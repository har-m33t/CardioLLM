"""
io.py — chunked HDF5 read helpers for the ARCHS4 human gene-count file.

Why this module exists
----------------------
The ARCHS4 H5 file is >30GB (Lachmann et al. 2018 originally 187,946 samples;
the current release is ~700k+ samples × ~35k genes). It does not fit in memory
on a typical workstation, so every EDA step has to stream over samples or
genes in chunks. This file centralises the H5 access so no downstream step
has to know the on-disk layout.

ARCHS4 H5 layout (as documented in the ARCHS4 paper's supplement and the
`archs4py` package):
    /data/expression       int matrix, shape (n_genes, n_samples), raw counts
    /meta/samples/*        per-sample metadata columns (bytes-encoded strings)
    /meta/genes/*          per-gene metadata columns

Sample metadata fields commonly present (byte strings on disk):
    geo_accession, series_id, source_name_ch1, characteristics_ch1, title,
    submission_date, singlecellprobability, readsaligned, readstotal

Gene metadata fields commonly present:
    gene_symbol (or symbol), ensembl_gene_id, gene_biotype (may be absent —
    older releases don't ship biotype; step 6 handles that gracefully).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

logger = logging.getLogger(__name__)


EXPRESSION_PATH = "data/expression"
SAMPLE_META_GROUP = "meta/samples"
GENE_META_GROUP = "meta/genes"

# ARCHS4 paper's recommended cutoff for the singlecellprobability metadata
# field: samples with sc_prob > this are flagged as likely single-cell. Any
# subsampling step (t-SNE reference distribution, correlation heatmap)
# excludes these samples upstream — see `filter_bulk_indices`. Whole-corpus
# steps that don't subsample (cohort, qc, gene_summary) still report both
# groups, per the "flag, don't drop" principle.
SINGLECELL_PROB_THRESHOLD = 0.5

# Fields we read for QC / cohort steps. Any field absent from a particular
# release is skipped by the callers (see `read_sample_field`).
SAMPLE_FIELDS_OF_INTEREST = (
    "geo_accession",
    "series_id",
    "source_name_ch1",
    "submission_date",
    "singlecellprobability",
    "readsaligned",
    "readstotal",
)

GENE_FIELDS_OF_INTEREST = (
    "gene_symbol",
    "symbol",
    "ensembl_gene_id",
    "gene_biotype",
)


@dataclass(frozen=True)
class Archs4Shape:
    n_genes: int
    n_samples: int


def open_h5(h5_path: Path) -> h5py.File:
    """Open the ARCHS4 H5 file read-only. Caller is responsible for closing."""
    return h5py.File(h5_path, "r")


def get_shape(h5: h5py.File) -> Archs4Shape:
    dset = h5[EXPRESSION_PATH]
    n_genes, n_samples = dset.shape
    return Archs4Shape(n_genes=n_genes, n_samples=n_samples)


def read_sample_field(h5: h5py.File, field: str) -> np.ndarray | None:
    """Return a per-sample metadata column as a numpy array, or None if absent.

    ARCHS4 stores strings as fixed-width bytes; we decode to str for
    convenience. Numeric fields (readsaligned, singlecellprobability) are
    returned as-is.
    """
    path = f"{SAMPLE_META_GROUP}/{field}"
    if path not in h5:
        return None
    arr = h5[path][:]
    if arr.dtype.kind in ("S", "O"):
        arr = np.array([x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def read_gene_field(h5: h5py.File, field: str) -> np.ndarray | None:
    path = f"{GENE_META_GROUP}/{field}"
    if path not in h5:
        return None
    arr = h5[path][:]
    if arr.dtype.kind in ("S", "O"):
        arr = np.array([x.decode("utf-8", errors="replace") if isinstance(x, bytes) else str(x) for x in arr])
    return arr


def gene_symbols(h5: h5py.File) -> np.ndarray:
    """Return per-gene symbol vector, falling back across common field names."""
    for candidate in ("gene_symbol", "symbol", "genes"):
        arr = read_gene_field(h5, candidate)
        if arr is not None:
            return arr
    raise KeyError("no gene symbol field found in ARCHS4 H5")


def read_gene_field_any(h5: h5py.File, *candidates: str) -> np.ndarray | None:
    """Return the first present per-gene field from `candidates`, or None.

    ARCHS4 field names have drifted across releases (`gene_biotype` →
    `biotype`, `ensembl_gene_id` → `ensembl_gene`). Callers pass every name
    they'd accept.
    """
    for c in candidates:
        arr = read_gene_field(h5, c)
        if arr is not None:
            return arr
    return None


def iter_sample_chunks(
    h5: h5py.File, chunk_size: int = 2048
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield successive (sample_slice, counts_chunk) pairs.

    counts_chunk has shape (n_genes, chunk_size_actual). Reading is done
    column-wise (samples on axis 1) which matches ARCHS4's on-disk layout —
    other orderings incur a large penalty.
    """
    dset = h5[EXPRESSION_PATH]
    n_genes, n_samples = dset.shape
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        sl = slice(start, stop)
        yield sl, dset[:, sl]


def iter_gene_chunks(
    h5: h5py.File, chunk_size: int = 512
) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield successive (gene_slice, counts_chunk) pairs, shape (chunk, n_samples).

    Gene-wise streaming is slower than sample-wise on the ARCHS4 layout
    (rows are not the primary axis) but is required for per-gene detection
    rate in step 6. Keep the chunk small.
    """
    dset = h5[EXPRESSION_PATH]
    n_genes, n_samples = dset.shape
    for start in range(0, n_genes, chunk_size):
        stop = min(start + chunk_size, n_genes)
        sl = slice(start, stop)
        yield sl, dset[sl, :]


def read_samples_by_index(h5: h5py.File, indices: np.ndarray) -> np.ndarray:
    """Materialise a (n_genes, len(indices)) counts matrix for the given samples.

    h5py requires monotonically increasing fancy indices — we sort, load, and
    return in the caller's original order.
    """
    indices = np.asarray(indices)
    order = np.argsort(indices)
    sorted_idx = indices[order]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    mat = h5[EXPRESSION_PATH][:, sorted_idx]
    return mat[:, inv]


def filter_bulk_indices(
    h5: h5py.File, sc_threshold: float = SINGLECELL_PROB_THRESHOLD
) -> tuple[np.ndarray, dict]:
    """Return the pool of sample indices that random subsampling may draw from.

    Applies ARCHS4's own `singlecellprobability` filter (paper cutoff 0.5):
    any sample with sc_prob > threshold is excluded from the pool. If the
    metadata field is absent from the release, no filter is applied and the
    stats dict records that fact — so downstream steps behave gracefully on
    older releases.

    Returns
    -------
    pool : np.ndarray[int]
        Sorted 1-D array of sample indices eligible for random subsampling.
    stats : dict
        `total`, `kept`, `excluded`, `excluded_pct`, `threshold`, plus a
        `note` when the release ships no singlecellprobability field. Meant
        to be embedded verbatim in step manifests.
    """
    n_samples = get_shape(h5).n_samples
    sc_prob = read_sample_field(h5, "singlecellprobability")
    if sc_prob is None:
        return np.arange(n_samples), {
            "total": int(n_samples),
            "kept": int(n_samples),
            "excluded": 0,
            "excluded_pct": 0.0,
            "threshold": float(sc_threshold),
            "note": "singlecellprobability absent from this release; no filter applied",
        }
    sc_prob = np.asarray(sc_prob, dtype=float)
    keep_mask = sc_prob <= sc_threshold
    pool = np.flatnonzero(keep_mask)
    excluded = int(n_samples - len(pool))
    return pool, {
        "total": int(n_samples),
        "kept": int(len(pool)),
        "excluded": excluded,
        "excluded_pct": round(100.0 * excluded / n_samples, 4),
        "threshold": float(sc_threshold),
    }


def subsample_from_pool(pool: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Draw `k` indices uniformly at random (without replacement) from `pool`.

    The pool is expected to be the output of `filter_bulk_indices` — a
    per-release, once-computed set of eligible sample indices. Every step
    that subsamples (quantile-norm reference, quantile-norm downstream matrix,
    t-SNE stability draws, correlation heatmap nested draw) must go through
    this helper so exclusions and randomness are centralised.
    """
    pool = np.asarray(pool)
    if k >= len(pool):
        return np.sort(pool.copy())
    rng = np.random.default_rng(seed)
    picked = pool[rng.choice(len(pool), size=k, replace=False)]
    return np.sort(picked)
