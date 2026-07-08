"""
labels.py — single streaming pass over sample metadata that produces the slim
per-sample label table every downstream section reads.

Why one pass, why parquet
-------------------------
The whole-dataset breakdown (section 1) and the CVD subtype breakdown
(section 2) both need the same three derived columns per sample:

    disease_category   — one of DISEASE_CATEGORIES + "unclear_unlabeled"
    is_cvd_pool        — bool: is_cvd_by_disease OR is_cvd_by_tissue
    cvd_subtype        — set only when is_cvd_pool

plus `patient_key`, `series_id`, and `sample_index`. Rather than re-scanning
the H5 metadata twice, we do one streaming pass over row slices, build the
label table, and write it to disk. Sections 1-3 then read that table plus the
existing per-sample QC CSV (which already has `genes_detected`) and do plain
pandas groupby's — no more H5 opens after this step.

Memory budget
-------------
24 GB total, real headroom often ~5 GB during a run. Loading all six string
columns at once for 1.1M rows in Python would be several GB. Instead we:
  * iterate row-slices of ~64k rows
  * for each slice, read the six needed metadata fields, decode bytes → str,
    lowercase, and concatenate the three "match" fields into one haystack
  * apply taxonomy + patient parse on the slice, append derived columns to
    output arrays, drop the slice's raw strings before continuing

Peak footprint per slice: ~200 MB for six 64k-string columns; that plus the
growing output arrays (5 fields × n_samples) stays well under the budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from . import patient_resolution as pr
from . import taxonomy as tx

logger = logging.getLogger(__name__)

# Row-slice size for the streaming pass. Picked so peak string-object memory
# stays under ~250 MB even when characteristics_ch1 values are on the long end.
DEFAULT_SLICE = 64_000


@dataclass(frozen=True)
class LabelTablePaths:
    """Files produced by build_label_table()."""
    labels_parquet: Path
    coverage_json: Path


def _decode(arr: np.ndarray) -> list[str]:
    """Turn an h5py object-string column into a list of Python strs."""
    out: list[str] = []
    for v in arr:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8", errors="replace"))
        elif v is None:
            out.append("")
        else:
            out.append(str(v))
    return out


def _read_slice_column(h5: h5py.File, path: str, sl: slice) -> list[str]:
    """Read a single sample-metadata column for one row slice, decoded to str.

    h5py returns an object dtype array of Python bytes / strings for GEO
    metadata fields; both branches decode to a plain list[str], which is what
    the taxonomy regexes expect.
    """
    if path not in h5:
        return [""] * (sl.stop - sl.start)
    return _decode(h5[path][sl])


def build_label_table(
    h5_path: Path,
    outdir: Path,
    slice_size: int = DEFAULT_SLICE,
    log_every: int = 5,
) -> LabelTablePaths:
    """Stream over all sample metadata, write the slim label parquet + coverage.

    Parameters
    ----------
    h5_path
        The ARCHS4 human H5 file.
    outdir
        Base output directory; a `labels/` subfolder is created if missing.
    slice_size
        Rows per streaming batch. Default sized for a 24GB machine; drop it
        (e.g. 16_000) if you're on a tighter box, raise it (e.g. 128_000) if
        you have RAM to burn and want a faster pass.
    log_every
        Print progress every N batches.
    """
    label_dir = outdir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as h5:
        n_samples = h5["data/expression"].shape[1]

        sample_index = np.arange(n_samples, dtype=np.int64)
        series_out = np.empty(n_samples, dtype=object)
        gsm_out = np.empty(n_samples, dtype=object)
        disease_out = np.empty(n_samples, dtype=object)
        subtype_out = np.empty(n_samples, dtype=object)
        is_cvd_disease = np.zeros(n_samples, dtype=bool)
        is_cvd_tissue = np.zeros(n_samples, dtype=bool)
        is_cvd_pool = np.zeros(n_samples, dtype=bool)
        # `n_disease_categories_matched` counts how many top-level disease
        # regexes hit the haystack for this sample. Used by section 3 to
        # quantify comorbidity — a CVD-pool sample with count >= 2 also had a
        # keyword hit in some non-CVD category, whose count the first-match
        # priority rule silently suppressed.
        n_categories_matched = np.zeros(n_samples, dtype=np.int32)
        patient_key_out = np.empty(n_samples, dtype=object)
        has_resolved_patient = np.zeros(n_samples, dtype=bool)

        n_with_key = 0
        n_batches = (n_samples + slice_size - 1) // slice_size

        for batch_i, start in enumerate(range(0, n_samples, slice_size)):
            stop = min(start + slice_size, n_samples)
            sl = slice(start, stop)

            title = _read_slice_column(h5, "meta/samples/title", sl)
            source = _read_slice_column(h5, "meta/samples/source_name_ch1", sl)
            chars = _read_slice_column(h5, "meta/samples/characteristics_ch1", sl)
            series = _read_slice_column(h5, "meta/samples/series_id", sl)
            gsm = _read_slice_column(h5, "meta/samples/geo_accession", sl)

            for i in range(stop - start):
                idx = start + i
                # Lowercased haystack for all keyword matching — the regexes
                # in taxonomy.py are case-insensitive but pre-lowering saves
                # allocations across ten regex calls per sample.
                hay = (
                    (title[i] or "") + " || "
                    + (source[i] or "") + " || "
                    + (chars[i] or "")
                ).lower()

                all_hits = tx.match_all_disease_categories(hay)
                n_categories_matched[idx] = len(all_hits)
                cat = all_hits[0] if all_hits else "unclear_unlabeled"
                disease_out[idx] = cat

                is_dz = (cat == "cardiovascular")  # by construction: matched CVD keyword
                is_tis = tx.is_cvd_by_tissue(hay)
                is_cvd_disease[idx] = is_dz
                is_cvd_tissue[idx] = is_tis
                if is_dz or is_tis:
                    is_cvd_pool[idx] = True
                    subtype_out[idx] = tx.assign_cvd_subtype(hay, is_dz)
                else:
                    subtype_out[idx] = ""  # NA sentinel; downstream filters on is_cvd_pool

                series_out[idx] = series[i]
                gsm_out[idx] = gsm[i]

                pid = pr.extract_patient_id(chars[i])
                pkey = pr.compose_patient_key(series[i], pid)
                if pkey is not None:
                    patient_key_out[idx] = pkey
                    has_resolved_patient[idx] = True
                    n_with_key += 1
                else:
                    # Fallback: sample-as-own-patient. Use the GEO accession so
                    # every downstream `.groupby("patient_key")` collapses at
                    # sample granularity for the unresolved rows.
                    fallback = gsm[i] or f"sample_{idx}"
                    patient_key_out[idx] = f"__unresolved__::{fallback}"

            if (batch_i % log_every == 0) or (batch_i + 1 == n_batches):
                logger.info(
                    "labels: batch %d/%d (rows %d..%d) — cumulative patient-key hits %d",
                    batch_i + 1, n_batches, start, stop, n_with_key,
                )

    df = pd.DataFrame({
        "sample_index": sample_index,
        "geo_accession": gsm_out,
        "series_id": series_out,
        "disease_category": disease_out,
        "is_cvd_disease": is_cvd_disease,
        "is_cvd_tissue": is_cvd_tissue,
        "is_cvd_pool": is_cvd_pool,
        "cvd_subtype": subtype_out,
        "n_disease_categories_matched": n_categories_matched,
        "patient_key": patient_key_out,
        "has_resolved_patient": has_resolved_patient,
    })

    labels_path = label_dir / "sample_labels.parquet"
    try:
        df.to_parquet(labels_path, index=False)
    except Exception as e:
        # Parquet needs pyarrow / fastparquet. If neither is installed we
        # fall back to CSV — slower but universally readable, and still fine
        # for the ~1M-row table (roughly 100 MB CSV).
        logger.warning("parquet write failed (%s); falling back to CSV", e)
        labels_path = label_dir / "sample_labels.csv"
        df.to_csv(labels_path, index=False)

    resolved_keys = df.loc[df["has_resolved_patient"], "patient_key"].unique()
    fallback_keys = df.loc[~df["has_resolved_patient"], "patient_key"].unique()
    coverage = pr.summarize_coverage(
        n_samples=len(df),
        n_with_key=n_with_key,
        n_unique_resolved=len(resolved_keys),
        n_unique_fallback=len(fallback_keys),
    )

    coverage_path = label_dir / "patient_coverage.json"
    import json as _json
    with open(coverage_path, "w") as f:
        _json.dump({
            "n_samples": coverage.n_samples,
            "n_with_resolvable_patient_key": coverage.n_with_key,
            "coverage_pct": coverage.coverage_pct,
            "n_unique_patients_resolved": coverage.n_unique_patients_resolved,
            "n_unique_patients_fallback": coverage.n_unique_patients_fallback,
            "n_unique_patients_total": (
                coverage.n_unique_patients_resolved + coverage.n_unique_patients_fallback
            ),
            "fallback_method": (
                "sample_as_own_patient — each sample without a resolvable "
                "(subject|patient|individual|donor)[_ ]id key in "
                "characteristics_ch1 is counted as its own patient. This "
                "upper-bounds the patient count and preserves per-disease "
                "sample totals."
            ),
        }, f, indent=2)

    logger.info(
        "labels: wrote %s (n=%d) and %s (coverage=%.2f%%)",
        labels_path, len(df), coverage_path, coverage.coverage_pct,
    )
    return LabelTablePaths(labels_parquet=labels_path, coverage_json=coverage_path)


def load_labels(labels_path: Path) -> pd.DataFrame:
    """Read the label table back, tolerating the parquet-or-CSV fallback."""
    if labels_path.suffix == ".parquet":
        return pd.read_parquet(labels_path)
    return pd.read_csv(labels_path)
