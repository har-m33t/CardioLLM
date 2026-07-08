"""
make_toy_data.py — synthesise a small ARCHS4-shaped H5 file for local testing.

Why
---
The real ARCHS4 human H5 is ~30GB and we should never touch it during
review/CI. Every step module in `eda/steps/` reads via `dataset/io.py`
against the exact group/dataset layout used by ARCHS4, so a tiny H5 with
the same schema exercises the whole pipeline end-to-end in seconds.

Layout produced (matches `dataset/io.py`):
    /data/expression                    int32,  shape (n_genes, n_samples)
    /meta/samples/geo_accession         S,      shape (n_samples,)
    /meta/samples/series_id             S,      shape (n_samples,)
    /meta/samples/source_name_ch1       S,      shape (n_samples,)
    /meta/samples/title                 S,      shape (n_samples,)
    /meta/samples/characteristics_ch1   S,      shape (n_samples,)
    /meta/samples/submission_date       S,      shape (n_samples,)
    /meta/samples/singlecellprobability f32,    shape (n_samples,)
    /meta/samples/readsaligned          i64,    shape (n_samples,)
    /meta/samples/readstotal            i64,    shape (n_samples,)
    /meta/genes/gene_symbol             S,      shape (n_genes,)
    /meta/genes/ensembl_gene_id         S,      shape (n_genes,)
    /meta/genes/gene_biotype            S,      shape (n_genes,)   [optional]

Counts are drawn from a negative-binomial-ish gene-mean model so QC / detection
rate / normalization all see realistic-shaped distributions.

Usage
-----
    python -m eda.dataset.make_toy_data --out /tmp/toy_archs4.h5
    python -m eda.dataset.make_toy_data --out /tmp/toy_archs4.h5 --n-genes 500 --n-samples 2000 --no-biotype
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _s(strings) -> np.ndarray:
    """Encode a list of Python strings as ARCHS4's fixed-width byte arrays."""
    return np.array([str(x).encode("utf-8") for x in strings])


# Toy CVD-flavored strings dropped into `title` / `characteristics_ch1` for a
# controllable fraction of samples so the elastic net label module has real
# positives to find. The list intentionally spans several of the keywords the
# elasticnet stage uses so the regex, not a single-keyword shortcut, is what
# lights them up.
_CVD_TOY_TITLES = (
    "Cardiac hypertrophy in mouse model",
    "Heart failure patient transcriptome",
    "Myocardial infarction time course",
    "Atherosclerosis progression study",
    "Cardiomyopathy dilated ventricle",
    "Coronary artery disease vs healthy",
    "Cardiac fibrosis after ischemic heart injury",
)
_CVD_TOY_CHARACTERISTICS = (
    "disease: heart failure",
    "condition: hypertension",
    "tissue: aortic tissue",
    "diagnosis: atrial fibrillation",
    "phenotype: cardiac hypertrophy",
)
_NON_CVD_TOY_TITLES = (
    "Bulk RNA-seq of leukocyte subsets",
    "Transcriptome of liver hepatocytes",
    "Skeletal muscle differentiation",
    "Neural progenitor cell profiling",
    "Colon adenocarcinoma vs adjacent normal",
)
_NON_CVD_TOY_CHARACTERISTICS = (
    "tissue: liver",
    "condition: healthy control",
    "cell type: hepatocyte",
    "disease: colorectal cancer",
    "phenotype: wild type",
)


def make_toy_h5(
    out_path: Path,
    n_genes: int = 500,
    n_samples: int = 2000,
    seed: int = 20260705,
    include_biotype: bool = True,
    cvd_positive_frac: float = 0.05,
) -> Path:
    """Write a synthetic ARCHS4-shaped H5 file to `out_path` and return it."""
    rng = np.random.default_rng(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-gene mean counts spanning several orders of magnitude — this gives
    # the detection-rate histogram real shape and forces normalization to
    # actually do something.
    gene_means = np.exp(rng.normal(loc=2.0, scale=1.8, size=n_genes))
    # Per-sample library-size multipliers so QC sees a spread (some outliers).
    sample_scale = np.exp(rng.normal(loc=0.0, scale=0.5, size=n_samples))
    lam = np.outer(gene_means, sample_scale)  # (n_genes, n_samples) rate
    counts = rng.poisson(lam).astype(np.int32)

    # Sample metadata.
    gsm = _s([f"GSM{1_000_000 + i}" for i in range(n_samples)])
    series = _s([f"GSE{200_000 + (i // 25)}" for i in range(n_samples)])
    source = _s(rng.choice(["blood", "brain", "liver", "muscle"], size=n_samples))

    # CVD-flavored title / characteristics for `cvd_positive_frac` of samples;
    # rest get neutral strings. This is what the elastic net label module reads.
    is_cvd = rng.uniform(size=n_samples) < cvd_positive_frac
    titles = [
        (rng.choice(_CVD_TOY_TITLES) if flag else rng.choice(_NON_CVD_TOY_TITLES))
        for flag in is_cvd
    ]
    chars = [
        (rng.choice(_CVD_TOY_CHARACTERISTICS) if flag else rng.choice(_NON_CVD_TOY_CHARACTERISTICS))
        for flag in is_cvd
    ]
    titles_enc = _s(titles)
    chars_enc = _s(chars)
    years = rng.integers(2012, 2025, size=n_samples)
    months = rng.integers(1, 13, size=n_samples)
    days = rng.integers(1, 28, size=n_samples)
    dates = _s([f"{y}-{m:02d}-{d:02d}" for y, m, d in zip(years, months, days)])
    # ~10% flagged as likely single-cell.
    sc_prob = rng.beta(1.2, 8.0, size=n_samples).astype(np.float32)
    reads_aligned = counts.sum(axis=0).astype(np.int64)
    reads_total = (reads_aligned * rng.uniform(1.05, 1.5, size=n_samples)).astype(np.int64)

    # Gene metadata.
    symbols = _s([f"TOYGENE{i:05d}" for i in range(n_genes)])
    ensembl = _s([f"ENSG{i:011d}" for i in range(n_genes)])
    if include_biotype:
        biotype = _s(rng.choice(
            ["protein_coding", "lncRNA", "pseudogene", "miRNA", "snoRNA"],
            size=n_genes,
            p=[0.6, 0.2, 0.1, 0.06, 0.04],
        ))

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data/expression", data=counts, chunks=(min(64, n_genes), min(64, n_samples)))
        m = h5.create_group("meta/samples")
        m.create_dataset("geo_accession", data=gsm)
        m.create_dataset("series_id", data=series)
        m.create_dataset("source_name_ch1", data=source)
        m.create_dataset("title", data=titles_enc)
        m.create_dataset("characteristics_ch1", data=chars_enc)
        m.create_dataset("submission_date", data=dates)
        m.create_dataset("singlecellprobability", data=sc_prob)
        m.create_dataset("readsaligned", data=reads_aligned)
        m.create_dataset("readstotal", data=reads_total)
        g = h5.create_group("meta/genes")
        g.create_dataset("gene_symbol", data=symbols)
        g.create_dataset("ensembl_gene_id", data=ensembl)
        if include_biotype:
            g.create_dataset("gene_biotype", data=biotype)

    return out_path


def main():
    p = argparse.ArgumentParser(description="Generate a small ARCHS4-shaped H5 for testing.")
    p.add_argument("--out", required=True, help="Output H5 path.")
    p.add_argument("--n-genes", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260705)
    p.add_argument("--no-biotype", action="store_true",
                   help="Omit /meta/genes/gene_biotype (exercises step 6's biotype-absent path).")
    p.add_argument("--cvd-positive-frac", type=float, default=0.05,
                   help="Fraction of samples given CVD-keyword titles/characteristics (elasticnet label positives).")
    args = p.parse_args()

    path = make_toy_h5(
        Path(args.out),
        n_genes=args.n_genes,
        n_samples=args.n_samples,
        seed=args.seed,
        include_biotype=not args.no_biotype,
        cvd_positive_frac=args.cvd_positive_frac,
    )
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
