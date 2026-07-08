"""
metrics.py — pure-pandas aggregation helpers shared by section 1 and section 2.

Both breakdown tables report the same five per-group metrics:

    N patients                       (unique patient_key)
    N samples                        (row count)
    Samples/patient (mean, med, IQR) (groupby patient_key, count → describe)
    Genes detected/sample (mean, med, IQR)
    N series                         (unique series_id)

Genes-detected comes from the existing whole-corpus QC CSV
(`eda_out/qc/qc_full_dataset.csv`) rather than re-streaming the H5 — that CSV
already has one row per sample with a `genes_detected` column produced by
`eda/steps/qc.py` using the same non-zero-count definition
(revised_eda_tod.md §0 requires staying consistent with the prior definition).
Reusing it cuts a full H5 sweep from the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_genes_detected(qc_csv_path: Path) -> pd.DataFrame:
    """Return a two-column frame [geo_accession, genes_detected].

    The QC CSV is ~67MB for the full ARCHS4 corpus; loading only the columns
    we need keeps the memory footprint under 100 MB. `usecols` also insulates
    us from any future QC additions that might inflate the file.
    """
    df = pd.read_csv(qc_csv_path, usecols=["geo_accession", "genes_detected"])
    return df


def assert_qc_matches_labels(
    labels_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    qc_csv_path: Path,
    labels_path: Path,
    tolerance_missing_frac: float = 0.001,
) -> None:
    """Guard against silently joining a QC file from a different ARCHS4 release.

    Fails LOUDLY (raises) if:
      * either input is empty,
      * QC row count differs from label row count by more than a rounding
        error (both should equal `n_samples` for the same release), or
      * more than `tolerance_missing_frac` of the label table's samples have
        no matching `geo_accession` in the QC file.

    Rationale: `genes_detected` is joined into both the whole-dataset and CVD
    subtype breakdowns; a release mismatch would silently produce wrong
    per-group means/medians while the manifest reports "step ok".
    """
    n_labels = len(labels_df)
    n_qc = len(qc_df)
    if n_labels == 0 or n_qc == 0:
        raise AssertionError(
            f"empty inputs: labels rows={n_labels} ({labels_path}), "
            f"qc rows={n_qc} ({qc_csv_path})"
        )
    if abs(n_labels - n_qc) > 0:
        raise AssertionError(
            "QC / label-table release mismatch: "
            f"QC {qc_csv_path} has {n_qc:,} rows but label table "
            f"{labels_path} has {n_labels:,} rows. These must both cover the "
            "same ARCHS4 release. Re-run the whole-corpus QC step against "
            "the same H5 used for this run before continuing."
        )
    label_gsms = set(labels_df["geo_accession"].dropna().astype(str))
    qc_gsms = set(qc_df["geo_accession"].dropna().astype(str))
    n_missing = len(label_gsms - qc_gsms)
    if n_missing / max(1, n_labels) > tolerance_missing_frac:
        raise AssertionError(
            f"QC / label-table release mismatch: {n_missing:,} of "
            f"{n_labels:,} label geo_accessions are absent from QC "
            f"({qc_csv_path}). Check that both were produced from the same "
            "ARCHS4 H5 release."
        )
    logger.info(
        "metrics: QC/label version check ok — %d rows on each side, "
        "%d label geo_accessions missing from QC (%.4f%%)",
        n_labels, n_missing, 100.0 * n_missing / max(1, n_labels),
    )


def _iqr(x: np.ndarray) -> tuple[float, float]:
    """Return (Q1, Q3). Empty input → (NaN, NaN) so downstream `format_iqr`
    can render "—" without special-casing."""
    if len(x) == 0:
        return (float("nan"), float("nan"))
    q1, q3 = np.percentile(x, [25, 75])
    return (float(q1), float(q3))


def summarize_distribution(values: np.ndarray) -> dict:
    """Compact mean/median/IQR/max/min summary for a numeric vector."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "q1": float("nan"), "q3": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": 0}
    q1, q3 = _iqr(values)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q1": q1, "q3": q3,
        "min": float(np.min(values)), "max": float(np.max(values)),
        "n": int(len(values)),
    }


def format_stats(stats: dict) -> str:
    """Human-facing "mean X, median Y, IQR [Q1, Q3]" formatter for tables."""
    if stats["n"] == 0:
        return "—"
    return (
        f"mean {stats['mean']:.1f}, "
        f"median {stats['median']:.1f}, "
        f"IQR [{stats['q1']:.1f}, {stats['q3']:.1f}]"
    )


def _resolution_counts(sub: pd.DataFrame) -> dict:
    """Patient-resolution stats for a subset. `has_resolved_patient` may be
    absent in tests or on stale label tables — treat as all-unresolved."""
    n_samples = len(sub)
    if "has_resolved_patient" in sub.columns:
        resolved_mask = sub["has_resolved_patient"].astype(bool)
    else:
        resolved_mask = pd.Series(False, index=sub.index)
    n_resolved_samples = int(resolved_mask.sum())
    n_patients_truly_resolved = int(
        sub.loc[resolved_mask, "patient_key"].nunique()
    )
    resolution_pct = (
        round(100.0 * n_resolved_samples / n_samples, 4) if n_samples else 0.0
    )
    return {
        "n_samples_with_resolved_patient_key": n_resolved_samples,
        "n_patients_truly_resolved": n_patients_truly_resolved,
        "resolution_pct": resolution_pct,
    }


def compute_group_metrics(
    df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute the five section-1/2 metrics per level of `group_col`.

    `df` must have columns:
        patient_key, series_id, genes_detected, sample_index, plus `group_col`
    and optionally `has_resolved_patient` for the resolution columns.

    Returns one row per group level with columns:
        n_patients, n_samples, n_series,
        samples_per_patient_{mean,median,q1,q3,max},
        genes_detected_{mean,median,q1,q3},
        n_samples_with_resolved_patient_key, n_patients_truly_resolved,
        resolution_pct
    """
    records = []
    for level, sub in df.groupby(group_col, sort=False, dropna=False):
        n_samples = len(sub)
        n_patients = sub["patient_key"].nunique()
        n_series = sub["series_id"].nunique()

        # samples-per-patient distribution: how many rows each patient contributes
        per_patient = sub.groupby("patient_key").size().to_numpy()
        spp = summarize_distribution(per_patient)

        gd = sub["genes_detected"].dropna().to_numpy()
        gds = summarize_distribution(gd)

        rec = {
            group_col: level,
            "n_patients": n_patients,
            "n_samples": n_samples,
            "n_series": n_series,
            "samples_per_patient_mean": spp["mean"],
            "samples_per_patient_median": spp["median"],
            "samples_per_patient_q1": spp["q1"],
            "samples_per_patient_q3": spp["q3"],
            "samples_per_patient_max": spp["max"],
            "genes_detected_mean": gds["mean"],
            "genes_detected_median": gds["median"],
            "genes_detected_q1": gds["q1"],
            "genes_detected_q3": gds["q3"],
            "genes_detected_n": gds["n"],
        }
        rec.update(_resolution_counts(sub))
        records.append(rec)
    return pd.DataFrame.from_records(records)


def total_row(df: pd.DataFrame, group_col: str) -> dict:
    """Aggregate everything into a single "Total" row, matching the table
    format in revised_eda_tod.md §1 and §2."""
    n_samples = len(df)
    n_patients = df["patient_key"].nunique()
    per_patient = df.groupby("patient_key").size().to_numpy()
    spp = summarize_distribution(per_patient)
    gd = df["genes_detected"].dropna().to_numpy()
    gds = summarize_distribution(gd)
    row = {
        group_col: "Total",
        "n_patients": n_patients,
        "n_samples": n_samples,
        "n_series": df["series_id"].nunique(),
        "samples_per_patient_mean": spp["mean"],
        "samples_per_patient_median": spp["median"],
        "samples_per_patient_q1": spp["q1"],
        "samples_per_patient_q3": spp["q3"],
        "samples_per_patient_max": spp["max"],
        "genes_detected_mean": gds["mean"],
        "genes_detected_median": gds["median"],
        "genes_detected_q1": gds["q1"],
        "genes_detected_q3": gds["q3"],
        "genes_detected_n": gds["n"],
    }
    row.update(_resolution_counts(df))
    return row
