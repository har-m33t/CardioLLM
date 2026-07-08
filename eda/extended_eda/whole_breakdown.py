"""
whole_breakdown.py — Section 1: whole-dataset disease-level breakdown.

Reads the label table (from `labels.py`) + genes-detected column (from the
existing whole-corpus QC CSV) and emits:

    whole_dataset_disease_breakdown.csv
        machine-readable, one row per disease category + a Total row.
    whole_dataset_disease_breakdown_display.md
        rendered markdown table for the write-up (mean/median/IQR strings).

Every category from `taxonomy.DISEASE_CATEGORIES` appears in the output,
including "unclear_unlabeled" — revised_eda_tod.md §1 mandates that unmatched
samples don't silently disappear.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import taxonomy as tx
from . import metrics as mx
from . import labels as lbl

logger = logging.getLogger(__name__)


def _order_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` rows sorted by the fixed taxonomy priority + unclear last."""
    order = list(tx.DISEASE_PRIORITY) + ["unclear_unlabeled"]
    df = df.copy()
    df["_ord"] = df["disease_category"].map({c: i for i, c in enumerate(order)})
    df = df.sort_values("_ord", na_position="last").drop(columns=["_ord"])
    return df


def _display_name(cat: str) -> str:
    if cat == "Total":
        return "Total"
    return tx.disease_category_display_name(cat)


def _render_markdown(df: pd.DataFrame) -> str:
    """Render the machine-readable table into the human-facing format from
    revised_eda_tod.md §1."""
    header = (
        "| Disease category | N patients (post-fallback) | N samples | "
        "Samples/patient (mean, median, IQR) | "
        "Genes detected/sample (mean, median, IQR) | N series | "
        "Resolution % | N patients (truly resolved) |\n"
        "|---|---:|---:|---|---|---:|---:|---:|"
    )
    lines = [header]
    for _, row in df.iterrows():
        cat = _display_name(row["disease_category"])
        spp = mx.format_stats({
            "mean": row["samples_per_patient_mean"],
            "median": row["samples_per_patient_median"],
            "q1": row["samples_per_patient_q1"],
            "q3": row["samples_per_patient_q3"],
            "n": row["n_patients"],
        })
        gds = mx.format_stats({
            "mean": row["genes_detected_mean"],
            "median": row["genes_detected_median"],
            "q1": row["genes_detected_q1"],
            "q3": row["genes_detected_q3"],
            "n": row["genes_detected_n"],
        })
        n_series = "—" if cat == "Total" and pd.isna(row["n_series"]) else f"{int(row['n_series']):,}"
        res_pct = row.get("resolution_pct", 0.0)
        res_pct_s = "—" if pd.isna(res_pct) else f"{float(res_pct):.2f}%"
        n_truly = row.get("n_patients_truly_resolved", 0)
        n_truly_s = "—" if pd.isna(n_truly) else f"{int(n_truly):,}"
        lines.append(
            f"| {cat} | {int(row['n_patients']):,} | {int(row['n_samples']):,} | "
            f"{spp} | {gds} | {n_series} | {res_pct_s} | {n_truly_s} |"
        )
    return "\n".join(lines) + "\n"


def run(
    labels_path: Path,
    qc_csv_path: Path,
    outdir: Path,
) -> Path:
    """Produce the section-1 CSV and rendered markdown table.

    Returns the path to the CSV; the markdown lives alongside it.
    """
    out = outdir / "section1_whole_breakdown"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("section1: loading label table %s", labels_path)
    labels = lbl.load_labels(labels_path)
    logger.info("section1: loading QC csv %s", qc_csv_path)
    gd = mx.load_genes_detected(qc_csv_path)
    mx.assert_qc_matches_labels(labels, gd, qc_csv_path, labels_path)
    merged = labels.merge(gd, on="geo_accession", how="left")

    # Ensure every taxonomy category is present in the output, even if a
    # category matched zero samples on this release — otherwise a reader
    # comparing two runs can't tell "0" from "we forgot to compute it".
    per_cat = mx.compute_group_metrics(merged, "disease_category")
    all_cats = list(tx.DISEASE_PRIORITY) + ["unclear_unlabeled"]
    present = set(per_cat["disease_category"])
    for c in all_cats:
        if c not in present:
            per_cat = pd.concat([per_cat, pd.DataFrame([{
                "disease_category": c,
                "n_patients": 0, "n_samples": 0, "n_series": 0,
                "samples_per_patient_mean": float("nan"),
                "samples_per_patient_median": float("nan"),
                "samples_per_patient_q1": float("nan"),
                "samples_per_patient_q3": float("nan"),
                "samples_per_patient_max": float("nan"),
                "genes_detected_mean": float("nan"),
                "genes_detected_median": float("nan"),
                "genes_detected_q1": float("nan"),
                "genes_detected_q3": float("nan"),
                "genes_detected_n": 0,
                "n_samples_with_resolved_patient_key": 0,
                "n_patients_truly_resolved": 0,
                "resolution_pct": 0.0,
            }])], ignore_index=True)

    per_cat = _order_categories(per_cat)
    tot = mx.total_row(merged, "disease_category")
    tot["n_series"] = float("nan")  # "—" in the rendered markdown; total series across categories overlaps
    per_cat = pd.concat([per_cat, pd.DataFrame([tot])], ignore_index=True)

    csv_path = out / "whole_dataset_disease_breakdown.csv"
    per_cat.to_csv(csv_path, index=False)
    logger.info("section1: wrote %s (%d rows)", csv_path, len(per_cat))

    md_path = out / "whole_dataset_disease_breakdown_display.md"
    md_path.write_text(_render_markdown(per_cat))
    logger.info("section1: wrote %s", md_path)
    return csv_path
