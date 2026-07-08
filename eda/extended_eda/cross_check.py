"""
cross_check.py — Section 3: consistency between the whole-dataset table's
`Cardiovascular` row and the section-2 CVD table's total.

Two definitions coexist in the extended EDA and both must be self-consistent:

  1. Section 1's "Cardiovascular" row uses the disease-keyword axis only —
     samples where `is_cvd_disease` (title/source/characteristics matched a
     CVD keyword). This is the same set the elastic-net weak label uses.
  2. Section 2's CVD pool is a UNION: `is_cvd_disease OR is_cvd_tissue`, so it
     legitimately covers more samples (a healthy-donor cardiac-tissue study
     without a disease keyword still enters the CVD scope).

The cross-check therefore has two clauses, not one:

  * `is_cvd_disease` count in the label table == section-1 `Cardiovascular`
    row's `n_samples` — this is the required strict equality that the todo's
    "same selection logic, same numbers" bullet is enforcing on the disease
    axis.
  * Section-2 total `n_samples` == count of `is_cvd_pool` in the label table,
    and is EXPECTED to be >= section 1's cardiovascular count. The excess is
    explained (tissue-only additions) rather than left ambiguous.

If either clause fails, we write a `discrepancy` field and log LOUDLY; the
orchestrator surfaces it in the manifest so it can't slip through.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import labels as lbl

logger = logging.getLogger(__name__)


def run(
    labels_path: Path,
    section1_csv: Path,
    section2_csv: Path,
    outdir: Path,
) -> Path:
    out = outdir / "section3_cross_check"
    out.mkdir(parents=True, exist_ok=True)

    labels = lbl.load_labels(labels_path)
    df1 = pd.read_csv(section1_csv)
    df2 = pd.read_csv(section2_csv)

    label_cvd_disease = int(labels["is_cvd_disease"].sum())
    label_cvd_pool = int(labels["is_cvd_pool"].sum())
    label_cvd_tissue_only = int((~labels["is_cvd_disease"] & labels["is_cvd_tissue"]).sum())

    row_s1 = df1[df1["disease_category"] == "cardiovascular"]
    s1_cvd_samples = int(row_s1["n_samples"].iloc[0]) if len(row_s1) else -1

    row_s2 = df2[df2["cvd_subtype"] == "Total CVD"]
    s2_total_samples = int(row_s2["n_samples"].iloc[0]) if len(row_s2) else -1

    strict_match_disease = (s1_cvd_samples == label_cvd_disease)
    pool_matches_label = (s2_total_samples == label_cvd_pool)
    pool_ge_disease = (s2_total_samples >= s1_cvd_samples)
    tissue_only_delta_matches = (
        s2_total_samples - s1_cvd_samples == label_cvd_tissue_only
    )

    result = {
        "label_table": {
            "n_cvd_by_disease_keyword": label_cvd_disease,
            "n_cvd_pool_disease_or_tissue": label_cvd_pool,
            "n_cvd_by_tissue_only": label_cvd_tissue_only,
        },
        "section1": {
            "cardiovascular_row_n_samples": s1_cvd_samples,
        },
        "section2": {
            "total_cvd_row_n_samples": s2_total_samples,
        },
        "checks": {
            "section1_cvd_equals_disease_keyword_count": strict_match_disease,
            "section2_total_equals_pool_count": pool_matches_label,
            "section2_total_ge_section1_cvd": pool_ge_disease,
            "tissue_only_delta_matches": tissue_only_delta_matches,
        },
        "discrepancy": None,
    }

    if not strict_match_disease:
        result["discrepancy"] = (
            f"Section 1 Cardiovascular row ({s1_cvd_samples}) does not match "
            f"the label table's is_cvd_disease count ({label_cvd_disease}). "
            "This must be reconciled before either table is reported."
        )
        logger.error("SECTION-3 DISCREPANCY: %s", result["discrepancy"])
    elif not pool_matches_label or not tissue_only_delta_matches:
        result["discrepancy"] = (
            f"Section 2 Total ({s2_total_samples}) is not consistent with the "
            f"label table's CVD-pool count ({label_cvd_pool}) or with the "
            f"tissue-only delta of {label_cvd_tissue_only}."
        )
        logger.error("SECTION-3 DISCREPANCY: %s", result["discrepancy"])
    else:
        logger.info(
            "section3 OK: disease-axis strict match (%d = %d); pool matches label; "
            "tissue-only additions to pool = %d",
            s1_cvd_samples, label_cvd_disease, label_cvd_tissue_only,
        )

    out_path = out / "cross_check.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("section3: wrote %s", out_path)
    return out_path
