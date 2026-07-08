"""
cvd_breakdown.py — Section 2: CVD-only breakdown by subtype.

Pool definition (per revised_eda_tod.md §2 and the PI's phrasing):

    CVD pool = { samples where is_cvd_disease OR is_cvd_tissue }

i.e. a sample enters the CVD scope either because a cardiovascular disease
keyword hit its metadata OR because the source_name_ch1 / characteristics_ch1
metadata names a cardiovascular tissue (heart, aorta, coronary artery, etc.).
Both flags are precomputed in the label table.

Within that pool we break by CVD subtype (heart failure / arrhythmia+AFib /
coronary artery disease / cardiomyopathy other / hypertension / other-unspec)
and report the same five metrics as section 1.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import taxonomy as tx
from . import metrics as mx
from . import labels as lbl

logger = logging.getLogger(__name__)


def _order_subtypes(df: pd.DataFrame) -> pd.DataFrame:
    order = list(tx.CVD_SUBTYPE_PRIORITY)
    df = df.copy()
    df["_ord"] = df["cvd_subtype"].map({c: i for i, c in enumerate(order)})
    df = df.sort_values("_ord", na_position="last").drop(columns=["_ord"])
    return df


def _display_name(st: str) -> str:
    if st == "Total CVD":
        return "Total CVD"
    return tx.cvd_subtype_display_name(st)


def _render_markdown(df: pd.DataFrame) -> str:
    header = (
        "| CVD subtype | N patients (post-fallback) | N samples | "
        "Samples/patient (mean, median, IQR) | "
        "Genes detected/sample (mean, median, IQR) | N series | "
        "Resolution % | N patients (truly resolved) |\n"
        "|---|---:|---:|---|---|---:|---:|---:|"
    )
    lines = [header]
    for _, row in df.iterrows():
        st = _display_name(row["cvd_subtype"])
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
        n_series = "—" if st == "Total CVD" and pd.isna(row["n_series"]) else f"{int(row['n_series']):,}"
        res_pct = row.get("resolution_pct", 0.0)
        res_pct_s = "—" if pd.isna(res_pct) else f"{float(res_pct):.2f}%"
        n_truly = row.get("n_patients_truly_resolved", 0)
        n_truly_s = "—" if pd.isna(n_truly) else f"{int(n_truly):,}"
        lines.append(
            f"| {st} | {int(row['n_patients']):,} | {int(row['n_samples']):,} | "
            f"{spp} | {gds} | {n_series} | {res_pct_s} | {n_truly_s} |"
        )
    return "\n".join(lines) + "\n"


def _canonicalize_subtypes(pool: pd.DataFrame) -> pd.DataFrame:
    """Bring the `cvd_subtype` column of a legacy label table in line with
    the current Issue-1 semantics:

      * All tissue-only samples (is_cvd_disease is False) → route to
        `tissue_only_disease_unconfirmed`, regardless of what specific
        subtype keyword may have matched under the old code. A tissue-only
        haystack that hit an acronym like "hfref" is still disease-unconfirmed
        by construction and MUST NOT be reported as disease-positive.
      * Any surviving legacy `other_unspecified_cvd` value on a
        is_cvd_disease-True sample → `disease_matched_subtype_unresolved`.

    Idempotent for label tables that were already produced by the current
    `assign_cvd_subtype`.
    """
    is_dz = pool["is_cvd_disease"].astype(bool)

    tissue_only_misrouted = (~is_dz) & (pool["cvd_subtype"] != "tissue_only_disease_unconfirmed")
    n_tissue_route = int(tissue_only_misrouted.sum())
    if n_tissue_route:
        prior_labels = pool.loc[tissue_only_misrouted, "cvd_subtype"].value_counts().to_dict()
        pool.loc[tissue_only_misrouted, "cvd_subtype"] = "tissue_only_disease_unconfirmed"
        logger.info(
            "section2: routed %d tissue-only samples out of specific/legacy "
            "subtypes into tissue_only_disease_unconfirmed (prior labels: %s)",
            n_tissue_route, prior_labels,
        )

    legacy_disease = (pool["cvd_subtype"] == tx.LEGACY_UNSPECIFIED_CVD_SUBTYPE) & is_dz
    n_legacy_disease = int(legacy_disease.sum())
    if n_legacy_disease:
        pool.loc[legacy_disease, "cvd_subtype"] = "disease_matched_subtype_unresolved"
        logger.info(
            "section2: canonicalized %d legacy `other_unspecified_cvd` "
            "disease-matched rows into `disease_matched_subtype_unresolved`",
            n_legacy_disease,
        )
    return pool


def run(
    labels_path: Path,
    qc_csv_path: Path,
    outdir: Path,
) -> Path:
    """Produce the section-2 CSV and rendered markdown table. Returns CSV path."""
    out = outdir / "section2_cvd_breakdown"
    out.mkdir(parents=True, exist_ok=True)

    logger.info("section2: loading label table %s", labels_path)
    labels = lbl.load_labels(labels_path)
    gd = mx.load_genes_detected(qc_csv_path)
    mx.assert_qc_matches_labels(labels, gd, qc_csv_path, labels_path)
    merged = labels.merge(gd, on="geo_accession", how="left")

    pool = merged[merged["is_cvd_pool"]].copy()
    pool = _canonicalize_subtypes(pool)
    logger.info(
        "section2: CVD pool = %d samples (of %d) — by disease: %d, by tissue: %d, both: %d",
        len(pool), len(merged),
        int(pool["is_cvd_disease"].sum()),
        int(pool["is_cvd_tissue"].sum()),
        int((pool["is_cvd_disease"] & pool["is_cvd_tissue"]).sum()),
    )

    per_subtype = mx.compute_group_metrics(pool, "cvd_subtype")

    all_subtypes = list(tx.CVD_SUBTYPE_PRIORITY)
    present = set(per_subtype["cvd_subtype"])
    for st in all_subtypes:
        if st not in present:
            per_subtype = pd.concat([per_subtype, pd.DataFrame([{
                "cvd_subtype": st,
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

    per_subtype = _order_subtypes(per_subtype)
    tot = mx.total_row(pool, "cvd_subtype")
    tot["cvd_subtype"] = "Total CVD"
    tot["n_series"] = float("nan")
    per_subtype = pd.concat([per_subtype, pd.DataFrame([tot])], ignore_index=True)

    csv_path = out / "cvd_disease_breakdown.csv"
    per_subtype.to_csv(csv_path, index=False)
    logger.info("section2: wrote %s (%d rows)", csv_path, len(per_subtype))

    md_path = out / "cvd_disease_breakdown_display.md"
    md_path.write_text(_render_markdown(per_subtype))
    logger.info("section2: wrote %s", md_path)

    # Persist pool composition alongside the table for the write-up and
    # cross-check step; a section-3 reader wants to see how the union split,
    # per-bucket resolution coverage, and how comorbid the CVD pool is with
    # the other disease categories (Issue 3 quantification).
    n_pool = int(len(pool))
    resolved_pool = pool["has_resolved_patient"].astype(bool)
    pool_res = int(resolved_pool.sum())

    def _bucket_stats(mask: pd.Series) -> dict:
        sub = pool.loc[mask]
        n = int(len(sub))
        n_res = int(sub["has_resolved_patient"].astype(bool).sum()) if n else 0
        return {
            "n_samples": n,
            "n_samples_with_resolved_patient_key": n_res,
            "n_patients_truly_resolved": int(
                sub.loc[sub["has_resolved_patient"].astype(bool), "patient_key"].nunique()
            ) if n else 0,
            "n_patients_after_fallback": int(sub["patient_key"].nunique()) if n else 0,
            "resolution_pct": round(100.0 * n_res / n, 4) if n else 0.0,
        }

    per_bucket = {
        st: _bucket_stats(pool["cvd_subtype"] == st)
        for st in tx.CVD_SUBTYPE_PRIORITY
    }

    if "n_disease_categories_matched" in pool.columns:
        n_comorbid_pool = int(
            (pool["is_cvd_disease"] & (pool["n_disease_categories_matched"] >= 2)).sum()
        )
        comorbid_note = None
    else:
        n_comorbid_pool = None
        comorbid_note = (
            "n_disease_categories_matched not present in label table — "
            "re-run the `labels` step to populate the comorbidity count."
        )

    pool_stats = {
        "n_pool_samples": n_pool,
        "n_by_disease_only": int((pool["is_cvd_disease"] & ~pool["is_cvd_tissue"]).sum()),
        "n_by_tissue_only": int((~pool["is_cvd_disease"] & pool["is_cvd_tissue"]).sum()),
        "n_by_both": int((pool["is_cvd_disease"] & pool["is_cvd_tissue"]).sum()),
        "n_by_disease_total": int(pool["is_cvd_disease"].sum()),
        "n_by_tissue_total": int(pool["is_cvd_tissue"].sum()),
        "cvd_pool_resolution": {
            "n_samples": n_pool,
            "n_samples_with_resolved_patient_key": pool_res,
            "resolution_pct": round(100.0 * pool_res / n_pool, 4) if n_pool else 0.0,
            "n_patients_truly_resolved": int(
                pool.loc[resolved_pool, "patient_key"].nunique()
            ) if n_pool else 0,
            "n_patients_after_fallback": int(pool["patient_key"].nunique()) if n_pool else 0,
        },
        "per_subtype_resolution": per_bucket,
        "comorbidity_with_non_cvd_categories": {
            "n_cvd_disease_pool_samples_matching_another_category": n_comorbid_pool,
            "denominator_is_cvd_disease": int(pool["is_cvd_disease"].sum()),
            "note": comorbid_note or (
                "Count of samples where is_cvd_disease is True AND at least one "
                "non-CVD category regex also matched. The first-match priority "
                "rule assigned these to `cardiovascular` in Section 1, "
                "suppressing the non-CVD category's count for these samples."
            ),
        },
    }
    import json as _json
    (out / "cvd_pool_composition.json").write_text(_json.dumps(pool_stats, indent=2))
    return csv_path
