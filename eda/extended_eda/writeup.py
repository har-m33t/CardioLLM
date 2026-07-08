"""
writeup.py — assembles the four markdown deliverables from revised_eda_tod.md:

    definitions.md            — before any numbers (patient resolution +
                                disease taxonomy, per §0)
    whole_dataset_writeup.md  — section 1 rendered table + narrative
    cvd_writeup.md            — section 2 rendered table + narrative
    eda_writeup.md            — overall write-up (§4): both tables, methods,
                                flags for cohorts too small to support 5-fold CV

Everything is a plain string-substitution over the CSV/JSON artifacts produced
by earlier steps — no re-aggregation happens here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from . import taxonomy as tx
from . import patient_resolution as pr

logger = logging.getLogger(__name__)

# Under this many patients, a subtype can't support a stratified 5-fold linear
# probe CV — revised_eda_tod.md §4 asks us to flag this at write-up time so it
# feeds into scoping the next stage.
MIN_PATIENTS_FOR_5FOLD_CV = 25


def _cvd_pool_resolution_md(pool: dict | None) -> str:
    """Render CVD-pool (and per-subtype) resolution stats. Returns an empty
    string if the pool JSON isn't available (definitions can be written before
    section 2 has run)."""
    if not pool or "cvd_pool_resolution" not in pool:
        return ""
    p = pool["cvd_pool_resolution"]
    per = pool.get("per_subtype_resolution", {})
    lines = [
        "",
        "Within the CVD pool (§2 pool = is_cvd_disease OR is_cvd_tissue):",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| CVD-pool samples | {p['n_samples']:,} |",
        f"| Samples with a resolvable patient key | {p['n_samples_with_resolved_patient_key']:,} |",
        f"| Resolution % | {p['resolution_pct']:.2f}% |",
        f"| Unique patients (truly resolved) | {p['n_patients_truly_resolved']:,} |",
        f"| Unique patients (after fallback) | {p['n_patients_after_fallback']:,} |",
        "",
        "Per subtype (post-Issue-1 bucket split):",
        "",
        "| Subtype | N samples | N resolved samples | Resolution % | Truly-resolved patients |",
        "|---|---:|---:|---:|---:|",
    ]
    for st in tx.CVD_SUBTYPE_PRIORITY:
        s = per.get(st, {"n_samples": 0, "n_samples_with_resolved_patient_key": 0,
                         "resolution_pct": 0.0, "n_patients_truly_resolved": 0})
        lines.append(
            f"| {tx.cvd_subtype_display_name(st)} | {s['n_samples']:,} | "
            f"{s['n_samples_with_resolved_patient_key']:,} | "
            f"{s['resolution_pct']:.2f}% | {s['n_patients_truly_resolved']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def _cvd_comorbidity_md(pool: dict | None) -> str:
    """One-paragraph render of the CVD-pool comorbidity count (Issue 3)."""
    if not pool or "comorbidity_with_non_cvd_categories" not in pool:
        return ""
    c = pool["comorbidity_with_non_cvd_categories"]
    n = c.get("n_cvd_disease_pool_samples_matching_another_category")
    denom = c.get("denominator_is_cvd_disease")
    if n is None or not denom:
        note = c.get("note", "")
        return f"\n\n**Comorbid overlap (this release).** {note}\n" if note else ""
    pct = 100.0 * n / denom
    return (
        f"\n\n**Comorbid overlap (this release).** {n:,} of {denom:,} "
        f"CVD-disease-matched pool samples ({pct:.2f}%) ALSO matched at least "
        "one non-CVD category's keyword net; those samples are reported under "
        "`Cardiovascular` in the Section 1 table and are therefore missing "
        "from whichever later-priority category also matched.\n"
    )


def _keyword_appendix() -> str:
    lines = ["## Appendix A — disease-category keyword lists\n"]
    for cat in list(tx.DISEASE_PRIORITY) + ["unclear_unlabeled"]:
        if cat == "unclear_unlabeled":
            lines.append(f"**{tx.disease_category_display_name(cat)}** — sentinel: any sample "
                         "that hit none of the category regexes above lands here.\n")
            continue
        kws = tx.DISEASE_CATEGORIES[cat]
        lines.append(f"**{tx.disease_category_display_name(cat)}** ({len(kws)} keywords):")
        lines.append(", ".join(f"`{k}`" for k in kws))
        lines.append("")

    lines.append("## Appendix B — CVD anatomical (tissue) patterns\n")
    lines.append(", ".join(f"`{k}`" for k in tx.CVD_TISSUE_PATTERNS))
    lines.append("")

    lines.append("## Appendix C — CVD subtype keyword lists\n")
    fallback_notes = {
        "disease_matched_subtype_unresolved": (
            "Non-keyword fallback for CVD-pool samples where `is_cvd_disease` "
            "is TRUE (a broad CVD disease keyword matched) but none of the "
            "specific subtype nets (heart failure, arrhythmia+AFib, CAD, "
            "cardiomyopathy-other, hypertension) matched. These samples ARE "
            "real disease-positive; only their subtype label is ambiguous."
        ),
        "tissue_only_disease_unconfirmed": (
            "Non-keyword fallback for CVD-pool samples where `is_cvd_disease` "
            "is FALSE — the sample entered the pool ONLY via a CVD tissue "
            "keyword. Disease status is unconfirmed; these samples MUST NOT "
            "be treated as CVD-disease-positive by any downstream consumer "
            "(e.g. the linear probe stage). Includes tissue-only haystacks "
            "that happened to contain a subtype acronym (e.g. \"hfref\", "
            "\"dcm\", \"cad\") — the broad CVD disease net's failure means "
            "we have insufficient evidence to trust that acronym as a "
            "confirmed disease label."
        ),
    }
    for st in tx.CVD_SUBTYPE_PRIORITY:
        kws = tx.CVD_SUBTYPES[st]
        if not kws:
            note = fallback_notes.get(
                st,
                "Non-keyword fallback bucket for samples in the CVD pool that "
                "didn't match any specific subtype.",
            )
            lines.append(f"**{tx.cvd_subtype_display_name(st)}** — {note}")
        else:
            lines.append(f"**{tx.cvd_subtype_display_name(st)}** ({len(kws)} keywords):")
            lines.append(", ".join(f"`{k}`" for k in kws))
        lines.append("")

    lines.append("## Appendix D — patient identifier key patterns\n")
    lines.append("Any of these keys, case-insensitive, in `characteristics_ch1` is treated as a patient identifier:")
    lines.append(", ".join(f"`{k}`" for k in pr.PATIENT_KEY_PATTERNS))
    lines.append("")
    return "\n".join(lines)


def write_definitions(
    coverage_json: Path,
    out_path: Path,
    pool_json: Path | None = None,
) -> None:
    cov = json.loads(Path(coverage_json).read_text())
    pool = json.loads(Path(pool_json).read_text()) if pool_json and Path(pool_json).exists() else None
    body = f"""# Extended EDA — definitions (§0)

This file states the patient-resolution method and its coverage %, and the
disease taxonomy used, **before** any numbers are generated in sections 1-3.

## Patient vs. sample

GEO / ARCHS4 organises metadata around **samples** (GSM accessions), not
patients. A single patient may contribute multiple samples (replicates,
timepoints, multiple tissues). There is no structured patient/subject ID field
in ARCHS4; patient identifiers, when present, live inside the free-text
`characteristics_ch1` column as `key: value` pairs.

### Resolution method

For each sample we scan `characteristics_ch1` for the first of these keys
(case-insensitive; longer variants preferred): `{", ".join(pr.PATIENT_KEY_PATTERNS)}`.
The value after the `:` (or `=`), stripped and lowercased, is the raw patient
id. Placeholder values (`NA`, `N/A`, `none`, `unknown`, `-`, `?`, `not available`)
are treated as absent — a placeholder is worse than nothing because it would
collapse all placeholder-having samples in a study into one fake patient.

The **canonical patient key** used by every "N patients" and "samples per
patient" aggregation is:

    patient_key = f"{{series_id}}::{{patient_id}}"

Requiring same-`series_id` scoping is the important guard: patient id "1" in
GSE A has nothing to do with patient id "1" in GSE B. Cross-series patient
identity would require a linking process (name/DOB/etc.) that GEO does not
expose.

### Fallback

For samples where **no** patient key is resolvable, the fallback is
sample-as-own-patient — each such sample counts as its own patient
(`patient_key = "__unresolved__::<geo_accession>"`). This upper-bounds
patient counts and preserves per-disease sample totals. **This means every
"N patients" number reported below is dominated by the sample-as-own-patient
fallback whenever coverage % is low, and should be read alongside "samples
per patient" (mean/median/IQR) — which surfaces the true replicate structure
in the fraction of samples that do have a resolvable key.**

### Coverage on this release

Whole corpus:

| Field | Value |
|---|---:|
| Total samples | {cov['n_samples']:,} |
| Samples with a resolvable patient key | {cov['n_with_resolvable_patient_key']:,} |
| Coverage % | {cov['coverage_pct']:.2f}% |
| Unique patients (resolved) | {cov['n_unique_patients_resolved']:,} |
| Unique patients (fallback = own sample) | {cov['n_unique_patients_fallback']:,} |
| Unique patients (total) | {cov['n_unique_patients_total']:,} |

{_cvd_pool_resolution_md(pool)}

## Disease taxonomy

We use MeSH-style broad categories rather than resolving every specific
named condition — this is intentionally a landscape view, not a clinical
registry. Each sample gets one category (first match wins in a fixed
priority order); anything that fails every keyword lands in an explicit
`Unclear / unlabeled` bucket rather than silently disappearing.

Categories (priority-ordered):
{chr(10).join(f"  {i+1}. {tx.disease_category_display_name(c)}" for i, c in enumerate(tx.DISEASE_PRIORITY))}
  {len(tx.DISEASE_PRIORITY)+1}. {tx.disease_category_display_name("unclear_unlabeled")}

**First-match consequence.** Category assignment is strictly first-match in
the ordered list above. A sample whose metadata hits keywords from multiple
categories (a comorbid study — e.g. "atherosclerosis in breast cancer
patients") is always assigned to whichever category appears higher in this
order, which inflates the earlier category's count and correspondingly
undercounts every later category's count of comorbid samples.
{_cvd_comorbidity_md(pool)}

### Genes-captured definition

`genes_detected` = count of genes with **non-zero** expression in a given
sample. Same definition as `eda/steps/qc.py` produced for the whole-corpus
QC step; we reuse that CSV rather than reintroduce a different threshold
(CPM etc.) here.

### CVD scope (used by section 2)

A sample enters the CVD pool if **either** condition holds:

* **CVD disease keyword hit** — the cardiovascular keyword list above matched
  in title, source_name_ch1, or characteristics_ch1.
* **CVD tissue keyword hit** — anatomical CVD terms (see Appendix B) matched
  in the same three fields.

This corresponds to the PI's phrasing: a sample counts as CVD if it
"comes from cardiovascular tissue, or cardiovascular disease".

{_keyword_appendix()}
"""
    Path(out_path).write_text(body)
    logger.info("definitions: wrote %s", out_path)


def _flag_undersized_subtypes(section2_csv: Path) -> list[str]:
    """Flag subtypes below the 5-fold CV floor. Reports BOTH the post-fallback
    N patients (which the table also uses) and truly-resolved N patients,
    because low-coverage subtypes can clear the floor after fallback while
    truly having <25 distinct annotated patients (Issue 2 — Arrhythmia+AFib
    was called out for exactly this)."""
    df = pd.read_csv(section2_csv)
    flags = []
    for _, row in df.iterrows():
        st = row["cvd_subtype"]
        if st == "Total CVD":
            continue
        n_p = int(row["n_patients"])
        n_s = int(row["n_samples"])
        if n_s <= 0:
            continue
        n_truly = int(row.get("n_patients_truly_resolved", 0))
        res_pct = float(row.get("resolution_pct", 0.0))
        below_post = n_p < MIN_PATIENTS_FOR_5FOLD_CV
        below_truly = n_truly < MIN_PATIENTS_FOR_5FOLD_CV
        if below_post:
            flags.append(
                f"* **{tx.cvd_subtype_display_name(st)}** — only {n_p} patients "
                f"post-fallback ({n_s} samples, {n_truly} truly resolved, "
                f"{res_pct:.2f}% coverage). Below the ~{MIN_PATIENTS_FOR_5FOLD_CV}-patient "
                "floor for a meaningful stratified 5-fold linear-probe CV."
            )
        elif below_truly:
            flags.append(
                f"* **{tx.cvd_subtype_display_name(st)}** — clears the "
                f"{MIN_PATIENTS_FOR_5FOLD_CV}-patient floor after fallback "
                f"({n_p} patients over {n_s} samples), but only {n_truly} "
                f"patients are truly resolved ({res_pct:.2f}% coverage). "
                "A stratified 5-fold CV over the resolved-only subset would "
                "fall under the floor; treat the post-fallback N as an "
                "upper bound only."
            )
    return flags


def _flag_series_dominance(section1_csv: Path, section2_csv: Path) -> list[str]:
    """Loose 'dominated by one or two series' flag — series_count / samples per
    series is the cheap proxy. If N_samples/N_series is large, few series
    supply most of the samples in that category, so batch effects will be a
    real concern later."""
    flags = []
    for label, path, key in (
        ("Section 1", section1_csv, "disease_category"),
        ("Section 2", section2_csv, "cvd_subtype"),
    ):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            name = row[key]
            if name in ("Total", "Total CVD"):
                continue
            n_s = int(row["n_samples"])
            n_ser = int(row["n_series"]) if not pd.isna(row["n_series"]) else 0
            if n_s >= 500 and n_ser > 0 and (n_s / n_ser) >= 200:
                disp = (
                    tx.disease_category_display_name(name)
                    if key == "disease_category"
                    else tx.cvd_subtype_display_name(name)
                )
                flags.append(
                    f"* **{label} — {disp}** — {n_s:,} samples across only {n_ser:,} series "
                    f"(≈ {n_s/n_ser:.0f} samples/series). Likely dominated by a handful of "
                    "large studies; batch-effect risk if used downstream without series-aware CV."
                )
    return flags


def write_whole_writeup(section1_csv: Path, section1_md: Path, coverage_json: Path, out_path: Path) -> None:
    cov = json.loads(Path(coverage_json).read_text())
    table = Path(section1_md).read_text()
    body = f"""# Section 1 — Whole-dataset disease-level breakdown

**Patient resolution.** {cov['coverage_pct']:.2f}% of the {cov['n_samples']:,}
samples in this ARCHS4 human release have a resolvable patient/subject/donor
key in `characteristics_ch1`; the remaining samples fall back to
sample-as-own-patient (see `definitions.md`). All per-patient numbers below
should be read with that caveat in mind.

**Category assignment.** Each sample gets exactly one disease category by
first-match against the MeSH-style keyword lists in `definitions.md`.
Samples that hit no keyword are surfaced explicitly in the
`Unclear / unlabeled` row (they are not dropped).

**Priority-order caveat (limitation).** Categories are assigned by the fixed
priority order documented in `definitions.md` (Cardiovascular first, then
Cancer/neoplasm, then Neurological, …). A sample whose metadata matches
keywords from multiple categories is always assigned to whichever category
appears higher in this order, which inflates that category's row and
correspondingly undercounts every later category's count of comorbid
samples. Downstream consumers that need a comorbidity-aware view should join
back to the label table's `n_disease_categories_matched` column.

**Genes-detected definition.** Count of genes with non-zero counts in a
given sample. This column is joined in from
`eda_out/qc/qc_full_dataset.csv`, which the whole-corpus QC step already
computed using the same definition.

## Table

{table}

CSV: `section1_whole_breakdown/whole_dataset_disease_breakdown.csv`.
"""
    Path(out_path).write_text(body)
    logger.info("whole_writeup: wrote %s", out_path)


def write_cvd_writeup(section2_csv: Path, section2_md: Path, coverage_json: Path, pool_json: Path, out_path: Path) -> None:
    cov = json.loads(Path(coverage_json).read_text())
    pool = json.loads(Path(pool_json).read_text())
    table = Path(section2_md).read_text()
    pool_res = pool.get("cvd_pool_resolution", {})
    cvd_pct = pool_res.get("resolution_pct", 0.0)
    cvd_n = pool_res.get("n_samples", pool.get("n_pool_samples", 0))
    cvd_resolved = pool_res.get("n_samples_with_resolved_patient_key", 0)
    truly = pool_res.get("n_patients_truly_resolved", 0)

    # Small-cohort watch after Issue 1 split: warn if any subtype falls under
    # the 25-patient CV floor on truly-resolved counts (even if the
    # post-fallback n_patients clears it).
    per_bucket = pool.get("per_subtype_resolution", {})
    truly_flags = []
    for st, s in per_bucket.items():
        if s.get("n_samples", 0) > 0 and s.get("n_patients_truly_resolved", 0) < MIN_PATIENTS_FOR_5FOLD_CV:
            truly_flags.append(
                f"* **{tx.cvd_subtype_display_name(st)}** — "
                f"{s['n_patients_truly_resolved']:,} truly-resolved patients "
                f"(out of {s['n_samples']:,} samples, "
                f"{s.get('n_patients_after_fallback', 0):,} post-fallback patients)."
            )
    truly_block = ""
    if truly_flags:
        truly_block = (
            "\n\n**Small-cohort watch (truly-resolved patient counts).** "
            f"The following subtypes fall below the {MIN_PATIENTS_FOR_5FOLD_CV}-patient "
            "floor when counting only samples with a resolvable "
            "`characteristics_ch1` patient key (rather than the post-fallback "
            "N patients reported in the table). Read this alongside the table:\n\n"
            + "\n".join(truly_flags) + "\n"
        )

    body = f"""# Section 2 — CVD-only breakdown by subtype

**Pool definition.** A sample enters the CVD pool if it matched either the
cardiovascular disease keyword net **or** a cardiovascular tissue term
(heart, aorta, coronary artery, etc.). This union captures the PI's phrasing:
CVD includes samples that "come from cardiovascular tissue, or cardiovascular
disease". See `definitions.md` Appendices A, B for the exact lists.

**Pool composition on this release**

| Route into the CVD pool | N samples |
|---|---:|
| CVD disease keyword only | {pool['n_by_disease_only']:,} |
| CVD tissue keyword only | {pool['n_by_tissue_only']:,} |
| Both keyword and tissue | {pool['n_by_both']:,} |
| **Total pool** | **{pool['n_pool_samples']:,}** |

**Two fallback buckets — do NOT collapse them.** After Issue 1, the residual
"other/unspecified" bucket is split into two, based on whether the sample
matched a cardiovascular disease keyword at all:

* **Disease-matched, subtype unresolved** — `is_cvd_disease` is TRUE but no
  specific subtype keyword net (heart failure, arrhythmia+AFib, CAD,
  cardiomyopathy-other, hypertension) matched. These are a real
  disease-positive subset with an ambiguous subtype label.
* **Tissue-only, disease status unconfirmed** — the sample entered the CVD
  pool ONLY through a cardiovascular tissue keyword; `is_cvd_disease` is
  FALSE. **These samples must NOT be treated as CVD-disease-positive by any
  downstream consumer** (in particular, the linear probe stage that reads
  this write-up when deciding what counts as a positive label). They belong
  in negative / unlabeled / control cohorts, not in the disease-positive
  cohort.

**Patient resolution.** Whole corpus: {cov['coverage_pct']:.2f}% of the
{cov['n_samples']:,} samples have a resolvable patient key. Within the CVD
pool specifically: {cvd_pct:.2f}% ({cvd_resolved:,} of {cvd_n:,} samples;
{truly:,} truly-resolved patients). Per-patient numbers within the CVD pool
follow the same convention as section 1 — read alongside samples-per-patient
and the truly-resolved column.

**Category-priority caveat.** The whole-dataset Section 1 table pins
Cardiovascular first in the priority order (see `definitions.md`). Any
CVD-disease-matched sample that ALSO matched a non-CVD category's keyword net
is counted under `Cardiovascular` in Section 1, suppressing that non-CVD
category's count for those comorbid samples. See
`cvd_pool_composition.json → comorbidity_with_non_cvd_categories` for the
per-release count.

## Table

{table}
{truly_block}
CSV: `section2_cvd_breakdown/cvd_disease_breakdown.csv`.
"""
    Path(out_path).write_text(body)
    logger.info("cvd_writeup: wrote %s", out_path)


def write_overall(
    section1_csv: Path,
    section1_md: Path,
    section2_csv: Path,
    section2_md: Path,
    coverage_json: Path,
    cross_check_json: Path,
    pool_json: Path,
    out_path: Path,
) -> None:
    cov = json.loads(Path(coverage_json).read_text())
    xc = json.loads(Path(cross_check_json).read_text())
    pool = json.loads(Path(pool_json).read_text())
    tbl1 = Path(section1_md).read_text()
    tbl2 = Path(section2_md).read_text()

    undersized = _flag_undersized_subtypes(section2_csv)
    dominated = _flag_series_dominance(section1_csv, section2_csv)

    xc_verdict = "consistent" if xc.get("discrepancy") is None else "**DISCREPANCY — see cross_check.json**"

    flags_body = ""
    if undersized or dominated:
        flags_body = "## Notable flags for downstream scoping\n\n"
        if undersized:
            flags_body += "**Subtypes too small for 5-fold stratified CV** "
            flags_body += f"(< {MIN_PATIENTS_FOR_5FOLD_CV} patients):\n\n" + "\n".join(undersized) + "\n\n"
        if dominated:
            flags_body += "**Categories/subtypes dominated by a few large series** (potential batch-effect risk):\n\n"
            flags_body += "\n".join(dominated) + "\n\n"
    else:
        flags_body = "## Notable flags for downstream scoping\n\n_No subtype fell below the "
        flags_body += f"{MIN_PATIENTS_FOR_5FOLD_CV}-patient CV floor and no category was "
        flags_body += "dominated ≥ 200 samples per series on this release._\n\n"

    body = f"""# Extended EDA write-up (§4)

Both tables and the methods behind them, in one place. Full keyword lists and
patient-resolution details live in `definitions.md`.

## Methods, briefly

* **Patient resolution.** {cov['coverage_pct']:.2f}% of {cov['n_samples']:,}
  samples have a resolvable patient key in `characteristics_ch1`; the rest
  fall back to sample-as-own-patient. Same-`series_id` scoping is required to
  count two samples as the same patient. Read every "N patients" number
  alongside its "samples/patient" distribution.
* **Disease taxonomy (§1).** MeSH-style broad categories, first-match wins
  against the fixed priority order in `definitions.md` (Cardiovascular first,
  then Cancer/neoplasm, then Neurological, then Infectious, then
  Metabolic/endocrine, then Autoimmune, then Respiratory, then Renal, then
  Musculoskeletal, then the `Unclear / unlabeled` sentinel). Explicit
  `Unclear / unlabeled` bucket for samples that hit no keyword.
  **Limitation:** comorbid samples matching multiple categories are always
  counted under whichever category appears higher in this order — inflating
  earlier categories and undercounting later ones for those samples.
* **CVD scope (§2).** Union of CVD disease keyword hit OR CVD tissue keyword
  hit. Pool composition on this release: disease-only {pool['n_by_disease_only']:,};
  tissue-only {pool['n_by_tissue_only']:,}; both {pool['n_by_both']:,};
  total {pool['n_pool_samples']:,}. Residual "other/unspecified" is split
  into `Disease-matched, subtype unresolved` and
  `Tissue-only, disease status unconfirmed` — the latter is NOT
  disease-positive and must be excluded from any downstream positive-label
  cohort.
* **Genes detected.** Reused from `eda_out/qc/qc_full_dataset.csv` (non-zero
  count definition).
* **Cross-check (§3).** Verdict: {xc_verdict}. Section-1 Cardiovascular row
  N samples = {xc['section1']['cardiovascular_row_n_samples']:,} equals the
  label table's `is_cvd_disease` count = {xc['label_table']['n_cvd_by_disease_keyword']:,};
  section-2 Total CVD N samples = {xc['section2']['total_cvd_row_n_samples']:,}
  equals the label table's `is_cvd_pool` count = {xc['label_table']['n_cvd_pool_disease_or_tissue']:,},
  which is the disease count plus {xc['label_table']['n_cvd_by_tissue_only']:,}
  tissue-only additions.

## Section 1 — Whole-dataset breakdown

{tbl1}

CSV: `section1_whole_breakdown/whole_dataset_disease_breakdown.csv`.

## Section 2 — CVD-only breakdown

{tbl2}

CSV: `section2_cvd_breakdown/cvd_disease_breakdown.csv`.

{flags_body}
See `definitions.md` for the full disease taxonomy, patient-key patterns,
and CVD subtype keyword lists (Appendices A-D).
"""
    Path(out_path).write_text(body)
    logger.info("overall writeup: wrote %s", out_path)
