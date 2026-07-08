"""
taxonomy.py — disease taxonomy for the extended EDA (revised_eda_tod.md §0-2).

Two taxonomies live here:

  1. `DISEASE_CATEGORIES` — MeSH-style broad categories for the whole-dataset
     breakdown in section 1. Meant to be a landscape view, not a clinical
     registry; each sample gets one category (first match wins in the fixed
     `DISEASE_PRIORITY` order), plus an explicit "unclear/unlabeled" bucket for
     anything that fails every keyword.
  2. `CVD_SUBTYPES` — one level deeper within CVD for the section 2 breakdown.
     Same first-match semantics, with an "other_unspecified_cvd" catch-all for
     samples that made it into the CVD pool but didn't hit a specific subtype.

The CVD keyword list under `DISEASE_CATEGORIES["cardiovascular"]` is the broad
net from elasticnet_todo.md §1 (verbatim, not the narrow HF-only list) — this
is required by section 3's cross-check: the "cardiovascular" row in the whole
table has to be computed from the same keyword set that the CVD-pool logic in
section 2 uses on the disease axis. If those diverge, section 1 and section 2
report inconsistent totals.

CVD tissue detection (`CVD_TISSUE_PATTERNS`) is independent of the disease
keyword net — it lets a sample enter the CVD pool via anatomy alone even
without a disease keyword (e.g. "left ventricle heart muscle" from a healthy-
donor study). Section 2's pool is the union: `is_cvd_tissue OR is_cvd_disease`.

Everything here is stateless and pure — no I/O, no matrix reads. `labels.py`
calls into these to produce the slim per-sample label table.
"""

from __future__ import annotations

import re
from typing import Iterable

# Priority order for assigning at most one disease category per sample.
# Cardiovascular comes first because section 3 cross-checks its row against the
# section-2 CVD-pool total on the disease axis — putting CVD later would mean a
# CVD-tagged cancer study (e.g. "atherosclerosis in breast cancer patients")
# gets counted under cancer and the two tables silently disagree.
DISEASE_PRIORITY: tuple[str, ...] = (
    "cardiovascular",
    "cancer_neoplasm",
    "neurological",
    "infectious",
    "metabolic_endocrine",
    "autoimmune",
    "respiratory",
    "renal",
    "musculoskeletal",
)

# Keyword nets per category. Match is case-insensitive substring; keywords
# were chosen to hit MeSH-style umbrella terms plus a few of the most common
# specific conditions inside each. `cardiovascular` mirrors elasticnet_todo.md.
DISEASE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "cardiovascular": (
        "cardiovasc",
        "cardiac",
        "heart failure",
        "myocardial infarct",
        "coronary artery",
        "atherosclerosis",
        "cardiomyopathy",
        "arrhythmia",
        "atrial fibrillation",
        "hypertension",
        "ischemic heart",
        "aortic",
        "vascular disease",
        "congestive heart",
        "cardiac hypertrophy",
        "cardiac fibrosis",
    ),
    "cancer_neoplasm": (
        "cancer",
        "carcinoma",
        "adenocarcinoma",
        "sarcoma",
        "leukemia",
        "lymphoma",
        "melanoma",
        "glioma",
        "glioblastoma",
        "tumor",
        "tumour",
        "neoplasm",
        "metastasis",
        "metastatic",
        "malignant",
        "myeloma",
        "myelodysplastic",
        "hepatocellular",
    ),
    "neurological": (
        "alzheimer",
        "parkinson",
        "huntington",
        "amyotrophic lateral sclerosis",
        "als disease",
        "multiple sclerosis",
        "epilep",
        "schizophrenia",
        "autism",
        "asd disorder",
        "dementia",
        "neurodegener",
        "stroke",
        "cerebral palsy",
        "migraine",
        "spinal cord injury",
    ),
    "infectious": (
        "hiv",
        "aids",
        "hepatitis",
        "tuberculosis",
        "influenza",
        "covid",
        "sars-cov",
        "sars cov",
        "coronavirus",
        "malaria",
        "sepsis",
        "bacterial infection",
        "viral infection",
        "pneumonia",
        "ebola",
        "zika",
        "dengue",
    ),
    "metabolic_endocrine": (
        "diabetes",
        "diabetic",
        "obesity",
        "obese",
        "metabolic syndrome",
        "insulin resistance",
        "thyroid",
        "hypothyroid",
        "hyperthyroid",
        "polycystic ovary",
        "pcos",
        "cushing",
        "adrenal insufficiency",
        "hyperlipidemia",
    ),
    "autoimmune": (
        "lupus",
        "systemic lupus",
        "rheumatoid arthritis",
        "psoriasis",
        "crohn",
        "ulcerative colitis",
        "inflammatory bowel disease",
        "ibd disease",
        "sjogren",
        "scleroderma",
        "vasculitis",
        "autoimmun",
        "type 1 diabetes",
        "graves disease",
        "hashimoto",
    ),
    "respiratory": (
        "asthma",
        "copd",
        "chronic obstructive pulmonary",
        "cystic fibrosis",
        "pulmonary fibrosis",
        "idiopathic pulmonary",
        "bronchi",
        "emphysema",
        "sleep apnea",
        "ards",
    ),
    "renal": (
        "renal",
        "kidney disease",
        "chronic kidney",
        "nephr",
        "glomerulonephritis",
        "dialysis",
        "polycystic kidney",
    ),
    "musculoskeletal": (
        "osteoarthritis",
        "osteoporosis",
        "arthritis",
        "musculoskeletal",
        "muscular dystrophy",
        "sarcopenia",
        "fibromyalgia",
        "ankylosing spondylitis",
    ),
}

# The "cardiovascular" keyword net is the disease axis of section 2's pool.
# Re-exported by name so cvd_breakdown.py doesn't have to duck-type through
# DISEASE_CATEGORIES — makes the section-3 cross-check obviously self-consistent.
CVD_DISEASE_KEYWORDS: tuple[str, ...] = DISEASE_CATEGORIES["cardiovascular"]

# Anatomical patterns: "sample sits on cardiovascular tissue". These are
# matched against source_name_ch1 + characteristics_ch1 + title, same as
# disease keywords. Chosen from what ARCHS4's tissue metadata actually
# surfaces (see labels.py inspection notes).
CVD_TISSUE_PATTERNS: tuple[str, ...] = (
    "heart",
    "cardiac muscle",
    "myocardium",
    "myocardial",
    "left ventricle",
    "right ventricle",
    "ventricular",
    "atrium",
    "atrial appendage",
    "aorta",
    "aortic",
    "coronary artery",
    "vascular smooth muscle",
    "endothelial",
    "cardiomyocyte",
    "cardiovascular",
)

# CVD subtype keyword nets. Priority order matters: heart failure & CAD are
# checked before generic "cardiomyopathy" so a DCM study doesn't get lumped
# into the "other cardiomyopathy" bucket.
#
# The last two entries are non-keyword fallback buckets that split what used
# to be a single "other/unspecified" catch-all — see `assign_cvd_subtype`:
#   * disease_matched_subtype_unresolved — is_cvd_disease is true but no
#     specific subtype keyword matched; a real disease-positive subset.
#   * tissue_only_disease_unconfirmed — sample entered the CVD pool via a
#     tissue keyword ONLY (is_cvd_disease is false). These are NOT confirmed
#     CVD cases and downstream consumers must not treat them as
#     disease-positive labels.
CVD_SUBTYPE_PRIORITY: tuple[str, ...] = (
    "heart_failure",
    "arrhythmia_afib",
    "coronary_artery_disease",
    "cardiomyopathy_other",
    "hypertension",
    "disease_matched_subtype_unresolved",
    "tissue_only_disease_unconfirmed",
)

# Legacy fallback label kept only so cvd_breakdown.py can canonicalize label
# tables produced before the split (see `canonicalize_legacy_subtype`).
LEGACY_UNSPECIFIED_CVD_SUBTYPE: str = "other_unspecified_cvd"

CVD_SUBTYPES: dict[str, tuple[str, ...]] = {
    "heart_failure": (
        "heart failure",
        "congestive heart",
        "hfref",
        "hfpef",
        "dilated cardiomyopathy",
        " dcm ",
        "(dcm)",
        "ischemic cardiomyopathy",
        " icm ",
        "(icm)",
    ),
    "arrhythmia_afib": (
        "arrhythmia",
        "atrial fibrillation",
        "afib",
        "a-fib",
        "ventricular tachycardia",
        "long qt",
        "brugada",
    ),
    "coronary_artery_disease": (
        "coronary artery disease",
        " cad ",
        "(cad)",
        "coronary artery",
        "myocardial infarct",
        "ischemic heart",
        "atherosclerosis",
    ),
    "cardiomyopathy_other": (
        "cardiomyopathy",
        "hypertrophic cardiomyopathy",
        "restrictive cardiomyopathy",
        "arrhythmogenic right ventricular cardiomyopathy",
        "arvc",
        "cardiac hypertrophy",
        "cardiac fibrosis",
    ),
    "hypertension": (
        "hypertension",
        "hypertensive",
        "high blood pressure",
    ),
    "disease_matched_subtype_unresolved": (
        # Non-keyword fallback for samples in the CVD pool where
        # `is_cvd_by_disease` is True but no specific subtype net matched.
        # `assign_cvd_subtype` never uses this list to match; present here
        # purely for documentation.
    ),
    "tissue_only_disease_unconfirmed": (
        # Non-keyword fallback for samples in the CVD pool via tissue keywords
        # only (`is_cvd_by_disease` False, `is_cvd_by_tissue` True). Disease
        # status is unconfirmed — treat as CVD-tissue-adjacent, not
        # disease-positive.
    ),
}


def _compile_keyword_regex(keywords: Iterable[str]) -> re.Pattern:
    """Case-insensitive substring OR-regex from an iterable of literal keywords."""
    escaped = [re.escape(k) for k in keywords]
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


# Precompiled per-category regexes; expensive to build, cheap to reuse across
# a million samples.
_DISEASE_RE: dict[str, re.Pattern] = {
    cat: _compile_keyword_regex(kws) for cat, kws in DISEASE_CATEGORIES.items()
}
_CVD_SUBTYPE_RE: dict[str, re.Pattern] = {
    st: _compile_keyword_regex(kws)
    for st, kws in CVD_SUBTYPES.items()
    if kws  # skip the empty catch-all
}
_CVD_TISSUE_RE: re.Pattern = _compile_keyword_regex(CVD_TISSUE_PATTERNS)


def assign_disease_category(haystack: str) -> str:
    """Return the disease category for one sample's concatenated metadata.

    `haystack` is the lowercased concatenation of title, source_name_ch1, and
    characteristics_ch1 for a single sample. Returns "unclear_unlabeled" when
    no category keyword matches — required by section 1 (unmatched samples
    must not disappear from the table).
    """
    for cat in DISEASE_PRIORITY:
        if _DISEASE_RE[cat].search(haystack):
            return cat
    return "unclear_unlabeled"


def is_cvd_by_disease(haystack: str) -> bool:
    """True iff any cardiovascular disease keyword hits the haystack."""
    return _DISEASE_RE["cardiovascular"].search(haystack) is not None


def is_cvd_by_tissue(haystack: str) -> bool:
    """True iff any cardiovascular anatomical keyword hits the haystack."""
    return _CVD_TISSUE_RE.search(haystack) is not None


def assign_cvd_subtype(haystack: str, is_cvd_disease: bool) -> str:
    """Return the CVD subtype for a sample known to be in the CVD pool.

    Callers must have already verified `is_cvd_by_disease or is_cvd_by_tissue`.
    `is_cvd_disease` gates BOTH the specific-subtype lookup and the fallback:

      * `is_cvd_disease` True + specific subtype net hits → that subtype.
      * `is_cvd_disease` True + no specific subtype hit →
        `disease_matched_subtype_unresolved` (real disease-positive subset
        with an unresolved subtype label).
      * `is_cvd_disease` False (tissue-only) → `tissue_only_disease_unconfirmed`
        UNCONDITIONALLY. A haystack whose only CVD hit was a tissue keyword
        cannot be reported as a disease-confirmed subtype even if a subtype
        acronym (e.g. "hfref", "dcm", "cad", "afib") happens to appear; the
        broad CVD disease net's failure means we have insufficient evidence
        of the disease itself. These samples MUST NOT be reported as
        disease-positive by any downstream consumer (e.g. the linear probe).
    """
    if not is_cvd_disease:
        return "tissue_only_disease_unconfirmed"
    for st in CVD_SUBTYPE_PRIORITY:
        if st not in _CVD_SUBTYPE_RE:
            continue
        if _CVD_SUBTYPE_RE[st].search(haystack):
            return st
    return "disease_matched_subtype_unresolved"


def canonicalize_legacy_subtype(subtype: str, is_cvd_disease: bool) -> str:
    """Map a subtype value read from an older label table onto the current
    two-bucket split. Idempotent for values already produced by the current
    `assign_cvd_subtype`.
    """
    if subtype != LEGACY_UNSPECIFIED_CVD_SUBTYPE:
        return subtype
    return (
        "disease_matched_subtype_unresolved"
        if is_cvd_disease
        else "tissue_only_disease_unconfirmed"
    )


def match_all_disease_categories(haystack: str) -> tuple[str, ...]:
    """Return every disease category whose keyword net hits the haystack.

    Unlike `assign_disease_category` (which is first-match-wins), this returns
    the full set so callers can detect comorbid samples (matching multiple
    categories) — used by Section 3's category-overlap count.
    """
    hits: list[str] = []
    for cat in DISEASE_PRIORITY:
        if _DISEASE_RE[cat].search(haystack):
            hits.append(cat)
    return tuple(hits)


def disease_category_display_name(cat: str) -> str:
    """Human-facing label for a category slug (used in write-up tables)."""
    return {
        "cardiovascular": "Cardiovascular",
        "cancer_neoplasm": "Cancer / neoplasm",
        "neurological": "Neurological",
        "infectious": "Infectious",
        "metabolic_endocrine": "Metabolic / endocrine",
        "autoimmune": "Autoimmune",
        "respiratory": "Respiratory",
        "renal": "Renal",
        "musculoskeletal": "Musculoskeletal",
        "unclear_unlabeled": "Unclear / unlabeled",
    }[cat]


def cvd_subtype_display_name(st: str) -> str:
    return {
        "heart_failure": "Heart failure (DCM + ICM)",
        "arrhythmia_afib": "Arrhythmia / AFib",
        "coronary_artery_disease": "Coronary artery disease",
        "cardiomyopathy_other": "Cardiomyopathy (other)",
        "hypertension": "Hypertension",
        "disease_matched_subtype_unresolved": "Disease-matched, subtype unresolved",
        "tissue_only_disease_unconfirmed": "Tissue-only, disease status unconfirmed",
        # Kept only for defensive rendering of stale label tables; canonicalized
        # away by cvd_breakdown before display.
        "other_unspecified_cvd": "Other / unspecified CVD (legacy)",
    }[st]
