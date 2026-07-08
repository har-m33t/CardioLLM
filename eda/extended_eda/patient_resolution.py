"""
patient_resolution.py — resolve a per-sample patient identifier from
`characteristics_ch1` when one is present, and fall back to sample-as-own-patient
otherwise (revised_eda_tod.md §0).

Why a fallback is needed
------------------------
GEO's `characteristics_ch1` is free-text, `key: value` pairs separated by
commas or newlines. Only ~18% of ARCHS4 samples (empirically, sampled 5k
rows on the 2026-07 release) declare *any* patient/subject/donor key.
Everything else has no resolvable identifier, and the section 0 mandate is to
document the fallback explicitly rather than silently assume — the standard
fallback (used here) is treat-each-sample-as-its-own-patient, which upper-bounds
the patient count and preserves per-disease sample totals.

What "same patient" means
-------------------------
Two samples are treated as the same patient iff:
  1. Both parsed a non-empty patient id from `characteristics_ch1`, AND
  2. They belong to the same `series_id` (GEO series), AND
  3. Their normalized patient ids compare equal.

Requiring same-series equality is the important guard: patient id "1" in GSE
A has nothing to do with patient id "1" in GSE B. Cross-series patient
identity would require a linking process (name/DOB/etc.) that GEO does not
expose. Without the series scope, we would collapse thousands of unrelated
studies into fake mega-patients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Every substring here is matched inside a `<key>: <value>` scan of
# characteristics_ch1. Ordering matters — the *longer* variants come first so
# "subject id" wins over the shorter "subject" prefix; picking the shorter one
# would truncate legitimate values.
PATIENT_KEY_PATTERNS: tuple[str, ...] = (
    "patient id",
    "patientid",
    "patient_id",
    "subject id",
    "subjectid",
    "subject_id",
    "individual id",
    "individualid",
    "individual_id",
    "donor id",
    "donorid",
    "donor_id",
    "patient",
    "subject",
    "individual",
    "donor",
)

# Combined regex: `(?P<key>one of the patterns) [space or nothing] : value`.
# Value runs to the next comma / semicolon / newline / end-of-string (GEO's
# usual field separators). `re.IGNORECASE` because keys arrive with any
# capitalization ("Subject", "Patient ID", "donorID", ...).
_PATIENT_RE = re.compile(
    r"(?:^|[,;\n])\s*(?P<key>"
    + "|".join(re.escape(k) for k in PATIENT_KEY_PATTERNS)
    + r")\s*[:=]\s*(?P<value>[^,;\n]+)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class PatientCoverage:
    """Aggregate stats emitted alongside the label table; embedded verbatim in
    the definitions/write-up so the "per patient" numbers are auditable."""
    n_samples: int
    n_with_key: int
    coverage_pct: float
    n_unique_patients_resolved: int
    n_unique_patients_fallback: int


def extract_patient_id(characteristics: str) -> str | None:
    """Return a normalized patient/subject/donor id from one sample's
    characteristics_ch1 string, or None if no such key/value is present.

    `characteristics` may be any casing / arbitrary whitespace. The regex
    scans for the first `(patient|subject|individual|donor)[_ ]?id?: <value>`
    fragment and returns `<value>` stripped and lower-cased. Empty values
    ("subject: -", "patient id: N/A", "donor: unknown") are treated as
    absent — a placeholder is worse than nothing because it would collapse
    all placeholder-having samples in a series into one fake patient.
    """
    if not characteristics:
        return None
    m = _PATIENT_RE.search(characteristics)
    if not m:
        return None
    val = m.group("value").strip().lower()
    if not val:
        return None
    # Filter out the common "no value here" placeholders.
    if val in {"na", "n/a", "none", "unknown", "-", "?", "not available"}:
        return None
    return val


def compose_patient_key(series_id: str | None, patient_id: str | None) -> str | None:
    """Combine series + patient id into the canonical "same patient" key.

    Returns None if either component is missing — the caller then applies the
    sample-as-own-patient fallback, which the label table records by writing
    the sample's own geo_accession as the patient key. This keeps every
    per-patient aggregation a plain pandas groupby without special-casing.
    """
    if not series_id or not patient_id:
        return None
    return f"{series_id}::{patient_id}"


def summarize_coverage(
    n_samples: int,
    n_with_key: int,
    n_unique_resolved: int,
    n_unique_fallback: int,
) -> PatientCoverage:
    pct = round(100.0 * n_with_key / n_samples, 4) if n_samples else 0.0
    return PatientCoverage(
        n_samples=int(n_samples),
        n_with_key=int(n_with_key),
        coverage_pct=pct,
        n_unique_patients_resolved=int(n_unique_resolved),
        n_unique_patients_fallback=int(n_unique_fallback),
    )
