"""Label vocabulary and output schema for Task 5.

The set of labels here is deliberately pinned in one module — both the LLM
prompt and the CSV writer read from it, so the two can't drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


CASE_CONTROL_LABELS: tuple[str, ...] = ("case", "control")

# Subtype labels. Kept intentionally short — every additional label the LLM
# has to pick from adds noise. If the reviewer needs a finer distinction they
# can override the proposal manually before Task 6 consumes the file.
SUBTYPE_LABELS: tuple[str, ...] = (
    "HCM",       # Hypertrophic cardiomyopathy
    "DCM",       # Dilated cardiomyopathy
    "HF",        # Heart failure, unspecified / mixed
    "HFrEF",     # Heart failure with reduced ejection fraction
    "HFpEF",     # Heart failure with preserved ejection fraction
    "MI",        # Myocardial infarction
    "CAD",       # Coronary artery disease
    "AF",        # Atrial fibrillation
    "PAH",       # Pulmonary arterial hypertension
    "AS",        # Aortic stenosis
    "other_CVD",
)

UNCERTAIN_LABEL = "uncertain"

LABEL_VOCAB: tuple[str, ...] = CASE_CONTROL_LABELS + SUBTYPE_LABELS + (UNCERTAIN_LABEL,)


@dataclass
class LabelProposal:
    """One row in ``label_proposals.csv``.

    ``evidence_quote`` is required to be a verbatim substring of the sample or
    series text so a reviewer can grep the source metadata for it as a
    spot-check.
    """

    sample_id: str
    proposed_label: str
    confidence: float
    evidence_quote: str
    uncertain_reason: str
    source_series_id: str
    model: str
    cached: bool

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "proposed_label",
    "confidence",
    "evidence_quote",
    "uncertain_reason",
    "source_series_id",
    "model",
    "cached",
)


def is_valid_label(label: str) -> bool:
    return label in LABEL_VOCAB
