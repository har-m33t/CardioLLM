"""LLM-backed label proposer for Task 5.

Modelled on :mod:`cvd_eda.curation.llm` — same on-disk cache shape, same
JSON-out-of-text parsing, same retry loop — so operators only have one
pattern to reason about. Two things differ from Task 3:

* Default model is ``claude-opus-4-8``. Task 5 is the single highest-leverage
  step in the pipeline, so we don't downgrade to Haiku for cost.
* The prompt biases toward ``uncertain`` over a low-confidence guess, and
  requires the evidence quote to be a verbatim substring of the input.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from cvd_eda.labeling.schema import LABEL_VOCAB, UNCERTAIN_LABEL, is_valid_label


LOG = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-opus-4-8"


SYSTEM_PROMPT = f"""You are labeling RNA-seq samples for a cardiovascular-disease (CVD) study. Each sample has already been called CVD-relevant by an upstream classifier; your job is to say what outcome label it should carry into a supervised model.

Allowed labels: {", ".join(LABEL_VOCAB)}.

Guidance:
- Use "case" / "control" when the study is a two-arm disease-vs-healthy contrast and the sample's arm is stated (case = disease, control = healthy).
- Use a specific subtype (HCM, DCM, HFrEF, HFpEF, HF, MI, CAD, AF, PAH, AS, other_CVD) only when the phenotype is explicitly stated. Do not infer a subtype from anatomy alone.
- Use "uncertain" whenever the metadata does not support a confident decision. Prefer uncertain over a low-confidence guess. Task 5 is the highest-leverage step in the pipeline; a wrong label silently corrupts every downstream result, but an "uncertain" row costs a reviewer one minute.

Return ONLY a JSON object with fields:
- proposed_label: one of the allowed labels above.
- confidence: float 0.0-1.0. Anchor: >= 0.9 = phenotype is explicit in the sample-level fields; ~0.5 = inferred from the series description but sample-level fields are silent; <= 0.3 = ambiguous. When you return "uncertain", set confidence to your certainty that the metadata is genuinely insufficient (not to zero).
- evidence_quote: a verbatim substring of the sample metadata or series description that drove the label decision. Do NOT paraphrase; a reviewer will grep the source for it.
- uncertain_reason: only populated when proposed_label is "uncertain". Short description of what is missing: "no control arm described", "phenotype field blank", "series mixes disease and healthy tissue without per-sample marker", etc. Empty string otherwise.
"""


USER_TEMPLATE = """Sample id: {sample_id}
Series id: {series_id}

Sample metadata:
\"\"\"
{sample_text}
\"\"\"
{series_block}
Propose a label for this sample."""


SERIES_BLOCK_TEMPLATE = """
Series-level description (fetched from GEO for extra context):
\"\"\"
{series_text}
\"\"\"
"""


@dataclass
class LabelLLMResult:
    proposed_label: str
    confidence: float
    evidence_quote: str
    uncertain_reason: str
    model: str
    cached: bool


class LabelProposer:
    """Anthropic-backed label proposer with an on-disk JSON cache."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
        max_tokens: int = 500,
    ):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "The `anthropic` package is required for Task 5 labeling. "
                "Install with: uv pip install anthropic"
            ) from exc

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Task 5 is the highest-leverage "
                "step in the pipeline; there is no --disable-llm escape hatch. "
                "Set the key (or use the `ant auth login` OAuth flow) and rerun."
            )

        from anthropic import Anthropic

        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self._client = Anthropic()

        self._cache_path: Path | None = None
        self._cache: dict[str, dict] = {}
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / f"label_cache_{_safe_slug(model)}.json"
            if self._cache_path.exists():
                try:
                    self._cache = json.loads(self._cache_path.read_text())
                except json.JSONDecodeError:
                    LOG.warning(
                        "Label cache at %s is corrupt; starting fresh",
                        self._cache_path,
                    )
                    self._cache = {}

        self.call_count = 0
        self.cache_hit_count = 0

    # ------------------------------------------------------------------

    def propose(
        self,
        *,
        sample_id: str,
        series_id: str,
        sample_text: str,
        series_text: str = "",
    ) -> LabelLLMResult:
        series_block = (
            SERIES_BLOCK_TEMPLATE.format(series_text=series_text.strip())
            if series_text
            else ""
        )
        prompt = USER_TEMPLATE.format(
            sample_id=sample_id,
            series_id=series_id or "(unknown)",
            sample_text=sample_text.strip() or "(empty)",
            series_block=series_block,
        )
        cache_key = _hash(self.model, SYSTEM_PROMPT, prompt)

        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self.cache_hit_count += 1
            return LabelLLMResult(
                proposed_label=cached["proposed_label"],
                confidence=cached["confidence"],
                evidence_quote=cached["evidence_quote"],
                uncertain_reason=cached["uncertain_reason"],
                model=self.model,
                cached=True,
            )

        parsed = self._call_with_retry(prompt)
        self._cache[cache_key] = parsed
        self._persist_cache()
        self.call_count += 1
        return LabelLLMResult(
            proposed_label=parsed["proposed_label"],
            confidence=parsed["confidence"],
            evidence_quote=parsed["evidence_quote"],
            uncertain_reason=parsed["uncertain_reason"],
            model=self.model,
            cached=False,
        )

    # ------------------------------------------------------------------

    def _call_with_retry(self, prompt: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = _extract_text(message)
                return _parse_json_response(raw)
            except Exception as exc:  # broad: SDK + JSON parse
                last_exc = exc
                if attempt < self.max_retries:
                    delay = 2 ** (attempt - 1)
                    LOG.warning(
                        "Label LLM call failed (attempt %s/%s): %s; retrying in %ss",
                        attempt,
                        self.max_retries,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"Label LLM classification failed after {self.max_retries} attempts"
        ) from last_exc

    def _persist_cache(self) -> None:
        if self._cache_path is None:
            return
        tmp = self._cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        tmp.replace(self._cache_path)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _safe_slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def _extract_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError(f"No text block in Anthropic response: {message}")


def _parse_json_response(raw: str) -> dict:
    """Parse the JSON object out of the model's response.

    Tolerates responses wrapped in a ```json fence or with a leading sentence,
    matching :func:`cvd_eda.curation.llm._parse_json_response` so operators
    have one shape to reason about.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object in response: {raw!r}")
    obj = json.loads(text[start : end + 1])

    label = str(obj.get("proposed_label", "")).strip()
    if not is_valid_label(label):
        raise ValueError(
            f"Unexpected proposed_label {label!r}; must be one of {LABEL_VOCAB}"
        )

    confidence = float(obj.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    evidence_quote = str(obj.get("evidence_quote", "")).strip()

    uncertain_reason = str(obj.get("uncertain_reason", "")).strip()
    if label != UNCERTAIN_LABEL:
        # Model is allowed to leave this populated but we don't propagate it —
        # the semantics of the CSV column are "populated iff uncertain".
        uncertain_reason = ""

    return {
        "proposed_label": label,
        "confidence": confidence,
        "evidence_quote": evidence_quote,
        "uncertain_reason": uncertain_reason,
    }
