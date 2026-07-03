"""LLM-backed plot interpretation.

Task 6's brief explicitly asks for "a short written interpretation of each
plot (not just the image)". This module packages up one plot's numeric
context — top-PC variance ratios, flagged confounders, cohort counts —
and asks the model for 2-4 sentences the reviewer can read alongside the
PNG.

Modelled on :mod:`cvd_eda.labeling.llm` — same Anthropic client wrapper,
same on-disk cache keyed by ``sha256(model || system || prompt)``, same
JSON-out-of-text parsing. Two things differ:

* No retries by default beyond three attempts; interpretations are
  advisory, not gating.
* An ``--disable-llm`` escape hatch exists: interpretation is a
  nice-to-have, not correctness-critical, and we do not want to hard-fail a
  full EDA run when the API is unreachable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path


LOG = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-opus-4-8"


SYSTEM_PROMPT = """You are annotating exploratory-data-analysis plots for a cardiovascular-disease RNA-seq study. A biomedical reviewer will read your interpretation next to the plot itself.

You will receive:
- The plot type (e.g. "PCA scatter colored by series_id").
- A compact JSON blob of the underlying summary statistics.

Return 2 to 4 sentences that:
1. State what the numbers say (not what the plot "looks like" — you cannot see it).
2. Call out anything that would change a downstream decision — dominant batch effects, one label class swamping the cohort, an obvious outlier tissue, etc.
3. If nothing surprising is present, say so plainly. Do not manufacture caveats.

Return ONLY a JSON object with a single key: interpretation.
"""


USER_TEMPLATE = """Plot: {plot_name}

Summary statistics:
```json
{context_json}
```

Write the interpretation."""


@dataclass
class InterpretationResult:
    plot_name: str
    interpretation: str
    model: str
    cached: bool


class PlotInterpreter:
    """Anthropic-backed interpreter with on-disk JSON cache.

    Constructor is a no-op when ``disabled=True`` — a stubbed interpreter
    that returns ``"(interpretation disabled)"`` for every plot. Lets the
    smoke test and CI runs stay offline without duplicating the call sites.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_dir: str | Path | None = None,
        disabled: bool = False,
        max_retries: int = 3,
        max_tokens: int = 400,
    ):
        self.model = model
        self.disabled = disabled
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.call_count = 0
        self.cache_hit_count = 0
        self._client = None
        self._cache_path: Path | None = None
        self._cache: dict[str, dict] = {}

        if disabled:
            return

        try:
            import anthropic  # noqa: F401
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise ImportError(
                "The `anthropic` package is required for LLM interpretation. "
                "Install with `uv pip install anthropic`, or rerun with "
                "--disable-llm-interpretation to skip this step."
            ) from exc

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Rerun with "
                "--disable-llm-interpretation if you want to skip Task 6's "
                "written interpretations."
            )

        from anthropic import Anthropic
        # Match Task 3's client construction: passing a pre-built bare
        # httpx.Client avoids a newer-httpx-only kwarg in anthropic's default.
        self._client = Anthropic(http_client=httpx.Client(timeout=60.0))

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_path = cache_dir / f"interpret_cache_{_safe_slug(model)}.json"
            if self._cache_path.exists():
                try:
                    self._cache = json.loads(self._cache_path.read_text())
                except json.JSONDecodeError:
                    LOG.warning(
                        "Interpretation cache at %s is corrupt; starting fresh",
                        self._cache_path,
                    )
                    self._cache = {}

    # ------------------------------------------------------------------

    def interpret(self, plot_name: str, context: dict) -> InterpretationResult:
        if self.disabled:
            return InterpretationResult(
                plot_name=plot_name,
                interpretation="(interpretation disabled)",
                model=self.model,
                cached=False,
            )

        context_json = json.dumps(context, indent=2, sort_keys=True, default=str)
        prompt = USER_TEMPLATE.format(plot_name=plot_name, context_json=context_json)
        cache_key = _hash(self.model, SYSTEM_PROMPT, prompt)

        if cache_key in self._cache:
            self.cache_hit_count += 1
            return InterpretationResult(
                plot_name=plot_name,
                interpretation=self._cache[cache_key]["interpretation"],
                model=self.model,
                cached=True,
            )

        parsed = self._call_with_retry(prompt)
        self._cache[cache_key] = parsed
        self._persist_cache()
        self.call_count += 1
        return InterpretationResult(
            plot_name=plot_name,
            interpretation=parsed["interpretation"],
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
                        "Interpretation LLM call failed (attempt %s/%s): %s; "
                        "retrying in %ss",
                        attempt, self.max_retries, exc, delay,
                    )
                    time.sleep(delay)
        raise RuntimeError(
            f"Interpretation LLM call failed after {self.max_retries} attempts"
        ) from last_exc

    def _persist_cache(self) -> None:
        if self._cache_path is None:
            return
        tmp = self._cache_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._cache, indent=2, sort_keys=True))
        tmp.replace(self._cache_path)


# ---------------------------------------------------------------------- helpers


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
    interpretation = str(obj.get("interpretation", "")).strip()
    if not interpretation:
        raise ValueError(f"Empty interpretation: {obj!r}")
    return {"interpretation": interpretation}
