"""Structured JSON audit log for Task 6.

Task 7 (Reporting Agent) reads every ``eda_run_log_*.json`` this module
writes, so the shape has to be stable across dataset runs. Mirrors
:class:`cvd_eda.processing.logging_utils.ProcessingLog` — same field
names where it makes sense so Task 7 has one shape to parse.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List


LOG = logging.getLogger(__name__)


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


@dataclass
class EDALog:
    dataset: str
    started_at: str = field(default_factory=_now_iso)
    finished_at: str = ""
    config: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    steps: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)
    interpretations: dict = field(default_factory=dict)
    flagged_confounders: List[dict] = field(default_factory=list)
    environment: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_step(self, name: str, report: Any) -> None:
        if hasattr(report, "__dataclass_fields__"):
            report = asdict(report)
        self.steps[name] = report

    def add_plot(self, name: str, path: Path) -> None:
        self.plots[name] = str(path)

    def add_interpretation(self, plot_name: str, text: str, cached: bool) -> None:
        self.interpretations[plot_name] = {"text": text, "cached": cached}

    def add_warning(self, message: str) -> None:
        LOG.warning(message)
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        LOG.error(message)
        self.errors.append(message)

    def finalize(self, output_path: Path) -> None:
        self.finished_at = _now_iso()
        self.environment = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        # Report objects can carry non-JSON-serializable pandas objects
        # (e.g. per-sample QC DataFrames). Strip those; the CSV summary
        # already persists them in a Task 7-consumable shape.
        _strip_dataframes(payload)
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2, default=str)
        LOG.info("Wrote EDA log → %s", output_path)


def _strip_dataframes(obj: Any) -> None:
    """In-place: drop pandas DataFrame values from a nested dict.

    We keep the numeric summaries (dicts, floats, lists) but replace any
    DataFrame with a shape marker. Reporting still has enough to work with
    via the ``outputs`` section, which points at the CSVs.
    """
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            if _is_dataframe(v):
                obj[k] = {"__dataframe__": True, "shape": list(v.shape)}
            else:
                _strip_dataframes(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if _is_dataframe(v):
                obj[i] = {"__dataframe__": True, "shape": list(v.shape)}
            else:
                _strip_dataframes(v)


def _is_dataframe(v: Any) -> bool:
    # Duck-typed to avoid a top-level pandas import for a check.
    return hasattr(v, "shape") and hasattr(v, "to_parquet") and hasattr(v, "columns")
