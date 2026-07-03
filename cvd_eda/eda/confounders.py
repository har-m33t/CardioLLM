"""Correlate top PCs against sample metadata covariates.

The goal is to flag the classic failure mode described in the task brief:
a non-biological variable (series/batch, sex, sequencing center) explains
more variance in the first few PCs than the disease label does. When that
happens we want the reviewer — and the LLM interpretation — to see it
before anyone trains a model.

Two association measures:
    * Categorical covariate  → eta² (variance explained by group means).
    * Continuous covariate   → Pearson r².

Both are scale-free, bounded in [0, 1], and directly comparable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


# Metadata columns we always test when they exist, plus anything a caller
# passes explicitly. Kept short so the report doesn't drown in noise.
DEFAULT_COVARIATES: tuple[str, ...] = (
    "label",
    "series_id",
    "sex", "gender",
    "race", "ethnicity",
    "tissue", "tissue_type", "body_site",
    "age", "age_at_diagnosis", "age_at_index",
    "rel_matched_keyword",
)


@dataclass
class ConfounderScreen:
    per_pc: pd.DataFrame                            # rows=PC, cols=covariate, values=eta²/r²
    flagged: List[dict] = field(default_factory=list)
    kind: Dict[str, str] = field(default_factory=dict)  # covariate → "categorical"/"continuous"


def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    """One-way eta² = SS_between / SS_total.

    Returns 0.0 when a single group has all the mass (no between-group
    variance to speak of) or when every value is identical.
    """
    df = pd.DataFrame({"value": values, "group": groups}).dropna()
    if df.empty or df["group"].nunique() < 2:
        return 0.0
    grand = df["value"].mean()
    group_means = df.groupby("group")["value"].mean()
    group_sizes = df.groupby("group")["value"].size()
    ss_between = float(((group_means - grand) ** 2 * group_sizes).sum())
    ss_total = float(((df["value"] - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    return ss_between / ss_total


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 3 or df["x"].std(ddof=0) == 0 or df["y"].std(ddof=0) == 0:
        return 0.0
    r = float(np.corrcoef(df["x"], df["y"])[0, 1])
    if not np.isfinite(r):
        return 0.0
    return r * r


def _covariate_kind(series: pd.Series) -> str:
    """Categorical if dtype is object/category OR fewer than 6 unique numeric values.

    The 6-unique heuristic catches ordinals that were stored as integers
    (0/1 sex codes, 1..5 stage) where treating them as continuous would
    understate their association.
    """
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.dropna().nunique()
        if nunique < 6:
            return "categorical"
        return "continuous"
    return "categorical"


def screen(
    pca_scores: pd.DataFrame,
    sample_meta: pd.DataFrame,
    *,
    covariates: Optional[List[str]] = None,
    top_pcs: int = 5,
    flag_threshold: float = 0.30,
) -> ConfounderScreen:
    """Compute per-PC association with each candidate covariate."""
    if covariates is None:
        covariates = [c for c in DEFAULT_COVARIATES if c in sample_meta.columns]

    pcs = list(pca_scores.columns[:top_pcs])
    result = pd.DataFrame(index=pcs, columns=covariates, dtype=float)
    kind_map: Dict[str, str] = {}
    flagged: List[dict] = []

    for cov in covariates:
        if cov not in sample_meta.columns:
            continue
        series = sample_meta[cov].reindex(pca_scores.index)
        kind = _covariate_kind(series)
        kind_map[cov] = kind
        for pc in pcs:
            pc_values = pca_scores[pc].to_numpy(dtype=float)
            if kind == "categorical":
                assoc = _eta_squared(pc_values, series.astype(str).values)
            else:
                assoc = _pearson_r2(pc_values, series.astype(float).values)
            result.loc[pc, cov] = round(assoc, 4)
            if assoc >= flag_threshold:
                flagged.append(
                    {
                        "pc": pc,
                        "covariate": cov,
                        "association": round(assoc, 4),
                        "kind": kind,
                    }
                )

    flagged.sort(key=lambda r: r["association"], reverse=True)
    return ConfounderScreen(per_pc=result, flagged=flagged, kind=kind_map)
