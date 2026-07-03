"""Sample-relationship analyses: PCA, sample-sample correlation, optional t-SNE.

Everything runs on a matrix that has been reduced to the
``top_variable_genes`` most variable rows so runtime stays bounded on
realistic ARCHS4/RECOUNT3 cohort sizes. The reduced matrix is exposed on
the report so downstream steps (confounder screen, correlation heatmap)
don't recompute it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


LOG = logging.getLogger(__name__)


@dataclass
class PCAResult:
    scores: pd.DataFrame                   # (n_samples, n_components) — index sample_id
    explained_variance_ratio: List[float]
    n_components: int
    n_genes_used: int


@dataclass
class RelationshipsReport:
    pca: PCAResult
    sample_corr: pd.DataFrame              # (n_samples, n_samples)
    linkage_order: List[str]               # sample_ids in leaf order
    tsne: Optional[pd.DataFrame] = None    # (n_samples, 2); None if skipped
    tsne_skip_reason: str = ""
    top_variable_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)


def _top_variable(expression: pd.DataFrame, k: int) -> pd.DataFrame:
    if expression.shape[0] <= k:
        return expression
    var = expression.var(axis=1)
    keep = var.sort_values(ascending=False).head(k).index
    return expression.loc[keep]


def run_pca(expression: pd.DataFrame, n_components: int) -> PCAResult:
    """PCA on samples-as-rows (expression is genes × samples)."""
    from sklearn.decomposition import PCA

    n_samples = expression.shape[1]
    max_components = min(n_components, n_samples - 1, expression.shape[0])
    max_components = max(max_components, 1)
    if max_components < n_components:
        LOG.info(
            "Capping PCA to %d components (had %d samples / %d genes).",
            max_components, n_samples, expression.shape[0],
        )

    X = expression.to_numpy().T  # (n_samples, n_genes)
    pca = PCA(n_components=max_components, random_state=0)
    scores = pca.fit_transform(X)
    scores_df = pd.DataFrame(
        scores,
        index=expression.columns,
        columns=[f"PC{i + 1}" for i in range(max_components)],
    )
    return PCAResult(
        scores=scores_df,
        explained_variance_ratio=[float(r) for r in pca.explained_variance_ratio_],
        n_components=max_components,
        n_genes_used=expression.shape[0],
    )


def sample_correlation(expression: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """(n_samples, n_samples) correlation matrix; sample IDs as both axes."""
    return expression.corr(method=method)


def hierarchical_order(corr: pd.DataFrame) -> List[str]:
    """Return sample IDs in the order produced by average-linkage clustering
    on ``1 - corr`` distance. Falls back to input order when clustering
    would degenerate (single sample).
    """
    if corr.shape[0] < 2:
        return list(corr.index.astype(str))

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    distance = 1.0 - corr.to_numpy()
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    # squareform expects a symmetric matrix with zero diagonal.
    distance = (distance + distance.T) / 2.0
    condensed = squareform(distance, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    return [str(corr.index[i]) for i in order]


def run_tsne(
    scores: pd.DataFrame,
    perplexity: float,
    random_state: int = 0,
) -> pd.DataFrame:
    """2D t-SNE on the PCA scores (dense enough to work; fast).

    Caller is expected to gate on ``len(scores) > perplexity + 1``; we
    raise below that to match sklearn's own contract.
    """
    from sklearn.manifold import TSNE

    if len(scores) <= perplexity:
        raise ValueError(
            f"t-SNE requires n_samples ({len(scores)}) > perplexity ({perplexity})."
        )
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(scores.to_numpy())
    return pd.DataFrame(
        embedding, index=scores.index, columns=["tSNE1", "tSNE2"]
    )


def analyze(
    expression: pd.DataFrame,
    *,
    top_variable_genes: int,
    n_pca_components: int,
    run_tsne_flag: bool,
    tsne_perplexity: float,
    tsne_random_state: int,
) -> RelationshipsReport:
    """Full sample-relationship suite in one call."""
    reduced = _top_variable(expression, top_variable_genes)
    pca_result = run_pca(reduced, n_pca_components)
    corr = sample_correlation(reduced)
    order = hierarchical_order(corr)

    tsne_df: Optional[pd.DataFrame] = None
    skip_reason = ""
    if run_tsne_flag:
        if len(pca_result.scores) <= tsne_perplexity:
            skip_reason = (
                f"n_samples={len(pca_result.scores)} <= perplexity={tsne_perplexity}"
            )
            LOG.info("Skipping t-SNE: %s", skip_reason)
        else:
            tsne_df = run_tsne(pca_result.scores, tsne_perplexity, tsne_random_state)
    else:
        skip_reason = "run_tsne=False in config"

    return RelationshipsReport(
        pca=pca_result,
        sample_corr=corr,
        linkage_order=order,
        tsne=tsne_df,
        tsne_skip_reason=skip_reason,
        top_variable_matrix=reduced,
    )
