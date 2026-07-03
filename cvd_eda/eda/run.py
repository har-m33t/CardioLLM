"""CLI entrypoint for Task 6: EDA Agent.

Drives one dataset through cohort composition, per-sample QC, PCA /
sample-sample correlation / optional t-SNE, and a top-PC-vs-covariate
confounder screen. Writes plots, a per-dataset summary-stats CSV, the
LLM interpretations, and a JSON audit log that Task 7 will consume.

The Task 5 human-review checkpoint is enforced here: the labels file must
have ``.reviewed.`` in its filename, or the caller must pass the explicit
``--allow-unreviewed-labels`` override (kept intentionally verbose to make
review-bypass obvious in shell history).

Typical invocation::

    python -m cvd_eda.eda.run \
        --dataset archs4 \
        --matrix       cvd_eda/logs/task4_out/cvd_matrix_archs4_normalized.parquet \
        --sample-meta  cvd_eda/logs/task4_out/cvd_sample_meta_archs4.parquet \
        --labels       cvd_eda/logs/label_proposals_archs4.reviewed.csv \
        --output-dir   cvd_eda/logs/task6_out/ \
        --llm-cache    cvd_eda/logs/eda_llm_cache/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from . import cohort, confounders, plotting, qc, relationships
from .config import EDAConfig
from .interpret import DEFAULT_MODEL, PlotInterpreter
from .loaders import load_dataset, load_gene_biotype_map, load_labels
from .logging_utils import EDALog


log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #


@dataclass
class RunConfig:
    dataset: str
    matrix_parquet: Path
    sample_meta_parquet: Path
    labels_csv: Path
    output_dir: Path
    biotype_tsv: Optional[Path] = None
    llm_cache_dir: Optional[Path] = None
    model: str = DEFAULT_MODEL
    disable_llm_interpretation: bool = False
    eda: EDAConfig = None  # populated by _cfg_from_args


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def _safe(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


# --------------------------------------------------------------------------- #
# Review-file gate — mirrors the Task 5 STOP banner
# --------------------------------------------------------------------------- #


_REVIEW_TOKEN = ".reviewed."


def _gate_review_file(labels_csv: Path, allow_unreviewed: bool) -> None:
    """Reject Task 5's raw output unless the caller explicitly overrides.

    Task 5's README instructs reviewers to save the corrected file as
    ``label_proposals.reviewed.csv``. Enforcing the convention here means a
    forgotten review step trips the CLI instead of silently corrupting Task 6.
    """
    name = labels_csv.name
    if _REVIEW_TOKEN in name or allow_unreviewed:
        return
    print(
        f"error: --labels {labels_csv} does not contain '{_REVIEW_TOKEN}' in the "
        f"filename. Task 5 emits a *proposal* file that requires human review "
        f"before Task 6 consumes it. Save the reviewed version as "
        f"`label_proposals.reviewed.csv` and pass that path, or explicitly opt "
        f"out with --allow-unreviewed-labels (do not do this in a real run).",
        file=sys.stderr,
    )
    raise SystemExit(2)


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


def _plot_coloring_columns(ds) -> list[str]:
    """Metadata columns to render as PCA/t-SNE color overlays.

    We always color by label. Beyond that, we opportunistically add
    series_id and any tissue/sex column that exists in the sample meta.
    """
    cols = ["label"]
    for extra in ("series_id", "sex", "gender", "tissue", "tissue_type", "body_site"):
        if extra in ds.sample_meta.columns and ds.sample_meta[extra].notna().any():
            cols.append(extra)
    # Deduplicate while preserving order.
    seen = set()
    return [c for c in cols if not (c in seen or seen.add(c))]


def _run_one_dataset(config: RunConfig) -> Path:
    ds_name = _safe(config.dataset)
    ds_out = config.output_dir / ds_name
    ds_out.mkdir(parents=True, exist_ok=True)
    plots_dir = ds_out / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    elog = EDALog(dataset=config.dataset, config=config.eda.as_dict())

    # ---- Load
    labels_df = load_labels(
        config.labels_csv, min_confidence=config.eda.min_label_confidence
    )
    biotype_map = load_gene_biotype_map(config.biotype_tsv)
    ds = load_dataset(
        dataset_name=config.dataset,
        matrix_parquet=config.matrix_parquet,
        sample_meta_parquet=config.sample_meta_parquet,
        labels_df=labels_df,
    )
    elog.inputs = {
        "matrix": str(config.matrix_parquet),
        "sample_meta": str(config.sample_meta_parquet),
        "labels": str(config.labels_csv),
        "biotype_map": str(config.biotype_tsv) if config.biotype_tsv else None,
        "n_samples_matrix": ds.n_samples_matrix,
        "n_samples_labeled": ds.n_samples_labeled,
        "n_samples_dropped_unlabeled": ds.n_samples_dropped_unlabeled,
        "n_genes": int(ds.expression.shape[0]),
    }
    if ds.expression.shape[1] < 2:
        elog.add_error(
            f"Dataset {config.dataset} has only {ds.expression.shape[1]} labeled "
            "sample(s); nothing to analyze. Aborting."
        )
        log_path = ds_out / f"eda_run_log_{ds_name}.json"
        elog.finalize(log_path)
        return log_path

    interpreter = PlotInterpreter(
        model=config.model,
        cache_dir=config.llm_cache_dir,
        disabled=config.disable_llm_interpretation,
    )

    # ---- Cohort composition
    cohort_report = cohort.summarize(ds)
    elog.add_step("cohort", cohort_report)
    label_plot = plotting.plot_label_bar(
        cohort_report.per_label,
        plots_dir / f"cohort_labels.{config.eda.plot_format}",
        config.eda.plot_dpi,
    )
    series_plot = plotting.plot_per_series_bar(
        cohort_report.per_series,
        plots_dir / f"cohort_series.{config.eda.plot_format}",
        config.eda.plot_dpi,
    )
    elog.add_plot("cohort_labels", label_plot)
    elog.add_plot("cohort_series", series_plot)
    _interpret_and_log(
        elog, interpreter,
        "cohort_labels", "Reviewed label counts",
        {"per_label": cohort_report.per_label, "n_samples": cohort_report.n_samples},
    )
    _interpret_and_log(
        elog, interpreter,
        "cohort_series", "Samples per series (top 20)",
        {
            "n_series_total": cohort_report.n_series,
            "top_series": cohort_report.per_series,
        },
    )

    # ---- Per-sample QC
    qc_report = qc.compute(ds, biotype_map=biotype_map)
    elog.add_step("qc", {
        "summary": qc_report.summary,
        "biotype_share_summary": qc_report.biotype_share_summary,
    })
    lib_plot = plotting.plot_library_size_hist(
        qc_report.per_sample,
        plots_dir / f"qc_library_size.{config.eda.plot_format}",
        config.eda.plot_dpi,
    )
    detected_plot = plotting.plot_genes_detected_hist(
        qc_report.per_sample,
        plots_dir / f"qc_genes_detected.{config.eda.plot_format}",
        config.eda.plot_dpi,
    )
    elog.add_plot("qc_library_size", lib_plot)
    elog.add_plot("qc_genes_detected", detected_plot)
    _interpret_and_log(
        elog, interpreter,
        "qc_library_size", "Library size distribution",
        qc_report.summary,
    )
    _interpret_and_log(
        elog, interpreter,
        "qc_genes_detected", "Genes detected per sample",
        qc_report.summary,
    )
    if qc_report.biotype_share is not None:
        biotype_plot = plotting.plot_biotype_share_box(
            qc_report.biotype_share,
            plots_dir / f"qc_biotype_share.{config.eda.plot_format}",
            config.eda.plot_dpi,
        )
        elog.add_plot("qc_biotype_share", biotype_plot)
        _interpret_and_log(
            elog, interpreter,
            "qc_biotype_share", "Biotype composition (top 10)",
            qc_report.biotype_share_summary or {},
        )

    # ---- Sample relationships
    rel = relationships.analyze(
        ds.expression,
        top_variable_genes=config.eda.top_variable_genes,
        n_pca_components=config.eda.n_pca_components,
        run_tsne_flag=config.eda.run_tsne,
        tsne_perplexity=config.eda.tsne_perplexity,
        tsne_random_state=config.eda.tsne_random_state,
    )
    elog.add_step(
        "relationships",
        {
            "pca": {
                "n_components": rel.pca.n_components,
                "explained_variance_ratio": rel.pca.explained_variance_ratio,
                "n_genes_used": rel.pca.n_genes_used,
            },
            "tsne_skip_reason": rel.tsne_skip_reason,
            "top_variable_genes_used": rel.top_variable_matrix.shape[0],
        },
    )
    coloring = _plot_coloring_columns(ds)
    for col in coloring:
        color_series = ds.sample_meta[col].reindex(rel.pca.scores.index)
        pca_plot = plotting.plot_pca_scatter(
            rel.pca.scores,
            color_series,
            rel.pca.explained_variance_ratio,
            plots_dir / f"pca_by_{_safe(col)}.{config.eda.plot_format}",
            config.eda.plot_dpi,
            color_name=col,
        )
        elog.add_plot(f"pca_by_{col}", pca_plot)
        if rel.tsne is not None:
            tsne_plot = plotting.plot_tsne_scatter(
                rel.tsne,
                color_series,
                plots_dir / f"tsne_by_{_safe(col)}.{config.eda.plot_format}",
                config.eda.plot_dpi,
                color_name=col,
            )
            elog.add_plot(f"tsne_by_{col}", tsne_plot)

    _interpret_and_log(
        elog, interpreter,
        "pca_variance", "PCA explained variance",
        {"explained_variance_ratio": rel.pca.explained_variance_ratio},
    )

    heatmap_plot = plotting.plot_sample_corr_heatmap(
        rel.sample_corr,
        rel.linkage_order,
        plots_dir / f"sample_correlation.{config.eda.plot_format}",
        config.eda.plot_dpi,
        max_samples=config.eda.heatmap_sample_cap,
    )
    elog.add_plot("sample_correlation", heatmap_plot)
    import numpy as np

    off = rel.sample_corr.to_numpy().copy()
    off[np.diag_indices_from(off)] = np.nan
    _interpret_and_log(
        elog, interpreter,
        "sample_correlation", "Sample-sample correlation heatmap",
        {
            "n_samples": int(rel.sample_corr.shape[0]),
            "off_diag_median": float(np.nanmedian(off)),
            "off_diag_min": float(np.nanmin(off)),
            "off_diag_max": float(np.nanmax(off)),
        },
    )

    # ---- Confounder screen
    conf = confounders.screen(
        rel.pca.scores,
        ds.sample_meta,
        top_pcs=config.eda.top_pcs_for_confounder_screen,
        flag_threshold=config.eda.confounder_association_flag,
    )
    elog.add_step("confounders", {
        "per_pc": conf.per_pc.astype(float).to_dict(),
        "flagged": conf.flagged,
        "kind": conf.kind,
    })
    elog.flagged_confounders = conf.flagged
    conf_plot = plotting.plot_confounder_heatmap(
        conf.per_pc.astype(float),
        plots_dir / f"confounder_screen.{config.eda.plot_format}",
        config.eda.plot_dpi,
    )
    elog.add_plot("confounder_screen", conf_plot)
    _interpret_and_log(
        elog, interpreter,
        "confounder_screen",
        "Top PCs vs covariates (association)",
        {
            "flagged": conf.flagged,
            "flag_threshold": config.eda.confounder_association_flag,
            "covariate_kinds": conf.kind,
        },
    )

    # ---- Persist tabular summaries
    stats_csv = ds_out / f"eda_summary_stats_{ds_name}.csv"
    _write_summary_stats(
        stats_csv,
        cohort_report=cohort_report,
        qc_report=qc_report,
        rel=rel,
        conf=conf,
    )
    qc_per_sample_csv = ds_out / f"per_sample_qc_{ds_name}.csv"
    qc_report.per_sample.to_csv(qc_per_sample_csv, index_label="sample_id")
    pca_scores_csv = ds_out / f"pca_scores_{ds_name}.csv"
    rel.pca.scores.to_csv(pca_scores_csv, index_label="sample_id")

    interpretations_json = ds_out / f"eda_interpretations_{ds_name}.json"
    interpretations_json.write_text(
        json.dumps(elog.interpretations, indent=2, sort_keys=True)
    )

    elog.outputs = {
        "plots_dir": str(plots_dir),
        "summary_stats_csv": str(stats_csv),
        "per_sample_qc_csv": str(qc_per_sample_csv),
        "pca_scores_csv": str(pca_scores_csv),
        "interpretations_json": str(interpretations_json),
        "llm_call_count": interpreter.call_count,
        "llm_cache_hit_count": interpreter.cache_hit_count,
    }
    log_path = ds_out / f"eda_run_log_{ds_name}.json"
    elog.finalize(log_path)
    return log_path


def _interpret_and_log(
    elog: EDALog,
    interpreter: PlotInterpreter,
    key: str,
    plot_name: str,
    context: dict,
) -> None:
    try:
        result = interpreter.interpret(plot_name, context)
    except Exception as exc:  # noqa: BLE001 — interpretation is advisory
        elog.add_warning(f"Interpretation failed for {key}: {exc}")
        return
    elog.add_interpretation(key, result.interpretation, result.cached)


def _write_summary_stats(
    csv_path: Path,
    *,
    cohort_report,
    qc_report,
    rel,
    conf,
) -> None:
    """One tall-format CSV with everything a Task 7 report needs.

    Columns: metric, key, value. Chosen deliberately over a wide format so
    it's cheap to append new metrics without breaking the schema.
    """
    rows = []
    rows.append({"metric": "cohort", "key": "n_samples", "value": cohort_report.n_samples})
    rows.append({"metric": "cohort", "key": "n_series", "value": cohort_report.n_series})
    for label, n in cohort_report.per_label.items():
        rows.append({"metric": "cohort.per_label", "key": label, "value": n})

    for name, block in qc_report.summary.items():
        for k, v in block.items():
            rows.append({"metric": f"qc.{name}", "key": k, "value": v})

    for i, r in enumerate(rel.pca.explained_variance_ratio, start=1):
        rows.append({"metric": "pca.explained_variance_ratio", "key": f"PC{i}", "value": r})

    for row in conf.flagged:
        rows.append({
            "metric": "confounders.flagged",
            "key": f"{row['pc']}:{row['covariate']}",
            "value": row["association"],
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False)


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cvd_eda.eda.run",
        description="Task 6: EDA Agent — runs and interprets the QC/EDA suite on "
                    "one labeled dataset. Requires a human-reviewed Task 5 labels file.",
    )
    p.add_argument("--dataset", required=True,
                   help="Dataset name for filenames (e.g. archs4, recount3_GTEX_HEART).")
    p.add_argument("--matrix", dest="matrix_parquet", required=True, type=Path,
                   help="Task 4 output: cvd_matrix_{dataset}_normalized.parquet.")
    p.add_argument("--sample-meta", dest="sample_meta_parquet", required=True, type=Path,
                   help="Task 4 output: cvd_sample_meta_{dataset}.parquet.")
    p.add_argument("--labels", dest="labels_csv", required=True, type=Path,
                   help="Task 5 output *after human review* — "
                        "e.g. label_proposals_{dataset}.reviewed.csv.")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory for plots, summary CSVs, and the JSON audit log.")
    p.add_argument("--gene-biotype-tsv", dest="biotype_tsv", type=Path, default=None,
                   help="Optional TSV [ensembl_id, biotype] to enable biotype-share QC.")
    p.add_argument("--allow-unreviewed-labels", action="store_true",
                   help="Skip the '.reviewed.' filename check on --labels. "
                        "Do not use in real runs — Task 5 requires human review.")

    p.add_argument("--llm-cache", dest="llm_cache_dir", type=Path, default=None,
                   help="Directory for the on-disk LLM interpretation cache.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Anthropic model for plot interpretation (default: {DEFAULT_MODEL}).")
    p.add_argument("--disable-llm-interpretation", action="store_true",
                   help="Skip the LLM interpretation step (still writes plots + stats).")

    # EDA config overrides — every field on EDAConfig gets a matching flag.
    p.add_argument("--top-variable-genes", type=int, default=None)
    p.add_argument("--n-pca-components", type=int, default=None)
    p.add_argument("--no-tsne", dest="run_tsne", action="store_false", default=None)
    p.add_argument("--tsne-perplexity", type=float, default=None)
    p.add_argument("--confounder-flag-threshold", type=float, default=None)
    p.add_argument("--top-pcs-for-confounder-screen", type=int, default=None)
    p.add_argument("--heatmap-sample-cap", type=int, default=None)
    p.add_argument("--plot-dpi", type=int, default=None)
    p.add_argument("--min-label-confidence", type=float, default=None)

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _cfg_from_args(args: argparse.Namespace) -> EDAConfig:
    overrides: dict = {}
    if args.top_variable_genes is not None:
        overrides["top_variable_genes"] = args.top_variable_genes
    if args.n_pca_components is not None:
        overrides["n_pca_components"] = args.n_pca_components
    if args.run_tsne is False:
        overrides["run_tsne"] = False
    if args.tsne_perplexity is not None:
        overrides["tsne_perplexity"] = args.tsne_perplexity
    if args.confounder_flag_threshold is not None:
        overrides["confounder_association_flag"] = args.confounder_flag_threshold
    if args.top_pcs_for_confounder_screen is not None:
        overrides["top_pcs_for_confounder_screen"] = args.top_pcs_for_confounder_screen
    if args.heatmap_sample_cap is not None:
        overrides["heatmap_sample_cap"] = args.heatmap_sample_cap
    if args.plot_dpi is not None:
        overrides["plot_dpi"] = args.plot_dpi
    if args.min_label_confidence is not None:
        overrides["min_label_confidence"] = args.min_label_confidence
    return EDAConfig(**overrides)


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    _gate_review_file(args.labels_csv, args.allow_unreviewed_labels)

    run_cfg = RunConfig(
        dataset=args.dataset,
        matrix_parquet=args.matrix_parquet,
        sample_meta_parquet=args.sample_meta_parquet,
        labels_csv=args.labels_csv,
        output_dir=args.output_dir,
        biotype_tsv=args.biotype_tsv,
        llm_cache_dir=args.llm_cache_dir,
        model=args.model,
        disable_llm_interpretation=args.disable_llm_interpretation,
        eda=_cfg_from_args(args),
    )

    _run_one_dataset(run_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
