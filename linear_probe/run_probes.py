"""run_probes.py — orchestrator for steps 5 and 7.

Sweeps every (variant, negative-pool) combination that has a cached embedding
parquet on disk, runs the linear probe once each, aggregates the per-run
results into `results/disease_classification_by_variant.csv` (§ 5), and
produces the scale-vs-performance comparison table + plot (§ 7) with the
elastic-net stage's PR-AUC drawn in as a reference line.

Deliberately re-uses `probe.py` per combination — one implementation, per the
TODO's rule "not five copies". This file only orchestrates + summarizes.

Elastic-net reference
---------------------
Reads mean PR-AUC from the elastic-net stage's outer-CV summary
(`eda/dataset/cvd_data/elasticnet_out/folds/cv_summary.json`) if present. If
not (e.g. that stage hasn't been re-run since a recent refactor), the plot
skips the reference line and writes a note into the comparison table.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_EMBEDDINGS = HERE / "embeddings"
DEFAULT_RESULTS = HERE / "results"

VARIANT_NOMINAL_MILLIONS = {
    "BulkFormer-37M": 37, "BulkFormer-50M": 50, "BulkFormer-93M": 93,
    "BulkFormer-127M": 127, "BulkFormer-147M": 147,
}
POOLS = ("neg_whole_corpus", "neg_hard")


def _log() -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    return logging.getLogger("linear_probe.run_probes")


def _elastic_net_reference(logger: logging.Logger) -> dict[str, float] | None:
    """Best-effort load of the elastic-net stage's PR-AUC for the plot."""
    candidates = [
        REPO / "eda" / "dataset" / "cvd_data" / "elasticnet_out" / "folds" / "cv_summary.json",
        REPO / "eda" / "dataset" / "cvd_data" / "elasticnet_out" / "evaluate" / "summary.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            # Best-effort key hunt — the elastic-net stage evolved schemas.
            for key in ("pr_auc_mean", "pr_auc"):
                if key in data:
                    logger.info(f"elastic-net reference: {p.name} → pr_auc={data[key]:.3f}")
                    return {"path": str(p.relative_to(REPO)), "pr_auc_mean": float(data[key])}
    logger.info("no elastic-net PR-AUC reference found — plot will omit the reference line")
    return None


def _run_probe(embeddings: Path, pool: str, outdir: Path,
               k_folds: int, seed: int, logger: logging.Logger) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", "-m", "linear_probe.probe",
           "--embeddings", str(embeddings),
           "--pool", pool,
           "--outdir", str(outdir),
           "--k-folds", str(k_folds),
           "--seed", str(seed)]
    logger.info(f"→ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO))
    return outdir / "probe_results.json"


def _aggregate(rows: list[dict], out_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["variant_millions_nominal"] = df["variant"].map(VARIANT_NOMINAL_MILLIONS)
    df = df.sort_values(["negative_pool", "variant_millions_nominal"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    logger.info(f"wrote {out_csv} ({len(df)} rows)")
    return df


def _plot_variant_comparison(df: pd.DataFrame, elastic_ref: dict | None,
                             out_png: Path, logger: logging.Logger) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"neg_whole_corpus": "#2E86AB", "neg_hard": "#E63946"}
    labels = {"neg_whole_corpus": "vs. whole-corpus non-CVD",
              "neg_hard":         "vs. tissue-only hard negatives"}

    for pool, sub in df.groupby("negative_pool"):
        sub = sub.sort_values("variant_millions_nominal")
        y = pd.to_numeric(sub["pr_auc_mean"], errors="coerce").to_numpy()
        yerr = pd.to_numeric(sub["pr_auc_std"], errors="coerce").to_numpy()
        # errorbar rejects None/NaN in yerr — mask to only points that actually ran.
        finite = np.isfinite(y) & np.isfinite(yerr)
        if not finite.any():
            logger.warning(f"pool={pool}: no probe runs produced numeric PR-AUC "
                           "(all folds skipped) — skipping series in the plot")
            continue
        x = sub["variant_millions_nominal"].to_numpy()[finite]
        ax.errorbar(x, y[finite], yerr=yerr[finite],
                    marker="o", capsize=3, color=colors.get(pool, "black"),
                    label=labels.get(pool, pool))

    if elastic_ref is not None:
        ax.axhline(elastic_ref["pr_auc_mean"], color="#333333", linestyle="--", linewidth=1,
                   label=f"elastic net baseline (PR-AUC={elastic_ref['pr_auc_mean']:.3f})")

    ax.set_xlabel("BulkFormer variant (params, millions — nominal)")
    ax.set_ylabel("PR-AUC (mean ± std across 5 outer folds)")
    ax.set_title("Disease classification: frozen BulkFormer + linear probe")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {out_png}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run linear probes across variants (steps 5+7).")
    parser.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--results-dir",    type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed",    type=int, default=20260707)
    parser.add_argument("--pools", nargs="*", default=list(POOLS),
                        help='Which negative pool(s) to run. Default: both.')
    args = parser.parse_args(argv)

    logger = _log()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    embeddings_files = sorted(args.embeddings_dir.glob("embeddings_BulkFormer-*.parquet"))
    if not embeddings_files:
        logger.error(f"no embeddings parquets found in {args.embeddings_dir} — "
                     "run linear_probe.extract first")
        return 1
    logger.info(f"found {len(embeddings_files)} embedding parquets: "
                f"{[p.name for p in embeddings_files]}")

    all_rows: list[dict] = []
    for emb_path in embeddings_files:
        variant_name = emb_path.stem.replace("embeddings_", "")
        for pool in args.pools:
            outdir = args.results_dir / variant_name / pool
            results_path = _run_probe(emb_path, pool, outdir, args.k_folds, args.seed, logger)
            data = json.loads(results_path.read_text())
            row = {"variant": data["variant"], "negative_pool": data["negative_pool"],
                   "n_samples": data["n_samples"], "n_positive": data["n_positive"],
                   "n_series": data["n_series"], **data["summary"]}
            all_rows.append(row)

    df = _aggregate(all_rows, args.results_dir / "disease_classification_by_variant.csv", logger)

    elastic_ref = _elastic_net_reference(logger)
    _plot_variant_comparison(df, elastic_ref,
                              args.results_dir / "variant_comparison.png", logger)

    # A tidier one-column-per-metric table for the plot's underlying numbers.
    df.to_csv(args.results_dir / "variant_comparison_table.csv", index=False)
    logger.info(f"wrote {args.results_dir / 'variant_comparison_table.csv'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
