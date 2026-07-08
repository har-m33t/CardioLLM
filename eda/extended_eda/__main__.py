"""
extended_eda.py — orchestrator CLI for the extended EDA (revised_eda_tod.md).

Pipeline order:
    1. labels          — one streaming pass over sample metadata to build
                         labels/sample_labels.parquet + labels/patient_coverage.json
    2. definitions     — writes extended_eda/definitions.md (§0)
    3. whole_breakdown — section 1 CSV + markdown table
    4. cvd_breakdown   — section 2 CSV + markdown table
    5. cross_check     — section 3 consistency JSON
    6. writeup         — assembles whole_dataset_writeup.md, cvd_writeup.md,
                         eda_writeup.md

Every step reads only the artifact produced by earlier steps, so any single
step can be re-run in isolation with `--only <step>`.

Usage
-----
    # Real ARCHS4 run:
    python -m eda.extended_eda \
        --h5 eda/dataset/cvd_data/archs4/human_gene_v2.latest.h5 \
        --qc-csv eda/dataset/cvd_data/eda_out/qc/qc_full_dataset.csv \
        --outdir eda/dataset/cvd_data/extended_eda

    # Auto-locate H5 and QC CSV under a data root:
    python -m eda.extended_eda \
        --data-root eda/dataset/cvd_data \
        --outdir eda/dataset/cvd_data/extended_eda

    # Just re-run one step (previous artifacts must already exist):
    python -m eda.extended_eda --data-root eda/dataset/cvd_data \
        --outdir eda/dataset/cvd_data/extended_eda --only writeup
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import (
    cross_check,
    cvd_breakdown,
    labels,
    whole_breakdown,
    writeup,
)

ALL_STEPS = (
    "labels",
    "definitions",
    "whole_breakdown",
    "cvd_breakdown",
    "cross_check",
    "writeup",
)


def setup_logging(outdir: Path) -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"extended_eda_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Avoid duplicating handlers when re-invoked in-process (tests).
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(fh)
    root.addHandler(sh)
    return logging.getLogger("extended_eda")


def resolve_h5(h5_arg: str | None, data_root: str | None) -> Path:
    if h5_arg:
        p = Path(h5_arg).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"H5 file not found: {p}")
        return p
    if not data_root:
        raise ValueError("either --h5 or --data-root must be provided")
    root = Path(data_root).expanduser() / "archs4"
    matches = sorted(root.glob("human_gene_*.h5"))
    if not matches:
        raise FileNotFoundError(f"no ARCHS4 human gene H5 file found under {root}")
    return matches[0]


def resolve_qc_csv(qc_arg: str | None, data_root: str | None) -> Path:
    if qc_arg:
        p = Path(qc_arg).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"QC CSV not found: {p}")
        return p
    if not data_root:
        raise ValueError("either --qc-csv or --data-root must be provided")
    root = Path(data_root).expanduser() / "eda_out" / "qc"
    candidate = root / "qc_full_dataset.csv"
    if not candidate.exists():
        raise FileNotFoundError(
            f"whole-corpus QC CSV not found at {candidate} — run `python -m eda.eda ... "
            "--only qc` first (extended EDA reuses its genes_detected column)"
        )
    return candidate


def _run_step(name: str, fn, logger: logging.Logger) -> dict:
    started = datetime.now(timezone.utc).isoformat()
    logger.info("=== step %s: start ===", name)
    try:
        result = fn()
        finished = datetime.now(timezone.utc).isoformat()
        logger.info("=== step %s: done -> %s ===", name, result)
        return {"step": name, "status": "ok", "started": started, "finished": finished, "output": str(result)}
    except Exception as e:
        finished = datetime.now(timezone.utc).isoformat()
        logger.error("=== step %s: FAILED (%s) ===\n%s", name, e, traceback.format_exc())
        return {"step": name, "status": "failed", "started": started, "finished": finished, "error": str(e)}


def _label_paths(outdir: Path) -> tuple[Path, Path]:
    """Guess the two label-step artifacts from an existing outdir. Used when
    `--only <step>` skips the labels step but a later step needs its paths."""
    labels_dir = outdir / "labels"
    parquet = labels_dir / "sample_labels.parquet"
    csv = labels_dir / "sample_labels.csv"
    if parquet.exists():
        labels_path = parquet
    elif csv.exists():
        labels_path = csv
    else:
        raise FileNotFoundError(
            f"neither {parquet} nor {csv} exists; run the `labels` step first"
        )
    coverage = labels_dir / "patient_coverage.json"
    if not coverage.exists():
        raise FileNotFoundError(f"{coverage} missing; re-run the `labels` step")
    return labels_path, coverage


def main():
    parser = argparse.ArgumentParser(description="Extended EDA (revised_eda_tod.md).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5", help="Path to the ARCHS4 human gene H5 file.")
    src.add_argument("--data-root", help="Base directory containing archs4/human_gene_*.h5 and eda_out/qc.")
    parser.add_argument("--qc-csv", help="Whole-corpus QC CSV with per-sample genes_detected. "
                                         "Defaults to <data-root>/eda_out/qc/qc_full_dataset.csv.")
    parser.add_argument("--outdir", required=True, help="Base output directory (e.g. .../extended_eda).")
    parser.add_argument("--only", default=None,
                        help=f"Comma-separated subset of steps to run. Default: all ({','.join(ALL_STEPS)}).")
    parser.add_argument("--slice-size", type=int, default=labels.DEFAULT_SLICE,
                        help="Rows per streaming batch in the labels step (RAM lever).")
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(outdir)

    h5_path = resolve_h5(args.h5, args.data_root)
    qc_csv = resolve_qc_csv(args.qc_csv, args.data_root)
    logger.info("H5:      %s", h5_path)
    logger.info("QC CSV:  %s", qc_csv)
    logger.info("outdir:  %s", outdir)

    steps_to_run = ALL_STEPS if args.only is None else tuple(s.strip() for s in args.only.split(","))
    unknown = [s for s in steps_to_run if s not in ALL_STEPS]
    if unknown:
        raise SystemExit(f"unknown step(s): {unknown}. valid: {ALL_STEPS}")

    # Paths we'll thread between steps. `labels_path` / `coverage_path` may be
    # produced by the labels step, or (for `--only <step>`) reloaded from disk.
    label_state: dict[str, Path] = {}

    def _labels():
        paths = labels.build_label_table(h5_path, outdir, slice_size=args.slice_size)
        label_state["labels_path"] = paths.labels_parquet
        label_state["coverage_json"] = paths.coverage_json
        return paths.labels_parquet

    def _definitions():
        if "coverage_json" not in label_state:
            _, cov = _label_paths(outdir)
            label_state["coverage_json"] = cov
        out_path = outdir / "definitions.md"
        # Pool JSON may not exist yet on a fresh run; write_definitions
        # gracefully omits the CVD-pool coverage block when it's absent, then
        # the writeup step re-runs definitions with the pool JSON present.
        pool_json = outdir / "section2_cvd_breakdown" / "cvd_pool_composition.json"
        writeup.write_definitions(
            label_state["coverage_json"],
            out_path,
            pool_json=pool_json if pool_json.exists() else None,
        )
        return out_path

    def _whole_breakdown():
        if "labels_path" not in label_state:
            lp, cov = _label_paths(outdir)
            label_state["labels_path"] = lp
            label_state["coverage_json"] = cov
        return whole_breakdown.run(label_state["labels_path"], qc_csv, outdir)

    def _cvd_breakdown():
        if "labels_path" not in label_state:
            lp, cov = _label_paths(outdir)
            label_state["labels_path"] = lp
            label_state["coverage_json"] = cov
        return cvd_breakdown.run(label_state["labels_path"], qc_csv, outdir)

    def _cross_check():
        if "labels_path" not in label_state:
            lp, cov = _label_paths(outdir)
            label_state["labels_path"] = lp
            label_state["coverage_json"] = cov
        s1 = outdir / "section1_whole_breakdown" / "whole_dataset_disease_breakdown.csv"
        s2 = outdir / "section2_cvd_breakdown" / "cvd_disease_breakdown.csv"
        return cross_check.run(label_state["labels_path"], s1, s2, outdir)

    def _writeup():
        if "coverage_json" not in label_state:
            _, cov = _label_paths(outdir)
            label_state["coverage_json"] = cov
        s1_csv = outdir / "section1_whole_breakdown" / "whole_dataset_disease_breakdown.csv"
        s1_md = outdir / "section1_whole_breakdown" / "whole_dataset_disease_breakdown_display.md"
        s2_csv = outdir / "section2_cvd_breakdown" / "cvd_disease_breakdown.csv"
        s2_md = outdir / "section2_cvd_breakdown" / "cvd_disease_breakdown_display.md"
        pool_json = outdir / "section2_cvd_breakdown" / "cvd_pool_composition.json"
        xc_json = outdir / "section3_cross_check" / "cross_check.json"
        # Re-emit definitions once section 2 has produced the pool JSON so
        # the CVD-pool coverage block and comorbidity paragraph land in
        # `definitions.md` even on a fresh full run (Issue 2/3).
        writeup.write_definitions(
            label_state["coverage_json"], outdir / "definitions.md",
            pool_json=pool_json,
        )
        writeup.write_whole_writeup(s1_csv, s1_md, label_state["coverage_json"],
                                    outdir / "whole_dataset_writeup.md")
        writeup.write_cvd_writeup(s2_csv, s2_md, label_state["coverage_json"], pool_json,
                                  outdir / "cvd_writeup.md")
        writeup.write_overall(s1_csv, s1_md, s2_csv, s2_md,
                              label_state["coverage_json"], xc_json, pool_json,
                              outdir / "eda_writeup.md")
        return outdir / "eda_writeup.md"

    step_runners = {
        "labels":          _labels,
        "definitions":     _definitions,
        "whole_breakdown": _whole_breakdown,
        "cvd_breakdown":   _cvd_breakdown,
        "cross_check":     _cross_check,
        "writeup":         _writeup,
    }

    manifest = {
        "h5_path": str(h5_path),
        "qc_csv": str(qc_csv),
        "outdir": str(outdir),
        "run_started": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }
    for name in steps_to_run:
        manifest["steps"].append(_run_step(name, step_runners[name], logger))
    manifest["run_finished"] = datetime.now(timezone.utc).isoformat()

    manifest_path = outdir / "logs" / "extended_eda_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
