"""
data.py — download the ARCHS4 human gene-count dataset.

Usage:
    python data.py --outdir ~/cvd_data

Design notes:
- Idempotent: re-running skips the download if the H5 file is already present
  (checked by filename pattern), so it's safe to re-run if interrupted.
- Logs to outdir/logs/ so you have a record of exactly what was pulled and when.

RECOUNT3 intentionally left out for now — whole-dataset EDA only needs one
corpus, and ARCHS4 is both larger and the corpus BulkFormer itself was
pretrained on. Add a RECOUNT3 loader back in later only if a concrete gap
shows up (e.g. needing more CVD case samples, or a second-corpus check).

Install: pip install archs4py
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHS4_ORGANISM = "human"
ARCHS4_VERSION = "latest"


def setup_logging(outdir: Path) -> logging.Logger:
    log_dir = outdir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"data_load_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.log"

    logger = logging.getLogger("data_loader")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_archs4(outdir: Path, logger: logging.Logger) -> dict:
    """Download the ARCHS4 human gene-count H5 file via archs4py, skipping if present."""
    import archs4py as a4

    archs4_dir = outdir / "archs4"
    archs4_dir.mkdir(parents=True, exist_ok=True)

    existing = list(archs4_dir.glob(f"{ARCHS4_ORGANISM}_gene_*.h5"))

    if existing:
        h5_path = existing[0]
        size_gb = h5_path.stat().st_size / 1e9
        logger.info(f"ARCHS4 file already present: {h5_path} ({size_gb:.1f} GB) — skipping download")
    else:
        logger.info(f"Downloading ARCHS4 {ARCHS4_ORGANISM} gene counts (version={ARCHS4_VERSION})...")
        logger.info("This is 30GB+ and will take a while — expected on first run.")
        h5_path_str = a4.download.counts(ARCHS4_ORGANISM, path=str(archs4_dir), version=ARCHS4_VERSION)
        h5_path = Path(h5_path_str)
        size_gb = h5_path.stat().st_size / 1e9
        logger.info(f"Downloaded: {h5_path} ({size_gb:.1f} GB)")

    return {
        "dataset": "archs4",
        "organism": ARCHS4_ORGANISM,
        "version": ARCHS4_VERSION,
        "path": str(h5_path),
        "size_gb": round(size_gb, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Download the ARCHS4 dataset.")
    parser.add_argument("--outdir", required=True, help="Base output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(outdir)

    manifest = {"run_timestamp": datetime.now(timezone.utc).isoformat(), "outdir": str(outdir)}

    try:
        manifest["archs4"] = load_archs4(outdir, logger)
    except Exception as e:
        logger.exception("ARCHS4 load failed")
        manifest["archs4"] = {"status": "failed", "error": str(e)}

    manifest_path = outdir / "logs" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Done. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()