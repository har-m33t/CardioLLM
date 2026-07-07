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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ARCHS4_ORGANISM = "human"
ARCHS4_VERSION = "latest"

# ARCHS4 human gene H5 has been ~30GB for recent releases. Anything under
# MIN_EXPECTED_SIZE_GB almost certainly represents a partial/failed download
# that a filename-only "already downloaded" check would silently accept.
# 20GB is a conservative floor — if a future release genuinely ships smaller
# than this, this constant must be lowered deliberately, not silently bypassed.
MIN_EXPECTED_SIZE_GB = 20.0

# Buffer above the expected download size to guard against filesystem overhead
# and any temporary/index files archs4py may write alongside the H5.
DISK_HEADROOM_GB = 10.0
EXPECTED_DOWNLOAD_GB = 30.0


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


def _validate_existing_file(h5_path: Path, logger: logging.Logger) -> float:
    """Return size in GB if the file looks complete; raise otherwise.

    A partial/failed download that matches the glob would be silently accepted
    by a filename-only check, then blow up later when h5py tries to open it or
    when a downstream step reads truncated rows. Reject anything below
    MIN_EXPECTED_SIZE_GB and let the caller decide (usually: delete and
    re-download).
    """
    size_gb = h5_path.stat().st_size / 1e9
    if size_gb < MIN_EXPECTED_SIZE_GB:
        raise RuntimeError(
            f"Existing file {h5_path} is only {size_gb:.1f} GB, below the "
            f"{MIN_EXPECTED_SIZE_GB:.0f} GB floor for an ARCHS4 human release. "
            "This looks like a truncated/failed download — delete it and re-run."
        )
    return size_gb


def _check_disk_space(target_dir: Path, needed_gb: float, logger: logging.Logger) -> None:
    """Fail fast if there isn't enough free space for the download.

    Prevents the silently-truncated-file class of bug: archs4py could
    otherwise write until the filesystem fills, producing a file that matches
    the glob on rerun and looks "done".
    """
    free_gb = shutil.disk_usage(target_dir).free / 1e9
    if free_gb < needed_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {target_dir}; need >= {needed_gb:.1f} GB "
            "for the ARCHS4 download. Free space or point --outdir elsewhere."
        )
    logger.info("disk space check: %.1f GB free at %s (need >= %.1f GB)",
                free_gb, target_dir, needed_gb)


def load_archs4(outdir: Path, logger: logging.Logger) -> dict:
    """Download the ARCHS4 human gene-count H5 file via archs4py, skipping if present."""
    import archs4py as a4

    archs4_dir = outdir / "archs4"
    archs4_dir.mkdir(parents=True, exist_ok=True)

    existing = list(archs4_dir.glob(f"{ARCHS4_ORGANISM}_gene_*.h5"))

    if existing:
        h5_path = existing[0]
        size_gb = _validate_existing_file(h5_path, logger)
        logger.info(f"ARCHS4 file already present: {h5_path} ({size_gb:.1f} GB) — skipping download")
    else:
        _check_disk_space(archs4_dir, EXPECTED_DOWNLOAD_GB + DISK_HEADROOM_GB, logger)
        logger.info(f"Downloading ARCHS4 {ARCHS4_ORGANISM} gene counts (version={ARCHS4_VERSION})...")
        logger.info("This is 30GB+ and will take a while — expected on first run.")
        h5_path_str = a4.download.counts(ARCHS4_ORGANISM, path=str(archs4_dir), version=ARCHS4_VERSION)
        h5_path = Path(h5_path_str)
        size_gb = _validate_existing_file(h5_path, logger)
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