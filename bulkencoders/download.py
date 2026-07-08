"""download.py — fetch BulkFormer pretrained checkpoints and support files.

Mirrors the assets published by the BulkFormer authors (KangBoming/BulkFormer):
  - 5 model-size variants (37M / 50M / 93M / 127M / 147M) from Google Drive
  - Support files (gene info, gene-interaction graph, ESM2 features, gene list)
    from Zenodo record 10.5281/zenodo.15744294

Everything lands under `bulkencoders/checkpoints/bulkformer/`:
    checkpoints/bulkformer/models/BulkFormer-{37M,50M,93M,127M,147M}.pt
    checkpoints/bulkformer/support/{bulkformer_gene_info.csv, G_tcga.pt,
        G_tcga_weight.pt, esm2_feature_concat.pt, interested_gene_list.pt}

Idempotent: re-running skips files that already exist and pass a minimum-size
sanity check. Delete a file to force re-download.

Usage:
    python -m bulkencoders.download                 # all models + support
    python -m bulkencoders.download --models 37M 50M
    python -m bulkencoders.download --skip-support
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gdown
import requests

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "checkpoints" / "bulkformer"


@dataclass(frozen=True)
class DriveModel:
    name: str          # e.g. "BulkFormer-37M"
    file_id: str       # Google Drive file ID
    min_mb: float      # sanity floor — smaller than this means the download failed


@dataclass(frozen=True)
class ZenodoFile:
    name: str
    url: str
    min_kb: float


# From https://github.com/KangBoming/BulkFormer/blob/main/model/README.md.
# Floors are calibrated at ~1 MB per M-param (37M weighs 53.5 MB on disk,
# implying float16 storage). Tight enough to catch the "Drive quota exceeded"
# HTML interstitial and mid-stream truncation, loose enough to pass a genuine
# complete file.
BULKFORMER_MODELS: tuple[DriveModel, ...] = (
    DriveModel("BulkFormer-37M",  "1qY2qaXfKfDot9EMcOF9gr8T0jghrH7km", min_mb=40),
    DriveModel("BulkFormer-50M",  "12ZYGYrZIQJyodaVicrJpnY_8_JG-hamK", min_mb=50),
    DriveModel("BulkFormer-93M",  "1s_3XoMaHiBfxi5C8D3bgwwzNrihafaIk", min_mb=90),
    DriveModel("BulkFormer-127M", "1-5AdgIpkm8dOm9tuwOXcqS1sUyGN6vHg", min_mb=120),
    DriveModel("BulkFormer-147M", "1UtqN_vCh3669Fs-GU5CTE7F7UnuQCAzN", min_mb=140),
)

# From Zenodo record 10.5281/zenodo.15744294 (the record the current data/README
# points to). Only the files needed to load and run the pretrained model —
# omits the multi-GB downstream-task datasets to keep the download tractable.
ZENODO_BASE = "https://zenodo.org/records/15744294/files"
BULKFORMER_SUPPORT: tuple[ZenodoFile, ...] = (
    ZenodoFile("bulkformer_gene_info.csv",  f"{ZENODO_BASE}/bulkformer_gene_info.csv?download=1",  min_kb=1_000),
    ZenodoFile("G_tcga.pt",                 f"{ZENODO_BASE}/G_tcga.pt?download=1",                 min_kb=1_000),
    ZenodoFile("G_tcga_weight.pt",          f"{ZENODO_BASE}/G_tcga_weight.pt?download=1",          min_kb=500),
    ZenodoFile("esm2_feature_concat.pt",    f"{ZENODO_BASE}/esm2_feature_concat.pt?download=1",    min_kb=50_000),
    ZenodoFile("interested_gene_list.pt",   f"{ZENODO_BASE}/interested_gene_list.pt?download=1",   min_kb=1),
    # gene_length_df.csv is needed by the TPM normalization step in the
    # extract_feature notebook — required to preprocess raw ARCHS4 counts.
    ZenodoFile("gene_length_df.csv",        f"{ZENODO_BASE}/gene_length_df.csv?download=1",        min_kb=500),
)


def _log() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("bulkencoders.download")


def _already_ok(path: Path, min_bytes: int, logger: logging.Logger) -> bool:
    """True if `path` exists and is at least `min_bytes` — skip re-download.

    The min-size floor is what protects against a filename-only "already there"
    check silently accepting a truncated/failed prior download.
    """
    if not path.exists():
        return False
    size = path.stat().st_size
    if size < min_bytes:
        logger.warning(
            f"{path.name} exists but is {size / 1e6:.1f} MB (< floor "
            f"{min_bytes / 1e6:.1f} MB) — treating as truncated and re-downloading"
        )
        path.unlink()
        return False
    logger.info(f"{path.name} already present ({size / 1e6:.1f} MB) — skipping")
    return True


def _download_drive(model: DriveModel, dest_dir: Path, logger: logging.Logger) -> Path:
    """Download one BulkFormer variant from Google Drive via `gdown`.

    `gdown` handles the virus-scan interstitial that plain HTTP hits for
    files >100MB — a raw `wget` on a Drive share URL just fetches the HTML
    warning page.
    """
    dest = dest_dir / f"{model.name}.pt"
    min_bytes = int(model.min_mb * 1e6)
    if _already_ok(dest, min_bytes, logger):
        return dest

    logger.info(f"downloading {model.name} from Google Drive (id={model.file_id})")
    url = f"https://drive.google.com/uc?id={model.file_id}"
    out_str = gdown.download(url, str(dest), quiet=False, fuzzy=True)
    if out_str is None or not Path(out_str).exists():
        raise RuntimeError(f"gdown returned no file for {model.name}")

    got = Path(out_str)
    if got != dest:
        got.rename(dest)

    size = dest.stat().st_size
    if size < min_bytes:
        raise RuntimeError(
            f"{model.name} downloaded to {size / 1e6:.1f} MB, below the "
            f"{model.min_mb:.0f} MB floor — likely truncated or a Drive quota error"
        )
    logger.info(f"  wrote {dest} ({size / 1e6:.1f} MB)")
    return dest


def _download_zenodo(item: ZenodoFile, dest_dir: Path, logger: logging.Logger) -> Path:
    """Stream a single Zenodo file to disk. Straightforward HTTPS."""
    dest = dest_dir / item.name
    min_bytes = int(item.min_kb * 1e3)
    if _already_ok(dest, min_bytes, logger):
        return dest

    logger.info(f"downloading {item.name} from Zenodo")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(item.url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
    tmp.rename(dest)

    size = dest.stat().st_size
    if size < min_bytes:
        raise RuntimeError(
            f"{item.name} downloaded to {size / 1e3:.1f} KB, below the "
            f"{item.min_kb:.0f} KB floor — likely truncated"
        )
    logger.info(f"  wrote {dest} ({size / 1e6:.1f} MB)")
    return dest


def _select_models(names: Iterable[str] | None) -> tuple[DriveModel, ...]:
    if not names:
        return BULKFORMER_MODELS
    known = {m.name.split("-")[1]: m for m in BULKFORMER_MODELS}  # "37M" -> model
    picked = []
    for n in names:
        key = n.upper().removeprefix("BULKFORMER-")
        if key not in known:
            raise SystemExit(f"unknown model {n!r}; choose from {sorted(known)}")
        picked.append(known[key])
    return tuple(picked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download BulkFormer checkpoints + support files.")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help=f"Root directory. Default: {DEFAULT_ROOT.relative_to(HERE.parent)}",
    )
    parser.add_argument(
        "--models", nargs="*", metavar="SIZE",
        help='Model variants to fetch (e.g. "37M 147M"). Default: all five.',
    )
    parser.add_argument("--skip-models",  action="store_true", help="Skip Google Drive checkpoints.")
    parser.add_argument("--skip-support", action="store_true", help="Skip Zenodo support files.")
    args = parser.parse_args(argv)

    logger = _log()
    root: Path = args.root
    models_dir = root / "models"
    support_dir = root / "support"
    models_dir.mkdir(parents=True, exist_ok=True)
    support_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"target root: {root}")

    if not args.skip_models:
        for m in _select_models(args.models):
            _download_drive(m, models_dir, logger)

    if not args.skip_support:
        for s in BULKFORMER_SUPPORT:
            _download_zenodo(s, support_dir, logger)

    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
