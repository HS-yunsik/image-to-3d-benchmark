"""
Download all model output PNG thumbnails from 3d-arena/3d-arena HuggingFace dataset.

Target: data/3darena_thumbs/outputs/{model}/{prompt}.png
Total:  ~2,542 files across 25 models

Usage:
    python scripts/download_thumbnails.py
    python scripts/download_thumbnails.py --only MeshFormer Strawberrry
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

ROOT     = Path(__file__).resolve().parent.parent
SAVE_DIR = ROOT / "data" / "3darena_thumbs"
LOG_FILE = ROOT / "logs" / "thumb_download.log"
REPO_ID  = "3d-arena/3d-arena"

LOG_FILE.parent.mkdir(exist_ok=True)


def log(msg: str) -> None:
    ts  = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="+", metavar="MODEL",
                        help="Download only these model folders")
    args = parser.parse_args()

    from huggingface_hub import list_repo_files, hf_hub_download

    token = os.environ.get("HF_TOKEN") or None

    # ── Enumerate target PNGs ──────────────────────────────────────────────
    log("Listing PNG files from HuggingFace …")
    all_files = list(list_repo_files(REPO_ID, repo_type="dataset", token=token))
    png_files = [
        f for f in all_files
        if f.endswith(".png") and f.startswith("outputs/")
    ]

    if args.only:
        only_set = set(args.only)
        png_files = [f for f in png_files if f.split("/")[1] in only_set]
        log(f"Filtering to models: {args.only}")

    total = len(png_files)
    log(f"Target: {total} PNG files")

    # ── Download loop ──────────────────────────────────────────────────────
    downloaded = 0
    skipped    = 0
    errors     = 0
    t_start    = time.time()

    for i, filepath in enumerate(sorted(png_files), 1):
        local_path = SAVE_DIR / filepath
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            skipped += 1
            if i % 200 == 0:
                elapsed = time.time() - t_start
                log(f"  [{i:4d}/{total}] skip={skipped} dl={downloaded} "
                    f"err={errors}  {elapsed:.0f}s elapsed")
            continue

        try:
            hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filepath,
                local_dir=str(SAVE_DIR),
                local_dir_use_symlinks=False,
                token=token,
            )
            downloaded += 1
        except Exception as e:
            errors += 1
            log(f"  ERROR [{filepath}]: {e}")

        if i % 100 == 0 or downloaded % 50 == 0:
            elapsed = time.time() - t_start
            rate    = downloaded / elapsed if elapsed > 0 else 0
            eta     = (total - i) / rate if rate > 0 else 0
            log(f"  [{i:4d}/{total}] dl={downloaded} skip={skipped} err={errors} "
                f"  {rate:.1f} dl/s  ETA {eta/60:.1f}min")

    elapsed = time.time() - t_start
    log(f"\nDone in {elapsed/60:.1f} min — "
        f"downloaded={downloaded}, skipped={skipped}, errors={errors}")
    log(f"Saved to: {SAVE_DIR / 'outputs'}")


if __name__ == "__main__":
    main()
