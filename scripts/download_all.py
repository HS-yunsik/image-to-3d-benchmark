"""Step 2: Download all mesh-based model outputs from 3D Arena.

- Reads data/model_inventory.csv to know which models are mesh-classified.
- Per model, downloads up to MAX_PER_MODEL alphabetically-sorted .glb/.obj files.
- Resumes via HF cache (already-downloaded files are no-op).
- Logs failures to logs/download_errors.csv (append).
- Writes data/manifest.csv at the end (and incrementally on each success).
- Tags Strawb3rry and Strawberrry as anonymous=True in manifest.

Run:
    conda activate 3darena
    python scripts/download_all.py --max-per-model 100
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "3d-arena/3d-arena"
REPO_TYPE = "dataset"
MESH_EXTS = (".glb", ".obj")

# Anonymous-submission models per 3D Arena Table 1
ANONYMOUS_MODELS = {"Strawb3rry", "Strawberrry"}

# Per CLAUDE.md, large per-mesh size — flag separately in progress display
LARGE_MODELS = {"Hi3DGen", "TripoSG", "Meshy-6", "Zaohaowu3D",
                "Unique3D", "TRELLIS.2-4B", "MeshFormer"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = PROJECT_ROOT / "meshes"
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

INVENTORY_CSV = DATA_DIR / "model_inventory.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
ERROR_CSV = LOG_DIR / "download_errors.csv"
PROGRESS_LOG = LOG_DIR / "download_progress.log"

MANIFEST_FIELDS = [
    "model", "filename", "prompt_name", "format",
    "size_mb", "abs_path", "anonymous", "downloaded_at",
]
ERROR_FIELDS = ["timestamp", "model", "filename", "error_type", "error_msg"]


def log(msg: str, fh=None) -> None:
    """Print to stdout AND optionally append to a log file."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def load_mesh_models() -> list[str]:
    """Read inventory CSV and return list of mesh-classified model names."""
    if not INVENTORY_CSV.exists():
        print(f"ERROR: {INVENTORY_CSV} missing. Run discover_models.py first.",
              file=sys.stderr)
        sys.exit(1)
    models = []
    with INVENTORY_CSV.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row["classification"] == "mesh":
                models.append(row["model"])
    return sorted(models)


def append_error(model: str, filename: str, err: BaseException) -> None:
    """Append a row to logs/download_errors.csv (creates header on first write)."""
    new_file = not ERROR_CSV.exists()
    with ERROR_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ERROR_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model,
            "filename": filename,
            "error_type": type(err).__name__,
            "error_msg": str(err)[:500],
        })


def append_manifest_row(row: dict) -> None:
    """Incremental manifest write (so partial progress survives interrupts)."""
    new_file = not MANIFEST_CSV.exists()
    with MANIFEST_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def already_in_manifest() -> set[tuple[str, str]]:
    """Return {(model, filename), ...} for entries already manifested."""
    if not MANIFEST_CSV.exists():
        return set()
    seen = set()
    with MANIFEST_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            seen.add((r["model"], r["filename"]))
    return seen


def fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1024 * 1024 * 1024:
        return f"{n_bytes / (1024**3):.2f} GB"
    return f"{n_bytes / (1024**2):.1f} MB"


def download_one_model(
    model: str, max_n: int, all_files: list[str],
    seen: set[tuple[str, str]], progress_fh,
) -> tuple[int, int, int]:
    """Download up to max_n meshes for one model.

    Returns (n_ok, n_fail, total_bytes_this_model).
    """
    files = sorted(
        f for f in all_files
        if f.startswith(f"outputs/{model}/")
        and f.lower().endswith(MESH_EXTS)
    )[:max_n]

    is_large = model in LARGE_MODELS
    is_anon = model in ANONYMOUS_MODELS
    tag = []
    if is_large: tag.append("LARGE")
    if is_anon: tag.append("ANON")
    tag_str = f" [{', '.join(tag)}]" if tag else ""

    log(f">>> {model}{tag_str}: {len(files)} files to attempt", progress_fh)

    n_ok = n_fail = 0
    total_bytes = 0
    t0 = time.time()

    for i, fpath in enumerate(files, 1):
        fname = fpath.split("/")[-1]
        if (model, fname) in seen:
            n_ok += 1
            continue
        try:
            local_path = hf_hub_download(
                repo_id=REPO_ID, filename=fpath, repo_type=REPO_TYPE,
                local_dir=str(LOCAL_DIR),
            )
            size_b = Path(local_path).stat().st_size
            total_bytes += size_b
            n_ok += 1
            append_manifest_row({
                "model": model,
                "filename": fname,
                "prompt_name": Path(fname).stem,
                "format": Path(fname).suffix.lstrip(".").lower(),
                "size_mb": round(size_b / (1024 * 1024), 3),
                "abs_path": local_path,
                "anonymous": is_anon,
                "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            })
        except (HfHubHTTPError, OSError, Exception) as e:
            n_fail += 1
            append_error(model, fname, e)

        # Progress update every 10 files (or at end), more often for LARGE models
        every = 5 if is_large else 10
        if i % every == 0 or i == len(files):
            elapsed = time.time() - t0
            rate = total_bytes / max(elapsed, 0.01)
            log(
                f"    {model}: {i}/{len(files)} "
                f"ok={n_ok} fail={n_fail} "
                f"size={fmt_size(total_bytes)} "
                f"({fmt_size(int(rate))}/s)",
                progress_fh,
            )

    log(f"<<< {model} done: ok={n_ok} fail={n_fail} "
        f"total={fmt_size(total_bytes)}", progress_fh)
    return n_ok, n_fail, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-model", type=int, default=100)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only download these models (space-separated)")
    args = parser.parse_args()

    models = load_mesh_models()
    if args.only:
        keep = set(args.only)
        unknown = keep - set(models)
        if unknown:
            print(f"WARN: unknown / non-mesh models in --only: {sorted(unknown)}",
                  file=sys.stderr)
        models = [m for m in models if m in keep]

    if not models:
        print("No mesh-classified models to download.", file=sys.stderr)
        sys.exit(1)

    seen = already_in_manifest()
    if seen:
        print(f"[download] Resuming - {len(seen)} files already in manifest", flush=True)

    print(f"[download] Listing repo files (one network call) ...", flush=True)
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)

    progress_fh = PROGRESS_LOG.open("a", encoding="utf-8")
    log(f"=== Download session START | models={len(models)} | "
        f"cap={args.max_per_model} ===", progress_fh)

    cum_ok = cum_fail = 0
    cum_bytes = 0
    started = time.time()

    for idx, model in enumerate(models, 1):
        log(f"\n[{idx}/{len(models)}] === {model} ===", progress_fh)
        try:
            ok, fail, nbytes = download_one_model(
                model, args.max_per_model, all_files, seen, progress_fh,
            )
        except Exception as e:
            log(f"FATAL on {model}: {type(e).__name__}: {e}", progress_fh)
            log(traceback.format_exc(), progress_fh)
            ok = fail = nbytes = 0
        cum_ok += ok
        cum_fail += fail
        cum_bytes += nbytes
        log(f"  cumulative: ok={cum_ok} fail={cum_fail} "
            f"size={fmt_size(cum_bytes)} "
            f"elapsed={(time.time()-started)/60:.1f}min", progress_fh)

    log(f"\n=== Download session END === ok={cum_ok} fail={cum_fail} "
        f"total={fmt_size(cum_bytes)}", progress_fh)
    progress_fh.close()

    print(f"\nManifest: {MANIFEST_CSV}")
    print(f"Errors:   {ERROR_CSV}")
    print(f"Log:      {PROGRESS_LOG}")


if __name__ == "__main__":
    main()
