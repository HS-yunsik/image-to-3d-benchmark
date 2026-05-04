"""Apply the full metric suite to every mesh in data/manifest.csv.

Reads manifest, runs metrics.compute_all on each row, writes
data/all_metrics.csv. Resume-safe: skips meshes that already have a row
in all_metrics.csv. Errors logged separately to logs/metric_errors.csv.

DO NOT RUN until download_all.py finishes (or until you're ready to
process whatever subset is currently downloaded).

    conda activate 3darena
    python run_metrics.py
    python run_metrics.py --only Hunyuan3D-2 TRELLIS   # subset
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# tqdm if available, otherwise dumb fallback
try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(it, **_kw):
        return it

from metrics import ALL_METRIC_KEYS, compute_all

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MANIFEST = DATA_DIR / "manifest.csv"
RESULTS = DATA_DIR / "all_metrics.csv"
ERRORS = LOG_DIR / "metric_errors.csv"

META_KEYS = ("model", "filename", "file_size_mb", "load_success", "error")
RESULT_FIELDS = list(META_KEYS) + list(ALL_METRIC_KEYS)
ERROR_FIELDS = ["timestamp", "model", "filename", "error"]


def already_processed() -> set[tuple[str, str]]:
    if not RESULTS.exists():
        return set()
    seen = set()
    with RESULTS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            seen.add((r["model"], r["filename"]))
    return seen


def append_result(row: dict) -> None:
    new_file = not RESULTS.exists()
    with RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS,
                                extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def append_error(model: str, filename: str, msg: str) -> None:
    new_file = not ERRORS.exists()
    with ERRORS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ERROR_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model, "filename": filename, "error": msg[:500],
        })


def load_manifest(only: list[str] | None = None) -> list[dict]:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run download_all.py first.",
              file=sys.stderr)
        sys.exit(1)
    rows: list[dict] = []
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if only and r["model"] not in only:
                continue
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these models")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total meshes (debugging)")
    args = ap.parse_args()

    rows = load_manifest(args.only)
    if not rows:
        print("No manifest rows match.", file=sys.stderr)
        return 1

    seen = already_processed()
    todo = [r for r in rows if (r["model"], r["filename"]) not in seen]
    if args.limit:
        todo = todo[: args.limit]

    print(f"manifest rows={len(rows)}  already done={len(seen)}  todo={len(todo)}",
          flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        return 0

    n_ok = n_fail = 0
    t0 = time.time()
    for r in tqdm(todo, desc="metrics", unit="mesh"):
        path = Path(r["abs_path"])
        try:
            metrics_row = compute_all(path, model=r["model"])
        except Exception as e:  # paranoia -- compute_all should not raise
            metrics_row = {
                "model": r["model"], "filename": r["filename"],
                "file_size_mb": None, "load_success": False,
                "error": f"{type(e).__name__}: {e}"[:500],
            }
            for k in ALL_METRIC_KEYS:
                metrics_row[k] = None

        append_result(metrics_row)
        if metrics_row.get("load_success") and not metrics_row.get("error"):
            n_ok += 1
        else:
            n_fail += 1
            append_error(metrics_row["model"], metrics_row["filename"],
                         metrics_row.get("error", ""))

    elapsed = time.time() - t0
    print(f"\nDone: ok={n_ok} fail={n_fail} elapsed={elapsed/60:.1f}min "
          f"rate={(n_ok+n_fail)/max(elapsed,0.01):.2f} meshes/s",
          flush=True)
    print(f"Results: {RESULTS}")
    print(f"Errors:  {ERRORS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
