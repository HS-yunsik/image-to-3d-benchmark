"""Apply the full metric suite to every mesh in data/manifest.csv.

Reads manifest, runs metrics.compute_all on each row, writes
data/all_metrics.csv. Resume-safe: skips meshes that already have a row
in all_metrics.csv. Errors logged separately to logs/metric_errors.csv.

Memory safeguards:
    - metrics.compute_all already does del + gc.collect() per mesh
    - This script monitors RSS via psutil and logs a WARN line every time a
      mesh pushes process RSS or system-used memory above MEM_WARN_PCT
    - Per-mesh wall-clock time is logged; meshes slower than SLOW_SEC are
      flagged but NOT killed (Windows lacks a clean cross-language SIGALRM
      and a multiprocessing supervisor is overkill for the expected costs
      after switching to the lightweight CC). If a runaway happens we
      Ctrl+C and the resume-safe append means we just skip the offender.

DO NOT RUN until download_all.py finishes (or until you're ready to
process whatever subset is currently downloaded).

    conda activate 3darena
    python run_metrics.py
    python run_metrics.py --only Hunyuan3D-2 TRELLIS   # subset
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(it, **_kw):
        return it

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore[assignment]

from metrics import ALL_METRIC_KEYS, compute_all

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

MANIFEST = DATA_DIR / "manifest.csv"
RESULTS = DATA_DIR / "all_metrics.csv"
ERRORS = LOG_DIR / "metric_errors.csv"
MEMLOG = LOG_DIR / "run_metrics_memory.log"

META_KEYS = ("model", "filename", "file_size_mb", "load_success", "error")
RESULT_FIELDS = list(META_KEYS) + list(ALL_METRIC_KEYS) + ["seconds"]
ERROR_FIELDS = ["timestamp", "model", "filename", "error"]

# Tunables
GC_EVERY = 50          # extra gc.collect() every N meshes
MEM_WARN_PCT = 80.0    # warn when system-wide memory usage crosses this
SLOW_SEC = 30.0        # flag meshes that take longer than this


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


def append_memlog(msg: str) -> None:
    with MEMLOG.open("a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")


def mem_snapshot() -> tuple[float, float, float]:
    """Return (proc_rss_mb, sys_used_pct, sys_avail_gb). All zero if no psutil."""
    if psutil is None:
        return 0.0, 0.0, 0.0
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / (1024 ** 2)
    vm = psutil.virtual_memory()
    return rss, vm.percent, vm.available / (1024 ** 3)


def load_manifest(
    only: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[dict]:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found. Run download_all.py first.",
              file=sys.stderr)
        sys.exit(1)
    exclude_set = set(exclude) if exclude else set()
    rows: list[dict] = []
    with MANIFEST.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if only and r["model"] not in only:
                continue
            if r["model"] in exclude_set:
                continue
            rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these models")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="Skip these models (e.g. incomplete downloads)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap total meshes (debugging)")
    args = ap.parse_args()

    rows = load_manifest(args.only, args.exclude)
    if not rows:
        print("No manifest rows match.", file=sys.stderr)
        return 1

    seen = already_processed()
    todo = [r for r in rows if (r["model"], r["filename"]) not in seen]
    if args.limit:
        todo = todo[: args.limit]

    print(f"manifest rows={len(rows)}  already done={len(seen)}  todo={len(todo)}",
          flush=True)
    print(f"psutil available: {psutil is not None}", flush=True)

    if not todo:
        print("Nothing to do.", flush=True)
        return 0

    rss0, sys_pct0, avail0 = mem_snapshot()
    append_memlog(f"START rss={rss0:.0f}MB sys={sys_pct0:.1f}% avail={avail0:.1f}GB")

    n_ok = n_fail = n_slow = 0
    last_warn_pct = 0.0
    pipeline_t0 = time.time()

    for i, r in enumerate(tqdm(todo, desc="metrics", unit="mesh"), 1):
        path = Path(r["abs_path"])
        t_mesh = time.time()
        try:
            metrics_row = compute_all(path, model=r["model"])
        except Exception as e:  # paranoia
            metrics_row = {
                "model": r["model"], "filename": r["filename"],
                "file_size_mb": None, "load_success": False,
                "error": f"{type(e).__name__}: {e}"[:500],
            }
            for k in ALL_METRIC_KEYS:
                metrics_row[k] = None
        elapsed = time.time() - t_mesh
        metrics_row["seconds"] = round(elapsed, 3)

        append_result(metrics_row)
        if metrics_row.get("load_success") and not metrics_row.get("error"):
            n_ok += 1
        else:
            n_fail += 1
            append_error(metrics_row["model"], metrics_row["filename"],
                         metrics_row.get("error", ""))

        if elapsed > SLOW_SEC:
            n_slow += 1
            append_memlog(
                f"SLOW {elapsed:.1f}s {r['model']}/{r['filename']} "
                f"size={r.get('size_mb', '?')}MB"
            )

        if i % GC_EVERY == 0:
            gc.collect()
            rss, sys_pct, avail = mem_snapshot()
            append_memlog(
                f"i={i}/{len(todo)} ok={n_ok} fail={n_fail} "
                f"rss={rss:.0f}MB sys={sys_pct:.1f}% avail={avail:.1f}GB"
            )
            # Once we cross MEM_WARN_PCT, log a clearly-marked warning
            # but don't auto-kill; user can Ctrl+C if needed.
            if sys_pct >= MEM_WARN_PCT and sys_pct - last_warn_pct >= 2.0:
                msg = (f"MEMORY WARN: system at {sys_pct:.1f}% "
                       f"(avail {avail:.1f}GB) after {i} meshes")
                append_memlog(msg)
                print(f"\n  !! {msg}", flush=True)
                last_warn_pct = sys_pct

    elapsed_total = time.time() - pipeline_t0
    rss, sys_pct, avail = mem_snapshot()
    append_memlog(f"END ok={n_ok} fail={n_fail} slow={n_slow} "
                  f"elapsed={elapsed_total/60:.1f}min "
                  f"rss={rss:.0f}MB sys={sys_pct:.1f}%")
    print(f"\nDone: ok={n_ok} fail={n_fail} slow(>{SLOW_SEC}s)={n_slow} "
          f"elapsed={elapsed_total/60:.1f}min "
          f"rate={(n_ok+n_fail)/max(elapsed_total,0.01):.2f} meshes/s",
          flush=True)
    print(f"Results: {RESULTS}")
    print(f"Errors:  {ERRORS}")
    print(f"Memlog:  {MEMLOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
