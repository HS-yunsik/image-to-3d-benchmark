"""Quick connected-components check on downloaded meshes.

Reads manifest.csv, picks the first N meshes per completed model, runs
ONLY the connected_components calculation, prints distribution per model.
This is meant to confirm whether the n=559 finding from the single-mesh
validation is typical or an outlier.

Output: logs/components_quick_check.txt
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh
from utils import load_mesh

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.csv"
LOG = ROOT / "logs" / "components_quick_check.txt"
CSV_OUT = ROOT / "data" / "components_preview.csv"
LOG.parent.mkdir(exist_ok=True)

N_PER_MODEL = 10  # cap per model


def count_components(path: Path) -> int:
    """Cheaper than full geometry.compute -- just split."""
    m = load_mesh(path)
    if len(m.vertices) == 0:
        return 0
    try:
        parts = m.split(only_watertight=False)
        return len(parts) if parts is not None else 1
    except Exception:
        return -1  # error sentinel


def main() -> int:
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found yet -- nothing downloaded?",
              file=sys.stderr)
        return 1

    # Group manifest rows by model
    by_model: dict[str, list[Path]] = defaultdict(list)
    with MANIFEST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_model[row["model"]].append(Path(row["abs_path"]))

    # Sort each model's list deterministically
    for m in by_model:
        by_model[m].sort()

    print(f"Models in manifest: {len(by_model)}", flush=True)
    for m in sorted(by_model):
        print(f"  {m}: {len(by_model[m])} files downloaded", flush=True)

    print(f"\nRunning components on first {N_PER_MODEL} per model ...\n",
          flush=True)

    results: dict[str, list[int]] = {}
    t0 = time.time()
    for model in sorted(by_model):
        sample = by_model[model][:N_PER_MODEL]
        comps = []
        for p in sample:
            n = count_components(p)
            comps.append(n)
        results[model] = comps
        valid = [c for c in comps if c >= 0]
        if valid:
            print(f"  {model:<20} n={len(valid)}/{len(comps):<3} "
                  f"min={min(valid):>5} med={int(median(valid)):>6} "
                  f"max={max(valid):>6} mean={mean(valid):>7.1f}",
                  flush=True)
        else:
            print(f"  {model:<20} ALL FAILED", flush=True)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)

    # Write detailed log
    with LOG.open("w", encoding="utf-8") as fh:
        fh.write("# components_quick_check\n")
        fh.write(f"# samples per model: up to {N_PER_MODEL}\n\n")
        fh.write(f"{'Model':<22} {'N':>4} {'Min':>6} {'Med':>6} "
                 f"{'Max':>6} {'Mean':>9}   per-mesh values\n")
        fh.write("-" * 110 + "\n")
        for model in sorted(results):
            comps = results[model]
            valid = [c for c in comps if c >= 0]
            if valid:
                fh.write(
                    f"{model:<22} {len(valid):>4} {min(valid):>6} "
                    f"{int(median(valid)):>6} {max(valid):>6} "
                    f"{mean(valid):>9.1f}   {valid}\n"
                )
            else:
                fh.write(f"{model:<22} (no valid samples) {comps}\n")

    # Long-format CSV: one row per (model, file, components)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "filename", "components"])
        writer.writeheader()
        for model in sorted(results):
            sample = by_model[model][:N_PER_MODEL]
            for path, n in zip(sample, results[model]):
                writer.writerow({"model": model, "filename": path.name,
                                 "components": n})

    print(f"\nLog: {LOG}", flush=True)
    print(f"CSV: {CSV_OUT}", flush=True)

    # Headline interpretation
    print("\n" + "=" * 70)
    print("HEADLINE")
    print("=" * 70)
    high_models = []
    low_models = []
    for model, comps in results.items():
        valid = [c for c in comps if c >= 0]
        if not valid:
            continue
        if median(valid) > 100:
            high_models.append((model, int(median(valid))))
        elif median(valid) <= 5:
            low_models.append((model, int(median(valid))))

    if high_models:
        print(f"  HIGH components (median > 100): "
              f"{sorted(high_models, key=lambda x: -x[1])}")
    if low_models:
        print(f"  LOW components (median <= 5): "
              f"{sorted(low_models, key=lambda x: x[1])}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
