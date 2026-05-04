"""Step 1: Discover and classify all models in the 3D Arena dataset.

Enumerates outputs/{model}/ folders, counts file extensions, and classifies
each model as mesh / splat / unknown. Writes data/model_inventory.csv and
prints a human-readable summary so the user can confirm before download.

Run:
    conda activate 3darena
    python scripts/discover_models.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, list_repo_files
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "3d-arena/3d-arena"
REPO_TYPE = "dataset"

# Extensions
MESH_EXTS = {".glb", ".obj", ".gltf", ".fbx"}
SPLAT_EXTS = {".ply", ".splat", ".spz"}
THUMB_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# Per CLAUDE.md, these are known splat models — keep an explicit deny list
# so they're flagged even if the auto-rule misses (e.g. mixed extension cases).
EXPLICIT_SPLAT = {
    "TRELLIS-3DGS",
    "LGM",
    "SAM-3D-Objects-3DGS",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
INVENTORY_CSV = DATA_DIR / "model_inventory.csv"

MAX_PER_MODEL_TARGET = 50  # Used only for size estimate


def fetch_repo_files() -> list[str]:
    """List every file in the dataset repo."""
    print(f"[discover] Listing files in {REPO_ID} ...", flush=True)
    try:
        files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    except HfHubHTTPError as e:
        print(f"[discover] Failed to list repo files: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[discover] Total files in repo: {len(files):,}", flush=True)
    return files


def fetch_repo_tree_with_sizes() -> dict[str, int]:
    """Map filename -> size_bytes for everything under outputs/.

    Uses HfApi.list_repo_tree which returns size info (list_repo_files does not).
    Recurses into outputs/ only to keep it cheap.
    """
    api = HfApi()
    sizes: dict[str, int] = {}
    print("[discover] Fetching file sizes (outputs/ tree) ...", flush=True)
    try:
        for entry in api.list_repo_tree(
            REPO_ID, repo_type=REPO_TYPE, path_in_repo="outputs", recursive=True
        ):
            # entry can be a RepoFile or RepoFolder; we want files
            if getattr(entry, "size", None) is not None:
                sizes[entry.path] = entry.size
    except Exception as e:
        print(f"[discover] WARN: tree size fetch failed ({type(e).__name__}: {e}); "
              "size estimates will be unavailable.", file=sys.stderr)
    return sizes


def classify(ext_counts: Counter, model_name: str) -> str:
    """Return one of: mesh / splat / unknown.

    Rule:
      - explicit deny list -> splat
      - majority of *content* files (excluding thumbnails) are mesh exts -> mesh
      - majority are splat exts -> splat
      - otherwise -> unknown
    """
    if model_name in EXPLICIT_SPLAT:
        return "splat"

    content = Counter()
    for ext, n in ext_counts.items():
        if ext in THUMB_EXTS:
            continue
        content[ext] += n

    total = sum(content.values())
    if total == 0:
        return "unknown"

    mesh_n = sum(n for ext, n in content.items() if ext in MESH_EXTS)
    splat_n = sum(n for ext, n in content.items() if ext in SPLAT_EXTS)

    if mesh_n / total >= 0.5:
        return "mesh"
    if splat_n / total >= 0.5:
        return "splat"
    return "unknown"


def main() -> None:
    files = fetch_repo_files()
    sizes = fetch_repo_tree_with_sizes()

    # Group files under outputs/{model}/...
    per_model_exts: dict[str, Counter] = defaultdict(Counter)
    per_model_files: dict[str, list[str]] = defaultdict(list)
    per_model_size: dict[str, int] = defaultdict(int)

    for f in files:
        if not f.startswith("outputs/"):
            continue
        parts = f.split("/", 2)
        if len(parts) < 3:
            continue  # outputs/<model>/  must have a child
        _, model, rest = parts
        ext = "." + rest.rsplit(".", 1)[-1].lower() if "." in rest else ""
        per_model_exts[model][ext] += 1
        per_model_files[model].append(f)
        per_model_size[model] += sizes.get(f, 0)

    if not per_model_exts:
        print("[discover] ERROR: no outputs/{model}/ files found.", file=sys.stderr)
        sys.exit(1)

    rows = []
    for model in sorted(per_model_exts):
        exts = per_model_exts[model]
        files_for_model = per_model_files[model]
        cls = classify(exts, model)

        # Count just the candidate mesh files for that model (what we'd actually download)
        mesh_files = [
            f for f in files_for_model
            if any(f.lower().endswith(e) for e in MESH_EXTS)
        ]
        n_mesh_files = len(mesh_files)

        # Average size per mesh file (bytes)
        mesh_size_total = sum(sizes.get(f, 0) for f in mesh_files)
        avg_mb = (mesh_size_total / n_mesh_files / 1_000_000) if n_mesh_files else 0.0

        # Estimated download size if we cap at MAX_PER_MODEL_TARGET
        n_dl = min(n_mesh_files, MAX_PER_MODEL_TARGET)
        est_dl_mb = avg_mb * n_dl

        ext_str = ", ".join(f"{e or '<none>'}={n}" for e, n in exts.most_common())

        rows.append({
            "model": model,
            "classification": cls,
            "n_files_total": sum(exts.values()),
            "n_mesh_files": n_mesh_files,
            "ext_distribution": ext_str,
            "avg_mesh_size_mb": round(avg_mb, 2),
            "est_dl_mb_at_50": round(est_dl_mb, 1),
        })

    # Write CSV
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[discover] Wrote {INVENTORY_CSV}", flush=True)

    # Pretty print summary, grouped by classification
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cls[r["classification"]].append(r)

    print("\n" + "=" * 78)
    print(f"3D Arena model inventory ({len(rows)} model folders found)")
    print("=" * 78)

    for cls in ("mesh", "splat", "unknown"):
        bucket = by_cls.get(cls, [])
        if not bucket:
            continue
        print(f"\n--- {cls.upper()} ({len(bucket)} models) ---")
        print(f"{'Model':<28} {'Files':>6} {'Mesh':>5} {'AvgMB':>7} {'Est@50':>8}")
        print("-" * 60)
        total_est = 0.0
        for r in sorted(bucket, key=lambda x: -x["n_mesh_files"]):
            print(f"{r['model']:<28} {r['n_files_total']:>6} "
                  f"{r['n_mesh_files']:>5} {r['avg_mesh_size_mb']:>7.2f} "
                  f"{r['est_dl_mb_at_50']:>7.1f}M")
            if cls == "mesh":
                total_est += r["est_dl_mb_at_50"]
        if cls == "mesh":
            print(f"\n  >>> Estimated total download (cap=50/model): "
                  f"{total_est/1024:.2f} GB")

    # Headline numbers
    n_mesh = len(by_cls.get("mesh", []))
    n_splat = len(by_cls.get("splat", []))
    n_unknown = len(by_cls.get("unknown", []))
    print("\n" + "=" * 78)
    print(f"Summary: mesh={n_mesh}, splat={n_splat} (excluded), unknown={n_unknown} (manual review)")
    print("=" * 78)
    print(f"Inventory CSV: {INVENTORY_CSV}")


if __name__ == "__main__":
    main()
