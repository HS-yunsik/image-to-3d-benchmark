"""IM-MA sanity check: download first 5 IM-MA meshes and verify they
contain valid geometry (vertex count, watertight, manifold).

Reason: IM-MA's average mesh size is 0.03 MB (30KB), suspicious of empty meshes.
Output: logs/im_ma_validation.txt + console report.
Exit code: 0 if at least one mesh has vertices, 1 if all are empty.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import trimesh
from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "3d-arena/3d-arena"
MODEL = "IM-MA"
N_SAMPLES = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = PROJECT_ROOT / "meshes"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "im_ma_validation.txt"


def load_concat(path: str) -> trimesh.Trimesh:
    """Load a glb/obj possibly as a Scene; return concatenated mesh."""
    obj = trimesh.load(path, force="mesh")
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(list(obj.geometry.values()))
    return obj


def main() -> int:
    print(f"[validate_im_ma] Listing {MODEL} mesh files ...", flush=True)
    all_files = list_repo_files(REPO_ID, repo_type="dataset")
    mesh_files = sorted(
        f for f in all_files
        if f.startswith(f"outputs/{MODEL}/") and f.lower().endswith((".glb", ".obj"))
    )[:N_SAMPLES]

    if not mesh_files:
        print(f"[validate_im_ma] No mesh files found for {MODEL}", file=sys.stderr)
        return 1

    print(f"[validate_im_ma] Downloading + checking {len(mesh_files)} samples ...",
          flush=True)

    rows: list[dict] = []
    for f in mesh_files:
        local_path = hf_hub_download(
            repo_id=REPO_ID, filename=f, repo_type="dataset",
            local_dir=str(LOCAL_DIR),
        )
        size_kb = Path(local_path).stat().st_size / 1024
        try:
            mesh = load_concat(local_path)
            n_verts = len(mesh.vertices)
            n_faces = len(mesh.faces)
            wt = bool(mesh.is_watertight) if n_verts else False
            # manifold edges: trimesh exposes .is_winding_consistent and edge counts
            try:
                non_manifold = len(mesh.edges) - len(mesh.edges_unique)
                # trimesh.Trimesh.edges has every face's 3 edges; non_manifold proxy:
                # edges that appear an odd or >2 number of times => not 2-manifold.
                from collections import Counter
                edge_counts = Counter(map(tuple, mesh.edges_sorted))
                bad = sum(1 for v in edge_counts.values() if v != 2)
                manifold_ratio = 1.0 - (bad / max(len(edge_counts), 1))
            except Exception:
                manifold_ratio = float("nan")

            rows.append({
                "file": Path(f).name,
                "size_kb": round(size_kb, 2),
                "verts": n_verts,
                "faces": n_faces,
                "watertight": wt,
                "manifold_ratio": round(manifold_ratio, 3) if manifold_ratio == manifold_ratio else "nan",
                "error": "",
            })
        except Exception as e:
            rows.append({
                "file": Path(f).name,
                "size_kb": round(size_kb, 2),
                "verts": 0, "faces": 0, "watertight": False,
                "manifold_ratio": "nan", "error": f"{type(e).__name__}: {e}",
            })

    # Decide verdict
    n_empty = sum(1 for r in rows if r["verts"] == 0)
    n_valid = len(rows) - n_empty
    verdict = (
        "ALL_EMPTY" if n_empty == len(rows)
        else "ALL_VALID" if n_empty == 0
        else f"PARTIAL ({n_valid} valid, {n_empty} empty)"
    )

    # Write log
    with LOG_FILE.open("w", encoding="utf-8") as fh:
        fh.write(f"IM-MA validation report\n")
        fh.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Repo: {REPO_ID}\n")
        fh.write(f"Samples: {len(rows)} (first {N_SAMPLES} alphabetical)\n")
        fh.write(f"Verdict: {verdict}\n\n")
        fh.write(
            f"{'File':<60} {'KB':>7} {'Verts':>7} {'Faces':>7} "
            f"{'WT':>4} {'Manif':>6} Error\n"
        )
        fh.write("-" * 120 + "\n")
        for r in rows:
            fh.write(
                f"{r['file']:<60} {r['size_kb']:>7.2f} {r['verts']:>7} "
                f"{r['faces']:>7} {str(r['watertight']):>4} "
                f"{str(r['manifold_ratio']):>6} {r['error']}\n"
            )

    # Console summary
    print("\n" + "=" * 78)
    print(f"IM-MA validation: {verdict}")
    print("=" * 78)
    print(f"{'File':<55} {'KB':>7} {'Verts':>7} {'Faces':>7} {'WT':>4} {'Manif':>6}")
    print("-" * 100)
    for r in rows:
        print(f"{r['file'][:54]:<55} {r['size_kb']:>7.2f} {r['verts']:>7} "
              f"{r['faces']:>7} {str(r['watertight']):>4} {str(r['manifold_ratio']):>6}")
    print(f"\nLog: {LOG_FILE}")

    return 0 if n_valid > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
