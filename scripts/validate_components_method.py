"""Confirm the new lightweight CC counts EXACTLY match mesh.split().

For each Phase-0 mesh, compute connected_components via:
  1. mesh.split(only_watertight=False) -> len()  (the heavyweight reference)
  2. metrics.geometry._count_components_lightweight(mesh)  (the new method)

A single discrepancy aborts with non-zero exit so we don't silently use a
wrong measure. Logs to logs/components_method_validation.txt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh

from metrics.geometry import _count_components_lightweight
from utils import load_mesh

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "components_method_validation.txt"
LOG.parent.mkdir(exist_ok=True)

TARGETS = [
    ("Hunyuan3D-2",   "Hunyuan3D-2/A_cartoon_house_with_red_roof.glb"),
    ("Hunyuan3D-2.1", "Hunyuan3D-2.1/A_cartoon_house_with_red_roof.glb"),
    ("TRELLIS",       "TRELLIS/A_cartoon_house_with_red_roof.glb"),
    ("InstantMesh",   "InstantMesh/A_cartoon_house_with_red_roof.glb"),
    ("Hi3DGen",       "Hi3DGen/A_cartoon_house_with_red_roof.glb"),
]


def main() -> int:
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    log("# components_method_validation")
    log(f"{'Model':<16} {'Verts':>8} {'Faces':>8} "
        f"{'split()':>9} {'split_s':>8} "
        f"{'lwt':>9} {'lwt_s':>8} {'match':>6}")
    log("-" * 90)

    n_diff = 0
    for name, rel in TARGETS:
        path = ROOT / "meshes" / "outputs" / rel
        if not path.exists():
            log(f"{name:<16} (missing)")
            continue
        mesh = load_mesh(path)

        # Reference: heavyweight split
        t0 = time.time()
        try:
            parts = mesh.split(only_watertight=False)
            ref = len(parts) if parts is not None else 1
        except Exception as e:
            ref = -1
            log(f"  split() error on {name}: {type(e).__name__}: {e}")
        t_ref = time.time() - t0

        # New method
        t0 = time.time()
        new = _count_components_lightweight(mesh)
        t_new = time.time() - t0

        match = (ref == new)
        if not match:
            n_diff += 1
        log(f"{name:<16} {len(mesh.vertices):>8} {len(mesh.faces):>8} "
            f"{ref:>9} {t_ref:>8.2f} "
            f"{new:>9} {t_new:>8.2f} "
            f"{('OK' if match else 'DIFF'):>6}")

    log("")
    log(f"Verdict: {n_diff} mismatches out of {len(TARGETS)}")

    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nLog: {LOG}")
    return 0 if n_diff == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
