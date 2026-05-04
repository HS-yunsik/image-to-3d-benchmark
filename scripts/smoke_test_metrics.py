"""Quick smoke test: run all 4 metric modules against one mesh per known
Phase-0 model. Confirms imports + signatures + that nothing crashes.

Run:
    conda activate 3darena
    python scripts/smoke_test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import load_mesh
from metrics import geometry, topology, uv, pbr

MESHES = [
    ("Hunyuan3D-2",   "meshes/outputs/Hunyuan3D-2/A_cartoon_house_with_red_roof.glb"),
    ("Hunyuan3D-2.1", "meshes/outputs/Hunyuan3D-2.1/A_cartoon_house_with_red_roof.glb"),
    ("TRELLIS",       "meshes/outputs/TRELLIS/A_cartoon_house_with_red_roof.glb"),
    ("InstantMesh",   "meshes/outputs/InstantMesh/A_cartoon_house_with_red_roof.glb"),
    ("Hi3DGen",       "meshes/outputs/Hi3DGen/A_cartoon_house_with_red_roof.glb"),
]


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    failures = 0

    print(f"{'Model':<16} {'V':>7} {'F':>7} {'WT':>5} {'Mfd':>6} "
          f"{'Tri%':>5} {'UV':>4} {'Tex':>4} {'Ch':>3} {'Res':>5}")
    print("-" * 80)

    for name, rel in MESHES:
        path = root / rel
        if not path.exists():
            print(f"{name:<16} (missing)")
            continue
        try:
            mesh = load_mesh(path)
            g = geometry.compute(mesh)
            t = topology.compute(mesh)
            u = uv.compute(mesh)
            p = pbr.compute(mesh)
            print(
                f"{name:<16} "
                f"{g['vertex_count']:>7} {g['face_count']:>7} "
                f"{str(g['watertight'])[0]:>5} "
                f"{g['manifold_edge_ratio']:>6.3f} "
                f"{t['triangle_ratio']*100:>4.0f}% "
                f"{str(u['has_uv'])[0]:>4} "
                f"{str(p['has_texture'])[0]:>4} "
                f"{p['pbr_channel_count']:>3} "
                f"{p['texture_resolution']:>5}"
            )
        except Exception as e:
            print(f"{name:<16} ERROR: {type(e).__name__}: {e}")
            failures += 1

    print("\nSmoke test:", "PASS" if failures == 0 else f"FAIL ({failures})")
    return failures


if __name__ == "__main__":
    sys.exit(main())
