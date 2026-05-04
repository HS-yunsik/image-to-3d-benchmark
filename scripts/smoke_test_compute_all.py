"""Smoke test the metrics.compute_all() entry point on Phase-0 meshes."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from metrics import ALL_METRIC_KEYS, compute_all

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    ("Hunyuan3D-2",   "Hunyuan3D-2/A_cartoon_house_with_red_roof.glb"),
    ("Hunyuan3D-2.1", "Hunyuan3D-2.1/A_cartoon_house_with_red_roof.glb"),
    ("TRELLIS",       "TRELLIS/A_cartoon_house_with_red_roof.glb"),
    ("InstantMesh",   "InstantMesh/A_cartoon_house_with_red_roof.glb"),
    ("Hi3DGen",       "Hi3DGen/A_cartoon_house_with_red_roof.glb"),
]


def main() -> int:
    print(f"Schema: {len(ALL_METRIC_KEYS)} metric keys + 5 meta keys")
    print(f"  metrics: {ALL_METRIC_KEYS}\n")

    failures = 0
    for name, rel in TARGETS:
        path = ROOT / "meshes" / "outputs" / rel
        if not path.exists():
            print(f"{name}: (missing)")
            continue
        try:
            row = compute_all(path, model=name)
            ok = row["load_success"] and not row["error"]
            sym = "OK  " if ok else "WARN"
            print(f"[{sym}] {name}")
            print(f"        file_size_mb={row['file_size_mb']}  "
                  f"verts={row['vertex_count']}  faces={row['face_count']}  "
                  f"comps={row['connected_components']}  "
                  f"area={row['surface_area']:.2f}  "
                  f"diag={row['bbox_diag']:.2f}  "
                  f"WT={row['watertight']}  "
                  f"UV={row['has_uv']}  Tex={row['has_texture']}")
            if row["error"]:
                print(f"        error: {row['error']}")
                failures += 1
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
            failures += 1

    print(f"\n{'PASS' if failures == 0 else 'FAIL'} ({failures} issues)")
    return failures


if __name__ == "__main__":
    sys.exit(main())
