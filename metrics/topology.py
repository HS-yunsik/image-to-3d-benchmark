"""D2: Topology metrics.

Phase 0 confirmed that every 3D Arena output is 100% triangulated -- they
are all marching-cubes derivatives. trimesh.load(force="mesh") additionally
triangulates any quad/ngon source faces, so by the time we see the mesh it's
guaranteed to be triangle-only. We surface that as an explicit metric (a
sanity check column for the paper) and reserve true quad/ngon detection for
a Phase 2 spot-check via Blender bpy on the original GLB.

Metric definitions (Phase 1)
----------------------------
triangle_ratio : float in [0, 1]
    Fraction of faces with exactly 3 vertices. Always 1.0 after trimesh.load
    in our pipeline; included as a column so the analysis can document this
    universal property of the dataset.

non_triangle_ratio : float in [0, 1]
    1 - triangle_ratio. Always 0.0 in our pipeline.
"""
from __future__ import annotations

from typing import Any

import trimesh


def compute(mesh: trimesh.Trimesh) -> dict[str, Any]:
    n_f = len(mesh.faces) if mesh.faces is not None else 0
    if n_f == 0:
        return {"triangle_ratio": float("nan"), "non_triangle_ratio": float("nan")}

    if mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3:
        return {"triangle_ratio": 1.0, "non_triangle_ratio": 0.0}

    # Defensive fallback for hypothetical mixed-face cases
    tri = sum(1 for f in mesh.faces if len(f) == 3)
    return {
        "triangle_ratio": round(tri / n_f, 4),
        "non_triangle_ratio": round(1.0 - tri / n_f, 4),
    }
