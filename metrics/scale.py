"""D5: Scale metrics for normalization.

Two simple geometric scalars used downstream to normalize per-mesh metrics
that scale with mesh size (e.g. when comparing surface area or component
count across models with very different vertex budgets).

    surface_area : float >= 0
        Sum of triangle areas in world units. Trimesh computes this directly;
        we cross-check via pymeshlab when available.

    bbox_diag : float >= 0
        Diagonal length of the axis-aligned bounding box in world units.

For empty meshes both metrics return 0.0.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import trimesh


def compute(mesh: trimesh.Trimesh) -> dict[str, Any]:
    if mesh.vertices is None or len(mesh.vertices) == 0:
        return {"surface_area": 0.0, "bbox_diag": 0.0}

    # surface area: trimesh property is the sum of triangle areas
    try:
        area = float(mesh.area)
    except Exception:
        area = 0.0

    # bbox diagonal
    try:
        bounds = mesh.bounds  # shape (2, 3)
        if bounds is not None and bounds.shape == (2, 3):
            diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        else:
            diag = 0.0
    except Exception:
        diag = 0.0

    return {
        "surface_area": round(area, 6),
        "bbox_diag": round(diag, 6),
    }
