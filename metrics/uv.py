"""D3: UV metrics.

Two complementary signals:

    has_uv : bool
        Whether the mesh carries any per-vertex UV coordinates.

    uv_bbox_efficiency : float in [0, 1]
        Bounding-box area of UV coordinates (clipped to the unit square).
        Fast coarse proxy for "atlas usage". A mesh that places UVs at the
        four corners would score 1.0 even if the actual triangles cover
        almost no area -- so this metric overestimates good packing.

    uv_packed_area : float >= 0
        Sum of triangulated UV-space face areas (shoelace formula). This is
        the *actual* area used. Interpretation:
            ~ 1.0  : ideal packing (UV atlas roughly fills the unit square)
            < 1.0  : atlas wastage (UVs cover < unit square)
            > 1.0  : UV overlap, i.e. the same texel is reused across faces
        Combined with bbox_efficiency, it tells us how well the atlas was packed.

All three return NaN-equivalents when there are no UVs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import trimesh


def _triangle_uv_area_sum(uv: np.ndarray, faces: np.ndarray) -> float:
    """Sum of |UV-triangle| areas using the shoelace / cross-product formula."""
    if len(faces) == 0:
        return 0.0
    pts = uv[faces]  # shape (n_faces, 3, 2)
    a = pts[:, 1] - pts[:, 0]
    b = pts[:, 2] - pts[:, 0]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return float(np.abs(cross).sum() * 0.5)


def compute(mesh: trimesh.Trimesh) -> dict[str, Any]:
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None) if visual is not None else None

    if uv is None or len(uv) == 0:
        return {
            "has_uv": False,
            "uv_bbox_efficiency": float("nan"),
            "uv_packed_area": float("nan"),
        }

    uv = np.asarray(uv, dtype=float)
    u_range = float(uv[:, 0].max() - uv[:, 0].min())
    v_range = float(uv[:, 1].max() - uv[:, 1].min())
    bbox_area = max(u_range, 0.0) * max(v_range, 0.0)
    bbox_eff = min(bbox_area, 1.0)

    packed = _triangle_uv_area_sum(uv, np.asarray(mesh.faces))

    return {
        "has_uv": True,
        "uv_bbox_efficiency": round(bbox_eff, 4),
        "uv_packed_area": round(packed, 4),
    }
