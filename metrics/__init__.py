"""Phase 1 metric suite for image-to-3D production-fitness evaluation.

Dimensions x metrics:
    D1 Geometry (7) : watertight, manifold/boundary/nonmanifold edge ratios,
                      connected_components, vertex_count, face_count
    D2 Topology (2) : triangle_ratio, non_triangle_ratio
    D3 UV       (3) : has_uv, uv_bbox_efficiency, uv_packed_area
    D4 PBR      (3) : has_texture, pbr_channel_count, texture_resolution
    D5 Scale    (2) : surface_area, bbox_diag (for normalization)

Total: 17 metrics + meta (model, file_size_mb, load_success, error).

The single entry point downstream code should use is `compute_all(mesh_path)`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

from . import geometry, pbr, scale, topology, uv  # noqa: F401

__all__ = ["geometry", "topology", "uv", "pbr", "scale", "compute_all",
           "ALL_METRIC_KEYS"]


# Sentinel values used when a metric module raises an exception. Keep keys
# stable so the resulting CSV has a consistent schema even when individual
# metric calls fail mid-pipeline.
_GEO_KEYS = (
    "watertight", "manifold_edge_ratio", "boundary_edge_ratio",
    "nonmanifold_edge_ratio", "connected_components",
    "vertex_count", "face_count",
)
_TOP_KEYS = ("triangle_ratio", "non_triangle_ratio")
_UV_KEYS = ("has_uv", "uv_bbox_efficiency", "uv_packed_area")
_PBR_KEYS = ("has_texture", "pbr_channel_count", "texture_resolution")
_SCL_KEYS = ("surface_area", "bbox_diag")

ALL_METRIC_KEYS = _GEO_KEYS + _TOP_KEYS + _UV_KEYS + _PBR_KEYS + _SCL_KEYS


def _empty_metrics() -> dict[str, Any]:
    return {k: None for k in ALL_METRIC_KEYS}


def _load_concat(path: Path) -> trimesh.Trimesh:
    """Load any glb/obj as a single concatenated Trimesh.

    Mirrors utils.load_mesh but kept inline to avoid an import cycle with
    the test suite's sys.path tricks. Returns an empty Trimesh on Scene with
    no geometry.
    """
    obj = trimesh.load(str(path), force="mesh")
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(list(obj.geometry.values()))
    if isinstance(obj, trimesh.Trimesh):
        return obj
    return trimesh.Trimesh()


def compute_all(
    mesh_path: str | Path,
    model: str | None = None,
) -> dict[str, Any]:
    """Compute all Phase 1 + D5 metrics for a single mesh file.

    Returns a flat dict including:
      - meta: model, filename, file_size_mb, load_success, error
      - one entry per metric in ALL_METRIC_KEYS

    Failures in individual metric modules are caught so other metrics still
    run. If load itself fails, all metric values are None and load_success
    is False with the error captured.
    """
    p = Path(mesh_path)
    out: dict[str, Any] = {
        "model": model or (p.parent.name if p.parent.name else ""),
        "filename": p.name,
        "file_size_mb": round(p.stat().st_size / (1024 * 1024), 4) if p.exists() else None,
        "load_success": False,
        "error": "",
    }
    out.update(_empty_metrics())

    if not p.exists():
        out["error"] = "file_missing"
        return out

    try:
        mesh = _load_concat(p)
    except Exception as e:
        out["error"] = f"load_failed: {type(e).__name__}: {e}"[:500]
        return out

    out["load_success"] = True

    # Run each metric module independently; record per-module errors but
    # keep going so downstream code always sees the same schema.
    errors: list[str] = []
    for module, keys in (
        (geometry, _GEO_KEYS),
        (topology, _TOP_KEYS),
        (uv, _UV_KEYS),
        (pbr, _PBR_KEYS),
        (scale, _SCL_KEYS),
    ):
        try:
            res = module.compute(mesh)
            for k in keys:
                if k in res:
                    out[k] = res[k]
        except Exception as e:
            errors.append(f"{module.__name__}: {type(e).__name__}: {e}")

    if errors:
        out["error"] = " | ".join(errors)[:500]

    return out
