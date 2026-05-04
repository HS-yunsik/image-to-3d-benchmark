"""Shared helpers for loading and inspecting 3D Arena meshes.

Refactored from phase0_*.py — keeps the I/O surface in one place so metric
modules can stay focused on their respective dimensions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import trimesh


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    """Load a glb/obj/gltf file and return a single concatenated Trimesh.

    trimesh.load returns either a Trimesh or a Scene. Scenes are flattened by
    concatenating all geometries — necessary because 3D Arena outputs sometimes
    wrap a single mesh in a Scene and sometimes don't.
    """
    obj = trimesh.load(str(path), force="mesh")
    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            return trimesh.Trimesh()
        return trimesh.util.concatenate(list(obj.geometry.values()))
    if isinstance(obj, trimesh.Trimesh):
        return obj
    # PointCloud or anything else: return an empty mesh so callers can
    # treat "no geometry" uniformly.
    return trimesh.Trimesh()


def load_scene(path: str | Path) -> Optional[trimesh.Scene]:
    """Load and return the raw Scene (or None if file is not a scene).

    Useful when metric modules need access to per-geometry materials/UVs that
    get lost on concatenation (e.g. PBR channel detection).
    """
    obj = trimesh.load(str(path), force="scene")
    return obj if isinstance(obj, trimesh.Scene) else None


def has_uv(mesh: trimesh.Trimesh) -> bool:
    """True if mesh has UV texture coordinates attached to its visual."""
    visual = getattr(mesh, "visual", None)
    if visual is None:
        return False
    uv = getattr(visual, "uv", None)
    return uv is not None and len(uv) > 0


def has_texture_image(mesh: trimesh.Trimesh) -> bool:
    """True if mesh's material carries a real texture image (not just a color)."""
    visual = getattr(mesh, "visual", None)
    if visual is None or not hasattr(visual, "material"):
        return False
    mat = visual.material
    if mat is None:
        return False
    # Different material classes use different attribute names — check broadly.
    for attr in ("baseColorTexture", "image", "diffuse_texture", "_image"):
        if getattr(mat, attr, None) is not None:
            return True
    return False


def safe_str(p: str | Path) -> str:
    return str(p).replace("\\", "/")
