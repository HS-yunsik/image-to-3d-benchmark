"""D4: PBR metrics.

These look at the material attached to a mesh's TextureVisuals. The 5
glTF PBR channels we recognize are:

    baseColorTexture          -- albedo / diffuse
    normalTexture             -- normal map
    metallicRoughnessTexture  -- packed metallic (B) + roughness (G)
    emissiveTexture           -- self-illumination
    occlusionTexture          -- baked AO

trimesh exposes these as attributes on PBRMaterial when loading glTF/GLB.
A material can exist without any image (e.g. only baseColorFactor as a flat
color, like InstantMesh); we explicitly check that the texture VALUE is
non-None to avoid Phase 0's bug where `hasattr(mat, 'baseColorTexture')`
returned True for an attribute set to None.

Metric definitions (Phase 1)
----------------------------
has_texture : bool
    True iff at least one of the 5 PBR channels has a non-None image.

pbr_channel_count : int in [0, 5]
    Count of distinct PBR channels with a non-None image.

texture_resolution : int >= 0
    max(width, height) of the largest texture image across all channels.
    0 if no textures are present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

PBR_CHANNELS = (
    "baseColorTexture",
    "normalTexture",
    "metallicRoughnessTexture",
    "emissiveTexture",
    "occlusionTexture",
)


def _texture_size(tex: Any) -> int | None:
    """Return max(width, height) of a texture image, or None."""
    if tex is None:
        return None
    # PIL.Image: .size is (w, h)
    size = getattr(tex, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return max(int(size[0]), int(size[1]))
    # Some wrappers expose .image
    img = getattr(tex, "image", None)
    if img is not None:
        size = getattr(img, "size", None)
        if isinstance(size, tuple) and len(size) == 2:
            return max(int(size[0]), int(size[1]))
    return None


def compute(mesh: trimesh.Trimesh, source_path: str | Path | None = None) -> dict[str, Any]:
    """Compute all D4 metrics. ``source_path`` is currently unused; reserved
    for a future pygltflib-based fallback if the trimesh material is ambiguous.
    """
    visual = getattr(mesh, "visual", None)
    mat = getattr(visual, "material", None) if visual is not None else None

    channels_found: set[str] = set()
    max_res = 0

    if mat is not None:
        for ch in PBR_CHANNELS:
            tex = getattr(mat, ch, None)
            if tex is not None:
                channels_found.add(ch)
                size = _texture_size(tex)
                if size:
                    max_res = max(max_res, size)
        # Some non-PBR materials only expose .image
        if not channels_found:
            img = getattr(mat, "image", None)
            if img is not None:
                channels_found.add("baseColorTexture")
                size = _texture_size(img)
                if size:
                    max_res = max(max_res, size)

    return {
        "has_texture": len(channels_found) > 0,
        "pbr_channel_count": len(channels_found),
        "texture_resolution": int(max_res),
    }
