"""Inspect what trimesh material objects actually expose, so we can
write accurate PBR detection logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import trimesh

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = [
    ("Hunyuan3D-2",   "Hunyuan3D-2/A_cartoon_house_with_red_roof.glb"),
    ("Hunyuan3D-2.1", "Hunyuan3D-2.1/A_cartoon_house_with_red_roof.glb"),
    ("TRELLIS",       "TRELLIS/A_cartoon_house_with_red_roof.glb"),
    ("InstantMesh",   "InstantMesh/A_cartoon_house_with_red_roof.glb"),
    ("Hi3DGen",       "Hi3DGen/A_cartoon_house_with_red_roof.glb"),
]


def inspect(name: str, path: Path) -> None:
    print(f"\n=== {name} ===")
    print(f"file: {path}")
    if not path.exists():
        print("  (missing)")
        return

    obj = trimesh.load(str(path), force="scene")
    if isinstance(obj, trimesh.Scene):
        print(f"  Scene with {len(obj.geometry)} geometry/-ies")
        for gname, geom in obj.geometry.items():
            print(f"  -- {gname} ({type(geom).__name__})")
            visual = getattr(geom, "visual", None)
            mat = getattr(visual, "material", None) if visual else None
            print(f"     visual: {type(visual).__name__ if visual else None}")
            print(f"     material: {type(mat).__name__ if mat else None}")
            if mat:
                # List all attributes that might be textures
                attrs = [a for a in dir(mat) if not a.startswith("_")]
                tex_attrs = []
                for a in attrs:
                    try:
                        v = getattr(mat, a)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    # Heuristic: looks like a PIL image or has size
                    if hasattr(v, "size") and not callable(v):
                        tex_attrs.append((a, type(v).__name__, getattr(v, "size", None)))
                    elif hasattr(v, "image"):
                        tex_attrs.append((a, type(v).__name__, "->image"))
                if tex_attrs:
                    print("     image-like attrs:")
                    for a, tn, s in tex_attrs:
                        print(f"       .{a}: {tn} size={s}")
                else:
                    print("     (no image-like attributes)")
                # Print primary glTF PBR attrs explicitly
                for ch in ("baseColorTexture", "normalTexture",
                           "metallicRoughnessTexture", "emissiveTexture",
                           "occlusionTexture"):
                    v = getattr(mat, ch, "<missing>")
                    if v is not None and v != "<missing>":
                        size = getattr(v, "size", None) or getattr(getattr(v, "image", None), "size", None)
                        print(f"     PBR.{ch}: {type(v).__name__} size={size}")
    else:
        print(f"  Trimesh: {type(obj).__name__}")


for name, rel in SAMPLES:
    inspect(name, ROOT / "meshes" / "outputs" / rel)
