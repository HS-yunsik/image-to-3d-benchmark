"""Precision validation of our 11 Phase 1 metrics on one well-known mesh.

Target: meshes/outputs/Hunyuan3D-2/A_cartoon_house_with_red_roof.glb
Cross-check our values against trimesh built-ins, pygltflib (raw glTF),
and pymeshlab. Differences are surfaced explicitly so we can decide whose
answer is correct.

Output: logs/metric_validation_single.txt
"""
from __future__ import annotations

import io
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from metrics import geometry, topology, uv, pbr
from utils import load_mesh

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "meshes" / "outputs" / "Hunyuan3D-2" / "A_cartoon_house_with_red_roof.glb"
LOG = ROOT / "logs" / "metric_validation_single.txt"
LOG.parent.mkdir(exist_ok=True)


class Tee:
    """Print to console AND collect for file writing."""
    def __init__(self) -> None:
        self.buf = io.StringIO()

    def __call__(self, *args) -> None:
        line = " ".join(str(a) for a in args)
        print(line, flush=True)
        self.buf.write(line + "\n")

    def save(self, path: Path) -> None:
        path.write_text(self.buf.getvalue(), encoding="utf-8")


def hr(t: Tee, title: str) -> None:
    t("")
    t("=" * 78)
    t(title)
    t("=" * 78)


def check_ours(t: Tee, mesh: trimesh.Trimesh) -> dict:
    hr(t, "OUR METRICS (metrics.{geometry,topology,uv,pbr})")
    g = geometry.compute(mesh)
    tp = topology.compute(mesh)
    u = uv.compute(mesh)
    p = pbr.compute(mesh)
    out = {**g, **tp, **u, **p}
    for k, v in out.items():
        t(f"  {k:30s} = {v}")
    return out


def cross_check_trimesh(t: Tee, mesh: trimesh.Trimesh) -> dict:
    hr(t, "TRIMESH BUILT-INS (alt views of the same metrics)")
    out = {}

    # Counts
    out["vertex_count_alt"] = len(mesh.vertices)
    out["face_count_alt"] = len(mesh.faces)
    t(f"  vertex_count (len(.vertices))     = {out['vertex_count_alt']}")
    t(f"  face_count   (len(.faces))        = {out['face_count_alt']}")

    # Watertight via trimesh
    out["watertight_alt"] = bool(mesh.is_watertight)
    t(f"  is_watertight                     = {out['watertight_alt']}")
    out["winding_consistent"] = bool(mesh.is_winding_consistent)
    t(f"  is_winding_consistent             = {out['winding_consistent']}")

    # Edge analysis -- recompute manifold ratio independently
    edge_counts = Counter(map(tuple, mesh.edges_sorted))
    n_unique = len(edge_counts)
    n_manifold = sum(1 for v in edge_counts.values() if v == 2)
    n_boundary = sum(1 for v in edge_counts.values() if v == 1)
    n_nonman = sum(1 for v in edge_counts.values() if v >= 3)
    out["n_unique_edges"] = n_unique
    out["manifold_edge_ratio_alt"] = round(n_manifold / n_unique, 4) if n_unique else float("nan")
    out["boundary_edge_ratio_alt"] = round(n_boundary / n_unique, 4) if n_unique else float("nan")
    out["nonmanifold_edge_ratio_alt"] = round(n_nonman / n_unique, 4) if n_unique else float("nan")
    t(f"  unique edges                      = {n_unique}")
    t(f"  edges shared by 1 face (boundary) = {n_boundary} ({out['boundary_edge_ratio_alt']:.4f})")
    t(f"  edges shared by 2 faces (manifold)= {n_manifold} ({out['manifold_edge_ratio_alt']:.4f})")
    t(f"  edges shared by >=3 (non-manifold)= {n_nonman} ({out['nonmanifold_edge_ratio_alt']:.4f})")
    sum_check = (out["manifold_edge_ratio_alt"] + out["boundary_edge_ratio_alt"]
                 + out["nonmanifold_edge_ratio_alt"])
    t(f"  sum of three ratios               = {sum_check:.4f}  (expected 1.0)")

    # Connected components -- multiple methods
    parts_all = mesh.split(only_watertight=False)
    parts_wt = mesh.split(only_watertight=True)
    out["components_all"] = len(parts_all) if parts_all is not None else 1
    out["components_watertight_only"] = len(parts_wt) if parts_wt is not None else 0
    t(f"  components (all pieces)           = {out['components_all']}")
    t(f"  components (watertight only)      = {out['components_watertight_only']}")

    # UV cross-check
    visual = mesh.visual
    if hasattr(visual, "uv") and visual.uv is not None:
        uv_arr = np.asarray(visual.uv, dtype=float)
        out["uv_count"] = len(uv_arr)
        out["uv_min"] = list(map(lambda x: round(float(x), 4), uv_arr.min(axis=0)))
        out["uv_max"] = list(map(lambda x: round(float(x), 4), uv_arr.max(axis=0)))
        t(f"  uv array length                   = {len(uv_arr)}")
        t(f"  uv min (u,v)                      = {out['uv_min']}")
        t(f"  uv max (u,v)                      = {out['uv_max']}")
    else:
        out["uv_count"] = 0
        t(f"  no UV array on visual")

    return out


def cross_check_pygltflib(t: Tee, path: Path) -> dict:
    hr(t, "PYGLTFLIB (raw glTF inspection)")
    out: dict = {}
    try:
        import pygltflib  # type: ignore
    except ImportError as e:
        t(f"  pygltflib not available: {e}")
        return out

    try:
        gltf = pygltflib.GLTF2().load(str(path))
    except Exception as e:
        t(f"  Load error: {type(e).__name__}: {e}")
        return out

    out["n_meshes"] = len(gltf.meshes or [])
    out["n_materials"] = len(gltf.materials or [])
    out["n_textures"] = len(gltf.textures or [])
    out["n_images"] = len(gltf.images or [])
    out["n_accessors"] = len(gltf.accessors or [])
    t(f"  meshes={out['n_meshes']} materials={out['n_materials']} "
      f"textures={out['n_textures']} images={out['n_images']}")

    # Look at each material's PBR channels
    pbr_channels: list[str] = []
    for i, mat in enumerate(gltf.materials or []):
        t(f"  material[{i}] name={mat.name}")
        pbr_obj = getattr(mat, "pbrMetallicRoughness", None)
        if pbr_obj is not None:
            for ch in ("baseColorTexture", "metallicRoughnessTexture"):
                tex_ref = getattr(pbr_obj, ch, None)
                if tex_ref is not None and getattr(tex_ref, "index", None) is not None:
                    pbr_channels.append(ch)
                    t(f"    .{ch} -> texture[{tex_ref.index}]")
        for ch in ("normalTexture", "emissiveTexture", "occlusionTexture"):
            tex_ref = getattr(mat, ch, None)
            if tex_ref is not None and getattr(tex_ref, "index", None) is not None:
                pbr_channels.append(ch)
                t(f"    .{ch} -> texture[{tex_ref.index}]")

    out["pbr_channels"] = sorted(set(pbr_channels))
    out["pbr_channel_count_alt"] = len(set(pbr_channels))
    t(f"  PBR channels found (alt)          = {out['pbr_channels']}")

    # UV presence: any mesh primitive has TEXCOORD_0?
    has_uv_alt = False
    for m in (gltf.meshes or []):
        for prim in (m.primitives or []):
            attrs = getattr(prim, "attributes", None)
            if attrs and getattr(attrs, "TEXCOORD_0", None) is not None:
                has_uv_alt = True
                break
        if has_uv_alt:
            break
    out["has_uv_alt"] = has_uv_alt
    t(f"  has TEXCOORD_0                    = {has_uv_alt}")

    # Image dimensions: try to read first image's bytes and ask PIL
    try:
        from PIL import Image  # type: ignore
        sizes = []
        for img in (gltf.images or []):
            # Image data can be in bufferView (binary GLB) or uri (external)
            if getattr(img, "bufferView", None) is not None:
                # Use trimesh / glb parser to get bytes
                buf = gltf.binary_blob() if hasattr(gltf, "binary_blob") else None
                if buf is None:
                    continue
                bv = gltf.bufferViews[img.bufferView]
                data = buf[bv.byteOffset:bv.byteOffset + bv.byteLength]
                im = Image.open(io.BytesIO(data))
                sizes.append((img.mimeType, im.size))
        out["image_sizes"] = [(m, list(s)) for m, s in sizes]
        out["max_image_dim"] = max((max(s[1]) for s in sizes), default=0)
        t(f"  image sizes                       = {out['image_sizes']}")
        t(f"  max image dimension               = {out['max_image_dim']}")
    except Exception as e:
        t(f"  PIL inspection failed: {type(e).__name__}: {e}")

    return out


def cross_check_pymeshlab(t: Tee, path: Path) -> dict:
    hr(t, "PYMESHLAB (independent computation)")
    out: dict = {}
    try:
        import pymeshlab  # type: ignore
    except ImportError as e:
        t(f"  pymeshlab not available: {e}")
        return out

    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(path))
        m = ms.current_mesh()

        out["vertex_count_pml"] = m.vertex_number()
        out["face_count_pml"] = m.face_number()
        t(f"  vertex_number()                   = {out['vertex_count_pml']}")
        t(f"  face_number()                     = {out['face_count_pml']}")

        # Topological measures filter
        try:
            tp = ms.get_topological_measures()
            for k, v in tp.items():
                out[f"pml_{k}"] = v
                t(f"  topo.{k:24s} = {v}")
        except Exception as e:
            t(f"  get_topological_measures failed: {type(e).__name__}: {e}")

        # Geometric measures
        try:
            gm = ms.get_geometric_measures()
            interesting = ("surface_area", "mesh_volume",
                           "shell_barycenter", "bbox_diag")
            for k in interesting:
                if k in gm:
                    out[f"pml_{k}"] = gm[k]
                    t(f"  geo.{k:24s} = {gm[k]}")
        except Exception as e:
            t(f"  get_geometric_measures failed: {type(e).__name__}: {e}")

    except Exception as e:
        t(f"  pymeshlab load/measure failed: {type(e).__name__}: {e}")

    return out


def reconcile(t: Tee, ours: dict, tm: dict, gltf: dict, pml: dict) -> None:
    hr(t, "RECONCILIATION")

    def cmp(label: str, a, b, tol: float = 1e-4) -> str:
        if a is None or b is None or (isinstance(a, float) and a != a):
            return f"  {label:35s} ours={a} alt={b}"
        try:
            ok = abs(float(a) - float(b)) < tol if isinstance(a, (int, float)) else (a == b)
        except Exception:
            ok = a == b
        flag = "OK" if ok else "DIFF"
        return f"  [{flag:4}] {label:35s} ours={a} alt={b}"

    t(cmp("vertex_count", ours.get("vertex_count"), tm.get("vertex_count_alt")))
    t(cmp("face_count", ours.get("face_count"), tm.get("face_count_alt")))
    t(cmp("watertight", ours.get("watertight"), tm.get("watertight_alt")))
    t(cmp("manifold_edge_ratio", ours.get("manifold_edge_ratio"),
          tm.get("manifold_edge_ratio_alt")))
    t(cmp("boundary_edge_ratio", ours.get("boundary_edge_ratio"),
          tm.get("boundary_edge_ratio_alt")))
    t(cmp("nonmanifold_edge_ratio", ours.get("nonmanifold_edge_ratio"),
          tm.get("nonmanifold_edge_ratio_alt")))
    t(cmp("connected_components", ours.get("connected_components"),
          tm.get("components_all")))

    if pml:
        t(cmp("vertex_count vs pymeshlab", ours.get("vertex_count"),
              pml.get("vertex_count_pml")))
        t(cmp("face_count vs pymeshlab", ours.get("face_count"),
              pml.get("face_count_pml")))
        # pymeshlab usually exposes connected_components_number or similar
        for k in pml:
            if "connected" in k.lower() or "component" in k.lower():
                t(cmp(f"components vs {k}", ours.get("connected_components"),
                      pml[k]))
            if "boundary_edges" in k.lower() or "non_manif" in k.lower():
                t(f"  pymeshlab {k} = {pml[k]}")

    if gltf:
        t(cmp("has_uv", ours.get("has_uv"), gltf.get("has_uv_alt")))
        t(cmp("pbr_channel_count", ours.get("pbr_channel_count"),
              gltf.get("pbr_channel_count_alt")))
        max_dim = gltf.get("max_image_dim", 0) or 0
        t(cmp("texture_resolution", ours.get("texture_resolution"), max_dim))


def main() -> int:
    t = Tee()
    if not TARGET.exists():
        t(f"ERROR: target file missing: {TARGET}")
        t.save(LOG)
        return 1

    t(f"# Single-mesh metric validation")
    t(f"# Generated: {datetime.now().isoformat(timespec='seconds')}")
    t(f"# Target:    {TARGET}")
    t(f"# Size (KB): {TARGET.stat().st_size / 1024:.2f}")

    mesh = load_mesh(TARGET)
    ours = check_ours(t, mesh)
    tm = cross_check_trimesh(t, mesh)
    gltf = cross_check_pygltflib(t, TARGET)
    pml = cross_check_pymeshlab(t, TARGET)
    reconcile(t, ours, tm, gltf, pml)

    t.save(LOG)
    print(f"\nLog: {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
