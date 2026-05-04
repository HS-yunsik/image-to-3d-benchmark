"""Unit tests for metrics/{geometry,topology,uv,pbr}.py.

Uses synthetic meshes (trimesh primitives) so each metric's expected value
is known by construction. Run as a script:

    conda activate 3darena
    python tests/test_metrics.py

No pytest required; uses plain asserts. Each test prints PASS/FAIL and we
exit non-zero if any test fails.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh

from metrics import geometry, topology, uv, pbr, scale, compute_all, ALL_METRIC_KEYS

# ---------- helpers ---------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
_results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, msg: str = "") -> None:
    _results.append((name, cond, msg))
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}{(' -- ' + msg) if msg and not cond else ''}")


def isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return False


# ---------- D1 geometry tests ----------------------------------------------

def test_closed_cube() -> None:
    print("\n[test] closed unit cube (watertight, manifold)")
    m = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    g = geometry.compute(m)
    check("vertex_count == 8", g["vertex_count"] == 8)
    check("face_count == 12", g["face_count"] == 12)
    check("watertight == True", g["watertight"] is True)
    check("manifold_edge_ratio == 1.0", g["manifold_edge_ratio"] == 1.0)
    check("boundary_edge_ratio == 0.0", g["boundary_edge_ratio"] == 0.0)
    check("nonmanifold_edge_ratio == 0.0", g["nonmanifold_edge_ratio"] == 0.0)
    check("connected_components == 1", g["connected_components"] == 1)


def test_open_plane() -> None:
    print("\n[test] single open quad (4 verts, 2 tris) -- not watertight")
    verts = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    faces = np.array([[0,1,2], [0,2,3]])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    g = geometry.compute(m)
    check("vertex_count == 4", g["vertex_count"] == 4)
    check("face_count == 2", g["face_count"] == 2)
    check("watertight == False", g["watertight"] is False)
    # 5 unique edges: 4 boundary + 1 shared
    check("boundary_edge_ratio == 4/5",
          abs(g["boundary_edge_ratio"] - 0.8) < 1e-6,
          f"got {g['boundary_edge_ratio']}")
    check("manifold_edge_ratio == 1/5",
          abs(g["manifold_edge_ratio"] - 0.2) < 1e-6,
          f"got {g['manifold_edge_ratio']}")
    check("nonmanifold_edge_ratio == 0.0", g["nonmanifold_edge_ratio"] == 0.0)
    check("connected_components == 1", g["connected_components"] == 1)


def test_two_disjoint_cubes() -> None:
    print("\n[test] two disjoint cubes (components==2)")
    a = trimesh.creation.box(extents=(1, 1, 1))
    b = trimesh.creation.box(extents=(1, 1, 1)).apply_translation((5, 0, 0))
    m = trimesh.util.concatenate([a, b])
    g = geometry.compute(m)
    check("connected_components == 2", g["connected_components"] == 2)
    check("watertight == True (both pieces closed)", g["watertight"] is True)
    check("manifold_edge_ratio == 1.0", g["manifold_edge_ratio"] == 1.0)


def test_nonmanifold_edge() -> None:
    print("\n[test] explicit non-manifold edge (3 faces share one edge)")
    # 4 vertices, edge (0,1) shared by 3 faces (T-junction structure)
    verts = np.array([
        [0, 0, 0],   # 0
        [1, 0, 0],   # 1
        [0, 1, 0],   # 2
        [0, 0, 1],   # 3
        [0, -1, 0],  # 4
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 4],
    ])
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    g = geometry.compute(m)
    check("nonmanifold_edge_ratio > 0", g["nonmanifold_edge_ratio"] > 0,
          f"got {g['nonmanifold_edge_ratio']}")


def test_empty_mesh() -> None:
    print("\n[test] empty mesh -- all sentinel values")
    m = trimesh.Trimesh()
    g = geometry.compute(m)
    check("vertex_count == 0", g["vertex_count"] == 0)
    check("face_count == 0", g["face_count"] == 0)
    check("watertight == False", g["watertight"] is False)
    check("connected_components == 0", g["connected_components"] == 0)
    check("manifold_edge_ratio is NaN", isnan(g["manifold_edge_ratio"]))


# ---------- D2 topology tests ----------------------------------------------

def test_topology_triangle() -> None:
    print("\n[test] triangulated mesh -> 100% tri ratio")
    m = trimesh.creation.box()
    t = topology.compute(m)
    check("triangle_ratio == 1.0", t["triangle_ratio"] == 1.0)
    check("non_triangle_ratio == 0.0", t["non_triangle_ratio"] == 0.0)


def test_topology_empty() -> None:
    print("\n[test] empty mesh topology -> NaN")
    t = topology.compute(trimesh.Trimesh())
    check("triangle_ratio is NaN", isnan(t["triangle_ratio"]))


# ---------- D3 UV tests -----------------------------------------------------

def test_uv_full_unit_square() -> None:
    print("\n[test] uv covering full unit square -> bbox=1.0, packed=1.0")
    verts = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    faces = np.array([[0,1,2], [0,2,3]])
    uv_arr = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=float)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv_arr)
    u = uv.compute(m)
    check("has_uv == True", u["has_uv"] is True)
    check("uv_bbox_efficiency ~ 1.0", abs(u["uv_bbox_efficiency"] - 1.0) < 1e-3)
    # Two triangles each of area 0.5 in UV -> total 1.0
    check("uv_packed_area ~ 1.0", abs(u["uv_packed_area"] - 1.0) < 1e-3,
          f"got {u['uv_packed_area']}")


def test_uv_quarter_square() -> None:
    print("\n[test] uv covering quarter square -> bbox=0.25")
    verts = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    faces = np.array([[0,1,2], [0,2,3]])
    uv_arr = np.array([[0,0], [0.5,0], [0.5,0.5], [0,0.5]], dtype=float)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv_arr)
    u = uv.compute(m)
    check("uv_bbox_efficiency ~ 0.25", abs(u["uv_bbox_efficiency"] - 0.25) < 1e-3)
    check("uv_packed_area ~ 0.25", abs(u["uv_packed_area"] - 0.25) < 1e-3)


def test_uv_missing() -> None:
    print("\n[test] mesh without UV -> has_uv=False, others NaN")
    m = trimesh.creation.box()
    u = uv.compute(m)
    check("has_uv == False", u["has_uv"] is False)
    check("uv_bbox_efficiency NaN", isnan(u["uv_bbox_efficiency"]))


# ---------- D4 PBR tests ----------------------------------------------------

def test_pbr_no_material() -> None:
    print("\n[test] mesh without texture -> has_texture=False, channels=0")
    m = trimesh.creation.box()
    p = pbr.compute(m)
    check("has_texture == False", p["has_texture"] is False)
    check("pbr_channel_count == 0", p["pbr_channel_count"] == 0)
    check("texture_resolution == 0", p["texture_resolution"] == 0)


def test_pbr_with_basecolor() -> None:
    print("\n[test] mesh with baseColorTexture -> 1 channel, correct resolution")
    try:
        from PIL import Image
    except ImportError:
        check("PIL available", False, "skipped: PIL/Pillow missing")
        return
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img)
    verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
    faces = np.array([[0,1,2]])
    uv_arr = np.array([[0,0], [1,0], [0,1]], dtype=float)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv_arr, material=mat)
    p = pbr.compute(m)
    check("has_texture == True", p["has_texture"] is True)
    check("pbr_channel_count == 1", p["pbr_channel_count"] == 1)
    check("texture_resolution == 256", p["texture_resolution"] == 256)


def test_pbr_multi_channel() -> None:
    print("\n[test] mesh with two PBR channels -> count == 2")
    try:
        from PIL import Image
    except ImportError:
        check("PIL available", False, "skipped: PIL/Pillow missing")
        return
    base = Image.new("RGB", (512, 512), (200, 200, 200))
    mr = Image.new("RGB", (1024, 1024), (0, 200, 0))  # bigger to test max
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base, metallicRoughnessTexture=mr,
    )
    verts = np.array([[0,0,0], [1,0,0], [0,1,0]], dtype=float)
    faces = np.array([[0,1,2]])
    uv_arr = np.array([[0,0], [1,0], [0,1]], dtype=float)
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv_arr, material=mat)
    p = pbr.compute(m)
    check("pbr_channel_count == 2", p["pbr_channel_count"] == 2)
    check("texture_resolution == 1024 (max of channels)",
          p["texture_resolution"] == 1024)


# ---------- D5 scale tests --------------------------------------------------

def test_scale_unit_cube() -> None:
    print("\n[test] unit cube -> area=6, bbox_diag=sqrt(3)")
    m = trimesh.creation.box(extents=(1, 1, 1))
    s = scale.compute(m)
    check("surface_area ~ 6.0", abs(s["surface_area"] - 6.0) < 1e-3)
    check("bbox_diag ~ sqrt(3)", abs(s["bbox_diag"] - math.sqrt(3)) < 1e-3)


def test_scale_empty() -> None:
    print("\n[test] empty mesh scale -> 0/0")
    s = scale.compute(trimesh.Trimesh())
    check("surface_area == 0", s["surface_area"] == 0.0)
    check("bbox_diag == 0", s["bbox_diag"] == 0.0)


# ---------- compute_all tests ----------------------------------------------

def test_compute_all_missing_file() -> None:
    print("\n[test] compute_all with missing file -> load_success=False")
    row = compute_all("does/not/exist.glb", model="phantom")
    check("load_success == False", row["load_success"] is False)
    check("error contains file_missing", "file_missing" in row["error"])
    check("all metric keys present",
          all(k in row for k in ALL_METRIC_KEYS))
    check("all metrics None", all(row[k] is None for k in ALL_METRIC_KEYS))


def test_compute_all_schema() -> None:
    print("\n[test] compute_all schema completeness")
    # Use a synthetic mesh saved to a temp file
    import tempfile
    m = trimesh.creation.box()
    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
        m.export(f.name)
        path = f.name
    try:
        row = compute_all(path, model="synthetic")
        check("load_success", row["load_success"] is True)
        check("model == synthetic", row["model"] == "synthetic")
        for k in ALL_METRIC_KEYS:
            check(f"key {k} present", k in row)
        check("watertight True", row["watertight"] is True)
        check("face_count == 12", row["face_count"] == 12)
        check("surface_area > 0", row["surface_area"] > 0)
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- runner ----------------------------------------------------------

def main() -> int:
    print("=" * 70)
    print("metrics unit tests (synthetic meshes)")
    print("=" * 70)

    tests = [
        # geometry
        test_closed_cube,
        test_open_plane,
        test_two_disjoint_cubes,
        test_nonmanifold_edge,
        test_empty_mesh,
        # topology
        test_topology_triangle,
        test_topology_empty,
        # uv
        test_uv_full_unit_square,
        test_uv_quarter_square,
        test_uv_missing,
        # pbr
        test_pbr_no_material,
        test_pbr_with_basecolor,
        test_pbr_multi_channel,
        # scale
        test_scale_unit_cube,
        test_scale_empty,
        # compute_all
        test_compute_all_missing_file,
        test_compute_all_schema,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            check(t.__name__ + " (CRASH)", False, f"{type(e).__name__}: {e}")

    print("\n" + "=" * 70)
    n = len(_results)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_fail = n - n_pass
    print(f"  {n_pass} / {n} checks passed, {n_fail} failed")
    if n_fail:
        print("\nFAILURES:")
        for name, ok, msg in _results:
            if not ok:
                print(f"  {name}: {msg}")
    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
