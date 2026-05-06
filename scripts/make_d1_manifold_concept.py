"""
Figure: Manifold vs. Non-manifold Edge concept for D1 Geometry slide.

Mesh A: clean box  — all edges count=2 (manifold)
Mesh B: box + fin triangles on 8 outer edges
        → 8 junction edges (count=3) + 16 boundary edges (count=1) = 24 NM total

Orange lines: junction edges only (the "3 faces share 1 edge" cases)

Output: outputs/fig_d1_manifold_concept.png  (dpi=200)
"""
from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import trimesh

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

GREEN  = "#1D9E75"
RED    = "#E24B4A"
ORANGE = "#EF9F27"

AZ, EL = 220, 28   # camera: upper-left, slightly above horizon


# ── Helpers ────────────────────────────────────────────────────────────────────

def _light(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    d = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return d / np.linalg.norm(d)

LIGHT = _light(AZ, EL)


def _shade(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return RGBA per-face colors with diffuse shading."""
    n = mesh.face_normals
    d = np.clip(n @ LIGHT, 0, 1)
    rgb  = np.outer(0.18 + 0.74 * d, np.array([0.82, 0.82, 0.82]))
    return np.clip(np.hstack([rgb, np.ones((len(d), 1))]), 0, 1)


# ── Mesh generation ────────────────────────────────────────────────────────────

def make_mesh_a() -> trimesh.Trimesh:
    """Manifold: clean box. Every edge shared by exactly 2 triangles."""
    m = trimesh.creation.box(extents=[2, 2, 2])
    return m


def _outer_edges(verts: np.ndarray, faces: np.ndarray) -> list[tuple[int, int]]:
    """
    Edges of a box that are real cube edges (length ≈ 2.0),
    not the face diagonals (length ≈ 2√2 ≈ 2.83).
    """
    cnt = Counter()
    for f in faces:
        for k in range(3):
            e = tuple(sorted([f[k], f[(k+1) % 3]]))
            cnt[e] += 1
    result = []
    for (vi, vj) in cnt:
        if abs(np.linalg.norm(verts[vi] - verts[vj]) - 2.0) < 0.05:
            result.append((vi, vj))
    return result


def make_mesh_b() -> tuple[trimesh.Trimesh, list[tuple[int, int]]]:
    """
    Box + fins on 4 top edges + 4 vertical edges = 8 fins.
    Fin tip placed outward (away from box centre) and slightly
    perpendicular so it is visible from az=220, el=28.
    """
    box   = trimesh.creation.box(extents=[2, 2, 2])
    verts = list(np.array(box.vertices, dtype=float))
    faces = list(np.array(box.faces,    dtype=int))

    outer = _outer_edges(np.array(verts), np.array(faces))

    # Separate into top (z≈1), bottom (z≈-1), and vertical edges
    def edge_kind(vi, vj):
        z0, z1 = verts[vi][2], verts[vj][2]
        if abs(z0 - 1.0) < 0.1 and abs(z1 - 1.0) < 0.1:  return "top"
        if abs(z0 + 1.0) < 0.1 and abs(z1 + 1.0) < 0.1:  return "bot"
        return "vert"

    top_edges  = [(vi, vj) for vi, vj in outer if edge_kind(vi, vj) == "top"]
    vert_edges = [(vi, vj) for vi, vj in outer if edge_kind(vi, vj) == "vert"]

    chosen = top_edges[:4] + vert_edges[:4]   # 8 fins total

    junction_edges = []

    for (va, vb) in chosen:
        pa, pb = np.array(verts[va]), np.array(verts[vb])
        mid    = (pa + pb) / 2.0

        # Outward direction (away from box origin, projected to XY-ish)
        out = mid.copy()
        if abs(edge_kind(va, vb) == "vert"):
            out[2] = 0          # vertical fins extend only horizontally
        norm = np.linalg.norm(out)
        out  = out / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

        # Fin tip: out + slight upward bias for top fins
        z_bias = 0.35 if edge_kind(va, vb) == "top" else 0.0
        fin_tip = mid + out * 1.15 + np.array([0, 0, z_bias])

        new_idx = len(verts)
        verts.append(fin_tip.tolist())
        faces.append([va, vb, new_idx])
        junction_edges.append((va, vb))

    mesh = trimesh.Trimesh(
        vertices=np.array(verts, dtype=float),
        faces=np.array(faces,    dtype=int),
        process=False,
    )
    return mesh, junction_edges


def nm_summary(mesh: trimesh.Trimesh) -> dict:
    cnt = Counter(map(tuple, mesh.edges_sorted))
    junct  = sum(1 for c in cnt.values() if c == 3)
    bound  = sum(1 for c in cnt.values() if c == 1)
    return {"junction": junct, "boundary": bound, "total": junct + bound}


# ── Rendering ──────────────────────────────────────────────────────────────────

def render_mesh(mesh: trimesh.Trimesh,
                nm_edges: list[tuple[int, int]] | None = None,
                size: int = 512) -> np.ndarray:
    dpi = 100
    fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi, facecolor="black")
    ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")

    fc = _shade(mesh)
    poly = Poly3DCollection(mesh.vertices[mesh.faces], zsort="average", shade=False)
    poly.set_facecolor(fc)
    poly.set_edgecolor("#3A3A3A")
    poly.set_linewidth(0.6)
    ax.add_collection3d(poly)

    if nm_edges:
        v = mesh.vertices
        for (a, b) in nm_edges:
            p1, p2 = v[a] + LIGHT * 0.07, v[b] + LIGHT * 0.07
            ax.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                      color=ORANGE, linewidth=5.5, zorder=200,
                      solid_capstyle="round", solid_joinstyle="round")

    lim = 2.3
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=EL, azim=AZ)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="black", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB").resize((size, size), Image.LANCZOS))


# ── Figure composition ─────────────────────────────────────────────────────────

def make_figure():
    print("Building meshes ...")
    mesh_a          = make_mesh_a()
    mesh_b, jedges  = make_mesh_b()
    stats_a         = nm_summary(mesh_a)
    stats_b         = nm_summary(mesh_b)

    print(f"  Mesh A: NM edges = {stats_a['total']}")
    print(f"  Mesh B: junction={stats_b['junction']}, "
          f"boundary={stats_b['boundary']}, total NM={stats_b['total']}")

    print("Rendering ...")
    img_a = render_mesh(mesh_a, None,   size=512)
    img_b = render_mesh(mesh_b, jedges, size=512)

    # ── Canvas ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9.5, 6.0), dpi=200, facecolor="white")

    fig.text(0.5, 0.992, "Manifold vs. Non-manifold Edge",
             ha="center", va="top", fontsize=15, fontweight="bold", color="#111111")

    XC = [0.285, 0.715]
    for xc, hdr, hc in zip(
        XC,
        ["Mesh A  —  Manifold", "Mesh B  —  Non-manifold"],
        [GREEN, RED],
    ):
        fig.text(xc, 0.945, hdr, ha="center", va="top",
                 fontsize=12, fontweight="bold", color=hc)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.045, right=0.975, top=0.908, bottom=0.275,
        hspace=0.0, wspace=0.035,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_a); ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2.2)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_b); ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(RED); sp.set_linewidth(2.2)

    fig.add_artist(plt.Line2D(
        [0.508, 0.508], [0.275, 0.908],
        transform=fig.transFigure, color="#cccccc", linewidth=0.9,
    ))

    # ── Info labels ─────────────────────────────────────────────────────────
    rows_a = [
        ("Manifold: ✓",                    GREEN,     True,  False),
        (f"Non-manifold edges: {stats_a['total']}", "#333333", False, False),
        ("Mesh processing: OK",            GREEN,     False, True ),
    ]
    rows_b = [
        ("Manifold: ✗",                    RED,       True,  False),
        (f"Non-manifold edges: {stats_b['total']}", "#333333", False, False),
        ("Boolean ops fail",               RED,       False, True ),
    ]

    for xc, rows in zip(XC, [rows_a, rows_b]):
        y = 0.248
        for text, col, bold, italic in rows:
            fig.text(xc, y, text, ha="center", va="top", fontsize=11.5,
                     fontweight="bold" if bold else "normal",
                     style="italic"    if italic else "normal",
                     color=col)
            y -= 0.058

    # ── Legend ──────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    fig.legend(
        handles=[
            Line2D([0], [0], color=ORANGE, lw=3.5,
                   label=(f"non-manifold edge  "
                          f"(3 faces share 1 edge)  x{stats_b['junction']}")),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=1, fontsize=9.5, framealpha=0.9, edgecolor="#cccccc",
    )

    # ── Save ────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_d1_manifold_concept.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved: {out}")
    print(f"\n=== Summary ===")
    print(f"  Non-manifold generation: box + fin triangles on 4 top + 4 vertical outer edges")
    print(f"  Mesh B junction edges  : {stats_b['junction']}  (3 faces per edge)")
    print(f"  Mesh B boundary edges  : {stats_b['boundary']}  (1 face per edge, from fin free sides)")
    print(f"  Total non-manifold     : {stats_b['total']}")


if __name__ == "__main__":
    make_figure()
