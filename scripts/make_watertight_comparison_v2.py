"""
Figure: Watertight comparison v2 -- D1 Geometry slide.

2x2 layout:
  Row 0: Normal shaded render   (Hi3DGen thumbnail | Strawberrry geometry render)
  Row 1: Wireframe              (Hi3DGen: green edges | Strawberrry: gray + red boundary)

Wireframe details:
  Hi3DGen    -- all edges green (#1D9E75), 0 boundary edges
  Strawberrry -- interior edges gray, boundary edges red (#E24B4A), 2,788 boundary edges

Output: outputs/fig_watertight_comparison_v2.png  (dpi=200)
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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image
import trimesh

ROOT      = Path(__file__).resolve().parent.parent
MESH_DIR  = ROOT / "meshes" / "outputs"
THUMB_DIR = ROOT / "data" / "3darena_thumbs" / "outputs"
OUT_DIR   = ROOT / "outputs"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

PROMPT = "A_heart_made_of_wood"
GREEN  = "#1D9E75"
RED    = "#E24B4A"

MODELS = [
    {"name": "Hi3DGen",     "elo": 1207, "watertight": True,  "cc": 2,  "n_boundary": 0,    "label": "production-fit", "color": GREEN},
    {"name": "Strawberrry", "elo": 1382, "watertight": False, "cc": 39, "n_boundary": 2788, "label": "non-fit",         "color": RED},
]


# ── Shared rotation (az=210, el=25) — matches render_geometry ─────────────────
def _rotate(verts: np.ndarray) -> np.ndarray:
    az = np.radians(210)
    el = np.radians(25)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,           0,           1]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    return (Rx @ Rz @ verts.T).T


def _load_mesh(glb_path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(glb_path), force="scene")
    if isinstance(scene, trimesh.Scene):
        return trimesh.util.concatenate(list(scene.geometry.values()))
    return scene


def _normalize(verts: np.ndarray) -> np.ndarray:
    verts = verts - verts.mean(axis=0)
    s = np.abs(verts).max()
    return verts / s if s > 0 else verts


def _make_3d_ax(size: int, dpi: int = 100):
    px = size / dpi
    fig = plt.figure(figsize=(px, px), dpi=dpi, facecolor="black")
    ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")
    return fig, ax


def _finish_3d_ax(fig, ax, size: int, dpi: int = 100) -> Image.Image:
    lim = 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="black", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    return img.resize((size, size), Image.LANCZOS)


# ── Normal shaded render ───────────────────────────────────────────────────────
def render_normal(model_name: str, size: int = 420) -> np.ndarray:
    official = THUMB_DIR / model_name / f"{PROMPT}.png"
    if official.exists():
        print(f"  {model_name}: official thumbnail")
        return np.array(Image.open(official).convert("RGB").resize((size, size), Image.LANCZOS))

    glb = MESH_DIR / model_name / f"{PROMPT}.glb"
    print(f"  {model_name}: geometry render ...")
    try:
        mesh  = _load_mesh(glb)
        verts = _normalize(np.array(mesh.vertices, dtype=float))
        faces = np.array(mesh.faces)
        if len(faces) > 40_000:
            np.random.seed(42)
            faces = faces[np.random.choice(len(faces), 40_000, replace=False)]
        verts = _rotate(verts)

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
        light = np.array([0.5, 0.5, 1.0]); light /= np.linalg.norm(light)
        diffuse = np.clip(normals @ light, 0, 1)
        base = np.array([0.62, 0.71, 0.80])
        fc = np.clip(np.outer(0.3 + 0.7 * diffuse, base), 0, 1)
        order = np.argsort((v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3)
        tris = verts[faces[order]]; fc = fc[order]

        fig2, ax2 = _make_3d_ax(size)
        poly = Poly3DCollection(tris, zsort="min", shade=False)
        poly.set_facecolor(fc); poly.set_edgecolor("none")
        ax2.add_collection3d(poly)
        return np.array(_finish_3d_ax(fig2, ax2, size))
    except Exception as e:
        print(f"  [WARN] normal render failed: {e}")
        return np.full((size, size, 3), 40, dtype=np.uint8)


# ── Wireframe render ───────────────────────────────────────────────────────────
def render_wireframe(model_name: str, highlight_boundary: bool, size: int = 420) -> np.ndarray:
    glb = MESH_DIR / model_name / f"{PROMPT}.glb"
    print(f"  {model_name}: wireframe render ...")
    try:
        mesh  = _load_mesh(glb)
        verts = _normalize(np.array(mesh.vertices, dtype=float))
        faces = np.array(mesh.faces)
        verts = _rotate(verts)

        if highlight_boundary:
            # Strawberrry: classify all edges from the full mesh
            ec = Counter(map(tuple, mesh.edges_sorted))
            int_list = [[a, b] for (a, b), c in ec.items() if c == 2]
            bnd_list = [[a, b] for (a, b), c in ec.items() if c == 1]

            int_arr = np.array(int_list, dtype=int) if int_list else np.empty((0, 2), int)
            bnd_arr = np.array(bnd_list, dtype=int) if bnd_list else np.empty((0, 2), int)

            MAX_INT = 6_000
            if len(int_arr) > MAX_INT:
                idx = np.random.RandomState(42).choice(len(int_arr), MAX_INT, replace=False)
                int_arr = int_arr[idx]
            print(f"    interior={len(int_arr)} (disp), boundary={len(bnd_arr)}")
        else:
            # Hi3DGen: subsample faces → extract unique edges (all shown in green)
            if len(faces) > 40_000:
                np.random.seed(42)
                face_sub = faces[np.random.choice(len(faces), 40_000, replace=False)]
            else:
                face_sub = faces
            all_e = np.vstack([face_sub[:, [0, 1]], face_sub[:, [1, 2]], face_sub[:, [2, 0]]])
            all_e = np.sort(all_e, axis=1)
            all_e = np.unique(all_e, axis=0)
            MAX_E = 18_000
            if len(all_e) > MAX_E:
                idx = np.random.RandomState(42).choice(len(all_e), MAX_E, replace=False)
                int_arr = all_e[idx]
            else:
                int_arr = all_e
            bnd_arr = np.empty((0, 2), int)
            print(f"    display edges={len(int_arr)} (green, no boundary)")

        fig2, ax2 = _make_3d_ax(size)

        # Interior / all edges
        if len(int_arr) > 0:
            segs = verts[int_arr]                           # (N, 2, 3)
            col   = "#888888" if highlight_boundary else GREEN
            lw    = 0.35     if highlight_boundary else 0.55
            alpha = 0.55     if highlight_boundary else 0.80
            lc = Line3DCollection(segs, colors=col, linewidths=lw, alpha=alpha)
            ax2.add_collection3d(lc)

        # Boundary edges (red, thick) — drawn last so they appear on top
        if len(bnd_arr) > 0:
            segs_bnd = verts[bnd_arr]                       # (N, 2, 3)
            lc_bnd = Line3DCollection(segs_bnd, colors=RED, linewidths=2.0, alpha=1.0)
            ax2.add_collection3d(lc_bnd)

        return np.array(_finish_3d_ax(fig2, ax2, size))
    except Exception as e:
        print(f"  [WARN] wireframe render failed: {e}")
        return np.full((size, size, 3), 40, dtype=np.uint8)


# ── Main figure ────────────────────────────────────────────────────────────────
def make_figure():
    print("=== Normal renders ===")
    norm = [render_normal(m["name"]) for m in MODELS]

    print("\n=== Wireframe renders ===")
    wire = [
        render_wireframe(MODELS[0]["name"], highlight_boundary=False),
        render_wireframe(MODELS[1]["name"], highlight_boundary=True),
    ]

    # ── Canvas ─────────────────────────────────────────────────────────────────
    # 8.5 x 8.0 in, white bg
    # GridSpec: 2 image rows x 2 model cols
    # Row labels and info text placed via fig.text

    fig = plt.figure(figsize=(8.5, 8.2), dpi=200, facecolor="white")

    # Overall title
    fig.text(0.5, 0.990,
             "Same input — different structural quality  (prompt: A heart made of wood)",
             ha="center", va="top", fontsize=10, color="#555555", style="italic")

    # Column header x centres (matched to gridspec cols below)
    XC = [0.355, 0.775]   # left col center, right col center

    for m, xc in zip(MODELS, XC):
        fig.text(xc, 0.965, m["name"],
                 ha="center", va="top", fontsize=15, fontweight="bold", color=m["color"])

    # GridSpec for 2x2 image grid
    gs = gridspec.GridSpec(
        nrows=2, ncols=2,
        figure=fig,
        height_ratios=[1, 1],
        hspace=0.055, wspace=0.040,
        left=0.09, right=0.985, top=0.948, bottom=0.265,
    )

    # Row labels (rotated, left of grid)
    fig.text(0.042, 0.760, "Surface\nRender",
             ha="center", va="center", fontsize=10, color="#444444",
             rotation=90, linespacing=1.6)
    fig.text(0.042, 0.420, "Wireframe",
             ha="center", va="center", fontsize=10, color="#444444",
             rotation=90)

    # ── Row 0: normal renders ──────────────────────────────────────────────────
    for j, (m, img) in enumerate(zip(MODELS, norm)):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(m["color"]); sp.set_linewidth(1.8)

    # ── Row 1: wireframe renders ───────────────────────────────────────────────
    for j, (m, img) in enumerate(zip(MODELS, wire)):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(m["color"]); sp.set_linewidth(1.8)

    # Horizontal separator between rows (approximate midpoint)
    fig.add_artist(plt.Line2D([0.09, 0.985], [0.585, 0.585],
                               transform=fig.transFigure,
                               color="#cccccc", linewidth=0.8, zorder=10))

    # ── Wireframe annotation (just below wireframe row) ────────────────────────
    wt_labels = [
        "✓ closed  (0 boundary edges)",
        "✗ open boundary  (2,788 edges)",
    ]
    for xc, m, lbl in zip(XC, MODELS, wt_labels):
        fig.text(xc, 0.258,
                 lbl,
                 ha="center", va="top", fontsize=11, fontweight="bold",
                 color=m["color"])

    # ── Info row ───────────────────────────────────────────────────────────────
    for xc, m in zip(XC, MODELS):
        wt_sym = "✓" if m["watertight"] else "✗"
        wt_col = GREEN if m["watertight"] else RED
        y = 0.204
        fig.text(xc, y, f"ELO: {m['elo']}",
                 ha="center", va="top", fontsize=10.5, color="#333333")
        y -= 0.053
        fig.text(xc, y, f"Watertight: {wt_sym}",
                 ha="center", va="top", fontsize=10.5, fontweight="bold", color=wt_col)
        y -= 0.053
        fig.text(xc, y, m["label"],
                 ha="center", va="top", fontsize=9.5, style="italic", color=m["color"])

    # ── Legend (horizontal, bottom center, below info) ─────────────────────────
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#888888", lw=1.4, label="interior edge"),
        Line2D([0], [0], color=RED,       lw=2.2, label="boundary edge (open)"),
        Line2D([0], [0], color=GREEN,     lw=1.4, label="closed mesh edge"),
    ]
    fig.legend(handles=legend_elements,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.005),
               ncol=3,
               fontsize=8.5,
               framealpha=0.88,
               edgecolor="#cccccc",
               title="Wireframe legend",
               title_fontsize=8.5)

    # ── Save ───────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_watertight_comparison_v2.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")

    print("\n=== Actual values (from CSV) ===")
    for m in MODELS:
        sym = "OK" if m["watertight"] else "FAIL"
        print(f"  {m['name']:<13} watertight={str(m['watertight']):<5} ({sym})"
              f"  CC={m['cc']}  boundary_edges={m['n_boundary']}  ELO={m['elo']}")


if __name__ == "__main__":
    make_figure()
