"""
Figure: Watertight comparison — D1 Geometry slide.

Layout:
  Top:   caption + small input image (centered)
  Main:  Hi3DGen mesh | Strawberrry mesh  (black bg cells, side by side)
  Info:  model name / ELO / Watertight symbol / fit label

Prompt: A_heart_made_of_wood
  Hi3DGen    watertight=True  (CC=2)   ELO=1207  -> production-fit  (green)
  Strawberrry watertight=False (CC=39)  ELO=1382  -> non-fit         (red)

Output: outputs/fig_watertight_comparison.png  (dpi=200)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import trimesh

ROOT      = Path(__file__).resolve().parent.parent
MESH_DIR  = ROOT / "meshes" / "outputs"
THUMB_DIR = ROOT / "data" / "3darena_thumbs" / "outputs"
INPUT_DIR = ROOT / "data" / "3darena_thumbs" / "inputs" / "images"
OUT_DIR   = ROOT / "outputs"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

PROMPT  = "A_heart_made_of_wood"
GREEN   = "#1D9E75"
RED     = "#E24B4A"

MODELS = [
    {"name": "Hi3DGen",    "elo": 1207, "watertight": True,  "cc": 2,  "label": "production-fit", "color": GREEN},
    {"name": "Strawberrry","elo": 1382, "watertight": False, "cc": 39, "label": "non-fit",         "color": RED},
]


def render_geometry(glb_path: Path, size: int = 420) -> Image.Image | None:
    """Black bg, az=210, el=25, FLIP_TOP_BOTTOM (GLTF Y-up fix). Same as fig1."""
    try:
        scene  = trimesh.load(str(glb_path), force="scene")
        meshes = list(scene.geometry.values()) if isinstance(scene, trimesh.Scene) else [scene]
        if not meshes:
            return None
        combined = trimesh.util.concatenate(meshes)
        verts = np.array(combined.vertices, dtype=float)
        faces = np.array(combined.faces)

        if len(faces) > 40_000:
            np.random.seed(42)
            faces = faces[np.random.choice(len(faces), 40_000, replace=False)]

        verts -= verts.mean(axis=0)
        s = np.abs(verts).max()
        if s > 0:
            verts /= s

        az = np.radians(210)
        el = np.radians(25)
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                       [np.sin(az),  np.cos(az), 0],
                       [0,           0,           1]])
        Rx = np.array([[1, 0,           0          ],
                       [0, np.cos(el), -np.sin(el)],
                       [0, np.sin(el),  np.cos(el)]])
        verts = (Rx @ Rz @ verts.T).T

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-9)
        light   = np.array([0.5, 0.5, 1.0])
        light  /= np.linalg.norm(light)
        diffuse = np.clip(normals @ light, 0, 1)
        base    = np.array([0.62, 0.71, 0.80])
        fc      = np.clip(np.outer(0.3 + 0.7 * diffuse, base), 0, 1)

        order = np.argsort((v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3)
        tris  = verts[faces[order]]
        fc    = fc[order]

        dpi = 100
        px  = size / dpi
        fig2 = plt.figure(figsize=(px, px), dpi=dpi, facecolor="black")
        ax2  = fig2.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")
        poly = Poly3DCollection(tris, zsort="min", shade=False)
        poly.set_facecolor(fc)
        poly.set_edgecolor("none")
        ax2.add_collection3d(poly)
        lim = 1.05
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)
        ax2.set_axis_off()
        ax2.set_box_aspect([1, 1, 1])

        buf = io.BytesIO()
        fig2.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                     facecolor="black", pad_inches=0)
        plt.close(fig2)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    except Exception as e:
        print(f"  [WARN] geometry render failed: {e}")
        return None


def get_thumbnail(model_name: str, size: int = 420) -> np.ndarray:
    official = THUMB_DIR / model_name / f"{PROMPT}.png"
    if official.exists():
        print(f"  {model_name}: official thumbnail")
        img = Image.open(official).convert("RGB").resize((size, size), Image.LANCZOS)
        return np.array(img)
    glb = MESH_DIR / model_name / f"{PROMPT}.glb"
    if glb.exists():
        print(f"  {model_name}: geometry render ...")
        img = render_geometry(glb, size=size)
        if img is not None:
            return np.array(img.resize((size, size), Image.LANCZOS))
    print(f"  {model_name}: placeholder (dark gray)")
    return np.full((size, size, 3), 40, dtype=np.uint8)


def make_figure():
    inp_path = INPUT_DIR / f"{PROMPT}.jpg"
    inp_img  = np.array(Image.open(inp_path).convert("RGB")) if inp_path.exists() \
               else np.full((200, 200, 3), 200, dtype=np.uint8)

    thumbs = [get_thumbnail(m["name"]) for m in MODELS]

    # ── Figure canvas (8 x 6 in, white bg) ────────────────────────────────────
    fig = plt.figure(figsize=(8, 6.0), dpi=200, facecolor="white")

    # ── Caption ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.985,
             "Same input, different structural quality",
             ha="center", va="top",
             fontsize=11, color="#555555", style="italic",
             transform=fig.transFigure)

    # ── Input image (small, centered at top) ──────────────────────────────────
    # Normalized coords: [left, bottom, width, height]
    inp_w, inp_h = 0.17, 0.22
    ax_inp = fig.add_axes([0.5 - inp_w / 2, 0.730, inp_w, inp_h])
    ax_inp.imshow(inp_img)
    ax_inp.set_xticks([]); ax_inp.set_yticks([])
    ax_inp.set_title("Input", fontsize=8.5, color="#666666", pad=3)
    for sp in ax_inp.spines.values():
        sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.8)

    # ── Thin horizontal separator ──────────────────────────────────────────────
    sep_y = 0.715
    fig.add_artist(plt.Line2D([0.03, 0.97], [sep_y, sep_y],
                               transform=fig.transFigure,
                               color="#cccccc", linewidth=0.7, zorder=10))

    # ── Mesh render cells ──────────────────────────────────────────────────────
    cell_top = sep_y - 0.015
    cell_bot = 0.235           # bottom of mesh image area
    cell_h   = cell_top - cell_bot
    cell_w   = 0.43
    gap      = 0.14            # gap between cells (holds the vertical divider)
    left_x   = 0.03
    right_x  = left_x + cell_w + gap

    ax_l = fig.add_axes([left_x, cell_bot, cell_w, cell_h])
    ax_l.imshow(thumbs[0])
    ax_l.set_xticks([]); ax_l.set_yticks([])
    for sp in ax_l.spines.values():
        sp.set_edgecolor("#444444"); sp.set_linewidth(1.2)

    ax_r = fig.add_axes([right_x, cell_bot, cell_w, cell_h])
    ax_r.imshow(thumbs[1])
    ax_r.set_xticks([]); ax_r.set_yticks([])
    for sp in ax_r.spines.values():
        sp.set_edgecolor("#444444"); sp.set_linewidth(1.2)

    # Thin vertical divider between cells
    div_x = left_x + cell_w + gap / 2
    fig.add_artist(plt.Line2D([div_x, div_x], [cell_bot, cell_top],
                               transform=fig.transFigure,
                               color="#cccccc", linewidth=0.7, zorder=10))

    # ── Info labels below each cell ────────────────────────────────────────────
    x_centers = [left_x + cell_w / 2, right_x + cell_w / 2]

    for m, xc in zip(MODELS, x_centers):
        wt_sym   = "✓" if m["watertight"] else "✗"   # ✓ or ✗
        wt_color = GREEN if m["watertight"] else RED

        y = cell_bot - 0.03
        # Model name
        fig.text(xc, y, m["name"],
                 ha="center", va="top", fontsize=13, fontweight="bold",
                 color=m["color"], transform=fig.transFigure)
        y -= 0.055
        # ELO
        fig.text(xc, y, f"ELO: {m['elo']}",
                 ha="center", va="top", fontsize=10, color="#333333",
                 transform=fig.transFigure)
        y -= 0.05
        # Watertight
        fig.text(xc, y, f"Watertight: {wt_sym}",
                 ha="center", va="top", fontsize=10.5, fontweight="bold",
                 color=wt_color, transform=fig.transFigure)
        y -= 0.05
        # Fit label
        fig.text(xc, y, m["label"],
                 ha="center", va="top", fontsize=9, style="italic",
                 color=m["color"], transform=fig.transFigure)

    # ── Save ───────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_watertight_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")

    # Terminal output of actual values
    print("\n=== Watertight actual values (CSV) ===")
    for m in MODELS:
        sym = "✓" if m["watertight"] else "✗"
        print(f"  {m['name']:<13} watertight={str(m['watertight']):<5}  ({sym})"
              f"  CC={m['cc']}   ELO={m['elo']}")


if __name__ == "__main__":
    make_figure()
