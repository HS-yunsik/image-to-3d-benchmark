"""
Figure 1: Qualitative comparison grid — simplified for paper submission.

Layout:
  - Header row: blank | ModelName\\nELO XXXX (x4)
  - 3 data rows:  input image+label | mesh render (x4, no cell labels)

Column order (left=production-fit):
  Input | InstantMesh(Q1) | Hi3DGen(Q3) | Strawberrry(Q2) | MeshFormer(Q4)

No divider lines, no CC labels, English text only.

Rendering:
  - Official 3darena thumbnail if available (InstantMesh, Hi3DGen)
  - Geometry render via matplotlib Poly3DCollection otherwise
    black bg, az=210, el=25, FLIP_TOP_BOTTOM (GLTF Y-up fix)

Outputs:
  outputs/fig1_qualitative_grid.png  (dpi=200)
  outputs/fig1_qualitative_grid.pdf
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import trimesh

ROOT       = Path(__file__).resolve().parent.parent
MESH_DIR   = ROOT / "meshes" / "outputs"
THUMB_DIR  = ROOT / "data" / "3darena_thumbs" / "outputs"
INPUT_DIR  = ROOT / "data" / "3darena_thumbs" / "inputs" / "images"
ISO_DIR    = ROOT / "data" / "iso3d_inputs" / "png@1024"
OUT_DIR    = ROOT / "outputs"
SUMMARY    = ROOT / "data" / "model_summary.csv"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

THUMB_PX = 400

_summary = pd.read_csv(SUMMARY).set_index("model")

def elo_val(name: str) -> str:
    if name in _summary.index:
        v = _summary.loc[name, "elo"]
        return str(int(v)) if pd.notna(v) else "—"
    return "—"

MODELS = [
    {"name": "InstantMesh"},
    {"name": "Hi3DGen"},
    {"name": "Strawberrry"},
    {"name": "MeshFormer"},
]
for m in MODELS:
    m["elo_str"] = elo_val(m["name"])

PROMPTS = [
    "A_cartoon_house_with_red_roof",
    "A_castle_made_of_cardboard",
    "A_heart_made_of_wood",
]

PROMPT_DISPLAY = {
    "A_cartoon_house_with_red_roof": "A cartoon house with red roof",
    "A_castle_made_of_cardboard":    "A castle made of cardboard",
    "A_heart_made_of_wood":          "A heart made of wood",
}

# Models that need a horizontal flip after geometry render
FLIP_LR_MODELS = {"MeshFormer"}


def render_geometry(glb_path: Path, size: int = THUMB_PX) -> Image.Image | None:
    """Black bg, az=210, el=25, FLIP_TOP_BOTTOM to fix GLTF Y-up inversion."""
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

        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
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
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_zlim(-lim, lim)
        ax2.set_axis_off()
        ax2.set_box_aspect([1, 1, 1])

        buf = io.BytesIO()
        fig2.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                     facecolor="black", pad_inches=0)
        plt.close(fig2)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)   # fix GLTF Y-up inversion
        return img
    except Exception as e:
        print(f"    [WARN] geometry render failed {glb_path.name}: {e}")
        return None


def get_thumbnail(model_name: str, prompt: str) -> np.ndarray:
    """Official thumbnail first; otherwise geometry render + optional LR flip."""
    official = THUMB_DIR / model_name / f"{prompt}.png"
    if official.exists():
        print(f"    {model_name}: official thumbnail")
        img = Image.open(official).convert("RGB").resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
        return np.array(img)
    glb = MESH_DIR / model_name / f"{prompt}.glb"
    if glb.exists():
        print(f"    {model_name}: geometry render ...")
        img = render_geometry(glb, size=THUMB_PX)
        if img is not None:
            if model_name in FLIP_LR_MODELS:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            return np.array(img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS))
    print(f"    {model_name}: placeholder")
    return np.full((THUMB_PX, THUMB_PX, 3), 40, dtype=np.uint8)


def make_grid():
    n_models  = len(MODELS)   # 4
    n_prompts = len(PROMPTS)  # 3
    n_cols    = 1 + n_models  # 5  (input + 4 model cols)

    HDR_H  = 0.70   # header row
    IMG_H  = 2.65   # image row
    CAP_H  = 0.40   # prompt label row (col 0 only)
    COL_W  = 2.65

    # GridSpec rows: 1 header + (img + cap) * 3 prompts
    height_ratios = [HDR_H] + [IMG_H, CAP_H] * n_prompts
    fig_h = sum(height_ratios)
    fig_w = n_cols * COL_W

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=200, facecolor="white")
    gs  = gridspec.GridSpec(
        nrows=1 + n_prompts * 2,
        ncols=n_cols,
        figure=fig,
        height_ratios=height_ratios,
        hspace=0.025, wspace=0.025,
        left=0.005, right=0.995, top=0.995, bottom=0.005,
    )

    # ── Header row ─────────────────────────────────────────────────────────────
    ax_h0 = fig.add_subplot(gs[0, 0])
    ax_h0.axis("off")
    ax_h0.text(0.5, 0.5, "Input\n(iso3d)", transform=ax_h0.transAxes,
               ha="center", va="center", fontsize=19, fontweight="bold", color="#111",
               linespacing=1.5)

    for j, m in enumerate(MODELS):
        ax_hj = fig.add_subplot(gs[0, j + 1])
        ax_hj.axis("off")
        ax_hj.text(0.5, 0.5, m['name'], transform=ax_hj.transAxes,
                   ha="center", va="center",
                   fontsize=19, fontweight="bold", color="#111")

    # ── Data rows ──────────────────────────────────────────────────────────────
    for i, prompt in enumerate(PROMPTS):
        img_row = 1 + i * 2
        cap_row = 1 + i * 2 + 1

        # Col 0: input image
        inp = INPUT_DIR / f"{prompt}.jpg"
        if not inp.exists():
            inp = ISO_DIR / f"{prompt}.png"
        iso_arr = np.array(Image.open(inp).convert("RGB")) if inp.exists() \
                  else np.full((THUMB_PX, THUMB_PX, 3), 200, dtype=np.uint8)

        ax_in = fig.add_subplot(gs[img_row, 0])
        ax_in.imshow(iso_arr)
        ax_in.set_xticks([]); ax_in.set_yticks([])
        for sp in ax_in.spines.values():
            sp.set_edgecolor("#999999"); sp.set_linewidth(0.8)

        ax_cap0 = fig.add_subplot(gs[cap_row, 0])
        ax_cap0.axis("off")
        ax_cap0.text(0.5, 0.65, PROMPT_DISPLAY[prompt],
                     transform=ax_cap0.transAxes,
                     ha="center", va="center",
                     fontsize=13, fontweight="bold", color="#111")

        # Cols 1-4: mesh renders only (no labels)
        for j, m in enumerate(MODELS):
            name = m["name"]
            print(f"  [{i+1}/{n_prompts}][{j+1}/{n_models}] {name} / {prompt}")
            thumb = get_thumbnail(name, prompt)

            ax_img = fig.add_subplot(gs[img_row, j + 1])
            ax_img.imshow(thumb)
            ax_img.set_xticks([]); ax_img.set_yticks([])
            for sp in ax_img.spines.values():
                sp.set_edgecolor("#555555"); sp.set_linewidth(0.7)

            # Caption row for model cols: empty (just consume the GridSpec slot)
            ax_cap = fig.add_subplot(gs[cap_row, j + 1])
            ax_cap.axis("off")

    # ── Save ───────────────────────────────────────────────────────────────────
    png_path = OUT_DIR / "fig1_qualitative_grid.png"

    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")

    sz_w = int(fig_w * 200)
    sz_h = int(fig_h * 200)
    print(f"Size:  {fig_w:.1f} x {fig_h:.1f} in  ({sz_w} x {sz_h} px at 200 dpi)")

    print("""
=== FIGURE CAPTION ===

Figure 1. Qualitative comparison of model outputs for three iso3d inputs
(rows) and four representative models (columns). Column headers report
ELO scores from the 3D Arena leaderboard. Detailed metric analysis
is presented in Section 5.
""")


if __name__ == "__main__":
    make_grid()
