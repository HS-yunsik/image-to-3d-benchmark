"""
Figure: Connected Components (CC) concept — Real mesh data.

LEFT:  Hi3DGen   "cartoon house"  CC = 18      → official 3darena thumbnail
RIGHT: MeshFormer "cartoon house" CC = 99,183  → mesh rendered all-red (triangle soup)

Output: outputs/fig_d1_cc_concept.png  (dpi=200)
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
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as sp_cc

ROOT      = Path(__file__).resolve().parent.parent
MESH_DIR  = ROOT / "meshes" / "outputs"
THUMB_DIR = ROOT / "data"  / "3darena_thumbs" / "outputs"
OUT_DIR   = ROOT / "outputs"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

GREEN = "#1D9E75"
RED   = "#E24B4A"

PROMPT   = "A_cartoon_house_with_red_roof"
PROMPT_G = PROMPT + ".glb"
PROMPT_P = PROMPT + ".png"

# Camera: match Hi3DGen thumbnail view (approx. 40° azimuth, 35° elevation)
AZ, EL = 40, 35


# ── Mesh + CC loading ──────────────────────────────────────────────────────────

def load_and_count(model: str) -> tuple[trimesh.Trimesh, int]:
    path = MESH_DIR / model / PROMPT_G
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    n = len(m.vertices)
    edges = m.edges_unique
    data  = np.ones(len(edges)*2, dtype=np.int8)
    rows  = np.concatenate([edges[:,0], edges[:,1]])
    cols  = np.concatenate([edges[:,1], edges[:,0]])
    adj   = csr_matrix((data,(rows,cols)), shape=(n,n))
    n_cc, _ = sp_cc(adj, directed=False)
    return m, n_cc


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _light(az_deg: float, el_deg: float) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    d = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return d / np.linalg.norm(d)

LIGHT = _light(AZ, EL)
FILL  = np.array([0.0, 0.0, 1.0])   # top fill


def _diffuse(normals: np.ndarray) -> np.ndarray:
    d_key  = np.clip(normals @ LIGHT, 0, 1)
    d_fill = np.clip(normals @ FILL,  0, 1)
    return 0.30 + 0.45 * d_key + 0.25 * d_fill


def render_red_mesh(mesh: trimesh.Trimesh, size: int = 512,
                    white_bg: bool = True) -> np.ndarray:
    """Render mesh in red shading (all faces = fragment color)."""
    dpi = 100
    bg  = "white" if white_bg else "black"
    fig = plt.figure(figsize=(size/dpi, size/dpi), dpi=dpi, facecolor=bg)
    ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor=bg)

    normals = mesh.face_normals
    d       = _diffuse(normals)
    fc_rgb  = np.outer(d, np.array([0.886, 0.294, 0.290]))   # #E24B4A
    fc_rgba = np.hstack([np.clip(fc_rgb, 0, 1),
                          np.ones((len(fc_rgb), 1))])

    poly = Poly3DCollection(mesh.vertices[mesh.faces], zsort="average", shade=False)
    poly.set_facecolor(fc_rgba)
    poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    v   = mesh.vertices
    cx, cy, cz = v.mean(axis=0)
    r   = np.abs(v - v.mean(axis=0)).max() * 1.05
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(cz-r, cz+r)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=EL, azim=AZ)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=bg, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = (Image.open(buf).convert("RGB")
               .transpose(Image.FLIP_TOP_BOTTOM)   # GLTF Y-up fix
               .resize((size, size), Image.LANCZOS))
    return np.array(img)


def load_thumbnail(model: str) -> np.ndarray:
    """Load official 3darena thumbnail as numpy RGB array."""
    path = THUMB_DIR / model / PROMPT_P
    img  = Image.open(str(path)).convert("RGB")
    # Convert white bg to match figure bg (keep white)
    return np.array(img.resize((512, 512), Image.LANCZOS))


# ── Figure composition ─────────────────────────────────────────────────────────

def make_figure() -> None:
    print("Loading meshes ...")
    _, n_cc_a = load_and_count("Hi3DGen")
    mesh_b, n_cc_b = load_and_count("MeshFormer")

    frag_b = n_cc_b - 1
    print(f"  Hi3DGen   CC = {n_cc_a:,}")
    print(f"  MeshFormer CC = {n_cc_b:,}  (main + {frag_b:,} fragments)")

    print("Loading Hi3DGen thumbnail ...")
    img_a = load_thumbnail("Hi3DGen")

    print("Rendering MeshFormer mesh (all red) ...")
    img_b = render_red_mesh(mesh_b, size=512, white_bg=True)

    # ── Canvas ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9.5, 6.2), dpi=200, facecolor="white")

    fig.text(0.5, 0.992,
             "Connected Components (CC)  —  Real mesh data",
             ha="center", va="top", fontsize=14, fontweight="bold", color="#111111")

    fig.text(0.5, 0.957,
             f'Same input: "{PROMPT.replace("_", " ")}"',
             ha="center", va="top", fontsize=9.5, color="#555555", style="italic")

    XC = [0.285, 0.715]
    for xc, hdr, hc in zip(
        XC,
        [f"Hi3DGen  —  CC = {n_cc_a:,}", f"MeshFormer  —  CC = {n_cc_b:,}"],
        [GREEN, RED],
    ):
        fig.text(xc, 0.927, hdr, ha="center", va="top",
                 fontsize=11.5, fontweight="bold", color=hc)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.045, right=0.975, top=0.894, bottom=0.275,
        hspace=0.0, wspace=0.035,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_a); ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2.5)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_b); ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(RED); sp.set_linewidth(2.5)

    fig.add_artist(plt.Line2D(
        [0.508, 0.508], [0.275, 0.894],
        transform=fig.transFigure, color="#cccccc", linewidth=0.9,
    ))

    rows_a = [
        (f"CC = {n_cc_a}",              GREEN,     True,  False),
        ("Mesh largely connected",       "#333333", False, False),
        ("Processable in DCC tools ✓",  GREEN,     False, True ),
    ]
    rows_b = [
        (f"CC = {n_cc_b:,}",            RED,       True,  False),
        ("Triangle soup — no topology", "#333333", False, False),
        (f"{frag_b:,} isolated triangles ✗", RED,  False, True ),
    ]

    for xc, rows in zip(XC, [rows_a, rows_b]):
        y = 0.248
        for text, col, bold, italic in rows:
            fig.text(xc, y, text, ha="center", va="top", fontsize=11.0,
                     fontweight="bold" if bold   else "normal",
                     style="italic"    if italic else "normal",
                     color=col)
            y -= 0.060

    fig.text(0.5, 0.052,
             "Real example: MeshFormer avg. CC = 97,891  (vs. ideal CC = 1)",
             ha="center", va="bottom",
             fontsize=10.5, fontweight="bold", color=RED)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor="#CCCCCC", edgecolor="#777777", lw=1.0,
                  label="Hi3DGen: clean official render"),
            Patch(facecolor=RED, edgecolor="#aa2222", lw=1.0,
                  label="MeshFormer: all faces = isolated CC (red = broken)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.062),
        ncol=2, fontsize=8.5, framealpha=0.90, edgecolor="#cccccc",
    )

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_d1_cc_concept.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out}")
    print(f"\n=== Summary ===")
    print(f"  LEFT  — Hi3DGen   : CC = {n_cc_a:,}  (official 3darena thumbnail)")
    print(f"  RIGHT — MeshFormer: CC = {n_cc_b:,}  (mesh rendered all-red)")
    print(f"  Isolated triangles (MeshFormer): {frag_b:,}")


if __name__ == "__main__":
    make_figure()
