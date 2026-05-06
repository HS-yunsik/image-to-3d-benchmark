"""
Figure: Watertight concept -- D1 Geometry slide.

Creates synthetic icosphere meshes (no real dataset).
  Mesh A: complete icosphere (watertight)
  Mesh B: icosphere with BFS-connected hole facing the camera + red boundary edges

Camera direction in world space: [-0.141, 0.622, 0.770]
  Derived from render rotation (az=210, el=25) + matplotlib default view (elev=30, azim=-60).
  The hole is placed at faces most aligned with this direction.

Output: outputs/fig_d1_watertight_concept.png  (dpi=200)
"""
from __future__ import annotations

import io
from collections import Counter, deque
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from PIL import Image
import trimesh

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"]       = 42

GREEN = "#1D9E75"
RED   = "#E24B4A"

# Camera direction in pre-rotation world space.
# inverse of (Rx(25) @ Rz(210)) applied to matplotlib cam [0.433, -0.75, 0.5]
CAM_DIR = np.array([-0.141, 0.622, 0.770])
CAM_DIR /= np.linalg.norm(CAM_DIR)


# ── Mesh creation ──────────────────────────────────────────────────────────────

def make_watertight_sphere(subdivisions: int = 3) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions)


def make_sphere_with_hole(subdivisions: int = 3,
                          n_remove: int = 40) -> tuple[trimesh.Trimesh, int]:
    """BFS-connected hole on the face most aligned with CAM_DIR."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)

    centroids = mesh.triangles_center          # (F, 3)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
    seed = int(np.argmax(c_norm @ CAM_DIR))

    # Build face adjacency
    adj: dict[int, list[int]] = {i: [] for i in range(len(mesh.faces))}
    for a, b in mesh.face_adjacency:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    removed: set[int] = set()
    q: deque[int] = deque([seed])
    while len(removed) < n_remove and q:
        curr = q.popleft()
        if curr in removed:
            continue
        removed.add(curr)
        for nb in adj[curr]:
            if nb not in removed:
                q.append(nb)

    mask = np.ones(len(mesh.faces), dtype=bool)
    mask[list(removed)] = False
    new_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces[mask].copy(),
        process=False,
    )
    return new_mesh, len(removed)


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _rotate(verts: np.ndarray) -> np.ndarray:
    """az=210, el=25 — same convention as existing renders."""
    az = np.radians(210)
    el = np.radians(25)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0,           0,           1]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    return (Rx @ Rz @ verts.T).T


def _normalize(verts: np.ndarray) -> np.ndarray:
    verts = verts - verts.mean(axis=0)
    s = np.abs(verts).max()
    return verts / s if s > 0 else verts


def _shade(verts_rot: np.ndarray, faces: np.ndarray,
           base: np.ndarray,
           cull_backface: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (face_colors, z-sorted faces).

    cull_backface: discard faces whose normal points away from the camera.
      Camera in post-rotation space ≈ [0.433, -0.75, 0.5] (az=210, el=25 +
      matplotlib default view elev=30, azim=-60).
      This makes the hole in Mesh B appear as pure black background.
    """
    _CAM_POSTROT = np.array([0.433, -0.75, 0.5])

    v0, v1, v2 = verts_rot[faces[:, 0]], verts_rot[faces[:, 1]], verts_rot[faces[:, 2]]
    n_raw = np.cross(v1 - v0, v2 - v0)

    if cull_backface:
        front = (n_raw @ _CAM_POSTROT) > 0
        faces = faces[front]
        v0, v1, v2 = verts_rot[faces[:, 0]], verts_rot[faces[:, 1]], verts_rot[faces[:, 2]]
        n_raw = np.cross(v1 - v0, v2 - v0)

    n = n_raw / (np.linalg.norm(n_raw, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.5, 0.5, 1.0]); light /= np.linalg.norm(light)
    d = np.clip(n @ light, 0, 1)
    fc = np.clip(np.outer(0.12 + 0.88 * d, base), 0, 1)
    order = np.argsort((v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3)
    return fc[order], faces[order]


def _fig_to_arr(fig, size: int, dpi: int = 100) -> np.ndarray:
    ax = fig.axes[0]
    lim = 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_axis_off(); ax.set_box_aspect([1, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor="black", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
    return np.array(img.resize((size, size), Image.LANCZOS))


# ── Per-mesh renders ───────────────────────────────────────────────────────────

def render_watertight(mesh: trimesh.Trimesh, size: int = 512) -> np.ndarray:
    v = _rotate(_normalize(np.array(mesh.vertices, dtype=float)))
    f = np.array(mesh.faces)
    base = np.array([0.85, 0.85, 0.85])
    fc, fs = _shade(v, f, base)

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="black")
    ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")
    poly = Poly3DCollection(v[fs], zsort="min", shade=False)
    poly.set_facecolor(fc); poly.set_edgecolor("none")
    ax.add_collection3d(poly)
    return _fig_to_arr(fig, size)


def render_open_mesh(mesh: trimesh.Trimesh,
                     size: int = 512) -> tuple[np.ndarray, int]:
    """Shaded open mesh + thick red boundary edges. Returns (image, n_boundary)."""
    verts_orig = np.array(mesh.vertices, dtype=float)
    faces      = np.array(mesh.faces)

    # Boundary edges from full mesh
    ec  = Counter(map(tuple, mesh.edges_sorted))
    bnd = np.array([[a, b] for (a, b), c in ec.items() if c == 1], dtype=int)
    n_boundary = len(bnd)

    v = _rotate(_normalize(verts_orig.copy()))
    base = np.array([0.85, 0.85, 0.85])
    fc, fs = _shade(v, faces, base)

    dpi = 100
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi, facecolor="black")
    ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="black")

    # Shaded mesh faces
    poly = Poly3DCollection(v[fs], zsort="min", shade=False)
    poly.set_facecolor(fc); poly.set_edgecolor("none")
    ax.add_collection3d(poly)

    # Red boundary edges (drawn after mesh → rendered on top)
    if n_boundary > 0:
        lc = Line3DCollection(v[bnd], colors=RED, linewidths=3.0, alpha=1.0)
        ax.add_collection3d(lc)

    return _fig_to_arr(fig, size), n_boundary


# ── Figure composition ─────────────────────────────────────────────────────────

def make_figure():
    print("Creating meshes ...")
    mesh_a = make_watertight_sphere(subdivisions=3)
    mesh_b, n_removed = make_sphere_with_hole(subdivisions=3, n_remove=40)

    print(f"  Mesh A: faces={len(mesh_a.faces)}, watertight={mesh_a.is_watertight}")
    print(f"  Mesh B: faces={len(mesh_b.faces)}, watertight={mesh_b.is_watertight}, removed={n_removed}")

    print("Rendering ...")
    img_a = render_watertight(mesh_a, size=512)
    img_b, n_boundary = render_open_mesh(mesh_b, size=512)

    print(f"  Mesh B boundary edges: {n_boundary}")

    # ── Canvas ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9.0, 5.5), dpi=200, facecolor="white")

    # Title
    fig.text(0.5, 0.992,
             "Watertight vs. Non-watertight Mesh",
             ha="center", va="top",
             fontsize=15, fontweight="bold", color="#111111")

    XC = [0.295, 0.720]   # column x-centres

    # Column headers
    for xc, hdr, hc in zip(
        XC,
        ["Mesh A  —  Watertight", "Mesh B  —  Non-watertight"],
        [GREEN, RED],
    ):
        fig.text(xc, 0.945, hdr,
                 ha="center", va="top",
                 fontsize=12, fontweight="bold", color=hc)

    # GridSpec: 1 row × 2 image cols
    gs = gridspec.GridSpec(
        nrows=1, ncols=2,
        figure=fig,
        hspace=0.0, wspace=0.035,
        left=0.055, right=0.975, top=0.908, bottom=0.215,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(img_a); ax_a.set_xticks([]); ax_a.set_yticks([])
    for sp in ax_a.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2.2)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.imshow(img_b); ax_b.set_xticks([]); ax_b.set_yticks([])
    for sp in ax_b.spines.values():
        sp.set_edgecolor(RED); sp.set_linewidth(2.2)

    # Vertical divider between cells
    fig.add_artist(plt.Line2D(
        [0.508, 0.508], [0.215, 0.908],
        transform=fig.transFigure,
        color="#cccccc", linewidth=0.9,
    ))

    # Info labels (3 rows each)
    rows_a = [
        ("Watertight: ✓", GREEN,     True,  False),
        ("Open edges: 0", "#333333", False, False),
        ("production-fit", GREEN,    False, True ),
    ]
    rows_b = [
        ("Watertight: ✗",           RED,      True,  False),
        (f"Open edges: {n_boundary}", "#333333", False, False),
        ("3D printing fails",        RED,      False, True ),
    ]

    for xc, rows in zip(XC, [rows_a, rows_b]):
        y = 0.188
        for text, col, bold, italic in rows:
            fig.text(xc, y, text,
                     ha="center", va="top",
                     fontsize=11.5,
                     fontweight="bold"   if bold   else "normal",
                     style="italic"      if italic else "normal",
                     color=col)
            y -= 0.060

    # ── Save ────────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "fig_d1_watertight_concept.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nSaved: {out}")
    print("\n=== Summary ===")
    print(f"  Faces removed  : {n_removed}")
    print(f"  Open boundary edges: {n_boundary}")


if __name__ == "__main__":
    make_figure()
