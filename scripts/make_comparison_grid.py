"""
Comparison grid: same prompt, multiple model thumbnails side by side.

Strategy:
  - Textured models (TRELLIS, Hunyuan3D-2, Hunyuan3D-2.1): extract embedded PNG
    from GLB bufferView via pygltflib.
  - Non-textured models (InstantMesh, Hi3DGen): render geometry via matplotlib
    Poly3DCollection (trimesh → numpy → mpl 3D).

Usage:
    python scripts/make_comparison_grid.py
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont
import pygltflib
import trimesh

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR   = ROOT / "meshes" / "outputs"
ISO_DIR    = ROOT / "data" / "iso3d_inputs" / "png@1024"
THUMB_DIR  = ROOT / "data" / "3darena_thumbs" / "outputs"
INPUT_DIR  = ROOT / "data" / "3darena_thumbs" / "inputs" / "images"
OUT_DIR    = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ALL_METRICS = ROOT / "data" / "all_metrics.csv"

# ── Model configuration ───────────────────────────────────────────────────────
MODELS = [
    {"name": "TRELLIS",       "elo": 1306, "elo_rank": 5,  "post_paper": False, "has_texture": True},
    {"name": "Hunyuan3D-2",   "elo": 1298, "elo_rank": 7,  "post_paper": False, "has_texture": True},
    {"name": "Hunyuan3D-2.1", "elo": None, "elo_rank": None, "post_paper": True, "has_texture": True},
    {"name": "InstantMesh",   "elo": 1278, "elo_rank": 8,  "post_paper": False, "has_texture": False},
    {"name": "Hi3DGen",       "elo": 1207, "elo_rank": 11, "post_paper": False, "has_texture": False},
]

THUMB_PX = 400   # thumbnail canvas size
PADDING  = 24    # padding around thumbnails


# ── Utility: extract embedded texture from GLB ────────────────────────────────
def extract_glb_texture(glb_path: Path) -> Image.Image | None:
    try:
        g = pygltflib.GLTF2().load(str(glb_path))
        if not g.images:
            return None
        img_def = g.images[0]
        if img_def.bufferView is not None:
            bv = g.bufferViews[img_def.bufferView]
            blob = g.binary_blob()
            img_bytes = blob[bv.byteOffset: bv.byteOffset + bv.byteLength]
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif img_def.uri and img_def.uri.startswith("data:"):
            import base64
            _, data = img_def.uri.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        return None
    except Exception as e:
        print(f"  [WARN] texture extract failed for {glb_path.name}: {e}")
        return None


# ── Utility: render geometry with matplotlib ──────────────────────────────────
def render_geometry(glb_path: Path, size: int = THUMB_PX) -> Image.Image | None:
    try:
        scene = trimesh.load(str(glb_path), force="scene")
        if isinstance(scene, trimesh.Scene):
            meshes = list(scene.geometry.values())
        else:
            meshes = [scene]

        if not meshes:
            return None

        # Merge all submeshes
        combined = trimesh.util.concatenate(meshes)
        verts = np.array(combined.vertices)
        faces = np.array(combined.faces)

        # Subsample faces for speed (up to 40k)
        MAX_FACES = 40_000
        if len(faces) > MAX_FACES:
            idx = np.random.choice(len(faces), MAX_FACES, replace=False)
            faces = faces[idx]

        # Normalize to unit cube
        center = verts.mean(axis=0)
        verts  = verts - center
        scale  = np.abs(verts).max()
        if scale > 0:
            verts = verts / scale

        # Isometric-like view: rotate 35° azimuth, 20° elevation
        az = np.radians(210)
        el = np.radians(25)
        Rz = np.array([[np.cos(az), -np.sin(az), 0],
                        [np.sin(az),  np.cos(az), 0],
                        [0,           0,           1]])
        Rx = np.array([[1, 0,          0         ],
                        [0, np.cos(el), -np.sin(el)],
                        [0, np.sin(el),  np.cos(el)]])
        R  = Rx @ Rz
        verts = (R @ verts.T).T

        # Compute face normals for shading
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-9)

        # Light direction (from top-right-front)
        light = np.array([0.5, 0.5, 1.0])
        light = light / np.linalg.norm(light)
        diffuse = np.clip(normals @ light, 0, 1)

        # Base color (cool gray-blue, same as 3D Arena style)
        base_rgb = np.array([0.62, 0.71, 0.80])
        ambient  = 0.3
        intensity = ambient + (1 - ambient) * diffuse
        face_colors = np.outer(intensity, base_rgb)
        face_colors = np.clip(face_colors, 0, 1)

        # Depth sort for painter's algorithm
        z_center = (v0[:, 2] + v1[:, 2] + v2[:, 2]) / 3
        order    = np.argsort(z_center)

        # Build Poly3DCollection
        tris = verts[faces[order]]  # (N, 3, 3)
        fc   = face_colors[order]   # (N, 3)

        dpi = 100
        px  = size / dpi
        fig = plt.figure(figsize=(px, px), dpi=dpi, facecolor="white")
        ax  = fig.add_axes([0, 0, 1, 1], projection="3d", facecolor="white")

        poly = Poly3DCollection(tris, zsort="min", shade=False)
        poly.set_facecolor(fc)
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)

        # Axis limits
        lim = 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    except Exception as e:
        print(f"  [WARN] geometry render failed for {glb_path.name}: {e}")
        import traceback; traceback.print_exc()
        return None


# ── Utility: placeholder image ────────────────────────────────────────────────
def make_placeholder(text: str, size: int = THUMB_PX) -> Image.Image:
    img  = Image.new("RGB", (size, size), color=(230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.text((size // 2, size // 2), text, fill=(120, 120, 120), anchor="mm")
    return img


# ── Utility: load per-mesh stats ──────────────────────────────────────────────
def load_stats(prompt_name: str) -> dict:
    """Return {model_name: {watertight, median_cc}} for a given prompt."""
    out = {}
    if not ALL_METRICS.exists():
        return out
    df = pd.read_csv(ALL_METRICS)
    df = df[df["filename"].str.startswith(prompt_name)]
    for _, row in df.iterrows():
        out[row["model"]] = {
            "watertight": bool(row.get("watertight", False)),
            "cc":         int(row.get("connected_components", 0)),
        }
    return out


# ── Core: get thumbnail PIL image for one model ───────────────────────────────
def get_thumbnail(model_cfg: dict, prompt_name: str) -> Image.Image:
    name = model_cfg["name"]

    # 1. Try official 3d-arena thumbnail
    official = THUMB_DIR / name / f"{prompt_name}.png"
    if official.exists():
        print(f"  {name}: using official thumbnail")
        return Image.open(official).convert("RGB")

    # 2. Try extracting embedded GLB texture
    glb = MESH_DIR / name / f"{prompt_name}.glb"
    if glb.exists() and model_cfg["has_texture"]:
        print(f"  {name}: extracting GLB texture …")
        img = extract_glb_texture(glb)
        if img:
            return img

    # 3. Geometry render via matplotlib
    mesh_path = glb if glb.exists() else MESH_DIR / name / f"{prompt_name}.obj"
    if mesh_path.exists():
        print(f"  {name}: rendering geometry …")
        img = render_geometry(mesh_path, size=THUMB_PX)
        if img:
            return img

    print(f"  {name}: all methods failed, using placeholder")
    return make_placeholder(name)


# ── Core: build caption text ──────────────────────────────────────────────────
def build_caption(model_cfg: dict, stats: dict) -> list[str]:
    name = model_cfg["name"]
    if model_cfg["post_paper"]:
        elo_str = "post-paper"
    else:
        elo_str = f"ELO {model_cfg['elo']} (#{model_cfg['elo_rank']})"

    s = stats.get(name, {})
    wt  = "✓" if s.get("watertight") else "✗"
    cc  = s.get("cc", "?")
    return [name, elo_str, f"Watertight: {wt}   CC: {cc}"]


# ── Core: draw single comparison grid ────────────────────────────────────────
def make_grid(prompt_name: str, out_path: Path):
    print(f"\n{'='*60}")
    print(f"Building grid: {prompt_name}")
    print(f"{'='*60}")

    stats = load_stats(prompt_name)

    # ── Collect thumbnails ──────────────────────────────────────────────────
    thumbnails: list[Image.Image] = []
    captions:   list[list[str]]  = []

    for m in MODELS:
        thumb = get_thumbnail(m, prompt_name)
        thumb = thumb.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
        thumbnails.append(thumb)
        captions.append(build_caption(m, stats))

    # ── Load input image (prefer 3d-arena official, fall back to iso3d) ────
    input_path = INPUT_DIR / f"{prompt_name}.jpg"
    if not input_path.exists():
        input_path = ISO_DIR / f"{prompt_name}.png"
    if input_path.exists():
        input_img = Image.open(input_path).convert("RGB")
        input_img = input_img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
        print(f"  Input image loaded: {input_path.name}")
    else:
        input_img = make_placeholder("Input image\nnot found")
        print(f"  Input image not found: {input_path}")

    # ── Layout constants ────────────────────────────────────────────────────
    n_models    = len(MODELS)
    CAPTION_H   = 72   # px for 3 caption lines below each thumbnail
    HEADER_H    = 44   # row label height
    ROW_INNER_H = THUMB_PX + CAPTION_H
    ROW_H       = ROW_INNER_H + HEADER_H

    TOTAL_W = PADDING + n_models * (THUMB_PX + PADDING)
    # Row 0: input (centred over n_models columns)
    # Row 1: model thumbnails
    TOTAL_H = (PADDING
               + HEADER_H + THUMB_PX + CAPTION_H  # row 0 (input)
               + PADDING
               + HEADER_H + THUMB_PX + CAPTION_H  # row 1 (models)
               + PADDING)

    canvas = Image.new("RGB", (TOTAL_W, TOTAL_H), color=(255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    # Try to load a font; fall back to default
    try:
        font_title   = ImageFont.truetype("arial.ttf", 18)
        font_model   = ImageFont.truetype("arialbd.ttf", 15)
        font_caption = ImageFont.truetype("arial.ttf", 13)
        font_label   = ImageFont.truetype("arialbd.ttf", 16)
    except OSError:
        try:
            fnt_path = "C:/Windows/Fonts/arial.ttf"
            font_title   = ImageFont.truetype(fnt_path, 18)
            font_model   = ImageFont.truetype(fnt_path, 15)
            font_caption = ImageFont.truetype(fnt_path, 13)
            font_label   = ImageFont.truetype(fnt_path, 16)
        except OSError:
            font_title = font_model = font_caption = font_label = ImageFont.load_default()

    # ── Row 0: input image (centred) ────────────────────────────────────────
    y0 = PADDING
    # section label
    label0 = "입력 2D 이미지 (iso3d)"
    draw.text((TOTAL_W // 2, y0 + HEADER_H // 2), label0,
              fill=(60, 60, 60), font=font_label, anchor="mm")

    img_x = (TOTAL_W - THUMB_PX) // 2
    img_y = y0 + HEADER_H
    # subtle border
    draw.rectangle([img_x - 2, img_y - 2, img_x + THUMB_PX + 2, img_y + THUMB_PX + 2],
                   outline=(180, 180, 180), width=2)
    canvas.paste(input_img, (img_x, img_y))
    draw.text((TOTAL_W // 2, img_y + THUMB_PX + 6),
              prompt_name.replace("_", " "),
              fill=(80, 80, 80), font=font_caption, anchor="mt")

    # ── Row 1: model thumbnails ──────────────────────────────────────────────
    y1 = y0 + HEADER_H + THUMB_PX + CAPTION_H + PADDING
    label1 = "모델별 출력 비교"
    draw.text((TOTAL_W // 2, y1 + HEADER_H // 2), label1,
              fill=(60, 60, 60), font=font_label, anchor="mm")

    for i, (thumb, caption) in enumerate(zip(thumbnails, captions)):
        tx = PADDING + i * (THUMB_PX + PADDING)
        ty = y1 + HEADER_H

        # Thin border around thumbnail
        draw.rectangle([tx - 1, ty - 1, tx + THUMB_PX + 1, ty + THUMB_PX + 1],
                       outline=(200, 200, 200), width=1)
        canvas.paste(thumb, (tx, ty))

        # Caption (3 lines)
        cx = tx + THUMB_PX // 2
        cy = ty + THUMB_PX + 8
        line_h = 18
        fonts = [font_model, font_caption, font_caption]
        colors = [(30, 30, 30), (80, 80, 80), (100, 100, 100)]
        for j, (line, fnt, col) in enumerate(zip(caption, fonts, colors)):
            draw.text((cx, cy + j * line_h), line, fill=col, font=fnt, anchor="mt")

    # ── Save ────────────────────────────────────────────────────────────────
    canvas.save(out_path, dpi=(150, 150))
    print(f"\n  Saved: {out_path}")
    print(f"  Size:  {canvas.width} × {canvas.height} px")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    prompts = [
        "A_cartoon_house_with_red_roof",
        "A_castle_made_of_cardboard",
        "A_heart_made_of_wood",
    ]

    for prompt in prompts:
        out_file = OUT_DIR / f"comparison_grid_{prompt.lower()}.png"
        make_grid(prompt, out_file)

    print("\nAll done.")


if __name__ == "__main__":
    np.random.seed(42)
    main()
