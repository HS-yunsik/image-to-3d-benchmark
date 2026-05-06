"""
Watertight visual comparison: MeshFormer (WT=0%) vs InstantMesh (WT=89%)
Both models have NO texture — fair visual comparison.
3 prompts x 2 models using official 3darena thumbnails.
Output: outputs/fig_watertight_visual_comparison.png  (dpi=200)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

ROOT      = Path(__file__).resolve().parent.parent
THUMB_DIR = ROOT / "data" / "3darena_thumbs" / "outputs"
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
C_RED    = "#E24B4A"
C_GREEN  = "#1D9E75"
C_WHITE  = "#FFFFFF"
C_BLACK  = "#000000"
C_DARK   = "#111111"
C_GRAY   = "#888888"
C_BGFIG  = "#FFFFFF"

# ── data ──────────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":      "MeshFormer",
        "elo":       1192,
        "rank":      12,
        "wt_pct":    0,
        "wt_ok":     False,
        "color":     C_RED,
    },
    {
        "name":      "InstantMesh",
        "elo":       1278,
        "rank":      8,
        "wt_pct":    89,
        "wt_ok":     True,
        "color":     C_GREEN,
    },
]

PROMPTS = [
    ("A_cartoon_house_with_red_roof", "house"),
    ("A_castle_made_of_cardboard",    "castle"),
    ("A_heart_made_of_wood",          "heart"),
]

# ── layout ────────────────────────────────────────────────────────────────────
N_ROWS   = len(PROMPTS)
N_COLS   = len(MODELS)
CELL_PX  = 320          # square cell size in pixels (display)
BORDER   = 5            # border thickness (pts)

# Figure dimensions (inches)
LABEL_W  = 0.55         # left label column width (in)
HDR_H    = 0.80         # column header height (in)
CAP_TOP  = 0.55         # top caption height (in)
CAP_BOT  = 0.60         # bottom caption height (in)
GAP      = 0.06         # gap between cells (in)
CELL_W   = 2.4          # cell width (in)
CELL_H   = 2.4          # cell height (in)

FIG_W = LABEL_W + N_COLS * CELL_W + (N_COLS - 1) * GAP + 0.25
FIG_H = CAP_TOP + HDR_H + N_ROWS * CELL_H + (N_ROWS - 1) * GAP + CAP_BOT

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=200, facecolor=C_BGFIG)

# ── coordinate helpers ────────────────────────────────────────────────────────
def col_left(ci):
    return (LABEL_W + ci * (CELL_W + GAP)) / FIG_W

def row_bottom(ri):
    # ri=0 is top row
    top_offset = CAP_TOP + HDR_H + ri * (CELL_H + GAP)
    return 1.0 - (top_offset + CELL_H) / FIG_H

def col_center(ci):
    return col_left(ci) + CELL_W / 2 / FIG_W

def row_center(ri):
    return row_bottom(ri) + CELL_H / 2 / FIG_H

# ── top caption ───────────────────────────────────────────────────────────────
fig.text(0.5, 1.0 - CAP_TOP * 0.35 / FIG_H,
         "Can you tell which mesh is watertight?",
         ha="center", va="center",
         fontsize=15, fontweight="bold", color=C_DARK,
         fontfamily="DejaVu Sans")

# ── column headers ────────────────────────────────────────────────────────────
hdr_top    = 1.0 - CAP_TOP / FIG_H
hdr_bottom = hdr_top - HDR_H / FIG_H

for ci, m in enumerate(MODELS):
    cx  = col_center(ci)
    cy  = (hdr_top + hdr_bottom) / 2
    col = m["color"]

    # background pill
    ax_hdr = fig.add_axes([col_left(ci), hdr_bottom,
                            CELL_W / FIG_W, HDR_H / FIG_H])
    ax_hdr.set_axis_off()
    ax_hdr.set_xlim(0, 1); ax_hdr.set_ylim(0, 1)
    ax_hdr.add_patch(FancyBboxPatch(
        (0.03, 0.06), 0.94, 0.88,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        facecolor=col, alpha=0.10,
        edgecolor=col, linewidth=2.2,
        transform=ax_hdr.transAxes, clip_on=False,
    ))

    # model name
    ax_hdr.text(0.5, 0.78, m["name"],
                ha="center", va="center",
                fontsize=13, fontweight="bold", color=C_DARK,
                fontfamily="DejaVu Sans")
    # ELO + rank
    ax_hdr.text(0.5, 0.50,
                f"ELO: {m['elo']}  (Rank {m['rank']})",
                ha="center", va="center",
                fontsize=10, color=C_GRAY,
                fontfamily="DejaVu Sans")
    # Watertight status
    wt_str = f"Watertight: {m['wt_pct']}%  {'checkmark' if m['wt_ok'] else 'cross'}"
    wt_sym = "✓" if m["wt_ok"] else "✗"
    wt_col = C_GREEN if m["wt_ok"] else C_RED
    ax_hdr.text(0.5, 0.23,
                f"Watertight: {m['wt_pct']}%  {wt_sym}",
                ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=wt_col,
                fontfamily="DejaVu Sans")

# ── row label column width in figure fraction ─────────────────────────────────
LABEL_FRAC = LABEL_W / FIG_W

# ── image cells ───────────────────────────────────────────────────────────────
for ri, (prompt_key, short_label) in enumerate(PROMPTS):
    rb = row_bottom(ri)
    rh = CELL_H / FIG_H

    # row label (left margin)
    fig.text(LABEL_FRAC * 0.55, rb + rh / 2,
             short_label,
             ha="center", va="center",
             fontsize=11, color=C_GRAY, style="italic",
             fontfamily="DejaVu Sans",
             rotation=0)

    for ci, m in enumerate(MODELS):
        cl  = col_left(ci)
        cw  = CELL_W / FIG_W

        # load image
        img_path = THUMB_DIR / m["name"] / (prompt_key + ".png")
        img = np.array(Image.open(img_path).convert("RGB")
                       .resize((CELL_PX, CELL_PX), Image.LANCZOS))

        # axes for image
        ax = fig.add_axes([cl, rb, cw, rh])
        ax.imshow(img, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])

        # colored border via spine
        col = m["color"]
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(BORDER)

# ── bottom caption ────────────────────────────────────────────────────────────
cap_y = (CAP_BOT * 0.62) / FIG_H
fig.text(0.5, cap_y,
         "Both look similar — but only InstantMesh passes production requirements.",
         ha="center", va="center",
         fontsize=12, color=C_DARK,
         fontfamily="DejaVu Sans")

# color legend inline
leg_y = (CAP_BOT * 0.22) / FIG_H
from matplotlib.patches import Patch
from matplotlib.lines import Line2D as L2D
handles = [
    Patch(facecolor=C_RED,   edgecolor=C_RED,   label="MeshFormer — Watertight 0%  ✗  (no texture)"),
    Patch(facecolor=C_GREEN, edgecolor=C_GREEN, label="InstantMesh — Watertight 89%  ✓  (no texture)"),
]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, leg_y / 0.6 - 0.02),
           ncol=2, fontsize=9.5,
           framealpha=0.0, edgecolor="none",
           handlelength=1.2, handleheight=0.8)

# ── save ──────────────────────────────────────────────────────────────────────
out = OUT_DIR / "fig_watertight_visual_comparison.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=C_BGFIG)
plt.close(fig)
print(f"Saved: {out}")
print(f"Grid: {N_ROWS} rows x {N_COLS} cols  ({N_ROWS*N_COLS} cells)")
for ri, (pk, sl) in enumerate(PROMPTS):
    print(f"  row {ri+1}: {sl:8s} ({pk})")
