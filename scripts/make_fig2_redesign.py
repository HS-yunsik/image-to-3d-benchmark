"""
Figure 2 redesign — Per-model Dimension Scores (v5).
Blues palette, English only, clean layout.
Output: outputs/fig2_heatmap_v5_redesign.png  (dpi=200)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA / "dimension_scores_v5.csv")
df = df[df["elo"].notna()].copy()
df = df.sort_values("elo", ascending=False).reset_index(drop=True)

score_matrix = df[["D1", "D2", "D3", "Total"]].values
models  = df["model"].tolist()
elos    = df["elo"].astype(int).tolist()
n_rows, n_cols = score_matrix.shape  # 16 × 4

# ── colormap ──────────────────────────────────────────────────────────────────
blues = ["#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1",
         "#6BAED6", "#4292C6", "#2171B5", "#08519C", "#08306B"]
CMAP = LinearSegmentedColormap.from_list("navy_blues", blues, N=256)
NORM = plt.Normalize(0, 1)

# ── column metadata ───────────────────────────────────────────────────────────
COL_META = [
    {"label": "D1 Geometry", "sub": "watertight · manifold · CC", "bg": "#E6F1FB", "tc": "#0C447C"},
    {"label": "D2 UV",       "sub": "has_uv · uv_packed",         "bg": "#E1F5EE", "tc": "#0F6E56"},
    {"label": "D3 PBR",      "sub": "pbr_ch · tex_res",           "bg": "#EEEDFE", "tc": "#3C3489"},
    {"label": "Total",       "sub": "D1+D2+D3",                   "bg": "#F1EFE8", "tc": "#5C5035"},
]

# ── layout (figure fractions, y=0 at bottom) ──────────────────────────────────
FIG_W, FIG_H = 10, 9
DPI = 200

LABEL_W   = 0.225   # row label column
RIGHT_PAD = 0.025
TITLE_H   = 0.060   # top title area
HDR_H     = 0.090   # column header band
BOT_H     = 0.095   # bottom (colorbar + footnote)

HEAT_L = LABEL_W
HEAT_R = 1.0 - RIGHT_PAD
HEAT_W = HEAT_R - HEAT_L
HEAT_T = 1.0 - TITLE_H - HDR_H
HEAT_B = BOT_H
HEAT_H = HEAT_T - HEAT_B

COL_W = HEAT_W / n_cols
ROW_H = HEAT_H / n_rows

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")

# ── row label background (alternating) ────────────────────────────────────────
for i in range(n_rows):
    y_bot = HEAT_B + (n_rows - i - 1) * ROW_H
    bg = "#F4F4F4" if i % 2 == 0 else "#FFFFFF"
    fig.patches.append(Rectangle(
        (0, y_bot), HEAT_L - 0.006, ROW_H,
        transform=fig.transFigure,
        facecolor=bg, edgecolor="none", clip_on=False, zorder=1,
    ))

# ── row labels: model name (bold) + ELO (gray) ────────────────────────────────
for i, (model, elo) in enumerate(zip(models, elos)):
    y_c = HEAT_B + (n_rows - i - 0.5) * ROW_H
    fig.text(0.012, y_c, model,
             ha="left", va="center",
             fontsize=10.5, fontweight="bold", color="#1C1C1C")
    fig.text(HEAT_L - 0.010, y_c, str(elo),
             ha="right", va="center",
             fontsize=9, color="#AAAAAA")

# ── main heatmap ───────────────────────────────────────────────────────────────
ax = fig.add_axes([HEAT_L, HEAT_B, HEAT_W, HEAT_H])
ax.imshow(score_matrix, cmap=CMAP, norm=NORM, aspect="auto",
          interpolation="nearest")
ax.set_axis_off()

# cell text: white on dark cells, dark on light cells
for i in range(n_rows):
    for j in range(n_cols):
        v = score_matrix[i, j]
        tc = "white" if v >= 0.60 else "#2C2C2C"
        ax.text(j, i, f"{v:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=tc, zorder=5)

# row separators (thin white lines)
for i in range(n_rows - 1):
    ax.axhline(i + 0.5, color="white", linewidth=0.8, zorder=4)

# border line between label area and heatmap
fig.add_artist(Line2D(
    [HEAT_L, HEAT_L], [HEAT_B, HEAT_T],
    transform=fig.transFigure,
    color="#CCCCCC", linewidth=1.2, zorder=10,
))

# ── column headers ─────────────────────────────────────────────────────────────
for j, meta in enumerate(COL_META):
    x_l = HEAT_L + j * COL_W
    ax_h = fig.add_axes([x_l, HEAT_T, COL_W, HDR_H])
    ax_h.set_axis_off()
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
    ax_h.add_patch(Rectangle((0, 0), 1, 1,
                              facecolor=meta["bg"], edgecolor="none",
                              transform=ax_h.transAxes))
    ax_h.text(0.5, 0.65, meta["label"],
              ha="center", va="center",
              fontsize=11, fontweight="bold", color=meta["tc"],
              transform=ax_h.transAxes)
    ax_h.text(0.5, 0.27, meta["sub"],
              ha="center", va="center",
              fontsize=8.5, color="#888888",
              transform=ax_h.transAxes)

# separator line between headers and heatmap
fig.add_artist(Line2D(
    [HEAT_L, HEAT_R], [HEAT_T, HEAT_T],
    transform=fig.transFigure,
    color="#BBBBBB", linewidth=1.2, zorder=10,
))

# ── title ──────────────────────────────────────────────────────────────────────
title_center = 1.0 - TITLE_H * 0.5
fig.text(0.5, title_center + 0.018,
         "Per-model Dimension Scores (v5)",
         ha="center", va="center",
         fontsize=14, fontweight="bold", color="#1C1C1C")
fig.text(0.5, title_center - 0.014,
         "16 ELO models, sorted by ELO (descending)",
         ha="center", va="center",
         fontsize=11, color="#777777")

# ── colorbar (horizontal, figure-width 30%, centered on heatmap) ──────────────
heatmap_cx = (HEAT_L + HEAT_R) / 2
cbar_w = 0.28
cbar_h = 0.018
cbar_x = heatmap_cx - cbar_w / 2
cbar_y = BOT_H * 0.45

cb_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
cb = plt.colorbar(mcm.ScalarMappable(norm=NORM, cmap=CMAP),
                  cax=cb_ax, orientation="horizontal")
cb.set_ticks([0, 0.5, 1.0])
cb.ax.set_xticklabels(["0  (worst)", "0.5", "1  (best)"], fontsize=9)
cb.ax.tick_params(labelsize=9, length=3)
cb_ax.set_title("Score", fontsize=9, color="#666666", pad=4)

# ── footnote ───────────────────────────────────────────────────────────────────
fig.text(0.012, 0.013,
         "Percentile rank [0,1].  "
         "D1=mean(wt, manifold, 1/CC).  "
         "D2=mean(has_uv, uv_packed).  "
         "D3=mean(pbr_ch, tex_res).",
         ha="left", va="bottom",
         fontsize=8.5, color="#AAAAAA")

# ── save ───────────────────────────────────────────────────────────────────────
out = OUT_DIR / "fig2_heatmap_v5_redesign.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
print(f"Grid: {n_rows} models × {n_cols} dimensions")
