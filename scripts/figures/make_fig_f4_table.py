"""
Figure F4 — Top-8 ELO models × Watertight table.
Output: outputs/fig_f4_watertight_table.png  (7×5 in, dpi=200)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── data ──────────────────────────────────────────────────────────────────────
ROWS = [
    (1, "Strawberrry",  1382,  0),
    (2, "Strawb3rry",   1370,  0),
    (3, "TRELLIS",      1306,  0),
    (4, "Zaohaowu3D",   1302,  0),
    (5, "Hunyuan3D-2",  1298,  0),
    (6, "InstantMesh",  1278, 89),
    (7, "Meshy-5",      1243,  0),
    (8, "Unique3D",     1230, 50),
]

# ── palette ───────────────────────────────────────────────────────────────────
C_HDR_BG   = "#3A3A3A"
C_WHITE    = "#FFFFFF"
C_RED_BG   = "#FCEBEB"
C_RED_TX   = "#E24B4A"
C_GRN_BG   = "#E1F5EE"
C_GRN_TX   = "#0F6E56"
C_DARK     = "#1C1C1C"
C_GRAY     = "#888888"
C_SEP      = "#DDDDDD"
C_SUM_BG   = "#F8D7D7"
C_SUM_TX   = "#C0392B"

# ── layout ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 7, 5
DPI = 200

N_DATA = len(ROWS)
N_ROWS = N_DATA + 1   # +1 summary row

HDR_H  = 0.13          # header row height (axes fraction)
SUM_H  = 0.10          # summary row height
DATA_H = (1.0 - HDR_H - SUM_H) / N_DATA

# Column x positions (left edges, axes fraction) and widths
# Rank | Model | ELO | Watertight(%)
COL_X = [0.00, 0.12, 0.58, 0.76]
COL_W = [0.12, 0.46, 0.18, 0.24]
HDR_LABELS = ["Rank", "Model", "ELO", "Watertight (%)"]
HDR_HA     = ["center", "left", "center", "center"]

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
ax  = fig.add_axes([0.03, 0.03, 0.94, 0.94])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()

def rect(x, y, w, h, fc, z=1):
    ax.add_patch(Rectangle((x, y), w, h,
                            facecolor=fc, edgecolor="none",
                            transform=ax.transAxes, clip_on=False, zorder=z))

def txt(x, y, s, fs=11, bold=False, color=C_DARK, ha="center", va="center"):
    ax.text(x, y, s,
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=color, ha=ha, va=va,
            transform=ax.transAxes, clip_on=False)

def hline(y, lw=0.5, color=C_SEP):
    ax.add_artist(Line2D([0, 1], [y, y],
                         color=color, linewidth=lw,
                         transform=ax.transAxes, clip_on=False, zorder=10))

# ── header ────────────────────────────────────────────────────────────────────
rect(0, 1 - HDR_H, 1, HDR_H, C_HDR_BG)
for ci, (lbl, ha) in enumerate(zip(HDR_LABELS, HDR_HA)):
    pad = 0.012 if ha == "left" else 0
    cx  = COL_X[ci] + (pad if ha == "left" else COL_W[ci] / 2)
    txt(cx, 1 - HDR_H / 2, lbl,
        fs=12, bold=True, color=C_WHITE, ha=ha)
hline(1 - HDR_H, lw=2.0, color="#111111")

# ── data rows ─────────────────────────────────────────────────────────────────
for ri, (rank, model, elo, wt) in enumerate(ROWS):
    wt_ok  = wt > 0
    row_bg = C_GRN_BG if wt_ok else C_RED_BG
    wt_col = C_GRN_TX if wt_ok else C_RED_TX
    wt_sym = "✓" if wt_ok else "✗"

    y_bot = (1 - HDR_H) - (ri + 1) * DATA_H
    y_mid = y_bot + DATA_H / 2

    rect(0, y_bot, 1, DATA_H, row_bg)

    # Rank
    txt(COL_X[0] + COL_W[0] / 2, y_mid, str(rank),
        fs=11, color=C_GRAY)

    # Model name
    txt(COL_X[1] + 0.012, y_mid, model,
        fs=11, bold=True, color=C_DARK, ha="left")

    # ELO
    txt(COL_X[2] + COL_W[2] / 2, y_mid, str(elo),
        fs=11, bold=True, color=C_DARK)

    # Watertight
    wt_str = f"{wt}%  {wt_sym}"
    txt(COL_X[3] + COL_W[3] / 2, y_mid, wt_str,
        fs=12, bold=True, color=wt_col)

    hline(y_bot, lw=0.5, color=C_SEP)

# ── summary row ───────────────────────────────────────────────────────────────
sum_y_bot = (1 - HDR_H) - N_DATA * DATA_H
sum_y_mid = sum_y_bot + SUM_H / 2

rect(0, sum_y_bot, 1, SUM_H, C_SUM_BG)
hline(sum_y_bot + SUM_H, lw=1.5, color="#BBBBBB")  # top border of summary

ax.text(0.5, sum_y_mid,
        "6 / 8 models   →   Watertight = 0%",
        fontsize=13, fontweight="bold",
        color=C_SUM_TX, ha="center", va="center",
        transform=ax.transAxes, clip_on=False)

hline(sum_y_bot, lw=0.5, color=C_SEP)

# ── save ──────────────────────────────────────────────────────────────────────
out = OUT_DIR / "fig_f4_watertight_table.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
