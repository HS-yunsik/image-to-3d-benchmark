"""
Spearman correlation results table for presentation slides.
Source: data/elo_correlation_v5.csv
Output: outputs/table_spearman_results.png  (7x4 in, dpi=200)
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
C_HDR_BG   = "#444441"
C_WHITE    = "#FFFFFF"
C_ROW_ALT  = "#F8F8F8"
C_TOTAL_BG = "#EEEDFE"
C_TOTAL_TX = "#3C3489"
C_TEX9_BG  = "#E6F1FB"
C_TEX9_TX  = "#0C447C"
C_RED      = "#E24B4A"
C_DARK     = "#1C1C1C"
C_GRAY     = "#888888"
C_SEP      = "#AAAAAA"

# ── table data (from CSV) ─────────────────────────────────────────────────────
raw = pd.read_csv(DATA / "elo_correlation_v5.csv")
lookup = {row["metric"]: row for _, row in raw.iterrows()}

ROWS = [
    # (display_name, metric_key, row_style)
    ("D1  Geometry",                         "D1 Geometry",          "plain"),
    ("D2  UV",                               "D2 UV",                "plain"),
    ("D3  PBR",                              "D3 PBR",               "plain"),
    ("Texture Res  (n = 16)",                "texture_resolution",   "plain"),
    ("Texture Res  (n = 9, texture-bearing)","tex_res_n9",           "tex9"),
    ("Total  (D1 + D2 + D3)",               "Total (D1+D2+D3)",     "total"),
]

def fmt_r(v: float) -> str:
    return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

def fmt_p(v: float) -> str:
    return f"{v:.3f}"

# ── layout constants ──────────────────────────────────────────────────────────
FIG_W, FIG_H = 7, 4
DPI          = 200

N_ROWS  = len(ROWS)
HDR_H   = 0.135       # header height (fraction of axes)
DATA_H  = (1.0 - HDR_H) / N_ROWS
PAD_TOP = 0.04        # top margin inside figure axes
PAD_BOT = 0.04

# column x-positions (normalised, left edges)
# Metric | r | p-value | Significant
COL_X = [0.00, 0.52, 0.70, 0.87]
COL_W = [0.52, 0.18, 0.17, 0.13]   # must sum to 1.0

HDR_LABELS = ["Metric", "r", "p-value", "Sig."]
HDR_HA     = ["left", "center", "center", "center"]

def row_y_bottom(ri: int) -> float:
    """Bottom of data row ri (0 = top row)."""
    return (1.0 - HDR_H) - (ri + 1) * DATA_H

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor="white")
ax  = fig.add_axes([0.03, 0.03, 0.94, 0.94])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()

def rect(x, y, w, h, fc, ec="none", lw=0.5, z=1):
    ax.add_patch(Rectangle((x, y), w, h,
                            facecolor=fc, edgecolor=ec, linewidth=lw,
                            transform=ax.transAxes, clip_on=False, zorder=z))

def txt(x, y, s, fs=11, bold=False, color=C_DARK, ha="center", va="center"):
    ax.text(x, y, s,
            fontsize=fs, fontweight="bold" if bold else "normal",
            color=color, ha=ha, va=va,
            transform=ax.transAxes, clip_on=False)

def hline(y, lw=0.6, color=C_SEP, ls="-", z=10):
    ax.add_artist(Line2D([0, 1], [y, y],
                         color=color, linewidth=lw, linestyle=ls,
                         transform=ax.transAxes, clip_on=False, zorder=z))

# ── header ────────────────────────────────────────────────────────────────────
rect(0, 1 - HDR_H, 1, HDR_H, C_HDR_BG)
for ci, (lbl, ha) in enumerate(zip(HDR_LABELS, HDR_HA)):
    cx = COL_X[ci] + (0.012 if ha == "left" else COL_W[ci] / 2)
    txt(cx, 1 - HDR_H / 2, lbl, fs=12, bold=True, color=C_WHITE, ha=ha)

hline(1 - HDR_H, lw=1.5, color="#222222")

# ── data rows ─────────────────────────────────────────────────────────────────
for ri, (display, key, style) in enumerate(ROWS):
    row   = lookup[key]
    r_val = float(row["r"])
    p_val = float(row["p_value"])
    sig   = bool(row["significant"])
    n_val = int(row["n"])

    y_bot = row_y_bottom(ri)
    y_mid = y_bot + DATA_H / 2

    # background
    if style == "total":
        bg = C_TOTAL_BG
    elif style == "tex9":
        bg = C_TEX9_BG
    else:
        bg = C_WHITE if ri % 2 == 0 else C_ROW_ALT
    rect(0, y_bot, 1, DATA_H, bg)

    # ── Metric label ──────────────────────────────────────────────────────────
    lbl_color = C_TOTAL_TX if style == "total" else (C_TEX9_TX if style == "tex9" else C_DARK)
    lbl_bold  = style == "total"
    txt(COL_X[0] + 0.012, y_mid, display,
        fs=11, bold=lbl_bold, color=lbl_color, ha="left")

    # ── r value ───────────────────────────────────────────────────────────────
    if style == "total":
        r_color = C_TOTAL_TX
        r_bold  = True
    elif key == "D1 Geometry":
        r_color = C_RED
        r_bold  = False
    else:
        r_color = C_DARK
        r_bold  = False
    txt(COL_X[1] + COL_W[1] / 2, y_mid, fmt_r(r_val),
        fs=11, bold=r_bold, color=r_color, ha="center")

    # ── p-value ───────────────────────────────────────────────────────────────
    p_color = C_TOTAL_TX if style == "total" else (C_TEX9_TX if style == "tex9" else C_DARK)
    txt(COL_X[2] + COL_W[2] / 2, y_mid, fmt_p(p_val),
        fs=11, color=p_color, ha="center")

    # ── significance ──────────────────────────────────────────────────────────
    sig_txt   = "★" if sig else "—"
    sig_color = C_RED if sig else C_GRAY
    sig_bold  = sig
    txt(COL_X[3] + COL_W[3] / 2, y_mid, sig_txt,
        fs=12, bold=sig_bold, color=sig_color, ha="center")

# ── separators ────────────────────────────────────────────────────────────────
# between each row
for ri in range(N_ROWS):
    hline(row_y_bottom(ri), lw=0.4, color="#DDDDDD")

# above Total row (slightly thick)
total_ri = next(i for i, (_, _, s) in enumerate(ROWS) if s == "total")
hline(row_y_bottom(total_ri) + DATA_H, lw=1.4, color="#888888")

# above tex9 row (thin dashed — subset marker)
tex9_ri = next(i for i, (_, k, _) in enumerate(ROWS) if k == "tex_res_n9")
hline(row_y_bottom(tex9_ri) + DATA_H, lw=0.8, color="#AAAAAA", ls=(0, (4, 3)))

# below header already drawn; just ensure bottom of last row is clean
hline(row_y_bottom(N_ROWS - 1), lw=0.4, color="#DDDDDD")

# ── footnote ──────────────────────────────────────────────────────────────────
ax.text(0.012, 0.018,
        "★ p < 0.05   |   n = 16 ELO models unless noted",
        fontsize=8.5, color=C_GRAY,
        transform=ax.transAxes, clip_on=False, va="bottom")

# ── save ──────────────────────────────────────────────────────────────────────
out = OUT_DIR / "table_spearman_results.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
print(f"Table: {N_ROWS} rows x 4 columns")
for display, key, _ in ROWS:
    row = lookup[key]
    print(f"  {display:<42}  r={float(row['r']):+.3f}  p={float(row['p_value']):.3f}  "
          f"sig={'★' if row['significant'] else '-'}")
