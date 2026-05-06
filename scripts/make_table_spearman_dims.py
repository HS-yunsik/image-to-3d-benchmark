"""
Per-metric Spearman correlations by dimension, 3-table layout.
Output: outputs/table_spearman_by_dimension.png  (7x9 in, dpi=200)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Build model-level averages ─────────────────────────────────────────────
summary = pd.read_csv(DATA / "model_summary.csv")
elo16   = summary[summary["elo"].notna()].copy()
models16 = set(elo16["model"])

am    = pd.read_csv(DATA / "all_metrics.csv")
am16  = am[am["model"].isin(models16)]

agg = am16.groupby("model", as_index=False).agg(
    avg_watertight          = ("watertight",         "mean"),
    avg_manifold_edge_ratio = ("manifold_edge_ratio","mean"),
    avg_connected_components= ("connected_components","mean"),
    avg_has_uv              = ("has_uv",             "mean"),
    avg_uv_packed_area      = ("uv_packed_area",     "mean"),
    avg_pbr_channel_count   = ("pbr_channel_count",  "mean"),
    avg_has_texture         = ("has_texture",         "mean"),
)
# avg_texture_resolution already in model_summary.csv

df = elo16.merge(agg, on="model", how="left")

# ── 2. Spearman for each metric ───────────────────────────────────────────────
def corr(series: pd.Series, elo: pd.Series):
    mask = series.notna() & elo.notna()
    r, p = spearmanr(series[mask], elo[mask])
    return float(r), float(p), int(mask.sum())

elo_s = df["elo"]

results = {}
results["watertight"]           = corr(df["avg_watertight"],           elo_s)
results["manifold_edge_ratio"]  = corr(df["avg_manifold_edge_ratio"],  elo_s)
results["connected_components"] = corr(df["avg_connected_components"], elo_s)
results["has_uv"]               = corr(df["avg_has_uv"],               elo_s)
results["uv_packed_area"]       = corr(df["avg_uv_packed_area"],       elo_s)
results["pbr_channel_count"]    = corr(df["avg_pbr_channel_count"],    elo_s)
results["texture_resolution"]   = corr(df["avg_texture_resolution"],  elo_s)

# texture_resolution n=9 (texture-bearing models only)
tex_df = df[df["avg_has_texture"] > 0]
r_t9, p_t9 = spearmanr(tex_df["avg_texture_resolution"], tex_df["elo"])
results["texture_resolution_n9"] = (float(r_t9), float(p_t9), len(tex_df))

# ── 3. Console output ─────────────────────────────────────────────────────────
print("\n=== Spearman correlations vs ELO (model-level averages) ===")
for k, (r, p, n) in results.items():
    sig = "★" if p < 0.05 else "-"
    print(f"  {k:<30}  r={r:+.3f}  p={p:.3f}  n={n}  {sig}")

# ── 4. Table data ─────────────────────────────────────────────────────────────
DIMS = [
    {
        "title":  "D1  Geometry",
        "hdr_bg": "#0C447C",
        "sig_bg": "#E6F1FB",
        "rows": [
            ("Watertight ratio",        "watertight"),
            ("Manifold edge ratio",     "manifold_edge_ratio"),
            ("Connected components",    "connected_components"),
        ],
    },
    {
        "title":  "D2  UV",
        "hdr_bg": "#0F6E56",
        "sig_bg": "#E1F5EE",
        "rows": [
            ("has_uv",                  "has_uv"),
            ("UV packed area",          "uv_packed_area"),
        ],
    },
    {
        "title":  "D3  PBR",
        "hdr_bg": "#3C3489",
        "sig_bg": "#EEEDFE",
        "rows": [
            ("PBR channel count",           "pbr_channel_count"),
            ("Texture resolution",          "texture_resolution"),
            ("Texture resolution  (n = 9)", "texture_resolution_n9"),
        ],
    },
]

# ── 5. Drawing ────────────────────────────────────────────────────────────────
C_WHITE  = "#FFFFFF"
C_ALTROW = "#F8F8F8"
C_DARK   = "#1C1C1C"
C_RED    = "#E24B4A"
C_GRAY   = "#999999"
C_SEP    = "#CCCCCC"

FIG_W, FIG_H = 7, 9
DPI          = 200

fig, axes = plt.subplots(3, 1, figsize=(FIG_W, FIG_H), dpi=DPI,
                          facecolor="white")
fig.subplots_adjust(left=0.03, right=0.97,
                    top=0.97,  bottom=0.02,
                    hspace=0.55)

COL_X = [0.00, 0.58, 0.76, 0.91]   # left edges (normalised)
COL_W = [0.58, 0.18, 0.15, 0.09]
HDR_LABELS = ["Metric", "r", "p-value", "Sig."]
HDR_HA     = ["left", "center", "center", "center"]

def draw_table(ax, dim: dict):
    n_rows = len(dim["rows"])
    HDR_H  = 0.20
    DATA_H = (1.0 - HDR_H) / n_rows

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    def rect(x, y, w, h, fc, z=1):
        ax.add_patch(Rectangle((x, y), w, h,
                               facecolor=fc, edgecolor="none",
                               transform=ax.transAxes,
                               clip_on=False, zorder=z))

    def cell_txt(x, y, s, fs=10, bold=False, color=C_DARK, ha="center"):
        ax.text(x, y, s,
                fontsize=fs, fontweight="bold" if bold else "normal",
                color=color, ha=ha, va="center",
                transform=ax.transAxes, clip_on=False)

    def hline(y, lw=0.5, color=C_SEP, ls="-"):
        ax.add_artist(Line2D([0, 1], [y, y],
                             color=color, linewidth=lw, linestyle=ls,
                             transform=ax.transAxes,
                             clip_on=False, zorder=10))

    # dimension title (above axes — use fig.text via offset)
    ax.text(0.0, 1.08, dim["title"],
            fontsize=13, fontweight="bold", color=dim["hdr_bg"],
            transform=ax.transAxes, clip_on=False, va="bottom")

    # header row
    rect(0, 1 - HDR_H, 1, HDR_H, dim["hdr_bg"])
    for ci, (lbl, ha) in enumerate(zip(HDR_LABELS, HDR_HA)):
        cx = COL_X[ci] + (0.012 if ha == "left" else COL_W[ci] / 2)
        cell_txt(cx, 1 - HDR_H / 2, lbl, fs=11, bold=True,
                 color=C_WHITE, ha=ha)
    hline(1 - HDR_H, lw=1.8, color="#111111")

    # data rows
    for ri, (label, key) in enumerate(dim["rows"]):
        r_val, p_val, n_val = results[key]
        sig = p_val < 0.05

        y_bot = (1 - HDR_H) - (ri + 1) * DATA_H
        y_mid = y_bot + DATA_H / 2

        # background
        bg = dim["sig_bg"] if sig else (C_WHITE if ri % 2 == 0 else C_ALTROW)
        rect(0, y_bot, 1, DATA_H, bg)

        # metric label
        cell_txt(COL_X[0] + 0.012, y_mid, label, fs=10, ha="left")

        # r value
        r_str   = f"+{r_val:.3f}" if r_val >= 0 else f"{r_val:.3f}"
        r_color = C_RED if r_val < 0 else C_DARK
        r_bold  = sig
        cell_txt(COL_X[1] + COL_W[1] / 2, y_mid, r_str,
                 fs=10, bold=r_bold, color=r_color)

        # p-value
        p_str = f"{p_val:.3f}"
        cell_txt(COL_X[2] + COL_W[2] / 2, y_mid, p_str, fs=10)

        # significance
        sig_str   = "★" if sig else "—"
        sig_color = C_RED if sig else C_GRAY
        cell_txt(COL_X[3] + COL_W[3] / 2, y_mid, sig_str,
                 fs=11, bold=sig, color=sig_color)

        hline(y_bot, lw=0.4, color="#DDDDDD")

    # bottom line
    hline((1 - HDR_H) - n_rows * DATA_H, lw=0.6, color=C_SEP)

for ax, dim in zip(axes, DIMS):
    draw_table(ax, dim)

out = OUT_DIR / "table_spearman_by_dimension.png"
fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {out}")
