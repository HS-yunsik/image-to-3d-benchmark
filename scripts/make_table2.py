"""
Table 2 — Raw metrics for presentation slides.
Source: data/metrics_table_v4.xlsx  (Raw Values sheet)
Output: outputs/table2_raw_metrics.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT    = Path(__file__).resolve().parent.parent
XLSX    = ROOT / "data" / "metrics_table_v4.xlsx"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── palette ──────────────────────────────────────────────────────────────────
BG_D1     = "#E6F1FB"
BG_D2     = "#E1F5EE"
BG_D3     = "#EEEDFE"
BG_TOP8   = "#FFFFFF"
BG_BOT8   = "#F8F8F8"
BG_GRP_D1 = "#C2D9F2"   # darker shade for group header row
BG_GRP_D2 = "#B8E8D4"
BG_GRP_D3 = "#D4D1FC"
BG_COL_HDR_DEFAULT = "#2C3E50"  # dark for Model/ELO header
BG_COL_HDR_D1 = "#1A6FA8"
BG_COL_HDR_D2 = "#1B8A5A"
BG_COL_HDR_D3 = "#5A4FA8"

RED   = "#E24B4A"
GREEN = "#0F6E56"
DARK  = "#1C1C1C"
GRAY  = "#666666"

NO_UV_MODELS = {
    "InstantMesh", "Unique3D", "Hi3DGen", "MeshFormer",
    "Real3D", "TripoSR", "IM-MA",
}

# ── column definitions ────────────────────────────────────────────────────────
# (src_col, display_label, group, col_width_weight)
COL_DEFS = [
    ("Model",            "Model",    "meta", 2.0),
    ("ELO",              "ELO",      "meta", 0.8),
    ("WT (%)",           "WT (%)",   "D1",   0.7),
    ("Avg CC",           "Avg CC",   "D1",   1.0),
    ("UV (%)",           "UV (%)",   "D2",   0.7),
    ("UV Pack Area",     "UV Pack",  "D2",   0.85),
    ("PBR Channels",     "PBR Ch",   "D3",   0.75),
    ("Texture Res (px)", "Tex Res",  "D3",   0.85),
]
N_COLS = len(COL_DEFS)

GROUP_INFO = {
    "D1": {"label": "D1  Geometry", "cols": [2, 3], "bg": BG_D1, "hdr": BG_COL_HDR_D1},
    "D2": {"label": "D2  UV",       "cols": [4, 5], "bg": BG_D2, "hdr": BG_COL_HDR_D2},
    "D3": {"label": "D3  PBR",      "cols": [6, 7], "bg": BG_D3, "hdr": BG_COL_HDR_D3},
}

# ── data formatting ───────────────────────────────────────────────────────────
def fmt_cell(model: str, col_src: str, val) -> str:
    no_uv = model in NO_UV_MODELS
    if col_src == "WT (%)":
        return str(int(val)) if not pd.isna(val) else "-"
    if col_src == "Avg CC":
        return f"{int(round(val)):,}" if not pd.isna(val) else "-"
    if col_src == "UV (%)":
        if no_uv:
            return "-"
        return str(int(val)) if not pd.isna(val) else "-"
    if col_src == "UV Pack Area":
        if no_uv or pd.isna(val):
            return "-"
        return f"{val:.2f}"
    if col_src == "PBR Channels":
        if no_uv or pd.isna(val):
            return "-"
        return f"{val:.1f}"
    if col_src == "Texture Res (px)":
        if no_uv or pd.isna(val) or val == 0:
            return "-"
        return f"{int(val)}px"
    if col_src == "ELO":
        return str(int(val))
    return str(val)


def cell_color(model: str, col_src: str, val, row_idx: int):
    """Return (text_color, is_bold)."""
    if col_src == "ELO":
        return DARK, True
    if col_src == "WT (%)":
        v = float(val) if not pd.isna(val) else 0
        return (GREEN if v > 0 else RED), False
    if col_src == "Avg CC":
        v = float(val) if not pd.isna(val) else 0
        return (RED if v > 10000 else DARK), False
    return DARK, False


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_rect(ax, x, y, w, h, fc, ec="none", lw=0.5, zorder=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                            linewidth=lw, transform=ax.transAxes,
                            clip_on=False, zorder=zorder))


def draw_text(ax, x, y, txt, fontsize=10, bold=False, color=DARK,
              ha="center", va="center"):
    ax.text(x, y, txt,
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=color,
            ha=ha, va=va,
            transform=ax.transAxes,
            clip_on=False)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    raw = pd.read_excel(XLSX, sheet_name="Raw Values")
    # keep only needed columns, already ELO-sorted in file
    src_cols = [c[0] for c in COL_DEFS]
    df = raw[src_cols].copy().reset_index(drop=True)
    n_rows = len(df)

    # ── layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 7), dpi=150, facecolor="white")
    ax  = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # Column x-positions (normalized)
    total_w   = sum(c[3] for c in COL_DEFS)
    col_widths = [c[3] / total_w for c in COL_DEFS]
    col_x = [sum(col_widths[:i]) for i in range(N_COLS)]   # left edge of each col

    # Row heights
    grp_h  = 0.075   # group header row
    col_h  = 0.072   # column label row
    data_h = (1.0 - grp_h - col_h) / n_rows
    grp_y  = 1.0 - grp_h
    col_y  = grp_y   - col_h
    # data rows: top-down
    def row_y(i):
        return col_y - (i + 1) * data_h

    # ── group header row ──────────────────────────────────────────────────────
    # Model / ELO cells (dark bg)
    for ci in [0, 1]:
        draw_rect(ax, col_x[ci], grp_y, col_widths[ci], grp_h, BG_COL_HDR_DEFAULT)

    for grp, info in GROUP_INFO.items():
        ci_start = info["cols"][0]
        ci_end   = info["cols"][-1]
        x0 = col_x[ci_start]
        x1 = col_x[ci_end] + col_widths[ci_end]
        draw_rect(ax, x0, grp_y, x1 - x0, grp_h, info["hdr"])
        draw_text(ax, (x0 + x1) / 2, grp_y + grp_h * 0.5,
                  info["label"], fontsize=11, bold=True, color="white")

    # ── column label row ──────────────────────────────────────────────────────
    col_bg_map = {
        "meta": BG_COL_HDR_DEFAULT,
        "D1":   BG_D1,
        "D2":   BG_D2,
        "D3":   BG_D3,
    }
    col_txt_map = {
        "meta": "white",
        "D1":   "#1A6FA8",
        "D2":   "#1B8A5A",
        "D3":   "#5A4FA8",
    }
    for ci, (src, lbl, grp, _) in enumerate(COL_DEFS):
        bg  = col_bg_map[grp]
        tc  = col_txt_map[grp]
        draw_rect(ax, col_x[ci], col_y, col_widths[ci], col_h, bg)
        draw_text(ax, col_x[ci] + col_widths[ci] / 2,
                  col_y + col_h * 0.5,
                  lbl, fontsize=10, bold=True, color=tc)

    # ── data rows ─────────────────────────────────────────────────────────────
    grp_bg_for_ci = {}
    for grp, info in GROUP_INFO.items():
        for ci in info["cols"]:
            grp_bg_for_ci[ci] = info["bg"]

    for ri, row in df.iterrows():
        model  = str(row["Model"])
        bg_row = BG_TOP8 if ri < 8 else BG_BOT8
        y_bot  = row_y(ri)

        for ci, (src, lbl, grp, _) in enumerate(COL_DEFS):
            val = row[src]
            txt = fmt_cell(model, src, val)
            tc, is_bold = cell_color(model, src, val, ri)

            # Cell background: group color tinted into row bg
            if grp in ("D1", "D2", "D3"):
                # blend group color lightly with row bg
                cell_bg = grp_bg_for_ci[ci] if bg_row == BG_TOP8 else "#F0F0F8" if grp == "D3" else "#F0F5F0" if grp == "D2" else "#EEF4FA"
            else:
                cell_bg = bg_row

            draw_rect(ax, col_x[ci], y_bot, col_widths[ci], data_h, cell_bg)

            ha = "left" if ci == 0 else "center"
            tx = col_x[ci] + (0.008 if ci == 0 else col_widths[ci] / 2)
            draw_text(ax, tx, y_bot + data_h * 0.5,
                      txt, fontsize=10, bold=is_bold, color=tc, ha=ha)

    # ── grid lines ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    def hline(y, lw=0.5, color="#CCCCCC", zorder=10):
        ax.add_artist(Line2D([0, 1], [y, y], color=color, linewidth=lw,
                             transform=ax.transAxes, clip_on=False, zorder=zorder))

    def vline(x, y0, y1, lw=0.7, color="#AAAAAA", zorder=10):
        ax.add_artist(Line2D([x, x], [y0, y1], color=color, linewidth=lw,
                             transform=ax.transAxes, clip_on=False, zorder=zorder))

    # header bottom (thick)
    hline(col_y, lw=2.0, color="#888888")
    # group header bottom
    hline(grp_y, lw=1.2, color="#AAAAAA")
    # between group header and column label
    # row separators
    for ri in range(n_rows):
        hline(row_y(ri), lw=0.4, color="#DDDDDD")

    # group boundary verticals (D1|D2 and D2|D3)
    for grp_idx, (grp, info) in enumerate(GROUP_INFO.items()):
        x_left = col_x[info["cols"][0]]
        vline(x_left, 0.0, grp_y + grp_h, lw=1.0, color="#AAAAAA")
    # right edge of D3
    last_ci = GROUP_INFO["D3"]["cols"][-1]
    vline(col_x[last_ci] + col_widths[last_ci],
          0.0, grp_y + grp_h, lw=1.0, color="#AAAAAA")

    # ELO | D1 boundary
    vline(col_x[2], 0.0, grp_y, lw=0.5, color="#CCCCCC")

    # ── top-8 / bottom-8 divider (slightly thicker) ───────────────────────────
    hline(row_y(7), lw=1.2, color="#AAAAAA")

    # ── save ──────────────────────────────────────────────────────────────────
    out = OUT_DIR / "table2_raw_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {out}")
    print(f"Table: {n_rows} rows x {N_COLS} columns")
    print(f"  Top-8  (ELO >= {int(df.iloc[7]['ELO'])}): white bg")
    print(f"  Bot-8  (ELO <  {int(df.iloc[8]['ELO'])}): gray bg")


if __name__ == "__main__":
    main()
