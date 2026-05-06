"""
Export 9-metric table as Excel + PNG.
Outputs:
  data/metrics_table_v4.xlsx   (Sheet1: Raw Values, Sheet2: Normalized Scores)
  outputs/fig_metrics_table_v4.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]        = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Load & merge ──────────────────────────────────────────────────────────────
summary = pd.read_csv(DATA / "model_summary.csv")
elo16   = summary[summary["elo"].notna()].sort_values("elo", ascending=False)

am   = pd.read_csv(DATA / "all_metrics.csv")
am16 = am[am["model"].isin(set(elo16["model"]))]
agg  = am16.groupby("model", as_index=False).agg(
    avg_uv_bbox=("uv_bbox_efficiency", "mean"),
    avg_uv_pack=("uv_packed_area",     "mean"),
)

df = elo16.merge(agg, on="model", how="left")
df = df.sort_values("elo", ascending=False).reset_index(drop=True)

scores = pd.read_csv(DATA / "dimension_scores_v4.csv")
df = df.merge(scores[["model", "wt_pct", "manifold_pct", "comp_inv",
                       "uv_bbox_pct", "uv_pack_pct", "pbr_pct", "texres_pct",
                       "D1", "D2_UV", "D3_PBR", "Total"]], on="model")

# ── Sheet 1: Raw Values ───────────────────────────────────────────────────────
raw = pd.DataFrame({
    "Model":          df["model"],
    "ELO":            df["elo"].astype(int),
    # D1 Geometry
    "WT (%)":         df["watertight_pct"].round(1),
    "Manifold (%)":   (df["manifold_edge_ratio"] * 100).round(2),
    "Avg CC":         df["avg_components"].round(1),
    # D2 UV
    "UV (%)":         df["has_uv_pct"].round(1),
    "UV BBox Eff":    df["avg_uv_bbox"].round(4),
    "UV Pack Area":   df["avg_uv_pack"].round(4),
    # D3 PBR
    "Texture (%)":    df["has_texture_pct"].round(1),
    "PBR Channels":   df["avg_pbr_channels"].round(2),
    "Texture Res (px)": df["avg_texture_resolution"].round(0).astype("Int64"),
})

# ── Sheet 2: Normalized Scores ───────────────────────────────────────────────
has_uv  = (df["has_uv_pct"]      / 100).round(3)
has_tex = (df["has_texture_pct"] / 100).round(3)

norm = pd.DataFrame({
    "Model":    df["model"],
    "ELO":      df["elo"].astype(int),
    # D1
    "wt_pct":       df["wt_pct"].round(3),
    "manifold_pct": df["manifold_pct"].round(3),
    "cc_inv":       df["comp_inv"].round(3),
    "D1":           df["D1"].round(3),
    # D2 UV
    "has_uv":       has_uv,
    "uv_bbox_pct":  df["uv_bbox_pct"].round(3),
    "uv_pack_pct":  df["uv_pack_pct"].round(3),
    "D2_UV":        df["D2_UV"].round(3),
    # D3 PBR
    "has_tex":      has_tex,
    "pbr_pct":      df["pbr_pct"].round(3),
    "texres_pct":   df["texres_pct"].round(3),
    "D3_PBR":       df["D3_PBR"].round(3),
    # Total
    "Total":        df["Total"].round(3),
})

# ── Save Excel ────────────────────────────────────────────────────────────────
xlsx_path = DATA / "metrics_table_v4.xlsx"
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    raw.to_excel(writer,  sheet_name="Raw Values",        index=False)
    norm.to_excel(writer, sheet_name="Normalized Scores", index=False)

    # Auto-width columns
    for sheet_name, frame in [("Raw Values", raw), ("Normalized Scores", norm)]:
        ws = writer.sheets[sheet_name]
        for col_idx, col in enumerate(frame.columns, 1):
            vals_len = frame[col].fillna("").astype(str).str.len().max()
            max_len  = max(len(str(col)), int(vals_len))
            ws.column_dimensions[
                ws.cell(row=1, column=col_idx).column_letter
            ].width = max_len + 3

print(f"[Excel] Saved: {xlsx_path}")

# ── PNG: two sub-tables ───────────────────────────────────────────────────────
models   = df["model"].tolist()
n_models = len(models)

# ── Table A data ──────────────────────────────────────────────────────────────
col_A = ["WT (%)", "Manifold (%)", "Avg CC",
         "UV (%)", "UV BBox", "UV Pack",
         "Tex (%)", "PBR ch", "TexRes (px)"]

def fmt_uv(v):
    return "N/A" if pd.isna(v) else f"{v:.3f}"

data_A = []
for _, r in df.iterrows():
    data_A.append([
        f"{r['watertight_pct']:.0f}",
        f"{r['manifold_edge_ratio']*100:.2f}",
        f"{r['avg_components']:.0f}",
        f"{r['has_uv_pct']:.0f}",
        fmt_uv(r["avg_uv_bbox"]),
        fmt_uv(r["avg_uv_pack"]),
        f"{r['has_texture_pct']:.0f}",
        f"{r['avg_pbr_channels']:.1f}",
        f"{r['avg_texture_resolution']:.0f}",
    ])

# ── Table B data ──────────────────────────────────────────────────────────────
col_B = ["wt", "manif", "cc_inv", "D1",
         "has_uv", "uv_bb", "uv_pk", "D2",
         "has_tx", "pbr", "txres", "D3", "Total"]

data_B = []
for _, r in df.iterrows():
    huv = r["has_uv_pct"] / 100
    htx = r["has_texture_pct"] / 100
    data_B.append([
        f"{r['wt_pct']:.3f}",   f"{r['manifold_pct']:.3f}", f"{r['comp_inv']:.3f}",  f"{r['D1']:.3f}",
        f"{huv:.3f}",            f"{r['uv_bbox_pct']:.3f}",  f"{r['uv_pack_pct']:.3f}", f"{r['D2_UV']:.3f}",
        f"{htx:.3f}",            f"{r['pbr_pct']:.3f}",      f"{r['texres_pct']:.3f}",  f"{r['D3_PBR']:.3f}",
        f"{r['Total']:.3f}",
    ])

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 21), dpi=200, facecolor="white")

# Color helpers
def cell_color_raw(col_name, val_str):
    try:
        v = float(val_str)
    except ValueError:
        return "#F5F5F5"
    if col_name in ("WT (%)", "UV (%)", "Tex (%)"):
        intensity = v / 100.0
        r = 1 - 0.55 * intensity
        g = 1 - 0.10 * intensity
        b = 1 - 0.55 * intensity
        return (max(0, min(1, r)), max(0, min(1, g)), max(0, min(1, b)))
    if col_name == "Avg CC":
        log_v = np.log1p(v)
        log_max = np.log1p(97890)
        intensity = log_v / log_max
        return (1, 1 - 0.65 * intensity, 1 - 0.65 * intensity)
    return "#FFFFFF"

NORM_DIM_COLS  = {0, 1, 2, 3}   # D1 group incl D1
UV_DIM_COLS    = {4, 5, 6, 7}
PBR_DIM_COLS   = {8, 9, 10, 11}
TOTAL_COL      = {12}
DIM_SCORE_COLS = {3, 7, 11, 12}  # D1, D2, D3, Total — slightly bold border

GROUP_COLORS = {
    "D1": "#E8F4FD",
    "D2": "#FFF3E0",
    "D3": "#F3E5F5",
    "Total": "#E8F5E9",
}

def cell_bg_norm(col_idx, val_str):
    try:
        v = float(val_str)
    except ValueError:
        return "#F5F5F5"
    if col_idx in NORM_DIM_COLS:
        base = "#E8F4FD"
    elif col_idx in UV_DIM_COLS:
        base = "#FFF3E0"
    elif col_idx in PBR_DIM_COLS:
        base = "#F3E5F5"
    else:
        base = "#E8F5E9"
    # heat overlay: blend base toward green (good) or red (bad)
    r_b = int(base[1:3], 16) / 255
    g_b = int(base[3:5], 16) / 255
    b_b = int(base[5:7], 16) / 255
    alpha = v
    rc = r_b * (1 - alpha * 0.35)
    gc = g_b + (1 - g_b) * 0.0 + alpha * 0.15
    bc = b_b * (1 - alpha * 0.35)
    return (max(0, min(1, rc)), max(0, min(1, gc)), max(0, min(1, bc)))


def draw_table(ax, data, col_labels, row_labels, title,
               col_group_spans=None, cell_color_fn=None):
    """Draw a custom matplotlib table."""
    ax.set_axis_off()
    n_rows = len(data)
    n_cols = len(col_labels)

    header_h = 0.10
    row_h    = (1.0 - header_h) / n_rows
    col_w    = 1.0 / (n_cols + 1.5)   # +1.5 for model name column
    model_w  = col_w * 1.5

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Title
    ax.text(0.5, 1.02, title, ha="center", va="bottom",
            fontsize=13, fontweight="bold", transform=ax.transAxes)

    # Header background
    from matplotlib.patches import FancyBboxPatch, Rectangle as Rect
    ax.add_patch(Rect((0, 1 - header_h), 1, header_h,
                      facecolor="#2C3E50", alpha=0.92, transform=ax.transAxes,
                      zorder=1, clip_on=False))

    # "Model" header
    ax.text(model_w / 2, 1 - header_h / 2, "Model",
            ha="center", va="center", fontsize=9.5, fontweight="bold",
            color="white", transform=ax.transAxes)

    # Column headers
    for j, lbl in enumerate(col_labels):
        x = model_w + col_w * j + col_w / 2
        ax.text(x, 1 - header_h / 2, lbl,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white", transform=ax.transAxes)

    # Group separator lines under header
    if col_group_spans:
        for (start, end, grp_label, grp_color) in col_group_spans:
            x0 = model_w + col_w * start
            x1 = model_w + col_w * end
            ax.add_patch(Rect((x0, 1 - header_h), x1 - x0, header_h * 0.28,
                              facecolor=grp_color, alpha=0.85,
                              transform=ax.transAxes, zorder=2, clip_on=False))
            ax.text((x0 + x1) / 2, 1 - header_h * 0.14, grp_label,
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white", transform=ax.transAxes, zorder=3)

    # Data rows
    for i, (row, model_name) in enumerate(zip(data, row_labels)):
        y_top = 1 - header_h - i * row_h
        row_bg = "#F9F9F9" if i % 2 == 0 else "#FFFFFF"
        ax.add_patch(Rect((0, y_top - row_h), 1, row_h,
                          facecolor=row_bg, transform=ax.transAxes,
                          zorder=0, clip_on=False))

        # Model name cell
        ax.text(model_w / 2, y_top - row_h / 2, model_name,
                ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#1f1f1f", transform=ax.transAxes)

        # Data cells
        for j, val in enumerate(row):
            x0 = model_w + col_w * j
            x1 = x0 + col_w
            xc = (x0 + x1) / 2

            # Cell background
            if cell_color_fn:
                bg = cell_color_fn(j, val) if callable(cell_color_fn) else row_bg
            else:
                bg = row_bg

            if isinstance(bg, tuple):
                ax.add_patch(Rect((x0, y_top - row_h), col_w, row_h,
                                  facecolor=bg, transform=ax.transAxes,
                                  zorder=1, clip_on=False))
            elif bg != row_bg:
                ax.add_patch(Rect((x0, y_top - row_h), col_w, row_h,
                                  facecolor=bg, transform=ax.transAxes,
                                  zorder=1, clip_on=False))

            is_bold = (val not in ("N/A", "0", "0.0", "0.00"))
            ax.text(xc, y_top - row_h / 2, val,
                    ha="center", va="center",
                    fontsize=8 if len(val) < 8 else 7,
                    fontweight="bold" if is_bold else "normal",
                    color="#1f1f1f", transform=ax.transAxes)

    # Grid lines (use ax.plot with transAxes)
    from matplotlib.lines import Line2D
    lw = 0.4
    for i in range(n_rows + 1):
        y = 1 - header_h - i * row_h
        ax.add_artist(Line2D([0, 1], [y, y], color="#cccccc", linewidth=lw,
                             transform=ax.transAxes, clip_on=False, zorder=5))
    for j in range(n_cols + 1):
        x = model_w + col_w * j
        ax.add_artist(Line2D([x, x], [0, 1], color="#cccccc", linewidth=lw,
                             transform=ax.transAxes, clip_on=False, zorder=5))
    ax.add_artist(Line2D([model_w, model_w], [0, 1], color="#888888",
                         linewidth=1.0, transform=ax.transAxes, clip_on=False, zorder=6))
    ax.add_artist(Line2D([0, 1], [1 - header_h, 1 - header_h], color="#888888",
                         linewidth=1.0, transform=ax.transAxes, clip_on=False, zorder=6))


# ── Two axes ──────────────────────────────────────────────────────────────────
ax1 = fig.add_axes([0.01, 0.52, 0.98, 0.44])
ax2 = fig.add_axes([0.01, 0.03, 0.98, 0.44])

# Table A group spans: (start_col, end_col_excl, label, color)
spans_A = [
    (0, 3, "D1  Geometry", "#1565C0"),
    (3, 6, "D2  UV",       "#E65100"),
    (6, 9, "D3  PBR",      "#6A1B9A"),
]

draw_table(
    ax1, data_A, col_A, models,
    title="표 A. 9개 메트릭 원시값 (Raw Values)",
    col_group_spans=spans_A,
    cell_color_fn=lambda j, v: cell_color_raw(col_A[j], v),
)

spans_B = [
    (0, 4,  "D1  Geometry", "#1565C0"),
    (4, 8,  "D2  UV",       "#E65100"),
    (8, 12, "D3  PBR",      "#6A1B9A"),
    (12, 13, "Total",       "#2E7D32"),
]

draw_table(
    ax2, data_B, col_B, models,
    title="표 B. 9개 메트릭 정규화 점수 (Percentile Rank [0,1]) + 차원 점수",
    col_group_spans=spans_B,
    cell_color_fn=lambda j, v: cell_bg_norm(j, v),
)

# Footer note
fig.text(0.5, 0.005,
         "* cc_inv = 1 - CC_percentile_rank (CC 낮을수록 좋음).  "
         "UV BBox·UV Pack·PBR·TexRes: UV/Texture 없는 모델은 NaN→0 처리 후 percentile rank.  "
         "D1=mean(wt,manif,cc_inv)  D2=mean(has_uv,uv_bb,uv_pk)  D3=mean(has_tx,pbr,txres)  Total=mean(D1,D2,D3)",
         ha="center", fontsize=8, color="#555")

out_png = OUT_DIR / "fig_metrics_table_v4.png"
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"[PNG]   Saved: {out_png}")
print("\nDone.")
