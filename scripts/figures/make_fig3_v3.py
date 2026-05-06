"""
Figure 3 v3 — revised dimension scoring with percentile-rank normalization.

Scoring rules:
  - All continuous metrics → percentile rank [0,1] within 16 ELO models
    (average rank for ties; NaN filled with 0 = worst before ranking)
  - "lower is better": connected_components, non_triangle_ratio → 1 - pct
  - Boolean per-model: has_uv, has_texture → raw proportion [0,1] (no percentile)
  - face_count: excluded

  D1 = mean(watertight_pct_rank, manifold_edge_ratio_rank, 1 - components_rank)
  D2 = mean(triangle_ratio_rank, 1 - non_triangle_rank)   ← 전 모델 동점 (상수)
  D3 = mean(has_uv_prop, uv_bbox_efficiency_rank, uv_packed_area_rank)
  D4 = mean(has_texture_prop, pbr_channel_count_rank, texture_resolution_rank)
  Total = mean(D1, D3, D4)   ← D2 제외 (변별력 없음)

Outputs:
  data/dimension_scores_v3.csv
  outputs/fig3_heatmap_v3.png  (300 dpi — does NOT touch v2)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"]        = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"]       = 42

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

V3_OUT_FIG = OUT_DIR / "fig3_heatmap_v3.png"
V3_OUT_CSV = DATA / "dimension_scores_v3.csv"
V2_FIG     = OUT_DIR / "fig3_heatmap_v2.png"   # must not be touched

assert V2_FIG.exists(), "Safety: fig3_heatmap_v2.png not found — abort before overwrite risk"


# ── 1. Load and filter to 16 ELO models ──────────────────────────────────────
summary = pd.read_csv(DATA / "model_summary.csv")
elo16 = summary[summary["elo"].notna()].copy().reset_index(drop=True)
assert len(elo16) == 16, f"Expected 16 ELO models, got {len(elo16)}"
models_set = set(elo16["model"])

# ── 2. Aggregate per-model from all_metrics.csv ───────────────────────────────
am = pd.read_csv(DATA / "all_metrics.csv")
am_elo = am[am["model"].isin(models_set)]

agg = (
    am_elo
    .groupby("model", as_index=False)
    .agg(
        avg_triangle_ratio    = ("triangle_ratio",      "mean"),
        avg_non_triangle_ratio= ("non_triangle_ratio",  "mean"),
        avg_uv_bbox_efficiency= ("uv_bbox_efficiency",  "mean"),  # NaN for no-UV
        avg_uv_packed_area    = ("uv_packed_area",      "mean"),  # NaN for no-UV
    )
)

df = elo16.merge(agg, on="model", how="left")
df = df.sort_values("elo", ascending=False).reset_index(drop=True)
n = len(df)  # 16

# ── 3. Percentile rank helper ─────────────────────────────────────────────────
def pct_rank(series: pd.Series, lower_is_better: bool = False) -> np.ndarray:
    """Percentile rank in [0,1]; NaN → 0 (treated as worst before ranking)."""
    vals = series.values.astype(float)
    vals = np.where(np.isnan(vals), 0.0, vals)
    ranks = rankdata(vals, method="average")   # 1 = smallest value
    pct   = (ranks - 1) / (n - 1)             # [0, 1]
    return 1 - pct if lower_is_better else pct

# ── 4. Component scores ───────────────────────────────────────────────────────
# D1 Geometry
wt_pct       = pct_rank(df["watertight_pct"])                       # higher → better
manifold_pct = pct_rank(df["manifold_edge_ratio"])                  # higher → better
comp_inv     = pct_rank(df["avg_components"], lower_is_better=True) # lower  → better

# D2 Topology
tri_pct      = pct_rank(df["avg_triangle_ratio"])                          # higher → better
non_tri_inv  = pct_rank(df["avg_non_triangle_ratio"], lower_is_better=True)# lower  → better

# D3 UV
has_uv_prop  = (df["has_uv_pct"] / 100).values                      # boolean proportion
uv_bbox_pct  = pct_rank(df["avg_uv_bbox_efficiency"])               # NaN→0 handled
uv_pack_pct  = pct_rank(df["avg_uv_packed_area"])                   # NaN→0 handled

# D4 PBR
has_tex_prop = (df["has_texture_pct"] / 100).values                 # boolean proportion
pbr_pct      = pct_rank(df["avg_pbr_channels"])
texres_pct   = pct_rank(df["avg_texture_resolution"])

# ── 5. Dimension & total scores ───────────────────────────────────────────────
d1    = np.column_stack([wt_pct,    manifold_pct, comp_inv ]).mean(axis=1)
d2    = np.column_stack([tri_pct,   non_tri_inv             ]).mean(axis=1)
d3    = np.column_stack([has_uv_prop, uv_bbox_pct, uv_pack_pct]).mean(axis=1)
d4    = np.column_stack([has_tex_prop, pbr_pct,   texres_pct ]).mean(axis=1)
total = np.column_stack([d1, d3, d4]).mean(axis=1)   # D2 제외

df["d1_v3"]    = d1
df["d2_v3"]    = d2
df["d3_v3"]    = d3
df["d4_v3"]    = d4
df["total_v3"] = total

# ── 6. Console: 64-cell score table ──────────────────────────────────────────
print("\n" + "=" * 78)
print("DIMENSION SCORES v3  (16 ELO models, ELO-sorted descending)")
print("=" * 78)
print(f"{'Model':<18} {'ELO':>5}  {'D1':>6}  {'D2':>6}  {'D3':>6}  {'D4':>6}  {'Total(D2제외)':>13}")
print("-" * 78)
for _, row in df.iterrows():
    print(f"{row['model']:<18} {int(row['elo']):>5}  "
          f"{row['d1_v3']:>6.3f}  {row['d2_v3']:>6.3f}  "
          f"{row['d3_v3']:>6.3f}  {row['d4_v3']:>6.3f}  "
          f"{row['total_v3']:>6.3f}")

# ── 7. Console: tie analysis ──────────────────────────────────────────────────
print("\n" + "=" * 78)
print("TIE ANALYSIS (identical score within dimension)")
print("=" * 78)
for dim, col in [("D1", "d1_v3"), ("D2", "d2_v3"), ("D3", "d3_v3"), ("D4", "d4_v3")]:
    vals = df[col].round(8)
    vc   = vals.value_counts()
    tied_groups = vc[vc > 1]
    tied_models = int(tied_groups.sum())
    print(f"  {dim}: {len(tied_groups)} tie group(s), {tied_models} models involved")
    for v, c in tied_groups.items():
        ms = df.loc[vals == v, "model"].tolist()
        print(f"       score={v:.4f} ({c} models) → {ms}")

# ── 8. Console: Spearman correlations ─────────────────────────────────────────
print("\n" + "=" * 78)
print("SPEARMAN CORRELATION (ELO score vs dimension score v3, n=16)")
print("=" * 78)
for dim, col in [("D1",          "d1_v3"),
                  ("D3",          "d3_v3"),
                  ("D4",          "d4_v3"),
                  ("Total(D2제외)", "total_v3")]:
    r, p = spearmanr(df["elo"], df[col])
    sig  = " *" if p < 0.05 else ""
    print(f"  ELO vs {dim:<14}  r = {r:+.4f}   p = {p:.4f}{sig}")
print(f"  ELO vs D2             r = N/A  (전 모델 동점 — 상수 입력)")

# ── 9. Save CSV ───────────────────────────────────────────────────────────────
out_cols = ["model", "elo",
            "d1_v3", "d2_v3", "d3_v3", "d4_v3", "total_v3",
            "wt_pct", "manifold_pct", "comp_inv",
            "tri_pct", "non_tri_inv",
            "uv_bbox_pct", "uv_pack_pct",
            "pbr_pct", "texres_pct"]

# add component columns to df for CSV
df["wt_pct"]     = wt_pct
df["manifold_pct"]= manifold_pct
df["comp_inv"]   = comp_inv
df["tri_pct"]    = tri_pct
df["non_tri_inv"]= non_tri_inv
df["uv_bbox_pct"]= uv_bbox_pct
df["uv_pack_pct"]= uv_pack_pct
df["pbr_pct"]    = pbr_pct
df["texres_pct"] = texres_pct

df[out_cols].to_csv(V3_OUT_CSV, index=False, float_format="%.6f")
print(f"\nSaved: {V3_OUT_CSV}")

# ── 10. Figure 3 v3 heatmap ───────────────────────────────────────────────────
D2_COL = 1  # D2 column index in the matrix

score_matrix = df[["d1_v3", "d2_v3", "d3_v3", "d4_v3", "total_v3"]].values
col_labels   = [
    "D1: Geometry",
    "D2: Topology",
    "D3: UV/Texture",
    "D4: PBR",
    "Total\n(D1+D3+D4)",
]
row_labels   = df["model"].tolist()
n_rows, n_cols = score_matrix.shape

fig, ax = plt.subplots(figsize=(13, 9.5), dpi=300)
im = ax.imshow(score_matrix, cmap="RdBu", vmin=0, vmax=1, aspect="auto")

# ── Cell text (D2 column drawn muted) ────────────────────────────────────────
for i in range(n_rows):
    for j in range(n_cols):
        v = score_matrix[i, j]
        if j == D2_COL:
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=11, color="#aaaaaa", style="italic", zorder=5)
        else:
            txt_color = "white" if v < 0.22 or v > 0.78 else "#1f1f1f"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color, zorder=5)

# ── Gray hatched overlay on D2 column ────────────────────────────────────────
from matplotlib.patches import Rectangle
ax.add_patch(Rectangle(
    (D2_COL - 0.5, -0.5), 1.0, n_rows,
    facecolor="#cccccc", alpha=0.45, zorder=4,
))

# ── D2 annotation below the column ───────────────────────────────────────────
ax.text(D2_COL, n_rows - 0.4,
        "모든 모델 동점\n(전 모델 삼각형 출력)\n→ Total 산출 제외",
        ha="center", va="top", fontsize=8, color="#666666",
        style="italic", clip_on=False, zorder=6,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.85))

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=11)
ax.tick_params(axis="x", which="both", length=0, pad=6)
ax.tick_params(axis="y", which="both", length=0, pad=4)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()

# ── Separators ────────────────────────────────────────────────────────────────
ax.axvline(n_cols - 1 - 0.5, color="#555", linewidth=2.5, linestyle="--")
for i in range(n_rows + 1):
    ax.axhline(i - 0.5, color="white", linewidth=1.5)
for j in range(n_cols + 1):
    ax.axvline(j - 0.5, color="white", linewidth=1.5)

ax.set_title("그림 3. 모델별 차원별 점수 v3 — 16 ELO 모델 (ELO 내림차순)",
              fontsize=14, fontweight="bold", pad=22, y=1.04)

cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.03, shrink=0.65)
cbar.set_label("차원 점수 (0=최하, 1=최상)", fontsize=11)
cbar.ax.tick_params(labelsize=10)
cbar.set_ticks([0, 0.5, 1.0])
cbar.set_ticklabels(["0\n(나쁨)", "0.5", "1.0\n(좋음)"])

fig.text(
    0.5, 0.005,
    "주: 점수는 16 ELO 모델 내 percentile rank 기반 [0,1].  "
    "D1: watertight·manifold·1/components.  "
    "D2: triangle·non-triangle (전 모델 동점, Total 제외).  "
    "D3: has_uv·uv_bbox·uv_packed.  "
    "D4: has_texture·pbr_ch·tex_res.  "
    "Total = mean(D1, D3, D4).",
    ha="center", fontsize=9, color="#555",
)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(V3_OUT_FIG, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {V3_OUT_FIG}")

# Safety check: v2 still exists and is untouched
assert V2_FIG.exists(), "fig3_heatmap_v2.png was accidentally deleted!"
print(f"Verified: {V2_FIG} still intact.")
