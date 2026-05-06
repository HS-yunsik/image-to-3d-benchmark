"""
Additional analysis:
[1] texture_resolution r=-0.50 artifact check (exclude has_texture=False)
[2] Composite dimension ranks (D1/D3/D4) vs ELO rank
[3] Category-level model rankings (excluding abstract/other)
[4] 2x2 ELO-high/low × watertight-high/low matrix
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

# ── data load ────────────────────────────────────────────────────────────────
metrics = pd.read_csv("data/all_metrics.csv")
summary = pd.read_csv("data/model_summary.csv")
cat_mat = pd.read_csv("data/category_model_matrix.csv")

elo_models = summary[summary["elo"].notna()].copy()
elo_models["elo_rank"] = elo_models["elo"].rank(ascending=False).astype(int)

results = {}

# ═══════════════════════════════════════════════════════════════════════════
# [1] texture_resolution artifact check
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("[1] texture_resolution r=-0.50 원인 분석")
print("=" * 60)

# NOTE: analyze.py correlates ELO RANK (rank 1=best) vs metric.
# Here we use raw ELO score, so signs are flipped vs elo_correlation.csv.
# Original r=-0.50 (rank-based) ≡ r=+0.50 (score-based): same finding.

elo_rank = elo_models["elo"].rank(ascending=False)

# rank-based (matches elo_correlation.csv)
r_rank_all, p_rank_all = spearmanr(elo_rank, elo_models["avg_texture_resolution"])
print(f"  전체 rank-based (n={len(elo_models)}): r={r_rank_all:.4f}, p={p_rank_all:.4f}  ← elo_correlation.csv 값")

# restrict to textured models only
tex_models = elo_models[elo_models["has_texture_pct"] > 0].copy()
elo_rank_tex = tex_models["elo"].rank(ascending=False)
r_rank_tex, p_rank_tex = spearmanr(elo_rank_tex, tex_models["avg_texture_resolution"])
print(f"  has_texture>0 rank-based (n={len(tex_models)}): r={r_rank_tex:.4f}, p={p_rank_tex:.4f}")

print("\n  텍스처 보유 모델의 ELO vs resolution:")
tex_disp = tex_models[["model", "elo", "avg_texture_resolution"]].sort_values("elo", ascending=False)
for _, row in tex_disp.iterrows():
    print(f"    {row['model']:<20} ELO={row['elo']:.0f}  res={row['avg_texture_resolution']:.0f}px")

# Sign check: original r is negative (rank-based), so "genuine" if same sign in restricted set
same_sign = (r_rank_tex * r_rank_all) > 0
stronger   = abs(r_rank_tex) > abs(r_rank_all)
if same_sign and stronger:
    verdict = "진짜 발견 + 더 강해짐 (no-texture 모델이 희석했었음)"
elif same_sign:
    verdict = "진짜 발견 (no-texture 포함해도 방향 동일)"
else:
    verdict = "artifact (no-texture 모델이 방향 역전)"
print(f"\n  판정: {verdict}")
print(f"  해석: 텍스처 없는 0px 모델이 포함돼 상관계수가 약해졌던 것. "
      f"텍스처 보유 모델 내에서 r={r_rank_tex:.2f}{'(더 강함)' if stronger else ''}")

results["tex_r_all"] = r_rank_all
results["tex_r_tex_only"] = r_rank_tex

# ═══════════════════════════════════════════════════════════════════════════
# [2] Composite dimension ranks
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[2] 모델별 종합 Dimension Rank")
print("=" * 60)

df = elo_models.copy()

# D1 rank: geometry quality
# watertight↑ + manifold_edge_ratio↑ + 1/avg_components↓ combined score
df["d1_wt_norm"]    = df["watertight_pct"] / 100.0
df["d1_man_norm"]   = df["manifold_edge_ratio"]
# components: lower is better → invert with log
df["d1_cc_inv"]     = 1.0 / (np.log1p(df["avg_components"]) + 1)
# normalise each sub-score to [0,1]
for col in ["d1_wt_norm", "d1_man_norm", "d1_cc_inv"]:
    cmin, cmax = df[col].min(), df[col].max()
    df[col] = (df[col] - cmin) / (cmax - cmin + 1e-9)
df["d1_score"] = (df["d1_wt_norm"] + df["d1_man_norm"] + df["d1_cc_inv"]) / 3.0
df["d1_rank"]  = df["d1_score"].rank(ascending=False).astype(int)

# D3 rank: UV completeness
df["d3_uv_norm"]  = df["has_uv_pct"] / 100.0
df["d3_tex_norm"] = df["has_texture_pct"] / 100.0
df["d3_score"]    = (df["d3_uv_norm"] + df["d3_tex_norm"]) / 2.0
df["d3_rank"]     = df["d3_score"].rank(ascending=False).astype(int)

# D4 rank: PBR quality
df["d4_ch_norm"]  = df["avg_pbr_channels"] / df["avg_pbr_channels"].max()
# resolution only meaningful for textured models; zero stays zero
df["d4_res_norm"] = df["avg_texture_resolution"] / df["avg_texture_resolution"].max()
df["d4_score"]    = (df["d4_ch_norm"] + df["d4_res_norm"]) / 2.0
df["d4_rank"]     = df["d4_score"].rank(ascending=False).astype(int)

# ELO rank (ascending=False → rank 1 = highest ELO)
df["elo_rank"] = df["elo"].rank(ascending=False).astype(int)

disp_cols = ["model", "elo", "elo_rank", "d1_score", "d1_rank", "d3_score", "d3_rank", "d4_score", "d4_rank"]
disp = df[disp_cols].sort_values("elo_rank")

print(f"\n  {'Model':<20} {'ELO':>6} {'ELO-R':>6} {'D1-R':>6} {'D3-R':>6} {'D4-R':>6} {'ΔD1':>5} {'ΔD3':>5} {'ΔD4':>5}")
print("  " + "-" * 72)
for _, row in disp.iterrows():
    delta_d1 = int(row["elo_rank"]) - int(row["d1_rank"])
    delta_d3 = int(row["elo_rank"]) - int(row["d3_rank"])
    delta_d4 = int(row["elo_rank"]) - int(row["d4_rank"])
    print(f"  {row['model']:<20} {row['elo']:>6.0f} {int(row['elo_rank']):>6} "
          f"{int(row['d1_rank']):>6} {int(row['d3_rank']):>6} {int(row['d4_rank']):>6} "
          f"{delta_d1:>+5} {delta_d3:>+5} {delta_d4:>+5}")

# Spearman ELO vs each dim rank score
r_d1, p_d1 = spearmanr(df["elo"], df["d1_score"])
r_d3, p_d3 = spearmanr(df["elo"], df["d3_score"])
r_d4, p_d4 = spearmanr(df["elo"], df["d4_score"])
print(f"\n  ELO vs D1(Geometry): r={r_d1:.3f} p={p_d1:.3f}")
print(f"  ELO vs D3(UV/Tex):   r={r_d3:.3f} p={p_d3:.3f}")
print(f"  ELO vs D4(PBR):      r={r_d4:.3f} p={p_d4:.3f}")

# Top-3 disconnect (|elo_rank - d1_rank|)
df["disconnect_d1"] = (df["elo_rank"] - df["d1_rank"]).abs()
df["disconnect_d3"] = (df["elo_rank"] - df["d3_rank"]).abs()
df["disconnect_d4"] = (df["elo_rank"] - df["d4_rank"]).abs()
df["disconnect_total"] = df["disconnect_d1"] + df["disconnect_d3"] + df["disconnect_d4"]

print("\n  가장 disconnect 큰 모델 top 3:")
for _, row in df.nlargest(3, "disconnect_total").iterrows():
    print(f"    {row['model']:<20}  ELO-R={int(row['elo_rank'])}  D1-R={int(row['d1_rank'])}  "
          f"D3-R={int(row['d3_rank'])}  D4-R={int(row['d4_rank'])}  total-Δ={int(row['disconnect_total'])}")

# ═══════════════════════════════════════════════════════════════════════════
# [3] Category-level analysis (exclude other/abstract)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[3] 카테고리별 분석 (abstract/other 제외)")
print("=" * 60)

EXCLUDE_CATS = {"other", "abstract"}

# Get watertight columns
wt_cols = [c for c in cat_mat.columns if c.startswith("wt%_")]
cc_cols = [c for c in cat_mat.columns if c.startswith("cc_med_")]

cat_filtered = cat_mat[~cat_mat["category"].isin(EXCLUDE_CATS)].copy()

# For each category, rank models by watertight%
model_names = [c.replace("wt%_", "") for c in wt_cols]

print(f"\n  {'Category':<14} {'Best watertight model':<22} {'wt%':>5}  {'Best CC model':<22} {'cc_med':>8}")
print("  " + "-" * 75)

cat_summary_rows = []
for _, row in cat_filtered.iterrows():
    cat = row["category"]
    wt_vals  = {m: row[f"wt%_{m}"]   for m in model_names if f"wt%_{m}"  in row.index}
    cc_vals  = {m: row[f"cc_med_{m}"] for m in model_names if f"cc_med_{m}" in row.index}

    wt_vals  = {k: v for k, v in wt_vals.items()  if pd.notna(v)}
    cc_vals  = {k: v for k, v in cc_vals.items()  if pd.notna(v)}

    best_wt_model = max(wt_vals, key=wt_vals.get) if wt_vals else "-"
    best_wt_val   = wt_vals.get(best_wt_model, float("nan"))
    best_cc_model = min(cc_vals, key=cc_vals.get) if cc_vals else "-"   # lower CC = cleaner
    best_cc_val   = cc_vals.get(best_cc_model, float("nan"))

    print(f"  {cat:<14} {best_wt_model:<22} {best_wt_val:>5.1f}  {best_cc_model:<22} {best_cc_val:>8.1f}")

    # watertight ranking for this category (top 3)
    wt_sorted = sorted(wt_vals.items(), key=lambda x: x[1], reverse=True)[:3]
    cat_summary_rows.append({
        "category": cat,
        "top_wt_1": wt_sorted[0][0] if len(wt_sorted) > 0 else "",
        "top_wt_1_val": wt_sorted[0][1] if len(wt_sorted) > 0 else 0,
        "top_wt_2": wt_sorted[1][0] if len(wt_sorted) > 1 else "",
        "top_wt_2_val": wt_sorted[1][1] if len(wt_sorted) > 1 else 0,
        "top_wt_3": wt_sorted[2][0] if len(wt_sorted) > 2 else "",
        "top_wt_3_val": wt_sorted[2][1] if len(wt_sorted) > 2 else 0,
        "best_cc_model": best_cc_model,
        "best_cc_val": best_cc_val,
    })

cat_summary_df = pd.DataFrame(cat_summary_rows)

# Show which models are consistently strong across categories
print("\n  모델별 카테고리 top-1 watertight 횟수:")
top1_counts = {}
for _, row in cat_summary_df.iterrows():
    m = row["top_wt_1"]
    if m:
        top1_counts[m] = top1_counts.get(m, 0) + 1
for m, cnt in sorted(top1_counts.items(), key=lambda x: -x[1]):
    print(f"    {m:<22} {cnt} 카테고리에서 1위")

# ═══════════════════════════════════════════════════════════════════════════
# [4] 2×2 ELO-high/low × watertight-high/low matrix
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[4] 2×2 Matrix: ELO high/low × watertight high/low")
print("=" * 60)

df2 = df.copy()
elo_median = df2["elo"].median()
# Most models have wt=0%, median would be 0 → degenerate split.
# Use >0% as the watertight threshold (binary: any vs none).
wt_threshold = 0.0

print(f"\n  ELO median={elo_median:.0f}, watertight threshold: >0%")
print(f"  (median would be {df2['watertight_pct'].median():.1f}% — degenerate; using >0% binary split)")

def quad(elo, wt, em, wt_th):
    elo_hi = elo >= em
    wt_hi  = wt > wt_th
    if elo_hi and wt_hi:
        return "Q1: ELO-High / WT-High (>0%)"
    elif elo_hi and not wt_hi:
        return "Q2: ELO-High / WT-None (=0%)"
    elif not elo_hi and wt_hi:
        return "Q3: ELO-Low  / WT-High (>0%)"
    else:
        return "Q4: ELO-Low  / WT-None (=0%)"

df2["quadrant"] = df2.apply(lambda r: quad(r["elo"], r["watertight_pct"], elo_median, wt_threshold), axis=1)

for q in ["Q1: ELO-High / WT-High (>0%)", "Q2: ELO-High / WT-None (=0%)",
          "Q3: ELO-Low  / WT-High (>0%)", "Q4: ELO-Low  / WT-None (=0%)"]:
    models_in_q = df2[df2["quadrant"] == q][["model", "elo", "watertight_pct"]].sort_values("elo", ascending=False)
    print(f"\n  {q} (n={len(models_in_q)}):")
    for _, row in models_in_q.iterrows():
        print(f"    {row['model']:<20}  ELO={row['elo']:.0f}  wt={row['watertight_pct']:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════
# Save to CSV (ELO models only — used for correlation analysis)
# ═══════════════════════════════════════════════════════════════════════════
out_cols = ["model", "elo", "elo_rank",
            "d1_score", "d1_rank", "d3_score", "d3_rank", "d4_score", "d4_rank",
            "disconnect_d1", "disconnect_d3", "disconnect_d4", "disconnect_total",
            "quadrant",
            "watertight_pct", "manifold_edge_ratio", "avg_components",
            "has_uv_pct", "has_texture_pct", "avg_pbr_channels", "avg_texture_resolution"]
df2[out_cols].to_csv("data/additional_analysis.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# [5] Full-model ranks (incl. post-paper) — used for Figure 3 heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("[5] 전체 모델 dimension rank (post-paper 포함)")
print("=" * 60)

full = summary.copy()

# D1 score
full["d1_wt_norm"]  = full["watertight_pct"] / 100.0
full["d1_man_norm"] = full["manifold_edge_ratio"]
full["d1_cc_inv"]   = 1.0 / (np.log1p(full["avg_components"]) + 1)
for col in ["d1_wt_norm", "d1_man_norm", "d1_cc_inv"]:
    cmin, cmax = full[col].min(), full[col].max()
    full[col] = (full[col] - cmin) / (cmax - cmin + 1e-9)
full["d1_score"] = (full["d1_wt_norm"] + full["d1_man_norm"] + full["d1_cc_inv"]) / 3.0
full["d1_rank"]  = full["d1_score"].rank(ascending=False).astype(int)

# D3 score
full["d3_uv_norm"]  = full["has_uv_pct"] / 100.0
full["d3_tex_norm"] = full["has_texture_pct"] / 100.0
full["d3_score"]    = (full["d3_uv_norm"] + full["d3_tex_norm"]) / 2.0
full["d3_rank"]     = full["d3_score"].rank(ascending=False).astype(int)

# D4 score
d4_max_ch  = full["avg_pbr_channels"].max() or 1
d4_max_res = full["avg_texture_resolution"].max() or 1
full["d4_ch_norm"]  = full["avg_pbr_channels"] / d4_max_ch
full["d4_res_norm"] = full["avg_texture_resolution"] / d4_max_res
full["d4_score"]    = (full["d4_ch_norm"] + full["d4_res_norm"]) / 2.0
full["d4_rank"]     = full["d4_score"].rank(ascending=False).astype(int)

# ELO rank — NaN for post-paper models
full["elo_rank"] = full["elo"].rank(ascending=False)

full_out = full[["model", "elo", "elo_rank",
                  "d1_score", "d1_rank", "d3_score", "d3_rank",
                  "d4_score", "d4_rank",
                  "watertight_pct", "manifold_edge_ratio", "avg_components",
                  "has_uv_pct", "has_texture_pct", "avg_pbr_channels",
                  "avg_texture_resolution", "post_paper"]]
full_out = full_out.sort_values(["elo_rank", "d1_rank"], na_position="last")
full_out.to_csv("data/full_model_ranks.csv", index=False)

print(f"\n  전체 {len(full)} 모델 (ELO {full['elo'].notna().sum()} + post-paper {full['elo'].isna().sum()})")
print(f"  {'Model':<20} {'ELO':>6} {'ELO-R':>6} {'D1-R':>6} {'D3-R':>6} {'D4-R':>6}")
print("  " + "-" * 60)
for _, row in full_out.iterrows():
    elo_str   = f"{row['elo']:.0f}" if pd.notna(row["elo"]) else "  N/A"
    elo_r_str = f"{int(row['elo_rank'])}" if pd.notna(row["elo_rank"]) else "N/A"
    print(f"  {row['model']:<20} {elo_str:>6} {elo_r_str:>6} "
          f"{int(row['d1_rank']):>6} {int(row['d3_rank']):>6} {int(row['d4_rank']):>6}")

print("\n" + "=" * 60)
print("저장: data/additional_analysis.csv (ELO 모델만)")
print("저장: data/full_model_ranks.csv (전체 모델, post-paper 포함)")
print("=" * 60)
