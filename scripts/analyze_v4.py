"""
v4 dimension system — Steps 2-6 (all-in-one)

Change from v3:  D2 Topology removed, D3→D2(UV), D4→D3(PBR)
Numerical values: identical to v3 (pure relabeling)

Outputs:
  data/dimension_scores_v4.csv
  data/elo_correlation_v4.csv
  outputs/fig2_heatmap_v4.png
  outputs/fig3_correlation_v4.png
  outputs/fig4_matrix_v4.png
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
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

rcParams["font.family"]        = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"]       = 42

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

COL_POS = "#0E9594"
COL_NEG = "#E07A5F"
COL_Q1  = "#0E9594"
COL_Q2  = "#E07A5F"
COL_Q3  = "#3D5A80"
COL_Q4  = "#888888"


# ── Step 2: Compute v4 scores ─────────────────────────────────────────────────

def pct_rank(series: pd.Series, n: int, lower_is_better: bool = False) -> np.ndarray:
    vals  = series.values.astype(float)
    vals  = np.where(np.isnan(vals), 0.0, vals)
    ranks = rankdata(vals, method="average")
    pct   = (ranks - 1) / (n - 1)
    return 1 - pct if lower_is_better else pct


def build_scores() -> pd.DataFrame:
    summary = pd.read_csv(DATA / "model_summary.csv")
    elo16   = summary[summary["elo"].notna()].copy().reset_index(drop=True)
    assert len(elo16) == 16, f"Expected 16 ELO models, got {len(elo16)}"
    models_set = set(elo16["model"])

    am     = pd.read_csv(DATA / "all_metrics.csv")
    am_elo = am[am["model"].isin(models_set)]

    agg = (
        am_elo
        .groupby("model", as_index=False)
        .agg(
            avg_uv_bbox_efficiency=("uv_bbox_efficiency", "mean"),
            avg_uv_packed_area    =("uv_packed_area",     "mean"),
        )
    )

    df = elo16.merge(agg, on="model", how="left")
    df = df.sort_values("elo", ascending=False).reset_index(drop=True)
    n  = len(df)

    # D1 Geometry (unchanged from v3)
    wt_pct       = pct_rank(df["watertight_pct"],         n)
    manifold_pct = pct_rank(df["manifold_edge_ratio"],    n)
    comp_inv     = pct_rank(df["avg_components"],         n, lower_is_better=True)

    # D2 UV  (was D3 in v3)
    has_uv_prop = (df["has_uv_pct"] / 100).values
    uv_bbox_pct = pct_rank(df["avg_uv_bbox_efficiency"],  n)
    uv_pack_pct = pct_rank(df["avg_uv_packed_area"],      n)

    # D3 PBR  (was D4 in v3)
    has_tex_prop = (df["has_texture_pct"] / 100).values
    pbr_pct      = pct_rank(df["avg_pbr_channels"],       n)
    texres_pct   = pct_rank(df["avg_texture_resolution"], n)

    d1    = np.column_stack([wt_pct, manifold_pct, comp_inv]).mean(axis=1)
    d2    = np.column_stack([has_uv_prop, uv_bbox_pct, uv_pack_pct]).mean(axis=1)
    d3    = np.column_stack([has_tex_prop, pbr_pct, texres_pct]).mean(axis=1)
    total = np.column_stack([d1, d2, d3]).mean(axis=1)

    df["D1"]      = d1
    df["D2_UV"]   = d2
    df["D3_PBR"]  = d3
    df["Total"]   = total
    df["wt_pct"]      = wt_pct
    df["manifold_pct"]= manifold_pct
    df["comp_inv"]    = comp_inv
    df["uv_bbox_pct"] = uv_bbox_pct
    df["uv_pack_pct"] = uv_pack_pct
    df["pbr_pct"]     = pbr_pct
    df["texres_pct"]  = texres_pct

    out_cols = ["model", "elo", "D1", "D2_UV", "D3_PBR", "Total",
                "wt_pct", "manifold_pct", "comp_inv",
                "uv_bbox_pct", "uv_pack_pct", "pbr_pct", "texres_pct"]
    df[out_cols].to_csv(DATA / "dimension_scores_v4.csv", index=False, float_format="%.6f")
    print(f"[Step 2] Saved: data/dimension_scores_v4.csv")

    # Console table
    print("\n" + "=" * 72)
    print("DIMENSION SCORES v4  (16 ELO models, ELO desc)")
    print(f"{'Model':<18} {'ELO':>5}  {'D1':>6}  {'D2_UV':>6}  {'D3_PBR':>7}  {'Total':>6}")
    print("-" * 72)
    for _, row in df.iterrows():
        print(f"{row['model']:<18} {int(row['elo']):>5}  "
              f"{row['D1']:>6.3f}  {row['D2_UV']:>6.3f}  "
              f"{row['D3_PBR']:>7.3f}  {row['Total']:>6.3f}")

    return df


# ── Step 3: Spearman correlations ─────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> dict:
    rows = []
    metrics_out = {}

    for label, col in [
        ("D1 Geometry", "D1"),
        ("D2 UV",       "D2_UV"),
        ("D3 PBR",      "D3_PBR"),
        ("Total",       "Total"),
    ]:
        r, p = spearmanr(df["elo"], df[col])
        sig  = " *" if p < 0.05 else ""
        print(f"  ELO vs {label:<16}  r = {r:+.4f}   p = {p:.4f}{sig}")
        rows.append({"metric": label, "r": r, "p": p, "n": len(df)})
        metrics_out[label] = (r, p, len(df))

    # texture_resolution subset (textured models only)
    tex  = df[df["has_texture_pct"] > 0] if "has_texture_pct" in df.columns else df
    r_t, p_t = spearmanr(tex["elo"], tex["avg_texture_resolution"])
    n_t = len(tex)
    print(f"  ELO vs texture_res (n={n_t}): r = {r_t:+.4f}  p = {p_t:.4f}")
    rows.append({"metric": "texture_resolution_texonly", "r": r_t, "p": p_t, "n": n_t})
    metrics_out["texture_resolution_texonly"] = (r_t, p_t, n_t)

    pd.DataFrame(rows).to_csv(DATA / "elo_correlation_v4.csv", index=False, float_format="%.6f")
    print(f"[Step 3] Saved: data/elo_correlation_v4.csv")
    return metrics_out


# ── Step 4: Quadrant verification ────────────────────────────────────────────

def verify_quadrant():
    add = pd.read_csv(DATA / "additional_analysis.csv")
    print("\n[Step 4] Quadrant distribution (v4 == v2, same source):")
    vc = add["quadrant"].value_counts()
    for q, cnt in vc.items():
        short = q.split(":")[0]
        models = add.loc[add["quadrant"] == q, "model"].tolist()
        print(f"  {short}: n={cnt}  {models}")
    return add


# ── Step 5a: fig2_heatmap_v4.png ─────────────────────────────────────────────

def make_heatmap(df: pd.DataFrame):
    score_matrix = df[["D1", "D2_UV", "D3_PBR", "Total"]].values
    col_labels   = ["D1: Geometry", "D2: UV/Texture", "D3: PBR", "Total\n(D1+D2+D3)"]
    row_labels   = df["model"].tolist()
    n_rows, n_cols = score_matrix.shape

    fig, ax = plt.subplots(figsize=(11, 9.5), dpi=300)
    im = ax.imshow(score_matrix, cmap="RdBu", vmin=0, vmax=1, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = score_matrix[i, j]
            txt_color = "white" if v < 0.22 or v > 0.78 else "#1f1f1f"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=12, fontweight="bold", color=txt_color, zorder=5)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.tick_params(axis="x", which="both", length=0, pad=6)
    ax.tick_params(axis="y", which="both", length=0, pad=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Total column separator
    ax.axvline(n_cols - 1 - 0.5, color="#555", linewidth=2.5, linestyle="--")
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_title("그림 2. 모델별 차원별 점수 v4 — 16 ELO 모델 (ELO 내림차순)",
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
        "D2: has_uv·uv_bbox·uv_packed.  "
        "D3: has_texture·pbr_ch·tex_res.  "
        "Total = mean(D1, D2, D3).",
        ha="center", fontsize=9, color="#555",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "fig2_heatmap_v4.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5a] Saved: {out}")


# ── Step 5b: fig3_correlation_v4.png ─────────────────────────────────────────

def make_correlation_bar(df: pd.DataFrame, metrics: dict):
    # Use same data sources as update_v2 but with v4 labels
    add = pd.read_csv(DATA / "additional_analysis.csv")

    # Recompute from v4 scores for D1/D2/D3
    r_d1, p_d1 = spearmanr(df["elo"], df["D1"])
    r_d2, p_d2 = spearmanr(df["elo"], df["D2_UV"])
    r_d3, p_d3 = spearmanr(df["elo"], df["D3_PBR"])
    r_tot, p_tot = spearmanr(df["elo"], df["Total"])

    # texture_resolution subset
    tex = add[add["has_texture_pct"] > 0]
    r_t, p_t = spearmanr(tex["elo"], tex["avg_texture_resolution"])
    n_t = len(tex)

    items = [
        ("D1\n(Geometry)",        r_d1,  p_d1,  16),
        ("D2\n(UV/Texture)",      r_d2,  p_d2,  16),
        ("D3\n(PBR)",             r_d3,  p_d3,  16),
        ("Total\n(D1+D2+D3)",     r_tot, p_tot, 16),
        (f"Texture Res.\n(텍스처 보유, n={n_t})", r_t, p_t, n_t),
    ]
    labels = [it[0] for it in items]
    rs     = [it[1] for it in items]
    ps     = [it[2] for it in items]
    ns     = [it[3] for it in items]
    colors = [COL_POS if r > 0 else COL_NEG for r in rs]

    fig, ax = plt.subplots(figsize=(13, 7.2), dpi=300)
    bars = ax.bar(labels, rs, color=colors, edgecolor="white",
                  linewidth=2.0, width=0.62, zorder=3)

    for bar, r, p, n in zip(bars, rs, ps, ns):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.03 if h >= 0 else -0.03),
                f"r = {r:+.3f}",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=13, fontweight="bold", color="#1f1f1f")
        sig_marker = " *" if p < 0.05 else ""
        ptext = f"p = {p:.3f}{sig_marker}\n(n = {n})"
        ax.text(bar.get_x() + bar.get_width() / 2,
                h * 0.5 if abs(h) > 0.18 else (h + (0.13 if h >= 0 else -0.13)),
                ptext,
                ha="center", va="center",
                fontsize=10.5,
                color="white" if abs(h) > 0.18 else "#444",
                fontweight="bold" if abs(h) > 0.18 else "normal")

    ax.axhline(0, color="black", linewidth=1.0, zorder=2)
    ax.axhspan(-0.1, 0.1, alpha=0.10, color="gray", zorder=1)
    ax.text(4.45, 0, "  무상관\n  (|r|<0.1)", ha="left", va="center",
            fontsize=9, color="gray", style="italic")

    ax.set_ylabel("Spearman 순위 상관계수  r", fontsize=13, fontweight="bold")
    ax.set_xlabel("자동 메트릭 차원", fontsize=13, fontweight="bold")
    ax.set_title("그림 3. ELO score 와 메트릭 차원별 Spearman 상관계수 — v4 (16 모델)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(-0.45, 0.95)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.01,
             "주: D1·D2·D3·Total 은 16 ELO 모델 종합 차원 점수 기반. "
             "Texture Res.는 has_texture>0 모델만. * = p<0.05.",
             ha="center", fontsize=10, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT_DIR / "fig3_correlation_v4.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5b] Saved: {out}")


# ── Step 5c: fig4_matrix_v4.png ──────────────────────────────────────────────

def make_matrix(add: pd.DataFrame):
    quad_label_map = {
        "Q1: ELO-High / WT-High (>0%)": "Q1",
        "Q2: ELO-High / WT-None (=0%)": "Q2",
        "Q3: ELO-Low  / WT-High (>0%)": "Q3",
        "Q4: ELO-Low  / WT-None (=0%)": "Q4",
    }
    quad_config = {
        "Q1": {"title": "Q1: 시각+Production", "sub": "ELO↑  ·  WT>0",
               "color": COL_Q1, "pos": (0, 0)},
        "Q2": {"title": "Q2: 시각형",           "sub": "ELO↑  ·  WT=0",
               "color": COL_Q2, "pos": (0, 1)},
        "Q3": {"title": "Q3: Production만",     "sub": "ELO↓  ·  WT>0",
               "color": COL_Q3, "pos": (1, 0)},
        "Q4": {"title": "Q4: 미흡",             "sub": "ELO↓  ·  WT=0",
               "color": COL_Q4, "pos": (1, 1)},
    }

    quad_models = {q: [] for q in quad_config}
    for _, row in add.sort_values("elo", ascending=False).iterrows():
        q = quad_label_map.get(row["quadrant"])
        if q:
            quad_models[q].append((row["model"], row["elo"], row["watertight_pct"]))

    n_total      = sum(len(v) for v in quad_models.values())
    largest_quad = max(len(v) for v in quad_models.values())
    elo_med      = add["elo"].median()

    fig = plt.figure(figsize=(13.5, 10.5), dpi=300, facecolor="white")
    gs  = GridSpec(2, 2, left=0.11, right=0.97, top=0.86, bottom=0.05,
                   wspace=0.07, hspace=0.10)

    for q, cfg in quad_config.items():
        r, c = cfg["pos"]
        ax = fig.add_subplot(gs[r, c])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        col    = cfg["color"]
        models = quad_models[q]
        n      = len(models)

        card = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                              boxstyle="round,pad=0.0,rounding_size=0.035",
                              facecolor=col, alpha=0.10,
                              edgecolor=col, linewidth=2.5,
                              transform=ax.transAxes, zorder=1)
        ax.add_patch(card)

        ax.text(0.06, 0.94, cfg["title"],
                ha="left", va="top", fontsize=19, fontweight="bold",
                color=col, zorder=3)
        ax.text(0.06, 0.86, cfg["sub"],
                ha="left", va="top", fontsize=12.5,
                color=col, style="italic", zorder=3)
        ax.text(0.94, 0.92, f"n = {n}",
                ha="right", va="center", fontsize=14, fontweight="bold",
                color="white",
                bbox=dict(facecolor=col, edgecolor="none",
                          boxstyle="round,pad=0.45", alpha=0.95),
                zorder=4)

        if q == "Q2" and n == largest_quad:
            ax.text(0.94, 0.78, "★ 최대 클러스터",
                    ha="right", va="center", fontsize=10.5, fontweight="bold",
                    color=col, style="italic", zorder=4)

        ax.plot([0.06, 0.94], [0.78, 0.78], color=col, alpha=0.45,
                linewidth=1.2, zorder=2)

        if n == 0:
            ax.text(0.5, 0.42, "(해당 모델 없음)",
                    ha="center", va="center", fontsize=12, color="#999",
                    style="italic", zorder=3)
        else:
            top_y = 0.72; bot_y = 0.07
            line_pair = (top_y - bot_y) / max(n, 1)
            for i, (name, elo, wt) in enumerate(models):
                y_top = top_y - i * line_pair
                ax.text(0.10, y_top, "●",
                        ha="left", va="top", fontsize=12, color=col, zorder=3)
                ax.text(0.16, y_top, name,
                        ha="left", va="top", fontsize=13.5, fontweight="bold",
                        color="#1c1c1c", zorder=3)
                stats = f"ELO {elo:.0f}   ·   WT {wt:.0f}%"
                ax.text(0.16, y_top - line_pair * 0.42, stats,
                        ha="left", va="top", fontsize=11,
                        color="#555", zorder=3)

    col_y = 0.905
    fig.text(0.32, col_y, "Watertight > 0%",
             ha="center", va="center", fontsize=15.5, fontweight="bold", color="#222")
    fig.text(0.73, col_y, "Watertight = 0%",
             ha="center", va="center", fontsize=15.5, fontweight="bold", color="#222")
    fig.text(0.045, 0.665, f"ELO ↑\n(>= {elo_med:.0f})",
             ha="center", va="center", fontsize=14, fontweight="bold", color="#222")
    fig.text(0.045, 0.255, f"ELO ↓\n(< {elo_med:.0f})",
             ha="center", va="center", fontsize=14, fontweight="bold", color="#222")
    fig.text(0.54, 0.965,
             "그림 4. ELO x Watertight 4분면 — 인기 모델 대부분이 production 부적합",
             ha="center", va="center", fontsize=16.5, fontweight="bold", color="#1f1f1f")
    fig.text(0.54, 0.015,
             f"주: ELO 중앙값 {elo_med:.0f} 기준 분할. Watertight >0% 임계. "
             f"전체 {n_total}개 모델(post-paper 제외).",
             ha="center", fontsize=10, color="#666")

    out = OUT_DIR / "fig4_matrix_v4.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5c] Saved: {out}  ({n_total} models, Q2={len(quad_models['Q2'])})")


# ── Step 6: Summary ───────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 60)
    print("v4 UPDATE SUMMARY")
    print("=" * 60)
    print("  D2 Topology: REMOVED (all models tied — 변별력 없음)")
    print("  D3 UV       -> D2 UV  (번호 변경, 수치 동일)")
    print("  D4 PBR      -> D3 PBR (번호 변경, 수치 동일)")
    print("  Total = mean(D1, D2_UV, D3_PBR)  (동일 공식)")
    print()
    print("  data/dimension_scores_v4.csv  ✓")
    print("  data/elo_correlation_v4.csv   ✓")
    print("  outputs/fig2_heatmap_v4.png   ✓")
    print("  outputs/fig3_correlation_v4.png ✓")
    print("  outputs/fig4_matrix_v4.png    ✓")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Building v4 dimension scores ...")
    print("=" * 60)
    df = build_scores()

    print("\n" + "=" * 60)
    print("Step 3: Spearman correlations ...")
    print("=" * 60)
    metrics = compute_correlations(df)

    print("\n" + "=" * 60)
    print("Step 4: Quadrant verification ...")
    print("=" * 60)
    add = verify_quadrant()

    print("\n" + "=" * 60)
    print("Step 5a: Heatmap v4 ...")
    print("=" * 60)
    make_heatmap(df)

    print("\n" + "=" * 60)
    print("Step 5b: Correlation bar chart v4 ...")
    print("=" * 60)
    make_correlation_bar(df, metrics)

    print("\n" + "=" * 60)
    print("Step 5c: 2x2 Matrix v4 ...")
    print("=" * 60)
    make_matrix(add)

    print_summary()
