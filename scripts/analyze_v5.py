"""
v5 dimension system — Steps 2-6 (all-in-one)

Changes from v4:
  - uv_bbox_efficiency removed (model avg range 0.046, no discriminative power)
  - has_texture removed (100% identical to has_uv, redundant)

Final metric composition (3 dims, 7 metrics):
  D1 Geometry: watertight, manifold_edge_ratio, connected_components
  D2 UV:       has_uv, uv_packed_area
  D3 PBR:      pbr_channel_count, texture_resolution
  Total = mean(D1, D2, D3)

Outputs:
  data/dimension_scores_v5.csv
  data/elo_correlation_v5.csv
  outputs/fig2_heatmap_v5.png
  outputs/fig3_correlation_v5.png
  outputs/fig4_matrix_v5.png  (or copy of v4 if quadrant unchanged)
"""
from __future__ import annotations

import shutil
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
from matplotlib.lines import Line2D

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

V4_CORR = {"D2_UV": 0.4194, "D3_PBR": 0.4154, "Total": 0.6500}


# ── helpers ───────────────────────────────────────────────────────────────────

def pct_rank(series: pd.Series, n: int, lower_is_better: bool = False) -> np.ndarray:
    vals  = series.values.astype(float)
    vals  = np.where(np.isnan(vals), 0.0, vals)
    ranks = rankdata(vals, method="average")
    pct   = (ranks - 1) / (n - 1)
    return 1 - pct if lower_is_better else pct


# ── Step 2: Dimension scores v5 ───────────────────────────────────────────────

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
        .agg(avg_uv_packed_area=("uv_packed_area", "mean"))
    )

    df = elo16.merge(agg, on="model", how="left")
    df = df.sort_values("elo", ascending=False).reset_index(drop=True)
    n  = len(df)

    # D1 Geometry
    wt_pct       = pct_rank(df["watertight_pct"],      n)
    manifold_pct = pct_rank(df["manifold_edge_ratio"], n)
    comp_inv     = pct_rank(df["avg_components"],      n, lower_is_better=True)

    # D2 UV  (has_uv + uv_packed_area only)
    has_uv_prop = (df["has_uv_pct"] / 100).values
    uv_pack_pct = pct_rank(df["avg_uv_packed_area"],   n)

    # D3 PBR  (pbr_channel_count + texture_resolution only)
    pbr_pct     = pct_rank(df["avg_pbr_channels"],        n)
    texres_pct  = pct_rank(df["avg_texture_resolution"],  n)

    d1    = np.column_stack([wt_pct, manifold_pct, comp_inv]).mean(axis=1)
    d2    = np.column_stack([has_uv_prop, uv_pack_pct]).mean(axis=1)
    d3    = np.column_stack([pbr_pct, texres_pct]).mean(axis=1)
    total = np.column_stack([d1, d2, d3]).mean(axis=1)

    df["D1"]    = d1
    df["D2"]    = d2
    df["D3"]    = d3
    df["Total"] = total

    # component columns for CSV
    df["wt_pct"]       = wt_pct
    df["manifold_pct"] = manifold_pct
    df["comp_inv"]     = comp_inv
    df["has_uv_prop"]  = has_uv_prop
    df["uv_pack_pct"]  = uv_pack_pct
    df["pbr_pct"]      = pbr_pct
    df["texres_pct"]   = texres_pct

    out_cols = ["model", "elo", "D1", "D2", "D3", "Total",
                "wt_pct", "manifold_pct", "comp_inv",
                "has_uv_prop", "uv_pack_pct",
                "pbr_pct", "texres_pct"]
    df[out_cols].to_csv(DATA / "dimension_scores_v5.csv", index=False, float_format="%.6f")
    print("[Step 2] Saved: data/dimension_scores_v5.csv")

    print("\n" + "=" * 68)
    print("DIMENSION SCORES v5  (16 ELO models, ELO desc)")
    print(f"{'Model':<18} {'ELO':>5}  {'D1':>6}  {'D2':>6}  {'D3':>6}  {'Total':>6}")
    print("-" * 68)
    for _, r in df.iterrows():
        print(f"{r['model']:<18} {int(r['elo']):>5}  "
              f"{r['D1']:>6.3f}  {r['D2']:>6.3f}  {r['D3']:>6.3f}  {r['Total']:>6.3f}")

    return df


# ── Step 3: Spearman correlations ─────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> dict:
    # avg_texture_resolution and has_texture_pct already in df via model_summary.csv
    df2 = df.copy()

    rows = []
    metrics_out = {}

    print("\n=== Spearman Correlation Results (v5) ===")
    for label, col in [
        ("D1 Geometry",      "D1"),
        ("D2 UV",            "D2"),
        ("D3 PBR",           "D3"),
        ("Total (D1+D2+D3)", "Total"),
    ]:
        r, p = spearmanr(df2["elo"], df2[col])
        sig  = "유의 *" if p < 0.05 else "무의"
        print(f"  {label:<20}  r = {r:+.3f}  p = {p:.3f}  (n=16)  [{sig}]")
        rows.append({"metric": label, "r": r, "p_value": p, "n": 16,
                     "significant": p < 0.05})
        metrics_out[label] = (r, p, 16)

    # texture_resolution n=16
    r_t16, p_t16 = spearmanr(df2["elo"], df2["avg_texture_resolution"])
    sig16 = "유의 *" if p_t16 < 0.05 else "무의"
    print(f"  {'texture_resolution':<20}  r = {r_t16:+.3f}  p = {p_t16:.3f}  (n=16)  [{sig16}]")
    rows.append({"metric": "texture_resolution", "r": r_t16, "p_value": p_t16,
                 "n": 16, "significant": p_t16 < 0.05})
    metrics_out["texture_resolution"] = (r_t16, p_t16, 16)

    # texture_resolution n=9 (texture-bearing models only)
    tex9 = df2[df2["has_texture_pct"] > 0]
    r_t9, p_t9 = spearmanr(tex9["elo"], tex9["avg_texture_resolution"])
    n9   = len(tex9)
    sig9 = "유의 *" if p_t9 < 0.05 else "무의"
    print(f"  {'tex_res (n=9)':<20}  r = {r_t9:+.3f}  p = {p_t9:.3f}  (n={n9})   [{sig9}]")
    rows.append({"metric": "tex_res_n9", "r": r_t9, "p_value": p_t9,
                 "n": n9, "significant": p_t9 < 0.05})
    metrics_out["tex_res_n9"] = (r_t9, p_t9, n9)

    # v4 comparison
    r_d2  = metrics_out["D2 UV"][0]
    r_d3  = metrics_out["D3 PBR"][0]
    r_tot = metrics_out["Total (D1+D2+D3)"][0]
    print("\n=== v4 대비 변화 ===")
    print(f"  D2 UV:   v4 r=+{V4_CORR['D2_UV']:.3f} -> v5 r={r_d2:+.3f}  "
          f"(변화: {r_d2 - V4_CORR['D2_UV']:+.3f})")
    print(f"  D3 PBR:  v4 r=+{V4_CORR['D3_PBR']:.3f} -> v5 r={r_d3:+.3f}  "
          f"(변화: {r_d3 - V4_CORR['D3_PBR']:+.3f})")
    print(f"  Total:   v4 r=+{V4_CORR['Total']:.3f} -> v5 r={r_tot:+.3f}  "
          f"(변화: {r_tot - V4_CORR['Total']:+.3f})")

    # Flag large changes
    for name, v4r, v5r in [("D2 UV", V4_CORR["D2_UV"], r_d2),
                             ("D3 PBR", V4_CORR["D3_PBR"], r_d3),
                             ("Total",  V4_CORR["Total"],  r_tot)]:
        if abs(v5r - v4r) > 0.05:
            print(f"  *** 주의: {name} |Δr|={abs(v5r-v4r):.3f} > 0.05 ***")

    pd.DataFrame(rows).to_csv(DATA / "elo_correlation_v5.csv",
                               index=False, float_format="%.6f")
    print(f"\n[Step 3] Saved: data/elo_correlation_v5.csv")
    return metrics_out


# ── Step 4: Quadrant verification ────────────────────────────────────────────

def verify_quadrant(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    add = pd.read_csv(DATA / "additional_analysis.csv")

    print("\n=== Step 4: 2x2 Quadrant (v5 == v4 확인) ===")
    vc = add["quadrant"].value_counts()
    changed = False
    v4_counts = {"Q1": 2, "Q2": 6, "Q3": 2, "Q4": 6}
    for q_full, cnt in vc.items():
        short = q_full.split(":")[0]
        models = add.loc[add["quadrant"] == q_full, "model"].tolist()
        match  = "(v4와 동일)" if v4_counts.get(short) == cnt else f"*** 변경! v4={v4_counts.get(short)} ***"
        print(f"  {short}: n={cnt}  {match}")
        print(f"         {models}")
        if v4_counts.get(short) != cnt:
            changed = True

    if not changed:
        print("  -> 분면 분류 v4와 완전 동일 ✓")
    return add, changed


# ── Step 5a: fig2_heatmap_v5.png ─────────────────────────────────────────────

def make_heatmap(df: pd.DataFrame):
    score_matrix = df[["D1", "D2", "D3", "Total"]].values
    col_labels   = [
        "D1: Geometry\n(watertight, manifold,\nconn_comp)",
        "D2: UV\n(has_uv,\nuv_packed)",
        "D3: PBR\n(pbr_ch,\ntex_res)",
        "Total\n(D1+D2+D3)",
    ]
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
    ax.set_xticklabels(col_labels, fontsize=10, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.tick_params(axis="x", which="both", length=0, pad=8)
    ax.tick_params(axis="y", which="both", length=0, pad=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    ax.axvline(n_cols - 1 - 0.5, color="#555", linewidth=2.5, linestyle="--")
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    ax.set_title("그림 2. 모델별 차원별 점수 v5 — 16 ELO 모델 (ELO 내림차순)\n"
                 "(7개 메트릭: uv_bbox_efficiency · has_texture 제거)",
                 fontsize=13, fontweight="bold", pad=28, y=1.06)

    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.03, shrink=0.65)
    cbar.set_label("차원 점수 (0=최하, 1=최상)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0\n(나쁨)", "0.5", "1.0\n(좋음)"])

    fig.text(
        0.5, 0.005,
        "주: percentile rank [0,1].  "
        "D1=mean(wt,manifold,1/cc).  "
        "D2=mean(has_uv,uv_packed).  "
        "D3=mean(pbr_ch,tex_res).  "
        "Total=mean(D1,D2,D3).",
        ha="center", fontsize=9, color="#555",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = OUT_DIR / "fig2_heatmap_v5.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5a] Saved: {out}")


# ── Step 5b: fig3_correlation_v5.png ─────────────────────────────────────────

def make_correlation_bar(df: pd.DataFrame, metrics: dict):
    df2 = df.copy()

    tex9 = df2[df2["has_texture_pct"] > 0]
    r_t9, p_t9 = spearmanr(tex9["elo"], tex9["avg_texture_resolution"])
    n9 = len(tex9)

    df2["Total_D2D3"] = (df2["D2"] + df2["D3"]) / 2
    r_d2d3, p_d2d3 = spearmanr(df2["elo"], df2["Total_D2D3"])

    items = [
        ("D1\n(Geometry)",      *metrics["D1 Geometry"]),
        ("D2\n(UV)",            *metrics["D2 UV"]),
        ("D3\n(PBR)",           *metrics["D3 PBR"]),
        ("Total\n(D2+D3)",      r_d2d3, p_d2d3, 16),
        ("Total\n(D1+D2+D3)",   *metrics["Total (D1+D2+D3)"]),
        (f"Texture Res.\n(n={n9})", r_t9, p_t9, n9),
    ]
    labels = [it[0] for it in items]
    rs     = [it[1] for it in items]
    ps     = [it[2] for it in items]
    ns     = [it[3] for it in items]
    colors = [COL_POS if r > 0 else COL_NEG for r in rs]

    fig, ax = plt.subplots(figsize=(15, 7.2), dpi=300)
    bars = ax.bar(labels, rs, color=colors, edgecolor="white",
                  linewidth=2.0, width=0.62, zorder=3)

    for bar, r, p, n in zip(bars, rs, ps, ns):
        h = bar.get_height()
        # r value + star
        star = " ★" if p < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                h + (0.03 if h >= 0 else -0.03),
                f"r = {r:+.3f}{star}",
                ha="center", va="bottom" if h >= 0 else "top",
                fontsize=13, fontweight="bold", color="#1f1f1f")
        # p-value
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
    ax.text(5.45, 0, "  무상관\n  (|r|<0.1)", ha="left", va="center",
            fontsize=9, color="gray", style="italic")

    ax.set_ylabel("Spearman 순위 상관계수  r", fontsize=13, fontweight="bold")
    ax.set_xlabel("자동 메트릭 차원", fontsize=13, fontweight="bold")
    ax.set_title("그림 3. ELO × 메트릭 Spearman 상관계수 — v5 (7개 메트릭, 16 모델)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(-0.45, 0.95)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.01,
             "주: D1·D2·D3·Total = 16 ELO 모델 종합 점수 기반. "
             "Texture Res. = has_texture>0 모델만. ★ = p<0.05.",
             ha="center", fontsize=10, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = OUT_DIR / "fig3_correlation_v5.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5b] Saved: {out}")


# ── Step 5c: fig4_matrix_v5.png ──────────────────────────────────────────────

def make_matrix(add: pd.DataFrame, changed: bool):
    src = OUT_DIR / "fig4_matrix_v4.png"
    dst = OUT_DIR / "fig4_matrix_v5.png"

    if not changed and src.exists():
        shutil.copy2(src, dst)
        print(f"[Step 5c] 분면 동일 → fig4_matrix_v4.png 복사: {dst}")
        return

    # Regenerate if changed
    print("[Step 5c] 분면 변경됨 → fig4_matrix_v5.png 재생성")
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
        r, c   = cfg["pos"]
        ax     = fig.add_subplot(gs[r, c])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        col    = cfg["color"]
        models = quad_models[q]
        n      = len(models)

        ax.add_patch(FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.0,rounding_size=0.035",
            facecolor=col, alpha=0.10, edgecolor=col, linewidth=2.5,
            transform=ax.transAxes, zorder=1))
        ax.text(0.06, 0.94, cfg["title"], ha="left", va="top",
                fontsize=19, fontweight="bold", color=col, zorder=3)
        ax.text(0.06, 0.86, cfg["sub"], ha="left", va="top",
                fontsize=12.5, color=col, style="italic", zorder=3)
        ax.text(0.94, 0.92, f"n = {n}", ha="right", va="center",
                fontsize=14, fontweight="bold", color="white",
                bbox=dict(facecolor=col, edgecolor="none",
                          boxstyle="round,pad=0.45", alpha=0.95), zorder=4)
        if q == "Q2" and n == largest_quad:
            ax.text(0.94, 0.78, "★ 최대 클러스터", ha="right", va="center",
                    fontsize=10.5, fontweight="bold", color=col,
                    style="italic", zorder=4)
        ax.plot([0.06, 0.94], [0.78, 0.78], color=col, alpha=0.45,
                linewidth=1.2, zorder=2)

        if n == 0:
            ax.text(0.5, 0.42, "(해당 모델 없음)", ha="center", va="center",
                    fontsize=12, color="#999", style="italic", zorder=3)
        else:
            top_y = 0.72; bot_y = 0.07
            line_pair = (top_y - bot_y) / max(n, 1)
            for i, (name, elo, wt) in enumerate(models):
                y_top = top_y - i * line_pair
                ax.text(0.10, y_top, "●", ha="left", va="top",
                        fontsize=12, color=col, zorder=3)
                ax.text(0.16, y_top, name, ha="left", va="top",
                        fontsize=13.5, fontweight="bold",
                        color="#1c1c1c", zorder=3)
                ax.text(0.16, y_top - line_pair * 0.42,
                        f"ELO {elo:.0f}   ·   WT {wt:.0f}%",
                        ha="left", va="top", fontsize=11,
                        color="#555", zorder=3)

    fig.text(0.32, 0.905, "Watertight > 0%", ha="center", va="center",
             fontsize=15.5, fontweight="bold", color="#222")
    fig.text(0.73, 0.905, "Watertight = 0%", ha="center", va="center",
             fontsize=15.5, fontweight="bold", color="#222")
    fig.text(0.045, 0.665, f"ELO ↑\n(>= {elo_med:.0f})", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#222")
    fig.text(0.045, 0.255, f"ELO ↓\n(< {elo_med:.0f})", ha="center", va="center",
             fontsize=14, fontweight="bold", color="#222")
    fig.text(0.54, 0.965,
             "그림 4. ELO x Watertight 4분면 — 인기 모델 대부분이 production 부적합",
             ha="center", va="center", fontsize=16.5, fontweight="bold",
             color="#1f1f1f")
    fig.text(0.54, 0.015,
             f"주: ELO 중앙값 {elo_med:.0f} 기준 분할. Watertight >0% 임계. "
             f"전체 {n_total}개 모델.",
             ha="center", fontsize=10, color="#666")

    plt.savefig(dst, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[Step 5c] Saved: {dst}")


# ── Step 6: Final summary ─────────────────────────────────────────────────────

def print_summary(metrics: dict, changed: bool):
    r_d1,  p_d1  = metrics["D1 Geometry"][:2]
    r_d2,  p_d2  = metrics["D2 UV"][:2]
    r_d3,  p_d3  = metrics["D3 PBR"][:2]
    r_tot, p_tot = metrics["Total (D1+D2+D3)"][:2]
    r_t16, p_t16 = metrics["texture_resolution"][:2]
    r_t9,  p_t9  = metrics["tex_res_n9"][:2]

    def sig(p): return "p<0.05 ★" if p < 0.05 else f"p={p:.3f}"

    print("\n" + "=" * 60)
    print("=== v5 최종 요약 ===")
    print("=" * 60)
    print("메트릭 구성:")
    print("  D1 Geometry (3개): watertight, manifold_edge_ratio, connected_components")
    print("  D2 UV       (2개): has_uv, uv_packed_area")
    print("  D3 PBR      (2개): pbr_channel_count, texture_resolution")
    print("  총계: 3개 차원 7개 메트릭")
    print()
    print("제거된 메트릭:")
    print("  uv_bbox_efficiency: 모델 평균 범위 0.046 (0.953~0.999), 변별력 없음")
    print("  has_texture:        has_uv와 완전 일치 (불일치 0건), 중복")
    print()
    print("Spearman 핵심 결과:")
    print(f"  D1 Geometry:        r = {r_d1:+.3f}  {sig(p_d1)}  (n=16)")
    print(f"  D2 UV:              r = {r_d2:+.3f}  {sig(p_d2)}  (n=16)")
    print(f"  D3 PBR:             r = {r_d3:+.3f}  {sig(p_d3)}  (n=16)")
    print(f"  Total (D1+D2+D3):   r = {r_tot:+.3f}  {sig(p_tot)}  (n=16)")
    print(f"  texture_resolution: r = {r_t16:+.3f}  {sig(p_t16)}  (n=16)")
    print(f"  tex_res (n=9):      r = {r_t9:+.3f}  {sig(p_t9)}  (n=9)")
    print()
    print(f"분면 분류: {'v4와 동일 (변경 없음)' if not changed else '*** 변경됨 ***'}")
    print()
    print("생성 파일:")
    print("  data/dimension_scores_v5.csv     ✓")
    print("  data/elo_correlation_v5.csv      ✓")
    print("  outputs/fig2_heatmap_v5.png      ✓")
    print("  outputs/fig3_correlation_v5.png  ✓")
    src = "fig4_matrix_v4.png 복사" if not changed else "재생성"
    print(f"  outputs/fig4_matrix_v5.png      ✓  ({src})")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Building v5 dimension scores ...")
    print("=" * 60)
    df = build_scores()

    print("\n" + "=" * 60)
    print("Step 3: Spearman correlations ...")
    print("=" * 60)
    metrics = compute_correlations(df)

    print("\n" + "=" * 60)
    print("Step 4: Quadrant verification ...")
    print("=" * 60)
    add, changed = verify_quadrant(df)

    print("\n" + "=" * 60)
    print("Step 5a: Heatmap v5 ...")
    print("=" * 60)
    make_heatmap(df)

    print("\n" + "=" * 60)
    print("Step 5b: Correlation bar chart v5 ...")
    print("=" * 60)
    make_correlation_bar(df, metrics)

    print("\n" + "=" * 60)
    print("Step 5c: 2x2 Matrix v5 ...")
    print("=" * 60)
    make_matrix(add, changed)

    print_summary(metrics, changed)
