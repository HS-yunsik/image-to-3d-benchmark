"""
Paper figures (한국어, 300dpi):
  Figure 1: 2x2 ELO × Watertight scatter matrix
  Figure 2: ELO vs dimension Spearman r bar chart
  Figure 3: Model dimension rank heatmap
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# ── Korean font ──────────────────────────────────────────────────────────────
rcParams["font.family"]      = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"]     = 42

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data" / "additional_analysis.csv"   # ELO models only (Fig 1, Fig 2)
DATA_FULL = ROOT / "data" / "full_model_ranks.csv"      # all models incl. post-paper (Fig 3)
OUT_DIR   = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── Quadrant colours ─────────────────────────────────────────────────────────
COL_Q1 = "#0E9594"  # teal — visual + production
COL_Q2 = "#E07A5F"  # coral — visual only
COL_Q3 = "#3D5A80"  # blue — production only
COL_Q4 = "#888888"  # gray — neither
COL_POS = "#0E9594"  # teal for positive r
COL_NEG = "#E07A5F"  # coral for negative r


# ════════════════════════════════════════════════════════════════════════════
# Figure 1: 2×2 ELO × Watertight scatter matrix
# ════════════════════════════════════════════════════════════════════════════
def figure1_matrix(df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)

    elo_med = df["elo"].median()
    wt_thresh = 0.0

    # Y-axis: use symlog-like trick — give the wt=0 row a dedicated band
    # so labels don't pile up. We'll plot wt=0 models slightly below 0 with jitter.
    xlim = (df["elo"].min() - 30, df["elo"].max() + 30)
    ylim = (-22, 108)

    # Quadrant backgrounds
    ax.add_patch(Rectangle((elo_med, wt_thresh + 0.5), xlim[1] - elo_med, 100 - wt_thresh,
                            facecolor=COL_Q1, alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((elo_med, ylim[0]), xlim[1] - elo_med, wt_thresh - ylim[0] + 0.5,
                            facecolor=COL_Q2, alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((xlim[0], wt_thresh + 0.5), elo_med - xlim[0], 100 - wt_thresh,
                            facecolor=COL_Q3, alpha=0.10, zorder=0))
    ax.add_patch(Rectangle((xlim[0], ylim[0]), elo_med - xlim[0], wt_thresh - ylim[0] + 0.5,
                            facecolor=COL_Q4, alpha=0.10, zorder=0))

    # Median dividers
    ax.axvline(elo_med, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax.axhline(wt_thresh, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)

    # Quadrant tags — Q1/Q3 in upper corners (wt>0 area is sparse).
    # Q2/Q4 in MIDDLE of empty wt>0 portion of their quadrant (won't conflict
    # with wt=0 data cluster which sits below y=0).
    ax.text(xlim[1] - 8, 102, "Q1: ELO-High / WT>0",
            ha="right", va="bottom", fontsize=12, color=COL_Q1, fontweight="bold")
    ax.text(xlim[1] - 8, 96, "(시각+production)",
            ha="right", va="top", fontsize=10, color=COL_Q1, style="italic")

    ax.text(xlim[0] + 8, 102, "Q3: ELO-Low / WT>0",
            ha="left", va="bottom", fontsize=12, color=COL_Q3, fontweight="bold")
    ax.text(xlim[0] + 8, 96, "(production만)",
            ha="left", va="top", fontsize=10, color=COL_Q3, style="italic")

    # Q2/Q4: label sits in bottom band BELOW the staggered model markers.
    # Put on outer edges, single short line.
    ax.text(xlim[1] - 8, -20.5, "Q2: 시각형 (ELO↑, WT=0)",
            ha="right", va="center", fontsize=11.5, color=COL_Q2, fontweight="bold")
    ax.text(xlim[0] + 8, -20.5, "Q4: 미흡 (ELO↓, WT=0)",
            ha="left", va="center", fontsize=11.5, color=COL_Q4, fontweight="bold")

    quad_color = {
        "Q1: ELO-High / WT-High (>0%)":  COL_Q1,
        "Q2: ELO-High / WT-None (=0%)":  COL_Q2,
        "Q3: ELO-Low  / WT-High (>0%)":  COL_Q3,
        "Q4: ELO-Low  / WT-None (=0%)":  COL_Q4,
    }

    # Stagger wt=0 markers vertically across up-to-three rows so labels can
    # sit clearly. Use a greedy assignment: each label goes to the row whose
    # last-placed label is furthest left (= most space).
    df_zero = df[df["watertight_pct"] == 0].sort_values("elo").reset_index(drop=True)
    df_pos  = df[df["watertight_pct"] >  0].copy()

    ROW_YS    = [-7, -14, -21]
    MIN_GAP   = 38   # ELO units required between same-row labels
    last_x    = [-1e9] * len(ROW_YS)
    zero_y    = {}
    for _, row in df_zero.iterrows():
        x = row["elo"]
        # row whose last_x is smallest → most horizontal slack
        best = min(range(len(ROW_YS)), key=lambda r: last_x[r])
        # If even the best row would still collide, just use the row with most slack
        zero_y[row["model"]] = ROW_YS[best]
        last_x[best] = x

    # Plot wt > 0 markers at true y
    for _, row in df_pos.iterrows():
        col = quad_color.get(row["quadrant"], "black")
        ax.scatter(row["elo"], row["watertight_pct"],
                    s=170, color=col, edgecolor="white", linewidth=1.8, zorder=10)
        ax.annotate(row["model"], (row["elo"], row["watertight_pct"]),
                     xytext=(8, 0), textcoords="offset points",
                     fontsize=11, ha="left", va="center", zorder=20,
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec="none", alpha=0.85))

    # Plot wt = 0 markers at jittered y, with leader lines back to y=0
    for _, row in df_zero.iterrows():
        col = quad_color.get(row["quadrant"], "black")
        y_jit = zero_y[row["model"]]
        # leader line
        ax.plot([row["elo"], row["elo"]], [0, y_jit],
                 color=col, alpha=0.35, linewidth=0.8, zorder=2)
        # marker on the line at y=0 (small) and the labelled marker at jittered y
        ax.scatter(row["elo"], 0, s=40, color=col, edgecolor="white",
                    linewidth=1.0, zorder=8)
        ax.scatter(row["elo"], y_jit, s=140, color=col, edgecolor="white",
                    linewidth=1.5, zorder=10)
        ax.annotate(row["model"], (row["elo"], y_jit),
                     xytext=(0, -11), textcoords="offset points",
                     fontsize=10.5, ha="center", va="top", zorder=20,
                     bbox=dict(boxstyle="round,pad=0.18", fc="white",
                               ec="none", alpha=0.9))

    ax.set_xlabel(f"ELO Score (3D Arena human preference)  →  중앙값 {elo_med:.0f}",
                  fontsize=13, fontweight="bold")
    ax.set_ylabel("Watertight 비율 (%)  ↑",
                  fontsize=13, fontweight="bold")
    ax.set_title("그림 1. ELO × Watertight 4분면 분포 — 시각 품질과 메쉬 구조의 disconnect",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Y ticks — only at meaningful values
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x", alpha=0.25, linestyle=":", zorder=0)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.tick_params(labelsize=11)

    # Footnote
    fig.text(0.5, 0.005,
              "주: y=0% 모델들은 가독성을 위해 두 줄로 분산 표시(점선이 실제 위치). "
              "ELO 중앙값 1207, watertight 임계 >0%.",
              ha="center", fontsize=9.5, color="#555")

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 v2: Quadrant card grid — emphasises the "Q2 = popular but
# production-unfit" message at a glance, without scatter-plot overcrowding.
# ════════════════════════════════════════════════════════════════════════════
def figure1_v2_cards(df: pd.DataFrame, out_path: Path):
    from matplotlib.gridspec import GridSpec

    elo_med = df["elo"].median()

    quad_label_map = {
        "Q1: ELO-High / WT-High (>0%)": "Q1",
        "Q2: ELO-High / WT-None (=0%)": "Q2",
        "Q3: ELO-Low  / WT-High (>0%)": "Q3",
        "Q4: ELO-Low  / WT-None (=0%)": "Q4",
    }

    # Layout follows the user spec:
    #   top-left  = Q1   top-right = Q2
    #   bot-left  = Q3   bot-right = Q4
    quad_config = {
        "Q1": {"title": "Q1: 시각+Production",  "sub": "ELO↑  ·  WT>0",
                "color": COL_Q1, "pos": (0, 0)},
        "Q2": {"title": "Q2: 시각형",            "sub": "ELO↑  ·  WT=0",
                "color": COL_Q2, "pos": (0, 1)},
        "Q3": {"title": "Q3: Production만",      "sub": "ELO↓  ·  WT>0",
                "color": COL_Q3, "pos": (1, 0)},
        "Q4": {"title": "Q4: 미흡",              "sub": "ELO↓  ·  WT=0",
                "color": COL_Q4, "pos": (1, 1)},
    }

    # Group models per quadrant, sorted by ELO desc within each
    quad_models = {q: [] for q in quad_config}
    for _, row in df.sort_values("elo", ascending=False).iterrows():
        q = quad_label_map.get(row["quadrant"])
        if q:
            quad_models[q].append((row["model"], row["elo"], row["watertight_pct"]))

    n_total       = sum(len(v) for v in quad_models.values())
    largest_quad  = max(len(v) for v in quad_models.values())

    fig = plt.figure(figsize=(13.5, 10.5), dpi=300, facecolor="white")
    gs  = GridSpec(2, 2,
                    left=0.11, right=0.97,
                    top=0.86, bottom=0.05,
                    wspace=0.07, hspace=0.10)

    for q, cfg in quad_config.items():
        r, c = cfg["pos"]
        ax = fig.add_subplot(gs[r, c])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()

        col = cfg["color"]
        models = quad_models[q]
        n = len(models)

        # Rounded card background
        from matplotlib.patches import FancyBboxPatch
        card = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle="round,pad=0.0,rounding_size=0.035",
                               facecolor=col, alpha=0.10,
                               edgecolor=col, linewidth=2.5,
                               transform=ax.transAxes, zorder=1)
        ax.add_patch(card)

        # Title strip at top of card
        ax.text(0.06, 0.94, cfg["title"],
                ha="left", va="top", fontsize=19, fontweight="bold",
                color=col, zorder=3)
        ax.text(0.06, 0.86, cfg["sub"],
                ha="left", va="top", fontsize=12.5,
                color=col, style="italic", zorder=3)

        # Count badge top-right
        ax.text(0.94, 0.92, f"n = {n}",
                ha="right", va="center", fontsize=14, fontweight="bold",
                color="white",
                bbox=dict(facecolor=col, edgecolor="none",
                           boxstyle="round,pad=0.45", alpha=0.95),
                zorder=4)

        # Highlight the "Q2 is biggest" message: subtle accent on Q2
        if q == "Q2" and n == largest_quad:
            ax.text(0.94, 0.78, "★ 최대 클러스터",
                     ha="right", va="center", fontsize=10.5, fontweight="bold",
                     color=col, style="italic", zorder=4)

        # Divider line below header
        ax.plot([0.06, 0.94], [0.78, 0.78], color=col, alpha=0.45,
                 linewidth=1.2, zorder=2)

        # Models list
        if n == 0:
            ax.text(0.5, 0.42, "(해당 모델 없음)",
                     ha="center", va="center", fontsize=12, color="#999",
                     style="italic", zorder=3)
        else:
            top_y      = 0.72
            bot_y      = 0.07
            available  = top_y - bot_y
            line_pair  = available / max(n, 1)   # space per model entry

            for i, (name, elo, wt) in enumerate(models):
                y_top = top_y - i * line_pair
                # bullet + model name
                ax.text(0.10, y_top, "●",
                         ha="left", va="top", fontsize=12, color=col, zorder=3)
                ax.text(0.16, y_top, name,
                         ha="left", va="top", fontsize=13.5, fontweight="bold",
                         color="#1c1c1c", zorder=3)
                # stats
                wt_str = f"WT {wt:.0f}%"
                stats  = f"ELO {elo:.0f}   ·   {wt_str}"
                # vertical offset: ~38% of line_pair below
                ax.text(0.16, y_top - line_pair * 0.42, stats,
                         ha="left", va="top", fontsize=11,
                         color="#555", zorder=3)

    # ── Outside labels ─────────────────────────────────────────────────────
    # Column headers (above the cards)
    col_y = 0.905
    fig.text(0.32, col_y, "Watertight > 0%",
              ha="center", va="center", fontsize=15.5, fontweight="bold",
              color="#222")
    fig.text(0.73, col_y, "Watertight = 0%",
              ha="center", va="center", fontsize=15.5, fontweight="bold",
              color="#222")

    # Row labels (left of cards)
    fig.text(0.045, 0.665, f"ELO ↑\n(≥ {elo_med:.0f})",
              ha="center", va="center", fontsize=14, fontweight="bold",
              color="#222")
    fig.text(0.045, 0.255, f"ELO ↓\n(< {elo_med:.0f})",
              ha="center", va="center", fontsize=14, fontweight="bold",
              color="#222")

    # Title
    fig.text(0.54, 0.965,
              "그림 1. ELO × Watertight 4분면 — 인기 모델 대부분이 production 부적합",
              ha="center", va="center", fontsize=16.5, fontweight="bold",
              color="#1f1f1f")

    # Footnote
    fig.text(0.54, 0.015,
              f"주: ELO 중앙값 {elo_med:.0f} 기준 분할. Watertight >0% 임계. "
              f"전체 {n_total}개 모델(post-paper 제외).",
              ha="center", fontsize=10, color="#666")

    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}  ({n_total} models, Q2={len(quad_models['Q2'])} largest)")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2: ELO vs dimension Spearman r bar chart
# ════════════════════════════════════════════════════════════════════════════
def figure2_correlation(out_path: Path):
    metrics = [
        ("D1\n(Geometry)",    -0.064),
        ("D3\n(UV/Texture)",  +0.340),
        ("D4\n(PBR)",         +0.320),
        ("Texture\nResolution\n(텍스처 보유 모델)", +0.701),
    ]
    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    colors = [COL_POS if v > 0 else COL_NEG for v in values]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    bars = ax.bar(labels, values, color=colors, edgecolor="white",
                   linewidth=2.0, width=0.62, zorder=3)

    # Value labels
    for bar, v in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.025 if h >= 0 else -0.025),
                 f"r = {v:+.3f}",
                 ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=13, fontweight="bold",
                 color="#1f1f1f")

    # r=0 baseline
    ax.axhline(0, color="black", linewidth=1.0, zorder=2)

    # Significance bands (rough thresholds)
    ax.axhspan(-0.1, 0.1, alpha=0.10, color="gray", zorder=1)
    ax.text(3.45, 0, "  무상관 영역\n  (|r|<0.1)", ha="left", va="center",
             fontsize=9, color="gray", style="italic")

    # Axes
    ax.set_ylabel("Spearman 순위 상관계수  r",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("자동 메트릭 차원", fontsize=13, fontweight="bold")
    ax.set_title("그림 2. ELO score 와 메트릭 차원별 Spearman 상관계수",
                  fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(-0.25, 0.85)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Footnote
    fig.text(0.5, 0.01,
              "주: D1·D3·D4 는 모델별 종합 차원 점수. Texture Resolution 은 has_texture>0 모델만 (n=8).",
              ha="center", fontsize=10, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3: Dimension rank heatmap
# ════════════════════════════════════════════════════════════════════════════
def figure3_heatmap(df: pd.DataFrame, out_path: Path):
    """Heatmap of dimension ranks.

    df may contain post-paper rows where elo_rank is NaN. Those rows render
    a gray "N/A" ELO cell while D1/D3/D4 cells use the diverging colormap.
    Sort order: ELO-ranked models first (by elo_rank ASC), then post-paper
    models (by d1_rank ASC) appended at the bottom with a divider line.
    """
    has_elo  = df[df["elo_rank"].notna()].copy().sort_values("elo_rank")
    no_elo   = df[df["elo_rank"].isna()].copy().sort_values("d1_rank")
    df_sorted = pd.concat([has_elo, no_elo], ignore_index=True)
    n_elo     = len(has_elo)

    rank_cols  = ["elo_rank", "d1_rank", "d3_rank", "d4_rank"]
    col_labels = ["ELO\n순위", "D1: Geometry\n순위", "D3: UV/Texture\n순위", "D4: PBR\n순위"]
    n_rows     = len(df_sorted)
    n_cols     = len(rank_cols)
    n_models   = n_rows  # total

    cmap = LinearSegmentedColormap.from_list(
        "rank_div",
        [(0.0, "#0E9594"),
         (0.5, "#F2E8CF"),
         (1.0, "#E07A5F")]
    )

    # Build masked matrix: NaN cells become masked, painted gray manually
    matrix     = df_sorted[rank_cols].values.astype(float)
    mask       = np.isnan(matrix)
    masked_arr = np.ma.array(matrix, mask=mask)

    fig_h = max(8, 0.5 * n_rows + 4)
    fig, ax = plt.subplots(figsize=(10, fig_h), dpi=300)

    cmap_w_bad = cmap.copy() if hasattr(cmap, "copy") else cmap
    cmap_w_bad.set_bad(color="#D9D9D9")
    im = ax.imshow(masked_arr, cmap=cmap_w_bad, vmin=1, vmax=n_models, aspect="auto")

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A",
                         ha="center", va="center",
                         fontsize=12, fontweight="bold", color="#666",
                         style="italic")
                continue
            v_int = int(v)
            ratio = (v - 1) / max(n_models - 1, 1)
            txt_color = "white" if (ratio < 0.20 or ratio > 0.80) else "#1f1f1f"
            ax.text(j, i, str(v_int),
                     ha="center", va="center",
                     fontsize=14, fontweight="bold", color=txt_color)

    # Y tick labels — italicise post-paper rows
    ax.set_yticks(range(n_rows))
    ylabels = []
    for _, row in df_sorted.iterrows():
        name = row["model"]
        if pd.isna(row["elo_rank"]):
            ylabels.append(f"{name}*")
        else:
            ylabels.append(name)
    ax.set_yticklabels(ylabels, fontsize=11)

    # Color the post-paper Y labels distinctly
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        if pd.isna(row["elo_rank"]):
            ax.get_yticklabels()[i].set_color("#666")
            ax.get_yticklabels()[i].set_style("italic")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", which="both", length=0, pad=6)
    ax.tick_params(axis="y", which="both", length=0, pad=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Cell borders
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    # Divider between ELO models and post-paper models
    if 0 < n_elo < n_rows:
        ax.axhline(n_elo - 0.5, color="#333", linewidth=2.0, linestyle="--",
                    alpha=0.7, zorder=10)

    title = f"그림 3. 모델별 차원별 순위 — {n_models}개 모델 (낮을수록 좋음)"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=22, y=1.04)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label("순위 (1 = 최상)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([1, n_models // 2, n_models])
    cbar.set_ticklabels(["1\n(좋음)", f"{n_models // 2}", f"{n_models}\n(나쁨)"])

    footer = ("주: D1/D3/D4 는 자동 메트릭 종합 점수 기준 순위 (전체 모델 대상). "
              f"* 표시 = post-paper 모델 (3D Arena ELO 미공개, n={n_rows - n_elo}). "
              "ELO 셀의 회색 N/A 는 인간 선호도 점수 부재를 의미.")
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9.5, color="#555",
              wrap=True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}  ({n_rows} models, {n_elo} with ELO + {n_rows - n_elo} post-paper)")


# ════════════════════════════════════════════════════════════════════════════
def main():
    df = pd.read_csv(DATA)
    print(f"Loaded {len(df)} ELO models from {DATA.name}")

    print("\n[Figure 1] 2×2 Matrix scatter ...")
    figure1_matrix(df, OUT_DIR / "fig1_matrix.png")

    print("\n[Figure 1 v2] Quadrant card grid ...")
    figure1_v2_cards(df, OUT_DIR / "fig1_matrix_v2.png")

    print("\n[Figure 2] Correlation bar chart ...")
    figure2_correlation(OUT_DIR / "fig2_correlation.png")

    if DATA_FULL.exists():
        df_full = pd.read_csv(DATA_FULL)
        print(f"\n[Figure 3] Heatmap with {len(df_full)} models from {DATA_FULL.name}")
        figure3_heatmap(df_full, OUT_DIR / "fig3_heatmap.png")
    else:
        print(f"\n[Figure 3] Heatmap with {len(df)} ELO-only models (full ranks CSV missing)")
        figure3_heatmap(df, OUT_DIR / "fig3_heatmap.png")

    print("\n완료. outputs/ 디렉토리에 3개 figure 저장됨.")


if __name__ == "__main__":
    main()
