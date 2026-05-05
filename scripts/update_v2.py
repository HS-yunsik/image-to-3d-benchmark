"""
Update master notes + figures to 16-model basis.

Steps:
  1. Recompute D1/D3/D4 + texture_resolution Spearman r/p using 16 ELO models
  2. Print to console (for paper_master_notes_v2.md)
  3. Regenerate fig2_correlation_v2.png with p-values shown
  4. Regenerate fig3_heatmap_v2.png with 16 ELO models only (ELO-sorted)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

rcParams["font.family"]        = "Malgun Gothic"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"]       = 42

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

ADD_ANALYSIS = DATA / "additional_analysis.csv"   # 16 ELO models
SUMMARY      = DATA / "model_summary.csv"

COL_POS = "#0E9594"
COL_NEG = "#E07A5F"


# ── Step 1+2: recompute correlations ─────────────────────────────────────────
def compute_correlations() -> dict:
    df = pd.read_csv(ADD_ANALYSIS)
    n  = len(df)
    print(f"[1] 상관계수 재계산 — n={n} ELO 모델 (16-model 기준)")
    print("-" * 60)

    metrics = {}

    # Score-based correlations (sign: positive = higher metric → higher ELO)
    for label, score_col in [
        ("D1 (Geometry)",   "d1_score"),
        ("D3 (UV/Texture)", "d3_score"),
        ("D4 (PBR)",        "d4_score"),
    ]:
        r, p = spearmanr(df["elo"], df[score_col])
        sig  = "p<0.05" if p < 0.05 else f"p={p:.3f}"
        print(f"  ELO vs {label:<18}  r = {r:+.4f}   {sig}   (n={n})")
        metrics[label] = (r, p, n)

    # texture_resolution — full 16 models (with sign-flipped: negative r in score-based corresponds to "higher rank# = lower res")
    r_tex_all, p_tex_all = spearmanr(df["elo"], df["avg_texture_resolution"])
    print(f"\n  ELO vs texture_resolution (전체 16개): r = {r_tex_all:+.4f}  "
          f"p = {p_tex_all:.4f}")

    # texture_resolution — textured models only
    tex = df[df["has_texture_pct"] > 0]
    r_tex_only, p_tex_only = spearmanr(tex["elo"], tex["avg_texture_resolution"])
    print(f"  ELO vs texture_resolution (텍스처 보유, n={len(tex)}): "
          f"r = {r_tex_only:+.4f}  p = {p_tex_only:.4f}")

    # Note: elo_correlation.csv reports r=-0.58 because that file uses ELO RANK
    # (rank 1 = best). Our score-based r above has the opposite sign but same
    # magnitude. We document both views in the master notes.
    elo_rank = df["elo"].rank(ascending=False)
    r_rank_all, p_rank_all = spearmanr(elo_rank, df["avg_texture_resolution"])
    print(f"\n  (참고) rank-based all   : r = {r_rank_all:+.4f} p = {p_rank_all:.4f}")
    r_rank_tex, p_rank_tex = spearmanr(elo_rank.loc[tex.index],
                                        tex["avg_texture_resolution"])
    print(f"  (참고) rank-based tex-on: r = {r_rank_tex:+.4f} p = {p_rank_tex:.4f}")

    metrics["texture_resolution_all"]    = (r_tex_all, p_tex_all, n)
    metrics["texture_resolution_texonly"] = (r_tex_only, p_tex_only, len(tex))
    metrics["texture_resolution_rank_all"]    = (r_rank_all, p_rank_all, n)
    metrics["texture_resolution_rank_texonly"] = (r_rank_tex, p_rank_tex, len(tex))

    # Disconnect top 3 (sanity check matches user's reported numbers)
    print("\n  Disconnect total 상위 3 (확인):")
    for _, row in df.nlargest(3, "disconnect_total").iterrows():
        print(f"    {row['model']:<14}  total-Δ = {int(row['disconnect_total'])}")

    return metrics, df


# ── Step 3: figure 2 v2 ──────────────────────────────────────────────────────
def figure2_v2(metrics: dict, out_path: Path):
    items = [
        ("D1\n(Geometry)",    *metrics["D1 (Geometry)"]),
        ("D3\n(UV/Texture)",  *metrics["D3 (UV/Texture)"]),
        ("D4\n(PBR)",         *metrics["D4 (PBR)"]),
        ("Texture Res.\n(텍스처 보유, n=9)",
            *metrics["texture_resolution_texonly"]),
    ]
    labels = [it[0] for it in items]
    rs     = [it[1] for it in items]
    ps     = [it[2] for it in items]
    ns     = [it[3] for it in items]
    colors = [COL_POS if r > 0 else COL_NEG for r in rs]

    fig, ax = plt.subplots(figsize=(10.5, 7.2), dpi=300)
    bars = ax.bar(labels, rs, color=colors, edgecolor="white",
                   linewidth=2.0, width=0.62, zorder=3)

    # r value above/below + p-value below
    for bar, r, p, n in zip(bars, rs, ps, ns):
        h = bar.get_height()
        # r value
        ax.text(bar.get_x() + bar.get_width() / 2,
                 h + (0.03 if h >= 0 else -0.03),
                 f"r = {r:+.3f}",
                 ha="center", va="bottom" if h >= 0 else "top",
                 fontsize=13, fontweight="bold", color="#1f1f1f")
        # p-value & n inside or outside bar
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
    ax.text(3.45, 0, "  무상관\n  (|r|<0.1)", ha="left", va="center",
             fontsize=9, color="gray", style="italic")

    ax.set_ylabel("Spearman 순위 상관계수  r",
                  fontsize=13, fontweight="bold")
    ax.set_xlabel("자동 메트릭 차원", fontsize=13, fontweight="bold")
    ax.set_title("그림 2. ELO score 와 메트릭 차원별 Spearman 상관계수 — 16 모델 기준",
                  fontsize=14, fontweight="bold", pad=14)
    ax.set_ylim(-0.45, 0.95)
    ax.tick_params(labelsize=11)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.01,
              "주: D1·D3·D4 는 16 ELO 모델 종합 차원 점수 기반. "
              "Texture Res.는 has_texture>0 모델만 (n=9). "
              "* = p<0.05.",
              ha="center", fontsize=10, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved: {out_path}")


# ── Step 4: figure 3 v2 (16 ELO models only, ELO-sorted) ────────────────────
def figure3_v2(df: pd.DataFrame, out_path: Path):
    # Re-rank D1/D3/D4 within the 16 ELO models (so ranks match the table)
    df = df.copy().sort_values("elo_rank").reset_index(drop=True)
    rank_cols  = ["elo_rank", "d1_rank", "d3_rank", "d4_rank"]
    col_labels = ["ELO\n순위", "D1: Geometry\n순위", "D3: UV/Texture\n순위", "D4: PBR\n순위"]
    matrix     = df[rank_cols].values.astype(int)
    n_rows, n_cols = matrix.shape
    n_models   = n_rows

    cmap = LinearSegmentedColormap.from_list(
        "rank_div",
        [(0.0, "#0E9594"),
         (0.5, "#F2E8CF"),
         (1.0, "#E07A5F")]
    )

    fig, ax = plt.subplots(figsize=(10, 9.5), dpi=300)
    im = ax.imshow(matrix, cmap=cmap, vmin=1, vmax=n_models, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            ratio = (v - 1) / max(n_models - 1, 1)
            txt_color = "white" if (ratio < 0.20 or ratio > 0.80) else "#1f1f1f"
            ax.text(j, i, str(v),
                     ha="center", va="center",
                     fontsize=14, fontweight="bold", color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df["model"].tolist(), fontsize=11)
    ax.tick_params(axis="x", which="both", length=0, pad=6)
    ax.tick_params(axis="y", which="both", length=0, pad=4)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(n_cols + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)

    ax.set_title(f"그림 3. 모델별 차원별 순위 — {n_models} 모델 (Zaohaowu3D 포함)",
                  fontsize=14, fontweight="bold", pad=22, y=1.04)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, shrink=0.7)
    cbar.set_label("순위 (1 = 최상)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks([1, n_models // 2, n_models])
    cbar.set_ticklabels(["1\n(좋음)", f"{n_models // 2}", f"{n_models}\n(나쁨)"])

    fig.text(0.5, 0.01,
              "주: ELO 순위는 3D Arena human preference 기준. "
              "D1/D3/D4 는 16 ELO 모델 내 자동 메트릭 종합 점수 기준 순위.",
              ha="center", fontsize=10, color="#555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    metrics, df = compute_correlations()

    print("\n[3] Figure 2 v2 (16-model + p-values) ...")
    figure2_v2(metrics, OUT_DIR / "fig2_correlation_v2.png")

    print("\n[4] Figure 3 v2 (16 ELO 모델, ELO 정렬) ...")
    figure3_v2(df, OUT_DIR / "fig3_heatmap_v2.png")

    return metrics


if __name__ == "__main__":
    main()
