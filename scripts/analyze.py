"""Stage 3: Analysis pipeline.

Inputs:
    data/all_metrics.csv          - per-mesh results from run_metrics.py
    data/elo_scores.csv           - ELO scores from 3D Arena leaderboard
    data/file_categories.csv      - prompt-level category labels

Outputs:
    data/model_summary.csv        - per-model aggregate metrics, ELO-sorted
    data/elo_correlation.csv      - Spearman r / p-value per metric vs ELO rank
    data/components_stats.csv     - connected_components distribution per model
    data/category_model_matrix.csv- watertight% / median CC per category x model
    data/disconnect_analysis.csv  - top ELO vs metric rank discrepancies

Usage:
    conda activate 3darena
    python scripts/analyze.py
    python scripts/analyze.py --metrics-csv data/all_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

METRICS_CSV  = DATA / "all_metrics.csv"
ELO_CSV      = DATA / "elo_scores.csv"
CATS_CSV     = DATA / "file_categories.csv"

SUMMARY_CSV   = DATA / "model_summary.csv"
CORR_CSV      = DATA / "elo_correlation.csv"
COMP_CSV      = DATA / "components_stats.csv"
CAT_CSV       = DATA / "category_model_matrix.csv"
DISC_CSV      = DATA / "disconnect_analysis.csv"

# Metrics included in ELO correlation (must match model_summary columns)
CORR_METRICS = [
    "watertight_pct",
    "manifold_edge_ratio",
    "nonmanifold_edge_ratio",
    "boundary_edge_ratio",
    "avg_components",
    "has_uv_pct",
    "has_texture_pct",
    "avg_pbr_channels",
    "avg_texture_resolution",
    "avg_vertex_count",
    "avg_face_count",
]

_BOOL_MAP = {"True": True, "False": False, True: True, False: False}


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {path} not found. Run run_metrics.py first.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    n_total = len(df)
    df = df[df["load_success"].map(_BOOL_MAP) == True].copy()
    print(f"  {len(df)}/{n_total} meshes loaded successfully "
          f"({n_total - len(df)} failures ignored)")
    df["prompt_name"] = df["filename"].str.rsplit(".", n=1).str[0]
    for col in ("watertight", "has_uv", "has_texture"):
        df[col] = df[col].map(_BOOL_MAP)
    return df


def load_elo(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARN: {path} not found -ELO data unavailable.", file=sys.stderr)
        return pd.DataFrame(columns=["model", "elo", "rank", "post_paper", "anonymous"])
    df = pd.read_csv(path)
    df["post_paper"] = df["post_paper"].map(_BOOL_MAP)
    df["anonymous"]  = df["anonymous"].map(_BOOL_MAP)
    df["elo"]  = pd.to_numeric(df["elo"],  errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df


def load_categories(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARN: {path} not found -category analysis skipped.", file=sys.stderr)
        return pd.DataFrame(columns=["prompt_name", "category"])
    return pd.read_csv(path)[["prompt_name", "category"]]


# ── Output 1: Model Summary ──────────────────────────────────────────────────

def build_model_summary(metrics: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    grp = metrics.groupby("model")
    summary = pd.DataFrame({
        "watertight_pct":         grp["watertight"].mean() * 100,
        "manifold_edge_ratio":    grp["manifold_edge_ratio"].mean(),
        "nonmanifold_edge_ratio": grp["nonmanifold_edge_ratio"].mean(),
        "boundary_edge_ratio":    grp["boundary_edge_ratio"].mean(),
        "avg_components":         grp["connected_components"].mean(),
        "has_uv_pct":             grp["has_uv"].mean() * 100,
        "has_texture_pct":        grp["has_texture"].mean() * 100,
        "avg_pbr_channels":       grp["pbr_channel_count"].mean(),
        "avg_texture_resolution": grp["texture_resolution"].mean(),
        "avg_vertex_count":       grp["vertex_count"].mean(),
        "avg_face_count":         grp["face_count"].mean(),
        "mesh_count":             grp.size(),
    }).reset_index()

    elo_cols = elo[["model", "elo", "rank", "post_paper", "anonymous"]].copy()
    summary = summary.merge(elo_cols, on="model", how="left")
    summary = summary.sort_values(["rank", "model"],
                                  ascending=[True, True],
                                  na_position="last").reset_index(drop=True)

    ordered = ["model", "elo", "rank", "post_paper", "anonymous", "mesh_count",
               "watertight_pct", "manifold_edge_ratio", "nonmanifold_edge_ratio",
               "boundary_edge_ratio", "avg_components", "has_uv_pct",
               "has_texture_pct", "avg_pbr_channels", "avg_texture_resolution",
               "avg_vertex_count", "avg_face_count"]
    return summary[[c for c in ordered if c in summary.columns]]


# ── Output 2: ELO Correlation ────────────────────────────────────────────────

def compute_elo_correlation(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[
        (summary["post_paper"] == False) & summary["elo"].notna()
    ].copy()

    if len(eligible) < 3:
        print(f"  WARN: only {len(eligible)} eligible models -need >=3 for correlation.",
              file=sys.stderr)
        return pd.DataFrame(columns=["metric", "spearman_r", "p_value", "n"])

    # Higher ELO → lower rank number (rank 1 = best)
    elo_rank = eligible["elo"].rank(ascending=False)
    rows = []
    for metric in CORR_METRICS:
        if metric not in eligible.columns:
            continue
        vals = eligible[metric]
        mask = vals.notna() & elo_rank.notna()
        n = mask.sum()
        if n < 3:
            continue
        r, p = stats.spearmanr(elo_rank[mask], vals[mask])
        rows.append({
            "metric": metric,
            "spearman_r": round(float(r), 4),
            "p_value":    round(float(p), 4),
            "n":          int(n),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("spearman_r", key=abs, ascending=False)
    return df


# ── Output 3: Components Distribution ────────────────────────────────────────

def compute_components_stats(metrics: pd.DataFrame) -> pd.DataFrame:
    cc = metrics.groupby("model")["connected_components"]
    out = cc.agg(
        median="median",
        mean="mean",
        std="std",
        min="min",
        max="max",
        count="count",
    ).reset_index()
    out["log_median"] = np.log1p(out["median"]).round(3)
    out["log_mean"]   = np.log1p(out["mean"]).round(3)
    for col in ("median", "mean", "std", "min", "max"):
        out[col] = out[col].round(1)
    return out.sort_values("median")


# ── Output 4: Category × Model Matrix ────────────────────────────────────────

def compute_category_model_matrix(
    metrics: pd.DataFrame,
    categories: pd.DataFrame,
) -> pd.DataFrame:
    if categories.empty:
        return pd.DataFrame()

    merged = metrics.merge(categories, on="prompt_name", how="left")
    merged = merged[merged["category"].notna() & (merged["category"] != "abstract")]

    if merged.empty:
        return pd.DataFrame()

    # watertight %
    wt = (merged.groupby(["category", "model"])["watertight"].mean() * 100).round(1)
    wt_wide = wt.unstack("model")
    wt_wide.columns = [f"wt%_{c}" for c in wt_wide.columns]

    # median connected components
    cc = merged.groupby(["category", "model"])["connected_components"].median().round(1)
    cc_wide = cc.unstack("model")
    cc_wide.columns = [f"cc_med_{c}" for c in cc_wide.columns]

    combined = pd.concat([wt_wide, cc_wide], axis=1).reset_index()
    return combined


# ── Output 5: Disconnect Analysis ────────────────────────────────────────────

def compute_disconnect(summary: pd.DataFrame) -> pd.DataFrame:
    eligible = summary[
        (summary["post_paper"] == False) & summary["elo"].notna()
    ].copy()

    if len(eligible) < 3:
        return pd.DataFrame()

    eligible["elo_rank"] = eligible["elo"].rank(ascending=False).astype(int)
    eligible["wt_rank"]  = eligible["watertight_pct"].rank(ascending=False).astype(int)
    eligible["cc_rank"]  = eligible["avg_components"].rank(ascending=True).astype(int)
    eligible["elo_vs_wt_gap"] = (eligible["elo_rank"] - eligible["wt_rank"]).abs()
    eligible["elo_vs_cc_gap"] = (eligible["elo_rank"] - eligible["cc_rank"]).abs()

    top_wt = eligible.nlargest(5, "elo_vs_wt_gap")[
        ["model", "elo_rank", "wt_rank", "elo_vs_wt_gap", "watertight_pct"]
    ]
    top_cc = eligible.nlargest(5, "elo_vs_cc_gap")[
        ["model", "elo_rank", "cc_rank", "elo_vs_cc_gap", "avg_components"]
    ]

    print("\n  ELO rank vs Watertight rank -top 5 disconnect:")
    print(top_wt.to_string(index=False))
    print("\n  ELO rank vs Components rank -top 5 disconnect:")
    print(top_cc.to_string(index=False))

    result_cols = ["model", "elo", "elo_rank",
                   "watertight_pct", "wt_rank", "elo_vs_wt_gap",
                   "avg_components", "cc_rank", "elo_vs_cc_gap"]
    result = eligible[[c for c in result_cols if c in eligible.columns]].copy()
    return result.sort_values("elo_vs_wt_gap", ascending=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", default=str(METRICS_CSV),
                    help="Path to all_metrics.csv (default: data/all_metrics.csv)")
    args = ap.parse_args()

    metrics_path = Path(args.metrics_csv)
    print(f"Loading metrics from {metrics_path} ...")
    metrics = load_metrics(metrics_path)
    print(f"  {metrics['model'].nunique()} models, {len(metrics)} meshes")

    elo  = load_elo(ELO_CSV)
    cats = load_categories(CATS_CSV)

    # [1] Model Summary
    print("\n[1] Model summary ...")
    summary = build_model_summary(metrics, elo)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(summary.to_string(index=False))
    print(f"  -> {SUMMARY_CSV}")

    # [2] ELO Correlation
    print("\n[2] ELO correlation ...")
    corr = compute_elo_correlation(summary)
    corr.to_csv(CORR_CSV, index=False)
    print(corr.to_string(index=False))
    print(f"  -> {CORR_CSV}")

    # [3] Components Distribution
    print("\n[3] Components distribution ...")
    comp_stats = compute_components_stats(metrics)
    comp_stats.to_csv(COMP_CSV, index=False)
    print(comp_stats.to_string(index=False))
    print(f"  -> {COMP_CSV}")

    # [4] Category × Model Matrix
    print("\n[4] Category x model matrix ...")
    cat_matrix = compute_category_model_matrix(metrics, cats)
    if not cat_matrix.empty:
        cat_matrix.to_csv(CAT_CSV, index=False)
        print(f"  {cat_matrix.shape[0]} categories, "
              f"{cat_matrix.shape[1]-1} metric-model columns")
        print(f"  -> {CAT_CSV}")
    else:
        print("  Skipped (no category data).")

    # [5] Disconnect Analysis
    print("\n[5] ELO-metric disconnect ...")
    disc = compute_disconnect(summary)
    if not disc.empty:
        disc.to_csv(DISC_CSV, index=False)
        print(f"  -> {DISC_CSV}")
    else:
        print("  Skipped (insufficient models).")

    print("\nDone.")


if __name__ == "__main__":
    main()
