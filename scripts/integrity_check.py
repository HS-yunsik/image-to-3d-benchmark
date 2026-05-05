"""
Data integrity check (read-only).

Checks for each mesh-classified model:
  [1] Download completeness — count of .glb/.obj files in meshes/outputs/{model}/
  [2] Metric application  — row count in data/all_metrics.csv
  [3] ELO availability    — non-NaN ELO in data/elo_scores.csv
  [4] Figure inclusion    — presence in additional_analysis.csv / full_model_ranks.csv
  [5] File integrity      — zero-byte .glb files, mesh load failures

Output:
  data/integrity_check.csv  (per-model status table)
  console summary           (action items: re-download / re-run metrics / regen figures)

DOES NOT modify, delete, or re-download anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATA      = ROOT / "data"
MESH_DIR  = ROOT / "meshes" / "outputs"
LOG_DIR   = ROOT / "logs"

INVENTORY    = DATA / "model_inventory.csv"
MANIFEST     = DATA / "manifest.csv"
ALL_METRICS  = DATA / "all_metrics.csv"
ELO_SCORES   = DATA / "elo_scores.csv"
ADD_ANALYSIS = DATA / "additional_analysis.csv"
FULL_RANKS   = DATA / "full_model_ranks.csv"
ERRORS_CSV   = LOG_DIR / "metric_errors.csv"

OUTPUT_CSV   = DATA / "integrity_check.csv"

# Cap used by download_all.py
DOWNLOAD_CAP = 100


def expected_count(inventory_row) -> int:
    """Expected file count for a model = min(n_mesh_files, DOWNLOAD_CAP)."""
    return min(int(inventory_row["n_mesh_files"]), DOWNLOAD_CAP)


def main() -> int:
    # ── Load inputs ─────────────────────────────────────────────────────────
    if not INVENTORY.exists():
        print(f"ERROR: missing {INVENTORY}", file=sys.stderr)
        return 1

    inventory = pd.read_csv(INVENTORY)
    inventory_mesh = inventory[inventory["classification"] == "mesh"].copy()
    print(f"Mesh-classified models in inventory: {len(inventory_mesh)}")

    # ELO data
    elo = pd.read_csv(ELO_SCORES) if ELO_SCORES.exists() else pd.DataFrame()
    elo_models     = set(elo[elo["elo"].notna()]["model"].tolist()) if not elo.empty else set()
    post_paper     = set(elo[elo["post_paper"] == True]["model"].tolist()) if not elo.empty else set()
    print(f"ELO-scored models: {len(elo_models)}, post-paper: {len(post_paper)}")

    # Metrics CSV — count per-model rows + load failures
    metrics_counts: dict[str, int] = {}
    metrics_failed: dict[str, int] = {}
    if ALL_METRICS.exists():
        m_df = pd.read_csv(ALL_METRICS)
        # load_success column (True/False as strings)
        success_mask = m_df["load_success"].astype(str).str.lower().isin(["true", "1"])
        for model, sub in m_df.groupby("model"):
            metrics_counts[model] = len(sub)
            metrics_failed[model] = int((~success_mask[sub.index]).sum())
    print(f"Metrics rows in all_metrics.csv: {sum(metrics_counts.values())} "
          f"({len(metrics_counts)} models)")

    # Figures inclusion sets
    fig_elo_models  = set()
    fig_full_models = set()
    if ADD_ANALYSIS.exists():
        fig_elo_models = set(pd.read_csv(ADD_ANALYSIS)["model"].tolist())
    if FULL_RANKS.exists():
        fig_full_models = set(pd.read_csv(FULL_RANKS)["model"].tolist())

    # metric errors log
    err_count_per_model: dict[str, int] = {}
    if ERRORS_CSV.exists():
        e_df = pd.read_csv(ERRORS_CSV)
        for model, sub in e_df.groupby("model"):
            err_count_per_model[model] = len(sub)
    else:
        print("logs/metric_errors.csv: not present (no recorded metric errors).")

    # ── Walk each mesh model and assemble row ───────────────────────────────
    rows = []
    for _, inv in inventory_mesh.iterrows():
        model    = inv["model"]
        expected = expected_count(inv)

        model_dir = MESH_DIR / model
        if model_dir.exists():
            files = sorted(list(model_dir.glob("*.glb")) + list(model_dir.glob("*.obj")))
            downloaded   = len(files)
            zero_size    = sum(1 for f in files if f.stat().st_size == 0)
        else:
            downloaded = 0
            zero_size  = 0

        applied   = metrics_counts.get(model, 0)
        load_fail = metrics_failed.get(model, 0)
        log_err   = err_count_per_model.get(model, 0)

        has_elo = model in elo_models
        is_post = model in post_paper

        in_fig_elo  = model in fig_elo_models
        in_fig_full = model in fig_full_models

        # Status logic
        issues = []
        if downloaded < expected:
            issues.append(f"DL {downloaded}/{expected}")
        if applied < downloaded:
            issues.append(f"metrics {applied}/{downloaded}")
        if zero_size > 0:
            issues.append(f"zero-byte={zero_size}")
        if load_fail > 0:
            issues.append(f"load-fail={load_fail}")
        if log_err > 0:
            issues.append(f"err-log={log_err}")

        if not has_elo and not is_post:
            issues.append("NO_ELO_UNEXPECTED")

        # Figure inclusion expectations
        # ELO models should be in fig_elo + fig_full
        # post-paper should be in fig_full only
        if has_elo and not in_fig_elo:
            issues.append("missing_from_fig_elo")
        if has_elo and not in_fig_full:
            issues.append("missing_from_fig_full")
        if is_post and not in_fig_full:
            issues.append("missing_from_fig_full(post-paper)")

        if not issues:
            status = "OK"
        elif any(s.startswith("DL") for s in issues):
            status = "INCOMPLETE_DOWNLOAD"
        elif any(s.startswith("metrics") or s.startswith("load") or s.startswith("err") for s in issues):
            status = "METRIC_ISSUE"
        elif any(s.startswith("missing_from_fig") for s in issues):
            status = "FIGURE_MISSING"
        else:
            status = "OTHER"

        rows.append({
            "model":            model,
            "expected":         expected,
            "downloaded":       downloaded,
            "metrics_applied":  applied,
            "load_failures":    load_fail,
            "zero_size_files":  zero_size,
            "elo_available":    has_elo,
            "post_paper":       is_post,
            "in_fig_elo":       in_fig_elo,
            "in_fig_full":      in_fig_full,
            "status":           status,
            "issues":           "; ".join(issues) if issues else "",
        })

    df = pd.DataFrame(rows).sort_values(["status", "model"]).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)

    # ── Console summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("MODEL-BY-MODEL STATUS")
    print("=" * 78)
    print(f"{'Model':<18} {'Exp':>4} {'DL':>4} {'Met':>4} {'Fail':>5} "
          f"{'ELO':>4} {'PP':>3} {'F1':>3} {'F3':>3}  Status / Issues")
    print("-" * 78)
    for _, r in df.iterrows():
        print(f"{r['model']:<18} {r['expected']:>4} {r['downloaded']:>4} "
              f"{r['metrics_applied']:>4} {r['load_failures']:>5} "
              f"{'Y' if r['elo_available'] else '-':>4} "
              f"{'Y' if r['post_paper'] else '-':>3} "
              f"{'Y' if r['in_fig_elo'] else '-':>3} "
              f"{'Y' if r['in_fig_full'] else '-':>3}  "
              f"{r['status']}  {r['issues']}")

    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY")
    print("=" * 78)
    by_status = df["status"].value_counts()
    for st, n in by_status.items():
        print(f"  {st:<22} {n}")

    # Action items
    redo_dl  = df[df["status"] == "INCOMPLETE_DOWNLOAD"]["model"].tolist()
    redo_met = df[df["status"] == "METRIC_ISSUE"]["model"].tolist()
    redo_fig = df[df["status"] == "FIGURE_MISSING"]["model"].tolist()

    print("\n" + "=" * 78)
    print("ACTION ITEMS")
    print("=" * 78)
    print(f"정상 모델 (status=OK):                   {(df['status'] == 'OK').sum()}개")
    print(f"재다운로드 필요 (INCOMPLETE_DOWNLOAD):  {redo_dl if redo_dl else '없음'}")
    print(f"메트릭 재실행 필요 (METRIC_ISSUE):       {redo_met if redo_met else '없음'}")
    print(f"figure 누락 (FIGURE_MISSING):           {redo_fig if redo_fig else '없음'}")
    print(f"분석 figure 갱신 필요:                   "
          f"{'yes' if (redo_dl or redo_met or redo_fig) else 'no'}")

    # Specific spotlight: previously-flagged models
    print("\n" + "=" * 78)
    print("이전 보고된 미완료 모델 점검")
    print("=" * 78)
    for spot in ("MeshFormer", "Unique3D", "Zaohaowu3D"):
        row = df[df["model"] == spot]
        if row.empty:
            print(f"  {spot:<18}  inventory에 없음")
            continue
        r = row.iloc[0]
        verdict = "✓ 완료" if r["status"] == "OK" else f"✗ {r['status']}: {r['issues']}"
        print(f"  {spot:<18}  expected={r['expected']:>3}  downloaded={r['downloaded']:>3}  "
              f"metrics={r['metrics_applied']:>3}  →  {verdict}")

    print(f"\n저장: {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
