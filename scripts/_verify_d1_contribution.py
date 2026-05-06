"""
D1이 Total 상관을 높이는지 엄밀하게 검증.

확인 항목:
1. 각 차원별 r 재확인
2. D1 vs (D2+D3) 상관: 정말 반상관인가?
3. r 차이(+0.109)의 신뢰구간 — 유의미한 차이인가?
4. Bootstrap CI (n=16 소표본 보정)
5. 모델별 D1 / D2+D3 / Total / ELO 순위 비교
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import t as t_dist

np.random.seed(42)

df = pd.read_csv("data/dimension_scores_v5.csv")
df = df[df["elo"].notna()].copy().reset_index(drop=True)
n = len(df)

df["Total_D2D3"] = (df["D2"] + df["D3"]) / 2

# ── 1. 각 상관 재확인 ─────────────────────────────────────────────────────────
print("=" * 60)
print("1. 상관계수 재확인")
print("=" * 60)
for label, col in [("D1",         "D1"),
                   ("D2",         "D2"),
                   ("D3",         "D3"),
                   ("Total_D2D3", "Total_D2D3"),
                   ("Total",      "Total")]:
    r, p = spearmanr(df["elo"], df[col])
    print(f"  {label:<12}  r = {r:+.4f}   p = {p:.4f}  {'★' if p < 0.05 else '-'}")

# ── 2. D1 vs D2/D3: 반상관 확인 ──────────────────────────────────────────────
print()
print("=" * 60)
print("2. D1 vs D2 / D3 / D2+D3 상관 (차원 간 관계)")
print("=" * 60)
for lbl, col in [("D2", "D2"), ("D3", "D3"), ("D2+D3", "Total_D2D3")]:
    r, p = spearmanr(df["D1"], df[col])
    print(f"  D1 vs {lbl:<8}  r = {r:+.4f}   p = {p:.4f}  {'★' if p < 0.05 else '-'}")

# ── 3. Bootstrap CI for each r ────────────────────────────────────────────────
print()
print("=" * 60)
print("3. Bootstrap 95% CI  (B=10000, n=16)")
print("=" * 60)

B = 10_000

def bootstrap_spearman_ci(x, y, B=10_000, alpha=0.05):
    idx = np.arange(len(x))
    boot_r = []
    for _ in range(B):
        s = np.random.choice(idx, size=len(idx), replace=True)
        r, _ = spearmanr(x[s], y[s])
        boot_r.append(r)
    lo = np.percentile(boot_r, 100 * alpha / 2)
    hi = np.percentile(boot_r, 100 * (1 - alpha / 2))
    return lo, hi

elo = df["elo"].values
for label, col in [("D1",         "D1"),
                   ("D2",         "D2"),
                   ("D3",         "D3"),
                   ("Total_D2D3", "Total_D2D3"),
                   ("Total",      "Total")]:
    r_obs, _ = spearmanr(elo, df[col].values)
    lo, hi   = bootstrap_spearman_ci(elo, df[col].values, B)
    print(f"  {label:<12}  r = {r_obs:+.4f}   95% CI [{lo:+.3f}, {hi:+.3f}]")

# ── 4. 두 상관계수 차이가 유의한가? ──────────────────────────────────────────
print()
print("=" * 60)
print("4. Total vs Total_D2D3 차이 유의성 (Bootstrap paired test)")
print("=" * 60)

diff_boot = []
for _ in range(B):
    s = np.random.choice(n, size=n, replace=True)
    r1, _ = spearmanr(elo[s], df["Total"].values[s])
    r2, _ = spearmanr(elo[s], df["Total_D2D3"].values[s])
    diff_boot.append(r1 - r2)

diff_obs = spearmanr(elo, df["Total"].values)[0] - spearmanr(elo, df["Total_D2D3"].values)[0]
p_diff   = np.mean(np.array(diff_boot) <= 0)   # one-sided: P(diff <= 0)
lo_diff  = np.percentile(diff_boot, 2.5)
hi_diff  = np.percentile(diff_boot, 97.5)

print(f"  관측 차이 (Total - Total_D2D3): Δr = {diff_obs:+.4f}")
print(f"  Bootstrap 95% CI of Δr:         [{lo_diff:+.3f}, {hi_diff:+.3f}]")
print(f"  P(Δr ≤ 0) one-sided:            p = {p_diff:.3f}")
if lo_diff > 0:
    print("  → CI가 0을 포함하지 않음: D1 포함이 유의미하게 상관을 높임 ✓")
else:
    print("  → CI가 0을 포함: 차이가 통계적으로 유의하지 않음 ✗")

# ── 5. 모델별 순위 비교 ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. 모델별 ELO 순위 vs D2+D3 순위 vs Total 순위")
print("   (D1이 어느 모델의 순위를 바꾸는지 확인)")
print("=" * 60)

from scipy.stats import rankdata
df["elo_rank"]     = rankdata(-df["elo"])
df["d2d3_rank"]    = rankdata(-df["Total_D2D3"])
df["total_rank"]   = rankdata(-df["Total"])
df["d1_rank"]      = rankdata(-df["D1"])
df["rank_shift"]   = df["d2d3_rank"] - df["total_rank"]  # + = moved up when D1 added

print(f"{'Model':<18} {'ELO_r':>6} {'D2D3_r':>7} {'Total_r':>8} {'Shift':>6}  D1_score")
print("-" * 60)
for _, row in df.sort_values("elo_rank").iterrows():
    shift_str = f"{row['rank_shift']:+.0f}" if row['rank_shift'] != 0 else "  ="
    print(f"{row['model']:<18} {row['elo_rank']:>6.0f} {row['d2d3_rank']:>7.0f} "
          f"{row['total_rank']:>8.0f} {shift_str:>6}  {row['D1']:.3f}")

print()
print("주: Shift > 0 = D1 추가 후 순위 하락 (D2D3_rank > Total_rank)")
print("    Shift < 0 = D1 추가 후 순위 상승")
