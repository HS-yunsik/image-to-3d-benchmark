"""D2+D3에서 공동순위(tie) 발생 구조 확인."""
import pandas as pd
from scipy.stats import rankdata, spearmanr

df = pd.read_csv("data/dimension_scores_v5.csv")
df = df[df["elo"].notna()].copy().reset_index(drop=True)

df["Total_D2D3"] = (df["D2"] + df["D3"]) / 2
df["elo_rank"]   = rankdata(-df["elo"]).astype(int)

print("모델별 D2 / D3 / D2+D3 실제 점수")
print(f"{'Model':<18} {'ELO_r':>6}  {'D1':>6}  {'D2':>6}  {'D3':>6}  {'D2+D3':>7}")
print("-" * 62)
for _, r in df.sort_values("elo_rank").iterrows():
    print(f"{r['model']:<18} {r['elo_rank']:>6}  {r['D1']:>6.3f}  "
          f"{r['D2']:>6.3f}  {r['D3']:>6.3f}  {r['Total_D2D3']:>7.3f}")

# D2+D3 공동순위 확인
print("\nD2+D3 점수 분포 — 공동순위(tie) 확인:")
vc = df["Total_D2D3"].round(4).value_counts().sort_index()
for val, cnt in vc.items():
    models = df[df["Total_D2D3"].round(4) == val]["model"].tolist()
    print(f"  D2+D3 = {val:.4f}  x{cnt}  → {models}")
