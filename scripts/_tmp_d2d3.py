import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv('data/dimension_scores_v5.csv')
df = df[df['elo'].notna()].copy()

df['Total_D2D3'] = (df['D2'] + df['D3']) / 2

r_d2d3, p_d2d3  = spearmanr(df['Total_D2D3'], df['elo'])
r_total, p_total = spearmanr(df['Total'],      df['elo'])
r1, p1 = spearmanr(df['D1'], df['elo'])
r2, p2 = spearmanr(df['D2'], df['elo'])
r3, p3 = spearmanr(df['D3'], df['elo'])

sig = lambda p: "★" if p < 0.05 else "-"

print(f"n = {len(df)}")
print()
print(f"Total_D2D3  (D2+D3):   r = {r_d2d3:+.3f}  p = {p_d2d3:.3f}  {sig(p_d2d3)}")
print(f"Total_D1D2D3 (D1+D2+D3): r = {r_total:+.3f}  p = {p_total:.3f}  {sig(p_total)}")
print()
print(f"  D1: r = {r1:+.3f}  p = {p1:.3f}  {sig(p1)}")
print(f"  D2: r = {r2:+.3f}  p = {p2:.3f}  {sig(p2)}")
print(f"  D3: r = {r3:+.3f}  p = {p3:.3f}  {sig(p3)}")
