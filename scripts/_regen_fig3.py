"""Regenerate fig3_correlation_v5.png only (adds Total D2+D3 bar)."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analyze_v5 import build_scores, compute_correlations, make_correlation_bar

df      = build_scores()
metrics = compute_correlations(df)
make_correlation_bar(df, metrics)
print("Done.")
