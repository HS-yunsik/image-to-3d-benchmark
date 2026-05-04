"""Preview visualization of the connected_components finding.

Reads data/components_preview.csv (long-format: model, filename, components)
and data/elo_scores.csv. Produces:

  output/figures/components_preview.png
    - Left  : per-model box plot (log-scale Y) of components
    - Right : ELO vs median(components) scatter, model-labeled

Also computes Spearman r between median(components) and ELO over the
intersection of models that have both a components sample and a paper-era
ELO. Splat-format models are excluded (we did not download their meshes).
"""
from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
COMP_CSV = ROOT / "data" / "components_preview.csv"
ELO_CSV = ROOT / "data" / "elo_scores.csv"
FIG_DIR = ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
FIG_OUT = FIG_DIR / "components_preview.png"


def load_components() -> dict[str, list[int]]:
    if not COMP_CSV.exists():
        print(f"ERROR: {COMP_CSV} missing -- run quick_components_check.py first.",
              file=sys.stderr)
        sys.exit(1)
    by_model: dict[str, list[int]] = defaultdict(list)
    with COMP_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                v = int(r["components"])
            except ValueError:
                continue
            if v >= 0:
                by_model[r["model"]].append(v)
    return dict(by_model)


def load_elo() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ELO_CSV.exists():
        return out
    with ELO_CSV.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                elo = float(r["elo"]) if r["elo"] else None
            except ValueError:
                elo = None
            try:
                rank = int(r["rank"]) if r["rank"] else None
            except ValueError:
                rank = None
            out[r["model"]] = {
                "elo": elo, "rank": rank,
                "format": r.get("format", ""),
                "post_paper": r.get("post_paper", "False") == "True",
                "anonymous": r.get("anonymous", "False") == "True",
            }
    return out


def spearman_r(x: list[float], y: list[float]) -> float:
    """Plain Spearman: rank-correlate x with y. Returns NaN if n < 3."""
    n = len(x)
    if n < 3:
        return float("nan")

    def ranks(vs: list[float]) -> list[float]:
        # Average ranks for ties
        order = sorted(range(n), key=lambda i: vs[i])
        ranks_ = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks_[order[k]] = avg
            i = j + 1
        return ranks_

    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    comps = load_components()
    elo = load_elo()
    if not comps:
        print("No components data.", file=sys.stderr)
        return 1

    # Order models by median components ascending (left = clean, right = shattered)
    ordered = sorted(comps.keys(), key=lambda m: median(comps[m]))
    print("Models in components data:", len(ordered))
    for m in ordered:
        med = int(median(comps[m]))
        e = elo.get(m, {}).get("elo")
        rk = elo.get(m, {}).get("rank")
        print(f"  {m:<22} n={len(comps[m]):>3} med={med:>7} "
              f"elo={e if e else '(post-paper)' :<5} rank={rk}")

    # ----- Plot -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-model box plot (log Y)
    ax = axes[0]
    data = [comps[m] for m in ordered]
    bp = ax.boxplot(data, labels=ordered, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.5))
    # Color: ELO-known = blue, post-paper = orange, missing = grey
    for patch, m in zip(bp["boxes"], ordered):
        info = elo.get(m, {})
        if info.get("post_paper"):
            patch.set_facecolor("#ffb84d")
        elif info.get("elo") is not None:
            patch.set_facecolor("#6db3f2")
        else:
            patch.set_facecolor("#cccccc")
    ax.set_yscale("log")
    ax.set_ylabel("connected_components (log scale)")
    ax.set_title(f"Components per mesh, n<={max(len(v) for v in comps.values())} per model")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Right: ELO vs median components scatter
    ax = axes[1]
    pairs = []
    for m in ordered:
        info = elo.get(m, {})
        if info.get("elo") is None or info.get("post_paper"):
            continue
        if info.get("format") == "Splat":
            # We don't have components for splat models in our analysis
            continue
        pairs.append((m, median(comps[m]), info["elo"]))

    if pairs:
        xs = [p[1] for p in pairs]
        ys = [p[2] for p in pairs]
        ax.scatter(xs, ys, s=70, c="#1f77b4", edgecolor="black", zorder=3)
        for m, x, y in pairs:
            ax.annotate(m, (x, y), xytext=(6, 4), textcoords="offset points",
                        fontsize=8)

        rho = spearman_r(xs, ys)
        ax.set_title(f"ELO vs median(components)   "
                     f"Spearman r = {rho:+.3f}   (n={len(pairs)})")
    else:
        ax.set_title("ELO vs median(components) -- no overlap data")

    # Also overlay post-paper models with ELO=NaN at their components value
    post_pairs = []
    for m in ordered:
        info = elo.get(m, {})
        if info.get("post_paper"):
            post_pairs.append((m, median(comps[m])))
    if post_pairs:
        # Plot below the lowest ELO value as triangles (no Y info)
        y_floor = min(p[2] for p in pairs) - 50 if pairs else 950
        for m, x in post_pairs:
            ax.scatter([x], [y_floor], s=70, c="#ffb84d",
                       edgecolor="black", marker="^", zorder=3)
            ax.annotate(f"{m} (post)", (x, y_floor), xytext=(6, -10),
                        textcoords="offset points", fontsize=7,
                        color="#996600")

    ax.set_xscale("log")
    ax.set_xlabel("median connected_components (log scale)")
    ax.set_ylabel("ELO (3D Arena leaderboard)")
    ax.grid(True, which="both", alpha=0.3)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#6db3f2", ec="black",
                      label="ELO known (paper Table 1)"),
        plt.Rectangle((0, 0), 1, 1, fc="#ffb84d", ec="black",
                      label="post-paper (no ELO)"),
        plt.Rectangle((0, 0), 1, 1, fc="#cccccc", ec="black",
                      label="no ELO match"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=140, bbox_inches="tight")
    print(f"\nSaved: {FIG_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
