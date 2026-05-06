"""
pymeshlab cross-check for watertight and manifold_edge_ratio.

For each mesh in all_metrics.csv (load_success=True):
  1. Load with pymeshlab, call get_topological_measures()
  2. Derive the same two metrics using pymeshlab fields:
       pml_manifold_edge_ratio = (edges - nm_edges - boundary_edges) / edges
       pml_watertight          = (nm_edges == 0) AND (boundary_edges == 0)
  3. Compare with trimesh values already stored in all_metrics.csv
  4. Flag discrepancies (|Δ| > FLOAT_TOL for ratio, exact for bool)

Metric equivalence:
  manifold_edge_ratio:
    trimesh  → Counter(edges_sorted), ratio of count==2 edges
    pymeshlab→ (edges_number - non_two_manifold_edges - boundary_edges) / edges_number
  watertight:
    trimesh  → is_watertight (manifold + no-boundary + consistent winding)
    pymeshlab→ non_two_manifold_edges==0 AND boundary_edges==0
    * Winding consistency is NOT checked by pymeshlab, so rare disagreements
      on this axis are expected and flagged separately.

Output: data/crosscheck_results.csv
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pymeshlab

ROOT      = Path(__file__).resolve().parent.parent
MESH_DIR  = ROOT / "meshes" / "outputs"
DATA      = ROOT / "data"
OUT_CSV   = DATA / "crosscheck_results.csv"
LOG_EVERY = 100
FLOAT_TOL = 1e-3


def pml_measures(path: str) -> dict | None:
    """Load mesh with pymeshlab and return topological measures dict."""
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        m = ms.get_topological_measures()
        edges     = m["edges_number"]
        nm_edges  = m["non_two_manifold_edges"]
        bnd_edges = m["boundary_edges"]
        if edges > 0:
            manif_r = (edges - nm_edges - bnd_edges) / edges
        else:
            manif_r = float("nan")
        wt = (nm_edges == 0) and (bnd_edges == 0)
        return {
            "pml_edges":          edges,
            "pml_nm_edges":       nm_edges,
            "pml_boundary_edges": bnd_edges,
            "pml_manifold_edge_ratio": round(manif_r, 6),
            "pml_watertight":     wt,
            "pml_two_manifold":   bool(m["is_mesh_two_manifold"]),
            "pml_error":          None,
        }
    except Exception as e:
        return {
            "pml_edges": None, "pml_nm_edges": None,
            "pml_boundary_edges": None,
            "pml_manifold_edge_ratio": None,
            "pml_watertight": None,
            "pml_two_manifold": None,
            "pml_error": str(e),
        }


def main():
    am = pd.read_csv(DATA / "all_metrics.csv")
    rows = am[am["load_success"] == True].copy().reset_index(drop=True)
    total = len(rows)
    print(f"Cross-check targets: {total} meshes\n")

    results = []
    t_start = time.time()

    for i, row in rows.iterrows():
        path = str(MESH_DIR / row["model"] / row["filename"])
        pml  = pml_measures(path)

        # --- comparison ---
        tm_wt    = bool(row["watertight"])
        tm_manif = float(row["manifold_edge_ratio"])

        if pml["pml_error"]:
            wt_match    = False
            manif_match = False
            wt_note     = f"pml_error: {pml['pml_error']}"
            manif_note  = wt_note
        else:
            pml_wt    = bool(pml["pml_watertight"])
            pml_manif = pml["pml_manifold_edge_ratio"]

            wt_match = (tm_wt == pml_wt)
            manif_match = (
                abs(tm_manif - pml_manif) <= FLOAT_TOL
                if not (pd.isna(tm_manif) or pml_manif is None)
                else False
            )

            # Distinguish winding-only mismatch from real discrepancy
            if not wt_match:
                if not pml_wt and tm_wt:
                    wt_note = "trimesh=WT, pml=not (boundary/nm issue)"
                elif pml_wt and not tm_wt:
                    # pml says closed+manifold but trimesh says not WT:
                    # most likely winding inconsistency (not checked by pml)
                    wt_note = "pml=closed+manifold, trimesh=not WT (winding?)"
                else:
                    wt_note = f"tm={tm_wt} pml={pml_wt}"
            else:
                wt_note = "OK"

            if not manif_match:
                delta = tm_manif - pml["pml_manifold_edge_ratio"]
                manif_note = (
                    f"tm={tm_manif:.4f} pml={pml['pml_manifold_edge_ratio']:.4f} "
                    f"delta={delta:+.4f}"
                )
            else:
                manif_note = "OK"

        rec = {
            "model":              row["model"],
            "filename":           row["filename"],
            # trimesh values
            "tm_watertight":      tm_wt,
            "tm_manifold_edge_ratio": tm_manif,
            # pymeshlab values
            **pml,
            # comparison
            "wt_match":           wt_match,
            "manif_match":        manif_match,
            "wt_note":            wt_note,
            "manif_note":         manif_note,
        }
        results.append(rec)

        if (i + 1) % LOG_EVERY == 0 or (i + 1) == total:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta  = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:4d}/{total}]  {elapsed:.0f}s elapsed  "
                  f"{rate:.1f} mesh/s  ETA {eta:.0f}s")

    df = pd.DataFrame(results)
    df.to_csv(OUT_CSV, index=False, float_format="%.6f")

    # --- summary ---
    n_ok   = len(df[df["pml_error"].isna()])
    n_err  = len(df[df["pml_error"].notna()])

    wt_ok      = df[df["pml_error"].isna()]["wt_match"].sum()
    wt_mismatch= n_ok - wt_ok
    wt_winding = df["wt_note"].str.contains("winding", na=False).sum()

    manif_ok      = df[df["pml_error"].isna()]["manif_match"].sum()
    manif_mismatch= n_ok - manif_ok

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Cross-check complete  ({elapsed:.1f}s, {total} meshes)")
    print(f"{'='*60}")
    print(f"  pymeshlab load errors  : {n_err}")
    print(f"  successfully compared  : {n_ok}")
    print()
    print(f"  watertight matches     : {wt_ok} / {n_ok}")
    print(f"  watertight mismatches  : {wt_mismatch}")
    if wt_mismatch > 0:
        print(f"    of which winding-only: {wt_winding}")
        real_wt = wt_mismatch - wt_winding
        print(f"    real discrepancies   : {real_wt}")
        if real_wt > 0:
            print("    sample:")
            sample = df[(~df["wt_match"]) & (~df["wt_note"].str.contains("winding", na=False))].head(5)
            for _, r in sample.iterrows():
                print(f"      {r['model']:18s} {r['filename'][:40]}  {r['wt_note']}")
    print()
    print(f"  manifold ratio matches : {manif_ok} / {n_ok}  (tol={FLOAT_TOL})")
    print(f"  manifold mismatches    : {manif_mismatch}")
    if manif_mismatch > 0:
        print("  sample:")
        for _, r in df[~df["manif_match"]].head(5).iterrows():
            print(f"    {r['model']:18s} {r['filename'][:40]}  {r['manif_note']}")
    print()
    print(f"  Saved: {OUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
