# Image-to-3D Production-Fitness Benchmark

A multi-dimensional mesh quality benchmark for image-to-3D generation models, evaluating **production fitness** rather than visual appeal alone.

> **Key insight**: ELO rankings (human preference) show near-zero correlation with geometry quality (r = −0.046), but significant correlation with texture resolution (r = +0.578★). Popular models ≠ production-ready models.

---

## Motivation

Existing image-to-3D evaluation methods (3DGen-Bench, Hi3DEval, MATE-3D) assess **rendered 2D images**, not mesh structure directly. Hi3DEval itself acknowledges: *"based on 2D renderings, these methods inherently struggle to capture the spatial continuity and structural complexity of 3D assets."*

This benchmark is the first to shift evaluation from rendering-based to **direct mesh structure measurement**, filling the gap that 3D Arena (Ebert, 2025) identified as future work:
- *"topology assessment is future work"*
- *"Multi-criteria assessment can disentangle aesthetic appeal from technical utility"*

---

## Evaluation Framework

We measure 7 metrics across 3 production-relevant dimensions:

| Dim | Metric | Description |
|-----|--------|-------------|
| **D1 Geometry** | `watertight_pct` | % of meshes that are watertight |
| | `manifold_edge_ratio` | Non-manifold edge ratio (lower = better) |
| | `connected_components` | Avg. # of disconnected parts |
| **D2 UV** | `has_uv` | Whether UV coordinates are present |
| | `uv_packed_area` | Effective UV utilization area |
| **D3 PBR** | `pbr_channel_count` | # of PBR channels (base/normal/roughness/metallic) |
| | `texture_resolution` | Average texture resolution (px) |

Dimension scores are computed as percentile ranks within the 16 ELO-matched models, then averaged per dimension.

---

## Key Findings

### Spearman Correlation with ELO (n = 16 models)

| Dimension | r | p | |
|-----------|---|---|-|
| D1 Geometry | −0.046 | 0.866 | no correlation |
| D2 UV | +0.461 | 0.073 | trend only |
| D3 PBR | +0.415 | 0.110 | trend only |
| Total (D2+D3) | +0.556 | 0.025 | ★ significant |
| **Total (D1+D2+D3)** | **+0.665** | **0.005** | **★ significant** |
| Texture resolution (n=16) | +0.578 | 0.019 | ★ significant |
| Texture resolution (n=9, textured only) | +0.710 | 0.032 | ★ significant |

### Four Core Findings

| # | Finding | Evidence |
|---|---------|----------|
| **F1** | Geometry quality uncorrelated with human preference | D1 r = −0.046 |
| **F2** | Texture resolution is the strongest single predictor | r = +0.578★ |
| **F3** | Multi-dimensional total score significantly predicts ELO | r = +0.665★ |
| **F4** | 6 of top-8 ELO models have 0% watertight meshes | Popular ≠ production-fit |

### Model Quadrant Analysis (ELO × Watertight)

| Quadrant | Models |
|----------|--------|
| Q1 Visual + Production | InstantMesh, Unique3D |
| Q2 Visual only (largest cluster) | Strawberrry, Strawb3rry, TRELLIS, Zaohaowu3D, Hunyuan3D-2, Meshy-5 |
| Q3 Production only | Hi3DGen, IM-MA |
| Q4 Neither | MeshFormer, SF3D, Real3D, SPAR3D, TripoSR, 3DTopia-XL |

---

## Dataset

- **3D Meshes**: [3d-arena/3d-arena](https://huggingface.co/datasets/3d-arena/3d-arena) (MIT License)
- **Input Images**: [dylanebert/iso3d](https://huggingface.co/datasets/dylanebert/iso3d) (100 images)
- **Coverage**: 20 models × ~100 meshes = **1,993 meshes** evaluated
- **Excluded**: Gaussian Splat models (TRELLIS-3DGS, LGM, SAM-3D-Objects-3DGS) — future work

---

## Repository Structure

```
├── metrics/                # Core metric implementations
│   ├── geometry.py         # D1: watertight, manifold, connected components
│   ├── uv.py               # D2: UV presence, packed area
│   ├── pbr.py              # D3: PBR channels, texture resolution
│   ├── topology.py         # Triangle/quad/ngon ratio
│   └── scale.py            # Vertex/face count
├── scripts/
│   ├── figures/            # Figure & table generation scripts
│   ├── download_all.py     # Download all meshes from HuggingFace
│   ├── analyze_v5.py       # Dimension scoring + Spearman correlation
│   ├── fetch_elo.py        # ELO score collection
│   └── ...
├── run_metrics.py          # Entry point: compute all metrics → CSV
├── data/
│   ├── all_metrics.csv     # Per-mesh metrics (1993 rows)
│   ├── model_summary.csv   # Per-model averages
│   └── dimension_scores_v5.csv
└── outputs/                # Figures and tables
```

---

## Setup

```bash
conda create -n 3darena python=3.11
conda activate 3darena
pip install trimesh huggingface_hub pillow numpy pygltflib scipy matplotlib seaborn
```

## Usage

```bash
# 1. Download meshes
python scripts/download_all.py

# 2. Compute metrics
python run_metrics.py

# 3. Run analysis and generate figures
python scripts/analyze_v5.py
```

---

## Acknowledgements

This benchmark uses mesh outputs from the [3D Arena](https://huggingface.co/datasets/3d-arena/3d-arena) dataset (Ebert, 2025, arXiv:2506.18787), released under the MIT License.

---

## License

MIT
