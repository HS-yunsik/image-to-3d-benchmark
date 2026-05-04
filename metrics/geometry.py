"""D1: Geometry metrics.

Each metric returned here measures a property of the mesh's vertex/face
structure. We avoid any rendering-based or learned-perceptual signals.

Metric definitions (Phase 1)
----------------------------
watertight : bool
    True iff every edge of the mesh is shared by exactly 2 faces. This is
    trimesh.is_watertight, which also requires winding-consistency. A
    watertight mesh has no holes and a well-defined inside/outside, which
    is required for boolean ops, slicing, simulation, etc.

manifold_edge_ratio : float in [0, 1]
    Fraction of unique edges that are shared by exactly 2 faces (i.e. proper
    interior edges). Higher is better. 1.0 implies the mesh is closed and
    has no T-junctions; 0.0 means every edge is either a boundary or non-
    manifold. This metric is independent from watertight (a closed sphere
    has 1.0; an open hemisphere also has high ratio but is not watertight).

boundary_edge_ratio : float in [0, 1]
    Fraction of unique edges that lie on a boundary (shared by exactly 1
    face). High values indicate open holes / open surfaces.

nonmanifold_edge_ratio : float in [0, 1]
    Fraction of unique edges shared by 3 or more faces. These are surface
    branches (T-junctions) and break most downstream pipelines (UV unwrap,
    physics, smoothing). Lower is better; ideally 0.0.

connected_components : int >= 0
    Number of disjoint mesh pieces. trimesh.split(only_watertight=False)
    counts every piece, including non-watertight ones. 1 is typical for
    a single object; >1 may mean floating geometry.

vertex_count, face_count : int >= 0
    Raw counts after trimesh.load (which auto-triangulates).

Note: manifold_edge_ratio + boundary_edge_ratio + nonmanifold_edge_ratio == 1.0
(within float precision) for any non-empty mesh.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as _scipy_cc


def _count_components_lightweight(mesh: trimesh.Trimesh) -> int:
    """Count connected components without materializing per-piece Trimeshes.

    mesh.split() builds one Trimesh per piece, which on shattered meshes
    (e.g. MeshFormer with ~98K components) costs many GB of RAM. We get the
    same count by running a scipy sparse-CC on the **face-adjacency graph**:

        nodes = faces
        edges = pairs of faces that share an edge (mesh.face_adjacency)

    This matches `mesh.split(only_watertight=False)` exactly because that is
    precisely the graph trimesh.split walks internally. Memory cost is
    O(n_faces + n_face_adjacency) which is two int32 arrays -- trivial.

    Returns 0 for an empty mesh and 1 if the face-adjacency graph is empty
    but at least one face exists (single-face mesh).
    """
    n_faces = 0 if mesh.faces is None else len(mesh.faces)
    if n_faces == 0:
        return 0

    try:
        adj = mesh.face_adjacency  # shape (n_pairs, 2), int
    except Exception:
        adj = None

    if adj is None or len(adj) == 0:
        return n_faces  # all faces are isolated islands

    rows = np.concatenate([adj[:, 0], adj[:, 1]])
    cols = np.concatenate([adj[:, 1], adj[:, 0]])
    data = np.ones(len(rows), dtype=bool)
    graph = csr_matrix(
        (data, (rows, cols)),
        shape=(n_faces, n_faces),
    )
    n_cc, _ = _scipy_cc(graph, directed=False)
    return int(n_cc)


EMPTY = {
    "watertight": False,
    "manifold_edge_ratio": float("nan"),
    "boundary_edge_ratio": float("nan"),
    "nonmanifold_edge_ratio": float("nan"),
    "connected_components": 0,
    "vertex_count": 0,
    "face_count": 0,
}


def compute(mesh: trimesh.Trimesh) -> dict[str, Any]:
    """Compute all D1 metrics for a single mesh."""
    n_v = len(mesh.vertices) if mesh.vertices is not None else 0
    n_f = len(mesh.faces) if mesh.faces is not None else 0

    if n_v == 0 or n_f == 0:
        return dict(EMPTY)

    watertight = bool(mesh.is_watertight)

    # Edge histogram. mesh.edges_sorted has every face-edge (so 3*n_f rows),
    # each edge sorted so (a, b) and (b, a) collapse. Counter on tuples gives
    # how many faces each unique edge belongs to.
    edge_counts = Counter(map(tuple, mesh.edges_sorted))
    n_unique = len(edge_counts)
    if n_unique == 0:
        manifold_r = boundary_r = nonmanifold_r = float("nan")
    else:
        n_manifold = sum(1 for v in edge_counts.values() if v == 2)
        n_boundary = sum(1 for v in edge_counts.values() if v == 1)
        n_nonman = sum(1 for v in edge_counts.values() if v >= 3)
        manifold_r = n_manifold / n_unique
        boundary_r = n_boundary / n_unique
        nonmanifold_r = n_nonman / n_unique

    n_components = _count_components_lightweight(mesh)

    return {
        "watertight": watertight,
        "manifold_edge_ratio": round(manifold_r, 4) if manifold_r == manifold_r else manifold_r,
        "boundary_edge_ratio": round(boundary_r, 4) if boundary_r == boundary_r else boundary_r,
        "nonmanifold_edge_ratio": round(nonmanifold_r, 4) if nonmanifold_r == nonmanifold_r else nonmanifold_r,
        "connected_components": int(n_components),
        "vertex_count": int(n_v),
        "face_count": int(n_f),
    }
