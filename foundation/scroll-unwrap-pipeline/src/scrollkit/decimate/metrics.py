"""Pure-numpy/scipy mesh + UV metrics backing the M2 geometry gates.

All UV comparisons are bit-exact (uint32 views of float32) — consistent with the
mesh-io-conventions rule that dedup/equality never uses epsilon merging.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def uv_signed_areas(wedge_uv: np.ndarray) -> np.ndarray:
    """Twice the signed UV-space area per face (float64). wedge_uv: (m,3,2)."""
    w = wedge_uv.astype(np.float64)
    e1 = w[:, 1] - w[:, 0]
    e2 = w[:, 2] - w[:, 0]
    return e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]


def uv_orientation_counts(wedge_uv: np.ndarray) -> dict:
    """Sign distribution of UV triangle areas: {flipped, zero, positive}."""
    a = uv_signed_areas(wedge_uv)
    return {
        "flipped": int((a < 0).sum()),
        "zero": int((a == 0).sum()),
        "positive": int((a > 0).sum()),
    }


def uv_chart_count(faces: np.ndarray, wedge_uv: np.ndarray) -> int:
    """Number of UV-connected components (charts).

    Corners weld into a single UV-vertex iff (mesh vertex id, exact UV bits) coincide;
    faces sharing a welded edge are UV-connected. Chart count = connected components of
    the UV-vertex graph (every UV-vertex belongs to >=1 face, so none are isolated).
    """
    m = int(faces.shape[0])
    if m == 0:
        return 0
    vid = faces.astype(np.int64).reshape(-1)  # (3m,)
    uvb = np.ascontiguousarray(wedge_uv.astype(np.float32, copy=False)).reshape(-1, 2).view(np.uint32)
    key = np.empty(3 * m, dtype=[("v", "i8"), ("u", "u4"), ("t", "u4")])
    key["v"] = vid
    key["u"] = uvb[:, 0]
    key["t"] = uvb[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    c = inv.reshape(m, 3)
    n = int(inv.max()) + 1
    rows = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
    cols = np.concatenate([c[:, 1], c[:, 2], c[:, 0]])
    g = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    ncomp, _ = connected_components(g, directed=False)
    return int(ncomp)


def edge_stats(vertices: np.ndarray, faces: np.ndarray) -> dict:
    """Boundary length/count + non-manifold edge count from raw arrays."""
    f = faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    nv = int(vertices.shape[0])
    key = e[:, 0] * nv + e[:, 1]
    uniq, counts = np.unique(key, return_counts=True)
    bnd = uniq[counts == 1]
    a, b = bnd // nv, bnd % nv
    v64 = vertices.astype(np.float64)
    blen = float(np.linalg.norm(v64[a] - v64[b], axis=1).sum())
    return {
        "edge_count": int(uniq.size),
        "boundary_edge_count": int((counts == 1).sum()),
        "boundary_length": blen,
        "nonmanifold_edge_count": int((counts > 2).sum()),
    }


def isolated_vertex_count(n_vertices: int, faces: np.ndarray) -> int:
    used = np.zeros(n_vertices, dtype=bool)
    used[faces.reshape(-1)] = True
    return int((~used).sum())


def mean_edge_length(vertices: np.ndarray, faces: np.ndarray) -> float:
    f = faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    nv = int(vertices.shape[0])
    uniq = np.unique(e[:, 0] * nv + e[:, 1])
    a, b = uniq // nv, uniq % nv
    v64 = vertices.astype(np.float64)
    return float(np.linalg.norm(v64[a] - v64[b], axis=1).mean())


def wedge_to_vertex_uv_exact(n_vertices: int, faces: np.ndarray, wedge_uv: np.ndarray) -> np.ndarray | None:
    """Convert wedge UVs to per-vertex UVs ONLY if bit-identical per vertex.

    Returns (n,2) float32 when every referenced vertex carries exactly one distinct
    (bit-exact) wedge UV across all its corners; otherwise None (caller keeps the
    wedge-dedup OBJ path). Unreferenced vertices would get UV (0,0) — callers must
    have removed those already.
    """
    vid = faces.astype(np.int64).reshape(-1)
    uv32 = np.ascontiguousarray(wedge_uv.astype(np.float32, copy=False)).reshape(-1, 2)
    out = np.zeros((n_vertices, 2), dtype=np.float32)
    out[vid] = uv32  # last write wins; equality below proves they all agreed
    if not np.array_equal(out[vid].view(np.uint32), uv32.view(np.uint32)):
        return None
    return out


def repair_uv_flips(faces: np.ndarray, wedge_uv: np.ndarray, max_iters: int = 25) -> tuple[np.ndarray, dict]:
    """Untangle flipped UV triangles left by the texture-aware quadric collapse.

    The collapse occasionally inverts a handful of tiny interior UV slivers (observed:
    ~1e-7 UV area, a few texels). Geometry is NOT touched; the offending UV-vertices
    (welded by exact (vertex id, uv bits)) are iteratively moved to the centroid of
    their UV neighbors (uniform-Laplacian untangling) until the global flipped count
    is zero IN FLOAT32 (the dtype the gates and the OBJ see). UV-boundary vertices are
    never moved, so chart outlines (and the texture-alpha silhouette) stay fixed.

    Returns (repaired wedge_uv float32, info dict). If the loop cannot reach zero
    flips the partially-repaired table is returned and the caller's gate fails
    honestly (info["flips_after"] > 0).
    """
    m = int(faces.shape[0])
    vid = faces.astype(np.int64).reshape(-1)
    uv32 = np.ascontiguousarray(wedge_uv.astype(np.float32, copy=True)).reshape(-1, 2)
    flips_before = int((uv_signed_areas(uv32.reshape(m, 3, 2)) < 0).sum())
    if flips_before == 0:
        return uv32.reshape(m, 3, 2), {"flips_before": 0, "flips_after": 0, "iters": 0, "moved_uvverts": 0}

    bits = uv32.view(np.uint32)
    key = np.empty(3 * m, dtype=[("v", "i8"), ("u", "u4"), ("t", "u4")])
    key["v"] = vid
    key["u"] = bits[:, 0]
    key["t"] = bits[:, 1]
    _, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    c = inv.reshape(m, 3)  # corner -> uvvert
    n = int(inv.max()) + 1
    uv = uv32[first_idx].astype(np.float64)  # working copy, one row per uvvert

    # UV-space edges (both directions) + boundary lock
    ea = np.concatenate([c[:, 0], c[:, 1], c[:, 2]])
    eb = np.concatenate([c[:, 1], c[:, 2], c[:, 0]])
    pair = np.minimum(ea, eb).astype(np.int64) * n + np.maximum(ea, eb)
    uniq_pair, pair_counts = np.unique(pair, return_counts=True)
    boundary_pairs = uniq_pair[pair_counts == 1]
    locked = np.zeros(n, dtype=bool)
    locked[boundary_pairs // n] = True
    locked[boundary_pairs % n] = True

    src = np.concatenate([ea, eb])  # neighbor accumulation: src receives dst's uv
    dst = np.concatenate([eb, ea])

    moved = np.zeros(n, dtype=bool)
    iters = 0
    prev_bad: int | None = None
    stall = 0
    for iters in range(1, max_iters + 1):
        tri = uv[c]  # float64 (m,3,2)
        e1 = tri[:, 1] - tri[:, 0]
        e2 = tri[:, 2] - tri[:, 0]
        area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        # check in float32 too: the gate (and the written OBJ) sees float32
        area32 = uv_signed_areas(tri.astype(np.float32))
        bad_faces = (area2 < 0) | (area32 < 0)
        nbad = int(bad_faces.sum())
        if nbad == 0:
            break
        offend = np.zeros(n, dtype=bool)
        offend[c[bad_faces].reshape(-1)] = True
        # pure point-Laplacian can stall on tangles — when the bad count stops
        # improving, progressively widen the relaxed region by UV one-rings.
        if prev_bad is not None and nbad >= prev_bad:
            stall += 1
            for _ in range(stall):
                grow = np.zeros(n, dtype=bool)
                grow[dst[offend[src]]] = True
                offend |= grow
        else:
            stall = 0
        prev_bad = nbad
        offend &= ~locked
        if not offend.any():
            break  # everything locked — cannot repair
        sel = offend[src]
        acc = np.zeros((n, 2))
        cnt = np.zeros(n)
        np.add.at(acc, src[sel], uv[dst[sel]])
        np.add.at(cnt, src[sel], 1.0)
        upd = offend & (cnt > 0)
        uv[upd] = acc[upd] / cnt[upd, None]
        moved |= upd

    out = uv.astype(np.float32)[c].reshape(m, 3, 2)
    flips_after = int((uv_signed_areas(out) < 0).sum())
    return out, {
        "flips_before": flips_before,
        "flips_after": flips_after,
        "iters": iters,
        "moved_uvverts": int(moved.sum()),
    }


def mesh_stats(vertices: np.ndarray, faces: np.ndarray, wedge_uv: np.ndarray) -> dict:
    """All gate-relevant scalar stats for one mesh state."""
    uv = uv_orientation_counts(wedge_uv)
    es = edge_stats(vertices, faces)
    return {
        "n_vertices": int(vertices.shape[0]),
        "n_faces": int(faces.shape[0]),
        "uv_flipped": uv["flipped"],
        "uv_zero": uv["zero"],
        "uv_positive": uv["positive"],
        "uv_charts": uv_chart_count(faces, wedge_uv),
        "boundary_edge_count": es["boundary_edge_count"],
        "boundary_length": es["boundary_length"],
        "nonmanifold_edge_count": es["nonmanifold_edge_count"],
        "isolated_vertices": isolated_vertex_count(int(vertices.shape[0]), faces),
        "uv_min": [float(x) for x in wedge_uv.reshape(-1, 2).min(axis=0)],
        "uv_max": [float(x) for x in wedge_uv.reshape(-1, 2).max(axis=0)],
    }
