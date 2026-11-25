"""Tiled local blend: refine a GrowPatch mesh toward a second trace of the same surface prediction.

Ported from the GrowPatch blend benchmark (growbench_tiled_blend.py, MIT). Tiles sit every SPACING voxels on the
mesh. In each tile a graph trace of the prediction runs from the mesh, and both surfaces are cast onto parallel
rays along the tile normal. Where both hit exactly one sheet within LIMIT voxels of each other, the mesh point moves
halfway toward the trace; everywhere else it stays. Tile moves are blended with cos^2 weights onto a 4x denser grid.
A tile whose trace finds no ridge near the mesh changes nothing.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import RADIUS, STEP, faces, frame, project
from .graph_trace import solve

LIMIT = 8.0            # largest disagreement, in voxels, at which a point still moves
SPACING = 32           # tile spacing in voxels (locked in the benchmark)
VISIBLE_TOL = 3.0      # a vertex takes a tile's move only if it lies this close to the tile's own surface depth
NEIGHBOR_RADIUS = 24.0
N_RAYS = 2 * (RADIUS // STEP) + 1


def grid_valid(xyz):
    return np.isfinite(xyz).all(-1) & (xyz >= 0).all(-1)


def upsample(xyz, valid, k):
    """Bilinear k-fold resampling of every complete quad. Shared edges get identical values."""
    xyz = np.asarray(xyz, dtype=np.float64)
    h, w, _ = xyz.shape
    full = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    out = np.full(((h - 1) * k + 1, (w - 1) * k + 1, 3), -1.0)
    ok = np.zeros(out.shape[:2], bool)
    I, J = np.nonzero(full)
    a00, a01, a10, a11 = xyz[I, J], xyz[I, J + 1], xyz[I + 1, J], xyz[I + 1, J + 1]
    for a in range(k + 1):
        for b in range(k + 1):
            s, t = a / k, b / k
            out[I * k + a, J * k + b] = (1 - s) * (1 - t) * a00 + (1 - s) * t * a01 + s * (1 - t) * a10 + s * t * a11
            ok[I * k + a, J * k + b] = True
    return out, ok


def grid_step(xyz, valid):
    both = valid[:, 1:] & valid[:, :-1]
    d = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)[both]
    return float(np.median(d)) if d.size else float("nan")


def tile_seeds(xyz, valid, spacing_vox):
    step = grid_step(xyz, valid)
    s = max(1, int(round(spacing_vox / step)))
    rows, cols = np.arange(s // 2, xyz.shape[0], s), np.arange(s // 2, xyz.shape[1], s)
    pts = xyz[valid]
    seeds = []
    for i in rows:
        for j in cols:
            if not valid[i, j]:
                continue
            p = xyz[i, j]
            near = pts[np.linalg.norm(pts - p, axis=1) <= NEIGHBOR_RADIUS]
            if len(near) < 6:
                continue
            centered = near - near.mean(0)
            _, vecs = np.linalg.eigh(centered.T @ centered)
            normal = vecs[:, 0]
            jj = j + 1 if j + 1 < xyz.shape[1] and valid[i, j + 1] else j - 1
            if jj < 0 or not valid[i, jj]:
                continue
            u = (xyz[i, jj] - p) * (1 if jj > j else -1)
            u = u - np.dot(u, normal) * normal
            if np.linalg.norm(u) < 1e-6:
                continue
            u /= np.linalg.norm(u)
            seeds.append(dict(grid_ij=[int(i), int(j)], seed_xyz=p.tolist(), normal_xyz=normal.tolist(),
                              tangent_u_xyz=u.tolist(), tangent_v_xyz=np.cross(normal, u).tolist()))
    return seeds, s, step


_VOLUME = None


def _init_worker(volume):
    global _VOLUME
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _VOLUME = np.load(volume, mmap_mode="r") if isinstance(volume, (str, os.PathLike)) else volume


def tile_shift(task):
    """Move field for one tile, or the reason it abstained."""
    site, tri = task["site"], task["triangles"]
    stock = project(tri, site)
    try:
        info = solve(_VOLUME, site)
    except ValueError as exc:
        return dict(index=task["index"], abstained=str(exc))
    gtri, _ = faces(info["field"]["xyz"], info["field"]["valid"])
    graph = project(gtri, site)
    unique_s = np.isfinite(stock.low) & (stock.high - stock.low <= 1)
    unique_g = np.isfinite(graph.low) & (graph.high - graph.low <= 1)
    both = unique_s & unique_g
    gap = np.zeros(stock.low.shape)
    gap[both] = graph.center[both] - stock.center[both]
    shared = both & (np.abs(gap) <= LIMIT)
    shift = np.where(shared, 0.5 * gap, 0.0)
    return dict(index=task["index"], shift=shift.astype(np.float32), center=np.where(unique_s, stock.center, np.nan).astype(np.float32),
                unique=unique_s, shared=shared)


def transfer(points, results, sites):
    """cos^2-weighted combination of the tile moves on the resampled vertices (shared-only)."""
    acc = np.zeros_like(points)
    wsum = np.zeros(len(points))
    for res in results:
        if "shift" not in res:
            continue
        site = sites[res["index"]]
        F = frame(site)
        seed = np.asarray(site["seed_xyz"], float)
        lo, hi = seed - RADIUS * 1.75, seed + RADIUS * 1.75
        cand = np.flatnonzero(((points >= lo) & (points <= hi)).all(1))
        if not len(cand):
            continue
        loc = (points[cand] - seed) @ F.T
        r = np.hypot(loc[:, 0], loc[:, 1])
        inside = r < RADIUS
        cand, loc, r = cand[inside], loc[inside], r[inside]
        fu, fv = (loc[:, 0] + RADIUS) / STEP, (loc[:, 1] + RADIUS) / STEP
        iu = np.clip(np.rint(fu).astype(int), 0, N_RAYS - 1)
        iv = np.clip(np.rint(fv).astype(int), 0, N_RAYS - 1)
        center = res["center"][iv, iu]
        visible = res["unique"][iv, iu] & np.isfinite(center) & (np.abs(loc[:, 2] - np.nan_to_num(center)) <= VISIBLE_TOL)
        visible &= res["shared"][iv, iu]
        if not visible.any():
            continue
        cand, fu, fv, r = cand[visible], fu[visible], fv[visible], r[visible]
        s = map_coordinates(res["shift"], [fv, fu], order=1, mode="nearest")
        w = np.cos(np.pi * r / (2 * RADIUS)) ** 2
        acc[cand] += (w * s)[:, None] * F[2]
        wsum[cand] += w
    disp = np.zeros_like(points)
    has = wsum > 0
    disp[has] = acc[has] / wsum[has, None]
    return disp


def tiled_blend(xyz, valid, volume, origin_xyz, spacing_vox=SPACING, workers=1):
    """Refine a GrowPatch vertex grid. Returns (refined xyz, validity, per-vertex move, report).

    `volume` is the ZYX surface prediction (array or .npy path) whose first voxel sits at global `origin_xyz`.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    sites, lattice, step = tile_seeds(xyz, valid, spacing_vox)
    for k, s in enumerate(sites):
        s["id"] = f"tile_{k:04d}"
        s["origin_xyz"] = [float(c) for c in origin_xyz]
    tri_all, _ = faces(xyz, valid)
    centroids = tri_all.mean(1)
    tasks = [dict(index=k, site=s,
                  triangles=tri_all[np.linalg.norm(centroids - np.asarray(s["seed_xyz"]), axis=1) <= RADIUS * 1.75 + 2 * step])
             for k, s in enumerate(sites)]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(volume,)) as pool:
            results = list(pool.map(tile_shift, tasks, chunksize=1))
    else:
        _init_worker(volume)
        results = [tile_shift(t) for t in tasks]
    k = max(1, int(round(step / STEP)))
    up, ok = upsample(xyz, valid, k)
    disp = transfer(up[ok], results, sites)
    out = up.copy()
    out[ok] = up[ok] + disp
    moved = np.linalg.norm(disp, axis=1)
    field = np.zeros(up.shape)
    field[ok] = disp
    report = dict(tiles=len(sites), tiles_solved=sum("shift" in r for r in results),
                  tiles_abstained=sum("abstained" in r for r in results), upsample=k,
                  vertices=int(ok.sum()), vertices_moved=int((moved > 1e-6).sum()),
                  move_median_vox=float(np.median(moved[moved > 1e-6])) if (moved > 1e-6).any() else 0.0,
                  move_p95_vox=float(np.percentile(moved[moved > 1e-6], 95)) if (moved > 1e-6).any() else 0.0)
    return out, ok, field, report
