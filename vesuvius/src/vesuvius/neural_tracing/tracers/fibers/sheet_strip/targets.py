"""Strip targets: the sheet height field of a crop from the tifxyz patches its fiber lies on.

A patch window (its grid nodes around the fiber, in crop-local (a, b, c) = (u, v, f) coordinates) is cut into two
triangles per grid cell whose three nodes are valid. Every strip grid point (f, v) that a triangle covers in the
(v, f) plane takes the triangle's height ``u`` there (barycentric). A grid point is supervised when at least one
triangle covers it and all covering triangles of every patch agree within ``fold_tolerance`` (no fold, no patches
disagreeing about the sheet), none of them is steeper than ``max_slope_deg`` against the crop's ``(v, f)`` plane (a
height field is ill-defined there) and its height lies within ``max_height`` of the axis. Everything else is
masked: a patch edge is not a sheet edge.
"""
from __future__ import annotations

import numpy as np


def patch_triangles(local, valid):
    """Triangles (T, 3, 3) of a window's crop-local nodes (h, w, 3) whose three nodes are valid (h, w)."""
    p00, p10, p01, p11 = local[:-1, :-1], local[1:, :-1], local[:-1, 1:], local[1:, 1:]
    v00, v10, v01, v11 = valid[:-1, :-1], valid[1:, :-1], valid[:-1, 1:], valid[1:, 1:]
    first = np.stack((p00, p10, p01), -2)[v00 & v10 & v01]
    second = np.stack((p11, p01, p10), -2)[v11 & v01 & v10]
    return np.concatenate((first, second)).astype(np.float64)


def triangle_heights(triangles, f, v, max_slope_deg):
    """For triangles (T, 3, 3) of (a, b, c) points and the grid rows ``f`` (Nf,) and columns ``v`` (Nv,): which
    triangle covers which grid point (T, Nf*Nv), its height ``a`` there, and whether the triangle is steep (T,)."""
    gv, gf = np.meshgrid(v, f)  # (Nf, Nv): b = v, c = f
    gb, gc = gv.reshape(-1), gf.reshape(-1)
    if not len(triangles):
        return np.zeros((0, len(gb)), bool), np.zeros((0, len(gb))), np.zeros(0, bool)
    a, b, c = triangles[..., 0], triangles[..., 1], triangles[..., 2]
    # Keep triangles whose (b, c) box meets the grid.
    near = ((b.max(1) >= gb.min()) & (b.min(1) <= gb.max()) & (c.max(1) >= gc.min()) & (c.min(1) <= gc.max()))
    a, b, c, triangles = a[near], b[near], c[near], triangles[near]
    normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    length = np.linalg.norm(normal, axis=1)
    steep = np.abs(normal[:, 0]) < np.cos(np.radians(max_slope_deg))*np.maximum(length, 1e-12)
    area = (b[:, 1]-b[:, 0])*(c[:, 2]-c[:, 0])-(b[:, 2]-b[:, 0])*(c[:, 1]-c[:, 0])  # twice the signed (b, c) area
    flat = np.abs(area) < 1e-9
    area = np.where(flat, 1., area)
    db, dc = gb[None]-b[:, :1], gc[None]-c[:, :1]
    l1 = (db*(c[:, 2:3]-c[:, :1])-(b[:, 2:3]-b[:, :1])*dc)/area[:, None]
    l2 = ((b[:, 1:2]-b[:, :1])*dc-db*(c[:, 1:2]-c[:, :1]))/area[:, None]
    l0 = 1-l1-l2
    eps = 1e-6
    covers = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps) & ~flat[:, None]  # edge-on triangles cover nothing
    height = l0*a[:, :1]+l1*a[:, 1:2]+l2*a[:, 2:3]
    return covers, height, steep


def strip_targets(windows, f, v, *, fold_tolerance=2., max_slope_deg=70., max_height=None):
    """Heights (Nf, Nv) float32 and supervision mask (Nf, Nv) bool from patch ``windows``: a list of crop-local node
    grids (h, w, 3) and their validity (h, w), one per patch (module docstring)."""
    count = np.zeros(len(f)*len(v), np.int64)
    total = np.zeros(len(f)*len(v))
    low = np.full(len(f)*len(v), np.inf)
    high = np.full(len(f)*len(v), -np.inf)
    bad = np.zeros(len(f)*len(v), bool)
    for local, valid in windows:
        if min(local.shape[:2]) < 2:
            continue
        covers, height, steep = triangle_heights(patch_triangles(local, valid), f, v, max_slope_deg)
        if not len(covers):
            continue
        bad |= (covers & steep[:, None]).any(0)
        good = covers & ~steep[:, None]
        count += good.sum(0)
        total += np.where(good, height, 0.).sum(0)
        low = np.minimum(low, np.where(good, height, np.inf).min(0))
        high = np.maximum(high, np.where(good, height, -np.inf).max(0))
    mean = total/np.maximum(count, 1)
    mask = (count > 0) & ~bad & (high-low <= fold_tolerance)
    if max_height is not None:
        mask &= np.abs(mean) <= max_height
    shape = (len(f), len(v))
    return np.where(mask, mean, 0.).astype(np.float32).reshape(shape), mask.reshape(shape)
