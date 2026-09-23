"""Shared sparse, occupancy-normalized filtering of signed patch normals."""
from __future__ import annotations

import numpy as np


def cube_neighbors(cells, queries, width=3):
    """Sparse equivalent of indexing a padded dense volume's cubic neighborhood."""
    if width < 1 or width % 2 != 1:
        raise ValueError('kernel width must be a positive odd integer')
    cells, queries = np.asarray(cells, dtype=np.int64), np.asarray(queries, dtype=np.int64)
    radius = width // 2
    offsets = np.stack(np.meshgrid(*([np.arange(-radius, radius+1)] * 3), indexing='ij'), axis=-1).reshape(-1, 3)
    if not len(cells):
        return np.full((len(queries), width**3), -1, dtype=np.int64)
    origin = cells.min(axis=0)
    shape = cells.max(axis=0) - origin + 1
    def keys(v):
        return (v[..., 0] * shape[1] + v[..., 1]) * shape[2] + v[..., 2]
    key = keys(cells - origin)
    order = np.argsort(key)
    ordered = key[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError('input must have at most one normal per voxel')
    neighbor = queries[:, None, :] + offsets - origin
    inside = ((neighbor >= 0) & (neighbor < shape)).all(axis=-1)
    wanted = keys(neighbor)
    at = np.minimum(np.searchsorted(ordered, wanted), len(ordered) - 1)
    found = inside & (ordered[at] == wanted)
    return np.where(found, order[at], -1)


def box_average(normals, neighbors, present=None):
    """Convolve vector*presence and presence, divide, then normalize the vector."""
    if present is None:
        present = np.ones(len(normals), dtype=bool)
    if not len(normals):
        count = np.zeros(len(neighbors), dtype=int)
        return np.zeros((len(neighbors), 3)), count, count.astype(float), count.astype(bool)
    at = np.maximum(neighbors, 0)
    keep = (neighbors >= 0) & present[at]
    count = keep.sum(axis=1)
    total = np.sum(np.where(keep[..., None], normals[at], 0.), axis=1)
    mean = total / np.maximum(count, 1)[:, None]
    length = np.linalg.norm(mean, axis=1)
    valid = (count > 0) & np.isfinite(mean).all(axis=1) & (length > 1e-8)
    unit = np.zeros_like(mean)
    unit[valid] = mean[valid] / length[valid, None]
    return unit, count, length, valid
