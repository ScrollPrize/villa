"""Crop geometry inside a point's 256^3 parent block (ZYX voxel coordinates, float64).

A training view is a 128^3 crop of the parent at `offset`, put through one of the 48 cube symmetries (axis
permutation, then flips) and, for teacher stage 1 only, a rotation. Column-vector convention:
    v = A (q - cin) + 63.5,   cin = offset + 63.5,
with q in parent coordinates and v in view coordinates; the image is pulled with A^T.
"""
import itertools

import numpy as np

CENTER = np.full(3, 63.5)
CORNERS = np.array(list(itertools.product((0., 127.), repeat=3)))
PERMS = list(itertools.permutations(range(3)))
# Crop offsets reserved for evaluation placements in the original runs; training never draws them.
EVAL_OFFSETS = np.array([[64, 64, 64], [23, 87, 105], [101, 35, 69]])


def rotation(angles):
    a, b, c = np.deg2rad(angles)
    ra = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rb = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rc = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rc @ rb @ ra


def construct(offset, perm, flips, angles):
    discrete = np.eye(3)[list(perm)]
    discrete = np.diag([(-1 if f else 1) for f in flips]) @ discrete
    A = discrete @ rotation(angles)
    cin = np.asarray(offset, np.float64) + CENTER
    M = A.T
    b = cin - M @ CENTER
    source = CORNERS @ M.T + b
    return dict(forward=A, input_center=cin, pull_matrix=M, pull_offset=b, source_corners=source)


def points(q, t):
    """Parent coordinates -> view coordinates."""
    return (np.asarray(q, np.float64) - t['input_center']) @ t['forward'].T + CENTER


def inverse(v, t):
    """View coordinates -> parent coordinates."""
    return (np.asarray(v, np.float64) - CENTER) @ t['forward'] + t['input_center']


def valid(t, anchor):
    """Crop inside the parent and the anchor point at least 3.5 voxels inside the view."""
    corners = t['source_corners']
    p = points(np.asarray(anchor)[None], t)[0]
    return bool(np.all(corners >= 0) & np.all(corners <= 255) & np.all(p >= 3.5) & np.all(p <= 123.5))


def visible_ids(parent_points, ids, t):
    """ids whose view coordinates lie in [3.5, 123.5]^3, their view coordinates (float32) and the mask."""
    q = points(parent_points, t)
    mask = np.all((q >= 3.5) & (q <= 123.5), axis=1)
    return np.asarray(ids)[mask].copy(), q[mask].astype(np.float32), mask


def support_box(t, parent):
    """Closed integer box (absolute coordinates) of the voxels a crop interpolates from."""
    lo = np.floor(t['source_corners'].min(0)).astype(int) + parent
    hi = np.ceil(t['source_corners'].max(0)).astype(int) + parent
    return lo, hi


def intersects(lo, hi, parents):
    """Does the closed box [lo, hi] touch any closed 256^3 box at `parents`?"""
    return bool(np.any(np.all((lo <= parents + 255) & (hi >= parents), axis=1)))


def draw_offset(rng):
    """Crop offset in [16, 112]^3, redrawn if it equals an evaluation placement."""
    while True:
        x = rng.integers(16, 113, 3)
        if not np.any(np.all(EVAL_OFFSETS == x, axis=1)):
            return x
