"""Centerline pair sampling for validated neighboring fibers.

All coordinates are crop-local trace voxels (a, b, c) = (u, v, forward); crop
volumes are laid out (c, b, a) like ``crop_local_grid``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree



PAIR_SAMPLING_VERSION = 3


@dataclass(frozen=True)
class ComponentRule:
    """Pair support/clearance settings; the old component sampler is retired."""
    threshold: float = .7  # presence in [0, 1]
    own_radius: float = 1.5  # minimum target clearance
    lateral_max: float = 12.  # matches the bank's mining radius; crop support still applies
    along_window: float = 2.  # negatives within this arc of their positive


def sample_pairs(curve, presence, crop, foreign_local, foreign_nearest, rng, *, positives=4, negatives=8,
                 forward=(1., 20.), margin=4., rule: ComponentRule = ComponentRule(),
                 appearance_crop=None, along_margin=0.):
    """Sample both classes uniformly on their interpolated centerlines.

    Presence and full appearance support are checked identically for both.
    Negative candidates must come from a validated bank; no expansion, snapping,
    phase adjustment, independent image augmentation or invented history is used.
    """
    curve = np.asarray(curve, np.float64)
    foreign_local = np.asarray(foreign_local, np.float64).reshape(-1, 3)
    support = appearance_crop or crop
    half = (support.width-1)*support.spacing/2-margin
    lower = np.array([-half, -half, -support.behind*support.spacing+along_margin])
    upper = np.array([half, half, (support.depth-1-support.behind)*support.spacing-along_margin])
    def supported(points):
        return ((points >= lower) & (points <= upper)).all(-1)
    pos = np.zeros((positives, 3), np.float32)
    pos_mask = np.zeros(positives, np.float32)
    neg = np.zeros((positives, negatives, 3), np.float32)
    neg_mask = np.zeros((positives, negatives), np.float32)
    eligible = np.flatnonzero((curve[:, 2] >= forward[0]) & (curve[:, 2] <= forward[1])
                             & supported(curve) & (volume_at(presence,crop,curve,order=1) >= rule.threshold))
    if not len(eligible) or len(curve) < 3:
        return pos, pos_mask, neg, neg_mask
    tree = cKDTree(curve)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))]
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    distance, nearest = tree.query(foreign_local)
    offset = foreign_local-curve[nearest]
    along = (offset*tangent[nearest]).sum(-1)
    lateral = np.linalg.norm(offset-along[:, None]*tangent[nearest], axis=-1)
    usable = (supported(foreign_local) & (distance > rule.own_radius)
              & (nearest > 0) & (nearest < len(curve)-1) & (lateral <= rule.lateral_max)
              & (volume_at(presence,crop,foreign_local,order=1) >= rule.threshold))
    for k, chunk in enumerate(np.array_split(eligible, positives)):
        if not len(chunk):
            continue
        j = int(rng.choice(chunk))
        pos[k], pos_mask[k] = curve[j], 1.
        candidates = np.flatnonzero(usable & (np.abs(arc[nearest]-arc[j]) <= rule.along_window))
        take = min(negatives, len(candidates))
        if take:
            chosen = rng.choice(candidates, size=take, replace=False)
            neg[k, :take], neg_mask[k, :take] = foreign_local[chosen], 1.
    return pos, pos_mask, neg, neg_mask


def crop_indices(crop, local):
    """Continuous volume indices (c, b, a) for crop-local (a, b, c) points."""
    local = np.asarray(local, np.float64).reshape(-1, 3)
    return np.c_[local[:, 2]/crop.spacing+crop.behind, local[:, 1]/crop.spacing+(crop.width-1)/2,
                 local[:, 0]/crop.spacing+(crop.width-1)/2]


def volume_at(volume, crop, local, *, order=0):
    """Nearest (default) or trilinear crop values at local points; zero outside."""
    index = crop_indices(crop, local)
    if order == 1:
        return ndimage.map_coordinates(np.asarray(volume, np.float32), index.T, order=1,
                                       mode='constant', cval=0., prefilter=False)
    if order != 0:
        raise ValueError('Only nearest or trilinear crop sampling is supported')
    index = np.rint(index).astype(np.int64)
    inside = np.all((index >= 0) & (index < np.asarray(np.shape(volume))), axis=1)
    out = np.zeros(len(index), np.asarray(volume).dtype)
    out[inside] = np.asarray(volume)[tuple(index[inside].T)]
    return out
