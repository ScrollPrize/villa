"""Centerline pair sampling for validated neighboring fibers.

All coordinates are crop-local trace voxels (a, b, c) = (u, v, forward); crop
volumes are laid out (c, b, a) like ``crop_local_grid``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree



PAIR_SAMPLING_VERSION = 5


@dataclass(frozen=True)
class ComponentRule:
    """Pair support/clearance settings; the old component sampler is retired."""
    own_radius: float = 1.5  # minimum target clearance
    lateral_max: float = 12.  # matches the bank's mining radius; crop support still applies
    along_window: float = 2.  # negatives within this arc of their positive


def sample_pairs(curve, crop, foreign_local, foreign_nearest, rng, *, positives=4, negatives=8,
                 forward=(1., 20.), margin=4., rule: ComponentRule = ComponentRule(),
                 appearance_crop=None, along_margin=0., path_ids=None, near_fraction=None,
                 near_distance=12., return_metadata=False):
    """Sample both classes uniformly on their interpolated centerlines.

    Full appearance support is checked identically for both. Positives come from
    the annotation and negatives from a validated bank, whose mining already
    required presence support, so neither is re-filtered on the presence crop.
    Negative candidates must come from that bank; no expansion, snapping,
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
    metadata = dict(negative_path_ids=np.full((positives,negatives),-1,np.int64),
                    negative_distance=np.zeros((positives,negatives),np.float32))
    def result():
        values = (pos,pos_mask,neg,neg_mask)
        return (*values,metadata) if return_metadata else values
    if near_fraction is not None and not 0 <= near_fraction <= 1:
        raise ValueError('Near-negative fraction must be in [0,1]')
    path_ids = np.asarray(path_ids if path_ids is not None else np.zeros(len(foreign_local)),np.int64)
    if path_ids.shape != (len(foreign_local),):
        raise ValueError('Every negative candidate needs a path ID')
    eligible = np.flatnonzero((curve[:, 2] >= forward[0]) & (curve[:, 2] <= forward[1])
                             & supported(curve))
    if not len(eligible) or len(curve) < 3:
        return result()
    tree = cKDTree(curve)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))]
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    distance, nearest = tree.query(foreign_local)
    offset = foreign_local-curve[nearest]
    along = (offset*tangent[nearest]).sum(-1)
    lateral = np.linalg.norm(offset-along[:, None]*tangent[nearest], axis=-1)
    usable = (supported(foreign_local) & (distance > rule.own_radius)
              & (nearest > 0) & (nearest < len(curve)-1) & (lateral <= rule.lateral_max))
    for k, chunk in enumerate(np.array_split(eligible, positives)):
        if not len(chunk):
            continue
        j = int(rng.choice(chunk))
        pos[k], pos_mask[k] = curve[j], 1.
        candidates = np.flatnonzero(usable & (np.abs(arc[nearest]-arc[j]) <= rule.along_window))
        if near_fraction is None:
            selections = [(0,rng.choice(candidates,size=min(negatives,len(candidates)),replace=False))] if len(candidates) else []
        else:
            near_count = int(np.floor(negatives*near_fraction+.5))
            selections = []
            for start,count,member in ((0,near_count,lateral <= near_distance),
                                       (near_count,negatives-near_count,lateral > near_distance)):
                ids = candidates[member[candidates]]
                # Visit each stored path once before taking its next point.
                groups = [list(rng.permutation(ids[path_ids[ids] == p])) for p in rng.permutation(np.unique(path_ids[ids]))]
                chosen = []
                while groups and len(chosen) < count:
                    for group in groups:
                        if len(chosen) == count:
                            break
                        chosen.append(group.pop())
                    groups = [g for g in groups if g]
                selections.append((start,np.asarray(chosen,np.int64)))
        for start,chosen in selections:
            slots = slice(start,start+len(chosen))
            neg[k,slots],neg_mask[k,slots] = foreign_local[chosen],1.
            metadata['negative_path_ids'][k,slots] = path_ids[chosen]
            metadata['negative_distance'][k,slots] = lateral[chosen]
    return result()


def crop_indices(crop, local):
    """Continuous volume indices (c, b, a) for crop-local (a, b, c) points."""
    local = np.asarray(local, np.float64).reshape(-1, 3)
    return np.c_[local[:, 2]/crop.spacing+crop.behind, local[:, 1]/crop.spacing+(crop.width-1)/2,
                 local[:, 0]/crop.spacing+(crop.width-1)/2]
