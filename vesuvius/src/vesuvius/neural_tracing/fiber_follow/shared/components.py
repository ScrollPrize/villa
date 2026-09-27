"""Presence components beside a traced fiber: negatives without annotating neighbors.

All coordinates are crop-local trace voxels (a, b, c) = (u, v, forward); crop
volumes are laid out (c, b, a) like ``crop_local_grid``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid


PAIR_SAMPLING_VERSION = 3


@dataclass(frozen=True)
class ComponentRule:
    threshold: float = .7  # presence in [0, 1]
    own_radius: float = 1.5  # a component reaching this close to the annotation is the traced fiber
    lateral_max: float = 10.  # farther components are not "at the same place"
    along_window: float = 2.  # negatives within this arc of their positive
    min_voxels: int = 8  # smallest qualifying lateral piece, in crop samples


def lateral_components(presence, crop: CropSpec, curve, rule: ComponentRule = ComponentRule()):
    """Label presence components and mark those lying beside the annotation.

    ``curve`` (N, 3) is the annotated fiber, densely sampled in traversal order
    and cut to its annotated range. A component is the traced fiber's own if
    any voxel lies within ``own_radius`` of the curve; touching neighbors merge
    into it and never become negatives. Every other component is lateral where
    its nearest curve point is interior: points whose nearest annotation is a
    curve end lie in front of or behind it (for example across a presence gap
    or past an unannotated end) and are excluded.

    Returns ``foreign`` (bool volume), the foreign voxels' ``local``
    coordinates and ``nearest`` curve indices, the component ``labels`` and the
    traced fiber's ``own`` labels, and a small ``counts`` summary.
    """
    curve = np.asarray(curve, np.float64)
    shape = (crop.depth, crop.width, crop.width)
    labels, count = ndimage.label(np.asarray(presence) >= rule.threshold, structure=np.ones((3, 3, 3)))
    index = np.argwhere(labels > 0)
    if len(curve) < 3 or not len(index):
        return dict(foreign=np.zeros(shape, bool), local=np.zeros((0, 3)), nearest=np.zeros(0, np.int64),
                    labels=labels, own=np.zeros(0, labels.dtype),
                    counts=dict(components=int(count), own=0, foreign_components=0))
    local = crop_local_grid(crop)[tuple(index.T)]
    distance, nearest = cKDTree(curve).query(local)
    component = labels[tuple(index.T)]
    own = np.unique(component[distance <= rule.own_radius])
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    offset = local-curve[nearest]
    along = (offset*tangent[nearest]).sum(-1)
    lateral = np.linalg.norm(offset-along[:, None]*tangent[nearest], axis=-1)
    beside = ((nearest > 0) & (nearest < len(curve)-1) & (lateral <= rule.lateral_max)
              & ~np.isin(component, own))
    sizes = np.bincount(component[beside], minlength=count+1)
    qualified = np.flatnonzero(sizes >= rule.min_voxels)
    keep = beside & np.isin(component, qualified)
    foreign = np.zeros(shape, bool)
    foreign[tuple(index[keep].T)] = True
    return dict(foreign=foreign, local=local[keep], nearest=nearest[keep], labels=labels, own=own,
                counts=dict(components=int(count), own=int(len(own)), foreign_components=int(len(qualified))))


def sample_pairs(curve, presence, crop, foreign_local, foreign_nearest, rng, *, positives=4, negatives=8,
                 forward=(1., 20.), margin=4., rule: ComponentRule = ComponentRule(), extra_negative=None,
                 appearance_crop=None, along_margin=0., component_labels=None, centerlines=False):
    """Positives on the annotation ahead, each with negatives beside it.

    Positives are spread over annotated points ``forward`` voxels ahead whose
    appearance receptive field (``margin`` laterally) stays inside the crop.
    Each negative shares its positive's fractional crop-grid coordinates, so
    both use identical trilinear embedding weights. Shifted candidates must
    retain interpolated presence, foreign membership, distance/arc bounds and
    a full appearance receptive field. Invalid candidates are omitted, never
    snapped back to the grid. Positives and their history are unchanged.
    ``extra_negative`` (e.g. a departed head) uses the same phase and support
    checks, but is exempt from the lateral/arc window; its component is required.
    With ``centerlines``, both classes are sampled uniformly on their supplied
    polylines with the same presence/support checks, without any grid snapping
    or phase adjustment. This preserves the native-traced negative geometry.
    """
    curve = np.asarray(curve, np.float64)
    foreign_local = np.asarray(foreign_local, np.float64).reshape(-1, 3)
    support = appearance_crop or crop
    half = (support.width-1)*support.spacing/2-margin
    lower = np.array([-half, -half, -support.behind*support.spacing+along_margin])
    upper = np.array([half, half, (support.depth-1-support.behind)*support.spacing-along_margin])
    origin = -crop.spacing*np.array([(crop.width-1)/2, (crop.width-1)/2, crop.behind])
    def supported(points):
        return ((points >= lower) & (points <= upper)).all(-1)
    pos = np.zeros((positives, 3), np.float32)
    pos_mask = np.zeros(positives, np.float32)
    neg = np.zeros((positives, negatives, 3), np.float32)
    neg_mask = np.zeros((positives, negatives), np.float32)
    eligible = np.flatnonzero((curve[:, 2] >= forward[0]) & (curve[:, 2] <= forward[1]) & supported(curve))
    if centerlines:
        eligible = eligible[volume_at(presence,crop,curve[eligible],order=1) >= rule.threshold]
    if not len(eligible) or len(curve) < 3:
        return pos, pos_mask, neg, neg_mask
    tree = cKDTree(curve)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=-1))]
    tangent = np.gradient(curve, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-9)
    foreign = np.zeros(np.shape(presence), np.float32)
    if len(foreign_local) and not centerlines:
        index = np.rint(crop_indices(crop, foreign_local)).astype(np.int64)
        foreign[tuple(index.T)] = 1.
    extra_component = None
    if extra_negative is not None and component_labels is not None:
        label = volume_at(component_labels, crop, [extra_negative])[0]
        if label > 0:
            extra_component = (component_labels == label).astype(np.float32)
    for k, chunk in enumerate(np.array_split(eligible, positives)):
        if not len(chunk):
            continue
        j = int(rng.choice(chunk))
        pos[k], pos_mask[k] = curve[j], 1.
        grid = (pos[k].astype(np.float64)-origin)/crop.spacing
        phase = (grid-np.floor(grid+.5))*crop.spacing
        slots = 0
        if extra_component is not None:
            base = origin+np.floor((np.asarray(extra_negative)-origin)/crop.spacing+.5)*crop.spacing
            point = base+phase
            if (supported(point) and tree.query(point)[0] > rule.own_radius
                    and volume_at(presence, crop, [point], order=1)[0] >= rule.threshold
                    and volume_at(extra_component, crop, [point], order=1)[0] >= rule.threshold):
                neg[k, 0], neg_mask[k, 0] = point, 1.
                slots = 1
        shifted = foreign_local if centerlines else foreign_local+phase
        distance, nearest = tree.query(shifted)
        offset = shifted-curve[nearest]
        along = (offset*tangent[nearest]).sum(-1)
        lateral = np.linalg.norm(offset-along[:, None]*tangent[nearest], axis=-1)
        weights = volume_at(presence, crop, shifted, order=1)
        keep = (supported(shifted) & (distance > rule.own_radius)
                & (nearest > 0) & (nearest < len(curve)-1) & (lateral <= rule.lateral_max)
                & (np.abs(arc[nearest]-arc[j]) <= rule.along_window)
                & (weights >= rule.threshold))
        if not centerlines:
            keep &= volume_at(foreign,crop,shifted,order=1) >= rule.threshold
        candidates = np.flatnonzero(keep)
        take = min(negatives-slots, len(candidates))
        if take:
            p = weights[candidates]+1e-6
            chosen = rng.choice(candidates, size=take, replace=False, p=None if centerlines else p/p.sum())
            neg[k, slots:slots+take] = shifted[chosen]
            neg_mask[k, slots:slots+take] = 1.
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
