"""Frontier-centered states, with candidate-specific history and new-step labels."""
from dataclasses import dataclass

import numba
import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.geometry import (
    CropSpec, frame_from_heading, normalize, random_rotation_about,
)


@dataclass
class BeamStateConfig:
    crop: CropSpec
    n_history: int = 128
    k_back: int = 32
    pool_size: int = 256  # training sample budget ONLY; inference scores all candidates
    tolerance: float = 1.5
    progress_slack: float = 2.
    lateral_sigmas: tuple = (.4, 1., 2.)
    lateral_probs: tuple = (.5, .35, .15)
    angle_sigmas_deg: tuple = (4., 10., 20.)
    angle_probs: tuple = (.5, .35, .15)

    @property
    def k_points(self):
        return self.k_back + 2  # history, parent, proposed endpoint


@numba.njit(cache=True)
def _tails(points, starts, ends, count, out, mask):
    """For each polyline ``points[starts[i]:ends[i]]``, the points 1, 2, ...
    ``count`` arclength units behind its last vertex (``resample_polyline`` of
    the reversed line at unit spacing, without its first sample). A line
    shorter than one unit yields its first vertex, as ``resample_polyline`` does."""
    for i in range(len(starts)):
        a, b = starts[i], ends[i]
        if b-a < 2:
            continue
        j = b-1  # walk segments (j, j-1) backwards; acc is the arclength at vertex j
        acc = 0.
        seg = 0.
        k = 0
        while k < count:
            t = k+1.
            while j > a:
                dx = points[j-1, 0]-points[j, 0]
                dy = points[j-1, 1]-points[j, 1]
                dz = points[j-1, 2]-points[j, 2]
                seg = np.sqrt(dx*dx+dy*dy+dz*dz)
                if acc+seg >= t-1e-9:
                    break
                acc += seg
                j -= 1
            if j == a:
                if k == 0:
                    out[i, 0] = points[a]
                    mask[i, 0] = 1.
                break
            f = min(max((t-acc)/seg, 0.), 1.) if seg > 0 else 0.
            for axis in range(3):
                out[i, k, axis] = points[j, axis]+f*(points[j-1, axis]-points[j, axis])
            mask[i, k] = 1.
            k += 1


def _behind_many(points, starts, ends, count):
    out = np.zeros((len(starts), count, 3))
    mask = np.zeros((len(starts), count), np.float32)
    _tails(points, np.asarray(starts, np.int64), np.asarray(ends, np.int64), count, out, mask)
    return out, mask


def _behind(points, count):
    out, mask = _behind_many(np.asarray(points, np.float64), [0], [len(points)], count)
    return out[0], mask[0]


def segment_labels(parents, endpoints, fiber, sign, tree, cfg):
    """(on-fiber, known, mean error) of each NEW segment parent -> endpoint.

    The whole segment is checked at <= 0.5 voxel spacing, including between its
    endpoints; this works with nondefault VC step lengths and old segments are
    not charged again.
    """
    P = len(parents)
    delta = endpoints-parents
    n = np.maximum(2, np.ceil(np.linalg.norm(delta, axis=1)/.5).astype(np.int64)+1)
    m = int(n.max())-1
    j = np.arange(1, m+1)
    valid = j[None] <= (n-1)[:, None]
    frac = np.where(valid, j[None]/np.maximum(n-1, 1)[:, None], 1.)
    samples = parents[:, None]+frac[..., None]*delta[:, None]
    dist, index = tree.query(samples.reshape(-1, 3))
    dist, index = dist.reshape(P, m), index.reshape(P, m)
    parent_progress = fiber.s[tree.query(parents)[1]]*sign
    progress = np.where(valid, fiber.s[index]*sign, -np.inf)
    running = np.maximum.accumulate(np.concatenate([parent_progress[:, None], progress], 1), 1)[:, :-1]
    at_end = index == (len(fiber.points)-1 if sign > 0 else 0)
    known = ~(at_end & (not fiber.endpoint_stop[1 if sign > 0 else 0]))
    failed = valid & known & ((dist > cfg.tolerance) | (progress < running-cfg.progress_slack))
    any_failed = failed.any(1)
    targets = (~any_failed).astype(np.float32)
    knowns = (any_failed | (known | ~valid).all(1)).astype(np.float32)
    errors = ((np.minimum(dist, 8)*valid).sum(1)/valid.sum(1)).astype(np.float32)
    return targets, knowns, errors


def pool_state(pool, cfg, fiber=None, sign=1., tree=None, rng=None, indices=None, anchor_index=None,
               crop_pos=None, crop_frame=None):
    """Use the best parent's current endpoint, never an old common ancestor.

    `indices` can subsample proposals for training, without changing the shared
    crop/history. Inference omits it and has no candidate count cap.
    `crop_pos`/`crop_frame` express the proposals in an existing crop instead
    (an already encoded scene); its history and footprint are not rebuilt.
    """
    paths = pool.paths
    if not len(pool) or paths.lengths.min() < 2:
        raise ValueError('a proposal needs a parent and an endpoint')
    points = paths.parent_points  # a candidate's path[:-1] is its parent's path
    anchor = int(np.argmin(pool.parent_losses)) if anchor_index is None else int(anchor_index)
    start, end = paths.parent_bounds(anchor)
    reference = points[start:end]
    pos = reference[-1].copy()
    heading = reference[-1]-reference[-2] if len(reference) > 1 else pool.step_directions[anchor]
    frame = frame_from_heading(normalize(heading))
    if rng is not None:
        frame = random_rotation_about(frame, rng.uniform(0, 2*np.pi))
    if crop_pos is not None:
        pos, frame = np.asarray(crop_pos, np.float64), np.asarray(crop_frame, np.float64)
    hist, hmask = _behind(reference, cfg.n_history)
    ids = np.arange(len(pool)) if indices is None else np.asarray(indices)
    starts, ends = paths.parent_bounds(ids)
    back, mask = _behind_many(points, starts, ends, cfg.k_back)
    parents, endpoints = points[ends-1], paths.endpoints[ids]
    world = np.concatenate([back[:, ::-1], parents[:, None], endpoints[:, None]], 1)
    point_mask = np.concatenate([mask[:, ::-1], np.ones((len(ids), 2), np.float32)], 1)
    candidates = (((world-pos) @ frame)*point_mask[..., None]).astype(np.float32)
    # supported_states recenters additional same-resolution crops when branches
    # spread apart, so unsupported proposals are never silently discarded.
    lower = np.array([cfg.crop.lateral_coords[0]]*2 + [cfg.crop.forward_coords[0]])
    upper = np.array([cfg.crop.lateral_coords[-1]]*2 + [cfg.crop.forward_coords[-1]])
    supported = ((candidates[:, -2:] >= lower) & (candidates[:, -2:] <= upper)).all((1, 2))
    item = dict(pos=pos, frame=frame, hist_local=((hist-pos) @ frame).astype(np.float32), hmask=hmask,
                candidates=candidates, point_mask=point_mask, cand_mask=np.ones(len(ids), np.float32),
                parent_loss=np.asarray(pool.parent_losses[ids], np.float32),
                step_length=np.asarray(pool.step_lengths[ids], np.float32),
                supported=supported.astype(np.float32),
                hand_loss=np.asarray(pool.losses[ids], np.float32),
                extra_world=np.concatenate([points, paths.endpoints]))
    if fiber is None:
        return item
    tree = cKDTree(fiber.points) if tree is None else tree
    targets, knowns, errors = segment_labels(parents, endpoints, fiber, sign, tree, cfg)
    item.update(onfiber=targets, label_mask=knowns*supported,
                quality=targets-.1*errors, fwd_error=errors[:, None])
    return item


def supported_states(pool, cfg, **kwargs):
    """Cover a spread-out beam with additional crops at the SAME resolution.

    Usually one crop covers every branch. If branches separate, recenter on an
    uncovered parent so every proposal is scored; no geometric pre-pruning.
    Yields original proposal indices and their state.
    """
    ids = np.asarray(kwargs.pop('indices', np.arange(len(pool))))
    while len(ids):
        anchor = int(ids[np.argmin(pool.parent_losses[ids])])
        item = pool_state(pool, cfg, indices=ids, anchor_index=anchor, **kwargs)
        supported = item['supported'] > 0
        if not supported[np.flatnonzero(ids == anchor)[0]]:
            raise ValueError('CT crop cannot cover even its anchor proposal; enlarge the crop')
        selected = ids[supported]
        # Candidate tensors only; leave global crop/history/holdout footprint.
        for key in ('candidates', 'point_mask', 'cand_mask', 'parent_loss', 'step_length',
                    'supported', 'hand_loss', 'onfiber', 'label_mask', 'quality', 'fwd_error'):
            if key in item:
                item[key] = item[key][supported]
        yield selected, item
        ids = ids[~supported]
