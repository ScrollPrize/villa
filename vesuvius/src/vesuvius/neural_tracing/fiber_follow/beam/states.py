"""Turn one beam candidate pool into a re-ranker state, with labels from dense GT.

A pool arrives at prune time with up to ``pool_size`` full paths that share a
common trunk. The state is anchored at the end of that trunk: the trunk is the
history (input channel and conditioning vector), and every candidate becomes a
fixed-length polyline of ``k_back`` trunk points, the anchor, and ``k_fwd`` new
points, all in the anchor's local frame. Labels compare only the new points
with the annotated curve, using the same tolerance/prefix/censoring semantics
as ``supervision.candidate_labels``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from vesuvius.neural_tracing.fiber_follow.geometry import (
    CropSpec, arclength, frame_from_heading, normalize, random_rotation_about, resample_polyline,
)
from vesuvius.neural_tracing.fiber_follow.beam.tube import tube_geometry


@dataclass
class BeamStateConfig:
    crop: CropSpec
    n_history: int = 128
    k_back: int = 16
    k_fwd: int = 8
    pool_size: int = 32
    tolerance: float = 1.5
    offtrack_distance: float = 3.5
    progress_slack: float = 2.0
    heatmap_target: str = 'tube'
    tube_sigma: float = 0.35
    lateral_sigmas: tuple = (0.4, 1.0, 2.0)
    lateral_probs: tuple = (0.5, 0.35, 0.15)
    angle_sigmas_deg: tuple = (4.0, 10.0, 20.0)
    angle_probs: tuple = (0.5, 0.35, 0.15)

    @property
    def k_points(self) -> int:
        return self.k_back + 1 + self.k_fwd


def common_trunk(paths) -> np.ndarray:
    """Longest shared prefix of the pool paths (they share ancestor nodes exactly)."""
    n = min(len(p) for p in paths)
    ref = paths[0][:n]
    same = np.ones(n, bool)
    for p in paths[1:]:
        same &= np.all(p[:n] == ref, axis=1)
    k = int(np.argmin(same)) if not same.all() else n
    return paths[0][:max(k, 1)]


def _behind(points: np.ndarray, count: int):
    """``count`` points at 1-voxel arc steps behind the last point of ``points``."""
    out = np.zeros((count, 3))
    mask = np.zeros(count, np.float32)
    if len(points) < 2:
        return out, mask
    back = resample_polyline(points[::-1], 1.0)[1:count + 1]
    out[:len(back)] = back
    mask[:len(back)] = 1
    return out, mask


def _ahead(points: np.ndarray, count: int):
    out = np.zeros((count, 3))
    mask = np.zeros(count, np.float32)
    if len(points) < 2:
        return out, mask
    fwd = resample_polyline(points, 1.0)[1:count + 1]
    out[:len(fwd)] = fwd
    mask[:len(fwd)] = 1
    return out, mask


def anchor_frame(pool, trunk: np.ndarray, rng=None):
    pos = trunk[-1].copy()
    if len(trunk) >= 2:
        heading = trunk[-1] - trunk[-2]
    else:
        heading = pool.paths[0][1] - pool.paths[0][0] if len(pool.paths[0]) > 1 else pool.step_directions[0]
    frame = frame_from_heading(normalize(heading))
    if rng is not None:
        frame = random_rotation_about(frame, rng.uniform(0, 2 * np.pi))
    return pos, frame


def pool_state(pool, cfg: BeamStateConfig, fiber=None, sign: float = 1.0, tree=None, rng=None) -> dict:
    """Model input geometry for a pool, plus labels when ``fiber`` is given.

    ``sign`` is the traversal direction along the fiber's arclength (+1 toward
    increasing ``fiber.s``). ``tree`` caches ``cKDTree(fiber.points)``.
    """
    P, K, kb, kf = cfg.pool_size, cfg.k_points, cfg.k_back, cfg.k_fwd
    trunk = common_trunk(pool.paths)
    pos, frame = anchor_frame(pool, trunk, rng)
    hist_world, hmask = _behind(trunk, cfg.n_history)
    back_world, back_mask = _behind(trunk, kb)

    n = min(len(pool), P)
    candidates = np.zeros((P, K, 3), np.float32)
    point_mask = np.zeros((P, K), np.float32)
    fwd_world = np.zeros((P, kf, 3))
    fwd_mask = np.zeros((P, kf), np.float32)
    cand_mask = np.zeros(P, np.float32)
    cand_mask[:n] = 1
    for i in range(n):
        new = pool.paths[i][len(trunk) - 1:]
        fwd_world[i], fwd_mask[i] = _ahead(new, kf)
        world = np.concatenate([back_world[::-1], pos[None], fwd_world[i]], 0)
        candidates[i] = (world - pos) @ frame
        point_mask[i] = np.concatenate([back_mask[::-1], [1.0], fwd_mask[i]])
    candidates *= point_mask[..., None]

    hand_loss = np.zeros(P, np.float32)
    hand_loss[:n] = pool.losses[:n]
    hand_rel = np.zeros(P, np.float32)
    hand_rel[:n] = np.clip(pool.losses[:n] - pool.losses[:n].min(), 0, 8)

    item = dict(pos=pos, frame=frame, hist_local=((hist_world - pos) @ frame).astype(np.float32),
                hmask=hmask, candidates=candidates, point_mask=point_mask, cand_mask=cand_mask,
                hand_loss=hand_loss, hand_rel=hand_rel,
                extra_world=np.concatenate([np.asarray(p) for p in pool.paths[:n]], 0))
    if fiber is None:
        return item

    if tree is None:
        tree = cKDTree(fiber.points)
    last = len(fiber.points) - 1
    d_pos, _ = tree.query(pos)
    offtrack = bool(d_pos > cfg.offtrack_distance)
    dist, index = tree.query(fwd_world.reshape(-1, 3))
    dist = dist.reshape(P, kf)
    index = index.reshape(P, kf)
    known = fwd_mask > 0
    # Continuation past an untagged annotation end is unknown, not wrong.
    at_end = (index == last) if sign > 0 else (index == 0)
    end_known = fiber.endpoint_stop[1 if sign > 0 else 0]
    known &= ~(at_end & (not end_known))
    prog = fiber.s[index] * sign
    # Progress along GT must not run backward from the anchor or an earlier point.
    pos_prog = fiber.s[tree.query(pos)[1]] * sign
    running = np.maximum.accumulate(np.concatenate([np.full((P, 1), pos_prog), prog], 1), axis=1)[:, :-1]
    regress = prog < running - cfg.progress_slack
    failed = known & ((dist > cfg.tolerance) | regress)
    if offtrack:
        failed = fwd_mask > 0
        known = fwd_mask > 0
    prefix_failed = np.cumsum(failed, 1) > 0
    prefix_known = np.cumprod(known, 1) > 0
    prefix_target = (~prefix_failed).astype(np.float32) * (fwd_mask > 0)
    prefix_mask = ((prefix_failed | prefix_known) & (fwd_mask > 0)).astype(np.float32)
    count = fwd_mask.sum(1)
    last_idx = np.clip(count - 1, 0, kf - 1).astype(int)
    rows = np.arange(P)
    label_mask = (cand_mask > 0) & (count > 0) & (prefix_mask[rows, last_idx] > 0)
    onfiber = ((~prefix_failed[rows, last_idx]) & label_mask).astype(np.float32)
    ok_mean = (prefix_target * prefix_mask).sum(1) / np.maximum(prefix_mask.sum(1), 1)
    err_mean = (np.minimum(dist, 8) * fwd_mask).sum(1) / np.maximum(fwd_mask.sum(1), 1)
    quality = (ok_mean - 0.1 * err_mean).astype(np.float32)
    item.update(prefix_target=prefix_target, prefix_mask=prefix_mask, onfiber=onfiber,
                label_mask=label_mask.astype(np.float32), quality=quality, offtrack=np.float32(offtrack),
                fwd_error=np.minimum(dist, 8).astype(np.float32) * fwd_mask)
    if cfg.heatmap_target == 'tube':
        item.update(tube_geometry(fiber.points, pos, frame, cfg.crop, cfg.tube_sigma, fiber.endpoint_stop, offtrack))
    return item
