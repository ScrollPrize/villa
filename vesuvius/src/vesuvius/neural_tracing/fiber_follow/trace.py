"""Autoregressive fiber tracing (model policy) and a direction-field baseline."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.data import build_inputs, read_blocks, render_count
from vesuvius.neural_tracing.fiber_follow.geometry import (
    CropSpec,
    block_start,
    crop_local_grid,
    frame_from_heading,
    normalize,
)
from vesuvius.neural_tracing.fiber_follow.model import FollowNet
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume


DEFAULT_CONFIDENCE = 0.7


@dataclass
class TraceParams:
    n_commit: int = 4  # maximum; confidence can shorten each commit
    max_len: float = 6000.0
    confidence: float = DEFAULT_CONFIDENCE
    loop_radius: float = 1.5
    loop_skip: int = 40
    explore_calls: int = 0  # collection only: bounded suffix after first would-stop

    def __post_init__(self):
        if self.n_commit < 1 or self.max_len <= 0 or not 0 <= self.confidence <= 1 or self.explore_calls < 0:
            raise ValueError('Invalid rollout parameters')


def point_samples(vol: FiberVolume, pts_xyz: np.ndarray) -> np.ndarray:
    """Nearest-voxel presence (0..1) at xyz points."""
    out = np.zeros(len(pts_xyz), np.float32)
    zyx = np.round(pts_xyz[:, ::-1]).astype(np.int64)
    for i, q in enumerate(zyx):
        out[i] = vol.presence.read(q, (1, 1, 1))[0, 0, 0] / 255.0
    return out


def field_axis(vol: FiberVolume, p_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Trilinear axis tensor + presence at a point -> (principal axis xyz, presence)."""
    if getattr(getattr(vol, 'spec', None), 'mode', None) == 'ct':
        # Initialization only: estimate a local ridge axis from presence, never
        # load or expose the predicted direction vectors to CT-only tracing.
        offsets = np.stack(np.meshgrid(*[np.arange(-3, 4)]*3, indexing='ij'), -1).reshape(-1, 3)
        points = p_xyz[None]+offsets
        weights = vol.presence.sample_nearest(points[:, ::-1]).astype(float)/255
        weights = np.where(weights >= .5*weights.max(), weights**2, 0)
        if weights.sum() <= 1e-8:
            raise ValueError('No presence support for a seed heading; supply an explicit heading')
        center = np.average(points, axis=0, weights=weights)
        delta = points-center
        covariance = (delta*weights[:, None]).T @ delta/weights.sum()
        _, axes = np.linalg.eigh(covariance)
        return axes[:, -1], float(weights.max()**.5)
    base = np.floor(p_xyz[::-1]).astype(np.int64)
    raw = vol.fiber_raw_block(base, (2, 2, 2))
    from vesuvius.neural_tracing.fiber_follow.volume import decode_raw

    d = decode_raw(torch.from_numpy(raw[None]))[0].numpy()  # C,2,2,2
    fz, fy, fx = p_xyz[::-1] - base
    w = np.array([1 - fz, fz])[:, None, None] * np.array([1 - fy, fy])[None, :, None] * np.array([1 - fx, fx])[None, None, :]
    v = (d * w[None]).sum((1, 2, 3))
    xx, yy, zz, xy, xz, yz = v[1:7]
    T = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    ev, evec = np.linalg.eigh(T)
    return evec[:, -1], float(v[0])


class ModelTracer:
    def __init__(self, model: FollowNet, vol: FiberVolume, crop: CropSpec, n_history: int = 128,
                 params: TraceParams | None = None, device: str = "cuda"):
        self.model, self.vol, self.crop = model, vol, crop
        self.n_history, self.p, self.device = n_history, params or TraceParams(), str(device)
        self.grid = torch.from_numpy(crop_local_grid(crop)).float().to(device)
        self.S = crop.block_size
        self.pool = ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)))

    def close(self):
        self.pool.shutdown(wait=True)

    @torch.no_grad()
    def trace(self, seeds_xyz, headings, histories=None, abort=None, on_decision=None):
        """Greedy rollout, checking supervised confidence before committing.

        on_decision(i, state) observes the exact inference input and proposals,
        before any commit; returning False censors the trace. abort(i, path)
        remains an optional post-commit geometric limit. Histories run oldest
        to newest and exclude the seed. Returned paths always start at the seed.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            return self._trace(seeds_xyz, headings, histories, abort, on_decision)
        finally:
            self.model.train(was_training)

    def _trace(self, seeds_xyz, headings, histories, abort, on_decision):
        from vesuvius.neural_tracing.fiber_follow.geometry import arclength, interp_at
        n = len(seeds_xyz)
        paths = [[np.asarray(s, np.float64)] for s in seeds_xyz]
        if histories is not None:
            paths = [list(np.asarray(h, np.float64))+[np.asarray(s, np.float64)] for h, s in zip(histories, seeds_xyz)]
        hist_start = [len(p)-1 for p in paths]
        frames = [frame_from_heading(h) for h in headings]
        active = np.ones(n, bool)
        reasons = ['']*n
        length = np.zeros(n)
        exploration = np.full(n, -1, int)
        last_segment = [np.asarray([p[-1]]) for p in paths]
        pp = self.p
        while active.any():
            idx = np.flatnonzero(active)
            pos = np.stack([paths[i][-1] for i in idx])
            fr = np.stack([frames[i] for i in idx])
            H = self.n_history
            hist_world = np.zeros((len(idx), H, 3))
            hm = np.zeros((len(idx), H), np.float32)
            for j, i in enumerate(idx):
                past = np.asarray(paths[i][-2*H-2:])
                arc = arclength(past)
                target = arc[-1]-np.arange(1, H+1)
                hm[j] = target >= 0
                hist_world[j] = interp_at(past, arc, target.clip(0))
            hist = np.einsum('bhi,bij->bhj', hist_world-pos[:, None], fr)
            raw, starts = read_blocks([dict(pos=p, frame=f) for p, f in zip(pos, fr)], self.vol, self.crop, self.pool)
            tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
            x = build_inputs(tensor(raw), tensor(starts).float(), tensor(pos).float(), tensor(fr).float(),
                             tensor(hist).float(), tensor(hm), self.grid, n_render=render_count(self.crop),
                             gate_direction=self.crop.gate_direction, input_scale=getattr(self.vol, 'input_scale', 1.),
                             history_sigma=self.crop.history_sigma, history_render=self.crop.history_render)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                out = self.model(x, tensor(hist).float(), tensor(hm))
            candidates, ranks, confidence = [out[k].float().cpu().numpy() for k in ('candidates', 'ranks', 'confidence')]
            for j, i in enumerate(idx):
                conf = np.minimum.accumulate(confidence[j], axis=-1)
                viable = conf[:, 0] >= pp.confidence
                chosen = int(np.argmax(np.where(viable, ranks[j], -np.inf))) if viable.any() else int(ranks[j].argmax())
                commit = min(pp.n_commit, int(np.sum(conf[chosen] >= pp.confidence)))
                would_stop = commit == 0
                if would_stop and exploration[i] < 0 and pp.explore_calls:
                    exploration[i] = 0
                exploratory = exploration[i] >= 0
                state = dict(pos=pos[j].copy(), frame=fr[j].copy(), hist=hist_world[j].copy(), hmask=hm[j].copy(),
                             candidates=candidates[j].copy(), rank_scores=ranks[j].copy(), confidence=conf.copy(),
                             chosen=chosen, n_commit=commit, would_stop=would_stop, exploratory=exploratory,
                             travelled=float(length[i]), last_segment=last_segment[i].copy())
                if on_decision is not None and on_decision(int(i), state) is False:
                    active[i], reasons[i] = False, 'oracle'
                    continue
                if exploratory and exploration[i] >= pp.explore_calls:
                    active[i], reasons[i] = False, 'exploration_limit'
                    continue
                if would_stop and not exploratory:
                    active[i], reasons[i] = False, 'confidence'
                    continue
                if exploratory:
                    exploration[i] += 1
                commit = max(1, commit)
                world = pos[j]+candidates[j, chosen, :commit] @ fr[j].T
                seg = np.concatenate([pos[j][None], world])
                new = []
                remaining = pp.max_len-length[i]
                for a, b in zip(seg[:-1], seg[1:]):
                    distance = float(np.linalg.norm(b-a))
                    if distance < 1e-8:
                        continue
                    used = min(remaining, distance)
                    end = a+(b-a)*(used/distance)
                    steps = max(1, int(np.ceil(used)))
                    new.extend(a+(end-a)*(k/steps) for k in range(1, steps+1))
                    remaining -= used
                    if remaining <= 1e-7:
                        break
                new = np.asarray(new)
                if not len(new) or not np.isfinite(new).all():
                    active[i], reasons[i] = False, 'invalid'
                    continue
                if np.any(new < 0) or np.any(new[:, ::-1] >= np.asarray(self.vol.shape)):
                    active[i], reasons[i] = False, 'bounds'
                    continue
                old = np.asarray(paths[i][hist_start[i]:-pp.loop_skip])
                if len(old) and np.min(np.linalg.norm(old[:, None]-new[None], axis=-1)) < pp.loop_radius:
                    active[i], reasons[i] = False, 'loop'
                    continue
                last_segment[i] = np.concatenate([pos[j][None], new])
                length[i] += float(arclength(last_segment[i])[-1])
                paths[i].extend(new)
                # Heading uses committed geometry only, never a future prediction.
                tangent = new[-1]-(new[-2] if len(new)>1 else pos[j])
                frames[i] = frame_from_heading(normalize(tangent), frames[i][:, 0])
                if abort is not None and abort(int(i), paths[i]):
                    active[i], reasons[i] = False, 'abort'
                elif length[i] >= pp.max_len-1e-6:
                    active[i], reasons[i] = False, 'max_len'
        return [np.asarray(p[h:]) for p, h in zip(paths, hist_start)], reasons


class FieldTracer:
    """Baseline: integrate the predicted fiber axis field with presence re-centring."""

    def __init__(self, vol: FiberVolume, step: float = 1.0, inertia: float = 0.5, recenter: float = 0.5,
                 min_presence: float = 0.12, patience: int = 12, max_len: float = 6000.0):
        self.vol = vol
        self.step, self.inertia, self.recenter = step, inertia, recenter
        self.min_presence, self.patience, self.max_len = min_presence, patience, max_len
        g = np.linspace(-1.5, 1.5, 7)
        self.plane = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)

    def _pres(self, pts):
        base = np.floor(pts.min(0)[::-1]).astype(np.int64)
        size = np.floor(pts.max(0)[::-1]).astype(np.int64) - base + 2
        blk = self.vol.presence.read(base, size).astype(np.float32) / 255.0
        q = pts[:, ::-1] - base
        i0 = np.floor(q).astype(int)
        f = q - i0
        out = 0
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (f[:, 0] if dz else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dx else 1 - f[:, 2])
                    out = out + w * blk[i0[:, 0] + dz, i0[:, 1] + dy, i0[:, 2] + dx]
        return out

    def trace_one(self, seed, heading):
        p = np.asarray(seed, np.float64)
        h = normalize(np.asarray(heading, np.float64))
        path = [p]
        low = 0
        for _ in range(int(self.max_len / self.step)):
            e, _ = field_axis(self.vol, p)
            if np.dot(e, h) < 0:
                e = -e
            h = normalize(self.inertia * h + (1 - self.inertia) * e)
            q = p + self.step * h
            fr = frame_from_heading(h)
            cand = q + self.plane[:, :1] * fr[:, 0] + self.plane[:, 1:] * fr[:, 1]
            w = self._pres(cand) ** 2
            if w.sum() > 1e-6:
                c = (cand * w[:, None]).sum(0) / w.sum()
                q = q + self.recenter * (c - q)
            pr = self._pres(q[None])[0]
            low = low + 1 if pr < self.min_presence else 0
            if low >= self.patience:
                del path[-(low - 1):]
                return np.asarray(path), "presence"
            if np.any(q < 0) or np.any(q[::-1] >= np.asarray(self.vol.shape)):
                return np.asarray(path), "bounds"
            h = normalize(q - p)
            p = q
            path.append(p)
        return np.asarray(path), "max_len"

    def trace(self, seeds, headings, histories=None):
        res = [self.trace_one(s, h) for s, h in zip(seeds, headings)]
        return [r[0] for r in res], [r[1] for r in res]
