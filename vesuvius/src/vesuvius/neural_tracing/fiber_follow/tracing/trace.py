"""Autoregressive fiber tracing with trusted twelve-voxel heading fits."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, normalize
from vesuvius.neural_tracing.fiber_follow.tracing.policy import DEFAULT_CONFIDENCE, DEFAULT_N_COMMIT, commit_prefix
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_path
from vesuvius.neural_tracing.fiber_follow.tracing.heading import trace_heading, TRACE_HEADING_POLICY, FRAME_POLICY, ct_frame

if TYPE_CHECKING:
    from vesuvius.neural_tracing.fiber_follow.models.model import ObservationFollower as FollowNet


@dataclass
class TraceParams:
    n_commit: int = DEFAULT_N_COMMIT  # maximum per decision, at most the model horizon; confidence can shorten each commit
    max_len: float = 6000.0
    confidence: float = DEFAULT_CONFIDENCE
    loop_radius: float = 1.5
    loop_skip: int = 40
    seed: int = 0  # stochastic sampler seed; each directed trace has its own stream
    # Rows per model forward call; 0 runs every active trace in one call. Chunking
    # bounds memory only: each row's inputs and outputs are unchanged.
    forward_chunk: int = 0

    def __post_init__(self):
        if self.n_commit < 1 or self.max_len <= 0 or not 0 <= self.confidence <= 1 or self.forward_chunk < 0:
            raise ValueError('Invalid rollout parameters')

    @classmethod
    def from_policy(cls, policy, **kwargs):
        """Rollout limits for a resolved ``OperatingPolicy``."""
        return cls(n_commit=policy.n_commit, confidence=policy.confidence, **kwargs)


def trace_history(path, size):
    """Inference history at unit arclength spacing, newest first, excluding head."""
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
    past = np.asarray(path[-2*size-2:])
    arc = arclength(past)
    target = arc[-1]-np.arange(1, size+1)
    return interp_at(past, arc, target.clip(0)), (target >= 0).astype(np.float32)


def advance_trace_path(path, frame, points, commit, heading_start, shape, params,
                       travelled=0., loop_start=0):
    """Shared committed geometry for inference and live training; no CT or labels.

    The caller applies commit_prefix first and resolves the returned heading's
    CT frame only after checking the new state's data split and read footprint.
    """
    from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
    path = np.asarray(path, dtype=np.float64)
    if commit < 1:
        return None, 'confidence'
    world = path[-1]+np.asarray(points)[:commit] @ np.asarray(frame).T
    if not np.isfinite(world).all():
        return None, 'invalid'
    seg = np.concatenate([path[-1][None], world])
    new = []
    first_connection_count = 0
    remaining = params.max_len-travelled
    for segment_index, (a, b) in enumerate(zip(seg[:-1], seg[1:])):
        distance = float(np.linalg.norm(b-a))
        if distance < 1e-8:
            continue
        used = min(remaining, distance)
        end = a+(b-a)*(used/distance)
        steps = max(1, int(np.ceil(used)))
        new.extend(a+(end-a)*(k/steps) for k in range(1, steps+1))
        if segment_index == 0:
            first_connection_count = len(new)
        remaining -= used
        if remaining <= 1e-7:
            break
    new = np.asarray(new)
    if not len(new) or not np.isfinite(new).all():
        return None, 'invalid'
    if np.any(new < 0) or np.any(new[:, ::-1] >= np.asarray(shape)):
        return None, 'bounds'
    old = path[loop_start:-params.loop_skip]
    if len(old) and np.min(np.linalg.norm(old[:, None]-new[None], axis=-1)) < params.loop_radius:
        return None, 'loop'
    last_segment = np.concatenate([path[-1][None], new])
    travelled += float(arclength(last_segment)[-1])
    old_size = len(path)
    path = np.concatenate([path, new])
    if heading_start >= old_size:
        heading_start = old_size+first_connection_count-1
    heading = trace_heading(path, heading_start, np.asarray(frame)[:, 2])
    return dict(path=path, last_segment=last_segment, travelled=travelled,
                heading_start=heading_start, heading=heading), ''


class ModelTracer:
    # When set, build_inputs also receives each trace's seed segment and length.
    path_context = False

    def __init__(self, model: FollowNet, vol: FiberVolume, crop: CropSpec, n_history: int = 128,
                 params: TraceParams | None = None, device: str = "cuda"):
        self.model, self.vol, self.crop = model, vol, crop
        self.n_history, self.p, self.device = n_history, params or TraceParams(), str(device)
        if self.p.n_commit > model.cfg.n_future:
            raise ValueError(f'n_commit={self.p.n_commit} exceeds the model horizon n_future={model.cfg.n_future}')
        self.grid = torch.from_numpy(crop_local_grid(crop)).float().to(device)
        self.S = crop.block_size
        self.pool = ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)))

    def close(self):
        self.pool.shutdown(wait=True)

    def forward(self, x, hist, hmask, sampling):
        """Model outputs for the active rows, in ``forward_chunk`` row chunks when set."""
        chunk = self.p.forward_chunk or len(hist)
        def rows(value, part):
            if isinstance(value, dict):
                return {k: rows(v, part) for k, v in value.items()}
            return value[part] if torch.is_tensor(value) and value.ndim and len(value) == len(hist) else value
        outputs = []
        for start in range(0, len(hist), chunk):
            part = slice(start, start+chunk)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.startswith('cuda')):
                outputs.append(self.model(rows(x, part), hist[part], hmask[part], **rows(sampling, part)))
        if len(outputs) == 1:
            return outputs[0]
        return {k: torch.cat([o[k] for o in outputs]) for k in outputs[0]
                if torch.is_tensor(outputs[0][k]) and outputs[0][k].ndim}

    def map(self, fn, values):
        """Per-trace CPU work in the tracer's pool; results keep input order."""
        pool = getattr(self, "pool", None)  # tests may build tracers without a pool
        return map(fn, values) if pool is None else pool.map(fn, values)

    def build_inputs(self, pos, frames, hist, hmask):
        raise NotImplementedError('Use the common observation tracer')

    @torch.no_grad()
    def trace(self, seeds_xyz, headings, histories=None, abort=None, on_decision=None, initial_states=None):
        """Greedy rollout, checking supervised confidence before committing.

        on_decision(i, state) observes the exact inference input and proposals,
        before any commit; returning False censors the trace. abort(i, path)
        remains an optional post-commit geometric limit. Histories run oldest
        to newest and exclude the seed. Returned paths always start at the seed.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            return self._trace(seeds_xyz, headings, histories, abort, on_decision, initial_states)
        finally:
            self.model.train(was_training)

    def _trace(self, seeds_xyz, headings, histories, abort, on_decision, initial_states=None):
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
        n = len(seeds_xyz)
        paths = [[np.asarray(s, np.float64)] for s in seeds_xyz]
        if histories is not None:
            paths = [list(np.asarray(h, np.float64))+[np.asarray(s, np.float64)] for h, s in zip(histories, seeds_xyz)]
        if initial_states is not None:
            if len(initial_states) != n:
                raise ValueError('One initial observed state is required per seed')
            if any(s['frame_policy'] != FRAME_POLICY for s in initial_states):
                raise ValueError('Unsupported crop frame policy; recollect replay')
            paths = [list(observed_path(dict(s, pos=p,
                      hist_local=(np.asarray(s['hist'])-p) @ s['frame'])))
                     for s,p in zip(initial_states,seeds_xyz)]
            if any(not len(path) or not np.allclose(path[-1], p, atol=1e-5, rtol=0)
                   for path, p in zip(paths, seeds_xyz)):
                raise ValueError('Initial observed prefix must end at resumed head')
        hist_start = [len(p)-1 for p in paths]
        frame_diagnostics = ([{} for _ in range(n)] if initial_states is None else
                             [dict(s.get('ct_frame_diagnostics', {})) for s in initial_states])
        frames = (list(self.map(lambda a: ct_frame(self.vol, a[0], a[1], diagnostics=a[2]),
                                zip(seeds_xyz, headings, frame_diagnostics)))
                  if initial_states is None else [np.asarray(s['frame']).copy() for s in initial_states])
        # Explicit histories supplied for a new trace are trusted. Resumed
        # states carry the exact acceptance boundary saved at their decision.
        heading_start = np.zeros(n, dtype=np.int64)
        if initial_states is not None:
            for i, state in enumerate(initial_states):
                heading_start[i] = int(state['heading_start'])
                if not 0 <= heading_start[i] <= len(paths[i]):
                    raise ValueError('Invalid trusted heading history boundary')
        references = [dict(seed_pos=np.asarray(p).copy(), seed_tangent=normalize(np.asarray(h)),
                           seed_age=0., seed_valid=True) for p, h in zip(seeds_xyz, headings)]
        if histories is not None:
            for reference, path in zip(references, paths):
                reference.update(seed_pos=np.asarray(path[0]).copy(), seed_age=float(arclength(np.asarray(path))[-1]))
        if initial_states is not None:
            for reference, state in zip(references, initial_states):
                reference.update({k: state[k] for k in SEED_FIELDS if k in state})
        active = np.ones(n, bool)
        reasons = ['']*n
        length = np.zeros(n)
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
                if initial_states is not None and length[i] == 0:
                    hist_world[j] = initial_states[i]['hist']
                    hm[j] = initial_states[i]['hmask']
                    continue
                hist_world[j], hm[j] = trace_history(paths[i], H)
            hist = np.einsum('bhi,bij->bhj', hist_world-pos[:, None], fr)
            tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
            context = {}
            if self.path_context:
                context['paths'] = [dict(observed_path=np.asarray(paths[i]),
                                         seed_segment=np.asarray(paths[i][hist_start[i]:hist_start[i]+64]),
                                         travelled=float(length[i]),
                                         **{**references[i], 'seed_age': references[i]['seed_age']+float(length[i])}) for i in idx]
            x = self.build_inputs(pos, fr, hist, hm, **context)
            sampling = {}
            if (hasattr(self.model, 'select_prediction')
                    or getattr(self.model.cfg, 'candidate_selection', 'prefix') == 'stop_fallback'):
                sampling.update(confidence_threshold=pp.confidence,n_commit=pp.n_commit)
            out = self.forward(x, tensor(hist).float(), tensor(hm), sampling)
            commits, allowed = commit_prefix(out['points'], out['confidence'], pp.confidence, pp.n_commit,
                                             self.model.cfg.max_recovery_distance)
            commits, allowed = [v.cpu().numpy() for v in (commits, allowed)]
            points, confidence = [out[k].float().cpu().numpy() for k in ('points', 'confidence')]
            # Next-step CT frames, resolved together after this step's commits.
            # Each depends only on its own trace, so the pool is a pure speedup.
            reframe = {}
            for j, i in enumerate(idx):
                conf = np.minimum.accumulate(confidence[j], axis=-1)
                commit = int(commits[j])
                recovery_blocked = not allowed[j]
                # Same-position refinement is already exhausted inside the model: a
                # rejected decision stops the trace immediately; nothing is forced.
                would_stop = commit == 0
                state = dict(pos=pos[j].copy(), frame=fr[j].copy(), hist=hist_world[j].copy(), hmask=hm[j].copy(),
                             points=points[j].copy(), confidence=conf.copy(), n_commit=commit, would_stop=would_stop,
                             recovery_allowed=bool(allowed[j]), recovery_blocked=bool(recovery_blocked),
                             travelled=float(length[i]), last_segment=last_segment[i].copy(),
                             observed_path=np.asarray(paths[i]).copy(),
                             heading_start=int(heading_start[i]), heading_policy=TRACE_HEADING_POLICY,
                             frame_policy=FRAME_POLICY, ct_frame_diagnostics=frame_diagnostics[i].copy())
                state.update({**references[i], 'seed_age': references[i]['seed_age']+float(length[i])})
                if on_decision is not None and on_decision(int(i), state) is False:
                    active[i], reasons[i] = False, 'oracle'
                    continue
                if recovery_blocked:
                    active[i], reasons[i] = False, 'recovery_limit'
                    continue
                if would_stop:
                    active[i], reasons[i] = False, 'confidence'
                    continue
                advanced, reason = advance_trace_path(
                    paths[i], fr[j], points[j], commit, int(heading_start[i]),
                    self.vol.shape, pp, length[i], hist_start[i])
                if advanced is None:
                    active[i], reasons[i] = False, reason
                    continue
                paths[i] = list(advanced['path'])
                last_segment[i] = advanced['last_segment']
                length[i] = advanced['travelled']
                heading_start[i] = advanced['heading_start']
                reframe[i] = advanced['heading']
                if abort is not None and abort(int(i), paths[i]):
                    active[i], reasons[i] = False, 'abort'
                elif length[i] >= pp.max_len-1e-6:
                    active[i], reasons[i] = False, 'max_len'
            update = lambda i: ct_frame(self.vol, paths[i][-1], reframe[i], frames[i],
                                        diagnostics=frame_diagnostics[i])
            for i, frame in zip(reframe, self.map(update, reframe)):
                frames[i] = frame
        return [np.asarray(p[h:]) for p, h in zip(paths, hist_start)], reasons
