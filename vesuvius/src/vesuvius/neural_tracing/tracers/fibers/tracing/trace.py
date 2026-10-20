"""Autoregressive fiber tracing with trusted twelve-voxel heading fits."""

from __future__ import annotations

import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, frame_from_heading, normalize
from vesuvius.neural_tracing.tracers.shared.policy import (DEFAULT_CONFIDENCE, DEFAULT_GATE, DEFAULT_N_COMMIT, GATES, gate_horizon,
    commit_count, selection_window)
from vesuvius.neural_tracing.tracers.shared.crop_sampling.volume import CTVolume
from vesuvius.neural_tracing.tracers.shared.reference import SEED_FIELDS, observed_path, trace_history
from vesuvius.neural_tracing.tracers.fibers.data.observations import observation_builder
from vesuvius.neural_tracing.tracers.fibers.frames.ct import (
    trace_heading, TRACE_HEADING_POLICY, FRAME_POLICY, FRAME_POLICIES, PREDICTION_FRAME_POLICY, ct_frame, fiber_family)
from vesuvius.neural_tracing.tracers.fibers.frames.learned import (frame_predictor, configured_frame_policy, predict_frames,
                                      predicted_axis, prediction_window)

if TYPE_CHECKING:
    from torch.nn import Module as FollowNet


@dataclass
class TraceParams:
    n_commit: int = DEFAULT_N_COMMIT  # points per committed decision, at most the model horizon
    max_len: float = 6000.0
    confidence: float = DEFAULT_CONFIDENCE
    gate: str = DEFAULT_GATE  # 'full': commit n_commit only if last-plane confidence passes; 'prefix': confident prefix
    loop_radius: float = 1.5
    loop_skip: int = 40
    seed: int = 0  # stochastic sampler seed; each directed trace has its own stream (decision_noise_key)
    # Rows per model forward call; 0 runs every active trace in one call. Chunking
    # bounds memory only: each row's inputs and outputs are unchanged.
    forward_chunk: int = 0
    # Model arithmetic on CUDA: 'bf16' autocast, or 'fp32' (no autocast; TF32 follows torch.backends flags).
    # fp32 keeps batch-composition rounding differences from growing into flipped decisions.
    precision: str = 'bf16'
    # Optional (off by default): a decision that would stop on confidence first retries once at the same head with its
    # crop refit to its own prediction (frames.learned.predicted_axis over the gate horizon); only a second stop
    # then ends the trace.
    refit_retry: bool = False

    def __post_init__(self):
        if (self.n_commit < 1 or self.max_len <= 0 or not 0 <= self.confidence <= 1 or self.forward_chunk < 0
                or self.gate not in GATES):
            raise ValueError('Invalid rollout parameters')
        if self.precision not in ('bf16', 'fp32'):
            raise ValueError("Rollout precision is 'bf16' or 'fp32'")

    @classmethod
    def from_policy(cls, policy, **kwargs):
        """Rollout limits for a resolved ``OperatingPolicy``."""
        return cls(n_commit=policy.n_commit, confidence=policy.confidence, gate=policy.gate, **kwargs)


def decision_noise_key(seed, start, decision):
    """Seed of one decision's sampled proposals: (sampler seed, the trace's start, decision index).

    The start is rounded to 1e-3 voxel, so a trace draws the same noise however it is batched.
    """
    import hashlib
    start = np.round(np.asarray(start, np.float64), 3)+0.  # +0. folds -0. into 0.
    digest = hashlib.blake2b(np.asarray([seed, decision], np.int64).tobytes()+start.tobytes(), digest_size=8).digest()
    return int.from_bytes(digest, 'little') & ((1 << 63)-1)


def advance_trace_path(path, frame, points, commit, heading_start, shape, params,
                       travelled=0., loop_start=0):
    """Shared committed geometry for inference and live training; no CT or labels.

    The caller applies commit_prefix first and resolves the returned heading's
    CT frame only after checking the new state's data split and read footprint.
    """
    from vesuvius.neural_tracing.tracers.shared.geometry import arclength
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
    if len(old):
        # Only points inside the new points' box grown by the loop radius can lie closer than it: the same decision
        # as the full distance matrix, without its cost growing with the trace length times the commit.
        near = np.all((old >= new.min(0)-params.loop_radius) & (old <= new.max(0)+params.loop_radius), 1)
        old = old[near]
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

    def __init__(self, model: FollowNet, vol: CTVolume, crop: CropSpec, n_history: int = 128,
                 params: TraceParams | None = None, device: str = "cuda"):
        self.model, self.vol, self.crop = model, vol, crop
        self.n_history, self.p, self.device = n_history, params or TraceParams(), str(device)
        self.frame_predictor = frame_predictor(model.cfg)
        self.frame_policy = configured_frame_policy(model.cfg)
        self.prediction_window = prediction_window(model.cfg)  # continuation crop axis from the previous prediction
        cfg = model.cfg
        self.refit_window = float(getattr(cfg, 'gate_horizon', cfg.n_future))*float(getattr(cfg, 'future_step', 1.))
        self.heading_policy = 'learned_heading_v1' if self.frame_predictor is not None else TRACE_HEADING_POLICY
        if self.p.n_commit > model.cfg.n_future:
            raise ValueError(f'n_commit={self.p.n_commit} exceeds the model horizon n_future={model.cfg.n_future}')
        self.pool = ThreadPoolExecutor(max(1, min(16, os.cpu_count() or 1)))

    def close(self):
        self.pool.shutdown(wait=True)

    def autocast(self):
        return torch.autocast('cuda', dtype=torch.bfloat16,
                              enabled=self.device.startswith('cuda') and self.p.precision == 'bf16')

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
            with self.autocast():
                call = getattr(self, 'forward_model', self.model)
                outputs.append(call(rows(x, part), hist[part], hmask[part], **rows(sampling, part)))
        if len(outputs) == 1:
            return outputs[0]
        def attempts(values):
            # Adaptive refinement stops once every row of a chunk is accepted, so chunks can return fewer
            # attempts. Extend them as the unchunked forward does: accepted rows repeat their last attempt, masked out.
            width = max(v.shape[1] for v in values)
            def pad(v):
                tail = v[:, -1:].expand(-1, width-v.shape[1], *v.shape[2:])
                return torch.cat((v, torch.zeros_like(tail) if v.dtype == torch.bool else tail), 1)
            return [pad(v) if v.shape[1] < width else v for v in values]
        return {k: torch.cat(attempts([o[k] for o in outputs]) if k.startswith('refinement_') else [o[k] for o in outputs])
                for k in outputs[0] if torch.is_tensor(outputs[0][k]) and outputs[0][k].ndim}

    def map(self, fn, values):
        """Per-trace CPU work in the tracer's pool; results keep input order."""
        pool = getattr(self, "pool", None)  # tests may build tracers without a pool
        return map(fn, values) if pool is None else pool.map(fn, values)

    def build_inputs(self, pos, frames, hist, hmask):
        raise NotImplementedError('Use the common observation tracer')

    def sequence_attach(self, x, idx, sequence, pos, frames, length):
        """Sequence follower (models/sequence.py): each row reads its trace's earlier history tokens (per-layer
        states, at most ``history_limit`` most recent) with their pose in this decision's frame."""
        from vesuvius.neural_tracing.tracers.models.sequence import relative_pose
        cfg = self.model.cfg
        limit = cfg.history_limit
        count = max(1, max(min(len(sequence[i]['heads']), limit) for i in idx))
        states = torch.zeros(len(idx), cfg.layers, count, cfg.hidden, device=self.device)
        relative = torch.zeros(len(idx), count, 3, device=self.device)
        age = torch.zeros(len(idx), count, device=self.device)
        padding = torch.ones(len(idx), count, dtype=torch.bool, device=self.device)
        for j, i in enumerate(idx):
            state = sequence[i]
            n = min(len(state['heads']), limit)
            if not n:
                continue
            heads = torch.from_numpy(np.stack(state['heads'][-n:])).to(self.device)
            relative[j, :n], age[j, :n] = relative_pose(
                heads, torch.tensor(state['travelled'][-n:], device=self.device, dtype=torch.float64),
                torch.from_numpy(pos[j]).to(self.device), torch.from_numpy(frames[j]).to(self.device),
                torch.tensor(float(length[i]), device=self.device, dtype=torch.float64))
            states[j, :, :n] = torch.stack([torch.stack(layer[-n:]) for layer in state['layers']])
            padding[j, :n] = False
        x.update(sequence_states=states, sequence_relative=relative, sequence_age=age, sequence_padding=padding)

    def sequence_record(self, cells, committed, sequence):
        """After commits: each (row, trace, head, frame-local committed points, travelled at the head) becomes its
        trace's next history token: the head crop's cells pooled at the head and its committed points, as in training."""
        if not committed:
            return
        rows = [c[0] for c in committed]
        points = [np.concatenate((np.zeros((1, 3)), c[3])) for c in committed]
        size = max(len(p) for p in points)
        local = torch.zeros(len(points), size, 3, device=self.device)
        valid = torch.zeros(len(points), size, dtype=torch.bool, device=self.device)
        for r, p in enumerate(points):
            local[r, :len(p)] = torch.from_numpy(p).float().to(self.device)
            valid[r, :len(p)] = True
        displacement = torch.stack([local[r, len(p)-1]-local[r, 0] for r, p in enumerate(points)])
        travelled = torch.tensor([c[4] for c in committed], device=self.device, dtype=torch.float32)
        model = self.model
        with self.autocast():
            tokens = model.history_input(model.step_features(cells[rows], local, valid), displacement, travelled)
            for r, (_, i, head, _, at) in enumerate(committed):
                state = sequence[i]
                limit = model.cfg.history_limit
                past = [torch.stack(layer[-limit:])[None] if layer else tokens.new_zeros(1, 0, tokens.shape[-1])
                        for layer in state['layers']]
                for layer, value in zip(state['layers'], model.extend_history(tokens[r:r+1], past)):
                    layer.append(value[0].float())
                state['heads'].append(np.asarray(head, np.float64).copy())
                state['travelled'].append(float(at))

    @torch.no_grad()
    def trace(self, seeds_xyz, headings, on_decision=None, initial_states=None, families=None, *,
              feed=None, before_step=None, on_finished=None):
        """Greedy rollout, checking supervised confidence before committing.

        on_decision(i, state) observes the exact inference input and proposals,
        before any commit; returning False censors the trace (reason 'oracle'),
        returning a string censors it with that reason. Returned paths always
        start at the seed.

        Streaming (fresh seeds, non-sequence models), all optional:
        feed(active) is called before every step with the number of running traces and may return
        (seeds, headings, families) to start; they become the next row indices.
        before_step(idx, pos, frames) sees the running rows' heads and crop frames before their inputs are built and
        may return {row: reason} to stop rows there.
        on_finished(i, path, reason) receives each trace when it ends; its returned path is then released (empty).
        """
        was_training = self.model.training
        self.model.eval()
        if (feed is not None or on_finished is not None) and initial_states is not None:
            raise ValueError('Streaming traces start from fresh seeds')
        try:
            return self._trace(seeds_xyz, headings, on_decision, initial_states, families, feed, before_step,
                               on_finished)
        finally:
            self.model.train(was_training)

    def _trace(self, seeds_xyz, headings, on_decision, initial_states=None, families=None, feed=None, before_step=None,
               on_finished=None):
        if not hasattr(self, 'stage_seconds'):
            self.stage_seconds = defaultdict(float)
        stamp = time.perf_counter()
        def mark(stage):
            nonlocal stamp
            now = time.perf_counter()
            self.stage_seconds[stage] += now-stamp
            stamp = now
        n = len(seeds_xyz)
        seeds_xyz = list(seeds_xyz)  # streaming appends rows
        predictor = getattr(self, 'frame_predictor', None)
        if families is None and initial_states is not None and predictor is not None:
            families = [s.get('fiber_family') for s in initial_states]
        if predictor is not None and (families is None or len(families) != n):
            raise ValueError('Learned crop frames require one H/V family per trace')
        families = [fiber_family(f) for f in families] if families is not None else [None]*n
        # Paths are (N, 3) float64 arrays (list-to-array conversions every step grow with the trace length).
        paths = [np.asarray(s, np.float64)[None] for s in seeds_xyz]
        if initial_states is not None:
            if len(initial_states) != n:
                raise ValueError('One initial observed state is required per seed')
            if any(s['frame_policy'] not in FRAME_POLICIES for s in initial_states):
                raise ValueError('Unsupported crop frame policy; recollect replay')
            paths = [np.asarray(observed_path(dict(s, pos=p,
                      hist_local=(np.asarray(s['hist'])-p) @ s['frame'])), np.float64).reshape(-1, 3)
                     for s,p in zip(initial_states,seeds_xyz)]
            if any(not len(path) or not np.allclose(path[-1], p, atol=1e-5, rtol=0)
                   for path, p in zip(paths, seeds_xyz)):
                raise ValueError('Initial observed prefix must end at resumed head')
        hist_start = [len(p)-1 for p in paths]
        frame_diagnostics = ([{} for _ in range(n)] if initial_states is None else
                             [dict(s.get('ct_frame_diagnostics', {})) for s in initial_states])
        if initial_states is not None:
            frames = [np.asarray(s['frame']).copy() for s in initial_states]
        elif predictor is not None:
            frames = predict_frames(predictor, self.vol, seeds_xyz, headings, [np.asarray(p) for p in paths], families,
                                    pool=getattr(self, 'pool', None))
            frame_diagnostics = [dict(source=3, energy=0., gap=0.) for _ in range(n)]
        else:
            frames = list(self.map(lambda a: ct_frame(self.vol, a[0], a[1], diagnostics=a[2]),
                                   zip(seeds_xyz, headings, frame_diagnostics)))
        # Resumed states carry the exact acceptance boundary saved at their decision.
        heading_start = np.zeros(n, dtype=np.int64)
        if initial_states is not None:
            for i, state in enumerate(initial_states):
                heading_start[i] = int(state['heading_start'])
                if not 0 <= heading_start[i] <= len(paths[i]):
                    raise ValueError('Invalid trusted heading history boundary')
        references = [dict(seed_pos=np.asarray(p).copy(), seed_tangent=normalize(np.asarray(h)),
                           seed_age=0., seed_valid=True) for p, h in zip(seeds_xyz, headings)]
        if initial_states is not None:
            for reference, state in zip(references, initial_states):
                reference.update({k: state[k] for k in SEED_FIELDS if k in state})
        sequence = ([dict(layers=[[] for _ in range(self.model.cfg.layers)], heads=[], travelled=[]) for _ in range(n)]
                    if getattr(self.model.cfg, 'model_type', '') == 'sequence' else None)
        if feed is not None and sequence is not None:
            raise ValueError('Streaming traces need a non-sequence follower')
        active = np.ones(n, bool)
        reasons = ['']*n
        length = np.zeros(n)
        last_segment = [np.asarray([p[-1]]) for p in paths]
        decisions = np.zeros(n, np.int64)  # per trace, for the flow sampler's decision keys
        # Each trace's previous prediction past what it committed (world, nearest first); None at a seed or restart.
        prediction = [None]*n
        predicted = np.zeros(n, bool)  # this decision's crop came from the previous prediction
        retried = np.zeros(n, bool)  # this head already had its confidence-stop retry
        pp = self.p

        def grow(new_seeds, new_headings, new_families):
            """Start fresh traces as new rows: the same initial state as the seeds above."""
            nonlocal heading_start, active, length, decisions, predicted, retried
            m = len(new_seeds)
            new_families = ([fiber_family(f) for f in new_families] if new_families is not None else [None]*m)
            if predictor is not None and any(f is None for f in new_families):
                raise ValueError('Learned crop frames require one H/V family per trace')
            new_paths = [np.asarray(q, np.float64)[None] for q in new_seeds]
            if predictor is not None:
                new_frames = predict_frames(predictor, self.vol, list(new_seeds), list(new_headings),
                                            [np.asarray(q) for q in new_paths], new_families, pool=getattr(self, 'pool', None))
                new_diag = [dict(source=3, energy=0., gap=0.) for _ in range(m)]
            else:
                new_diag = [{} for _ in range(m)]
                new_frames = list(self.map(lambda a: ct_frame(self.vol, a[0], a[1], diagnostics=a[2]),
                                           zip(new_seeds, new_headings, new_diag)))
            seeds_xyz.extend(new_seeds)
            families.extend(new_families)
            paths.extend(new_paths)
            hist_start.extend([0]*m)
            frame_diagnostics.extend(new_diag)
            frames.extend(new_frames)
            references.extend(dict(seed_pos=np.asarray(q).copy(), seed_tangent=normalize(np.asarray(h)),
                                   seed_age=0., seed_valid=True) for q, h in zip(new_seeds, new_headings))
            reasons.extend(['']*m)
            last_segment.extend(np.asarray([q[-1]]) for q in new_paths)
            prediction.extend([None]*m)
            heading_start = np.concatenate([heading_start, np.zeros(m, np.int64)])
            active = np.concatenate([active, np.ones(m, bool)])
            length = np.concatenate([length, np.zeros(m)])
            decisions = np.concatenate([decisions, np.zeros(m, np.int64)])
            predicted = np.concatenate([predicted, np.zeros(m, bool)])
            retried = np.concatenate([retried, np.zeros(m, bool)])

        def finish(rows):
            for i in rows:
                on_finished(int(i), np.asarray(paths[i][hist_start[i]:]), reasons[i])
                paths[i], hist_start[i], last_segment[i], prediction[i] = np.zeros((0, 3)), 0, None, None
                references[i] = frames[i] = None

        while True:
            stamp = time.perf_counter()
            if feed is not None:
                more = feed(int(active.sum()))
                if more is not None and len(more[0]):
                    grow(*more)
            mark('feed')
            if not active.any():
                break
            idx = np.flatnonzero(active)
            if before_step is not None:
                stops = before_step(idx, np.stack([paths[i][-1] for i in idx]), np.stack([frames[i] for i in idx]))
                if stops:
                    for i, reason in stops.items():
                        active[i], reasons[i] = False, reason
                    if on_finished is not None:
                        finish(list(stops))
                    idx = np.flatnonzero(active)
                    if not len(idx):
                        continue
            was_active = idx
            mark('checks')
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
                                         fiber_family=families[i],
                                         seed_segment=np.asarray(paths[i][hist_start[i]:hist_start[i]+64]),
                                         travelled=float(length[i]),
                                         **{**references[i], 'seed_age': references[i]['seed_age']+float(length[i])}) for i in idx]
            x = self.build_inputs(pos, fr, hist, hm, **context)
            mark('inputs')
            if sequence is not None:
                self.sequence_attach(x, idx, sequence, pos, fr, length)
            sampling = {}
            if hasattr(self.model, 'select_prediction'):
                sampling.update(confidence_threshold=pp.confidence,
                                n_commit=selection_window(pp.n_commit, gate_horizon(self.model.cfg), pp.gate))
            if getattr(self.model.cfg, 'flow_samples', 0):
                x['flow_noise_keys'] = tensor(np.asarray([decision_noise_key(pp.seed, seeds_xyz[i], decisions[i])
                                                          for i in idx], np.int64))
            out = self.forward(x, tensor(hist).float(), tensor(hm), sampling)
            decisions[idx] += 1
            commits, allowed = commit_count(out['points'], out['confidence'], pp.confidence, pp.n_commit,
                                            self.model.cfg.max_recovery_distance, pp.gate, gate_horizon(self.model.cfg))
            commits, allowed = [v.cpu().numpy() for v in (commits, allowed)]
            points, confidence = [out[k].float().cpu().numpy() for k in ('points', 'confidence')]
            mark('forward')
            # Geometry of each accepted row depends only on that row. Reuse
            # the process pool, preserving row order and all scalar arithmetic.
            # Callbacks can inspect/change state, so keep their original serial
            # ordering rather than speculatively advancing callback-controlled rows.
            pool = getattr(self, 'axis_pool', None)
            parallel_advanced = {}
            if pool is not None and on_decision is None:
                jobs = [(j, i) for j, i in enumerate(idx) if allowed[j] and commits[j] > 0]
                tasks = [(paths[i], fr[j], points[j], int(commits[j]), int(heading_start[i]),
                          self.vol.shape, pp, length[i], hist_start[i]) for j, i in jobs]
                if tasks:
                    parallel_advanced = dict(zip(
                        [j for j, _ in jobs], pool.map(advance_trace_path, *zip(*tasks), chunksize=8)))
            # Next-step CT frames, resolved together after this step's commits.
            # Each depends only on its own trace, so the pool is a pure speedup.
            reframe = {}
            committed = []  # sequence history: steps that advanced this round
            for j, i in enumerate(idx):
                conf = np.minimum.accumulate(confidence[j], axis=-1)
                commit = int(commits[j])
                recovery_blocked = not allowed[j]
                # Same-position refinement is already exhausted inside the model: a
                # rejected decision stops the trace immediately; nothing is forced.
                would_stop = commit == 0
                refit = None
                if would_stop and pp.refit_retry and not retried[i] and not recovery_blocked:
                    refit = predicted_axis(pos[j]+points[j] @ fr[j].T, pos[j], self.refit_window)
                if on_decision is None:
                    verdict = None
                else:
                    state = dict(pos=pos[j].copy(), frame=fr[j].copy(), hist=hist_world[j].copy(), hmask=hm[j].copy(),
                               points=points[j].copy(), confidence=conf.copy(), n_commit=commit, would_stop=would_stop,
                               recovery_blocked=bool(recovery_blocked),
                               travelled=float(length[i]), last_segment=last_segment[i].copy(),
                               observed_path=np.asarray(paths[i]).copy(),
                               heading_start=int(heading_start[i]),
                               frame_policy=(initial_states[i]['frame_policy'] if initial_states is not None and length[i] == 0
                                             else PREDICTION_FRAME_POLICY if predicted[i]
                                             else getattr(self, 'frame_policy', FRAME_POLICY)),
                               fiber_family=families[i], ct_frame_diagnostics=frame_diagnostics[i].copy(),
                               refit_retry=refit is not None)
                    state.update({**references[i], 'seed_age': references[i]['seed_age']+float(length[i])})
                    verdict = on_decision(int(i), state)
                if verdict is False or isinstance(verdict, str):
                    active[i], reasons[i] = False, verdict if isinstance(verdict, str) else 'oracle'
                    continue
                if recovery_blocked:
                    active[i], reasons[i] = False, 'recovery_limit'
                    continue
                if would_stop and refit is not None:
                    # Retry once from the same head, the crop refit to this decision's own predicted fiber.
                    frames[i] = frame_from_heading(refit, fr[j][:, 0])
                    retried[i], predicted[i] = True, True
                    continue
                if would_stop:
                    active[i], reasons[i] = False, 'confidence'
                    continue
                if j in parallel_advanced:
                    advanced, reason = parallel_advanced[j]
                else:
                    advanced, reason = advance_trace_path(
                        paths[i], fr[j], points[j], commit, int(heading_start[i]),
                        self.vol.shape, pp, length[i], hist_start[i])
                if advanced is None:
                    active[i], reasons[i] = False, reason
                    continue
                if sequence is not None:
                    committed.append((j, i, pos[j].copy(), points[j][:commit].copy(), float(length[i])))
                paths[i] = advanced['path']
                last_segment[i] = advanced['last_segment']
                length[i] = advanced['travelled']
                heading_start[i] = advanced['heading_start']
                reframe[i] = advanced['heading']
                retried[i] = False
                prediction[i] = (pos[j]+points[j] @ fr[j].T)[commit:]
                if length[i] >= pp.max_len-1e-6:
                    active[i], reasons[i] = False, 'max_len'
            if sequence is not None:
                self.sequence_record(out['sequence_cells'], committed, sequence)
            mark('advance')
            if predictor is not None:
                pending = [i for i in reframe if active[i]]
                updated = predict_frames(predictor, self.vol, [paths[i][-1] for i in pending],
                    [reframe[i] for i in pending], [np.asarray(paths[i]) for i in pending],
                    [families[i] for i in pending], previous=[frames[i] for i in pending], pool=getattr(self, 'pool', None))
                for i, frame in zip(pending, updated):
                    frames[i] = frame
                    frame_diagnostics[i] = dict(source=3, energy=0., gap=0.)
            else:
                update = lambda i: ct_frame(self.vol, paths[i][-1], reframe[i], frames[i],
                                            diagnostics=frame_diagnostics[i])
                for i, frame in zip(reframe, self.map(update, reframe)):
                    frames[i] = frame
            # Continuations: forward along the previous prediction's axis, roll from the frame just built. Each axis
            mark('heading')
            # depends only on its own trace; an optional process pool (axis_pool) computes them in parallel with the
            # same values. (Threads are avoided here: running predicted_axis in the thread pool crashed CPython 3.14.)
            rows = [i for i in reframe if self.prediction_window and prediction[i] is not None and len(prediction[i])]
            pool = getattr(self, 'axis_pool', None)
            heads_ = [paths[i][-1] for i in rows]
            futures_ = [prediction[i] for i in rows]
            if pool is not None and rows:
                axes = list(pool.map(predicted_axis, futures_, heads_, [self.prediction_window]*len(rows),
                                     chunksize=max(1, len(rows)//32)))
            else:
                axes = [predicted_axis(f, h, self.prediction_window) for f, h in zip(futures_, heads_)]
            axis_by_row = dict(zip(rows, axes))
            for i in reframe:
                axis = axis_by_row.get(i)
                if axis is not None:
                    frames[i] = frame_from_heading(axis, np.asarray(frames[i])[:, 0])
                    frame_diagnostics[i] = dict(frame_diagnostics[i], source=4)
                predicted[i] = axis is not None
            mark('axes')
            if on_finished is not None:
                finish(was_active[~active[was_active]])
            mark('finish')
        return [np.asarray(p[h:]) for p, h in zip(paths, hist_start)], reasons


class FiberTracer(ModelTracer):
    path_context = True

    def __init__(self,model,*args,**kwargs):
        super().__init__(model,*args,**kwargs)
        self.observations = observation_builder(model.cfg)

    def build_inputs(self,pos,frames,hist,hmask,paths=None):
        items = [dict(pos=p,frame=f,hist_local=h,hmask=m,frame_policy=getattr(self, 'frame_policy', FRAME_POLICY))
                 for p,f,h,m in zip(pos,frames,hist,hmask)]
        if paths is not None:
            for item,path in zip(items,paths):
                item.update({k:path[k] for k in SEED_FIELDS if k in path})
                item['observed_path'] = path['observed_path']
                if 'fiber_family' in path:
                    item['fiber_family'] = path['fiber_family']
        def move(value):
            return {k: move(v) for k, v in value.items()} if isinstance(value, dict) else value.to(self.device)
        return move(self.observations.images(items,self.vol,self.pool))
