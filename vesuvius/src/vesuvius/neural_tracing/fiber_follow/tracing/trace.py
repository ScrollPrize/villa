"""Autoregressive fiber tracing with trusted twelve-voxel heading fits."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid, normalize
from vesuvius.neural_tracing.fiber_follow.tracing.policy import (DEFAULT_CONFIDENCE, DEFAULT_GATE, DEFAULT_N_COMMIT, GATES, gate_horizon,
    commit_count, selection_window)
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolume
from vesuvius.neural_tracing.fiber_follow.shared.reference import SEED_FIELDS, observed_path
from vesuvius.neural_tracing.fiber_follow.tracing.heading import trace_heading, TRACE_HEADING_POLICY, FRAME_POLICY, FRAME_POLICIES, ct_frame, fiber_family
from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import frame_predictor, configured_frame_policy, predict_frames

if TYPE_CHECKING:
    from vesuvius.neural_tracing.fiber_follow.models.model import ObservationFollower as FollowNet


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
        self.frame_predictor = frame_predictor(model.cfg)
        self.frame_policy = configured_frame_policy(model.cfg)
        self.heading_policy = 'learned_heading_v1' if self.frame_predictor is not None else TRACE_HEADING_POLICY
        if self.p.n_commit > model.cfg.n_future:
            raise ValueError(f'n_commit={self.p.n_commit} exceeds the model horizon n_future={model.cfg.n_future}')
        self.grid = torch.from_numpy(crop_local_grid(crop)).float().to(device)
        self.S = crop.block_size
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
                outputs.append(self.model(rows(x, part), hist[part], hmask[part], **rows(sampling, part)))
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

    # Decision memory: each decision's encoder entry is recorded once and read by later
    # decisions of the same trace; entries for a pre-existing prefix are encoded once
    # from their crops. Records and features are pruned to what later selections can read.

    def sequence_attach(self, x, idx, sequence, pos, frames, length):
        """Sequence follower (models/sequence.py): each row reads its trace's earlier history tokens (per-layer
        states, at most ``history_limit`` most recent) with their pose in this decision's frame."""
        from vesuvius.neural_tracing.fiber_follow.models.sequence import relative_pose
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

    @staticmethod
    def memory_start(paths):
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
        from vesuvius.neural_tracing.fiber_follow.data.decision_memory import simulated_decisions
        memory = []
        for path in paths:
            path = np.asarray(path, np.float64)
            records = simulated_decisions(path, arclength(path)) if len(path) > 1 else []
            memory.append(dict(decisions=records, features={}, next_key=0))
        return memory

    def memory_attach(self, x, idx, memory):
        """Replace memory crops by recorded features: (B, SLOTS, *cfg.memory_entry_shape)."""
        encode, keys, valid = x['history_encode'], x['history_keys'].cpu().numpy(), x['history_valid'].cpu().numpy()
        entries = self._memory_entries
        features = torch.zeros((*encode.shape, *self.model.cfg.memory_entry_shape), device=encode.device,
                               dtype=torch.bfloat16)
        flat = encode.flatten().nonzero().flatten()
        if len(flat):
            with self.autocast():
                fresh = self.model.encode_memory_crops(x, flat)
            for value, cell in zip(fresh, flat.tolist()):
                j, slot = divmod(cell, encode.shape[1])
                state = memory[idx[j]]
                record = entries[j][slot]['record']
                record['key'], state['next_key'] = state['next_key'], state['next_key']+1
                state['features'][record['key']] = value
                keys[j, slot] = record['key']
        for j, i in enumerate(idx):
            for slot in np.flatnonzero(valid[j]):
                features[j, slot] = memory[i]['features'][int(keys[j, slot])]
        for key in ('history_crops', 'history_references', 'history_reference_mask'):
            x.pop(key)
        x['history_encode'] = torch.zeros_like(encode)
        x['history_features'] = features

    @staticmethod
    def memory_record(entries, idx, pos, frames, paths, memory):
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
        from vesuvius.neural_tracing.fiber_follow.models.decision_memory import prune_decisions
        for j, i in enumerate(idx):
            state = memory[i]
            length = float(arclength(np.asarray(paths[i], np.float64))[-1])
            key, state['next_key'] = state['next_key'], state['next_key']+1
            state['features'][key] = entries[j].detach()
            records = state['decisions']+[dict(travelled=length, pos=pos[j].copy(), frame=frames[j].copy(), key=key)]
            records = [records[k] for k in prune_decisions([r['travelled'] for r in records], length)]
            kept = {r['key'] for r in records}
            state['features'] = {k: v for k, v in state['features'].items() if k in kept}
            state['decisions'][:] = records

    @torch.no_grad()
    def trace(self, seeds_xyz, headings, histories=None, abort=None, on_decision=None, initial_states=None, families=None):
        """Greedy rollout, checking supervised confidence before committing.

        on_decision(i, state) observes the exact inference input and proposals,
        before any commit; returning False censors the trace. abort(i, path)
        remains an optional post-commit geometric limit. Histories run oldest
        to newest and exclude the seed. Returned paths always start at the seed.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            return self._trace(seeds_xyz, headings, histories, abort, on_decision, initial_states, families)
        finally:
            self.model.train(was_training)

    def _trace(self, seeds_xyz, headings, histories, abort, on_decision, initial_states=None, families=None):
        from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at
        n = len(seeds_xyz)
        predictor = getattr(self, 'frame_predictor', None)
        if families is None and initial_states is not None and predictor is not None:
            families = [s.get('fiber_family') for s in initial_states]
        if predictor is not None and (families is None or len(families) != n):
            raise ValueError('Learned crop frames require one H/V family per trace')
        families = [fiber_family(f) for f in families] if families is not None else [None]*n
        paths = [[np.asarray(s, np.float64)] for s in seeds_xyz]
        if histories is not None:
            paths = [list(np.asarray(h, np.float64))+[np.asarray(s, np.float64)] for h, s in zip(histories, seeds_xyz)]
        if initial_states is not None:
            if len(initial_states) != n:
                raise ValueError('One initial observed state is required per seed')
            if any(s['frame_policy'] not in FRAME_POLICIES for s in initial_states):
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
        if initial_states is not None:
            frames = [np.asarray(s['frame']).copy() for s in initial_states]
        elif predictor is not None:
            frames = predict_frames(predictor, self.vol, seeds_xyz, headings, [np.asarray(p) for p in paths], families,
                                    pool=getattr(self, 'pool', None))
            frame_diagnostics = [dict(source=3, energy=0., gap=0.) for _ in range(n)]
        else:
            frames = list(self.map(lambda a: ct_frame(self.vol, a[0], a[1], diagnostics=a[2]),
                                   zip(seeds_xyz, headings, frame_diagnostics)))
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
        memory = self.memory_start(paths) if getattr(self.model.cfg, 'memory', 'slabs') == 'decisions' else None
        sequence = ([dict(layers=[[] for _ in range(self.model.cfg.layers)], heads=[], travelled=[]) for _ in range(n)]
                    if getattr(self.model.cfg, 'model_type', '') == 'sequence' else None)
        active = np.ones(n, bool)
        reasons = ['']*n
        length = np.zeros(n)
        last_segment = [np.asarray([p[-1]]) for p in paths]
        decisions = np.zeros(n, np.int64)  # per trace, for the flow sampler's decision keys
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
                                         fiber_family=families[i],
                                         seed_segment=np.asarray(paths[i][hist_start[i]:hist_start[i]+64]),
                                         travelled=float(length[i]),
                                         **{**references[i], 'seed_age': references[i]['seed_age']+float(length[i])}) for i in idx]
                if memory is not None:
                    for path, i in zip(context['paths'], idx):
                        path['memory_decisions'] = memory[i]['decisions']
            x = self.build_inputs(pos, fr, hist, hm, **context)
            if memory is not None:
                self.memory_attach(x, idx, memory)
            if sequence is not None:
                self.sequence_attach(x, idx, sequence, pos, fr, length)
            sampling = {}
            if (hasattr(self.model, 'select_prediction')
                    or getattr(self.model.cfg, 'candidate_selection', 'prefix') == 'stop_fallback'):
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
            if memory is not None:
                self.memory_record(out['memory_entry'], idx, pos, fr, paths, memory)
            points, confidence = [out[k].float().cpu().numpy() for k in ('points', 'confidence')]
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
                state = dict(pos=pos[j].copy(), frame=fr[j].copy(), hist=hist_world[j].copy(), hmask=hm[j].copy(),
                             points=points[j].copy(), confidence=conf.copy(), n_commit=commit, would_stop=would_stop,
                             recovery_allowed=bool(allowed[j]), recovery_blocked=bool(recovery_blocked),
                             travelled=float(length[i]), last_segment=last_segment[i].copy(),
                             observed_path=np.asarray(paths[i]).copy(),
                             heading_start=int(heading_start[i]),
                             heading_policy='learned_heading_v1' if predictor is not None else TRACE_HEADING_POLICY,
                             frame_policy=(initial_states[i]['frame_policy'] if initial_states is not None and length[i] == 0
                                           else getattr(self, 'frame_policy', FRAME_POLICY)),
                             fiber_family=families[i], ct_frame_diagnostics=frame_diagnostics[i].copy())
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
                if sequence is not None:
                    committed.append((j, i, pos[j].copy(), points[j][:commit].copy(), float(length[i])))
                paths[i] = list(advanced['path'])
                last_segment[i] = advanced['last_segment']
                length[i] = advanced['travelled']
                heading_start[i] = advanced['heading_start']
                reframe[i] = advanced['heading']
                if abort is not None and abort(int(i), paths[i]):
                    active[i], reasons[i] = False, 'abort'
                elif length[i] >= pp.max_len-1e-6:
                    active[i], reasons[i] = False, 'max_len'
            if sequence is not None:
                self.sequence_record(out['sequence_cells'], committed, sequence)
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
        return [np.asarray(p[h:]) for p, h in zip(paths, hist_start)], reasons
