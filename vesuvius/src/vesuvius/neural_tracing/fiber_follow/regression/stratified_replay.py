"""Sparse encoder gradients over a complete chronological memory replay.

Only three selected historical crops are retained on the CPU. Other observations
retain detached compact features made by the live training model (and therefore
possibly older weights). Every writer transition is recomputed at current weights
within one optimizer update. This is a selective, stale-feature approximation,
not an unbiased estimator of full-history encoder backpropagation.
"""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def stratified_indices(length, rng):
    """One observation per available age band; endpoint is always trained separately."""
    selected = []
    for lo, hi in ((1, 4), (5, 16), (17, length-1)):
        hi = min(hi, length-1)
        if hi >= lo:
            selected.append(length-1-int(rng.integers(lo, hi+1)))
    return set(selected)


def take_row(batch, j):
    return {k: take_row(v, j) if isinstance(v, dict) else v[j:j+1].detach().clone()
            for k, v in batch.items()}


class StratifiedReplay:
    def __init__(self):
        self.streams = {}
        self.pending = []
        self._compiled = None

    def transition(self, raw, compiled):
        """The writer transition; compiled when the training forward is.

        A replay runs up to feature_stream_steps small transitions in sequence,
        so eager execution is dominated by kernel launches. Batch size and
        state shapes are fixed, so a handful of static graphs cover every call.
        """
        base = getattr(raw, 'replay_transition', raw.recurrent_memory.observe_tokens)
        if not compiled:
            return base
        if self._compiled is None or self._compiled[0] is not raw:
            self._compiled = (raw, torch.compile(base, dynamic=False))
        return self._compiled[1]

    def record(self, cpu, output):
        if 'replay_select' not in cpu:
            return
        for j, (key, reset, end, selected) in enumerate(zip(cpu['stream_id'].tolist(),
                cpu['stream_reset'].tolist(), cpu['stream_end'].tolist(), cpu['replay_select'].tolist())):
            if reset:
                self.streams[key] = []
            if key not in self.streams:
                raise ValueError('Replay continuation without reset')
            rows = self.streams[key]
            # Clone only the small observation, never a view retaining the whole
            # encoder output or a selected crop's other batch members.
            row = dict(features=tuple(output['observation_'+name][j:j+1].detach().clone()
                                      for name in ('tokens', 'xyz', 'valid')),
                       pose={name: cpu['x'][name][j:j+1].clone() for name in
                             ('query_position', 'query_frame', 'feature_seed_here')},
                       batch=take_row(cpu if end else {k: cpu[k] for k in ('x', 'hist', 'hmask')}, j)
                             if selected or end else None)
            rows.append(row)
            if end:
                self.pending.append(self.streams.pop(key))

    def backward(self, model, total, *, device, tolerance, n_commit, confidence_weight,
                 candidate_weight, memory_probe_weight):
        from .train import move_batch
        from .supervision import loss_terms, memory_probe_terms
        raw = getattr(model, '_orig_mod', model)
        weight = raw.cfg.feature_replay_weight
        metrics = dict(replay_endpoints=0, replay_observations=0, replay_encoder_crops=0, replay_loss=0.)
        transition = self.transition(raw, compiled=raw is not model)
        while self.pending:
            rows = self.pending.pop(0)
            if len(rows) < 2 or not weight:
                continue
            state = None
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=torch.device(device).type == 'cuda'):
                for row in rows[:-1]:
                    if row['batch'] is not None:
                        batch = move_batch(row['batch'], device)
                        # Checkpoint only selected image encodings. Do not detach
                        # the recurrent state between selected observations.
                        features = checkpoint(raw.observation_features, batch['x'], batch['hist'], batch['hmask'],
                                              use_reentrant=False, preserve_rng_state=False)
                        metrics['replay_encoder_crops'] += 1
                    else:
                        features = row['features']
                    pose = move_batch(row['pose'], device)
                    state, _ = transition(*features, pose, state)
                    state = {k: v for k, v in state.items() if k != 'probe'}
                batch = move_batch(rows[-1]['batch'], device)
                kwargs = dict(candidates=batch['candidate_points']) if 'candidate_points' in batch else {}
                output = model(batch['x'], batch['hist'], batch['hmask'], memory=state, **kwargs)
                terms = loss_terms(output, batch, raw.cfg, tolerance, n_commit=n_commit)
                loss = terms['geometry_per_state'].sum()+confidence_weight*terms['confidence_per_state'].sum()
                if 'proposal_per_state' in terms:
                    loss = loss+terms['proposal_per_state'].sum()
                if 'candidate_per_state' in terms:
                    loss = loss+candidate_weight*terms['candidate_per_state'].sum()
                if 'memory_probe' in output and 'memory_target_identity' in batch:
                    probe = memory_probe_terms(output, batch, departed_weight=raw.cfg.memory_departed_weight)
                    loss = loss+memory_probe_weight*(probe['memory_identity_per_state'].sum()+probe['memory_offset_per_state'].sum())
                loss = loss*(weight/total)
            loss.backward()
            # Stays on the device; the training update resolves all sums with one transfer.
            metrics['replay_loss'] = metrics['replay_loss']+loss.detach().double()
            metrics['replay_endpoints'] += 1
            metrics['replay_observations'] += len(rows)
            metrics['replay_encoder_crops'] += 1
        return metrics
