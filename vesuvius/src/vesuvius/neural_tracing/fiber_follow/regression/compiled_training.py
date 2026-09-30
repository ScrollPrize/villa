"""Fixed tensor contracts for training; adaptive inference remains in model.py.

There are four deliberate compiled operations: a fixed crop batch, candidate
scoring, a single replay encoding, and a single replay memory transition.
Padding, stream ownership and truncation of recurrent gradients stay outside
these graphs. No parameter or checkpoint schema is changed.
"""
import torch

from .identity_decisions import CANDIDATE_COUNT


def fixed_rows(value, size):
    """Pad with finite copies; the caller discards the extra output rows."""
    if not 0 < len(value) <= size:
        raise ValueError(f'Expected 1..{size} rows, got {len(value)}')
    if len(value) < size:
        value = torch.cat((value, value[:1].expand(size-len(value), *value.shape[1:])))
    # Singleton dimensions can otherwise retain the stride of an indexed view.
    return torch.empty_like(value, memory_format=torch.contiguous_format).copy_(value)


def differentiable_state(state, size):
    """Uniform autograd metadata without reconnecting a detached history.

    Constant/detached fields become leaves whose input gradients are discarded.
    Already attached fields retain their graph, including learned reset slots
    and the preceding decision in a truncated-BPTT chunk.
    """
    result = {}
    for name in sorted(state):
        value = state[name]
        value = fixed_rows(value, size)
        if value.is_floating_point() and not value.requires_grad:
            value.requires_grad_(True)
        result[name] = value
    return result


def pack_feature_chunks(chunks, size):
    """Pack independent partial chunks within their existing optimizer update.

    Never join consecutive chunks of one stream: that would change its BPTT
    boundary. Full worker chunks are reused without copying their large crops.
    Real rows alone reach the losses, metrics and persistent stream state.
    """
    if any('feature_sequence' not in chunk or 'stream_id' not in chunk['feature_sequence'][0]
           for chunk in chunks):
        return chunks
    seen, complete, pending = set(), [], []
    def row(batch, j):
        return {key: row(value, j) if isinstance(value, dict) else value[j:j+1]
                for key, value in batch.items()}
    for chunk in chunks:
        sequence = chunk['feature_sequence']
        names = sequence[0]['stream_id'].tolist()
        if seen.intersection(names):
            return chunks
        seen.update(names)
        if len(names) == size:
            complete.append(chunk)
            continue
        if len(names) > size:
            raise ValueError('Worker chunk exceeds compiled stream batch')
        for name in names:
            rows = []
            for batch in sequence:
                ids = batch['stream_id'].tolist()
                if name in ids:
                    rows.append(row(batch, ids.index(name)))
            pending.append(rows)
    def schema(batch):
        return tuple((key, schema(value) if isinstance(value, dict) else (value.dtype, value.shape[1:]))
                     for key, value in sorted(batch.items()))
    if pending and any(schema(rows[0]) != schema(pending[0][0]) for rows in pending):
        return chunks
    def combine(rows):
        result = {}
        for key in rows[0]:
            values = [r[key] for r in rows]
            if isinstance(values[0], dict):
                result[key] = combine(values)
            elif len(values) == 1:
                result[key] = values[0]
            elif values[0].is_pinned() and not any(v.requires_grad for v in values):
                target = torch.empty((sum(len(v) for v in values), *values[0].shape[1:]),
                                     dtype=values[0].dtype, pin_memory=True)
                result[key] = torch.cat(values, out=target)
            else:
                result[key] = torch.cat(values)
        return result
    pending.sort(key=len, reverse=True)
    for start in range(0, len(pending), size):
        streams = pending[start:start+size]
        complete.append(dict(feature_sequence=[combine([rows[t] for rows in streams if t < len(rows)])
                                               for t in range(max(map(len, streams)))]))
    return complete


class CompiledTrainingModel(torch.nn.Module):
    def __init__(self, model, batch_size=2, *, backend=None):
        super().__init__()
        if batch_size < 1:
            raise ValueError('Positive compiled batch size required')
        self._orig_mod = model
        self.batch_size = batch_size
        options = dict(dynamic=False, fullgraph=True)
        if backend is not None:
            options['backend'] = backend
        self._crop = torch.compile(model.training_forward, **options)
        self._candidates = torch.compile(model.score_candidates, **options)
        self._observation = torch.compile(model.replay_observation_features, **options)
        self._transition = torch.compile(model.recurrent_memory.observe_tokens, **options)
        self._refined = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__('_orig_mod'), name)

    def begin_update(self):
        self._refined = torch.zeros((), device=next(self.parameters()).device, dtype=torch.bool)

    def pack_batches(self, batches):
        return pack_feature_chunks(batches, self.batch_size)

    def finish_update(self):
        # Masked, unexecuted retries must preserve AdamW's grad=None semantics.
        if self.cfg.recurrent_refinement_steps and not bool(self._refined):
            for module in (self.refinement_fusion, self.refinement_stage):
                for parameter in module.parameters():
                    parameter.grad = None

    def inputs(self, x, hist, hmask, size):
        b = len(hist)
        defaults = dict(seed=hist.new_zeros(b, 1, 3), seed_mask=hist.new_zeros(b, 1),
                        seed_tangent=hist.new_zeros(b, 3), seed_age=hist.new_zeros(b))
        names = ('fine', 'seed', 'seed_mask', 'seed_tangent', 'seed_age',
                 'query_frame', 'query_position', 'feature_seed_here')
        image = {name: fixed_rows(x[name] if name in x else defaults[name], size) for name in names}
        return image, fixed_rows(hist, size), fixed_rows(hmask, size)

    def forward(self, x, hist, hmask, candidates=None, memory=None, confidence_threshold=.5, n_commit=None):
        if not self.training or not torch.is_grad_enabled():
            return self._orig_mod(x, hist, hmask, candidates=candidates, memory=memory,
                                  confidence_threshold=confidence_threshold, n_commit=n_commit)
        size, actual = self.batch_size, len(hist)
        if 'feature_seed_x' in x:
            seed_x = x['feature_seed_x']
            empty = torch.zeros_like(hmask)
            seed_ctx = self._orig_mod.context(seed_x, torch.zeros_like(hist), empty)
            memory, _ = self._orig_mod.observe_context(seed_ctx, seed_x, torch.zeros_like(hist), empty, memory)
        if memory is None:
            memory = self.initial_memory(actual, hist.device)
        image, history, mask = self.inputs(x, hist, hmask, size)
        state = differentiable_state(memory, size)
        threshold = hist.new_full((), confidence_threshold)
        output, context = self._crop(image, history, mask, state, threshold)
        if candidates is not None:
            count = candidates.shape[1]
            if not 0 < count <= CANDIDATE_COUNT:
                raise ValueError(f'Expected 1..{CANDIDATE_COUNT} candidate paths')
            curves = candidates.detach()
            if count < CANDIDATE_COUNT:
                curves = torch.cat((curves, curves[:, :1].expand(-1, CANDIDATE_COUNT-count, -1, -1)), 1)
            scored = self._candidates(context, fixed_rows(curves, size))
            output.update({name: value[:, :count] for name, value in scored.items()})
        output = {name: value[:actual] for name, value in output.items()}
        if self._refined is None:
            self.begin_update()
        self._refined = self._refined | output['refinement_mask'][:, 1:].any()
        return self.select_prediction(output, confidence_threshold, n_commit)

    def replay_observation_features(self, x, hist, hmask):
        # Historical crops are selected and encoded individually.
        if not torch.is_grad_enabled():
            return self._orig_mod.observation_features(x, hist, hmask)
        return self._observation(*self.inputs(x, hist, hmask, 1))

    def replay_transition(self, tokens, local_xyz, valid, pose, state=None):
        if state is None:
            state = self.initial_memory(1, tokens.device)
        tokens = fixed_rows(tokens, 1)
        if not tokens.requires_grad:
            tokens.requires_grad_(True)
        pose = {name: fixed_rows(pose[name], 1) for name in
                ('query_position', 'query_frame', 'feature_seed_here')}
        return self._transition(tokens, fixed_rows(local_xyz, 1), fixed_rows(valid, 1),
                                pose, differentiable_state(state, 1))
