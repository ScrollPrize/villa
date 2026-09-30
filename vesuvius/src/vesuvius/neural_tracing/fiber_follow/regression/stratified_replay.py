"""Visual history collection and differentiable memory reconstruction at decisions.

All observations contribute detached appearance tokens. Only crops selected for
future encoder gradients are retained on the CPU. A decision reconstructs every
preceding writer transition at current weights and re-encodes up to three past
crops. This is sparse encoder training with possibly stale cached features, not
full-history image backpropagation. There is one task loss per selected decision.
"""


def stratified_indices(length, rng):
    """One observation per age band; the current decision is encoded separately."""
    selected = []
    for lo, hi in ((1, 4), (5, 16), (17, length-1)):
        hi = min(hi, length-1)
        if hi >= lo:
            selected.append(length-1-int(rng.integers(lo, hi+1)))
    return set(selected)


def take_rows(batch, indices):
    return {k: take_rows(v, indices) if isinstance(v, dict) else v[indices]
            for k, v in batch.items()}


def take_row(batch, j):
    return {k: take_row(v, j) if isinstance(v, dict) else v[j:j+1].detach().clone()
            for k, v in batch.items()}


class ObservationHistory:
    """Own detached stream evidence; never persist a writer autograd graph."""
    def __init__(self):
        self.streams = {}

    def start(self, cpu):
        keys = cpu['stream_id'].tolist()
        if len(set(keys)) != len(keys):
            raise ValueError('A crop batch must contain distinct streams')
        for key, reset, index in zip(keys, cpu['stream_reset'].tolist(), cpu['stream_index'].tolist()):
            if reset:
                if index or key in self.streams:
                    raise ValueError('Stream reset must start a new history at index zero')
                self.streams[key] = []
            if key not in self.streams or len(self.streams[key]) != index:
                raise ValueError('Missing or out-of-order observation history')

    def record(self, cpu, features):
        for j, key in enumerate(cpu['stream_id'].tolist()):
            rows = self.streams[key]
            index = int(cpu['stream_index'][j])
            if len(rows) != index:
                raise ValueError('Observation history must be recorded exactly once')
            if bool(cpu['stream_end'][j]):
                del self.streams[key]
                continue
            for row in rows:
                if row['retain_until'] <= index:
                    row['batch'] = None
            retain = int(cpu['retain_until'][j])
            rows.append(dict(features=tuple(v[j:j+1].detach().clone() for v in features),
                pose={name: cpu['x'][name][j:j+1].clone() for name in
                      ('query_position', 'query_frame', 'feature_seed_here')},
                batch=take_row({k: cpu[k] for k in ('x', 'hist', 'hmask')}, j) if retain > index else None,
                retain_until=retain))

    def reconstruct(self, model, cpu, device):
        """Only the prefix before this decision enters its gradient-bearing memory."""
        from .train import move_batch, training_observation_features, training_memory_transition
        if len(cpu['hist']) != 1:
            raise ValueError('Reconstruct one independent decision history at a time')
        rows = self.streams[int(cpu['stream_id'][0])]
        if len(rows) != int(cpu['stream_index'][0]):
            raise ValueError('Decision must follow exactly its causal observation prefix')
        selected = set(cpu['encoder_indices'][0].tolist())-{-1}
        if any(i < 0 or i >= len(rows) for i in selected):
            raise ValueError('Encoder selection must precede the decision')
        state = None
        for i, row in enumerate(rows):
            if i in selected:
                if row['batch'] is None:
                    raise ValueError('Selected historical crop was not retained')
                batch = move_batch(row['batch'], device)
                features = training_observation_features(model, batch['x'], batch['hist'], batch['hmask'])
            else:
                features = row['features']
            state, _ = training_memory_transition(model, *features, move_batch(row['pose'], device), state)
        return state, dict(memory_replay_observations=len(rows), history_encoder_crops=len(selected))
