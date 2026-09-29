"""Causal, supervised streams of main crops, shared across gradient chunks.

Workers own observation streams; the training process owns their tensor states.
Only geometry/labels, never encoder features, are created in loader workers.
"""
import numpy as np
import torch

from .spatial_sequences import decision_at
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, frame_from_heading, interp_at, normalize
from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed


FEATURE_SAMPLING_REVISION = 3  # causal headings, generator-certified history, equal crop weights

def seed_observation(item, cfg):
    return dict(pos=np.asarray(item['seed_pos']).copy(),
        frame=frame_from_heading(normalize(np.asarray(item['seed_tangent']))),
        hist_local=np.zeros((cfg.n_history, 3)), hmask=np.zeros(cfg.n_history),
        seed_pos=np.asarray(item['seed_pos']).copy(), seed_tangent=np.asarray(item['seed_tangent']).copy(),
        seed_valid=True, seed_age=0., memory_warm=True)


def stream_rows(item, builder, band=None):
    """Use observed paths (including drift), not annotation trajectories as inputs.

    Retain the actual seed and the newest memory_steps historical decisions.
    Every selected position is encoded once; known membership supplies supervision.
    Unknown bridge states retain their observations but have masked supervision.
    Reject an entire stream if any crop/label crosses the held-out band.
    """
    cfg = builder.cfg
    track = item.get('memory_track')
    if track is None or not len(track['pos']):
        mask = np.asarray(item['hmask']) > 0
        hist = np.asarray(item['hist_local'])[mask][::-1] @ np.asarray(item['frame']).T+item['pos']
        path = np.concatenate((hist, np.asarray(item['pos'])[None]))
        arc = arclength(path)
        sample_s = np.arange(0, arc[-1], cfg.memory_stride)
        if len(path) > 1 and arc[-1] > 0:
            pos = interp_at(path, arc, sample_s)
            behind = interp_at(path, arc, np.maximum(0, sample_s-.5))
            keep = np.linalg.norm(pos-behind, axis=1)>1e-6
            seed_here = bool(item.get('seed_valid', False) and np.linalg.norm(pos[0]-item['seed_pos'])<1e-4)
            if seed_here:
                keep[0] = True
            frames = np.asarray([frame_from_heading(item['seed_tangent'] if k==0 and seed_here else normalize(p-q))
                for k,(p,q) in enumerate(zip(pos,behind)) if keep[k]]).reshape(-1,3,3)
            pos = pos[keep]
        else:
            pos, frames = np.empty((0, 3)), np.empty((0, 3, 3))
        membership = np.zeros(len(pos))
        if not item.get('_generated_original_history', False):
            membership[:] = np.nan
            if '_constructed_path' in item and len(pos):
                from .neighbor_mining import exact_nearest
                distance, _, segment, u = exact_nearest(pos, item['_constructed_path'])
                arcs = item['_constructed_arc']
                at = arcs[segment]+u*np.diff(arcs)[segment]
                # Reconstructed chords can cut across bends: certify only points
                # still close to the generated path and outside its unknown bridge.
                close = distance<=.25
                membership[close & (at<=item['_leave_arc']+1e-6)] = 0.
                membership[close & (at>=item['_reach_arc']-1e-6)] = 1.
        track = dict(pos=pos, frame=frames, offtrack=membership, offset=np.full((len(pos),3),np.nan))
    # Label/history construction still sees the whole causal observed prefix;
    # the feature stream only reads the retained keyframes and its immutable seed.
    track = {key: np.asarray(value).copy() for key, value in track.items()}
    horizon = cfg.feature_stream_steps if cfg.feature_memory_revision == 2 else cfg.memory_steps
    indices = list(range(max(0, len(track['pos'])-horizon), len(track['pos'])))
    if item.get('seed_valid', False):
        seed = seed_observation(item, cfg)
        # Prepending supplies causal history; do not duplicate an existing seed.
        if not len(track['pos']) or np.linalg.norm(track['pos'][0]-seed['pos']) > 1e-4:
            certified = item.get('_seed_original_certified', False)
            additions = dict(pos=seed['pos'], frame=seed['frame'], offtrack=0. if certified else np.nan,
                             offset=np.zeros(3) if certified else np.full(3,np.nan))
            track = {key: np.concatenate((np.asarray(additions[key])[None], value)) for key, value in track.items()}
            indices = [j+1 for j in indices]
        if not indices or indices[0] != 0:
            indices.insert(0, 0)
    source = dict(item, memory_track=track)
    rows, observed = [], []
    for j in indices:
        # Current crop is emitted separately, with its original hard-example labels.
        if np.linalg.norm(track['pos'][j]-item['pos']) < 1e-4:
            continue
        row = decision_at(source, builder, j, observed_indices=observed)
        if row is None:
            return []
        if not np.isfinite(track['offtrack'][j]):
            row['identity_observable'] = False
        rows.append(row)
        observed.append(j)
    endpoint = dict(item, memory_track={key: value[observed].copy() for key, value in track.items()})
    # Re-evaluate identity observability against evidence actually retained by
    # this stream. A truncated-away observation cannot justify supervision.
    augmentation = {key: endpoint[key] for key in
                    ('photometric', 'drop_presence', 'blur_sigma', 'identity_seed') if key in endpoint}
    endpoint.pop('_identity_prepared', None)
    fiber = endpoint.get('supervision_fiber')
    if fiber is None:
        fiber = builder.fibers[endpoint['fiber_ref'][0]]
    builder.prepare(endpoint, fiber, np.random.default_rng(item['identity_seed']))
    endpoint.update(augmentation)
    rows.append(endpoint)
    for row in rows:
        # prepare() above established observability from the causal prefix.
        # images() must never reconstruct historical crops for these rows.
        row['memory_warm'] = True
        if not training_state_allowed(row, cfg.fine, band) or not builder.footprint_allowed(row, band):
            return []
    return rows


def sequence_batches(builder, items, vol, *, band=None, worker=0, requested_fraction=0.):
    """Yield bounded chunks; ids survive worker interleaving and optimizer updates."""
    streams = [stream_rows(item, builder, band) for item in items]
    streams = [rows for rows in streams if rows]
    if not streams:
        return
    group = getattr(builder, '_feature_group', 0)
    builder._feature_group = group+1
    ids = [(worker << 48)+(group << 16)+j for j in range(len(streams))]
    length = builder.cfg.feature_sequence_length
    selected = None
    if builder.cfg.feature_memory_revision == 2 and builder.cfg.feature_replay_weight:
        from .stratified_replay import stratified_indices
        selected = [stratified_indices(len(rows), np.random.default_rng(int(rows[-1].get('identity_seed', 0))))
                    for rows in streams]
    for lo in range(0, max(map(len, streams)), length):
        sequence = []
        for t in range(lo, lo+length):
            active = [j for j, rows in enumerate(streams) if t < len(rows)]
            if not active:
                break
            batch = builder([streams[j][t] for j in active], vol)
            batch['stream_id'] = torch.tensor([ids[j] for j in active], dtype=torch.long)
            batch['stream_reset'] = torch.full((len(active),), t == 0, dtype=torch.bool)
            batch['stream_end'] = torch.tensor([t == len(streams[j])-1 for j in active])
            batch['decision_requested'] = torch.full((len(active),), requested_fraction)
            if selected is not None:
                batch['replay_select'] = torch.tensor([t in selected[j] for j in active])
            sequence.append(batch)
        yield dict(feature_sequence=sequence)


def sequence_steps(batch):
    return batch.get('feature_sequence', [batch])


class FeatureStreamStates:
    """Training-only state ownership, with explicit reset, detach and eviction."""
    def __init__(self):
        self.states = {}
        from .stratified_replay import StratifiedReplay
        self.replay = StratifiedReplay()

    def incoming(self, model, batch, device):
        initial = model.initial_memory(1, device)
        rows = []
        for key, reset in zip(batch['stream_id'].tolist(), batch['stream_reset'].tolist()):
            if reset:
                self.states.pop(key, None)
                rows.append(initial)
            elif key in self.states:
                rows.append(self.states[key])
            else:
                raise ValueError(f'Missing carried feature memory for stream {key}; a reset must precede continuation')
        return {name: torch.cat([row[name] for row in rows]) for name in initial}

    def update(self, batch, output, names):
        for j, (key, end) in enumerate(zip(batch['stream_id'].tolist(), batch['stream_end'].tolist())):
            if end:
                self.states.pop(key, None)
            else:
                self.states[key] = {name: output['memory_'+name][j:j+1] for name in names}

    def detach(self):
        self.states = {key: {name: value.detach() for name, value in state.items()}
                       for key, state in self.states.items()}
