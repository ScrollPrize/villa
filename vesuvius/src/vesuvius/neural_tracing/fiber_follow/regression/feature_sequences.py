"""Causal observation streams with sparse supervised decision positions.

Workers own observation streams; the training process owns their tensor states.
Only geometry/labels, never encoder features, are created in loader workers.
"""
import numpy as np
import torch

from .spatial_sequences import decision_at
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, frame_from_heading, interp_at, normalize
from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed


FEATURE_SAMPLING_REVISION = 8  # sparse decisions with one reconstructed-memory objective

def seed_observation(item, cfg):
    return dict(pos=np.asarray(item['seed_pos']).copy(),
        frame=frame_from_heading(normalize(np.asarray(item['seed_tangent']))),
        hist_local=np.zeros((cfg.n_history, 3)), hmask=np.zeros(cfg.n_history),
        seed_pos=np.asarray(item['seed_pos']).copy(), seed_tangent=np.asarray(item['seed_tangent']).copy(),
        seed_valid=True, seed_age=0., memory_warm=True)


def stream_rows(item, builder, band=None):
    """Use observed paths (including drift), not annotation trajectories as inputs.

    Retain the actual seed and the newest feature_stream_steps observations.
    Known membership supplies supervision at the sampled decision positions.
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
    horizon = cfg.feature_stream_steps
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


class SwitchCropBudget:
    """Admit whole streams using actual crop costs, without extra active state.

    Ordinary streams earn room for switches; excess switch proposals are dropped
    before image I/O. At completed-group boundaries switches cannot exceed the
    target share. Selection can undershoot if suitable switch proposals are rare.
    """
    def __init__(self, fraction):
        if not 0 <= fraction < 1:
            raise ValueError('Switch crop fraction must be in [0, 1)')
        self.fraction = fraction
        self.ordinary = self.switch = 0
        self.rejected_streams = 0

    def admit(self, streams):
        self.ordinary += sum(len(rows) for rows in streams if rows[-1].get('source') != 3)
        accepted = []
        for rows in streams:
            if rows[-1].get('source') == 3:
                proposed = self.switch+len(rows)
                if proposed > self.fraction*(self.ordinary+proposed)+1e-9:
                    self.rejected_streams += 1
                    continue
                self.switch = proposed
            accepted.append(rows)
        return accepted


def sequence_batches(builder, items, vol, *, band=None, worker=0, requested_fraction=0.):
    """Yield bounded chunks; ids survive worker interleaving and optimizer updates."""
    streams = [stream_rows(item, builder, band) for item in items]
    failed_pairs = {item['pair_observation_seed'] for item, rows in zip(items, streams)
                    if not rows and 'pair_observation_seed' in item}
    streams = [rows for item, rows in zip(items, streams)
               if rows and (not failed_pairs or item.get('pair_observation_seed') not in failed_pairs)]
    fraction = builder.cfg.feature_switch_crop_fraction
    if fraction >= 0:
        if not hasattr(builder, '_switch_crop_budget'):
            builder._switch_crop_budget = SwitchCropBudget(fraction)
        streams = builder._switch_crop_budget.admit(streams)
    if not streams:
        return
    plans = [decision_plan(len(rows), builder.cfg, np.random.default_rng(int(rows[-1]['identity_seed'])))
             for rows in streams]
    group = getattr(builder, '_feature_group', 0)
    builder._feature_group = group+1
    ids = [(worker << 48)+(group << 16)+j for j in range(len(streams))]
    length = builder.cfg.feature_sequence_length
    for lo in range(0, max(map(len, streams)), length):
        sequence = []
        for t in range(lo, lo+length):
            active = [j for j, rows in enumerate(streams) if t < len(rows)]
            if not active:
                break
            weights = torch.tensor([plans[j][0][t] for j in active], dtype=torch.float32)
            batch = builder([streams[j][t] for j in active], vol, decision_mask=weights > 0)
            batch['stream_id'] = torch.tensor([ids[j] for j in active], dtype=torch.long)
            batch['stream_reset'] = torch.full((len(active),), t == 0, dtype=torch.bool)
            batch['stream_end'] = torch.tensor([t == len(streams[j])-1 for j in active])
            batch['stream_index'] = torch.full((len(active),), t, dtype=torch.long)
            batch['loss_weight'] = weights
            batch['decision_mask'] = batch['loss_weight'] > 0
            batch['encoder_indices'] = torch.tensor(np.stack([plans[j][1][t] for j in active]), dtype=torch.long)
            batch['retain_until'] = torch.tensor([plans[j][2][t] for j in active], dtype=torch.long)
            batch['decision_requested'] = torch.full((len(active),), requested_fraction)
            sequence.append(batch)
        yield dict(feature_sequence=sequence)


def decision_plan(length, cfg, rng):
    """Uniform auxiliary queries; stratified encoder gradients for each causal prefix.

    Selected histories share the history budget. Their inclusion probability
    k/(length-1) makes this an unbiased estimator of the *direct* history loss
    for a fixed model/state, not of full-history encoder backpropagation.
    ``retain_until`` releases CPU image crops after their final selected use.
    """
    from .stratified_replay import stratified_indices
    if length < 1:
        raise ValueError('A stream needs at least one observation')
    count = min(cfg.feature_history_decisions, length-1) if cfg.feature_history_loss_fraction else 0
    queries = sorted(rng.choice(length-1, count, replace=False).tolist()) if count else []
    weights = np.zeros(length)
    if queries:
        weights[queries] = cfg.feature_history_loss_fraction/count
    weights[-1] = 1-cfg.feature_history_loss_fraction if queries else 1.
    indices = np.full((length, 3), -1, dtype=np.int64)
    retained = np.full(length, -1, dtype=np.int64)
    for t in queries+[length-1]:
        selected = sorted(stratified_indices(t+1, rng))
        indices[t, :len(selected)] = selected
        retained[selected] = t
    return weights, indices, retained


def sequence_steps(batch):
    return batch.get('feature_sequence', [batch])


def decision_count(batch):
    """Independent non-stream batches are already supervised decision crops."""
    return int(batch['decision_mask'].sum()) if 'stream_id' in batch else len(batch['hist'])
