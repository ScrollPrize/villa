"""Inputs for decision memory (``memory='decisions'``): earlier decisions of the same trace.

A state's earlier decisions come from ``item['memory_decisions']``: records with the
observed-path arclength (``travelled``), head ``pos`` and crop ``frame`` of each decision,
and a ``key`` under which its encoder entry was recorded (training chains, tracer). A
state without recorded decisions (fresh samples, a resumed or seeded prefix) gets
simulated decisions every ``DECISION_SPACING`` voxels behind its head. Selected entries
without a recorded key are encoded from their own CT crop (same crop specification,
references and frame rules as a decision there), so both kinds of entry are identical
model inputs. Only small geometry and, for unrecorded entries, CT crops leave here.
"""
import time

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, interp_at, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.shared.reference import observed_path
from vesuvius.neural_tracing.fiber_follow.models.decision_memory import (
    SLOTS, PATH_OFFSETS, GRID, DECISION_SPACING, select_decisions)

SAMPLING_REVISION = 'decision_memory_v1'


def simulated_decisions(path, arc):
    """Decision records every DECISION_SPACING voxels behind the head, plus the path start."""
    length = float(arc[-1])
    at = length-DECISION_SPACING*np.arange(1, int(np.floor(length/DECISION_SPACING))+1)
    at = np.unique(np.r_[0., at[at > 1e-6]]) if length > 1e-6 else np.zeros(0)
    return [dict(travelled=float(t), pos=interp_at(path, arc, np.array([t]))[0], frame=None, key=None) for t in at]


def memory_layout(item):
    """Selected earlier decisions of this state: geometry, record and whether to encode.

    An unrecorded entry gets the tracer's heading prior at that point (``trace_heading``:
    the 12-voxel fit of the committed path, else the seed heading); ``orient_entries``
    then resolves its crop frame with the tracer's frame rule.
    """
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import trace_heading
    path = observed_path(item)
    arc = arclength(path)
    length = float(arc[-1])
    records = item.get('memory_decisions')
    if records is None:
        records = item.setdefault('_simulated_decisions', simulated_decisions(path, arc))
    records = [r for r in records if r['travelled'] < length-1e-6]
    heading = item['seed_tangent'] if item.get('seed_valid', False) else np.asarray(item['frame'])[:, 2]
    entries = []
    for index in select_decisions([r['travelled'] for r in records], length):
        record = records[index]
        at = float(record['travelled'])
        pos = np.asarray(record['pos'], np.float64)
        entry = dict(record=record, travelled=at, age=length-at, pos=pos, seed=at <= 1e-6,
                     encode=record.get('key') is None,
                     observed_path=np.concatenate((path[arc < at], pos[None])),
                     fiber_family=item.get('fiber_family'))
        if record.get('frame') is not None:
            entry.update(frame=np.asarray(record['frame'], np.float64), frame_policy=item.get('frame_policy'))
        else:
            entry['prior'] = trace_heading(entry['observed_path'], 0, heading)
            entry['frame'] = frame_from_heading(entry['prior'])  # provisional, for read planning
        entries.append(entry)
    return entries, path, arc, length


def entry_path_samples(path, arc, entry):
    """Committed path at PATH_OFFSETS behind the entry's head, in the entry's frame."""
    at = entry['travelled']+np.asarray(PATH_OFFSETS)
    valid = at >= -1e-8
    clipped = np.clip(at, 0., arc[-1])
    points = (interp_at(path, arc, clipped)-entry['pos']) @ entry['frame']
    before = interp_at(path, arc, np.clip(clipped-.25, 0., arc[-1]))
    after = interp_at(path, arc, np.clip(clipped+.25, 0., arc[-1]))
    tangent = (after-before) @ entry['frame']
    norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    return points, tangent/np.maximum(norm, 1e-8), valid & (norm[:, 0] > 1e-8)


def entry_item(item, entry, cfg):
    """The observation a decision at this entry would have had (crop frame and references)."""
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import trace_history
    from vesuvius.neural_tracing.fiber_follow.data.observations import reference_layout
    world, mask = trace_history(list(entry['observed_path']), cfg.n_history)
    observation = dict(pos=entry['pos'], frame=entry['frame'], hist_local=(world-entry['pos']) @ entry['frame'], hmask=mask)
    if item.get('seed_valid', False):
        observation.update(seed_pos=item['seed_pos'], seed_tangent=item['seed_tangent'],
                           seed_age=max(0., float(item['seed_age'])-entry['age']), seed_valid=True)
    return reference_layout(observation, cfg)


def memory_bounds(item, cfg, vol, predictor):
    from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import crop_frame_bounds
    entries = memory_layout(item)[0]
    for entry in entries:
        if entry['encode']:
            yield from crop_frame_bounds(entry, cfg.fine, vol, predictor)


def memory_allowed(item, cfg, band):
    """Unrecorded entries read CT and must respect the held-out band like any state."""
    from vesuvius.neural_tracing.fiber_follow.data.data import training_state_allowed
    return all(training_state_allowed(dict(pos=e['pos'], frame=e['frame']), cfg.fine, band)
               for e in memory_layout(item)[0] if e['encode'])


def orient_entries(layouts, vol, cfg, pool=None):
    """Final crop frames for unrecorded entries, by the tracer's decision rule; written back to records.

    Like a tracer decision: the learned predictor resolves heading and normal from the
    ``trace_heading`` prior and the committed path, with roll anchored to the preceding
    decision (here the preceding entry of the same state; none for the first). Without a
    learned predictor, the CT frame rule with the same prior and anchor.
    """
    from vesuvius.neural_tracing.fiber_follow.tracing.crop_frames import frame_predictor, predict_frames
    from vesuvius.neural_tracing.fiber_follow.tracing.heading import ct_frame
    predictor = frame_predictor(cfg)
    pending = [[e for e in entries if e['encode'] and e['record'].get('frame') is None] for _, entries in layouts]
    previous = [None]*len(pending)
    # Entries of one state chain their roll anchor, so resolve them rank by rank across states.
    for rank in range(max((len(p) for p in pending), default=0)):
        batch = [(row, p[rank]) for row, p in enumerate(pending) if rank < len(p)]
        learned = [(row, e) for row, e in batch if predictor is not None and e.get('fiber_family')]
        if learned:
            frames = predict_frames(predictor, vol, [e['pos'] for _, e in learned], [e['prior'] for _, e in learned],
                [e['observed_path'] for _, e in learned], [e['fiber_family'] for _, e in learned],
                previous=[previous[row] for row, _ in learned], pool=pool)
            for (_, entry), frame in zip(learned, frames):
                entry['_learned_frame'] = np.asarray(frame, np.float64)
        for row, entry in batch:
            if '_learned_frame' in entry:
                frame = entry.pop('_learned_frame')
                entry['frame_source'] = 3
            else:
                diagnostics = {}
                frame = ct_frame(vol, entry['pos'], entry['prior'], previous[row], diagnostics=diagnostics)
                entry['frame_source'] = diagnostics.get('source', -1)
            entry['frame'] = entry['record']['frame'] = frame
            previous[row] = frame


def memory_inputs(items, vol, cfg, pool=None):
    """Decision-memory tensors; crops only for selected entries without a recorded key."""
    from vesuvius.neural_tracing.fiber_follow.data.observations import visible_points
    from vesuvius.neural_tracing.fiber_follow.data.crop_sampling import scalar_crops, empty_image_batch
    started = time.perf_counter()
    layouts = [memory_layout(item) for item in items]
    orient_entries([(item, layout[0]) for item, layout in zip(items, layouts)], vol, cfg, pool)
    n, paths = len(items), len(PATH_OFFSETS)
    valid = torch.zeros(n, SLOTS, dtype=torch.bool)
    encode = torch.zeros(n, SLOTS, dtype=torch.bool)
    keys = torch.full((n, SLOTS), -1, dtype=torch.int64)
    pose, ages, overlap = torch.zeros(n, SLOTS, 14), torch.zeros(n, SLOTS), torch.zeros(n, SLOTS)
    frame_source = torch.full((n, SLOTS), -1, dtype=torch.int64)
    points = torch.zeros(n, SLOTS, paths, 3)
    tangents = torch.zeros(n, SLOTS, paths, 3)
    path_valid = torch.zeros(n, SLOTS, paths, dtype=torch.bool)
    observations = []
    for row, (item, (entries, path, arc, _)) in enumerate(zip(items, layouts)):
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        for slot, entry in enumerate(entries):
            valid[row, slot] = True
            values = entry_path_samples(path, arc, entry)
            for tensor, value in zip((points, tangents, path_valid), values):
                tensor[row, slot] = torch.as_tensor(value)
            relative = (entry['pos']-pos) @ frame
            rotation = frame.T @ entry['frame']
            pose[row, slot] = torch.tensor(np.r_[relative/128., rotation.ravel(), np.log1p(entry['age'])/8., float(entry['seed'])])
            ages[row, slot] = entry['age']
            world = GRID @ entry['frame'].T+entry['pos']
            overlap[row, slot] = float(visible_points((world-pos) @ frame, cfg.fine).mean())
            if entry['encode']:
                encode[row, slot] = True
                frame_source[row, slot] = entry.get('frame_source', -1)
                observations.append(entry_item(item, entry, cfg))
            else:
                keys[row, slot] = int(entry['record']['key'])
        item['_memory_entries'] = entries
    shape = (len(observations), 1, cfg.fine.depth, cfg.fine.width, cfg.fine.width)
    if observations:
        crops = empty_image_batch(shape)
        scalar_crops(observations, vol, cfg.fine, pool, presence=False, out=crops.numpy())
    else:
        crops = torch.zeros(shape)
    references = torch.from_numpy(np.stack([o['reference_points'] for o in observations]).astype(np.float32)
                                  if observations else np.zeros((0, cfg.n_history+1, 3), np.float32))
    reference_mask = torch.from_numpy(np.stack([o['reference_mask'] for o in observations]).astype(np.float32)
                                      if observations else np.zeros((0, cfg.n_history+1), np.float32))
    elapsed = torch.full((n,), (time.perf_counter()-started)/max(1, n))
    chain = torch.tensor([int(item.get('live_chain_id', -1)) for item in items], dtype=torch.int64)
    return dict(history_valid=valid, history_encode=encode, history_keys=keys, history_chain=chain,
                history_pose=pose, history_ages=ages, history_overlap=overlap, history_load_seconds=elapsed,
                history_frame_source=frame_source, history_frame_energy=torch.zeros(n, SLOTS),
                history_frame_gap=torch.zeros(n, SLOTS), history_path_points=points, history_path_tangents=tangents,
                history_path_valid=path_valid, history_crops=crops, history_references=references,
                history_reference_mask=reference_mask)
