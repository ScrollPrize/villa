"""Rebuild causal observation sequences, without GT inputs.

A state may carry ``memory_track``: the heads actually observed before it, one
per tracing decision (replay) or along a constructed wrong-fiber path. Its
labels are auxiliary targets only; the writer sees positions, frames and CT.
Without a track, observations are reconstructed from the saved history.
"""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, frame_from_heading, normalize
from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed

TRACK_KEYS = ('pos', 'frame', 'offtrack', 'offset')  # offset: world vector to the original fiber


def memory_crop(cfg):
    if cfg.memory_version >= 3:
        return cfg.fine
    n = cfg.memory_patch_size
    return CropSpec(depth=n, width=n, behind=n//2, spacing=cfg.fine.spacing)


def memory_count(items, cfg):
    """Fixed training shape; online calls contain just a single new observation."""
    return 1 if all(i.get('memory_warm', False) for i in items) else cfg.memory_steps+1


def memory_layout(item, cfg):
    """Oldest to newest observed positions; current head is always last.

    Patches have a deterministic tangent-aligned frame, independent of current
    crop roll. Past tangents use only observations available at that position.
    Online tracing supplies only the new head after memory initialization.
    Track observations record their ``track`` index for auxiliary labels.
    """
    pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
    warm = bool(item.get('memory_warm', False))
    observations = []
    track = item.get('memory_track')
    if not warm and track is not None:
        n = len(track['pos'])
        for i in range(max(0, n-cfg.memory_steps), n):
            point, heading = np.asarray(track['pos'][i], np.float64), np.asarray(track['frame'][i], np.float64)[:, 2]
            if np.isfinite(point).all() and np.isfinite(heading).all() and np.linalg.norm(heading) > 1e-6:
                observation_frame = (np.asarray(track['frame'][i]) if cfg.memory_version >= 3
                                     else frame_from_heading(normalize(heading)))
                observations.append(dict(pos=point, frame=observation_frame, track=i))
    elif not warm:
        hist = np.asarray(item.get('hist_local', np.zeros((cfg.n_history, 3)))) @ frame.T+pos
        mask = np.asarray(item.get('hmask', np.zeros(cfg.n_history))).astype(bool)
        contiguous = np.logical_and.accumulate(mask)
        for age in range(cfg.memory_steps*cfg.memory_stride, 0, -cfg.memory_stride):
            i = age-1
            if i >= len(hist) or not contiguous[i] or not np.isfinite(hist[i]).all():
                continue
            if i+1 < len(hist) and mask[i+1]:
                heading = hist[i]-hist[i+1]
            else:
                heading = np.asarray(item.get('seed_tangent', frame[:, 2]))
            if not np.isfinite(heading).all() or np.linalg.norm(heading) < 1e-6:
                heading = frame[:, 2]
            observations.append(dict(pos=hist[i], frame=frame_from_heading(normalize(heading))))
    observations.append(dict(pos=pos, frame=frame if cfg.memory_version >= 3 else frame_from_heading(frame[:, 2])))
    seed = None
    if not warm and item.get('seed_valid', False):
        point = np.asarray(item['seed_pos'])
        tangent = np.asarray(item['seed_tangent'])
        if np.isfinite(point).all() and np.isfinite(tangent).all() and np.linalg.norm(tangent) > 1e-6:
            seed = dict(pos=point, frame=frame_from_heading(normalize(tangent)))
    return observations, seed


def memory_allowed(item, cfg, band):
    observations, seed = memory_layout(item, cfg)
    return all(training_state_allowed(o, memory_crop(cfg), band)
               for o in observations+([] if seed is None else [seed]))


def memory_images(items, vol, cfg, image_crop, pool=None):
    layouts = [memory_layout(i, cfg) for i in items]
    count = memory_count(items, cfg)
    b, n = len(items), cfg.memory_patch_size
    unified = cfg.memory_version >= 3
    shape = (cfg.fine.depth,cfg.fine.width,cfg.fine.width) if unified else (n,n,n)
    patches = torch.zeros(b, count-1 if unified else count, 2, *shape)
    mask = torch.zeros(b, count, dtype=torch.bool)
    positions = torch.zeros(b, count, 3)
    frames = torch.eye(3).expand(b, count, -1, -1).clone()
    seed_patch = torch.zeros(b, 2, *shape)
    seed_valid = torch.zeros(b, dtype=torch.bool)
    seed_position = torch.zeros(b, 3)
    seed_frame = torch.eye(3).expand(b, -1, -1).clone()
    reads, destinations = [], []
    for j, (observations, seed) in enumerate(layouts):
        for k, obs in enumerate(observations, count-len(observations)):
            if not unified or k < count-1:  # current crop is already in x['fine']
                reads.append(obs); destinations.append((j, k))
            mask[j, k] = True
            positions[j, k] = torch.from_numpy(np.array(obs['pos'], np.float32))  # copy: replay arrays are read-only mmaps
            frames[j, k] = torch.from_numpy(np.array(obs['frame'], np.float32))
        if seed is not None:
            reads.append(seed); destinations.append((j, -1)); seed_valid[j] = True
            seed_position[j] = torch.from_numpy(np.array(seed['pos'], np.float32))
            seed_frame[j] = torch.from_numpy(np.array(seed['frame'], np.float32))
    # Bound temporary source blocks for a production microbatch of sequences.
    read_batch = 2 if unified else 32
    for start in range(0, len(reads), read_batch):
        images = image_crop(reads[start:start+read_batch], vol, memory_crop(cfg), pool)
        for image, (j, k) in zip(images, destinations[start:start+read_batch]):
            if k < 0:
                seed_patch[j] = image
            else:
                patches[j, k] = image
    out = dict(memory_mask=mask, memory_positions=positions, memory_frames=frames,
               memory_seed_valid=seed_valid, memory_seed_position=seed_position, memory_seed_frame=seed_frame)
    out.update({'history_crops' if unified else 'memory_patches':patches,
                'seed_crop' if unified else 'memory_seed_patch':seed_patch})
    return out


def memory_targets(items, cfg):
    """Per-observation probe targets, aligned with ``memory_images``; never model inputs.

    Identity is 1 until the trace's confirmed departure from its original fiber.
    Offsets point from the observation to that fiber, in the observation's
    frame, known only while on it and within the recovery distance. Tracks
    label every write; otherwise only the head is labeled. Burn-in writes and
    states whose memory cannot distinguish the original fiber are unlabeled.
    """
    count = memory_count(items, cfg)
    b = len(items)
    identity = np.zeros((b, count), np.float32)
    identity_mask = np.zeros((b, count), bool)
    offset = np.zeros((b, count, 3), np.float32)
    offset_mask = np.zeros((b, count), bool)
    burn = max(0, count-cfg.memory_grad_steps-1)
    for j, item in enumerate(items):
        if not item.get('identity_observable', True):
            continue
        observations, _ = memory_layout(item, cfg)
        track = item.get('memory_track')
        for k, obs in enumerate(observations, count-len(observations)):
            if k < burn:
                continue
            if 'track' in obs:
                off, vector = float(track['offtrack'][obs['track']]), np.asarray(track['offset'][obs['track']], np.float64)
            elif k == count-1 and 'offtrack' in item:
                off = float(item['offtrack'])
                vector = np.full(3, np.nan)
                if not off and 'gt_history' in item and item['gt_history_mask'][0] > 0:
                    vector = np.asarray(item['frame']) @ np.asarray(item['gt_history'][0], np.float64)
            else:
                continue
            if not np.isfinite(off):
                continue
            identity[j, k], identity_mask[j, k] = 1.-off, True
            local = vector @ obs['frame']
            if not off and np.isfinite(local).all() and np.linalg.norm(local) <= cfg.max_recovery_distance:
                offset[j, k], offset_mask[j, k] = local, True
    return dict(memory_target_identity=torch.from_numpy(identity),
                memory_target_identity_mask=torch.from_numpy(identity_mask),
                memory_target_offset=torch.from_numpy(offset),
                memory_target_offset_mask=torch.from_numpy(offset_mask))
