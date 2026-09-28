"""Rebuild causal observation sequences from saved histories, without GT inputs."""
import numpy as np
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, frame_from_heading, normalize
from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed


def memory_crop(cfg):
    n = cfg.memory_patch_size
    return CropSpec(depth=n, width=n, behind=n//2, spacing=cfg.fine.spacing)


def memory_layout(item, cfg):
    """Oldest to newest observed positions; current head is always last.

    Patches have a deterministic tangent-aligned frame, independent of current
    crop roll. Past tangents use only observations available at that position.
    Online tracing supplies only the new head after memory initialization.
    """
    pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
    hist = np.asarray(item.get('hist_local', np.zeros((cfg.n_history, 3)))) @ frame.T+pos
    mask = np.asarray(item.get('hmask', np.zeros(cfg.n_history))).astype(bool)
    contiguous = np.logical_and.accumulate(mask)
    warm = bool(item.get('memory_warm', False))
    observations = []
    if not warm:
        for age in range(cfg.memory_steps*cfg.memory_stride, 0, -cfg.memory_stride):
            i = age-1
            if not contiguous[i] or not np.isfinite(hist[i]).all():
                continue
            if i+1 < len(hist) and mask[i+1]:
                heading = hist[i]-hist[i+1]
            else:
                heading = np.asarray(item.get('seed_tangent', frame[:, 2]))
            if not np.isfinite(heading).all() or np.linalg.norm(heading) < 1e-6:
                heading = frame[:, 2]
            observations.append(dict(pos=hist[i], frame=frame_from_heading(normalize(heading))))
    observations.append(dict(pos=pos, frame=frame_from_heading(frame[:, 2])))
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
    # Fixed training shape; online calls contain just a single new observation.
    count = 1 if all(i.get('memory_warm', False) for i in items) else cfg.memory_steps+1
    b, n = len(items), cfg.memory_patch_size
    patches = torch.zeros(b, count, 2, n, n, n)
    mask = torch.zeros(b, count, dtype=torch.bool)
    positions = torch.zeros(b, count, 3)
    frames = torch.eye(3).expand(b, count, -1, -1).clone()
    seed_patch = torch.zeros(b, 2, n, n, n)
    seed_valid = torch.zeros(b, dtype=torch.bool)
    seed_position = torch.zeros(b, 3)
    seed_frame = torch.eye(3).expand(b, -1, -1).clone()
    reads, destinations = [], []
    for j, (observations, seed) in enumerate(layouts):
        for k, obs in enumerate(observations, count-len(observations)):
            reads.append(obs); destinations.append((j, k))
            mask[j, k] = True
            positions[j, k] = torch.as_tensor(obs['pos'], dtype=torch.float32)
            frames[j, k] = torch.as_tensor(obs['frame'], dtype=torch.float32)
        if seed is not None:
            reads.append(seed); destinations.append((j, -1)); seed_valid[j] = True
            seed_position[j] = torch.as_tensor(seed['pos'], dtype=torch.float32)
            seed_frame[j] = torch.as_tensor(seed['frame'], dtype=torch.float32)
    # Bound temporary source blocks for a production microbatch of sequences.
    for start in range(0, len(reads), 32):
        images = image_crop(reads[start:start+32], vol, memory_crop(cfg), pool)
        for image, (j, k) in zip(images, destinations[start:start+32]):
            if k < 0:
                seed_patch[j] = image
            else:
                patches[j, k] = image
    return dict(memory_patches=patches, memory_mask=mask, memory_positions=positions,
                memory_frames=frames, memory_seed_patch=seed_patch, memory_seed_valid=seed_valid,
                memory_seed_position=seed_position, memory_seed_frame=seed_frame)
