"""Rebuild causal observation sequences, without GT inputs.

A state may carry ``memory_track``: the heads actually observed before it, one
per tracing decision (replay) or along a constructed wrong-fiber path. Its
labels are auxiliary targets only; the writer sees positions, frames and CT.
Without a track, observations are reconstructed from the saved history.
"""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.geometry import frame_from_heading, normalize
from vesuvius.neural_tracing.fiber_follow.shared.data import training_state_allowed

TRACK_KEYS = ('pos', 'frame', 'offtrack', 'offset')  # offset: world vector to the original fiber


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
                observations.append(dict(pos=point, frame=frame_from_heading(normalize(heading)), track=i))
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
    observations.append(dict(pos=pos, frame=frame))
    seed = None
    if not warm and item.get('seed_valid', False):
        point = np.asarray(item['seed_pos'])
        tangent = np.asarray(item['seed_tangent'])
        if np.isfinite(point).all() and np.isfinite(tangent).all() and np.linalg.norm(tangent) > 1e-6:
            seed = dict(pos=point, frame=frame_from_heading(normalize(tangent)))
    return observations, seed


def memory_allowed(item, cfg, band):
    if not training_state_allowed(item, cfg.fine, band):
        return False
    if item.get('seed_valid', False) and not item.get('memory_warm', False):
        from .feature_sequences import seed_observation
        return training_state_allowed(seed_observation(item, cfg), cfg.fine, band)
    return True
