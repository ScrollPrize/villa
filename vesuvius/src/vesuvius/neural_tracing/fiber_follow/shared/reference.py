"""Observed original-fiber references, independent of annotation labels."""
import numpy as np

from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, normalize


SEED_FIELDS = ('seed_pos', 'seed_tangent', 'seed_age', 'seed_valid')
SEED_DEFAULTS = dict(seed_pos=lambda n: np.zeros((n, 3), np.float32),
                     seed_tangent=lambda n: np.zeros((n, 3), np.float32),
                     seed_age=lambda n: np.zeros(n, np.float32),
                     seed_valid=lambda n: np.zeros(n, bool))


def observed_seed(pos, frame, hist_local, hmask):
    """The start of a synthetic observed path, or its head when history is absent.

    This never looks up GT. Replay must preserve its actual reference rather
    than reconstructing one from a truncated (possibly wrong-fiber) history.
    """
    world = np.asarray(hist_local)[np.asarray(hmask) > 0][::-1] @ np.asarray(frame).T+pos
    path = np.concatenate((world, np.asarray(pos)[None]))
    tangent = path[min(2, len(path)-1)]-path[0]
    if np.linalg.norm(tangent) < 1e-6:
        tangent = np.asarray(frame)[:, 2]
    return dict(seed_pos=path[0].copy(), seed_tangent=normalize(tangent),
                seed_age=float(arclength(path)[-1]), seed_valid=True)


def observed_path(item):
    """Complete committed prefix, including head; never bridge a remote seed."""
    if 'observed_path' in item:
        path = np.asarray(item['observed_path'], dtype=np.float64)
    else:
        hist = np.asarray(item.get('hist_local', np.empty((0, 3))))
        mask = np.asarray(item.get('hmask', np.zeros(len(hist)))).astype(bool)
        path = np.concatenate((hist[mask][::-1] @ item['frame'].T+item['pos'],
                               np.asarray(item['pos'])[None]))
    if path.ndim != 2 or path.shape[1] != 3 or not len(path) or not np.isfinite(path).all():
        raise ValueError('History requires a finite, nonempty observed polyline')
    if not np.allclose(path[-1], item['pos'], atol=1e-5, rtol=0):
        raise ValueError('Observed prefix must end at the decision head')
    if item.get('seed_valid', False) and not np.allclose(path[0], item['seed_pos'], atol=1e-5, rtol=0):
        raise ValueError('Complete observed prefix required for a remote seed; recollect replay')
    # Consecutive stationary points have no arclength. Retain loops/wrong turns.
    return path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]]

