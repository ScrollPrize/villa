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
