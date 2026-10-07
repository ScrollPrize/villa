"""Tube scoring behaviour on a straight synthetic fiber (u = x is the sheet normal, v = y the strip width)."""
from types import SimpleNamespace

import numpy as np

from vesuvius.neural_tracing.fiber_follow.evaluation.tube_scoring import replay_stop, score_tube

FRAME = np.eye(3)  # columns (u, v, f)
SWAPPED = np.eye(3)[:, [1, 0, 2]]


def fiber(length=200.):
    z = np.arange(0., length+1e-9, 1.)
    return SimpleNamespace(points=np.c_[np.zeros_like(z), np.zeros_like(z), z], s=z, length=length,
                           endpoint_stop=(False, False))


def path(offset, length, t0=20.):
    """A trace from t0 along +z sampled every 0.5 voxels; offset(z) gives its (x, y) displacement."""
    z = np.arange(t0, t0+length+1e-9, .5)
    return np.c_[np.array([offset(v) for v in z]), z]


def score(trace, reason='confidence', frames=None, **kw):
    travelled = np.arange(0., np.linalg.norm(np.diff(trace, axis=0), axis=1).sum()+1e-9, 8.)
    frames = [FRAME]*len(travelled) if frames is None else frames(travelled)
    return score_tube(trace, fiber(), 20., 1., travelled, frames, reason, **kw)


def test_annotation_end_passed_off_centre_is_unscored_not_wrong():
    # In the tube (|u| 1.4, |v| 2.8) but 3.1 voxels from the axis at the end, then away: the length past the
    # end is unscored, never wrong, and no identity loss.
    off = lambda z: (1.4, 2.8) if z < 200 else (1.4+.5*(z-200), 2.8)
    s = score(path(off, 280.))
    assert s['end'] == 'end' and s['reached_end'] and not s['loss']
    assert s['wrong_length'] == 0. and s['unscored_length'] > 90.


def test_width_slide_stays_in_tube_but_normal_step_is_an_excursion():
    slide = score(path(lambda z: (0., 2.5) if 60 < z < 100 else (0., 0.), 120.))
    assert slide['excursions'] == 0 and slide['wrong_length'] == 0.
    step = score(path(lambda z: (2., 0.) if 60 < z < 70 else (0., 0.), 120.))
    assert step['excursions'] == 1 and not step['loss'] and 9. < step['wrong_length'] < 12.  # includes the 2-voxel jump in


def test_frame_freezes_at_last_in_tube_decision():
    # The tracer's frame rotates during the excursion; measured in it the normal step would look like a width
    # offset inside the tube. The frozen frame keeps the fiber's orientation.
    def frames(travelled):
        return [SWAPPED if 44 <= t < 80 else FRAME for t in travelled]
    s = score(path(lambda z: (2., 0.) if 60 < z < 100 else (0., 0.), 140.), frames=frames)
    assert s['excursions'] == 1


def test_losses_and_stop_classification():
    lost = score(path(lambda z: (max(0., .2*(z-60)), 0.), 100.))
    assert lost['loss'] and lost['loss_kind'] == 'unlabelled' and lost['end'] == 'lost'
    assert 40. < lost['loss_at'] < 50.
    labelled = score(path(lambda z: (max(0., .2*(z-60)), 0.), 100.), foreign=lambda p: np.full(len(p), .5))
    assert labelled['loss_kind'] == 'switch'
    on = path(lambda z: (0., 0.), 60.)
    assert score(on)['end'] == 'premature'
    assert score(on, reason='max_len')['end'] == 'censored'
    near_end = path(lambda z: (0., 0.), 172.)  # stopped 8 voxels short of the annotation end
    assert score(near_end)['end'] == 'end'
    replay = score(path(lambda z: (0., 0.), 120.), stop_at=40.)
    assert replay['end'] == 'premature' and 39. < replay['total_length'] < 41.
    assert replay_stop(np.array([0., 8., 16.]), np.array([.9, .6, .4]), .7, final_was_stop=True) == 8.
    assert replay_stop(np.array([0., 8., 16.]), np.array([.9, .8, .6]), .7, final_was_stop=True) is None
    assert replay_stop(np.array([0., 8., 16.]), np.array([.9, .8, .6]), .7, final_was_stop=False) == 16.
