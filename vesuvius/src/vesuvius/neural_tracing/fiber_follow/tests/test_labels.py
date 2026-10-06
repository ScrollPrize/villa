"""Training labels: departures follow the evaluation rule, and replayed states are supervised only beyond the tip."""
import json

import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_follow.data import data as D
from vesuvius.neural_tracing.fiber_follow.tracing.collection import DecisionCollector
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, frame_from_heading


def fiber(length=100, z=0, endpoints=(False, False)):
    p = np.c_[np.arange(length + 1), np.zeros(length + 1), np.full(length + 1, z)].astype(float)
    return D.TracedFiber('test.json', p, arclength(p), 'H', endpoint_stop=endpoints)


def sample_config():
    return D.SampleConfig(crop=CropSpec(depth=12, width=9, behind=2), n_history=4, n_future=4)


def test_replay_uses_committed_prefix_and_supervises_only_beyond_tip():
    from replay_fixtures import replay_states
    f, cfg = fiber(), sample_config()
    reverse, sign = True, -1
    prefix = np.c_[50+sign*np.arange(-4, 1), np.full(5, .2), np.zeros(5)]
    frame = frame_from_heading(np.array([sign, 0., 0.]))
    op = replay_states([f], [dict(t=50., reverse=reverse, pos=prefix[-1], frame=frame, hist=prefix[-2::-1],
                                  hmask=np.ones(4), match_distance=.2, window_distance=.2, travelled=4.,
                                  seed_pos=prefix[0], seed_valid=True, seq_start=0, seq_end=5)],
                       # The archive can contain later points, but the row must never expose them.
                       track=np.concatenate((prefix, [[99., 7., 0.]])))
    ds = D.FollowDataset([f], None, cfg, None)
    item = ds.replay_item(op, 0, np.random.default_rng(1))
    np.testing.assert_array_equal(item['observed_path'], prefix)
    np.testing.assert_array_equal(item['seed_pos'], prefix[0])
    assert item['seed_valid'] and item['source'] == D.SOURCE['replay'] and item['source_step'] == 1000
    np.testing.assert_allclose(item['hist_local'] @ frame.T+item['pos'], prefix[-2::-1])
    target = item['fut_local'] @ frame.T+item['pos']
    np.testing.assert_allclose(target[:, 0], 50+sign*cfg.future_s)
    np.testing.assert_allclose(target[:, 1:], 0.)
    assert item['fmask'].all() and item['geometry_valid'] and not item['terminal']
    assert item['labeler_state']['t'] == 50. and item['labeler_state']['last_travelled'] == 4.


def decision_at(x, travelled, previous=None, would_stop=False, y=0.):
    pos = np.array([x, y, 0.])
    segment = np.array([pos]) if previous is None else np.array([previous, pos])
    return dict(pos=pos, frame=frame_from_heading(np.array([1., 0, 0])), hist=np.zeros((4, 3)),
                hmask=np.zeros(4), would_stop=would_stop, n_commit=0 if would_stop else 4,
                points=np.zeros((4, 3)), confidence=np.ones(4), heading_start=0, travelled=travelled,
                seed_pos=segment[0], seed_tangent=np.array([1., 0, 0]), seed_age=travelled, seed_valid=False,
                last_segment=segment)


def test_departure_is_the_evaluation_rule_and_a_return_restores_supervision():
    # Three consecutive committed points beyond 3 voxels (score_trace tolerance and
    # patience), dated at the run's first point, including runs spanning commits.
    from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, RECOVERABLE, REPLAY_CLASS
    c = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert c(decision_at(50, 0))
    # Isolated off-track points are not departures.
    probe = DecisionCollector(fiber(length=300), 0, 50, 1, sample_config(), stride=16)
    assert probe(decision_at(50, 0))
    path = np.array([[50., 0, 0], [51, 3.5, 0], [52, 3.5, 0], [53, 0, 0], [54, 3.5, 0], [55, 0, 0]])
    d = decision_at(55, float(arclength(path)[-1]))
    d['last_segment'] = path
    assert probe(d) and probe.labeler.departure_distance is None
    assert all(np.isnan(r['departure_distance']) for r in probe.rows)
    first = np.array([[50., 0, 0], [51, 0, 0], [52, 0, 0], [53, 3.5, 0], [54, 3.5, 0]])
    d = decision_at(54, float(arclength(first)[-1]), y=3.5)
    d['last_segment'] = first
    assert c(d) and not c.rows[-1]['bad_run'] >= 3 and np.isnan(c.rows[-1]['departure_distance'])
    assert c.rows[-1]['supervision'] == RECOVERABLE  # displaced now, but not yet a departure event
    second = np.array([[54., 3.5, 0], [55, 3.5, 0], [56, 3.5, 0]])
    d = decision_at(56, d['travelled']+2., y=3.5)
    d['last_segment'] = second
    assert c(d) and c.rows[-1]['departure_distance'] == pytest.approx(arclength(first[:4])[-1])
    assert c.rows[-1]['supervision'] == RECOVERABLE and c.rows[-1]['geometry_valid']
    assert c.rows[0]['pre_excursion'] and c.rows[0]['hard']
    # Back on the fiber: following again; the historical departure is retained.
    back = np.array([[56., 3.5, 0], [60, 0, 0], [70, 0, 0]])
    d = decision_at(70, d['travelled']+float(arclength(back)[-1]))
    d['last_segment'] = back
    assert c(d) and c.rows[-1]['supervision'] == FOLLOWING and c.rows[-1]['geometry_valid']
    assert np.isfinite(c.rows[-1]['departure_distance']) and c.rows[-1]['event_id'] == -1
    classes = [r['replay_class'] for r in c.finish()]
    assert classes == [REPLAY_CLASS['pre_excursion'], REPLAY_CLASS['recoverable'], REPLAY_CLASS['recoverable'],
                       REPLAY_CLASS['ordinary']]
