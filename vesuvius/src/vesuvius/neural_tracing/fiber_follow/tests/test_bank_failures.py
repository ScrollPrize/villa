"""Continuous trusted-bank contacts, collector events and replay provenance, geometric difficulty."""
import numpy as np
import pytest

from test_neighbor_bank import make_bank, add_shard, publish
from sampling_fixtures import clean_sample
from vesuvius.neural_tracing.fiber_follow.data.bank_geometry import difficulty_scores, first_foreign_contact, tube_intervals, BankSwitchDetector
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionConfig
from vesuvius.neural_tracing.fiber_follow.tracing.collection import DecisionCollector, append_traces, collected_states
from vesuvius.neural_tracing.fiber_follow.data.data import OnPolicyStates, TracedFiber
from vesuvius.neural_tracing.fiber_follow.data.state_labels import FOLLOWING, REASON, REPLAY_CLASS, TERMINAL
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength, frame_from_heading


def vertical(x, lo=0., hi=200.):
    return np.array([[x, 0., lo], [x, 0., hi]])


def test_exact_entry_between_heads_and_subsample_brief_contact():
    segment = np.array([[0., 0., 50.], [3., 0., 58.]])
    distance, pos = first_foreign_contact(segment, vertical(3.), vertical(0.))
    np.testing.assert_allclose(pos, [2.25, 0., 56.], atol=1e-12)
    assert distance == pytest.approx(np.linalg.norm(segment[1]-segment[0])*.75)
    # A tiny capsule contact would be missed by regular 0.25-voxel samples.
    segment = np.array([[-1., .999999, 0.], [1., .999999, 0.]])
    far_target = vertical(30.)
    event = first_foreign_contact(segment, vertical(0., -1., 1.), far_target, 1., 1.)
    assert event is not None
    assert event[1][0] == pytest.approx(-np.sqrt(1-.999999**2))
    # Shared tubes are ambiguous, missing contact is None, capsule ends bound the interval.
    assert first_foreign_contact(vertical(0., 20, 30), vertical(0.), vertical(0.)) is None
    assert first_foreign_contact(vertical(2., 20, 30), vertical(5.), vertical(0.)) is None
    # Being close to both annotations does not certify the foreign identity.
    assert first_foreign_contact(vertical(1., 20, 30), vertical(1.5), vertical(0.)) is None
    np.testing.assert_allclose(tube_intervals(np.array([0., 0., -2.]), np.array([0., 0., 2.]),
                                            vertical(0., 0., 10.), 1.), [[.25, 1.]])
    assert first_foreign_contact(vertical(5., 14, 20), vertical(5., 0, 10), vertical(0.)) is None
    assert first_foreign_contact(np.array([[3., 0., 50.]]), vertical(3.), vertical(0.))[0] == 0


def test_bank_lookup_uses_relationship_arcs_not_mining_anchor(tmp_path):
    bank, _ = make_bank(tmp_path)
    shard = add_shard(tmp_path, 0, x=3., z_range=(20., 180.))
    shard['anchor_range'] = [5000., 5000.]
    publish(tmp_path, [shard])
    # Curved/winding annotations can map a mined neighbor far from its seed arc.
    assert len(bank.paths(0, 50.)) == 1
    event = BankSwitchDetector([bank]).first_contact(0, 50., np.array([[0.,0.,50.], [3.,0.,58.]]))
    assert event is not None
    np.testing.assert_allclose(event['pos'], [2.25,0.,56.])
    # Even a different winding's nearest arc cannot hide spatial contact.
    assert BankSwitchDetector([bank]).first_contact(0, 5000., np.array([[0.,0.,50.], [3.,0.,58.]])) is not None


def decision(pos, travelled, previous=None, *, stop=False, reverse=False):
    pos = np.asarray(pos, float)
    heading = np.array([0., 0., -1. if reverse else 1.])
    cfg = CoordinateRegressionConfig()
    return dict(pos=pos, frame=frame_from_heading(heading),
        hist=pos-np.arange(1, cfg.n_history+1)[:, None]*heading,
        hmask=np.ones(cfg.n_history), would_stop=stop, n_commit=0 if stop else 4,
        points=np.zeros((cfg.n_future, 3)), confidence=np.ones(cfg.n_future), heading_start=0,
        seed_pos=pos if previous is None else np.asarray(previous, float), seed_tangent=heading, seed_age=travelled,
        seed_valid=True, observed_path=np.array([pos] if previous is None else [previous, pos]),
        travelled=travelled, last_segment=np.array([pos] if previous is None else [previous, pos]))


def provenance():
    return dict(step=1000, cache_id='test', volume=dict(grid_scale=8.))


def test_close_neighbor_switch_is_terminal_before_distance_departure_and_saved(tmp_path):
    reverse = True
    bank, parent = make_bank(tmp_path/'bank')
    publish(bank.root, [add_shard(bank.root, 0, x=3., z_range=(20., 180.))])
    cfg = clean_sample(CoordinateRegressionConfig())
    start = 140. if reverse else 50.
    sign = -1 if reverse else 1
    collector = DecisionCollector(parent, 0, start, sign, cfg,
                                  bank_detector=BankSwitchDetector([bank]))
    a, b = np.array([0., 0., start]), np.array([3., 0., start+sign*8])
    first = decision(a, 0., reverse=reverse)
    second = decision(b, np.linalg.norm(b-a), a, reverse=reverse)
    second['seed_pos'] = a
    assert collector(first)
    assert collector(second)
    before, after = collector.rows
    assert before['supervision'] == FOLLOWING and before['pre_excursion'] and before['event_id'] == after['event_id']
    # Certified contact is terminal while the head is still within the departure distance.
    assert after['supervision'] == TERMINAL and after['supervision_reason'] == REASON['switch']
    assert after['match_distance'] <= 3. and after['switched']
    np.testing.assert_array_equal(collector.track, np.stack((a,b)))
    assert [r['prefix_end'] for r in collector.rows] == [1,2]
    np.testing.assert_allclose(after['switch_pos'], [2.25, 0., start+sign*6])
    assert after['switch_distance'] == pytest.approx(np.linalg.norm(b-a)*.75)
    assert after['switch_bank_path'].endswith('shards/0000#0')
    assert after['switch_bank_run'] == bank.run['digest']
    rows, track = [], []
    append_traces([collector], rows, track)
    assert [r['replay_class'] for r in rows] == [REPLAY_CLASS['pre_excursion'], REPLAY_CLASS['terminal']]
    states = collected_states(rows, track, [parent], provenance())
    path = tmp_path/'replay.npz'
    states.save(path)
    loaded = OnPolicyStates.load(path)
    np.testing.assert_array_equal(loaded.supervision, [FOLLOWING, TERMINAL])
    np.testing.assert_array_equal(loaded.switch_pos, states.switch_pos)
    np.testing.assert_array_equal(loaded.switch_bank_path, states.switch_bank_path)
    np.testing.assert_array_equal(loaded.observed_prefix(1), np.stack((a,b)))
    np.testing.assert_array_equal(loaded.observed_prefix(0), a[None])


def test_terminal_commit_keeps_event_without_inventing_decision(tmp_path):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=3., z_range=(20.,180.))])
    collector = DecisionCollector(parent, 0, 50., 1, clean_sample(CoordinateRegressionConfig()),
                                  bank_detector=BankSwitchDetector([bank]))
    assert collector(decision([0,0,50], 0.))
    collector.observe_final_path(np.array([[0.,0.,50.], [3.,0.,58.]]))
    assert len(collector.rows) == 1 and len(collector.track) == 2
    assert collector.rows[0]['supervision'] == FOLLOWING and collector.rows[0]['pre_excursion']
    assert collector.labeler.switch is not None
    np.testing.assert_allclose(collector.labeler.switch['switch_pos'], [2.25,0,56])


def test_difficulty_is_orientation_invariant_and_ranks_near_similar_curved_converging():
    points = vertical(0.)
    parent = TracedFiber('parent', points, arclength(points), 'H')
    near, far = vertical(3., 20., 180.), vertical(24., 20., 180.)
    assert difficulty_scores(near, parent)[0] > difficulty_scores(far, parent)[0]
    z = np.linspace(20, 180, 81)
    bend = np.c_[8+3*np.sin(z/12), np.zeros_like(z), z]
    scores = difficulty_scores(bend, parent)
    assert scores[1] > difficulty_scores(near, parent)[1]
    np.testing.assert_allclose(scores, difficulty_scores(bend[::-1], parent), atol=1e-12)
    converging = np.array([[20.,0.,20.], [3.,0.,180.]])
    assert difficulty_scores(converging, parent)[2] > difficulty_scores(near, parent)[2]


def test_hard_sampling_keeps_uniform_support_and_does_not_change_bank(tmp_path):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=3., z_range=(20,180)),
                       add_shard(tmp_path, 1, x=24., z_range=(20,180))])
    before = bank.provenance()
    uniform, hard = [], []
    a, b = np.random.default_rng(12), np.random.default_rng(12)
    for _ in range(240):
        uniform.append(bank.draw_path(a)[1][0,0])
        hard.append(bank.draw_path(b, hard_fraction=.5)[1][0,0])
    assert set(hard) == {3., 24.}
    assert np.mean(np.asarray(hard) == 3.) > np.mean(np.asarray(uniform) == 3.)
    assert bank.provenance() == before
