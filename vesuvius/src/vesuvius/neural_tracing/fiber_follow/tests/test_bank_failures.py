"""Continuous trusted-bank contacts, replay provenance, and geometric difficulty."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from test_neighbor_bank import make_bank, add_shard, publish
from test_neighbor_following import clean_sample
from vesuvius.neural_tracing.fiber_follow.regression.bank_geometry import (
    difficulty_scores, first_foreign_contact, tube_intervals, BankSwitchDetector,
)
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentitySampling
from vesuvius.neural_tracing.fiber_follow.shared.collect import DecisionCollector, append_traces, track_arrays
from vesuvius.neural_tracing.fiber_follow.shared.data import (
    OnPolicyStates, FollowDataset, fiber_manifest, replay_pools, TracedFiber,
)
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


def test_ambiguous_shared_tubes_missing_contact_and_capsule_ends():
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
    cfg = DirectConfig()
    return dict(pos=pos, frame=frame_from_heading(heading),
        hist=pos-np.arange(1, cfg.n_history+1)[:, None]*heading,
        hmask=np.ones(cfg.n_history), exploratory=False, would_stop=stop,
        travelled=travelled, last_segment=np.array([pos] if previous is None else [previous, pos]))


@pytest.mark.parametrize('reverse', [False, True])
def test_close_neighbor_switch_is_labeled_before_distance_departure_and_saved(tmp_path, reverse):
    bank, parent = make_bank(tmp_path/'bank')
    publish(bank.root, [add_shard(bank.root, 0, x=3., z_range=(20., 180.))])
    cfg = clean_sample(DirectConfig())
    start = 140. if reverse else 50.
    sign = -1 if reverse else 1
    collector = DecisionCollector(parent, 0, start, sign, cfg,
                                  bank_detector=BankSwitchDetector([bank]))
    a, b = np.array([0., 0., start]), np.array([3., 0., start+sign*8])
    assert collector(decision(a, 0., reverse=reverse))
    assert collector(decision(b, np.linalg.norm(b-a), a, reverse=reverse))
    before, after = collector.rows
    assert before['failure_kind'] == 4 and not before['offtrack']
    assert after['failure_kind'] == 1 and after['offtrack']
    np.testing.assert_array_equal(collector.track, np.stack((a,b)))
    assert [r['prefix_end'] for r in collector.rows] == [1,2]
    assert after['switch_decision'] == 0
    np.testing.assert_allclose(after['switch_pos'], [2.25, 0., start+sign*6])
    assert after['switch_distance'] == pytest.approx(np.linalg.norm(b-a)*.75)
    assert after['switch_bank_path'].endswith('shards/0000#0')
    assert after['switch_bank_run'] == bank.run['digest']
    rows, track = [], []
    append_traces([collector], rows, track)
    states = OnPolicyStates(manifest=fiber_manifest([parent]),
        **{k: np.asarray([row[k] for row in rows]) for k in
           OnPolicyStates.FIELDS+tuple(OnPolicyStates.OPTIONAL)+tuple(OnPolicyStates.ROW_TRACK)},
        **track_arrays(track))
    path = tmp_path/'replay.npz'
    states.save(path)
    loaded = OnPolicyStates.load(path)
    np.testing.assert_array_equal(loaded.failure_kind, [4, 1])
    np.testing.assert_array_equal(loaded.switch_pos, states.switch_pos)
    np.testing.assert_array_equal(loaded.switch_bank_path, states.switch_bank_path)
    np.testing.assert_array_equal(loaded.observed_prefix(1), np.stack((a,b)))
    np.testing.assert_array_equal(loaded.observed_prefix(0), a[None])


def test_premature_stopping_and_real_endpoint_overshoot_are_distinct(tmp_path):
    _, parent = make_bank(tmp_path)
    parent = replace(parent, endpoint_stop=(True, True))
    cfg = clean_sample(DirectConfig())
    collector = DecisionCollector(parent, 0, 150., 1, cfg)
    assert collector(decision([0, 0, 150], 0, stop=True))
    assert collector.rows[-1]['failure_kind'] == 2
    assert not collector.rows[-1]['offtrack']
    assert collector(decision([0, 0, 202], 52, [0, 0, 150]))
    assert collector.rows[-1]['failure_kind'] == 3 and collector.rows[-1]['offtrack']


def test_terminal_commit_keeps_event_without_inventing_decision(tmp_path):
    bank, parent = make_bank(tmp_path)
    publish(tmp_path, [add_shard(tmp_path, 0, x=3., z_range=(20.,180.))])
    collector = DecisionCollector(parent, 0, 50., 1, clean_sample(DirectConfig()),
                                  bank_detector=BankSwitchDetector([bank]))
    assert collector(decision([0,0,50], 0.))
    collector.observe_final_path(np.array([[0.,0.,50.], [3.,0.,58.]]))
    assert len(collector.rows) == 1 and len(collector.track) == 2
    assert collector.rows[0]['failure_kind'] == 4 and not collector.rows[0]['offtrack']
    np.testing.assert_allclose(collector.rows[0]['switch_pos'], [2.25,0,56])


def replay_fixture():
    # Unequal row counts should not overwhelm rarer failure types or fibers.
    kinds = np.r_[np.zeros(40, int), np.ones(100, int), [2, 3, 4]]
    n = len(kinds)
    return OnPolicyStates(manifest=[], failure_kind=kinds,
        fiber_idx=np.r_[np.zeros(90, int), np.ones(n-90, int)], t=np.zeros(n), reverse=np.zeros(n, bool),
        pos=np.zeros((n,3)), frame=np.tile(np.eye(3), (n,1,1)), hist=np.zeros((n,1,3)), hmask=np.ones((n,1)),
        offtrack=(kinds == 1) | (kinds == 3), hard=np.ones(n, bool), exploratory=np.zeros(n, bool),
        drift=np.tile([.5,1.25,1.75,2.5], (n+3)//4)[:n])


def test_failure_replay_balances_categories_then_fibers_with_drift_budget():
    op = replay_fixture()
    ds = FollowDataset([SimpleNamespace(length=100.)], None, None, None, fresh_fraction=0.,
                       batch_builder=SimpleNamespace(sampling=IdentitySampling(replay_failure_fraction=.5)))
    ds.recent_pools = replay_pools([op], failures=True)
    rng = np.random.default_rng(8)
    counts = np.zeros(9)
    switches = np.zeros(2)
    for _ in range(12000):
        _, band, cache, row = ds.draw_replay(rng)
        counts[band] += 1
        if band == 5:
            switches[cache.fiber_idx[row]] += 1
    assert counts[:4].sum()/counts.sum() == pytest.approx(.5, abs=.02)
    np.testing.assert_allclose(counts[5:]/counts.sum(), [.125]*4, atol=.015)
    assert switches[0]/switches.sum() == pytest.approx(.5, abs=.04)
    # Exhausted failure category falls back to drift, not fabricated negatives.
    ds.recent_pools[4:] = [{} for _ in range(5)]
    assert all(ds.draw_replay(rng)[1] < 4 for _ in range(100))


def test_legacy_cache_without_failure_fields_defaults_to_existing_labels(tmp_path):
    op = replay_fixture()
    op.failure_kind[:] = 0
    path = tmp_path/'legacy.npz'
    op.save(path)
    # Mimic old archives that predate all switch/failure metadata.
    with np.load(path) as archive:
        data = {k:archive[k] for k in archive.files if not k.startswith('switch_') and k not in ('failure_kind','travelled')}
    np.savez(path, **data)
    loaded = OnPolicyStates.load(path)
    assert not loaded.failure_kind.any() and (loaded.switch_decision == -1).all()
    assert np.isnan(loaded.switch_pos).all()
    assert replay_pools([loaded], failures=True)[4]
    np.testing.assert_array_equal(loaded.offtrack, op.offtrack)


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
