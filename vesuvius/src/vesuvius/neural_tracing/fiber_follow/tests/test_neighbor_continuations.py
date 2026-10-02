"""Certified synthetic failures: trace-noised original prefix, smooth bridge, short neighbor tail."""
import numpy as np
import pytest

from test_neighbor_bank import make_bank, add_shard, publish
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
from vesuvius.neural_tracing.fiber_follow.shared.data import SOURCE, SampleConfig
from vesuvius.neural_tracing.fiber_follow.shared.state_labels import REASON, TERMINAL


def configuration():
    model = DirectConfig(n_history=64,n_future=4)
    sample = SampleConfig(crop=model.fine,n_history=64,recent_history_points=64,n_future=4)
    return model,sample


def test_synthetic_failure_is_a_certified_terminal_switch_with_noisy_original_prefix(tmp_path):
    bank,fiber = make_bank(tmp_path,with_path=True)
    _,sample = configuration()
    state = wrong_continuation(bank,sample,np.random.default_rng(8),tail_length_range=(4.,12.))
    assert state is not None and state['source'] == SOURCE['synthetic']
    assert state['supervision'] == TERMINAL and state['supervision_reason'] == REASON['switch']
    assert state['terminal'] and not state['geometry_valid'] and state['confidence_valid']
    # The original prefix carries tracing error, zero at its annotated seed; the tail is exact.
    path = state['observed_path']
    np.testing.assert_allclose(path[0], state['seed_pos'])
    assert state['trace_noise_sigma'] > 0
    prefix = path[:int(np.searchsorted(state['_constructed_arc'], state['_leave_arc']))]
    lateral = np.abs(prefix[:, 0]-fiber.points[0, 0])
    assert lateral[0] == pytest.approx(0., abs=1e-9) and lateral.max() > 1e-3
    assert bank.clear_of_target(state['fiber_ref'][0], path[-1:]).all()


def test_synthetic_terminal_discovers_paths_published_after_empty_lookup(tmp_path):
    bank,fiber = make_bank(tmp_path)
    cfg,sample = configuration()
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank,sampling=IdentitySampling(synthetic_tail=(4.,12.)))
    rng = np.random.default_rng(8)
    assert builder.synthetic_terminal(sample,rng) is None
    publish(tmp_path,[add_shard(tmp_path,0)])
    state = builder.synthetic_terminal(sample,rng)
    assert state is not None and state['source'] == SOURCE['synthetic'] and bank.shard_count == 1


def test_repeated_prefix_start_has_a_nonzero_observed_seed_heading(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_follow.regression import neighbor_continuations as module
    bank,_=make_bank(tmp_path,with_path=True)
    _,sample=configuration()
    sample.trace_noise_sigma = (0., 0.)
    original=module.interp_at
    def repeated_start(points,arc,at):
        result=original(points,arc,at)
        if len(at)>2:
            result[1]=result[0]
        return result
    monkeypatch.setattr(module,'interp_at',repeated_start)
    state=wrong_continuation(bank,sample,np.random.default_rng(8),tail_length_range=(4.,12.))
    assert state is not None
    path=state['observed_path']
    np.testing.assert_array_equal(path[0],path[1])
    assert np.linalg.norm(state['seed_tangent'])==pytest.approx(1.)
    moving=np.flatnonzero(np.linalg.norm(path-path[0],axis=1)>1e-6)[0]
    assert np.dot(state['seed_tangent'],path[moving]-path[0])>0


def test_requested_tails_are_honored_and_short_paths_are_not_silently_substituted(tmp_path):
    bank,_ = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(20.,180.))])
    _,sample = configuration()
    rng = np.random.default_rng(5)
    states = [wrong_continuation(bank,sample,rng,tail_length_range=(4.,16.)) for _ in range(60)]
    lengths = [s['bank_tail_length'] for s in states]
    assert 4. <= min(lengths) < 6. and 14. < max(lengths) <= 16.
    assert {s['fiber_ref'][2] for s in states} == {False,True}
    short,_ = make_bank(tmp_path/'short',with_path=True)
    assert wrong_continuation(short,sample,rng,tail_length_range=(128.,128.)) is None
    # A distant switch gets a longer bridge without clipping the tail.
    bank,_=make_bank(tmp_path/'distant')
    publish(bank.root,[add_shard(bank.root,0,x=30.,z_range=(20.,180.))])
    cfg=DirectConfig()
    from sampling_fixtures import clean_sample
    state=wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(64.,64.))
    assert state is not None and state['bank_tail_length']==64.
    assert state['bank_transition_length'] >= 60.
    assert state['terminal'] and not state['geometry_valid']
    assert wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(128.,128.)) is None


def test_invalid_tail_ranges_fail_closed():
    for lengths in ((0.,128.),(128.,4.),(4.,float('nan')),(4.,float('inf')),(4.,)):
        with pytest.raises(ValueError,match='tail lengths'):
            IdentitySampling(synthetic_tail=lengths)
