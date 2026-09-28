"""Smooth bank departures and selective replacement of DAgger departures."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_neighbor_bank import make_bank, add_shard, publish
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder, IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset, SampleConfig, collate_targets
from vesuvius.neural_tracing.fiber_follow.shared.labels import prefix_labels


def configuration():
    model = DirectConfig(n_history=64,n_future=4)
    sample = SampleConfig(crop=model.fine,n_history=64,recent_history_points=64,n_future=4)
    return model,sample


@pytest.mark.parametrize('seed',range(6))
def test_smooth_bridge_retains_own_history_and_rejects_wrong_tail(tmp_path,seed):
    bank,fiber = make_bank(tmp_path,with_path=True)
    cfg,sample = configuration()
    rng = np.random.default_rng(seed)
    state = wrong_continuation(bank,sample,rng,tail_length_range=(4.,12.))
    assert state is not None and state['offtrack'] == 1 and state['source'] == 3
    np.testing.assert_allclose(state['pos'][:2],[6.,0.],atol=1e-8)
    assert bank.clear_of_target(0,state['pos']).all()
    assert not state['fmask'].any() and not state['dense_mask'].any()
    history = state['hist_local'] @ state['frame'].T+state['pos']
    valid = history[state['hmask']>0]
    assert (abs(valid[:,0])<1e-8).sum() >= 8
    assert ((valid[:,0]>0) & (valid[:,0]<6)).any()
    np.testing.assert_allclose(valid[0,:2],[6.,0.],atol=1e-8)
    steps = np.diff(np.concatenate((state['pos'][None],valid)),axis=0)
    assert np.linalg.norm(steps,axis=1).max() <= 1.000001
    assert np.linalg.norm(steps,axis=1).min() > .98
    directions = steps/np.linalg.norm(steps,axis=1)[:,None]
    assert (directions[1:]*directions[:-1]).sum(1).min() > .97
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank)
    builder.prepare(state,fiber,rng)
    on=state['reference_on_fiber']
    assert on[:cfg.n_history].sum() >= 1
    assert state['identity_observable'] == bool(on[:-1].sum() >= 2 or on[-1])
    assert not on[:-1][state['reference_mask'][:-1]==0].any()
    assert on[-1]  # immutable seed remains a reference outside the current crop
    assert state['reference_on_fiber'][0] == 0
    points = torch.zeros(1,4,3)
    points[0,:,2] = torch.arange(1,5)
    target,known,_ = prefix_labels(points,collate_targets([state]))
    assert not target.any() and known.all()


def test_only_recent_departure_slots_are_replaced_and_missing_bank_abstains(tmp_path):
    bank,fiber = make_bank(tmp_path,with_path=True)
    cfg,sample = configuration()
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank,
        sampling=IdentitySampling(bank_wrong_continuation_probability=1.,bank_wrong_continuation_tail=(4.,12.)))
    rng = np.random.default_rng(8)
    for source in (0,1,2):
        for stratum in range(5):
            if (source,stratum) == (2,4):
                assert builder.replace_replay(source,stratum,sample,rng)['source'] == 3
            else:
                assert builder.replace_replay(source,stratum,sample,rng) is None
    empty,_ = make_bank(tmp_path/'empty')
    builder.negative_bank = empty
    assert builder.replace_replay(2,4,sample,rng) is None


def test_wrong_continuations_discover_paths_published_after_empty_lookup(tmp_path):
    bank,fiber = make_bank(tmp_path)
    cfg,sample = configuration()
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank,
        sampling=IdentitySampling(bank_wrong_continuation_probability=1.,bank_wrong_continuation_tail=(4.,12.)))
    rng = np.random.default_rng(8)
    assert builder.replace_replay(2,4,sample,rng) is None
    publish(tmp_path,[add_shard(tmp_path,0)])
    state = builder.replace_replay(2,4,sample,rng)
    assert state is not None and state['source'] == 3 and bank.shard_count == 1


def test_unsafe_synthetic_departure_falls_back_to_original_replay(tmp_path):
    bank,fiber = make_bank(tmp_path,with_path=True)
    cfg,sample = configuration()
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank,
        sampling=IdentitySampling(bank_wrong_continuation_probability=1.,bank_wrong_continuation_tail=(4.,12.)))
    ds = object.__new__(FollowDataset)
    ds.fibers,ds.cfg,ds.batch_builder = [fiber],sample,builder
    ds.state_allowed = lambda state: state['source'] != 3
    replay = SimpleNamespace(fiber_idx=[0],t=[100.],reverse=[False],offtrack=[True],
        pos=np.array([[7.,0.,100.]]),frame=np.eye(3)[None],
        hist=np.c_[np.full(64,7.),np.zeros(64),100.-np.arange(1,65)][None],
        hmask=np.ones((1,64)),provenance={'step':1000})
    state = ds.replay_item((2,4,replay,0),np.random.default_rng(8))
    assert state['source'] == 2 and state['source_step'] == 1000
    np.testing.assert_array_equal(state['pos'],replay.pos[0])


@pytest.mark.parametrize('seed',range(6))
def test_long_departure_uses_remote_seed_for_supervision(tmp_path,seed):
    bank,fiber = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(20.,180.))])
    cfg = DirectConfig(n_future=4)
    sample = SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=4)
    builder = IdentityObservationBuilder(cfg,[fiber],negative_bank=bank,
        sampling=IdentitySampling(bank_wrong_continuation_probability=1.,
                                  bank_wrong_continuation_tail=(128.,128.)))
    state = builder.replace_replay(2,4,sample,np.random.default_rng(seed))
    assert state is not None and state['bank_tail_length'] == 128.
    history = state['hist_local'] @ state['frame'].T+state['pos']
    np.testing.assert_allclose(history[:,0],6.,atol=1e-5)
    assert state['hmask'].all() and not state['dense_mask'].any()
    builder.prepare(state,fiber,np.random.default_rng(seed))
    assert not state['reference_on_fiber'][:cfg.n_history].any()
    assert state['reference_on_fiber'][-1]
    assert state['identity_observable']
    assert not state['visible_seed_mask'].any()
    state['seed_valid'] = False
    builder.prepare(state,fiber,np.random.default_rng(seed))
    assert not state['identity_observable']


def test_variable_tails_reach_128_and_short_paths_are_not_silently_substituted(tmp_path):
    bank,_ = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(20.,180.))])
    _,sample = configuration()
    rng = np.random.default_rng(5)
    states = [wrong_continuation(bank,sample,rng) for _ in range(100)]
    lengths = [s['bank_tail_length'] for s in states]
    assert min(lengths) < 12. and 120. < max(lengths) <= 128.
    assert {s['fiber_ref'][2] for s in states} == {False,True}
    short,_ = make_bank(tmp_path/'short',with_path=True)
    assert wrong_continuation(short,sample,rng,tail_length_range=(128.,128.)) is None




def test_length_filter_discovers_long_paths_among_short_shards(tmp_path):
    bank,_ = make_bank(tmp_path)
    shards = [add_shard(tmp_path,i,z_range=(20.,60.)) for i in range(20)]
    shards.append(add_shard(tmp_path,20,z_range=(20.,180.)))
    for shard in shards:
        shard.update(path_lengths=[160. if shard['begin']==20 else 40.],training_indices=[0],
                     draw_indices=[0],draw_candidates=1)
    publish(tmp_path,shards)
    _,sample = configuration()
    for seed in range(10):
        state = wrong_continuation(bank,sample,np.random.default_rng(seed),
                                   tail_length_range=(128.,128.),prefer_long=True)
        assert state is not None and state['bank_tail_length'] == 128.


@pytest.mark.parametrize('lengths',[(0.,128.),(128.,4.),(4.,float('nan')),(4.,float('inf')),(4.,)])
def test_invalid_tail_ranges_fail_closed(lengths):
    with pytest.raises(ValueError,match='tail lengths'):
        IdentitySampling(bank_wrong_continuation_tail=lengths)
