"""Positive bank supervision, symmetric distant queries and shared-bank roles."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_neighbor_bank import make_bank,add_shard,publish,item
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder,IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import DirectConfig,DirectFollower
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import following_sample
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
from vesuvius.neural_tracing.fiber_follow.shared.components import ComponentRule
from vesuvius.neural_tracing.fiber_follow.shared.data import FollowDataset,SampleConfig,continuation_targets,ZBand


def clean_sample(cfg):
    return SampleConfig(crop=cfg.fine,n_history=cfg.n_history,n_future=cfg.n_future,
        no_history_prob=0.,short_history_prob=0.,lateral_sigmas=(0.,),lateral_probs=(1.,),
        angle_sigmas_deg=(0.,),angle_probs=(1.,),history_jitter=0.,history_drift=0.,history_wobble=0.)


@pytest.mark.parametrize('seed',range(4))
def test_bank_following_uses_its_own_fiber_and_censors_cut_ends(tmp_path,seed):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=24.,z_range=(20.,180.))])
    cfg = DirectConfig()
    sample = clean_sample(cfg)
    state = following_sample(bank,sample,np.random.default_rng(seed))
    assert state['source'] == 4 and not state['offtrack'] and state['dense_mask'].all()
    assert state['fmask'].all() and not state['endpoint_known']
    fiber = state['supervision_fiber']
    assert fiber.endpoint_stop == (False,False)
    ds = object.__new__(FollowDataset)
    ds.fibers,ds.cfg,ds.exclude,ds.additional_crops = [parent],sample,bank.band,()
    ds.batch_builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank)
    state = ds.prepare(state,np.random.default_rng(30))
    world = state['identity_curve'] @ state['frame'].T+state['pos']
    np.testing.assert_allclose(world[:,0],24.,atol=1e-6)
    assert ds.state_allowed(state)
    assert not bank.clear_of_state(state,state['pos']).any()
    assert bank.clear_of_state(state,np.array([0.,0.,state['pos'][2]])).all()
    end = continuation_targets(fiber,fiber.length-1.,False,fiber.points[-2],np.eye(3),sample)
    assert not end['endpoint_known'] and not end['fmask'][1:].any()


def test_bank_following_probability_empty_bank_and_legacy_resume(tmp_path):
    bank,parent = make_bank(tmp_path)
    cfg = DirectConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,
        sampling=IdentitySampling(bank_following_probability=1.))
    sample = clean_sample(cfg)
    assert builder.replace_fresh(sample,np.random.default_rng(1)) is None
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(20.,180.))])
    builder.sampling = replace(builder.sampling,bank_following_probability=.1)
    draws=[builder.replace_fresh(sample,np.random.default_rng(seed)) for seed in range(200)]
    assert 8 <= sum(x is not None for x in draws) <= 35

def test_covered_primary_sampling_keeps_history_and_prepares_only_once(tmp_path):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(60.,180.))])
    cfg = DirectConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,augment=True,
        sampling=IdentitySampling(bank_coverage_probability=1.))
    rng = np.random.default_rng(17)
    state = builder.replace_fresh(clean_sample(cfg),rng)
    assert state['source'] == 0 and state['location_source'] == 5
    assert state['reference_on_fiber'][:cfg.n_history].sum() >= 2
    assert 'supervision_fiber' not in state
    seed = state['identity_seed']
    builder.prepare(state,parent,rng)
    assert state['identity_seed'] == seed and '_identity_prepared' not in state


def test_separate_roles_supply_both_bands_without_changing_following_source(tmp_path):
    bank,parent = make_bank(tmp_path/'outer')
    near,_ = make_bank(tmp_path/'near')
    publish(bank.root,[add_shard(bank.root,0,x=24.,z_range=(20.,180.))])
    publish(near.root,[add_shard(near.root,0,x=6.,z_range=(20.,180.))])
    cfg = DirectConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,near_negative_bank=near,
        following_bank=near,sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),
            negative_near_fraction=.5,bank_following_probability=1.))
    state = item(cfg)
    images={'fine':torch.ones(1,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width)}
    labels = builder.identity_targets([state],images)
    assert labels['negative_mask'].all()
    assert (labels['negative_distance'][...,:4] <= 12).all()
    assert (labels['negative_distance'][...,4:] > 12).all()
    state = builder.replace_fresh(clean_sample(cfg),np.random.default_rng(3))
    np.testing.assert_allclose(state['supervision_fiber'].points[:,0],6.)




def test_positive_bank_path_uses_annotated_parent_as_negative(tmp_path):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=24.,z_range=(20.,180.))])
    cfg = DirectConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,
        sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.)))
    state=following_sample(bank,clean_sample(cfg),np.random.default_rng(20))
    state=builder.prepare(state,state['supervision_fiber'],np.random.default_rng(21))
    images={'fine':torch.ones(1,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width)}
    labels=builder.identity_targets([state],images)
    assert labels['positive_mask'].all() and labels['negative_mask'].all()
    world=labels['identity_points'][0].numpy() @ state['frame'].T+state['pos']
    np.testing.assert_allclose(world[:4,0],24.,atol=1e-6)
    np.testing.assert_allclose(world[4:,0],0.,atol=1e-6)




def test_distant_switch_gets_longer_bridge_without_clipping_tail(tmp_path):
    bank,_=make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=30.,z_range=(20.,180.))])
    cfg=DirectConfig()
    state=wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(64.,64.))
    assert state is not None and state['bank_tail_length']==64.
    assert state['bank_transition_length'] >= 60.
    assert state['offtrack'] and not state['dense_mask'].any()
    assert wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(128.,128.)) is None
