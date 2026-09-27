"""Positive bank supervision, symmetric distant queries and shared-bank roles."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_neighbor_bank import make_bank,add_shard,publish,item
from vesuvius.neural_tracing.fiber_follow.regression.data import IdentityObservationBuilder,IdentitySampling
from vesuvius.neural_tracing.fiber_follow.regression.model import IdentityConfig,IdentityFollower
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_following import following_sample
from vesuvius.neural_tracing.fiber_follow.regression.neighbor_continuations import wrong_continuation
from vesuvius.neural_tracing.fiber_follow.regression.train import resolve_bank_option
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
    cfg = IdentityConfig()
    sample = clean_sample(cfg)
    state = following_sample(bank,sample,np.random.default_rng(seed))
    assert state['source'] == 4 and not state['offtrack'] and state['dense_mask'].all()
    assert state['fmask'].all() and not state['endpoint_known']
    fiber = state['supervision_fiber']
    assert fiber.endpoint_stop == (False,False)
    ds = object.__new__(FollowDataset)
    ds.fibers,ds.cfg,ds.exclude,ds.additional_crops = [parent],sample,bank.band,(cfg.coarse,)
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
    cfg = IdentityConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,
        sampling=IdentitySampling(bank_following_probability=1.))
    sample = clean_sample(cfg)
    assert builder.replace_fresh(sample,np.random.default_rng(1)) is None
    publish(tmp_path,[add_shard(tmp_path,0,z_range=(20.,180.))])
    builder.sampling = replace(builder.sampling,bank_following_probability=.1)
    draws=[builder.replace_fresh(sample,np.random.default_rng(seed)) for seed in range(200)]
    assert 8 <= sum(x is not None for x in draws) <= 35
    assert resolve_bank_option(None,None,'bank_following_probability',.1,0.) == .1
    saved={'training_options':{}}
    assert resolve_bank_option(None,saved,'bank_following_probability',.1,0.) == 0.
    with pytest.raises(ValueError,match='Resume option differs'):
        resolve_bank_option(.1,saved,'bank_following_probability',.1,0.)


@pytest.mark.parametrize('reverse',[False,True])
def test_outer_bank_supplies_identity_queries_outside_main_crop(tmp_path,reverse):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=31.5,z_range=(20.,180.))])
    cfg = IdentityConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,
        sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),query_patches=True))
    state = item(cfg,reverse)
    presence=np.ones((1,builder.pair_crop.depth,builder.pair_crop.width,builder.pair_crop.width),np.float32)
    images={'fine':torch.ones(1,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width)}
    labels=builder.identity_targets([state],images,presence)
    assert labels['positive_mask'].all() and labels['negative_mask'].all()
    assert not labels['foreign'].any()  # distant geometry is outside confidence's fine crop
    query=labels['identity_points'][0].numpy()
    world=query @ state['frame'].T+state['pos']
    np.testing.assert_allclose(world[:4,0],0.,atol=1e-6)
    np.testing.assert_allclose(world[4:,0],31.5,atol=1e-6)
    assert abs(query[4:,0]).min() > (cfg.fine.width-1)*cfg.fine.spacing/2


def test_positive_bank_path_uses_annotated_parent_as_negative(tmp_path):
    bank,parent = make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=24.,z_range=(20.,180.))])
    cfg = IdentityConfig()
    builder = IdentityObservationBuilder(cfg,[parent],negative_bank=bank,
        sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),query_patches=True))
    state=following_sample(bank,clean_sample(cfg),np.random.default_rng(20))
    state=builder.prepare(state,state['supervision_fiber'],np.random.default_rng(21))
    presence=np.ones((1,builder.pair_crop.depth,builder.pair_crop.width,builder.pair_crop.width),np.float32)
    images={'fine':torch.ones(1,2,cfg.fine.depth,cfg.fine.width,cfg.fine.width)}
    labels=builder.identity_targets([state],images,presence)
    assert labels['positive_mask'].all() and labels['negative_mask'].all()
    world=labels['identity_points'][0].numpy() @ state['frame'].T+state['pos']
    np.testing.assert_allclose(world[:4,0],24.,atol=1e-6)
    np.testing.assert_allclose(world[4:,0],0.,atol=1e-6)


def test_query_reads_are_identical_for_both_classes_and_guard_holdout(tmp_path,monkeypatch):
    import vesuvius.neural_tracing.fiber_follow.regression.data as data
    cfg=IdentityConfig()
    builder=IdentityObservationBuilder(cfg,sampling=IdentitySampling(rule=ComponentRule(lateral_max=32.),query_patches=True))
    state=builder.prepare(item(cfg),make_bank(tmp_path)[1],np.random.default_rng(0))
    points=torch.tensor([[[0.,0.,4.],[30.,0.,4.],[0.,0.,0.]]])
    valid=torch.tensor([[1.,1.,0.]])
    reads=[]
    def read(items,vol,crop,**kwargs):
        reads.extend(items)
        assert crop == cfg.patch_crop
        return torch.ones(len(items),1,crop.depth,crop.width,crop.width)
    monkeypatch.setattr(data,'scalar_crops',read)
    x=builder.query_inputs([state],None,points,valid)
    assert len(reads)==2
    for row,p in zip(reads,points[0,:2].numpy()):
        np.testing.assert_allclose(row['pos'],state['pos']+state['frame'] @ p)
        np.testing.assert_array_equal(row['frame'],state['frame'])
    assert x['identity_query_patches'][0,:2].all() and not x['identity_query_patches'][0,2].any()
    # Rotate lateral extent into z: query support reaches the holdout even
    # when the original head-centered crop does not.
    state['frame']=np.array([[0.,0.,1.],[0.,1.,0.],[1.,0.,0.]])
    assert not builder.footprint_allowed(state,ZBand(125.,130.))


def test_distant_switch_gets_longer_bridge_without_clipping_tail(tmp_path):
    bank,_=make_bank(tmp_path)
    publish(tmp_path,[add_shard(tmp_path,0,x=30.,z_range=(20.,180.))])
    cfg=IdentityConfig()
    state=wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(64.,64.))
    assert state is not None and state['bank_tail_length']==64.
    assert state['bank_transition_length'] >= 60.
    assert state['offtrack'] and not state['dense_mask'].any()
    assert wrong_continuation(bank,clean_sample(cfg),np.random.default_rng(4),tail_length_range=(128.,128.)) is None


def test_separate_query_embeddings_use_ct_and_backpropagate_outside_crop():
    from test_identity import config,batch,forward
    from vesuvius.neural_tracing.fiber_follow.regression.supervision import identity_terms
    torch.manual_seed(31)
    cfg=config()
    model=IdentityFollower(cfg).eval()
    data=batch(cfg,1)
    points=data['identity_points']
    points[...,0]=30.  # All queries are outside this small main crop.
    crop=cfg.patch_crop
    patches=torch.rand(1,points.shape[1],crop.depth,crop.width,crop.width,requires_grad=True)
    data['x'].update(identity_query_patches=patches,identity_query_mask=torch.ones(points.shape[:2]))
    output=forward(model,data)
    assert output['query_support'].all()
    direct=model.appearance(patches.reshape(-1,1,crop.depth,crop.width,crop.width),same=False)
    torch.testing.assert_close(output['query_embedding'],direct.reshape(1,points.shape[1],cfg.embedding))
    loss=identity_terms(output,data,cfg)['identity_per_state'].sum()
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert torch.isfinite(patches.grad).all() and patches.grad[:,:2].abs().sum() > 0
    assert patches.grad[:,2:].abs().sum() > 0
    # Query class/order and the main crop do not affect the CT embedding.
    permutation=torch.randperm(points.shape[1])
    data['identity_points']=points[:,permutation]
    data['x']['identity_query_patches']=patches.detach()[:,permutation]
    data['x']['fine'].zero_()
    with torch.no_grad():
        reordered=forward(model,data)
    torch.testing.assert_close(reordered['query_embedding'],output['query_embedding'][:,permutation])
