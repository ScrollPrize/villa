"""Regression checks for centerline sampling and full history anchors."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.regression.supervision import identity_terms
from vesuvius.neural_tracing.fiber_follow.shared.components import (
    ComponentRule, sample_pairs,
)
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, crop_local_grid


def fixture(width=41):
    crop = CropSpec(depth=45, width=width, behind=8, spacing=.5)
    curve = np.c_[np.full(81, .13), np.full(81, -.17), np.linspace(-3.87, 12.13, 81)]
    foreign = curve+np.array([4.06,.04,0.])
    return crop,curve,foreign


def test_near_outer_slots_visit_distinct_paths_before_reusing_one():
    crop,curve,_ = fixture(width=129)
    lines = [curve+[x,0,0] for x in (4,6,8,10,16,18,20,22)]
    points = np.concatenate(lines)
    ids = np.repeat(np.arange(8),len(curve))
    result = sample_pairs(curve,crop,points,np.zeros(len(points)),np.random.default_rng(4),
        margin=0.,rule=ComponentRule(lateral_max=32.),path_ids=ids,near_fraction=.5,return_metadata=True)
    _,pm,_,nm,meta = result
    assert pm.all() and nm.all()
    assert (meta['negative_distance'][:,:4] <= 12).all()
    assert (meta['negative_distance'][:,4:] > 12).all()
    for row in meta['negative_path_ids']:
        assert len(np.unique(row)) == 8
    # A missing band stays masked, even if the other band has surplus points.
    _,_,_,nm,meta = sample_pairs(curve,crop,points[ids>=4],np.zeros((ids>=4).sum()),
        np.random.default_rng(4),margin=0.,rule=ComponentRule(lateral_max=32.),path_ids=ids[ids>=4],
        near_fraction=.5,return_metadata=True)
    assert not nm[:,:4].any() and nm[:,4:].all()
    assert (meta['negative_path_ids'][:,:4] == -1).all()


def test_seed_reference_fallback_supplies_gradients_without_doubling_loss():
    cfg = SimpleNamespace(n_history=2)
    history = torch.tensor([[[.8,.6],[.8,.6],[.6,.8],[.6,.8]]],requires_grad=True)
    query = torch.tensor([[[1.,0.],[0.,1.]]],requires_grad=True)
    output = dict(reference_embedding=history,reference_mask=torch.ones(1,4),query_embedding=query,query_support=torch.ones(1,2))
    batch = dict(reference_on_fiber=torch.tensor([[0.,0.,1.,1.]]),positive_mask=torch.ones(1,1),
                 negative_mask=torch.ones(1,1,1),identity_seed_fallback=torch.tensor([True]))
    terms = identity_terms(output,batch,cfg)
    assert terms['identity_count'] == 1 and terms['identity_anchor_source'].item() == 2
    terms['identity_per_state'].sum().backward()
    assert history.grad[:,:2].abs().sum() == 0 and history.grad[:,2:].abs().sum() > 0
    assert query.grad.abs().sum() > 0
    saved=output['reference_mask'].clone()
    output['reference_mask'][:,2:]=0
    assert identity_terms(output,batch,cfg)['identity_count'] == 0
    output['reference_mask']=saved
    batch['reference_on_fiber'].fill_(1)
    expected = identity_terms(output,batch,cfg)['identity_per_state']
    batch['identity_seed_fallback'].fill_(True)
    current = identity_terms(output,batch,cfg)
    assert current['identity_anchor_source'].item() == 1
    torch.testing.assert_close(current['identity_per_state'],expected)


def test_identity_source_and_distance_metrics_pool_across_microbatches():
    from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import identity_training_groups
    cfg = SimpleNamespace(n_history=2)
    output = dict(reference_embedding=torch.tensor([[[1.,0.]]*4]*3),reference_mask=torch.ones(3,4),
        query_embedding=torch.tensor([[[1.,0.],[0.,1.],[-1.,0.]]]*3),query_support=torch.ones(3,3))
    batch = dict(reference_on_fiber=torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.],[0.,0.,0.,0.]]),
        positive_mask=torch.ones(3,1),negative_mask=torch.ones(3,1,2),
        identity_seed_fallback=torch.ones(3,dtype=torch.bool),source=torch.tensor([0,3,0]),
        location_source=torch.tensor([5,0,0]),negative_path_ids=torch.tensor([[[2,3]]]*3),
        negative_distance=torch.tensor([[[6.,24.]]]*3),negative_near_distance=torch.full((3,),12.))
    def groups(out,labels):
        terms = identity_terms(out,labels,cfg)
        terms.update(geometry_per_state=torch.ones(len(labels['source'])),
                     confidence_per_state=torch.full((len(labels['source']),),.5))
        return identity_training_groups(out,labels,terms,cfg,.1)
    full = groups(output,batch)
    pooled = {}
    for start,end in ((0,1),(1,3)):
        part = groups({k:v[start:end] for k,v in output.items()},{k:v[start:end] for k,v in batch.items()})
        for name,row in part.items():
            target = pooled.setdefault(name,{})
            for key,value in row.items():
                target[key] = target.get(key,0)+value
    for name,row in full.items():
        assert row == pytest.approx(pooled[name])
    assert full['source/fresh']['eligible_states'] == 1
    assert full['source/wrong_continuation']['seed_states'] == 1
    assert full['distance/near']['distinct_paths'] == 3


@pytest.mark.parametrize('width', [40, 41])
def test_centerline_queries_are_not_displaced_and_keep_valid_support(width):
    crop,curve,foreign = fixture(width)
    kwargs = dict(positives=4, negatives=8, margin=2., along_margin=1.)
    result = sample_pairs(curve,crop,foreign,np.arange(len(foreign)),np.random.default_rng(7),**kwargs)
    repeated = sample_pairs(curve,crop,foreign,np.arange(len(foreign)),np.random.default_rng(7),**kwargs)
    for a,b in zip(result,repeated):
        np.testing.assert_array_equal(a,b)
    pos,pm,neg,nm = result
    assert pm.all() and nm.all()
    for p,points in zip(pos,neg):
        assert np.linalg.norm(curve-p,axis=1).min() < 1e-6
        assert (np.linalg.norm(foreign[:,None]-points[None],axis=-1).min(0) < 1e-6).all()
        assert np.max(np.abs(points[:,2]-p[2])) <= ComponentRule().along_window+1e-6
    bounds = crop_local_grid(crop)
    assert np.all(neg[...,2] >= bounds[...,2].min()+1.)
    assert np.all(neg[...,2] <= bounds[...,2].max()-1.)


def test_missing_paths_and_receptive_field_exclusions_stay_unknown():
    crop,curve,foreign = fixture()
    for points in (np.empty((0,3)),foreign+np.array([5.,0.,0.])):
        _,pm,_,nm = sample_pairs(curve,crop,points,np.empty(0,int),np.random.default_rng(7),margin=2.)
        assert pm.all() and not nm.any()


def test_full_target_history_scores_all_candidates_and_padding_has_no_effect():
    cfg = SimpleNamespace(n_history=4)
    history = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]], requires_grad=True)
    output = dict(reference_embedding=history, reference_mask=torch.ones(1, 4),
                  query_embedding=torch.tensor([[[0., 1.], [1., 0.], [-1., 0.]]]),
                  query_support=torch.ones(1, 3, dtype=torch.bool))
    batch = dict(reference_on_fiber=torch.ones(1, 4), positive_mask=torch.ones(1, 1),
                 negative_mask=torch.tensor([[[1., 0.]]]))
    full = identity_terms(output, batch, cfg)['identity_per_state']
    full.sum().backward()
    assert (history.grad.norm(dim=-1) > 0).all()  # includes all older positive-history patches
    output['query_embedding'][0, 2] = torch.tensor([1000., -1000.])
    torch.testing.assert_close(identity_terms(output, batch, cfg)['identity_per_state'], full)
    output['reference_mask'] = torch.tensor([[1., 1., 0., 0.]])
    short = identity_terms(output, batch, cfg)['identity_per_state']
    assert short.item() > full.item()+1.  # truncation really would weaken this positive; we do not do it
    batch['negative_mask'].zero_()
    terms = identity_terms(output, batch, cfg)
    assert terms['identity_count'] == 0 and terms['identity_per_state'].item() == 0
