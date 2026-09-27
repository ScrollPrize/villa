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
    presence = np.ones((crop.depth,crop.width,crop.width),np.float32)
    return crop,curve,presence,foreign


@pytest.mark.parametrize('width', [40, 41])
def test_centerline_queries_are_not_displaced_and_keep_valid_support(width):
    crop,curve,presence,foreign = fixture(width)
    kwargs = dict(positives=4, negatives=8, margin=2., along_margin=1.)
    result = sample_pairs(curve,presence,crop,foreign,np.arange(len(foreign)),np.random.default_rng(7),**kwargs)
    repeated = sample_pairs(curve,presence,crop,foreign,np.arange(len(foreign)),np.random.default_rng(7),**kwargs)
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


def test_same_presence_threshold_applies_to_positive_and_negative_queries():
    crop,curve,presence,foreign = fixture()
    presence.fill(.69)
    _,pm,_,nm = sample_pairs(curve,presence,crop,foreign,np.arange(len(foreign)),np.random.default_rng(7))
    assert not pm.any() and not nm.any()


def test_missing_paths_and_receptive_field_exclusions_stay_unknown():
    crop,curve,presence,foreign = fixture()
    for points in (np.empty((0,3)),foreign+np.array([5.,0.,0.])):
        _,pm,_,nm = sample_pairs(curve,presence,crop,points,np.empty(0,int),np.random.default_rng(7),margin=2.)
        assert pm.all() and not nm.any()


def test_full_target_history_scores_all_candidates_and_padding_has_no_effect():
    cfg = SimpleNamespace(recent_patches=4)
    history = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]], requires_grad=True)
    output = dict(history_embedding=history, patch_mask=torch.ones(1, 4),
                  query_embedding=torch.tensor([[[0., 1.], [1., 0.], [-1., 0.]]]),
                  query_support=torch.ones(1, 3, dtype=torch.bool))
    batch = dict(patch_on_fiber=torch.ones(1, 4), positive_mask=torch.ones(1, 1),
                 negative_mask=torch.tensor([[[1., 0.]]]))
    full = identity_terms(output, batch, cfg)['identity_per_state']
    full.sum().backward()
    assert (history.grad.norm(dim=-1) > 0).all()  # includes all older positive-history patches
    output['query_embedding'][0, 2] = torch.tensor([1000., -1000.])
    torch.testing.assert_close(identity_terms(output, batch, cfg)['identity_per_state'], full)
    output['patch_mask'] = torch.tensor([[1., 1., 0., 0.]])
    short = identity_terms(output, batch, cfg)['identity_per_state']
    assert short.item() > full.item()+1.  # truncation really would weaken this positive; we do not do it
    batch['negative_mask'].zero_()
    terms = identity_terms(output, batch, cfg)
    assert terms['identity_count'] == 0 and terms['identity_per_state'].item() == 0
