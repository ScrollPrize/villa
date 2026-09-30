"""Regression checks for centerline sampling."""

import numpy as np
import pytest

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
