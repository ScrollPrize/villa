import numpy as np
import pytest

from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_reference_subset import (
    clipped_intervals, remap_endpoints, retained_links, reanchored_centers, audit_subset)
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import radial_initial_geometry


def test_exact_crossings_and_disconnected_reentry():
    xyz=np.array([[0,0,0],[0,0,10],[0,0,0],[0,0,10.]])
    np.testing.assert_allclose(clipped_intervals(xyz,[2,8]),[[2,8],[12,18],[22,28]])
    np.testing.assert_allclose(clipped_intervals(xyz,[0,8]),[[0,8],[12,28]])


def test_on_boundary_flat_and_tangency():
    np.testing.assert_allclose(clipped_intervals([[0,0,2],[4,0,2]],[2,8]),[[0,4]])
    assert clipped_intervals([[0,0,0],[0,0,2],[0,0,0]],[2,8]).shape==(0,2)


def test_remap_original_arc_negative_direction_and_no_bridge():
    d=dict(start=np.array([0.,10.]),end=np.array([10.,20.]),links=np.array([[0,1,-10,0.]]))
    direction=np.array([-1.,-1.]);parent=np.array([0,0,1]);starts=np.array([2.,8.,10.]);ends=np.array([4.,10.,12.])
    owner,ds=remap_endpoints(np.array([0,0,0,1]),np.array([-3.,-6.,-9.,-1.]),d,parent,starts,ends,direction)
    np.testing.assert_array_equal(owner,[0,-1,1,2]);np.testing.assert_allclose(ds[owner>=0],[-1,-1,-1])
    np.testing.assert_allclose(retained_links(d,parent,starts,ends,direction),[[1,2,-2,0]])
    assert retained_links(d,parent,starts,np.array([4.,9.,12.]),direction).shape==(0,4)


def test_continuation_roundoff_cannot_drop_exact_shared_cut():
    cut=10000.1;d=dict(start=np.array([.3,cut]),end=np.array([cut,cut+10]),links=np.array([[0,1,np.nextafter(cut-.3,np.inf),0]]))
    links=retained_links(d,np.array([0,1]),np.array([cut-1,cut]),np.array([cut,cut+1]),np.ones(2))
    np.testing.assert_array_equal(links,[[0,1,1,0]])


def test_audit_identity_degrees_and_clipped_samples():
    d=dict(source_id=np.array([3]),source_index=np.array([0]),family=np.array([0]),parent_piece=np.array([0]),
           start=np.array([2.]),end=np.array([8.]),lengths=np.array([6.]),node_ds=np.array([[0.,3.,6.]]),
           i=np.array([0]),j=np.array([0]),ds_a=np.array([1.]),ds_b=np.array([5.]),degree=np.array([2.]),links=np.empty((0,4)))
    parent=dict(d,start=np.array([0.]),end=np.array([10.]))
    p=dict(source_id=d['source_id'],start=d['start'],end=d['end'],owner=np.array([0,0]),arc=np.array([2.,8.]),
           offset=np.array([0,2]),xyz=np.array([[0,0,2.],[0,0,8.]]),weight=np.array([3.,3.]),chosen_winding=np.array([10.]))
    assert audit_subset(d,parent,p,[2,8],[10,19])['validated']
    assert audit_subset(d,parent,dict(p,chosen_winding=np.array([np.nan])),[2,8],None)['validated']
    with pytest.raises(ValueError,match='reference selection'):
        audit_subset(d,parent,dict(p,chosen_winding=np.array([np.nan])),[2,8],[10,19])
    with pytest.raises(ValueError,match='degree'):audit_subset(dict(d,degree=np.array([1.])),parent,p,[2,8],[10,19])
    with pytest.raises(ValueError,match='identity'):audit_subset(d,parent,dict(p,start=np.array([1.])),[2,8],[10,19])


def test_full_corpus_radial_prior_not_stretched():
    original=dict(radius=np.arange(1000.)+100)
    cfg=dict(winding_count=151)
    centers,widths=radial_initial_geometry(original,cfg)
    selected=np.arange(10,20)
    subset=dict(radius=original['radius'][selected],radial_initial_centers=centers[selected],radial_initial_widths=widths)
    actual=radial_initial_geometry(subset,cfg)
    np.testing.assert_array_equal(actual[0],centers[selected]);np.testing.assert_array_equal(actual[1],widths)
    with pytest.raises(ValueError,match='count mismatch'):radial_initial_geometry(subset,dict(winding_count=100))


def test_reanchor_preserves_physical_u_across_phase_wrap():
    d=dict(start=np.array([10.,10.]),phase=np.array([.9,.4]),family=np.array([0,1]))
    widths=np.array([100.,200.,300.]);centers=np.array([1.,1.])
    actual=reanchored_centers(d,np.array([0,1]),np.array([50.,50.]),np.array([.1,.5]),np.array([1.,-1.]),centers,widths)
    # H prior U=280, shifted to320 =>q=2+20/300, minusphase. V U unchanged.
    np.testing.assert_allclose(actual,[2+20/300-.1,.9],rtol=1e-6)
