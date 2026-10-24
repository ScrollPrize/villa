import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_anchored_dataset import curve_samples,one_hop,target_spans,anchored_mask
from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import pair_cost


def test_filter_keeps_saved_event_metadata_aligned():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_soft_winding import constraint_family_filter
    d=dict(family=np.array([0,1]),i=np.array([0,0]),j=np.array([1,1]),
        ds_a=np.zeros(2),ds_b=np.zeros(2),sign=np.ones(2),distance=np.ones(2),parallel=np.array([False,True]),
        event_ids=np.array([42,43]),phase_a=np.array([1.,2.]),phase_b=np.array([3.,4.]),phase_delta=np.array([5.,6.]))
    result,_=constraint_family_filter(d,True)
    for k in ('event_ids','phase_a','phase_b','phase_delta'):
        np.testing.assert_equal(result[k],d[k][:1])


def test_perpendicular_support_filter_remaps_all_edges_and_preserves_prior(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_anchored_dataset as a
    d={k:np.arange(4) for k in ('source_index','source_id','initial_k','start','end','phase','z','radius','lengths')}
    d.update(family=np.array([0,0,1,0]),node_ds=np.zeros((4,3)),node_phase=np.zeros((4,3)),
             node_z=np.zeros((4,3)),ids=np.arange(4),earlier_labels=np.arange(4),width_initial=np.ones(3),
             i=np.array([1,0,1]),j=np.array([2,1,3]),parallel=np.array([False,True,True]),links=np.empty((0,4)))
    for k in ('ds_a','ds_b','phase_a','phase_b','phase_delta','distance','sign','event_ids'):d[k]=np.arange(3)
    monkeypatch.setattr(a.uv,'radial_initial_geometry',lambda d,c:(np.arange(4)+.25,np.ones(3)))
    out,stats=a.perpendicular_supported_graph(d,{})
    np.testing.assert_array_equal(out['source_id'],[1,2])
    np.testing.assert_array_equal(out['i'],[0]);np.testing.assert_array_equal(out['j'],[1])
    np.testing.assert_array_equal(out['event_ids'],[0])
    np.testing.assert_array_equal(out['radial_initial_centers'],[1.25,2.25])
    assert stats['removed_source_ids']==[0,3] and stats['removed_constraints']==2
    assert stats['removed_perpendicular_constraints']==0
    assert a.perpendicular_supported_graph(out,{})[0] is out


def test_sampling_retains_vertices_and_bounds_gaps():
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[200.,0.,0.]])
    s,p=curve_samples(xyz,64)
    assert 1 in s and np.diff(s).max()<=64
    np.testing.assert_equal(p[:,0],s)


def test_one_hop_does_not_recurse_and_keeps_whole_sources():
    d=dict(source_id=np.array([10,10,20,30]),i=np.array([1,2]),j=np.array([2,3]))
    np.testing.assert_equal(one_hop(d,[10]),[10,20])


def test_segment_intersections_not_only_anchor():
    spans=target_spans(np.array([0.,100.]),np.array([[0,0,0],[0,0,100]]),
        np.array([10.,20.]),[25,75],[14,16])
    np.testing.assert_allclose(spans,[[35.,65.]])


def test_unanchored_component_is_uncertain():
    d=dict(family=np.zeros(4),i=np.array([0,2]),j=np.array([1,3]),links=np.empty((0,4)))
    np.testing.assert_equal(anchored_mask(d,[0]),[True,True,False,False])


def test_disabled_bridge_cannot_anchor_retained_island():
    d=dict(family=np.zeros(3),i=np.array([0,1]),j=np.array([1,2]),links=np.empty((0,4)))
    np.testing.assert_equal(anchored_mask(d,[0],[True,False,True]),[True,False,False])


def test_fixed_reference_delta_and_width_independence():
    a=torch.tensor([[3.,.2,100.,4.]],requires_grad=True)
    b=torch.tensor([[3.,.2,100.,100.]])
    fa=torch.tensor([0]);fb=torch.tensor([1]);zero=torch.zeros(1)
    cfg=dict(winding_sigma=.5,distance_sigma=.25,perpendicular_offset=.5)
    values=[]
    for width in (100.,1000.):
        w=torch.full((10,),width)
        values.append(pair_cost(a,b,fa,fb,zero,zero,w,torch.ones(1),torch.tensor([.5]),torch.tensor([False]),cfg,fixed_b=True))
    for x,y in zip(*values):torch.testing.assert_close(x,y)
    sum(x.sum() for x in values[0]).backward()
    assert a.grad is not None and b.grad is None
    # Enabled probability of GT is not involved in pair costs, including a
    # deliberately negative sentinel: it cannot suppress reference supervision.
    b[:,3]=-100
    other=pair_cost(a,b,fa,fb,zero,zero,torch.full((10,),100.),torch.ones(1),torch.tensor([.5]),torch.tensor([False]),cfg,fixed_b=True)
    for x,y in zip(values[0],other):torch.testing.assert_close(x,y)
