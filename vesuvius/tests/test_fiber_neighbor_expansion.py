import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_neighbor_expansion import z_spans, interval_overlap
from vesuvius.neural_tracing.fiber_trace_3d.fiber_scaled_lbfgs import scaled_joint


def test_z_spans_preserve_multiple_passes_and_boundary_interpolation():
    xyz=np.array([[0.,0.,0.],[0.,0.,20.],[0.,0.,0.]])
    assert z_spans(xyz,[5.,10.])==[[5.,10.],[30.,35.]]
    assert z_spans(np.array([[0.,0.,7.],[10.,0.,7.]]),[5.,10.])==[[0.,10.]]
    assert not z_spans(xyz,[30.,40.])


def test_winding_overlap_is_not_rounded_or_padded():
    assert interval_overlap([2.1,2.5],[2.5,3.8])
    assert not interval_overlap([2.1,2.49],[2.5,3.8])


def test_new_neighbor_initialization_uses_fractional_width_map(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_neighbor_expansion as m
    d=dict(source_id=np.array([10,20]),family=np.array([0,1]),phase=np.array([.2,.2]),
           z=np.zeros(2),start=np.zeros(2),node_ds=np.ones((2,2)),
           i=np.array([0]),j=np.array([1]),ds_a=np.array([100.]),ds_b=np.array([50.]),
           sign=np.ones(1),distance=np.array([.5]),parallel=np.array([False]))
    old_d=dict(source_id=np.array([10]))
    sol=dict(state=np.array([[2.,.2,300.,100.]]),widths=np.full(8,1000.),
             straight_anchor_u=np.array([.2]),straight_anchor_v=np.array([300.]))
    monkeypatch.setattr(m,'map_geometry',lambda *args:(None,lambda owner,arc:(np.full(len(owner),2.3),np.full(len(owner),300.))))
    state=m.warm_state(d,old_d,sol,{})
    np.testing.assert_allclose(state[0],sol['state'][0])
    np.testing.assert_allclose(state[1,:3],[2.1,.2,250.])


def test_gt_retained_even_below_confidence_or_outside_range(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_neighbor_expansion as m
    d=dict(source_id=np.array([10,20,30]),family=np.array([0,1,1]),
           start=np.zeros(3),end=np.full(3,10.),i=np.array([0,0]),j=np.array([1,2]),links=np.empty((0,4)))
    curves={s:({},np.array([[0.,0.,0.],[10.,0.,0.]])) for s in d['source_id']}
    def query(owner,arc):return np.array([2.,5.,2.])[owner],np.zeros(len(owner))
    monkeypatch.setattr(m,'map_geometry',lambda *args:(np.array([0.,0.,.9]),query))
    selected,audit=m.select(d,{}, {},[10,20],curves,[-1,1],.5)
    assert selected=={30}
    assert all(row['selected'] for row in audit['records'])


def test_conditional_fit_freezes_old_anchors_and_widths():
    class Curve(torch.nn.Module):
        straight=True
        def __init__(self):
            super().__init__();self.owner=torch.tensor([0,1]);self.u=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64));self.v=torch.nn.Parameter(torch.zeros(2,dtype=torch.float64))
    class Ext(torch.nn.Module):
        def __init__(self):super().__init__();self.curve=Curve()
    ext=Ext();state=torch.zeros(2,4,dtype=torch.float64,requires_grad=True)
    raw=torch.tensor([1000.,.1],dtype=torch.float64,requires_grad=True)
    original=raw.detach().clone()
    def objective():
        val=(state[:,0]-1).square().sum()+(ext.curve.u-2).square().sum()+(ext.curve.v-3).square().sum()+(raw-1).square().sum()
        return val,{}
    scaled_joint(objective,state,raw,ext,50,monotone_widths=True,relative_widths=True,
        optimizer_kind='torch_lbfgs',soft_minimum_width=True,freeze_widths=True,
        frozen_fiber_mask=[True,False],phase_coordinate_scale=1.,vertical_coordinate_scale=1.)
    torch.testing.assert_close(raw,original)
    assert state[0,0]==0 and ext.curve.u[0]==0 and ext.curve.v[0]==0
    assert abs(state[1,0].item()-1)<1e-5
    assert abs(ext.curve.u[1].item()-2)<1e-5 and abs(ext.curve.v[1].item()-3)<1e-5
