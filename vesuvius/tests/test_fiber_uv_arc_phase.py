import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_arc_phase as ap


def test_phase_arc_and_cpu_agree():
    widths=torch.tensor([100.,200.,400.],dtype=torch.float64)
    bounds=torch.cat((torch.zeros(1,dtype=torch.float64),widths.cumsum(0)))
    state=torch.zeros(2,12,dtype=torch.float64,requires_grad=True)
    phase=torch.tensor([.5,.5],dtype=torch.float64);family=torch.tensor([0,1])
    ds=torch.tensor([150.,150.],dtype=torch.float64)
    q=ap.coarse_coordinates(state,phase,family,ds,widths,bounds)
    assert q[0,0].item()==pytest.approx(1.5)
    assert q[1,0].item()==pytest.approx(.5)
    q.sum().backward();assert state.grad[:,4].abs().min()>0
    d=dict(phase=phase.numpy(),family=family.numpy())
    cpu=ap.coarse_points(d,state.detach().numpy(),np.zeros(2,int),widths.numpy(),np.arange(2),ds.numpy())
    np.testing.assert_allclose(cpu,q.detach().numpy()[:,0])
    changed=state.detach().clone();changed[:,6:]=.2
    torch.testing.assert_close(ap.coarse_coordinates(changed,phase,family,ds,widths,bounds),q)


def test_bucket_phase_gauge_and_width_gradients():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    widths=torch.tensor([100.,200.,400.],dtype=torch.float64,requires_grad=True)
    state=torch.zeros(2,12,dtype=torch.float64)
    state[0,4]=.5;state[1,4]=-.5
    phase=torch.zeros(2,dtype=torch.float64);family=torch.zeros(2,dtype=torch.int64)
    ds=torch.full((2,),75.,dtype=torch.float64)
    def run(w):
        bounds=torch.cat((w.new_zeros(1),w.cumsum(0)))
        return ap.coarse_coordinates(state,phase,family,ds,w,bounds)
    q=run(widths)
    torch.testing.assert_close(q[0,0],q[1,1])
    assert torch.autograd.gradcheck(run,(widths,),atol=1e-6)


def test_audit_ignores_local_angle_and_seam():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    k=3;state=np.zeros((2,3*k+3));widths=np.array([100.,200.,400.])
    state[0,k+1]=.5;state[1,k+1]=-.5
    d=dict(i=np.array([0]),j=np.array([1]),phase=np.zeros(2),family=np.array([0,1]),
           ds_a=np.array([0.]),ds_b=np.array([300.]),phase_delta=np.array([1234.]),bucket_shift=np.array([2000.]))
    pick=np.array([0,1]);bounds=np.r_[0.,np.cumsum(widths)]
    geometry=(None,None,np.ones(2,bool),pick,np.array([50.,50.]),np.zeros(2),bounds)
    _,_,delta,active=uv.edge_readout(d,geometry,widths,np.array([0]),dict(arc_phase_constraints=True),state)
    np.testing.assert_array_equal(delta,[0.]);assert active.all()


def test_arc_gt_not_angles():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_gt_constraint_score import event_residuals
    d=dict(phase=np.array([.25]),family=np.array([0]),start=np.array([10.]),end=np.array([110.]),node_ds=np.array([[0.,50.,100.]]))
    state=np.zeros((1,12));widths=np.array([100.,200.,400.]);pick=np.array([0])
    events=[dict(test_piece=0,gt_piece=0,test_arc=60.,gt_arc=60.,signed_distance=.1,family_delta=0,parallel=True)]
    value=event_residuals(events,d,state,pick,3,d,dict(state=state,widths=widths),pick,dict(arc_phase_constraints=True),{},widths)
    np.testing.assert_allclose(value,0)
