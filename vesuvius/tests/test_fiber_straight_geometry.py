import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve, saved_points
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import width_boundaries
from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft


def setup():
    d=dict(family=np.array([0,1]),start=np.zeros(2),node_ds=np.array([[0.,27.],[0.,-21.]]))
    owner=np.array([0,0,0,1,1,1]);arc=np.array([0.,10.,27.,0.,10.,21.])
    cfg=dict(straight_fixed_length_geometry=True,shared_curve_integer_geometry=True)
    c=SharedCurve(d,owner,arc,cfg,'cpu')
    state=torch.tensor([[1.,.23,11.,100.],[2.,.37,25.,100.]],dtype=torch.float64)
    w=torch.tensor([8.,11.,17.,23.,29.],dtype=torch.float64)
    c.initialize(state,w)
    return d,c,state,w


def test_exact_length_all_hypotheses_and_saved_restore():
    d,c,s,w=setup()
    ids=c.owner;ds=(c.arc-c.start[ids])*c.direction[ids]
    labels=torch.tensor([[-2.,0.,1.,3.,7.]],dtype=torch.float64).expand(len(ids),-1)
    q,_,v=c.hypothesis_queries(s,w,(ids,ds,None,None,None),labels)
    x=soft.uv.q_to_u(q,w,width_boundaries(w))
    r=torch.tensor([1,2,4,5]);l=r-1
    length=torch.hypot(x[r]-x[l],v[r]-v[l])
    torch.testing.assert_close(length,(c.arc[r]-c.arc[l])[:,None].expand_as(length),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(v[:3],v[:1].expand(3,-1))
    torch.testing.assert_close(x[3:],x[3:4].expand(3,-1))
    c.terms(torch.nn.Parameter(s.clone()),w)
    saved=dict(c.arrays(),state=s.numpy(),widths=w.numpy())
    actual=saved_points(d,saved,s.numpy(),w.numpy(),ids.numpy(),ds.numpy())
    expected=c.hypothesis_queries(s,w,(ids,ds,None,None,None),s[ids,0,None])
    for a,b in zip(actual,expected):np.testing.assert_allclose(a,b[:,0].detach().numpy(),atol=1e-12)
    assert sum(p.numel() for p in c.parameters())==2*len(s)
    with pytest.raises(ValueError,match='across straight/free'):
        c.restore(dict(shared_curve_phase_v=np.zeros((6,2))))


def test_inverse_geometry_gradients_and_boundary_continuity():
    _,c,s,w=setup()
    ids=torch.tensor([0,1]);ds=torch.tensor([13.2,-9.7],dtype=torch.float64)
    labels=torch.tensor([[.8,2.1],[1.4,3.2]],dtype=torch.float64,requires_grad=True)
    w=w.requires_grad_()
    def f(ww,kk):
        return torch.stack(c.hypothesis_queries(s,ww,(ids,ds,None,None,None),kk),-1)
    assert torch.autograd.gradcheck(f,(w,labels),eps=1e-6,atol=1e-6,rtol=1e-4)
    b=width_boundaries(w)
    x=b[2].detach()+torch.tensor([-1e-7,1e-7],dtype=torch.float64)
    q=soft.uv.u_to_q(x,w,b)
    assert float((q[1]-q[0]).detach())<1e-6
