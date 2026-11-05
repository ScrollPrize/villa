import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_kernels import interpolate_offsets,BatchedInterpolation


@pytest.mark.parametrize('values',[[],[2,2,2],[48,3,8,4,16,32,48,2,8,16]])
def test_sigma_partition_preserves_order(values):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import sigma_group_plan
    radius=torch.tensor(values,dtype=torch.long);bounds=[4,8,16,32,48]
    a=sigma_group_plan(radius,bounds);b=sigma_group_plan(radius,bounds,True)
    assert len(a)==len(b)
    for (upper,x),(other,y) in zip(a,b):
        assert upper==other
        torch.testing.assert_close(x,y,rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_interpolation_backward(dtype):
    torch.manual_seed(4)
    offsets=torch.randn(5,4,2,dtype=dtype,requires_grad=True)
    labels=torch.tensor([[-1,0,1,2],[2,3,4,5]])
    active=torch.tensor([[True,True,True,False],[True,True,False,False]])
    ids=torch.tensor([0,1,0,1]);left=torch.tensor([0,2,-1,2]);right=torch.tensor([1,4,-1,4])
    t=torch.tensor([.25,.8,.4,1.2],dtype=dtype,requires_grad=True)
    k=torch.tensor([[-1.3,.2,2.4],[2.1,3.2,5.2],[.1,1.4,4.3],[1.2,2.3,3.4]],dtype=dtype,requires_grad=True)
    args=offsets,labels,active,ids,left,right,t,k
    a=interpolate_offsets(*args);b=BatchedInterpolation.apply(*args)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    weight=torch.randn_like(a)
    ga=torch.autograd.grad((a*weight).sum(),(offsets,t,k))
    gb=torch.autograd.grad((b*weight).sum(),(offsets,t,k))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_affine_backward(dtype):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_kernels import affine_coordinates,BatchedAffine
    torch.manual_seed(7)
    base=torch.randn(7,4,2,dtype=dtype,requires_grad=True)
    local=torch.randn_like(base,requires_grad=True)
    u=torch.randn(3,4,3,dtype=dtype,requires_grad=True)
    v=torch.randn_like(u,requires_grad=True)
    owner=torch.tensor([0,0,1,2,2,2,0]);alpha=torch.linspace(-.2,1.2,7,dtype=dtype)
    args=base,local,u,v,owner,alpha
    a=affine_coordinates(*args);b=BatchedAffine.apply(*args)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    weight=torch.randn_like(a)
    ga=torch.autograd.grad((a*weight).sum(),args[:4])
    gb=torch.autograd.grad((b*weight).sum(),args[:4])
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)


def test_batched_backward_full_objective():
    from .test_fiber_affine_geometry import setup
    from .test_fiber_intermediate_uv import costs
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_geometry_cache import geometry_evaluation
    d,cfg,td,s,w,e,_=setup()
    torch.manual_seed(11)
    with torch.no_grad():
        for affine in (e.affine,e.curve.affine,e.intermediate.affine):
            affine.u.add_(torch.randn_like(affine.u)*.03)
            affine.v.add_(torch.randn_like(affine.v)*.03)
    s=s.detach().requires_grad_();w=w.detach().requires_grad_()
    params=[s,w,*e.parameters()]
    cfg['reuse_objective_geometry']=True
    e.intermediate.config['reuse_objective_geometry']=True
    def evaluate(enabled):
        e.intermediate.config['batched_interpolation_backward']=enabled
        for affine in (e.affine,e.curve.affine,e.intermediate.affine):
            affine.batched_backward=enabled
        with geometry_evaluation():
            loss=sum(e.terms(s,w,td,torch.arange(3)).values())+sum(x.sum() for x in costs(cfg,td,s,w,e))
            return loss,torch.autograd.grad(loss,params,allow_unused=True)
    a,ga=evaluate(False);b,gb=evaluate(True)
    torch.testing.assert_close(a,b)
    for x,y in zip(ga,gb):
        if x is None or y is None:assert x is y
        else:torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-5)
