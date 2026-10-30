import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


@pytest.mark.parametrize('dtype',[torch.float64,torch.float32])
@pytest.mark.parametrize('ties',[False,True])
def test_prefix_sign_value_and_gradients(dtype,ties):
    torch.manual_seed(12)
    p=torch.rand(3,13,dtype=dtype,requires_grad=True)
    b=torch.rand(3,13,dtype=dtype,requires_grad=True)
    q=torch.randn(3,13,dtype=dtype)*3+100
    y=torch.randn(3,13,dtype=dtype)*3+100
    phase=torch.tensor([.1,.4,.9],dtype=dtype)
    family=torch.tensor([1.,-1.,0.],dtype=dtype)
    if ties:
        q=torch.arange(13,dtype=dtype).expand(3,-1).clone()+100
        y=q.clone();phase.zero_();family.zero_()
    q.requires_grad_();y.requires_grad_()
    sign=torch.tensor([-1.,0.,1.],dtype=dtype)
    actual=uv.prefix_sign_terms(p,b,q,y,phase,family,sign)
    wrong=uv.pair_geometry(q,y,q*0,y*0,phase,family,sign,{})[2]
    expected=(p[:,:,None]*b[:,None,:]*wrong).sum((1,2))
    tol=1e-4 if dtype==torch.float32 else 1e-10
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    ga=torch.autograd.grad(actual.sum(),(p,b,q,y),retain_graph=True)
    gb=torch.autograd.grad(expected.sum(),(p,b,q,y))
    for a,b in zip(ga,gb):torch.testing.assert_close(a,b,atol=tol,rtol=tol)


def test_window_boundary_and_mass():
    p=torch.tensor([[.4,.2,.1,0.],[0.,0.,0.,0.]])
    q=torch.arange(4.).expand(2,-1)
    a,x,_,coverage=uv.local_winding_window(p,q,q,1)
    torch.testing.assert_close(a.sum(1),p.sum(1))
    assert a[0,0]==0 and a[0,1]>a[0,2]
    torch.testing.assert_close(coverage[:,0],torch.tensor([.6/.7,0.]))
    assert torch.isfinite(a).all()


def test_local_full_range_and_sign_global():
    torch.manual_seed(1)
    p=torch.rand(3,9,dtype=torch.float64);p=p/p.sum(1,keepdim=True)*.7
    q=torch.randn_like(p);v=torch.randn_like(p)*100
    z=torch.zeros(3,dtype=p.dtype);sign=torch.tensor([-1,0,1],dtype=p.dtype)
    config=dict(layout_mode='shared_anchor',distance_objective='squared')
    args=(p,p,q,q+1,v,v+3,z,z,sign,z+1,torch.zeros(2,2))
    full=uv.pair_objective(*args,config)
    torch.testing.assert_close(full,uv.pair_objective(*args,dict(config,local_window_radius=8)))
    config.update(distance_weight=0.,metric_weight=0.)
    full=uv.pair_objective(*args,config)
    for r in (0,1,2):
        torch.testing.assert_close(full,uv.pair_objective(*args,dict(config,local_window_radius=r)))


def test_gathered_full_coverage_loss_and_gradients():
    torch.manual_seed(9)
    p=torch.rand(2,9,dtype=torch.float64);p[:,4]=10
    inputs=[p.requires_grad_(),p.clone().detach().requires_grad_()]
    inputs += [torch.randn_like(p).requires_grad_() for _ in range(4)]
    z=torch.zeros(2,dtype=p.dtype)
    fixed=(z+.2,z+.5,z+1,z+1,torch.zeros(2,2))
    config=dict(layout_mode='shared_anchor',distance_objective='squared')
    full=uv.pair_objective(*inputs,*fixed,config)
    local=uv.pair_objective(*inputs,*fixed,dict(config,local_window_radius=4))
    torch.testing.assert_close(full,local)
    a=torch.autograd.grad(full,inputs,retain_graph=True)
    b=torch.autograd.grad(local,inputs)
    for x,y in zip(a,b):torch.testing.assert_close(x,y)


def test_near_equal_float32_prefix():
    q=torch.tensor([[149.,149.00002,149.00004]],requires_grad=True)
    y=torch.tensor([[149.00001,149.00003,149.00005]],requires_grad=True)
    p=torch.tensor([[.2,.3,.4]],requires_grad=True);z=torch.zeros(1)
    actual=uv.prefix_sign_terms(p,p,q,y,z,z,z+1)
    expected=(p[:,:,None]*p[:,None,:]*uv.pair_geometry(q,y,q*0,y*0,z,z,z+1,{})[2]).sum()
    torch.testing.assert_close(actual.sum(),expected,atol=1e-9,rtol=1e-5)
