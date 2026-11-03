import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_coordinates import combined_coordinates
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import q_to_u, u_to_q


def evaluate(a, d, w, combined):
    c = torch.cat((w.new_zeros(1), w.cumsum(0)))
    if combined:
        return combined_coordinates(a, d, w, c)
    u = q_to_u(a, w, c) + d
    return u, u_to_q(u, w, c)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('outputs', ['u', 'q', 'both'])
def test_values_gradients(dtype, outputs):
    torch.manual_seed(19)
    a = (torch.rand(8, 7, dtype=dtype)*10-2).requires_grad_()
    d = (torch.randn(8, 1, dtype=dtype)*300).requires_grad_()
    w = (torch.rand(7, dtype=dtype)*100+100).requires_grad_()
    gu, gq = torch.randn_like(a), torch.randn_like(a)
    results = []
    for combined in [False, True]:
        u, q = evaluate(a, d, w, combined)
        loss = (u*gu).sum() if outputs == 'u' else (q*gq).sum()
        if outputs == 'both': loss = loss+(u*gu).sum()
        results.append((u, q, *torch.autograd.grad(loss, (a,d,w))))
    for x, y in zip(*results):
        torch.testing.assert_close(x,y,rtol=3e-5 if dtype==torch.float32 else 1e-10,
                                   atol=3e-4 if dtype==torch.float32 else 1e-10)


def test_gradcheck():
    a = torch.tensor([[-1.2,.2,1.4,2.1,4.3]],dtype=torch.float64,requires_grad=True)
    d = torch.tensor([[23.]],dtype=torch.float64,requires_grad=True)
    w = torch.tensor([110.,130.,150.],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,d,w:evaluate(a,d,w,True),(a,d,w))


def test_boundaries_zero_distance_and_extrapolation():
    w = torch.tensor([100.,200.,300.],dtype=torch.float64,requires_grad=True)
    a = torch.tensor([[-2.,0.,1.,2.,3.,5.]],dtype=torch.float64,requires_grad=True)
    u,q = evaluate(a,torch.zeros_like(a),w,True)
    torch.testing.assert_close(q,a,rtol=0,atol=0)
    ga,gw = torch.autograd.grad(q.sum(),(a,w))
    torch.testing.assert_close(ga,torch.ones_like(a),rtol=0,atol=0)
    torch.testing.assert_close(gw,torch.zeros_like(w),rtol=0,atol=0)
    ru,rq = evaluate(a,torch.zeros_like(a),w,False)
    torch.testing.assert_close(u,ru)
    torch.testing.assert_close(q,rq)


def test_noncontiguous():
    a=torch.rand(5,4,dtype=torch.float64).T.requires_grad_()
    d=torch.ones(4,1,dtype=torch.float64,requires_grad=True)
    w=torch.tensor([100.,200.],dtype=torch.float64,requires_grad=True)
    u,q=evaluate(a,d,w,True)
    assert all(torch.isfinite(x).all() for x in torch.autograd.grad((u+q).sum(),(a,d,w)))


def test_full_state_and_softplus_width_chain():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import coordinates, width_boundaries
    torch.manual_seed(13)
    k=7
    state=(torch.randn(5,3*k+3,dtype=torch.float64)*.05).requires_grad_()
    raw=(torch.rand(k,dtype=torch.float64)*200).requires_grad_()
    phase=torch.rand(5,dtype=torch.float64)
    z=torch.rand(5,dtype=torch.float64)*500
    family=torch.tensor([0,1,0,1,0])
    ds=torch.tensor([-211.,0.,451.,20.,0.],dtype=torch.float64)
    weights=[torch.randn(5,k,dtype=torch.float64) for _ in range(3)]
    results=[]
    for backend in ['reference','combined']:
        w,c=width_boundaries(raw)
        out=coordinates(state,phase,z,family,ds,w,c,backend)
        loss=sum((x*y).sum() for x,y in zip(out,weights))
        results.append((*out,*torch.autograd.grad(loss,(state,raw))))
    for x,y in zip(*results):torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('division',['standard','precise'])
@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA needs managed validation')
def test_cuda_values_gradients_determinism(division):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_coordinates import validate_cuda
    torch.use_deterministic_algorithms(True)
    assert validate_cuda(division)['repeat_bitwise']


def test_invalid_division():
    with pytest.raises(ValueError,match='Unknown coordinate division'):
        combined_coordinates(torch.zeros(1,1),torch.zeros(1,1),torch.ones(1),torch.tensor([0.,1.]),'typo')


@pytest.mark.parametrize('division',['standard','precise'])
def test_division_plumbing(monkeypatch,division):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_coordinates as module
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import coordinates
    original=module.combined_coordinates
    seen=[]
    def spy(a,d,w,c,mode):
        seen.append(mode)
        return original(a,d,w,c,mode)
    monkeypatch.setattr(module,'combined_coordinates',spy)
    coordinates(torch.zeros(1,9),torch.zeros(1),torch.zeros(1),torch.zeros(1,dtype=torch.long),
                torch.zeros(1),torch.ones(2),torch.tensor([0.,1.,2.]),'combined',division)
    assert seen==[division]
