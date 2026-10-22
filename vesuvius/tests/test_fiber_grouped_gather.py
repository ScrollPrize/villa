import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_grouped_gather import width_gather_mode,width_lookup
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
from vesuvius.neural_tracing.fiber_trace_3d import fiber_soft_winding as soft


@pytest.mark.parametrize('mode',['select','grouped'])
def test_width_maps_values_and_gradients(mode):
    widths=torch.tensor([11.,13.,17.,19.],dtype=torch.float64,requires_grad=True)
    q=torch.tensor([[-1.2,.3,1.7],[1.7,3.2,6.1]],dtype=torch.float64,requires_grad=True)
    def calc():
        bounds=torch.cat((widths.new_zeros(1),widths.cumsum(0)))
        u=uv.q_to_u(q,widths,bounds)
        return uv.u_to_q(u+3.2,widths,bounds).square().sum()+u.square().sum()
    original=calc();g=torch.autograd.grad(original,(q,widths))
    with width_gather_mode(mode):
        new=calc();gg=torch.autograd.grad(new,(q,widths))
    torch.testing.assert_close(original,new,rtol=0,atol=0)
    for a,b in zip(g,gg):torch.testing.assert_close(a,b)
    with width_gather_mode(mode):
        values=torch.randn(5,dtype=torch.float64,requires_grad=True)
        index=torch.tensor([[3,3,1],[3,1,1]])
        assert torch.autograd.gradcheck(lambda v:width_lookup(v,index),(values,))
        empty=width_lookup(values,torch.empty((0,2),dtype=torch.long))
        assert torch.equal(torch.autograd.grad(empty.sum(),values)[0],torch.zeros_like(values))


def test_pair_scope_and_context_reset(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_grouped_gather import _mode
    monkeypatch.setattr(soft,'validate',lambda cfg:None)
    def pair():return _mode.get()
    def solve_impl(config,device,pair_cost,*args):
        assert _mode.get()=='original'
        assert pair_cost()=='select'
        assert _mode.get()=='original'
        raise RuntimeError('test unwind')
    monkeypatch.setattr(soft,'_solve',solve_impl)
    cfg=dict(width_gather_scope='pairs',width_gather_mode='select')
    with pytest.raises(ValueError,match='requires a pair cost'):soft.solve(cfg,device='cpu')
    with pytest.raises(ValueError,match='audit requires'):
        soft.solve(dict(cfg,indexing_audit=True),device='cpu',pair_cost=pair)
    with pytest.raises(RuntimeError,match='test unwind'):soft.solve(cfg,device='cpu',pair_cost=pair)
    assert _mode.get()=='original'
