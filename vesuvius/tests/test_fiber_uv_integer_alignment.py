import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv

CFG=dict(unified_winding_loss=True,integer_alignment_loss=True,
    bucket_position_split=True,bucket_shared_phase=True,layout_mode='shared_anchor',
    distance_objective='squared',distance_sigma=.5,distance_weight=.25,
    position_distance_weight=2.,sign_weight=0.,metric_weight=0.,loss_reference_width=8000.)

def loss(delta,mu=.7,fd=1.,compiled=False,coarse_delta=None,charge=None):
    a=torch.zeros_like(delta);p=torch.ones_like(delta);w=torch.full((1,),8000.,dtype=delta.dtype)
    fn=torch.compile(uv.pair_objective,backend='aot_eager',fullgraph=True) if compiled else uv.pair_objective
    return fn(p,p,a,delta,a,a,torch.zeros(1),torch.tensor([fd]),torch.ones(1),
              torch.tensor([mu]),None,CFG,charge=charge,coarse=(a,delta if coarse_delta is None else coarse_delta,w))

@pytest.mark.parametrize('mu',[.1,.5,.7,1.2])
def test_fractional_measurement_does_not_move_zero_target(mu):
    d=torch.tensor([[0.]],dtype=torch.float64,requires_grad=True)
    value=loss(d,mu)
    assert torch.autograd.grad(value,d)[0].item()==0
    assert loss(torch.tensor([[.1]],dtype=torch.float64),mu)>value
    assert loss(torch.tensor([[-.1]],dtype=torch.float64),mu)>value

def test_integer_evidence_not_fractional_target_and_not_free_tails():
    z=torch.tensor([[0.]],dtype=torch.float64)
    torch.testing.assert_close(loss(z,mu=1.,fd=1.),loss(z+1,mu=1.,fd=1.))
    assert loss(z+10,mu=1.,fd=1.)>loss(z,mu=1.,fd=1.)
    assert loss(z+.2,mu=.7)>loss(z,mu=.7)

def test_phase_gradient_and_compiled_parity():
    d=torch.tensor([[.2]],dtype=torch.float64,requires_grad=True)
    v=loss(d);grad=torch.autograd.grad(v,d)[0]
    assert grad.item()>0
    vc=loss(d,compiled=True);gc=torch.autograd.grad(vc,d)[0]
    torch.testing.assert_close(v,vc);torch.testing.assert_close(grad,gc)

def test_integer_target_uses_unwrapped_coordinates_not_raw_indices():
    # A bucket +1 with phase -1 produces the same full coordinate and objective.
    a=torch.tensor([[.5]],dtype=torch.float64)
    b=torch.tensor([[1.-.5]],dtype=torch.float64)
    torch.testing.assert_close(loss(b-a),loss(torch.zeros_like(a)))

def test_geometry_required():
    with pytest.raises(ValueError,match='unified'):
        uv.reject_superseded_objectives(dict(integer_alignment_loss=True))

@pytest.mark.parametrize('fd,step,violation',[(1.,0.,False),(1.,-1.,True),(-1.,1.,False),(-1.,0.,True)])
def test_integer_hv_sign_order(fd,step,violation):
    d=torch.tensor([[step]],dtype=torch.float64)
    difference=loss(d,fd=fd,charge=torch.tensor([100.]))-loss(d,fd=fd)
    assert difference.item()==(100. if violation else 0.)

def test_fine_residual_not_coarse_or_measurement_difference():
    fine=torch.tensor([[.3]],dtype=torch.float64,requires_grad=True)
    coarse=torch.tensor([[.2]],dtype=torch.float64)
    value=loss(fine,mu=.7,coarse_delta=coarse)
    nll=.5*((.5-.7)/.5)**2
    expected=.25*(nll+2*torch.exp(torch.tensor(-nll))*.5*(fine/.5)**2)
    torch.testing.assert_close(value,expected.squeeze(),atol=1e-8,rtol=1e-6)
