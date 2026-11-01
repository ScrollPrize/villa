import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def test_constraint_equality_is_separate_from_global_z_prior():
    v=torch.tensor([[500.]],requires_grad=True)
    z_a=torch.tensor([[0.]])
    z_b=torch.tensor([[1000.]])
    q=torch.zeros_like(v);zero=torch.zeros(1)
    metric=uv.pair_geometry(q,q,v,v,zero,zero,zero,{})[-1]
    assert metric.item()==0
    # Both point priors remain nonzero although the connection is satisfied.
    torch.testing.assert_close(uv.vertical_layout_prior(v,z_a,{}),torch.tensor([[.0025]]))
    torch.testing.assert_close(uv.vertical_layout_prior(v,z_b,{}),torch.tensor([[.0025]]))
    unequal=uv.pair_geometry(q,q,v,v+128,zero,zero,zero,{})[-1]
    torch.testing.assert_close(unequal,torch.tensor([[[.01]]]))


def test_unary_prior_is_weak_independent_and_allows_z_reversal():
    v=torch.tensor([[0.],[100.],[200.]],requires_grad=True)
    z=torch.tensor([[200.],[100.],[0.]])  # Physical Z decreases along increasing V.
    cost=uv.vertical_layout_prior(v,z,{'prior_weight':999.})
    torch.testing.assert_close(cost,torch.tensor([[.0004],[0.],[.0004]]))
    cost.mean().backward()
    torch.testing.assert_close(v.grad,torch.tensor([[-4e-6/3],[0.],[4e-6/3]]))
    assert torch.isfinite(cost).all()  # No rejection, direction check or slope term.
    assert torch.count_nonzero(uv.vertical_layout_prior(v,z,{'v_prior_weight':0}))==0


def test_old_target_array_is_not_loaded_or_required():
    keys=('family','phase','z','degree','node_ds','node_phase','node_z','i','j','ds_a','ds_b',
          'phase_a','phase_b','phase_delta','distance','sign','parallel','links')
    data={k:np.zeros(1) for k in keys}
    assert 'target_v' not in uv._tensor_data(data,'cpu')
    data['target_v']=np.array([float('nan')])
    assert 'target_v' not in uv._tensor_data(data,'cpu')


@pytest.mark.parametrize('config',[{'v_prior_scale':0},{'v_prior_scale':float('inf')},
                                  {'v_prior_weight':-1},{'v_prior_weight':float('nan')}])
def test_invalid_vertical_prior(config):
    with pytest.raises(ValueError):uv.vertical_layout_prior(torch.zeros(1),torch.zeros(1),config)
