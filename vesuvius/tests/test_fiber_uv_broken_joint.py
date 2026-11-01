import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_layout import (
    rejection_weights,validate_joint_rejection,calibrated_sign_charge)


def test_fixed_cost_independent_of_degree():
    degree=torch.tensor([0.,1.,1000.])
    weights=rejection_weights(degree,degree.mean())
    torch.testing.assert_close(weights,torch.ones(3))
    probabilities=torch.tensor([.1,.5,1.])
    torch.testing.assert_close(10*probabilities*weights,torch.tensor([1.,5.,10.]))


@pytest.mark.parametrize('constraint_cost,gradient_sign',[(0,1),(100,-1)])
def test_broken_can_move_both_ways_from_first_update(constraint_cost,gradient_sign):
    # Two winding hypotheses plus broken, initialized with1%broken probability.
    embedding=torch.nn.Embedding.from_pretrained(torch.tensor([[0.,0.,-3.902]]),freeze=False,sparse=True)
    optimizer=torch.optim.SparseAdam(embedding.parameters(),lr=.03)
    before=embedding.weight.detach().clone()
    p=embedding(torch.tensor([0])).softmax(-1)
    loss=constraint_cost*p[:,:2].sum()+10*p[:,-1].sum()
    loss.backward()
    assert embedding.weight.grad.coalesce().values()[0,-1]*gradient_sign>0
    optimizer.step()
    assert (embedding.weight[0,-1]-before[0,-1])*gradient_sign<0


def test_no_hidden_warmup():
    validate_joint_rejection({})
    validate_joint_rejection({'rejection_warmup':0})
    with pytest.raises(ValueError,match='from update0'):
        validate_joint_rejection({'rejection_warmup':.5})


def test_fixed_charge_matches_fixed_rejection_cost():
    degree=torch.tensor([0.,1000.])
    charge=calibrated_sign_charge(degree,degree.flip(0),degree.mean(),100,10,10.,4.,'fixed')
    torch.testing.assert_close(charge,torch.full((2,),400.))
