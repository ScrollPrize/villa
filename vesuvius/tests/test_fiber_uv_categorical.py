import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_categorical import (
    accumulate_pair_costs,categorical_target,ce_logit_step)


@pytest.mark.parametrize('choice',[0,2])
def test_tiny_compatible_or_broken_option_recovers(choice):
    logits=torch.tensor([[0.,0.,0.]],requires_grad=True)
    with torch.no_grad():logits[0,choice]=-30
    costs=torch.full((1,3),100.,requires_grad=True)
    with torch.no_grad():costs[0,choice]=0
    target=categorical_target(costs,.25)
    assert not target.requires_grad
    updated=ce_logit_step(logits,target,4.)
    assert updated[0,choice]-logits[0,choice]>3.99
    for _ in range(20):updated=ce_logit_step(updated,target,4.)
    assert updated.softmax(1)[0,choice]>.9


def test_both_endpoint_aggregation_matches_energy_derivative():
    p=torch.tensor([[.1,.7,.2],[.4,.1,.5],[.2,.2,.6]],dtype=torch.double,requires_grad=True)
    i=torch.tensor([0,0]);j=torch.tensor([1,2])
    cost=torch.tensor([[[0.,20.],[5.,0.]],[[10.,0.],[0.,8.]]],dtype=torch.double)
    # Original E^-1 pair sum plus N^-1 broken-node charge.
    energy=(cost*p[i,:2,None]*p[j,None,:2]).sum()/2+(p[:,2]*10).sum()/3
    expected=torch.autograd.grad(energy,p)[0]*3
    actual=torch.zeros_like(p);actual[:,2]=10
    accumulate_pair_costs(actual,i,j,cost,p,3/2)
    torch.testing.assert_close(actual,expected)
    split=torch.zeros_like(p);split[:,2]=10
    for e in range(2):accumulate_pair_costs(split,i[e:e+1],j[e:e+1],cost[e:e+1],p,3/2)
    torch.testing.assert_close(split,actual)


def test_broken_neighbor_is_zero_cost_and_multiple_events_combine():
    p=torch.tensor([[.9,.1,0.],[0.,0.,1.],[1.,0.,0.]])
    out=torch.zeros_like(p)
    accumulate_pair_costs(out,torch.tensor([0,0]),torch.tensor([1,2]),torch.ones(2,2,2)*10,p,1)
    torch.testing.assert_close(out[0],torch.tensor([10.,10.,0.]))
    assert categorical_target(torch.tensor([[20.,30.,10.]]),1)[0,2]>.99


def test_step_and_temperature_validation():
    x=torch.zeros(1,3)
    with pytest.raises(ValueError):categorical_target(x,0)
    with pytest.raises(ValueError):ce_logit_step(x,x,10)


def test_full_conditional_energy_derivative_with_priors_and_links():
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_uv_categorical import conditional_costs
    state=torch.zeros(3,9);state[:,:3]=torch.tensor([[1.,2.,-1.],[2.,1.,-.5],[.2,.1,-1.]])
    widths=torch.tensor([200.,300.],dtype=torch.double)
    data={key:torch.zeros(2,dtype=torch.double) for key in
          ('ds_a','ds_b','phase_a','phase_b','phase_delta','distance','sign')}
    data.update(i=torch.tensor([0,1]),j=torch.tensor([1,2]),family=torch.tensor([0,1,0]),
                phase=torch.tensor([.1,.2,.3],dtype=torch.double),z=torch.zeros(3,dtype=torch.double),
                degree=torch.tensor([1.,2.,1.],dtype=torch.double),
                node_ds=torch.zeros(3,3,dtype=torch.double),node_phase=torch.zeros(3,3,dtype=torch.double),
                node_z=torch.zeros(3,3,dtype=torch.double),links=torch.tensor([[0.,2.,1.,2.]],dtype=torch.double))
    data['sign'][:]=1;data['distance'][:]=.5
    table=torch.tensor(uv.make_distance_table(1,.5))
    cfg=dict(layout_mode='shared_anchor',batch_size=1,sign_charge_multiplier=4)
    costs,energy=conditional_costs(state,widths,data,table,cfg,True)
    cost2,energy2=conditional_costs(state,widths,data,table,dict(cfg,batch_size=2),True)
    torch.testing.assert_close(costs,cost2);torch.testing.assert_close(energy,energy2)
    p=state[:,:3].softmax(1)
    derivative=p*(costs-(costs*p).sum(1,keepdim=True))/3
    for row,col in [(0,0),(1,2),(2,1)]:
        plus=state.clone();minus=state.clone();eps=.01
        plus[row,col]+=eps;minus[row,col]-=eps
        ep=conditional_costs(plus,widths,data,table,cfg,True)[1]
        em=conditional_costs(minus,widths,data,table,cfg,True)[1]
        torch.testing.assert_close((ep-em)/(2*eps),derivative[row,col],atol=.002,rtol=.002)
