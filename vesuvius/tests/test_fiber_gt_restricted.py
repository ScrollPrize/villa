import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_current_gt_solve import enclosure_subset
from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import pair_cost


def test_enclosure_targets_orientation_and_subset():
    d=dict(family=np.array([0,1,1]),source_id=np.array([10,20,30]),
        i=np.array([0,1,0]),j=np.array([1,0,2]),ds_a=np.array([0.,0.,5.]),
        ds_b=np.array([0.,10.,5.]),sign=np.array([1.,1.,1.]),
        parallel=np.array([False,False,True]),distance=np.array([.2,.9,1.]))
    selected,audit=enclosure_subset(d)
    assert selected['distance'].tolist()==[0.,1.]
    assert audit['original_events']==[0,1]
    assert audit['unconnected_sources']==[30]


def test_sign_only_ignores_distance_and_calibration():
    a=torch.tensor([[3.,.1,0.,100.]],requires_grad=True)
    b=torch.tensor([[2.,.1,100.,100.]],requires_grad=True)
    f=torch.tensor([0]);g=torch.tensor([1]);zero=torch.zeros(1);w=torch.full((10,),1000.)
    cfg=dict(gt_pair_mode='sign_only',winding_sigma=.2,sign_violation_weight=30.,
             soft_pair_weight=1.,probability_pair_weight=.3)
    def cost(dist,offset):return pair_cost(a,b,f,g,zero,zero,w,torch.ones(1),torch.tensor([dist]),
                torch.zeros(1,dtype=torch.bool),dict(cfg,perpendicular_offset=offset))
    x=cost(.2,.5);y=cost(99.,0.)
    for u,v in zip(x,y):torch.testing.assert_close(u,v)
    assert x[1]==0 and x[2]==0
    x[0].sum().backward()
    assert b.grad[0,1]<0 # gradient descent moves B outward to repair order


def test_fixed_target_cost_at_target_contains_only_hypothesis_variance():
    a=torch.tensor([[3.,.1,0.,100.]])
    b=torch.tensor([[4.,.1,0.,100.]],requires_grad=True)
    args=(a,b,torch.tensor([0]),torch.tensor([1]),torch.zeros(1),torch.zeros(1),
        torch.full((10,),1000.),torch.ones(1),torch.ones(1),torch.zeros(1,dtype=torch.bool))
    cfg=dict(gt_pair_mode='enclosure',winding_sigma=.2,soft_pair_weight=1.,probability_pair_weight=0.)
    result=pair_cost(*args,cfg)
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_implicit_winding import hypotheses
    h,lp=hypotheses(a,.2)
    variance=(lp.exp()*(h[...,0]-a[:,:1]).square()).sum(1)
    torch.testing.assert_close(result[0],8*variance)  # Var(kB-kA) / distance_sigma**2
    assert result[1]==0 and result[2]==0
