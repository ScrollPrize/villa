import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_winding_order import order_margin
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as imp


def test_family_order_and_reversed_endpoints():
    delta=np.array([0.,-.51,1.,.49])
    family=np.array([1,1,-1,-1])
    parallel=np.zeros(4,dtype=bool)
    result=order_margin(delta,np.ones(4),family,parallel)
    assert (result<0).tolist()==[False,True,False,True]
    np.testing.assert_allclose(result,order_margin(-delta,-np.ones(4),-family,parallel))
    assert order_margin(0.,1,1,np.bool_(False),0.)==0
    assert order_margin(.9,1,-1,np.bool_(False),0.)<0


def test_lasagna_calibration_does_not_change_sign_mass():
    a=torch.tensor([[0.,.2,0.,10.]])
    b=torch.tensor([[0.,.2,0.,10.]])
    widths=torch.ones(10)*1000
    results=[];losses=[]
    for calibration in (.1,.5,1.5):
        diag={}
        loss=imp.pair_cost(a,b,torch.tensor([0]),torch.tensor([1]),
            torch.zeros(1),torch.zeros(1),widths,torch.ones(1),torch.ones(1)*.3,
            torch.zeros(1,dtype=torch.bool),dict(winding_sigma=.5,
            perpendicular_offset=calibration),diagnostics=diag)
        results.append(diag['wrong_probability']);losses.append(loss[0])
    for r in results[1:]:torch.testing.assert_close(results[0],r)
    assert not torch.allclose(losses[0],losses[-1])
