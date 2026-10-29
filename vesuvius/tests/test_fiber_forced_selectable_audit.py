from types import SimpleNamespace
import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_shared_curve import SharedCurve
from vesuvius.neural_tracing.fiber_trace_3d.fiber_forced_selectable_audit import transfer_parent_geometry


def test_transfer_preserves_all_hypotheses_not_just_map():
    def bundle(family):
        n=len(family)
        d=dict(family=np.array(family),start=np.zeros(n),node_ds=np.tile([0.,100.],(n,1)))
        curve=SharedCurve(d,np.arange(n),np.zeros(n),dict(straight_fixed_length_geometry=True),'cpu')
        return SimpleNamespace(curve=curve,cavity=SimpleNamespace(values=torch.ones(n,dtype=torch.float64)))
    source=bundle([0,1]);target=bundle([0,0,1,1])
    parent=torch.tensor([0,0,1,1]);shift=torch.tensor([0.,170.,0.,-70.],dtype=torch.float64)
    with torch.no_grad():
        source.curve.u.copy_(torch.tensor([.3,.7]))
        source.curve.v.copy_(torch.tensor([50.,90.]))
    source.cavity.values=torch.tensor([.2,1.3],dtype=torch.float64)
    ids=np.arange(4);local=np.array([10.,40.,15.,25.])
    target.curve.add_plan('a',ids,local)
    target.curve.add_plan('b',ids,local+10)
    original=dict(target.curve.plans)
    transfer_parent_geometry(source,target,parent,shift)
    torch.testing.assert_close(target.cavity.values,source.cavity.values[parent],rtol=0,atol=0)
    state=torch.zeros(2,4,dtype=torch.float64)
    widths=torch.tensor([100.,150.,300.,600.,1000.,1500.],dtype=torch.float64)
    # Include fractional labels used by the continuous guidance loss, too.
    labels=torch.tensor([[-2.,0.,1.,1.5,2.,4.,7.]],dtype=torch.float64).expand(4,-1)
    for key,plan in original.items():
        owner,arc=plan[:2]
        expected=source.curve.hypothesis_queries(state,widths,(parent[owner],arc+shift[owner],None,None,None),labels)
        actual=target.curve.hypothesis_queries(state[parent],widths,target.curve.plans[key],labels)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(target.curve.pair_plan_values[:,0,0],torch.as_tensor(local)+shift)
