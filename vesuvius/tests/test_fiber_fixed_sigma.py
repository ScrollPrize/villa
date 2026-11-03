import numpy as np
import torch
from vesuvius.neural_tracing.fiber_trace_3d.fiber_cavity_sigma import FixedSigma


def test_fixed_sigma_never_uses_geometry_and_roundtrips():
    model=FixedSigma(dict(family=np.array([0,1])),dict(winding_sigma=.5,sigma_max=8),'cpu')
    model.restore({})
    for step in range(12):model.update(None,None,None,step)
    torch.testing.assert_close(model.values,torch.full((2,),.5))
    saved=model.arrays();model.restore(saved)
    assert model.diagnostics()['cavity_rounds']==0
    kwargs=model.pair_kwargs(dict(i=torch.tensor([0]),j=torch.tensor([1])),torch.tensor([0]))
    assert kwargs['sigma_a'].item()==kwargs['sigma_b'].item()==.5
