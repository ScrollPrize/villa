import numpy as np
import torch
from scipy.special import softmax
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def test_expansion_preserves_geometry_and_broken_probability():
    rng=np.random.default_rng(42)
    state=rng.normal(size=(4,303)).astype(np.float64)
    widths=np.linspace(200,500,100)
    expanded,w=uv.expand_winding_state(state,widths,151)
    p=softmax(state[:,:101],axis=1);q=softmax(expanded[:,:152],axis=1)
    np.testing.assert_allclose(q[:,151],p[:,100])
    np.testing.assert_allclose(q[:,:100],p[:,:100]*100/151)
    assert (q[:,100:151]>0).all()
    for ds in (-1000.,0.,1000.):
        args=[torch.tensor([-.3,.5,1.2,.9]),torch.arange(4,dtype=torch.float64),
              torch.tensor([0,1,0,1]),torch.full((4,),ds,dtype=torch.float64)]
        def geometry(s,width):
            wt=torch.from_numpy(width)
            return uv.coordinates(torch.from_numpy(s),*args,wt,torch.cat((wt.new_zeros(1),wt.cumsum(0))))
        old=geometry(state,widths);new=geometry(expanded,w)
        for a,b in zip(old,new):
            np.testing.assert_allclose(a,b[:,:100],atol=1e-9)


def test_no_expansion_is_identity():
    state=np.zeros((1,303));widths=np.full(100,200.)
    a,b=uv.expand_winding_state(state,widths,100)
    assert a is state and b is widths
