import numpy as np
import pytest
import torch
from scipy.special import logsumexp,softmax
from vesuvius.neural_tracing.fiber_trace_3d import fiber_uv_layout as uv


def evidence():
    r=np.linspace(1000.,20000.,20)
    lo,hi=np.quantile(r,[.001,.999])
    return dict(radius=r,family=np.arange(20)%2,
        initial_k=np.clip(5+85*(r-lo)/max(hi-lo,1),0,99).astype(np.float32),
        width_initial=np.maximum(2*np.pi*np.interp(np.arange(100),
            np.linspace(5,90,100),np.linspace(lo,hi,100)),100.).astype(np.float32))


def test_historical_initialization_identical():
    d=evidence();state,w=uv.fresh_initial_state(d,dict(layout_mode='shared_anchor'))
    old=np.zeros((20,303),np.float32)
    old[:,:100]=np.maximum(-14.,-.5*((np.arange(100)[None,:]-d['initial_k'][:,None])/3.)**2)
    old[:,100]=logsumexp(old[:,:100],axis=1)+np.log(.01/.99)
    np.testing.assert_array_equal(state,old)
    np.testing.assert_array_equal(w,d['width_initial'])


def test_fresh151_geometry_and_probability():
    d=evidence()
    # Deliberately poisoned cached priors: new dimensions must derive from radius.
    d['initial_k'][:]=np.nan;d['width_initial'][:]=np.nan
    state,w=uv.fresh_initial_state(d,dict(layout_mode='shared_anchor',winding_count=151))
    assert state.shape==(20,456) and w.shape==(151,)
    assert np.isfinite(state).all() and np.isfinite(w).all()
    assert np.all(state[:,152:]==0)
    p=softmax(state[:,:152],axis=1)
    np.testing.assert_allclose(p[:,-1],.01,rtol=1e-6)
    assert (p>0).all() and p[:,:151].argmax(1).max()>130
    assert w[-1]>w[0]>100


def test_fresh151_geometry_and_widths_trainable():
    state,w=uv.fresh_initial_state(evidence(),dict(layout_mode='shared_anchor',winding_count=151))
    s=torch.tensor(state,requires_grad=True);raw=torch.nn.Parameter(uv.inverse_widths(torch.tensor(w)))
    opt=torch.optim.Adam([s,raw],lr=.01)
    widths,bounds=uv.width_boundaries(raw)
    u,v,q=uv.coordinates(s,torch.full((20,),.2),torch.zeros(20),
        torch.arange(20)%2,torch.ones(20),widths,bounds)
    loss=((u[:,50]/1e5-2)**2+(v[:,50]/1000-1)**2).mean()
    loss.backward();opt.step()
    assert not torch.equal(s[:,152:],torch.from_numpy(state)[:,152:])
    assert not torch.equal(uv.width_boundaries(raw)[0],torch.from_numpy(w))


@pytest.mark.parametrize('config',[{'winding_count':1},{'initial_sigma':0}])
def test_invalid_initialization(config):
    with pytest.raises(ValueError):uv.fresh_initial_state(evidence(),config)
