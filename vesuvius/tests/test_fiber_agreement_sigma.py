import numpy as np
import torch
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_agreement_sigma import (
    evidence_moments,combine_evidence,source_moments,AgreementSigma)


def combine(means,variances):
    m=torch.tensor(means,dtype=torch.float64);v=torch.tensor(variances,dtype=torch.float64)
    return combine_evidence(evidence_moments(m,v,torch.ones_like(m)).sum(0),m.new_tensor(0),4.,8)


def test_agreement_and_conflict():
    assert combine([0],[1.25])[1]==pytest.approx(1.25)
    assert combine([0,0],[1.25,1.25])[1]==pytest.approx(.625)
    assert combine([-2,2],[1.25,1.25])[1]==pytest.approx(4.)
    assert combine([0,0,0,0],[1.25]*4)[1]<combine([0,0],[1.25]*2)[1]


def test_duplicates_siblings_and_disabled_evidence():
    def run(src,m,r):
        m=torch.tensor(m,dtype=torch.float64)
        total,_,_=source_moments(torch.zeros(len(m),dtype=torch.long),torch.tensor(src),m,
            torch.ones_like(m),torch.tensor(r,dtype=torch.float64),4)
        return combine_evidence(total,torch.zeros(4),4.,8)
    a=run([1],[0],[1]);b=run([1,1],[0,0],[1,1])
    torch.testing.assert_close(a[1],b[1])
    assert run([1,1],[-2,2],[1,1])[1][0]>a[1][0]
    assert run([1,2],[0,100],[1,0])[1][0]==a[1][0]
    assert run([1],[0],[0])[1][0]==4


def test_live_continuation_and_roundtrip(monkeypatch):
    # Same endpoint geometry API as the solve, with a simple analytic fixture.
    class Geometry:
        def __init__(self,ext,state,widths,edges,keys):pass
        def points_both(self,a,b):
            return ((a[:,:,0]+a[:,:,1],a[:,:,1],a[:,:,2]),
                    (b[:,:,0]+b[:,:,1],b[:,:,1],b[:,:,2]))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_trace_3d.fiber_intermediate_uv.SharedPairGeometry',Geometry)
    d=dict(source_id=np.array([1,1,2]),family=np.array([0,0,1]),i=np.array([1]),j=np.array([2]),
        ds_a=np.zeros(1),ds_b=np.zeros(1),sign=np.ones(1),distance=np.array([.5]),parallel=np.array([False]),
        links=np.array([[0,1,0,0]]))
    cfg=dict(winding_sigma=.5,sigma_max=4,cavity_damping=1,cavity_update_every=1,cavity_constraint_sigma=.25)
    data={k:torch.tensor(v) for k,v in d.items() if k!='source_id'}
    state=torch.tensor([[0.,0,0,100],[0.,0,0,100],[0.,0,0,100]],dtype=torch.float64)
    ext=AgreementSigma(d,cfg,'cpu');ext.update(state,torch.ones(20),data,0,extension=object())
    assert len(ext.src)==4 # Both measured edge and continuation, both directions.
    assert len(ext.values)==3 and not ext.values.requires_grad
    saved=ext.arrays();other=AgreementSigma(d,cfg,'cpu');other.restore(saved)
    ext.update(state,torch.ones(20),data,1,extension=object())
    other.update(state,torch.ones(20),data,1,extension=object())
    torch.testing.assert_close(ext.values,other.values)
    # No return echo on an isolated two-piece chain: sigma doesn't collapse.
    d['i']=np.array([],dtype=int);d['j']=np.array([],dtype=int)
    for key in ('ds_a','ds_b','sign','distance','parallel'):d[key]=d[key][:0]
    data={k:torch.tensor(v) for k,v in d.items() if k!='source_id'}
    ext=AgreementSigma(d,cfg,'cpu')
    for it in range(10):ext.update(state,torch.ones(20),data,it,extension=object())
    assert ext.values[0]>1 and ext.values[1]>1
    before=ext.shift.clone();state[1,1]=1
    ext.update(state,torch.ones(20),data,10,extension=object())
    assert not torch.equal(before,ext.shift)
