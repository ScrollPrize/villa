import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_trace_3d import fiber_implicit_winding as density
from vesuvius.neural_tracing.fiber_trace_3d import fiber_gt_winding_reference as ref


def test_only_global_offset_is_removed(monkeypatch):
    d=dict(i=np.array([0,0]),j=np.array([1,1]),ds_a=np.zeros(2),ds_b=np.ones(2),
           source_id=np.array([1,3]),phase_a=np.array([-5.,-4.]),phase_b=np.zeros(2))
    r=dict(fiber_offsets={'1':9,'3':4},excluded_source_ids=[7],events=[
        dict(index=e,source_a=1,source_b=3,phase_a=float(d['phase_a'][e]),phase_b=0.) for e in range(2)])
    s=dict(state=np.zeros((2,4)),widths=np.ones(10))
    def points(d,s,st,w,ids,ds,shared_only):
        return np.where(ds==0,np.array([4.,5.]),4.)+12,None,None
    monkeypatch.setattr(ref,'saved_points',points)
    result=ref.compare(d,s,r)
    assert result['global_shift']==12
    assert result['error_windings']['max']==0
    d['phase_a'][0]=-.3
    with pytest.raises(ValueError,match='mismatch'):ref.compare(d,s,r)


def test_distribution_center_is_not_phase_or_realized_position():
    state=torch.tensor([[5.5,2.3,0.,100.]])
    h,lp=density.hypotheses(state,.5)
    labels=h[0,:,0];p=lp[0].exp()
    torch.testing.assert_close(p[labels==5],p[labels==6])
    modes=h[0,(labels==5)|(labels==6)]
    torch.testing.assert_close(modes[:,0]+modes[:,1],torch.tensor([7.3,8.3]))
    torch.testing.assert_close(density.map_state(state)[:,0],torch.tensor([5.]))
    torch.testing.assert_close(torch.remainder(modes[:,1],1),torch.tensor([.3,.3]))


def test_reference_uses_integer_label(monkeypatch):
    d=dict(i=np.array([0]),j=np.array([1]),ds_a=np.zeros(1),ds_b=np.zeros(1),
           source_id=np.array([1,3]),phase_a=np.array([2.3]),phase_b=np.array([2.3]))
    r=dict(fiber_offsets={'1':5,'3':6},excluded_source_ids=[],events=[
        dict(index=0,source_a=1,source_b=3,phase_a=2.3,phase_b=2.3)])
    s=dict(state=np.array([[5.5,2.3,0.,100.],[6.1,2.3,0.,100.]]),widths=np.ones(20))
    def points(d,s,st,w,ids,ds,shared_only):return st[ids,0]+st[ids,1],st[ids,1],st[ids,2]
    monkeypatch.setattr(ref,'saved_points',points)
    result=ref.compare(d,s,r)
    assert result['realized_labels']==[5.,6.]
    assert result['error_windings']['max']==0
