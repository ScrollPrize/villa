import numpy as np
import io
import torch
import pytest
from .test_fiber_affine_geometry import setup
from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv import BucketUV


def test_predicted_only_preserves_shared_geometry_and_learning_rates():
    d,cfg,td,s,w,full,_=setup()
    cfg=dict(cfg,predicted_only_geometry=True,intermediate_uv_geometry=False,
             affine_local_lr_factor=.1,affine_coarse_lr_factor=3.,bucket_local_lr_multiplier=10.)
    full.config.update(cfg)
    e=BucketUV(d,cfg,'cpu');e.prepare_fields(d,{})
    e.update(s,w,td,0)
    assert e.values is None and e.affine is None and e.intermediate is None
    assert all(name.startswith('curve.') for name,_ in e.named_parameters())
    torch.testing.assert_close(e.curve.coordinates(),full.curve.coordinates())
    original={id(g['params'][0]):g['lr'] for g in full.optimizer_groups()}
    expected=[original[id(p)] for p in (full.curve.u,full.curve.v,full.curve.affine.u,full.curve.affine.v)]
    assert [g['lr'] for g in e.optimizer_groups()]==expected
    terms=e.terms(s,w,td,torch.arange(len(td['i'])))
    reference=full.curve.terms(s,w)
    assert terms.keys()==reference.keys()
    for k in terms:torch.testing.assert_close(terms[k],reference[k])
    assert not any('meeting' in k or 'attachment' in k for k in terms)
    sum(terms.values()).backward()
    saved=e.arrays()
    assert 'bucket_uv_positions' not in saved and 'intermediate_uv_scaled_qv' not in saved
    buffer=io.BytesIO();np.savez(buffer,**saved);buffer.seek(0)
    restored=BucketUV(d,cfg,'cpu')
    with np.load(buffer) as checkpoint:restored.restore(checkpoint)
    torch.testing.assert_close(e.curve.coordinates(),restored.curve.coordinates())
    with pytest.raises(ValueError,match='start fresh'):restored.restore(full.arrays())
    h,_=e.local_hypotheses(s)
    assert e.shared_positions(s,w).shape==(len(e.owner),h.shape[1],2)


def test_predicted_only_visualization_uses_saved_shared_positions():
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_bucket_uv_visualization import geometry
    d,cfg,td,s,w,_,_=setup()
    cfg=dict(cfg,predicted_only_geometry=True,intermediate_uv_geometry=False,
             bucket_uv_dynamic_same_sheet=True,bucket_uv_perpendicular_nodes_only=True)
    e=BucketUV(d,cfg,'cpu');e.prepare_fields(d,{})
    e.update(s,w,td,0)
    saved=dict(e.arrays(),state=s.detach().numpy(),widths=w.detach().numpy(),
               winding_sigma=e.cavity.values.detach().numpy())
    buffer=io.BytesIO();np.savez(buffer,**saved);buffer.seek(0)
    with np.load(buffer) as checkpoint:
        before,after,maximum,predicted,actual,_,_=geometry(d,checkpoint,cfg,show_all=True)
    np.testing.assert_array_equal(predicted,actual)
    np.testing.assert_array_equal(maximum,np.zeros(len(s)))
    for left,right in zip(before[1]+before[2],after[1]+after[2]):
        np.testing.assert_array_equal(left,right)
