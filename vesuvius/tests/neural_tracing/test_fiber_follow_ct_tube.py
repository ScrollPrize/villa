from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow import data as D
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, arclength, crop_local_grid, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.supervision import tube_loss, teacher_candidates, loss_fn
from vesuvius.neural_tracing.fiber_follow.tube import tube_geometry, render_tube
from vesuvius.neural_tracing.fiber_follow.volume import FiberVolume, FiberVolumeSpec
from vesuvius.neural_tracing.fiber_follow.trace import field_axis, ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.train import save_checkpoint, load_checkpoint


def array_at(path, values):
    path.mkdir(parents=True)
    (path/'.zarray').write_text(json.dumps(dict(shape=list(values.shape), chunks=list(values.shape),
        dtype='|u1', fill_value=0, order='C', filters=None, compressor=None, zarr_format=2)))
    (path/'0.0.0').write_bytes(values.astype(np.uint8).tobytes())


def ct_volume(tmp_path):
    presence = tmp_path/'fields/test_presence.ome.zarr/3'
    p = np.zeros((8, 8, 8), np.uint8)
    p[4, 4, :] = 255
    array_at(presence, p)
    z,y,x = np.indices((16,16,16))
    array_at(tmp_path/'ct/0', x+2*y+3*z)
    return FiberVolume(FiberVolumeSpec(str(tmp_path/'fields'), ct_zarr=str(tmp_path/'ct'),
                       ct_level=0, ct_grid_scale=4., inputs='ct'), cache_bytes=1<<20)


def test_native_ct_sampling_matches_source_coordinates_in_both_samplers(tmp_path, monkeypatch):
    vol = ct_volume(tmp_path)
    assert vol.nx is None and vol.ny is None
    assert vol.input_scale == 2 and vol.channels == vol.raw_channels == 1
    # No direction arrays exist. Presence becomes unavailable after initialization.
    heading, strength = field_axis(vol, np.array([4.,4.,4.]))
    assert abs(heading[0]) > .999 and strength == 1
    monkeypatch.setattr(vol.presence, 'read', lambda *args: pytest.fail('presence read during model sampling'))
    crop = CropSpec(depth=4, width=5, behind=1, spacing=.5)
    item = dict(pos=np.array([4.,4.,4.]), frame=np.eye(3), hist_local=np.zeros((4,3)), hmask=np.zeros(4))
    grid = torch.from_numpy(crop_local_grid(crop)).float()
    expected_points = 2*(item['pos']+grid.numpy())
    expected = (expected_points[...,0]+2*expected_points[...,1]+3*expected_points[...,2])/255
    outputs = []
    for fused in (False, True):
        monkeypatch.setattr(D, 'FUSED_SAMPLER', fused)
        batch = D.collate_with_volume([item], vol, crop, grid)
        assert batch['x'].shape == (1,2,4,5,5)
        np.testing.assert_allclose(batch['x'][0,0], expected, atol=2e-4)
        assert not batch['x'][0,1].any()
        outputs.append(batch['x'])
    torch.testing.assert_close(*outputs, atol=2e-4, rtol=0)


def test_ct_scale_mismatch_is_rejected(tmp_path):
    vol = ct_volume(tmp_path)
    with pytest.raises(ValueError, match='do not align'):
        FiberVolume(replace(vol.spec, ct_grid_scale=8.))


def test_ct_diagnostics_use_ct_and_trace_units(tmp_path):
    vol = ct_volume(tmp_path)
    assert vol.sample_image_nearest(np.array([[4.,4.,4.]]))[0] == 48


def test_gaussian_tube_matches_exact_segment_distance_between_sparse_vertices():
    crop = CropSpec(depth=9, width=9, behind=4)
    points = np.array([[0.,0.,-20.],[0.,0.,20.]])
    item = tube_geometry(points, np.zeros(3), np.eye(3), crop, 1., (False,False))
    target, mask = render_tube(item,crop,1.)
    assert mask.all()
    np.testing.assert_allclose(target[:,4,4],1.)
    np.testing.assert_allclose(target[:,4,5],np.exp(-.5),rtol=1e-6)
    np.testing.assert_allclose(target[:,4,6],np.exp(-2),rtol=1e-6)


def test_gaussian_tube_retains_turns_and_multiple_crossings():
    crop = CropSpec(depth=9,width=9,behind=4)
    points = np.array([[-2.,0.,-3.],[-2.,0.,2.],[2.,0.,2.],[2.,0.,-3.]])
    item = tube_geometry(points,np.zeros(3),np.eye(3),crop,.5,(True,True))
    target,mask = render_tube(item,crop,.5)
    assert target[4,4,2] == target[4,4,6] == target[6,4,4] == 1
    assert target[4,4,4] < .001
    assert mask.all()


@pytest.mark.parametrize('physical',[False,True])
def test_tube_masks_unknown_annotation_end_and_preserves_physical_end(physical):
    crop=CropSpec(depth=9,width=9,behind=4)
    item=tube_geometry(np.array([[0.,0.,-20.],[0.,0.,1.]]),np.zeros(3),np.eye(3),crop,1.,(False,physical))
    target,mask=render_tube(item,crop,1.)
    assert mask[5,4,4] == 1
    assert mask[6,4,4] == physical
    assert target[6,4,4] == pytest.approx(np.exp(-.5))
    logits=torch.zeros((1,*target.shape),requires_grad=True)
    loss=tube_loss(logits,torch.from_numpy(target[None]),torch.from_numpy(mask[None]))
    loss.backward()
    assert (logits.grad[0,6,4,4] != 0).item() == physical


def test_departed_tube_has_no_position_gradient():
    crop=CropSpec(depth=9,width=9,behind=4)
    item=tube_geometry(np.array([[0.,0.,-20.],[0.,0.,20.]]),np.zeros(3),np.eye(3),crop,1.,(False,False),True)
    target,mask=render_tube(item,crop,1.)
    logits=torch.zeros((1,*target.shape),requires_grad=True)
    tube_loss(logits,torch.from_numpy(target[None]),torch.from_numpy(mask[None])).backward()
    assert not logits.grad.any()


def test_tube_target_requires_matching_holdout():
    item=dict(pos=np.array([0.,0.,0.]), frame=np.eye(3), tube_segments=np.array([[[0.,0.,110.],[1.,0.,110.]]]))
    assert not D.training_state_allowed(item,CropSpec(depth=4,width=5,behind=1),D.ZBand(100,120))


@pytest.mark.parametrize('history_render', ['points', 'segments'])
def test_ct_tube_forward_backward_checkpoint_and_tracer_use_same_inputs(tmp_path,monkeypatch,history_render):
    torch.set_num_threads(1)
    vol=ct_volume(tmp_path)
    crop=CropSpec(depth=12,width=9,behind=2,spacing=.5,history_render=history_render,history_sigma=.35)
    cfg=FollowNetConfig(in_channels=2,depth=12,width=9,behind=2,spacing=.5,
        widths=(8,16),hidden=16,n_future=4,future_step=1.,hist_points=2,hist_stride=2,
        heat_bins=7,heat_spacing=.5,n_candidates=2,norm='group',heatmap_target='tube',tube_sigma=.35)
    sample=D.SampleConfig(crop=crop,n_history=4,n_future=4,future_step=1.,n_candidates=2,
                          heatmap_target='tube',tube_sigma=.35)
    points=np.array([[x,4.,4.] for x in np.linspace(0,7,15)])
    fiber=D.TracedFiber('synthetic',points,arclength(points),'')
    pos=np.array([4.,4.,4.]); frame=frame_from_heading(np.array([1.,0.,0.]))
    item=D.label_state(fiber,pos,frame,np.array([[3.,4.,4.],[2.,4.,4.],[1.,4.,4.],[0.,4.,4.]]),np.ones(4),sample,t=4.,reverse=False)
    grid=torch.from_numpy(crop_local_grid(crop)).float()
    batch=D.collate_with_volume([item],vol,crop,grid)
    model=FollowNet(cfg)
    output=model(batch['x'].float(),batch['hist'],batch['hmask'],teacher_candidates(batch,cfg))
    assert output['tube_logits'].shape == (1,12,9,9)
    loss,metrics=loss_fn(output,batch,cfg)
    loss.backward()
    assert torch.isfinite(loss)
    for layer in (model.heat_head,model.encoders[0][0],model.rank_head,model.confidence_head):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
    checkpoint=tmp_path/'tube.pt'
    save_checkpoint(checkpoint,model,vol.spec,sample)
    loaded,loaded_crop,nh,spec,_=load_checkpoint(checkpoint,'cpu')
    assert loaded.cfg==cfg and loaded_crop==crop and spec==vol.spec and nh==4
    captured=[]
    hook=loaded.register_forward_pre_hook(lambda module,args: captured.append(args[0].detach().clone()))
    monkeypatch.setattr(vol.presence,'read',lambda *args:pytest.fail('presence read during rollout'))
    tracer=ModelTracer(loaded,vol,crop,4,TraceParams(max_len=1),device='cpu')
    try:
        tracer.trace(pos[None],np.array([[1.,0.,0.]]), histories=[np.array([[0.,4.,4.],[1.,4.,4.],[2.,4.,4.],[3.,4.,4.]])])
    finally:
        tracer.close(); hook.remove()
    torch.testing.assert_close(captured[0],batch['x'].float(),atol=3e-4,rtol=0)
