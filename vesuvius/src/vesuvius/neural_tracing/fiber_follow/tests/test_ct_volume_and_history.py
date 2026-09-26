"""CT-only volume sampling and rendered observed history."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared import data as D
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, TracedFiber, make_sample
from vesuvius.neural_tracing.fiber_follow.shared.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec, arclength, crop_local_grid, render_history
from vesuvius.neural_tracing.fiber_follow.shared.trace import field_axis
from vesuvius.neural_tracing.fiber_follow.shared.volume import FiberVolume, FiberVolumeSpec


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
    assert vol.input_scale == 2 and vol.channels == 1
    # Presence becomes unavailable after initialization.
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


@pytest.mark.parametrize('mask', [[1,1,1],[1,0,1],[0,1,1],[0,0,0]])
def test_connected_history_torch_matches_fused_sampler(mask):
    crop = CropSpec(depth=17, width=9, behind=12, spacing=.5, history_render='segments', history_sigma=.35)
    grid = crop_local_grid(crop)
    hist = np.array([[0.,0.,-1.],[1.,0.,-3.],[0.,0.,-5.]])
    actual = render_history(torch.tensor(hist[None]),torch.tensor([mask]),torch.tensor(grid),.35,'segments')[0,0]
    fused = sample_crop(np.zeros((1,16,16,16),np.uint8),np.zeros(3),np.ones(3)*8,np.eye(3),
                        grid.reshape(-1,3),hist,np.array(mask),2,.35,'segments')[-1].reshape(actual.shape)
    np.testing.assert_allclose(actual,fused,atol=1e-7,rtol=1e-6)
    if not any(mask):
        assert not actual.any()


def test_connected_history_fills_between_sparse_points_and_reaches_current_position():
    grid = torch.tensor([[[[0.,0.,-2.],[.35,0.,-2.],[0.,0.,0.]]]])
    hist=torch.tensor([[[0.,0.,-1.],[0.,0.,-3.]]]);mask=torch.ones((1,2))
    connected=render_history(hist,mask,grid,.35,'segments').flatten()
    torch.testing.assert_close(connected,torch.tensor([1.,np.exp(-.5),1.],dtype=torch.float32))
    assert render_history(hist,mask,grid,.35,'points').flatten()[0]<.02
    # A missing history point must not connect the more distant history to the origin.
    assert render_history(hist,torch.tensor([[0.,1.]]),grid,.35,'segments').flatten()[2]<1e-10


def test_connected_history_does_not_bridge_masked_gaps():
    hist=torch.tensor([[[0.,0.,-1.],[0.,0.,-3.],[0.,0.,-5.]]])
    grid=torch.tensor([[[[0.,0.,-3.]]]])
    assert render_history(hist,torch.tensor([[1.,0.,1.]]),grid,.35,'segments').item()<1e-6
    assert not render_history(hist[:,:0],torch.empty((1,0)),grid,.35,'segments').any()


def test_zero_jitter_and_drift_leave_straight_history():
    points=np.array([[0.,0.,0.],[0.,0.,200.]])
    fiber=TracedFiber('line',points,arclength(points),'')
    cfg=SampleConfig(n_history=64, history_jitter=0,history_wobble=0,history_drift=0,no_history_prob=0)
    smooth=make_sample(fiber,100,False,cfg,np.random.default_rng(31))
    jitter=make_sample(fiber,100,False,replace(cfg,history_jitter=.25),np.random.default_rng(31))
    np.testing.assert_allclose(np.diff(smooth['hist_local'],n=2,axis=0),0,atol=1e-12)
    assert np.abs(np.diff(jitter['hist_local'],n=2,axis=0)).max()>.1
    wobble=make_sample(fiber,100,False,replace(cfg,history_wobble=1),np.random.default_rng(31))
    assert not np.allclose(smooth['hist_local'],wobble['hist_local'])
    assert np.abs(np.diff(wobble['hist_local'],n=2,axis=0)).max()<.2
