from dataclasses import replace
import numpy as np
import pytest
import torch
from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec, crop_local_grid, render_history, arclength
from vesuvius.neural_tracing.fiber_follow.fast_sample import sample_crop
from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, TracedFiber, make_sample


@pytest.mark.parametrize('mask', [[1,1,1],[1,0,1],[0,1,1],[0,0,0]])
def test_connected_history_torch_matches_fused_sampler(mask):
    crop = CropSpec(depth=17, width=9, behind=12, spacing=.5, history_render='segments', history_sigma=.35)
    grid = crop_local_grid(crop)
    hist = np.array([[0.,0.,-1.],[1.,0.,-3.],[0.,0.,-5.]])
    actual = render_history(torch.tensor(hist[None]),torch.tensor([mask]),torch.tensor(grid),.35,'segments')[0,0]
    fused = sample_crop(np.zeros((1,16,16,16),np.uint8),np.zeros(3),np.ones(3)*8,np.eye(3),
                        grid.reshape(-1,3),False,hist,np.array(mask),2,.35,'segments')[-1].reshape(actual.shape)
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


def test_zero_jitter_retains_only_smooth_history_drift():
    points=np.array([[0.,0.,0.],[0.,0.,200.]])
    fiber=TracedFiber('line',points,arclength(points),'')
    cfg=SampleConfig(n_history=64, history_jitter=0,history_wobble=0,no_history_prob=0)
    smooth=make_sample(fiber,100,False,cfg,np.random.default_rng(31))
    jitter=make_sample(fiber,100,False,replace(cfg,history_jitter=.25),np.random.default_rng(31))
    np.testing.assert_allclose(np.diff(smooth['hist_local'],n=2,axis=0),0,atol=1e-12)
    assert np.abs(np.diff(jitter['hist_local'],n=2,axis=0)).max()>.1
    wobble=make_sample(fiber,100,False,replace(cfg,history_wobble=1),np.random.default_rng(31))
    assert not np.allclose(smooth['hist_local'],wobble['hist_local'])
    assert np.abs(np.diff(wobble['hist_local'],n=2,axis=0)).max()<.2


def test_legacy_crop_keeps_replay_provenance_compatible():
    legacy=dict(depth=64,width=64,behind=16,spacing=.5,gate_direction=False)
    assert CropSpec(**legacy).replay_dict()==legacy
    crop=CropSpec(**legacy,history_render='segments',history_sigma=.35)
    assert CropSpec(**crop.replay_dict())==crop
