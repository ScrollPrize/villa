from dataclasses import replace


import copy


import math


import numpy as np


import pytest


import torch


from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, TracedFiber, continuation_targets


from vesuvius.neural_tracing.fiber_follow.geometry import arclength, frame_from_heading


from vesuvius.neural_tracing.fiber_follow.model import (
    FollowNet, FollowNetConfig, flow_targets, observable_half_width, prior_mean,
)


from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn


from vesuvius.neural_tracing.fiber_follow.train import fit_flow_sigma, update_ema


def config():
    return FollowNetConfig(in_channels=3, depth=24, width=17, behind=6, spacing=.5,
                           widths=(8, 16), hidden=16, n_future=8, future_step=1.,
                           hist_points=4, hist_stride=2, recent_history_points=8, flow_sigma=((1., 1.),)*8,
                           flow_stencil_radius=.5,  # leaves 3.5 observable voxels in the 4-voxel half-width
                           flow_layers=1, flow_heads=2, flow_steps=3, flow_draws=2, norm='group')


def inputs(cfg):
    x = torch.randn(1, 3, cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, cfg.hist_points*cfg.hist_stride, 3)
    hist[..., 2] = -torch.arange(1, hist.shape[1]+1)
    return x, hist, torch.ones(hist.shape[:2])


def test_follower_labels_do_not_construct_gaussian_targets():
    cfg = SampleConfig(n_future=8, future_step=1.)
    points = np.array([[float(x), 0., 0.] for x in range(30)])
    fiber = TracedFiber('line', points, arclength(points), '')
    out = continuation_targets(fiber, 10., False, np.array([10., 0., 0.]),
                               frame_from_heading(np.array([1., 0., 0.])), cfg)
    assert not any('tube' in k or 'heat' in k for k in out)
    assert out['plane_mask'].sum() == 8


def test_prior_is_straight_ahead_whatever_the_history_says():
    cfg = config()
    _, hist, mask = inputs(cfg)
    hist[..., 0] = .5*hist[..., 2]
    hist[..., 1] = -.25*hist[..., 2]
    planes = cfg.future_step*torch.arange(1, cfg.n_future+1)
    for supplied in (mask, torch.zeros_like(mask)):
        mu = prior_mean(hist, supplied, cfg)
        assert mu.shape == (1, cfg.n_future, 3)
        assert not mu[..., :2].any()
        torch.testing.assert_close(mu[0, :, 2], planes)


def test_flow_targets_censor_the_prefix_once_the_curve_leaves_the_crop():
    cfg = config()
    half = observable_half_width(cfg)
    assert half == pytest.approx(3.5)
    batch = targets(cfg)
    batch['plane_ab'][0, 4, 0] = half+.5   # leaves at plane 5
    batch['plane_ab'][0, 5:, 0] = 0.       # and re-enters
    x1, mask, censored = flow_targets(batch, cfg)
    assert mask[0].tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert censored[0].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    torch.testing.assert_close(x1[0, :, 2], cfg.future_step*torch.arange(1, cfg.n_future+1))
    # Unannotated planes never break the observable prefix, whatever their padding holds.
    gap = targets(cfg)
    gap['plane_mask'][0, 2] = 0
    gap['plane_ab'][0, 2] = float('nan')
    _, mask, censored = flow_targets(gap, cfg)
    assert mask[0].tolist() == [1, 1, 0, 1, 1, 1, 1, 1] and not censored.any()
    # Departed states are neither known nor censored.
    gone = targets(cfg)
    gone['offtrack'].fill_(1)
    gone['plane_ab'][0, 1, 0] = 100.
    _, mask, censored = flow_targets(gone, cfg)
    assert not mask.any() and not censored.any()
    # The loss report and the scale calibration apply the same rule.
    model = FollowNet(cfg)
    x, hist, hmask = inputs(cfg)
    row = targets(cfg, hist, hmask)
    row['plane_ab'][0, 4, 0] = half+.5
    out = model(x, hist, hmask, targets=row)
    assert out['flow_known_fraction'].item() == .5 and out['flow_censored_fraction'].item() == .5
    clean = targets(cfg, hist, hmask)
    rows = {k: torch.cat([clean[k], clean[k], row[k]]) for k in ('hist', 'hmask', 'plane_ab', 'plane_mask', 'offtrack')}
    sigma, report = fit_flow_sigma(iter([rows]), cfg, states=3)
    assert report['known_counts'] == [3]*4+[2]*4 and report['censored_counts'] == [0]*4+[1]*4
    assert len(sigma) == cfg.n_future
    with pytest.raises(ValueError, match='fit crop'):
        FollowNet(replace(cfg, flow_stencil_radius=(cfg.width-1)*cfg.spacing/2))


def test_scale_calibration_uses_masked_history_residuals_and_voxel_floor():
    cfg = config()
    _, hist, mask = inputs(cfg)
    hist = hist.expand(6, -1, -1).clone()
    mask = mask.expand(6, -1).clone()
    hist[..., 0] = hist[..., 2] * torch.arange(6)[:, None]*.02
    mu = prior_mean(hist, mask, cfg)[..., :2]
    # Residuals stay inside the observable extent; the floor still binds on small scales.
    scales = torch.stack([torch.linspace(.25, 1.5, 8), torch.full((8,), .1)], -1)
    residual = torch.tensor([-2., -1., 1., 2., 1000., 1e6])[:, None, None]*scales
    batch = dict(hist=hist, hmask=mask, plane_ab=mu+residual,
                 plane_mask=torch.ones(6, 8), offtrack=torch.tensor([0, 0, 0, 0, 1, 0]))
    batch['plane_mask'][0, -1] = 0
    batch['plane_ab'][0, -1] = float('nan')
    # The sixth row must not enter calibration; the fifth is an offtrack row.
    sigma, report = fit_flow_sigma(iter([batch]), cfg, states=5)
    expected = residual[:4].double().std(0, correction=0).clamp_min(1)
    expected[-1] = residual[1:4, -1].double().std(0, correction=0).clamp_min(1)
    torch.testing.assert_close(torch.tensor(sigma).double(), expected, atol=1e-6, rtol=1e-6)
    assert report['states'] == 5 and report['known_counts'] == [4]*7+[3]
    assert report['censored_counts'] == [0]*8
    assert (torch.tensor(sigma)[:, 0] == 1).sum() > 0 and (torch.tensor(sigma)[:, 0] > 1).sum() > 0
    batch['plane_mask'][:, -1] = 0
    with pytest.raises(ValueError, match='at least two known targets'):
        fit_flow_sigma(iter([batch]), cfg, states=5)



def targets(cfg, hist=None, hmask=None):
    q = (cfg.n_future-1)*4+1
    z = torch.linspace(1, cfg.n_future, q)
    ab = torch.stack([.04*z*z, .1*z], -1)[None]
    if hist is None:
        _, hist, hmask = inputs(cfg)
    gt_history = torch.zeros(1, cfg.recent_history_points+1, 3)
    gt_history[0, :, 2] = -torch.arange(cfg.recent_history_points+1)
    return dict(dense_ab=ab, dense_mask=torch.ones(1, q), offtrack=torch.zeros(1),
                plane_ab=ab[:, ::4], plane_mask=torch.ones(1, cfg.n_future),
                gt_history=gt_history, gt_history_mask=torch.ones(1, cfg.recent_history_points+1),
                hist=hist, hmask=hmask, endpoint_known=torch.zeros(1), end_local=torch.zeros(1, 3))
