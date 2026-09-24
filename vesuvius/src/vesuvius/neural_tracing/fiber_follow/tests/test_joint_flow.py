"""Contracts of the joint flow generator: pinning, censoring, learning, selection, isolation."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, TracedFiber, continuation_targets
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.model import FollowNet, FollowNetConfig, select_candidates
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, teacher_candidates


def config():
    return FollowNetConfig(in_channels=3, depth=24, width=17, behind=6, spacing=.5,
                           widths=(8, 16), hidden=16, n_future=8, future_step=1.,
                           hist_points=4, hist_stride=2, clean_points=4, n_candidates=3,
                           flow_layers=1, flow_heads=2, flow_steps=3, flow_samples=6, flow_draws=2, norm='group')


def inputs(cfg):
    x = torch.randn(1, 3, cfg.depth, cfg.width, cfg.width)
    hist = torch.zeros(1, cfg.hist_points*cfg.hist_stride, 3)
    hist[..., 2] = -torch.arange(1, hist.shape[1]+1)
    return x, hist, torch.ones(hist.shape[:2])


def targets(cfg, hist=None, hmask=None):
    q = (cfg.n_future-1)*4+1
    z = torch.linspace(1, cfg.n_future, q)
    ab = torch.stack([.04*z*z, .1*z], -1)[None]
    if hist is None:
        _, hist, hmask = inputs(cfg)
    clean = torch.zeros(1, cfg.clean_points+1, 3)
    clean[0, :, 2] = -torch.arange(cfg.clean_points+1)
    return dict(dense_ab=ab, dense_mask=torch.ones(1, q), offtrack=torch.zeros(1),
                plane_ab=ab[:, ::4], plane_mask=torch.ones(1, cfg.n_future),
                clean_local=clean, clean_mask=torch.ones(1, cfg.clean_points+1),
                hist=hist, hmask=hmask, endpoint_known=torch.zeros(1), end_local=torch.zeros(1, 3),
                replay_valid=torch.zeros(1, cfg.n_candidates),
                replay_candidates=torch.zeros(1, cfg.n_candidates, cfg.n_future, 3))


def test_samples_pin_future_planes_and_outputs_have_scorer_shapes():
    torch.manual_seed(1)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    out = model(x, hist, mask, generator=torch.Generator().manual_seed(0))
    P = 1+cfg.clean_points+cfg.n_future
    assert out['samples'].shape == (1, cfg.flow_samples, P, 3)
    assert out['candidates'].shape == (1, cfg.n_candidates, cfg.n_future, 3)
    assert out['clean_history'].shape == (1, cfg.clean_points+1, 3)
    assert out['ranks'].shape == (1, cfg.n_candidates) and out['confidence'].shape == (1, cfg.n_candidates, cfg.n_future)
    assert out['candidate_support'].shape == (1, cfg.n_candidates)
    planes = cfg.future_step*torch.arange(1, cfg.n_future+1)
    torch.testing.assert_close(out['candidates'][..., 2], planes.expand(1, cfg.n_candidates, -1))
    torch.testing.assert_close(out['samples'][:, :, 1+cfg.clean_points:, 2], planes.expand(1, cfg.flow_samples, -1))
    assert torch.isfinite(out['samples']).all()
    assert (out['confidence'][..., 1:] <= out['confidence'][..., :-1]+1e-6).all()
    assert 'flow_loss' not in out


def test_sampling_is_reproducible_with_a_seeded_generator():
    torch.manual_seed(2)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    a = model(x, hist, mask, generator=torch.Generator().manual_seed(5))['samples']
    b = model(x, hist, mask, generator=torch.Generator().manual_seed(5))['samples']
    c = model(x, hist, mask, generator=torch.Generator().manual_seed(6))['samples']
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_flow_loss_reaches_image_history_and_flow_but_scorer_never_trains_the_flow():
    torch.manual_seed(3)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    out = model(x, hist, mask, teacher_candidates(batch, cfg), targets=batch)
    assert out['flow_loss'].requires_grad
    out['flow_loss'].backward()
    for parameter in (model.flow.velocity.weight, model.flow.input[0].weight, model.encoders[0][0].weight,
                      model.history[0].weight, model.flow.context.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert model.rank_head.weight.grad is None and model.confidence_head.weight.grad is None
    model.zero_grad()
    out = model(x, hist, mask, teacher_candidates(batch, cfg), targets=batch)
    loss, metrics = loss_fn(out, batch, cfg, flow_weight=0.)
    loss.backward()
    assert model.rank_head.weight.grad.abs().sum() > 0
    assert model.flow.velocity.weight.grad is None or not model.flow.velocity.weight.grad.any()
    assert model.flow.input[0].weight.grad is None or not model.flow.input[0].weight.grad.any()
    assert {'flow', 'flow_past', 'flow_future', 'candidate_support', 'oracle_recall', 'clean_current_error'} <= metrics.keys()


@pytest.mark.parametrize('case', ['unknown', 'offtrack'])
def test_unlabeled_states_have_no_flow_loss(case):
    torch.manual_seed(4)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    if case == 'unknown':
        batch['plane_mask'].zero_()
        batch['clean_mask'].zero_()
    else:
        batch['offtrack'].fill_(1)
    out = model(x, hist, mask, targets=batch)
    assert out['flow_loss'].item() == 0 and out['flow_known_fraction'].item() == 0
    out['flow_loss'].backward()
    assert not model.flow.velocity.weight.grad.any()


def test_unknown_future_tail_and_absent_history_are_censored():
    torch.manual_seed(5)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    mask[:, 2:] = 0  # history supplied for two points only
    batch = targets(cfg, hist, mask)
    batch['plane_mask'][:, 3:] = 0
    x1, token_mask = model.polyline_targets(batch)
    n = cfg.clean_points
    assert token_mask[0, :n+1].tolist() == [1, 1, 1, 0, 0]
    assert token_mask[0, n+1:].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    torch.testing.assert_close(x1[0, n+1:, 2], cfg.future_step*torch.arange(1, cfg.n_future+1))
    torch.testing.assert_close(x1[0, :n+1], batch['clean_local'][0])
    # Loss and gradient reflect only the known tokens.
    out = model(x, hist, mask, targets=batch)
    assert 0 < out['flow_known_fraction'].item() < 1 and out['flow_loss'].item() > 0


def test_prior_uses_observed_history_where_supplied_and_pins_future_planes():
    cfg = config()
    model = FollowNet(cfg)
    _, hist, mask = inputs(cfg)
    hist[:, :, 0] = 2.
    mask[:, 3:] = 0
    mu, valid = model.flow.prior(hist, mask)
    n = cfg.clean_points
    torch.testing.assert_close(mu[0, 0], torch.zeros(3))
    torch.testing.assert_close(mu[0, 1:4], hist[0, :3])
    torch.testing.assert_close(mu[0, 4], torch.tensor([0., 0., -4.]))
    torch.testing.assert_close(mu[0, n+1:, 2], cfg.future_step*torch.arange(1, cfg.n_future+1))
    assert not mu[0, n+1:, :2].any()
    assert valid[0].tolist() == [1, 1, 1, 1, 0] + [0]*cfg.n_future


def test_flow_optimization_learns_a_curved_continuation():
    torch.manual_seed(21)
    cfg = replace(config(), flow_draws=8)
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    losses = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(150):
        optimizer.zero_grad()
        out = model(x, hist, mask, targets=batch, generator=generator)
        losses.append(out['flow_loss'].item())
        out['flow_loss'].backward()
        optimizer.step()
    assert np.mean(losses[-5:]) < np.mean(losses[:5])*.6
    model.eval()
    out = model(x, hist, mask, generator=torch.Generator().manual_seed(1))
    error = (out['candidates'][..., :2] - batch['plane_ab'][:, None]).norm(dim=-1).mean(-1)
    assert error.min() < 1.5  # some mode follows the curve after a short fit


def test_select_candidates_returns_distinct_modes_with_support():
    paths = torch.zeros(1, 8, 6, 3)
    paths[0, :5, :, 0] = torch.tensor([0., .2, -.2, .1, -.1])[:, None]   # tight cluster at u=0
    paths[0, 5:7, :, 0] = torch.tensor([4., 4.3])[:, None]                # second mode at u=4
    paths[0, 7, :, 0] = 10.                                               # isolated outlier
    index, support = select_candidates(paths, 3, 1.5, 4)
    picks = paths[0, index[0], 0, 0].tolist()
    assert picks[0] == 0. and picks[1] in (4., 4.3) and picks[2] == 10.
    assert support[0].tolist() == pytest.approx([5/8, 2/8, 1/8])
    # Fewer modes than slots: remaining slots are distinct real samples, farthest from the selection.
    index, _ = select_candidates(paths[:, :5], 3, 1.5, 4)
    assert len(set(index[0].tolist())) == 3
    with pytest.raises(ValueError):
        select_candidates(paths[:, :2], 3, 1.5, 4)


def test_follower_labels_do_not_construct_gaussian_targets():
    cfg = SampleConfig(n_future=8, future_step=1.)
    points = np.array([[float(x), 0., 0.] for x in range(30)])
    fiber = TracedFiber('line', points, arclength(points), '')
    out = continuation_targets(fiber, 10., False, np.array([10., 0., 0.]),
                               frame_from_heading(np.array([1., 0., 0.])), cfg)
    assert not any('tube' in k or 'heat' in k for k in out)
    assert out['plane_mask'].sum() == 8


def test_config_validation():
    with pytest.raises(ValueError):
        FollowNet(replace(config(), flow_samples=2))
    with pytest.raises(ValueError):
        FollowNet(replace(config(), hidden=15))
    with pytest.raises(ValueError):
        FollowNet(replace(config(), prior_scale=0.))


def test_beam_owns_its_optional_head_without_flow_parameters():
    from vesuvius.neural_tracing.fiber_follow.beam.model import BeamNetConfig, BeamRankNet
    cfg = BeamNetConfig(**config().to_dict())
    model = BeamRankNet(cfg)
    assert not hasattr(model, 'flow')
    x, hist, mask = inputs(cfg)
    paths = torch.randn(1, 2, 8, 3)
    out = model(x, hist, mask, paths, torch.ones(1, 2, 8), torch.zeros(1, 2))
    assert out['ranks'].shape == (1, 2) and out['tube_logits'].shape == x.shape[:1]+x.shape[2:]
