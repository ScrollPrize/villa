"""Future flow: fixed observed history, censoring, learning, and all-sample scoring."""
from dataclasses import replace

import copy
import math

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.data import SampleConfig, TracedFiber, continuation_targets
from vesuvius.neural_tracing.fiber_follow.geometry import arclength, frame_from_heading
from vesuvius.neural_tracing.fiber_follow.model import (
    FollowNet, FollowNetConfig, candidate_support, flow_targets, observable_half_width, prior_mean,
)
from vesuvius.neural_tracing.fiber_follow.supervision import loss_fn, teacher_candidates
from vesuvius.neural_tracing.fiber_follow.train import fit_flow_sigma, update_ema


def config():
    return FollowNetConfig(in_channels=3, depth=24, width=17, behind=6, spacing=.5,
                           widths=(8, 16), hidden=16, n_future=8, future_step=1.,
                           hist_points=4, hist_stride=2, recent_history_points=4, flow_sigma=((1., 1.),)*8,
                           flow_stencil_radius=.5,  # leaves 3.5 observable voxels in the 4-voxel half-width
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
    gt_history = torch.zeros(1, cfg.recent_history_points+1, 3)
    gt_history[0, :, 2] = -torch.arange(cfg.recent_history_points+1)
    return dict(dense_ab=ab, dense_mask=torch.ones(1, q), offtrack=torch.zeros(1),
                plane_ab=ab[:, ::4], plane_mask=torch.ones(1, cfg.n_future),
                gt_history=gt_history, gt_history_mask=torch.ones(1, cfg.recent_history_points+1),
                hist=hist, hmask=hmask, endpoint_known=torch.zeros(1), end_local=torch.zeros(1, 3),
                replay_valid=torch.zeros(1, cfg.flow_samples),
                replay_candidates=torch.zeros(1, cfg.flow_samples, cfg.n_future, 3))


def test_samples_pin_future_planes_and_outputs_have_scorer_shapes():
    torch.manual_seed(1)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    out = model(x, hist, mask, generator=torch.Generator().manual_seed(0))
    P = cfg.n_future
    assert out['samples'].shape == (1, cfg.flow_samples, P, 3)
    assert out['candidates'].shape == (1, cfg.flow_samples, cfg.n_future, 3)
    assert 'clean_history' not in out and 'corrected_path' not in out
    assert out['ranks'].shape == (1, cfg.flow_samples) and out['confidence'].shape == (1, cfg.flow_samples, cfg.n_future)
    assert out['candidate_support'].shape == (1, cfg.flow_samples)
    planes = cfg.future_step*torch.arange(1, cfg.n_future+1)
    torch.testing.assert_close(out['candidates'][..., 2], planes.expand(1, cfg.flow_samples, -1))
    torch.testing.assert_close(out['samples'][..., 2], planes.expand(1, cfg.flow_samples, -1))
    torch.testing.assert_close(out['candidates'], out['samples'])
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
    assert {'flow', 'candidate_support', 'oracle_recall', 'observed_current_error'} <= metrics.keys()
    assert not any(k.startswith('clean_') or k == 'flow_past' for k in metrics)


@pytest.mark.parametrize('case', ['known', 'unknown', 'offtrack'])
def test_skipping_metrics_preserves_loss_and_gradients(case, monkeypatch):
    from vesuvius.neural_tracing.fiber_follow import supervision
    torch.manual_seed(73)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    if case == 'unknown':
        batch['plane_mask'].zero_()
        batch['dense_mask'].zero_()
    elif case == 'offtrack':
        batch['offtrack'].fill_(1)
    out = model(x, hist, mask, teacher_candidates(batch, cfg), targets=batch)
    weights = dict(flow_weight=.7, rank_weight=1.3, confidence_weight=.4)
    full_loss, metrics = loss_fn(out, batch, cfg, **weights)
    parameters = tuple(model.parameters())
    full_grad = torch.autograd.grad(full_loss, parameters, retain_graph=True, allow_unused=True)

    def unexpected_metric(*args, **kwargs):
        raise AssertionError('Non-logging steps must skip diagnostic selection and scalar extraction')
    with monkeypatch.context() as patch:
        patch.setattr(supervision, 'choose_candidate', unexpected_metric)
        patch.setattr(torch.Tensor, 'item', unexpected_metric)
        loss, skipped = loss_fn(out, batch, cfg, compute_metrics=False, **weights)
        grad = torch.autograd.grad(loss, parameters, allow_unused=True)
    assert skipped == {} and 'oracle_recall' in metrics
    torch.testing.assert_close(loss, full_loss, rtol=0, atol=0)
    for actual, expected in zip(grad, full_grad):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('case', ['unknown', 'offtrack'])
def test_unlabeled_states_have_no_flow_loss(case):
    torch.manual_seed(4)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    if case == 'unknown':
        batch['plane_mask'].zero_()
        batch['gt_history_mask'].zero_()
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
    x1, token_mask = model.future_targets(batch)
    assert token_mask[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    torch.testing.assert_close(x1[0, :, 2], cfg.future_step*torch.arange(1, cfg.n_future+1))
    # Loss and gradient reflect only the known tokens.
    out = model(x, hist, mask, targets=batch)
    assert 0 < out['flow_known_fraction'].item() < 1 and out['flow_loss'].item() > 0


def test_prior_contains_only_future_planes():
    cfg = config()
    model = FollowNet(cfg)
    _, hist, mask = inputs(cfg)
    hist[:, :, 0] = 2.
    mask[:, 3:] = 0
    mu = model.flow.prior(hist, mask)
    assert mu.shape == (1, cfg.n_future, 3)
    torch.testing.assert_close(mu[0, :, 2], cfg.future_step*torch.arange(1, cfg.n_future+1))
    assert not mu[..., :2].any()


def test_history_targets_never_affect_generation_or_flow_loss():
    torch.manual_seed(6)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    original = hist.clone()
    batch = targets(cfg, hist, mask)
    first = model(x, hist, mask, targets=batch, generator=torch.Generator().manual_seed(8))
    batch['gt_history'].fill_(1000)
    batch['gt_history_mask'].zero_()
    second = model(x, hist, mask, targets=batch, generator=torch.Generator().manual_seed(8))
    for key in ('samples', 'candidates', 'ranks', 'confidence', 'flow_loss'):
        torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(hist, original)


def test_unobserved_history_and_unknown_future_cannot_influence_known_velocities():
    torch.manual_seed(7)
    cfg = replace(config(), flow_layers=2)
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    mask[:, 2:] = 0
    features, context = model.encode(x, hist, mask)
    future = torch.zeros(1, 1, cfg.n_future, 2, requires_grad=True)
    fixed = model.flow.conditioning(features.float(), hist, mask, model.sampling_grid)
    known = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    time = torch.full((1, 1), .9)
    velocity = model.flow(features.float(), context, future, time, fixed,
                          model.sampling_grid, future_mask=known)
    corrupted = future.detach().clone()
    corrupted[:, :, 3:] = float('nan')
    changed_history = hist.clone()
    changed_history[:, 2:] = float('nan')
    other_features, other_context = model.encode(x, changed_history, mask)
    other_fixed = model.flow.conditioning(other_features.float(), changed_history, mask, model.sampling_grid)
    other = model.flow(other_features.float(), other_context, corrupted, time, other_fixed,
                       model.sampling_grid, future_mask=known)
    torch.testing.assert_close(velocity[:, :, :3], other[:, :, :3])
    velocity[:, :, :3].sum().backward()
    assert not future.grad[:, :, 3:].any()
    assert future.grad[:, :, :3].abs().sum() > 0
    # The training bridge must also ignore arbitrary missing target values.
    batch = targets(cfg, hist, mask)
    batch['plane_mask'] = known.float()
    a = model.flow_loss(features.float(), context, hist, mask, batch, torch.Generator().manual_seed(5))
    batch['plane_ab'][:, 3:] = float('nan')
    b = model.flow_loss(features.float(), context, hist, mask, batch, torch.Generator().manual_seed(5))
    torch.testing.assert_close(a['flow_loss'], b['flow_loss'])


def test_observed_history_conditions_flow_without_being_generated():
    torch.manual_seed(8)
    cfg = config()
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    features, context = model.encode(x, hist, mask)
    future = torch.zeros(1, 1, cfg.n_future, 2)
    # Keep global context and image encoding fixed to test the history tokens.
    hist.requires_grad_()
    fixed = model.flow.conditioning(features.float(), hist, mask, model.sampling_grid)
    velocity = model.flow(features.float(), context, future, torch.full((1, 1), .5),
                          fixed, model.sampling_grid)
    velocity.sum().backward()
    assert hist.grad[:, :cfg.recent_history_points].abs().sum() > 0
    assert not hist.grad[:, cfg.recent_history_points:].any()
    assert velocity.shape == future.shape


def test_bfloat16_training_handles_missing_history_and_partial_future():
    torch.manual_seed(9)
    cfg = config()
    model = FollowNet(cfg).train()
    x, hist, mask = inputs(cfg)
    mask.zero_()
    batch = targets(cfg, hist, mask)
    batch['plane_mask'][:, 3:] = 0
    batch['dense_mask'][:, 9:] = 0
    with torch.backends.mkldnn.flags(enabled=False):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model(x, hist, mask, teacher_candidates(batch, cfg), targets=batch)
        loss, _ = loss_fn(out, batch, cfg)
        loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_flow_optimization_learns_a_curved_continuation():
    torch.manual_seed(21)
    cfg = replace(config(), flow_draws=8)
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    optimizer = torch.optim.Adam(model.parameters(), lr=.005)
    losses = []
    generator = torch.Generator().manual_seed(0)
    for _ in range(450):
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


def test_support_for_all_samples_teachers_and_replay_excludes_self_votes():
    paths = torch.zeros(1, 4, 6, 3)
    paths[0, :, :, 0] = torch.tensor([0., .2, 4., 10.])[:, None]
    extras = torch.zeros(1, 3, 6, 3)
    extras[0, :, :, 0] = torch.tensor([.1, 4., 20.])[:, None]
    support = candidate_support(torch.cat([paths, extras], 1), paths, 1.5, 4)
    assert support[0].tolist() == pytest.approx([1/3, 1/3, 0., 0., .5, .25, 0.])
    # Coincident samples remain separate proposals; their other votes all count.
    paths.zero_()
    torch.testing.assert_close(candidate_support(paths, paths, 1.5, 4), torch.ones(1, 4))
    assert candidate_support(paths[:, :1], paths[:, :1], 1.5, 4).item() == 0


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
        FollowNet(replace(config(), flow_samples=0))
    with pytest.raises(ValueError):
        FollowNet(replace(config(), hidden=15))
    with pytest.raises(ValueError):
        FollowNet(replace(config(), flow_sigma=()))


def test_beam_owns_its_optional_head_without_flow_parameters():
    from vesuvius.neural_tracing.fiber_follow.beam.model import BeamNetConfig, BeamRankNet
    cfg = BeamNetConfig(**config().to_dict())
    model = BeamRankNet(cfg)
    assert not hasattr(model, 'flow')
    x, hist, mask = inputs(cfg)
    paths = torch.randn(1, 2, 8, 3)
    out = model(x, hist, mask, paths, torch.ones(1, 2, 8), torch.zeros(1, 2))
    assert out['ranks'].shape == (1, 2) and out['tube_logits'].shape == x.shape[:1]+x.shape[2:]


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
    with pytest.raises(ValueError, match='observable'):
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


def test_normalized_loss_and_stratified_times_match_the_training_bridge(monkeypatch):
    # A wider crop keeps the unit-scale targets below observable; censoring is tested separately.
    cfg = replace(config(), flow_draws=8, width=65, flow_sigma=tuple((float(k), float(k+1)) for k in range(1, 9)))
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    hist[..., 0] = .1*hist[..., 2]
    features, context = model.encode(x, hist, mask)
    fixed = model.flow.conditioning(features.float(), hist, mask, model.sampling_grid)
    batch = targets(cfg, hist, mask)
    # Scale and mean cancel: every supervised normalized target is exactly one.
    batch['plane_ab'] = fixed['mu'][..., :2] + model.flow.sigma
    batch['plane_mask'][:, -2:] = 0
    batch['plane_ab'][:, -2:] = float('nan')
    captured = []
    def zero_field(features, context, y, t, fixed, grid, future_mask=None):
        captured.append((y.detach().clone(), t.detach().clone()))
        return torch.zeros_like(y)
    monkeypatch.setattr(model.flow, 'forward', zero_field)
    result = model.flow_loss(features.float(), context, hist, mask, batch, torch.Generator().manual_seed(9), fixed)
    y, t = captured[0]
    assert torch.equal((t*cfg.flow_draws).long(), torch.arange(cfg.flow_draws)[None])
    # Recover source noise from the observed bridge and known unit endpoint.
    y0 = (y[:, :, :-2]-t[..., None, None])/(1-t[..., None, None])
    expected = (1-y0).square().mean()
    torch.testing.assert_close(result['flow_loss'], expected)
    assert result['flow_known_fraction'].item() == .75
    assert result['flow_censored_fraction'].item() == 0


def test_midpoint_has_second_order_convergence_in_normalized_space(monkeypatch):
    cfg = replace(config(), flow_sigma=tuple((float(k), float(k+1)) for k in range(1, 9)))
    model = FollowNet(cfg).eval()
    x, hist, mask = inputs(cfg)
    hist[..., 0] = .1*hist[..., 2]
    features, context = model.encode(x, hist, mask)
    fixed = model.flow.conditioning(features.float(), hist, mask, model.sampling_grid)
    calls = []
    def field(features, context, y, t, fixed, grid, future_mask=None):
        calls.append(t[0, 0].item())
        return y + t[..., None, None]
    monkeypatch.setattr(model.flow, 'forward', field)
    initial = torch.randn(1, cfg.flow_samples, cfg.n_future, 2, generator=torch.Generator().manual_seed(3))
    exact = math.e*initial + math.e-2  # y' = y+t, integrated from 0 to 1
    errors = []
    for steps in (4, 8, 16):
        calls.clear()
        paths = model.sample(features.float(), context, hist, mask, steps=steps,
                             generator=torch.Generator().manual_seed(3), fixed=fixed)
        normalized = (paths[..., :2]-fixed['mu'][:, None, :, :2])/model.flow.sigma
        errors.append((normalized-exact).abs().mean().item())
        assert len(calls) == 2*steps and min(calls) == 0 and max(calls) < 1
        assert calls[1] == pytest.approx(.5/steps)
        torch.testing.assert_close(paths[..., 2], model.flow.planes.expand(1, cfg.flow_samples, -1))
    assert errors[0]/errors[1] > 3.5 and errors[1]/errors[2] > 3.5


def test_observation_patches_are_static_always_valid_and_use_voxel_pitch(monkeypatch):
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    mask.zero_()
    features, context = model.encode(x, hist, mask)
    queries = []
    original = model.flow.sample_features
    def capture(features, coordinates, grid):
        queries.append(coordinates.detach().clone())
        return original(features, coordinates, grid)
    monkeypatch.setattr(model.flow, 'sample_features', capture)
    fixed = model.flow.conditioning(features.float(), hist, mask, model.sampling_grid)
    assert fixed['valid'][:, model.flow.n_history:].all()
    assert not fixed['valid'][:, 1:model.flow.n_history].any()
    torch.testing.assert_close(fixed['coordinates'][:, model.flow.n_history:], fixed['mu'])
    assert model.flow.stencil.shape == (9, 3)
    assert model.flow.stencil[:, :2].abs().max() == cfg.flow_stencil_radius
    other = FollowNet(replace(cfg, spacing=1.))
    torch.testing.assert_close(model.flow.stencil, other.flow.stencil)
    batch = targets(cfg, hist, mask)
    batch['plane_mask'].zero_()
    model.flow_loss(features.float(), context, hist, mask, batch, fixed=fixed)
    model.sample(features.float(), context, hist, mask, steps=2, fixed=fixed)
    assert len(queries) == 1+1+4  # one static read, one loss call, four midpoint stages
    assert all(q.shape[-2] == cfg.n_future for q in queries[1:])
    # Always-valid observation tokens continue to supply gradients with partial targets.
    batch['plane_mask'][:, :2] = 1
    loss = model.flow_loss(features.float(), context, hist, mask, batch, fixed=fixed)['flow_loss']
    loss.backward()
    assert model.flow.kind_embedding.weight.grad[1].abs().sum() > 0


def test_group_norm_train_and_eval_produce_the_same_samples_and_scores():
    torch.manual_seed(71)
    model = FollowNet(config())
    x, hist, mask = inputs(model.cfg)
    with torch.no_grad():
        train = model.train()(x, hist, mask, generator=torch.Generator().manual_seed(3))
        evaluate = model.eval()(x, hist, mask, generator=torch.Generator().manual_seed(3))
    for key in ('samples', 'ranks', 'confidence'):
        torch.testing.assert_close(train[key], evaluate[key], atol=2e-5, rtol=2e-5)


def test_scorer_uses_support_and_every_extra_gets_measured_support():
    torch.manual_seed(19)
    cfg = config()
    model = FollowNet(cfg)
    x, hist, mask = inputs(cfg)
    batch = targets(cfg, hist, mask)
    extra = teacher_candidates(batch, cfg)
    out = model(x, hist, mask, extra)
    assert out['candidate_support'].shape == (1, cfg.flow_samples+extra.shape[1])
    torch.testing.assert_close(out['candidates'][:, :cfg.flow_samples], out['samples'])
    features, _ = model.encode(x, hist, mask)
    support = out['candidate_support'].detach().requires_grad_()
    ranks, confidence = model.score_candidates(features, out['candidates'], hist, mask, support)
    assert torch.autograd.grad(ranks.sum(), support, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(confidence.sum(), support)[0].abs().sum() > 0


def test_ema_updates_parameters_without_gradients_and_copies_buffers():
    model = FollowNet(config())
    ema = copy.deepcopy(model).requires_grad_(False).eval()
    before = ema.flow.velocity.weight.clone()
    with torch.no_grad():
        model.flow.velocity.weight.add_(2)
        model.flow.sigma.add_(1)
    update_ema(ema, model, step=1, decay=.999)
    torch.testing.assert_close(ema.flow.velocity.weight, before+2*(1-2/11))
    torch.testing.assert_close(ema.flow.sigma, model.flow.sigma)
    before = ema.flow.velocity.weight.clone()
    update_ema(ema, model, step=100000, decay=.999)
    torch.testing.assert_close(ema.flow.velocity.weight, before*.999+model.flow.velocity.weight*.001)
    assert all(not p.requires_grad and p.grad is None for p in ema.parameters())
