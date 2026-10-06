"""Flow matching (models/flow.py) on the crop transformer: the common trainer, proposals, selection and losses."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_batch, ct_volume, flow_config, run_document
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.flow import fit_flow_sigma, flow_targets
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import (
    prepare_training, optimizer_update, save_checkpoint, load_checkpoint,
    initialize_training_optimizer, initialize_model_weights,
)
from vesuvius.neural_tracing.fiber_follow.train.runloop import training_rng_state
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


def config(**options):
    return flow_config(**dict(dict(flow_draws=2), **options))


def take(value, index):
    return {k: take(v, index) for k, v in value.items()} if isinstance(value, dict) else value[index]


def test_flow_draws_and_trace_rows_are_independent_and_unknown_targets_are_censored():
    cfg = config(); model = build_model(cfg).eval(); b = coordinate_batch(cfg)
    with torch.no_grad():
        out = model(b['x'], b['hist'], b['hmask'])
        for row in range(2):
            one = take(b, slice(row, row+1))
            result = model(one['x'], one['hist'], one['hmask'])
            torch.testing.assert_close(result['points'], out['points'][row:row+1], atol=2e-5, rtol=2e-5)
        ctx = model.context(b['x'], b['hist'], b['hmask']); model.prepare_prediction(ctx, b['hist'])
        y, t = torch.randn(2, 2, 4, 2), torch.rand(2, 2)
        both = model.velocity_field(ctx, y, t)
        solo = model.velocity_field(ctx, y[:, :1], t[:, :1])
        torch.testing.assert_close(both[:, :1], solo, atol=1e-6, rtol=1e-5)
        b['geometry_valid'].zero_(); b['dense_ab'].fill_(float('nan'))
        target, mask = flow_targets(b, cfg)
        assert not mask.any() and not target.any()
        result = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
        assert torch.isfinite(result['flow_per_state']).all() and not result['flow_per_state'].any()


def test_flow_scales():
    cfg = config(); b = coordinate_batch(cfg)
    b['dense_ab'][0, :, 0] = -2.; b['dense_ab'][1, :, 0] = 2.
    assert fit_flow_sigma(iter([b]), cfg, 2) == ((2., 1.),)*cfg.n_future
    b['geometry_valid'].zero_()
    with pytest.raises(ValueError, match='two known targets'):
        fit_flow_sigma(iter([b]), cfg, 2)


def test_flow_optimizer_ema_checkpoint_and_resume(tmp_path):
    torch.manual_seed(32)
    cfg = config(); model, ema = initialize_model_weights(cfg, 'cpu')
    args = SimpleNamespace(lr=.001, reset_optimizer=False)
    opt, _, _ = initialize_training_optimizer(model, ema, args)
    prepare_training(model, backend='eager'); b = coordinate_batch(cfg)
    metrics = optimizer_update(model, ema, opt, [b], 1, .001)
    assert metrics['prediction_loss_type'] == 'flow' and metrics['flow'] > 0
    path = tmp_path/'last.pt'
    sample = SampleConfig(crop=cfg.fine, n_history=cfg.n_history, n_future=cfg.n_future)
    save_checkpoint(path, model, ema, FiberVolumeSpec('unused', inputs='ct'), sample,
                    dict(step=1, optimizer=opt.state_dict(), rng=training_rng_state()))
    loaded, _, _, _, ck = load_checkpoint(path, 'cpu')
    assert loaded.cfg == cfg
    restored, restored_ema = initialize_model_weights(cfg, 'cpu')
    restored_opt, done, _ = initialize_training_optimizer(restored, restored_ema, args, ck)
    assert done == 1 and restored_opt.state
    prepare_training(restored, backend='eager')
    # Fixed velocity draws isolate update/accumulation semantics from RNG order.
    b['flow_noise'] = torch.randn(2, cfg.flow_draws, cfg.n_future, 2)
    b['flow_times'] = torch.rand(2, cfg.flow_draws)
    optimizer_update(model, ema, opt, [b], 2, .001, compute_metrics=False)
    optimizer_update(restored, restored_ema, restored_opt, [b],
                     2, .001, compute_metrics=False)
    for p, q in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_flow_loss_and_gradients_are_weighted_per_state_across_microbatches():
    torch.manual_seed(53)
    cfg = config(); whole = build_model(cfg); split = copy.deepcopy(whole)
    b = coordinate_batch(cfg)
    b['dense_mask'][1, 5:] = 0  # Unequal numbers of known planes must not reweight states.
    b['flow_noise'] = torch.randn(2, cfg.flow_draws, cfg.n_future, 2)
    b['flow_times'] = torch.tensor([[.1, .6], [.2, .8]])
    def objective(model, data):
        out = model.select_prediction(model.training_forward(data['x'], data['hist'], data['hmask'], .5, data))
        terms = loss_terms(out, data, cfg)
        return (terms['geometry_per_state']+.5*terms['confidence_per_state']).sum()/2
    expected = objective(whole, b); expected.backward()
    actual = 0.
    for index in range(2):
        loss = objective(split, take(b, slice(index, index+1)))
        loss.backward(); actual += loss.detach()
    torch.testing.assert_close(actual, expected.detach(), atol=1e-6, rtol=1e-5)
    for (name, p), q in zip(whole.named_parameters(), split.parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_inference_samples_retry_after_the_zero_start_path_only_when_it_is_not_accepted():
    torch.manual_seed(17)
    model = build_model(config()).eval()
    sampled = build_model(config(flow_samples=3, flow_sample_scale=1.5)).eval()
    sampled.load_state_dict(model.state_dict())
    b = coordinate_batch(model.cfg)
    with torch.no_grad():
        for threshold in (0., .5, 1.):
            plain = model(b['x'], b['hist'], b['hmask'], threshold)
            out = sampled(b['x'], b['hist'], b['hmask'], threshold)
            assert out['refinement_points'].shape[1] == 4
            torch.testing.assert_close(out['refinement_points'][:, 0], plain['points'])
            torch.testing.assert_close(out['refinement_confidence'][:, 0], plain['confidence'])
            assert not torch.equal(out['refinement_points'][:, 1], out['refinement_points'][:, 0])
            index = torch.arange(len(out['points']))
            torch.testing.assert_close(out['points'], out['refinement_points'][index, out['selected_refinement']])
            if threshold == 0.:  # the zero-start path is accepted, so no sample is used
                assert (out['selected_refinement'] == 0).all()
                torch.testing.assert_close(out['points'], plain['points'])


def test_sampled_proposals_are_scored_in_training_and_the_zero_start_path_is_unchanged():
    torch.manual_seed(23)
    model = build_model(config())
    sampled = build_model(config(flow_samples=2, flow_sample_scale=2.))
    sampled.load_state_dict(model.state_dict())
    b = coordinate_batch(model.cfg)
    out = sampled.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    plain = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    assert out['refinement_points'].shape[1] == 3 and out['refinement_mask'].all()
    torch.testing.assert_close(out['refinement_points'][:, 0], plain['refinement_points'][:, 0])
    terms = loss_terms(sampled.select_prediction(out), b, sampled.cfg)
    assert int(terms['refinement_attempts_sum']) == 2*3
    terms['confidence_per_state'].sum().backward()
    assert all(float(m.weight.grad.abs().sum()) > 0 for m in (sampled.score[0], sampled.hazard))


def test_unknown_planes_are_attended_only_with_own_path_and_the_scale_floor_bounds_the_prior():
    torch.manual_seed(29)
    b = coordinate_batch(config())
    b['dense_mask'][:, b['dense_mask'].shape[1]//2:] = 0  # the trailing planes have no target
    for mode, attended in (('padded', False), ('own_path', True)):
        model = build_model(config(flow_unknown_planes=mode)).eval()
        _, known = flow_targets(b, model.cfg)
        assert known.any() and not known.all()
        losses = []
        for seed in (0, 1):
            noise = torch.randn(2, model.cfg.flow_draws, model.cfg.n_future, 2, generator=torch.Generator().manual_seed(7))
            # Change the Gaussian draw at target-less planes only.
            noise[~known[:, None].expand(-1, model.cfg.flow_draws, -1)] = torch.randn(
                int((~known).sum())*model.cfg.flow_draws, 2, generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                out = model.training_forward(b['x'], b['hist'], b['hmask'], .5,
                                             dict(b, flow_noise=noise, flow_times=torch.tensor([[.3, .8]]*2)))
            losses.append(out['flow_per_state'])
        assert torch.isfinite(losses[0]).all()
        assert (not torch.allclose(losses[0], losses[1])) == attended
    cfg = config(flow_sigma_floor=3., flow_sigma=((3., 3.),)*4)
    data = coordinate_batch(cfg)
    data['dense_ab'][0, :, 0] = -2.; data['dense_ab'][1, :, 0] = 2.
    assert fit_flow_sigma(iter([data]), cfg, 2) == ((3., 3.),)*cfg.n_future
    with pytest.raises(ValueError, match='scale floor'):
        config(flow_sigma_floor=3.)  # the fixture's unit scales lie below the floor


def test_best_selection_ranks_all_proposals_and_noise_only_drops_the_zero_start():
    from vesuvius.neural_tracing.fiber_follow.models.model import select_refinement
    cfg = config()
    points = torch.zeros(1, 3, cfg.n_future, 3)
    points[..., 2] = torch.arange(1., cfg.n_future+1)
    confidence = torch.tensor([[[.6]*cfg.n_future, [.9]*cfg.n_future, [.95]*cfg.n_future]])
    output = dict(refinement_points=points, refinement_confidence=confidence,
                  refinement_mask=torch.ones(1, 3, dtype=torch.bool),
                  **{f'refinement_{k}': confidence for k in ('hazard_logits', 'confidence_logits')})
    # Proposal 0 is accepted, so a retry keeps it; one ranking over all takes the most confident.
    assert int(select_refinement(output, cfg, .5)['selected_refinement']) == 0
    assert int(select_refinement(output, cfg, .5, retry=False)['selected_refinement']) == 2
    torch.manual_seed(37)
    model = build_model(config(flow_samples=3, flow_zero_start=False, flow_selection='best')).eval()
    b = coordinate_batch(model.cfg)
    with torch.no_grad():
        out = model(b['x'], b['hist'], b['hmask'])
    assert out['refinement_points'].shape[1] == 3 and out['refinement_points'][:, :, :, :2].abs().sum() > 0
    with pytest.raises(ValueError, match='zero start'):
        config(flow_zero_start=False)


def test_sample_threshold_holds_samples_to_a_higher_bar_than_the_zero_start():
    torch.manual_seed(41)
    cfg = config(flow_samples=2, flow_selection='best', flow_sample_threshold=.8)
    model = build_model(cfg)
    points = torch.zeros(1, 3, cfg.n_future, 3)
    points[..., 2] = torch.arange(1., cfg.n_future+1)
    confidence = torch.tensor([[[.52]*cfg.n_future, [.75]*cfg.n_future, [.85]*cfg.n_future]])
    output = dict(refinement_points=points, refinement_confidence=confidence,
                  refinement_mask=torch.ones(1, 3, dtype=torch.bool),
                  **{f'refinement_{k}': confidence for k in ('hazard_logits', 'confidence_logits')})
    out = model.select_prediction(output, .5)
    # Sample 1 (0.75) falls below the 0.8 bar and cannot commit; sample 2 (0.85) clears it and is ranked
    # on its own confidence, above the zero start.
    assert int(out['selected_refinement']) == 2
    torch.testing.assert_close(out['confidence'], torch.full((1, cfg.n_future), .85))
    # Below the bar a sample is out of the running, however it compares with the zero start.
    output['refinement_confidence'] = torch.tensor([[[.52]*cfg.n_future, [.75]*cfg.n_future, [.78]*cfg.n_future]])
    out = model.select_prediction(output, .5)
    assert int(out['selected_refinement']) == 0
    from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix
    barred = model.select_prediction(dict(output, refinement_mask=torch.tensor([[False, False, True]])), .5)
    assert int(commit_prefix(barred['points'], barred['confidence'], .5, cfg.n_future)[0]) == 0


def test_keyed_proposal_noise_depends_only_on_each_rows_key_and_the_tracer_keys_each_decision(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
    torch.manual_seed(43)
    model = build_model(config(flow_samples=3))
    hist = torch.zeros(3, model.cfg.n_history, 3)
    keys = torch.tensor([11, 22, 33])
    together = model.proposal_starts(hist, keys)
    alone = model.proposal_starts(hist[:1], keys[1:2])
    torch.testing.assert_close(alone[0], together[1], rtol=0, atol=0)  # independent of batch neighbours
    torch.testing.assert_close(model.proposal_starts(hist, keys), together, rtol=0, atol=0)
    assert not torch.equal(together[0, 1:], together[2, 1:]) and not together[:, 0].any()
    assert not torch.equal(model.proposal_starts(hist)[:, 1:], model.proposal_starts(hist)[:, 1:])  # global RNG

    flow = build_model(config(flow_samples=2)).eval()
    vol = ct_volume(tmp_path)

    def trace_keys():
        keys, forward = [], flow.forward
        def capture(x, *args, **kwargs):
            keys.append(x['flow_noise_keys'].tolist())
            return forward(x, *args, **kwargs)
        flow.forward = capture
        tracer = FiberTracer(flow, vol, flow.cfg.fine, flow.cfg.n_history,
                             TraceParams(n_commit=1, max_len=4., confidence=0.), device="cpu")
        try:
            tracer.trace(np.array([[24., 24., 24.]]), np.array([[.3, .4, .8660254]]))
        finally:
            tracer.close()
            del flow.forward
        return keys
    first, second = trace_keys(), trace_keys()
    assert first == second and len(first) > 3
    assert len({k[0] for k in first}) == len(first)  # a fresh key every decision


def test_pseudo_huber_flow_loss_is_half_the_squared_distance_for_small_residuals_and_linear_beyond_c():
    torch.manual_seed(37)
    b = coordinate_batch(config())
    noise = torch.randn(2, 2, 4, 2)*3
    target, known = flow_targets(b, config())
    # A zero velocity makes the residual (noise - target) per known path point (unit scales).
    distance = (noise-target[:, None]).norm(dim=-1)[known[:, None].expand(-1, 2, -1)].reshape(2, -1)
    for loss, c, expected in (('mse', 1., .5*distance.square()),
                              ('pseudo_huber', 1., ((1+distance.square()).sqrt()-1)),
                              ('pseudo_huber', .5, .25*((1+(distance/.5).square()).sqrt()-1))):
        model = build_model(config(flow_loss=loss, flow_huber_c=c))
        model.velocity_field = lambda ctx, y, t, known=None: torch.zeros_like(y)
        out = model.training_forward(b['x'], b['hist'], b['hmask'], .5, dict(b, flow_noise=noise))
        torch.testing.assert_close(out['flow_per_state'], expected.mean(-1).float(), rtol=1e-5, atol=1e-6)
    huber = ((1+distance.square()).sqrt()-1)
    assert (huber <= .5*distance.square()+1e-6).all()  # never above the squared loss; linear for large residuals
    with pytest.raises(ValueError, match='Flow loss'):
        config(flow_loss='l1')


def test_flow_geometry_weight_trains_the_integrated_zero_start_path_and_nothing_else_sees_its_gradient():
    torch.manual_seed(41)
    plain = build_model(config(flow_samples=2))
    model = build_model(config(flow_samples=2, flow_geometry_weight=.5))
    model.load_state_dict(plain.state_dict())
    b = coordinate_batch(model.cfg)
    b['flow_noise'] = torch.randn(2, 2, 4, 2)
    torch.manual_seed(5); base = plain.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    torch.manual_seed(5); out = model.training_forward(b['x'], b['hist'], b['hmask'], .5, b)
    assert 'flow_geometry_points' not in base and out['flow_geometry_points'].requires_grad
    # Same proposals, scores and flow loss; the proposals reach the scorer detached.
    for key in ('refinement_points', 'refinement_confidence', 'flow_per_state'):
        torch.testing.assert_close(out[key], base[key], atol=2e-5, rtol=1e-5)
    assert not out['refinement_points'].requires_grad
    torch.testing.assert_close(out['flow_geometry_points'], out['refinement_points'][:, 0], atol=2e-5, rtol=1e-5)
    terms = loss_terms(model.select_prediction(out), b, model.cfg)
    torch.testing.assert_close(terms['geometry_per_state'],
                               out['flow_per_state']+.5*terms['flow_path_geometry_per_state'])
    assert 'flow_path_geometry_per_state' not in loss_terms(plain.select_prediction(base), b, plain.cfg)
    terms['flow_path_geometry_per_state'].sum().backward()
    assert model.velocity.weight.grad.abs().sum() > 0
    assert all(p.grad is None or not p.grad.any() for m in (model.score, model.hazard) for p in m.parameters())
    model.eval()
    with torch.no_grad():
        assert 'flow_geometry_points' not in model.training_forward(b['x'], b['hist'], b['hmask'], .5)
    with pytest.raises(ValueError, match='zero-start path'):
        config(flow_samples=2, flow_zero_start=False, flow_geometry_weight=.5)


def test_flow_loss_and_geometry_options_reach_the_model_config():
    resolve = lambda **fields: run_config.model_config(run_config.resolve(run_document('flow', **fields)))
    cfg = resolve(flow_loss='mse', flow_huber_c=2, flow_geometry_weight=.25)
    assert (cfg.flow_loss, cfg.flow_huber_c, cfg.flow_geometry_weight) == ('mse', 2., .25)
    default = resolve()
    assert (default.flow_loss, default.flow_geometry_weight, default.flow_sigma) == ('pseudo_huber', 0., ())
