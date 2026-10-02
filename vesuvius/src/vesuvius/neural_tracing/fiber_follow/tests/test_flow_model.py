"""The common trainer operates both generators without a separate flow training path."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from model_fixtures import coordinate_config, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.models.model import CoordinateRegressionFollower, build_model
from vesuvius.neural_tracing.fiber_follow.models.flow import FlowConfig, fit_flow_sigma, flow_targets
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import (
    prepare_training, training_prediction, optimizer_update, save_checkpoint, load_checkpoint,
    initialize_training_optimizer, initialize_model_weights, build_parser, model_config_from_args,
)
from vesuvius.neural_tracing.fiber_follow.train.runloop import training_rng_state
from vesuvius.neural_tracing.fiber_follow.data.data import SampleConfig
from vesuvius.neural_tracing.fiber_follow.data.volume import FiberVolumeSpec


def config(**options):
    return FlowConfig(**(coordinate_config().to_dict() | dict(model_type='flow_matching',
        recurrent_refinement_steps=0, flow_steps=2, flow_draws=2, flow_sigma=((1., 1.),)*4) | options))


def take(value, index):
    return {k: take(v, index) for k, v in value.items()} if isinstance(value, dict) else value[index]


def test_flow_shares_components_and_compiles_with_finite_gradients():
    torch.manual_seed(31)
    cfg = config()
    model, coordinate = build_model(cfg), CoordinateRegressionFollower(coordinate_config())
    for name in ('encoder', 'history_encoder', 'history_attention', 'confidence_scorer', 'path_geometry'):
        assert type(getattr(model, name)) is type(getattr(coordinate, name))
    b = coordinate_batch(cfg)
    b['flow_noise'] = torch.randn(2, cfg.flow_draws, cfg.n_future, 2)
    b['flow_times'] = torch.tensor([[.1, .7], [.2, .9]])
    eager = model.select_prediction(model.training_forward(b['x'], b['hist'], b['hmask'], .5, b))
    compiled = prepare_training(copy.deepcopy(model), backend='eager')
    out = training_prediction(compiled, b['x'], b['hist'], b['hmask'], targets=b)
    for key in eager:
        torch.testing.assert_close(out[key], eager[key], atol=1e-6, rtol=1e-5)
    terms = loss_terms(out, b, cfg)
    (terms['geometry_per_state']+.5*terms['confidence_per_state']).mean().backward()
    for module in (compiled.velocity, compiled.encoder.patch_projection, compiled.history_encoder.convolution[0],
                   compiled.confidence_scorer.failure, compiled.history_encoder.path_projection[-1]):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0
    assert out['refinement_points'].shape[1] == 1
    assert out['solver_points'].shape[1] == cfg.flow_steps+1
    assert not out['solver_points'][:, 0, :, :2].any()


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


def test_flow_scales_and_default_factory():
    args = build_parser().parse_args(['--name', 'test'])
    assert model_config_from_args(args).model_type == 'coordinate_regression'
    args.model = 'flow_matching'
    assert isinstance(model_config_from_args(args), FlowConfig)
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
