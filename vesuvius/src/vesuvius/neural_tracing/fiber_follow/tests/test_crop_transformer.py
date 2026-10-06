"""Crop transformer (models/crop_transformer.py): CNN tokens and one transformer, as regression or flow."""
import copy

import pytest
import torch

from model_fixtures import coordinate_batch, ct_volume, run_document
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, config_from_checkpoint
from vesuvius.neural_tracing.fiber_follow.models.crop_transformer import FlowConfig, RegressionConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train import run_config
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import prepare_training, training_prediction

SMALL = dict(fine=CropSpec(depth=32, width=16, behind=8, spacing=.5), n_future=8, gate_plane=4, hidden=32, layers=2,
             heads=2, ffn=64, cnn_channels=(4, 8, 16), cnn_blocks=(1, 1, 1), n_history=32)
# The plain flow model: time added to the path tokens once, padded unknown planes, squared error, one path.
PLAIN_FLOW = dict(flow_steps=2, flow_draws=3, flow_sigma=((1., 1.),)*8, flow_samples=0, flow_time_conditioning='input',
                  flow_sigma_floor=1., flow_unknown_planes='padded', flow_loss='mse')


def small(flow=False, **options):
    torch.manual_seed(0)
    if flow:
        cfg = FlowConfig(**dict(SMALL, **dict(PLAIN_FLOW, **options)))
    else:
        cfg = RegressionConfig(**dict(SMALL, recurrent_refinement_steps=2, **options))
    return build_model(cfg)


def test_regression_predicts_every_plane_with_refinement_slots():
    model = small().eval()
    b = coordinate_batch(model.cfg, 3)
    with torch.no_grad():
        out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
    assert out['points'].shape == (3, 8, 3) and out['refinement_points'].shape == (3, 3, 8, 3)
    assert out['refinement_mask'].all()  # nothing accepted at threshold 1: every pass ran
    assert torch.all(out['confidence'][:, 1:] <= out['confidence'][:, :-1])
    assert (out['points'][:, 0, :2].norm(dim=-1) <= 6.).all()


def test_path_token_sets_and_absent_references_are_independent():
    model = small(flow=True).eval()
    b = coordinate_batch(model.cfg, 2)
    with torch.no_grad():
        ctx = model.context(b['x'], b['hist'], b['hmask'])
        tokens = torch.randn(2, 2, 8, 32)
        both = model.run_paths(ctx, tokens)
        changed = tokens.clone()
        changed[:, 1] += 1.
        torch.testing.assert_close(model.run_paths(ctx, changed)[:, 0], both[:, 0])  # sets never read each other
        padding = torch.zeros(2, 2, 8, dtype=torch.bool)
        padding[:, 0, 5:] = True
        padded = model.run_paths(ctx, tokens, padding)
        changed = tokens.clone()
        changed[:, 0, 5:] += 1.
        torch.testing.assert_close(model.run_paths(ctx, changed, padding)[:, 0, :5], padded[:, 0, :5])
        # Context tokens never read path tokens; absent references are not read at all.
        hmask = b['hmask'].clone()
        hmask[:, 10:] = 0
        hist = b['hist'].clone()
        reference = model.context(b['x'], hist, hmask)
        hist[:, 10:] += 3.
        moved = model.context(b['x'], hist, hmask)
        for (k0, v0), (k1, v1) in zip(reference['context'], moved['context']):
            visible = ~reference['padding']
            torch.testing.assert_close(k1[:, :, visible[0]], k0[:, :, visible[0]])
            torch.testing.assert_close(v1[:, :, visible[0]], v0[:, :, visible[0]])


@pytest.mark.parametrize('flow, options', [(False, {}), (True, {}), (True, dict(flow_time_conditioning='adaln')),
                                            (True, dict(flow_time_conditioning='adaln_zero'))])
def test_compiled_training_matches_eager_with_finite_gradients(flow, options):
    model = small(flow=flow, **options)
    b = coordinate_batch(model.cfg, 2)
    if flow:
        b['flow_noise'] = torch.randn(2, model.cfg.flow_draws, 8, 2)
        b['flow_times'] = torch.rand(2, model.cfg.flow_draws)
    eager = model.select_prediction(model.training_forward(b['x'], b['hist'], b['hmask'], torch.tensor(2.), b), .5)
    compiled = prepare_training(copy.deepcopy(model), 2, backend='eager')
    out = training_prediction(compiled, b['x'], b['hist'], b['hmask'], targets=b, retry_threshold=2.)
    for key in eager:
        torch.testing.assert_close(out[key], eager[key], atol=1e-5, rtol=1e-4)
    terms = loss_terms(out, b, compiled.cfg, refinement_loss='all', policy_threshold=.5)
    assert ('flow_per_state' in out) == flow
    (terms['geometry_per_state']+.5*terms['confidence_per_state']).mean().backward()
    heads = (compiled.velocity, compiled.score[0], compiled.hazard) if flow else (
        compiled.coordinates, compiled.hazard, compiled.refinement_fusion[0])
    for module in (compiled.cnn.stages[0].blocks[0].conv1.conv, compiled.cell_token, compiled.reference_token[0],
                   compiled.layers[0].qkv, compiled.layers[-1].ffn[0], *heads):
        assert module.weight.grad is not None and torch.isfinite(module.weight.grad).all()
        assert module.weight.grad.abs().sum() > 0


def test_configuration_defaults_run_configuration_and_checkpoint():
    cfg = RegressionConfig()
    assert (cfg.cnn_channels, cfg.cnn_blocks, cfg.layers, cfg.heads, cfg.ffn) == ((32, 64, 128), (1, 2, 2), 12, 8, 2048)
    assert cfg.token_stride == (8, 8, 8) and cfg.recurrent_refinement_steps == 3 and (cfg.n_future, cfg.gate_plane) == (35, 16)
    assert (cfg.fine.depth, cfg.fine.width, cfg.fine.behind) == (144, 104, 72)
    assert run_config.model_config(run_config.resolve(run_document('regression'))).to_dict() == dict(
        cfg.to_dict(), frame_checkpoint=run_config.model_defaults('regression')['frame_checkpoint'])
    assert config_from_checkpoint(dict(model_type='regression', model_cfg=cfg.to_dict())) == cfg
    flow = run_config.model_config(run_config.resolve(run_document('flow', cnn_channels=[16, 32], cnn_blocks=[1, 1])))
    assert type(flow) is FlowConfig and flow.token_stride == (4, 4, 4)
    assert (flow.flow_samples, flow.flow_time_conditioning, flow.flow_sigma_floor, flow.flow_loss) == (4, 'adaln_zero', 3., 'pseudo_huber')
    flow.flow_sigma = ((3., 3.),)*flow.n_future
    assert config_from_checkpoint(dict(model_type='flow', model_cfg=flow.to_dict())) == flow
    with pytest.raises(ValueError, match='Unknown model settings'):
        run_config.resolve(run_document('regression', memory='decisions'))
    with pytest.raises(ValueError, match='Unsupported model type'):
        config_from_checkpoint(dict(model_type='unified', model_cfg={}))


def test_tracing_runs_the_regression_model(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
    from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams
    import numpy as np
    model = small().eval()
    tracer = FiberTracer(model, ct_volume(tmp_path), model.cfg.fine, model.cfg.n_history,
                         TraceParams(n_commit=2, max_len=4., confidence=0.), device='cpu')
    try:
        paths, reasons = tracer.trace(np.array([[24., 24., 24.]]), np.array([[.3, .4, .8660254]]))
    finally:
        tracer.close()
    assert reasons == ['max_len'] and len(paths[0]) > 1


def test_adaln_starts_as_the_input_conditioned_flow_model_and_modulates_only_velocity_tokens():
    plain, adaln = small(flow=True), small(flow=True, flow_time_conditioning='adaln')
    missing, unexpected = adaln.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(('time_modulation', 'output_modulation')) for k in missing)
    b = coordinate_batch(adaln.cfg, 2)
    y, t = torch.randn(2, 3, 8, 2), torch.rand(2, 3)
    with torch.no_grad():
        ctx = adaln.context(b['x'], b['hist'], b['hmask'])
        torch.testing.assert_close(adaln.velocity_field(ctx, y, t), plain.velocity_field(ctx, y, t))  # zero init
        for linear in (*adaln.time_modulation, adaln.output_modulation):
            linear.weight.normal_(std=.05)
        modulated = adaln.velocity_field(ctx, y, t)
        assert not torch.allclose(modulated, plain.velocity_field(ctx, y, t))
        # Draws stay independent; proposal scores carry no time, so modulation never reaches them.
        torch.testing.assert_close(adaln.velocity_field(ctx, y[:, :1], t[:, :1]), modulated[:, :1], atol=1e-5, rtol=1e-4)
        points = adaln.to_points(y[:, 0])
        torch.testing.assert_close(adaln.hazard_logits(ctx, points), plain.hazard_logits(ctx, points))
    b['flow_noise'], b['flow_times'] = torch.randn(2, adaln.cfg.flow_draws, 8, 2), torch.rand(2, adaln.cfg.flow_draws)
    out = adaln.training_forward(b['x'], b['hist'], b['hmask'], torch.tensor(.5), b)
    out['flow_per_state'].sum().backward()
    assert all(linear.weight.grad.abs().sum() > 0 for linear in (*adaln.time_modulation, adaln.output_modulation))


def test_adaln_zero_velocity_blocks_start_as_the_identity_and_open_with_training():
    model = small(flow=True, flow_time_conditioning='adaln_zero')
    b = coordinate_batch(model.cfg, 2)
    y, t = torch.randn(2, 3, 8, 2), torch.rand(2, 3)
    moved = y.clone()
    moved[:, :, 3] += 2.
    others = [i for i in range(8) if i != 3]
    with torch.no_grad():
        ctx = model.context(b['x'], b['hist'], b['hmask'])
        # Identity blocks: a plane's velocity reads neither the other planes nor the crop tokens through attention.
        torch.testing.assert_close(model.velocity_field(ctx, moved, t)[:, :, others],
                                   model.velocity_field(ctx, y, t)[:, :, others])
        for linear in (*model.time_modulation, model.output_modulation):
            linear.weight.normal_(std=.05)
        assert not torch.allclose(model.velocity_field(ctx, moved, t)[:, :, others],
                                  model.velocity_field(ctx, y, t)[:, :, others])
    fresh = small(flow=True, flow_time_conditioning='adaln_zero')
    b['flow_noise'], b['flow_times'] = torch.randn(2, fresh.cfg.flow_draws, 8, 2), torch.rand(2, fresh.cfg.flow_draws)
    fresh.training_forward(b['x'], b['hist'], b['hmask'], torch.tensor(.5), b)['flow_per_state'].sum().backward()
    assert all(m.weight.grad.abs().sum() > 0 for m in fresh.time_modulation)  # the gates learn from the first step
