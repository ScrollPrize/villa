"""Unified crop model (models/unified.py): CNN tokens and one transformer, as regression ('unified') or flow."""
import copy

import pytest
import torch

from model_fixtures import REQUIRED, coordinate_batch
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.unified import UnifiedConfig, UnifiedFlowConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import (
    build_parser, checkpoint_config, model_config_from_args, prepare_training, training_prediction)

SMALL = dict(fine=CropSpec(depth=32, width=16, behind=8, spacing=.5), n_future=8, gate_plane=4, hidden=32, layers=2,
             heads=2, ffn=64, cnn_channels=(4, 8, 16), cnn_blocks=(1, 1, 1), n_history=32)


def small(flow=False, **options):
    torch.manual_seed(0)
    if flow:
        cfg = UnifiedFlowConfig(**dict(SMALL, flow_steps=2, flow_draws=3, flow_sigma=((1., 1.),)*8, **options))
    else:
        cfg = UnifiedConfig(**dict(SMALL, recurrent_refinement_steps=2, **options))
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


@pytest.mark.parametrize('flow', [False, True])
def test_compiled_training_matches_eager_with_finite_gradients(flow):
    model = small(flow=flow)
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


def test_configuration_from_arguments_and_checkpoint():
    args = build_parser().parse_args(REQUIRED+['--model', 'unified', '--crop-depth', '144', '--crop-width', '104',
                                               '--crop-behind', '72', '--n-future', '35', '--gate-plane', '16',
                                               '--recurrent-refinement-steps', '3'])
    cfg = model_config_from_args(args)
    assert (cfg.cnn_channels, cfg.cnn_blocks, cfg.layers, cfg.heads, cfg.ffn) == ((32, 64, 128), (1, 2, 2), 12, 8, 2048)
    assert cfg.token_stride == (8, 8, 8) and cfg.recurrent_refinement_steps == 3 and cfg.memory == 'none'
    assert checkpoint_config(dict(model_type='unified', model_cfg=cfg.to_dict())) == cfg
    args = build_parser().parse_args(REQUIRED+['--model', 'unified_flow', '--unified-cnn-channels', '16', '32',
                                               '--unified-cnn-blocks', '1', '1'])
    flow = model_config_from_args(args)
    assert type(flow) is UnifiedFlowConfig and flow.token_stride == (4, 4, 4)
    flow.flow_sigma = ((1., 1.),)*flow.n_future
    assert checkpoint_config(dict(model_type='unified_flow', model_cfg=flow.to_dict())) == flow
    with pytest.raises(ValueError):
        UnifiedConfig(memory='decisions')


def test_tracing_runs_the_unified_model(tmp_path):
    from test_rollout_threading import ct_volume
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
