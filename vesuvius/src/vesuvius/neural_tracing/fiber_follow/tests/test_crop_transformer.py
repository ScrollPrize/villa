"""Crop transformer (models/crop_transformer.py): CNN tokens and one transformer, as regression or flow."""
import copy

import pytest
import torch

from model_fixtures import coordinate_batch, ct_volume
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.models.crop_transformer import FlowConfig, RegressionConfig
from vesuvius.neural_tracing.fiber_follow.shared.geometry import CropSpec
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.train.train import prepare_training, training_prediction

SMALL = dict(fine=CropSpec(depth=32, width=16, behind=8, spacing=.5), n_future=8, gate_plane=4, hidden=32, layers=2,
             heads=2, ffn=64, cnn_channels=(4, 8, 16), cnn_blocks=(1, 1, 1), n_history=32)
# A small flow model with one path.
PLAIN_FLOW = dict(flow_steps=2, flow_draws=3, flow_sigma=((1., 1.),)*8, flow_samples=0, flow_sigma_floor=1.)


def small(flow=False, **options):
    torch.manual_seed(0)
    if flow:
        cfg = FlowConfig(**dict(SMALL, **dict(PLAIN_FLOW, **options)))
    else:
        cfg = RegressionConfig(**dict(SMALL, recurrent_refinement_steps=2, **options))
    return build_model(cfg)


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


def test_adaln_zero_velocity_blocks_start_as_the_identity_and_open_with_training():
    model = small(flow=True)
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
    fresh = small(flow=True)
    b['flow_noise'], b['flow_times'] = torch.randn(2, fresh.cfg.flow_draws, 8, 2), torch.rand(2, fresh.cfg.flow_draws)
    fresh.training_forward(b['x'], b['hist'], b['hmask'], torch.tensor(.5), b)['flow_per_state'].sum().backward()
    assert all(m.weight.grad.abs().sum() > 0 for m in fresh.time_modulation)  # the gates learn from the first step


def test_scoring_pass_scores_each_finished_proposal_without_steering_it():
    model = small(scoring_pass=True).eval()
    b = coordinate_batch(model.cfg, 2)
    with torch.no_grad():
        out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
        changed = copy.deepcopy(model)
        changed.score[2].weight.mul_(-1.)  # only the scoring tokens' input changes
        out2 = changed(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
    # The initial proposal does not depend on its score (refinement passes read it back as feedback, by design).
    torch.testing.assert_close(out2['initial_points'], out['initial_points'])
    assert not torch.allclose(out2['confidence'], out['confidence'])  # the confidence comes from the scoring pass
