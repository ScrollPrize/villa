"""Prefix invariance, unrestricted observation reads and censored likelihoods."""
import math

import pytest
import torch

from model_fixtures import config
from model_fixtures import slab_batch as memory_batch
from model_fixtures import proposal_output
from label_fixtures import set_terminal, set_unknown
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.survival_confidence import (
    survival_loss, survival_predictions,
)


def scoring_context(model, batch):
    ctx = model.context(batch['x'], batch['hist'], batch['hmask'])
    model.decoder_memory(ctx)
    return ctx


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_confidence_head_computes_fp32_inside_bf16_autocast(device):
    torch.manual_seed(65)
    cfg = config()
    scorer = build_model(cfg).to(device).confidence_scorer
    width = 27*(cfg.channels+1)+cfg.hidden+1
    spatial = torch.randn(2, 4, 4, width, device=device, requires_grad=True)
    points = torch.zeros(2, 4, 3, device=device)
    points[..., 2] = torch.arange(1, 5, device=device)
    memory = torch.randn(2, 9, cfg.hidden, device=device, requires_grad=True)
    padding = torch.zeros(2, 9, device=device, dtype=torch.bool)
    inputs = {}
    def capture(name):
        def hook(module, args):
            assert not torch.is_autocast_enabled(device)
            assert args[0].dtype == torch.float32
            inputs[name] = args[0].detach()
        return hook
    hooks = [scorer.norm.register_forward_pre_hook(capture('norm')),
             scorer.failure.register_forward_pre_hook(capture('failure'))]
    try:
        with torch.autocast(device, dtype=torch.bfloat16):
            logits = scorer(spatial, points, scorer.project_memory(memory, padding), padding, (memory, padding))
    finally:
        for hook in hooks:
            hook.remove()
    expected = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(inputs['norm'], scorer.norm.normalized_shape,
                                       scorer.norm.weight, scorer.norm.bias, scorer.norm.eps),
        scorer.failure.weight, scorer.failure.bias).squeeze(-1)
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert not torch.equal(logits, logits.bfloat16().float())
    values = (scorer.norm.weight, scorer.failure.weight)
    # CPU oneDNN BF16 attention backward is not supported on every host.
    # CUDA additionally checks gradients through the surrounding BF16 scorer.
    if device == 'cuda':
        values += (spatial, memory)
    for grad in torch.autograd.grad(logits.square().mean(), values):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_replacing_truncating_or_extending_suffix_preserves_prefix():
    torch.manual_seed(61)
    model = build_model(config(scorer_layers=4)).train(True)
    first = memory_batch(model.cfg, 1)
    with torch.no_grad():
        batch = memory_batch(model.cfg, 1, step=1)
        ctx = scoring_context(model, batch)
        curve = torch.zeros(1, 4, 3)
        curve[..., 2] = torch.arange(1, 5)
        curve[0, :, 0] = torch.tensor([.2, -.4, .7, 1.])
        baseline = model.hazard_logits(ctx, curve)
        changed = curve.clone()
        changed[:, 2:, :2] = torch.tensor([[2., -2.], [-3., 3.]])
        alternative = model.hazard_logits(ctx, changed)
        torch.testing.assert_close(baseline[:, :2], alternative[:, :2], rtol=0, atol=0)
        assert not torch.equal(baseline[:, 2:], alternative[:, 2:])
        # Different sequence lengths may select different floating-point kernels.
        short = model.hazard_logits(ctx, curve[:, :2])
        torch.testing.assert_close(baseline[:, :2], short, rtol=1e-5, atol=2e-6)
        extension = torch.cat((curve, torch.tensor([[[2., 1., 5.], [3., -1., 6.]]])), 1)
        extended = model.hazard_logits(ctx, extension)
        torch.testing.assert_close(baseline, extended[:, :4], rtol=1e-5, atol=2e-6)
        for other in (alternative[:, :2], short, extended[:, :2]):
            torch.testing.assert_close(survival_predictions(baseline[:, :2])[1],
                                       survival_predictions(other)[1], rtol=1e-5, atol=2e-6)


def test_segment_reads_cover_interior_and_do_not_receive_generator_state(monkeypatch):
    model = build_model(config()).eval()
    batch = memory_batch(model.cfg, 1)
    ctx = scoring_context(model, batch)
    curve = torch.tensor([[[1., 0., 1.], [3., 2., 2.], [2., 0., 3.], [0., 0., 4.]]], requires_grad=True)
    sampled = []
    original = model.evidence
    def capture(ctx, points, stage):
        sampled.append(points.detach().clone())
        return original(ctx, points, stage)
    monkeypatch.setattr(model, 'evidence', capture)
    logits = model.confidence_logits(ctx, torch.full((1, 4, model.cfg.hidden), float('nan')), curve)
    torch.testing.assert_close(sampled[0][0, 4:8], torch.tensor([
        [1.5, .5, 1.25], [2., 1., 1.5], [2.5, 1.5, 1.75], [3., 2., 2.]]))
    logits.sum().backward()
    assert curve.grad is None
    assert all(p.grad is None for p in model.decoder.parameters())
    assert model.coordinates.weight.grad is None
    assert model.encoder.stem[0].weight.grad.abs().sum() > 0


def test_first_segment_reads_all_observations_but_no_future_segment_features():
    torch.manual_seed(62)
    model = build_model(config())
    scorer = model.confidence_scorer
    width = 27*(model.cfg.channels+1)+model.cfg.hidden+1
    spatial = torch.randn(1, 4, 4, width, requires_grad=True)
    points = torch.zeros(1, 4, 3)
    points[..., 2] = torch.arange(1, 5)
    memory = torch.randn(1, 9, model.cfg.hidden, requires_grad=True)
    padding = torch.zeros(1, 9, dtype=torch.bool)
    padding[:, -1] = True
    logits = scorer(spatial, points, scorer.project_memory(memory, padding), padding, (memory, padding))
    logits[:, 0].sum().backward()
    assert len(scorer.layers) == 2
    for layer in scorer.layers:
        grad = layer.linear1.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert spatial.grad[:, 0].abs().sum() > 0
    assert spatial.grad[:, 1:].count_nonzero() == 0
    assert (memory.grad[:, :-1].abs().sum(-1) > 0).all()
    assert memory.grad[:, -1].count_nonzero() == 0


def test_survival_probabilities_are_conditional_products_and_numerically_stable():
    hazards = torch.tensor([[.1, .2, .3, .4]])
    logits, confidence = survival_predictions(torch.logit(hazards))
    torch.testing.assert_close(confidence, torch.tensor([[.9, .72, .504, .3024]]))
    torch.testing.assert_close(logits.sigmoid(), confidence)
    assert (confidence[:, 1:] <= confidence[:, :-1]).all()
    extreme = torch.tensor([[-1000., -100., 0., 100., 1000.]], requires_grad=True)
    logits, confidence = survival_predictions(extreme)
    assert torch.isfinite(logits).all() and torch.isfinite(confidence).all()
    logits.sum().backward()
    assert torch.isfinite(extreme.grad).all()


def test_first_failure_censoring_and_missing_gaps_have_exact_likelihood_and_gradients():
    hazards = torch.logit(torch.tensor([[.1, .2, .3, .4]])).repeat(5, 1).requires_grad_()
    labels = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.],
                           [1., 1., 0., 0.], [0., 0., 0., 0.]])
    known = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1],
                          [1, 0, 1, 1], [0, 0, 0, 0]], dtype=torch.bool)
    loss, valid = survival_loss(hazards, labels, known)
    expected = torch.tensor([-math.log(.9*.8*.3), -math.log(.9*.8), -math.log(.1), -math.log(.9), 0.])
    torch.testing.assert_close(loss, expected)
    assert valid.tolist() == [[True, True, True, False], [True, True, False, False],
                              [True, False, False, False], [True, False, False, False], [False]*4]
    loss.sum().backward()
    assert hazards.grad[~valid].count_nonzero() == 0
    assert (hazards.grad[valid] != 0).all()


def test_generated_losses_keep_survival_semantics_and_mask_unavailable_supervision():
    cfg = config()
    batch = memory_batch(cfg, 2)
    points = torch.zeros(2, cfg.n_future, 3)
    points[..., 2] = torch.arange(1, cfg.n_future+1)
    points.requires_grad_()
    hazards = torch.zeros(2, cfg.n_future, requires_grad=True)
    set_terminal(batch, 0)
    set_unknown(batch, 1)
    out = proposal_output(points[:, None], hazards[:, None])
    terms = loss_terms(out, batch, cfg, n_commit=1)
    torch.testing.assert_close(terms['confidence_per_state'], torch.tensor([math.log(2), 0.]))
    # Unknown (unobservable) states teach nothing; terminal states teach no geometry.
    assert terms['geometry_per_state'].eq(0).all()
    (terms['confidence_per_state'].sum()+terms['geometry_per_state'].sum()).backward()
    assert hazards.grad[0, 0] < 0 and hazards.grad[:, 1:].count_nonzero() == 0
    assert hazards.grad[1].count_nonzero() == 0
    assert points.grad is None or points.grad.count_nonzero() == 0
