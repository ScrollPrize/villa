"""Fixed compiler contracts without changing adaptive losses or stream gradients."""
import copy

import pytest
import torch

from model_fixtures import config as cfg, slab_batch as memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    prepare_training, optimizer_update, training_prediction,
    begin_training_update, finish_training_update,
)
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms


def predict(model, *args, **kwargs):
    # The adaptive inference implementation is an independent numerical oracle.
    return (training_prediction(model, *args, **kwargs) if hasattr(model, 'training_batch_size')
            else model(*args, **kwargs))


def compare_gradients(left, right):
    for (name, p), (_, q) in zip(left.named_parameters(), right.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6, msg=name)










def test_partial_acceptance_keeps_policy_and_geometry_gradients():
    torch.manual_seed(115)
    eager = build_model(cfg(recurrent_refinement_steps=2))
    raw = copy.deepcopy(eager)
    compiled = prepare_training(raw, backend='eager')
    batch = memory_batch(eager.cfg, 2)
    with torch.no_grad():
        confidence = eager(batch['x'], batch['hist'], batch['hmask'])['refinement_confidence'][:, 0, -1]
    assert confidence[0] != confidence[1]
    threshold = float(confidence.mean())
    predictions = []
    for model in (eager, compiled):
        out = predict(model, batch['x'], batch['hist'], batch['hmask'], confidence_threshold=threshold)
        terms = loss_terms(out, batch, eager.cfg)
        (terms['geometry_per_state'].sum()+terms['confidence_per_state'].sum()).backward()
        predictions.append(out)
    assert predictions[0]['refinement_mask'][:, 1].sum() == 1
    used = predictions[0]['refinement_mask'].shape[1]
    assert torch.equal(predictions[0]['refinement_mask'], predictions[1]['refinement_mask'][:, :used])
    assert torch.equal(predictions[0]['selected_refinement'], predictions[1]['selected_refinement'])
    torch.testing.assert_close(predictions[0]['points'], predictions[1]['points'], rtol=1e-5, atol=1e-6)
    compare_gradients(eager, raw)




def test_masked_retries_preserve_adamw_skipped_parameter_updates():
    torch.manual_seed(113)
    eager = build_model(cfg(recurrent_refinement_steps=2))
    raw = copy.deepcopy(eager)
    compiled = prepare_training(raw, backend='eager')
    prepare_training(eager, backend='eager')
    models = (eager, compiled)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1) for model in models]
    emas = [build_model(eager.cfg), build_model(eager.cfg)]
    for ema in emas:
        ema.load_state_dict(raw.state_dict())
    # Populate identical momentum, then accept every initial proposal. A zero
    # grad would incorrectly decay parameters and advance optimizer moments.
    for model, opt in zip(models, optimizers):
        for module in (model.refinement_fusion, model.refinement_stage):
            for p in module.parameters():
                p.grad = torch.ones_like(p)
        opt.step()
    protected = {name: p.detach().clone() for name, p in raw.named_parameters() if name.startswith('refinement_')}
    data = [memory_batch(eager.cfg)]
    for model, opt, ema in zip(models, optimizers, emas):
        with torch.no_grad():
            model.confidence_scorer.failure.weight.zero_()
            model.confidence_scorer.failure.bias.fill_(-20.)
        optimizer_update(model, ema, opt, data, 2, .001, compute_metrics=False)
    compare_gradients(eager, raw)
    for name, p in raw.named_parameters():
        if name in protected:
            torch.testing.assert_close(p, protected[name], rtol=0, atol=0)
            assert optimizers[1].state[p]['step'] == 1
    assert all(p.grad is None for p in raw.refinement_fusion.parameters())




def test_fixed_rows_avoids_full_crop_copy_but_normalizes_singleton_strides():
    from vesuvius.neural_tracing.fiber_follow.regression.train import fixed_rows
    value = torch.randn(2, 2, 8, 9, 9, requires_grad=True)
    assert fixed_rows(value, 2) is value
    indexed = torch.randn(3, 2, 1, 3).transpose(1, 2)
    assert indexed.is_contiguous()  # Singleton strides still differ from canonical.
    result = fixed_rows(indexed, 3)
    assert result.stride() == (6, 6, 3, 1)
    torch.testing.assert_close(result, indexed)
    padded = fixed_rows(value[:1], 2)
    padded.sum().backward()
    torch.testing.assert_close(value.grad[0], torch.full_like(value[0], 2.))
    assert value.grad[1].eq(0).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_compiled_loss_preserves_terms_and_prediction_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    torch.manual_seed(341)
    model = build_model(cfg(encoder='patch4', token_only=True, recurrent_refinement_steps=1)).to(device)
    batch = move_batch(memory_batch(model.cfg), device)
    with torch.no_grad():
        prediction = model(batch['x'], batch['hist'], batch['hmask'])
    prediction = {k: v.detach().requires_grad_(v.is_floating_point()) for k, v in prediction.items()}
    compiled = torch.compile(loss_terms, fullgraph=True, dynamic=False,
                             options=dict(emulate_precision_casts=True))
    inputs = [v for v in prediction.values() if v.requires_grad]
    terms, gradients = [], []
    for loss in (loss_terms, compiled):
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
            result = loss(prediction, batch, model.cfg)
            value = (result['geometry_per_state']+.5*result['confidence_per_state']).sum()
        terms.append(result)
        gradients.append(torch.autograd.grad(value, inputs, allow_unused=True))
    for key in terms[0]:
        torch.testing.assert_close(terms[0][key], terms[1][key], rtol=3e-5, atol=2e-6)
    for a, b in zip(*gradients):
        if a is None:
            # AOTAutograd can materialize zeros for metric-only outputs.
            assert b is None or b.eq(0).all()
        else:
            torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
