"""Fixed compiler contracts without changing adaptive losses or stream gradients."""
import copy

import torch

from model_fixtures import aligned_batch, aligned_config
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model
from vesuvius.neural_tracing.fiber_follow.regression.train import (
    prepare_training, optimizer_update, training_prediction,
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
    eager = build_model(aligned_config(recurrent_refinement_steps=2))
    raw = copy.deepcopy(eager)
    compiled = prepare_training(raw, backend='eager')
    batch = aligned_batch(eager.cfg, 2)
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
    model = prepare_training(build_model(aligned_config(recurrent_refinement_steps=2)), backend='eager')
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.1)
    ema = build_model(model.cfg)
    ema.load_state_dict(model.state_dict())
    # Populate momentum, then accept every initial proposal. A zero grad would
    # incorrectly decay parameters and advance optimizer moments.
    for module in (model.refinement_fusion, model.refinement_stage):
        for p in module.parameters():
            p.grad = torch.ones_like(p)
    opt.step()
    protected = {name: p.detach().clone() for name, p in model.named_parameters() if name.startswith('refinement_')}
    trained = model.coordinates.weight.detach().clone()
    with torch.no_grad():
        model.confidence_scorer.failure.weight.zero_()
        model.confidence_scorer.failure.bias.fill_(-20.)
    optimizer_update(model, ema, opt, [aligned_batch(model.cfg)], 2, .001, compute_metrics=False)
    for name, p in model.named_parameters():
        if name in protected:
            torch.testing.assert_close(p, protected[name], rtol=0, atol=0)
            assert opt.state[p]['step'] == 1
    assert all(p.grad is None for p in model.refinement_fusion.parameters())
    assert not torch.equal(trained, model.coordinates.weight)


def test_compiled_loss_preserves_terms_and_prediction_gradients():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from vesuvius.neural_tracing.fiber_follow.regression.train import move_batch
    torch.manual_seed(341)
    model = build_model(aligned_config(recurrent_refinement_steps=1)).to(device)
    batch = move_batch(aligned_batch(model.cfg), device)
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
