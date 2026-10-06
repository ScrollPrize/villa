"""Training updates: compiled losses match eager, skipped refinement passes leave AdamW state alone, and
refinement_loss 'all' trains every pass."""
import torch

from model_fixtures import coordinate_batch, coordinate_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.train.train import prepare_training, optimizer_update, training_prediction
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms


def test_masked_retries_preserve_adamw_skipped_parameter_updates():
    torch.manual_seed(113)
    model = prepare_training(build_model(coordinate_config(recurrent_refinement_steps=2)), backend='eager')
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
        model.hazard.weight.zero_()
        model.hazard.bias.fill_(-20.)
    optimizer_update(model, ema, opt, [coordinate_batch(model.cfg)], 2, .001, compute_metrics=False)
    for name, p in model.named_parameters():
        if name in protected:
            torch.testing.assert_close(p, protected[name], rtol=0, atol=0)
            assert opt.state[p]['step'] == 1
    assert all(p.grad is None for p in model.refinement_fusion.parameters())
    assert not torch.equal(trained, model.coordinates.weight)


def test_compiled_loss_preserves_terms_and_prediction_gradients():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from vesuvius.neural_tracing.fiber_follow.train.train import move_batch
    torch.manual_seed(341)
    model = build_model(coordinate_config(recurrent_refinement_steps=1)).to(device)
    batch = move_batch(coordinate_batch(model.cfg), device)
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


def test_refinement_loss_all_keeps_initial_weight_and_reports_policy_attempts():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_follow.train.train import (
        REFINE_ALL, initialize_model_weights, initialize_training_optimizer, optimizer_update)
    torch.manual_seed(115)
    model = build_model(coordinate_config(recurrent_refinement_steps=2))
    batch = coordinate_batch(model.cfg, 2)
    with torch.no_grad():
        confidence = model(batch['x'], batch['hist'], batch['hmask'])['refinement_confidence'][:, 0, -1]
    threshold = float(confidence.mean())
    model = prepare_training(model, backend='eager')
    args = (model, batch['x'], batch['hist'], batch['hmask'], threshold)
    policy = training_prediction(*args)
    every = training_prediction(*args, retry_threshold=REFINE_ALL)
    assert every['refinement_mask'].all()
    old = loss_terms(policy, batch, model.cfg, 2.)
    initial_only = loss_terms(every, batch, model.cfg, 2., refinement_loss='all', refinement_weight=0., policy_threshold=threshold)
    accepted = int(confidence.argmax())
    for key in ('geometry_per_state', 'confidence_per_state'):
        torch.testing.assert_close(initial_only[key][accepted], old[key][accepted])  # accepted outright: same loss
    # The attempt metric counts what the operating policy runs, not the training-only passes.
    assert int(initial_only['refinement_attempts_sum']) == int(policy['refinement_mask'].sum())
    full = loss_terms(every, batch, model.cfg, 2., refinement_loss='all', refinement_weight=1., policy_threshold=threshold)
    (full['geometry_per_state'].sum()+full['confidence_per_state'].sum()).backward()
    assert model.refinement_stage.weight.grad[1].abs().sum() > 0  # the last pass trains even for the accepted row
    torch.manual_seed(2)
    trained, ema = initialize_model_weights(coordinate_config(recurrent_refinement_steps=2), 'cpu')
    opt, _, _ = initialize_training_optimizer(trained, ema, SimpleNamespace(lr=.001, reset_optimizer=False))
    prepare_training(trained, backend='eager')
    metrics = optimizer_update(trained, ema, opt, [coordinate_batch(trained.cfg, 2)], 1, .001, tolerance=2.,
                               confidence_threshold=.4, gate='full', refinement_loss='all')
    assert torch.isfinite(torch.tensor(metrics['loss'])) and 1 <= metrics['refinement_attempts_mean'] <= 3
