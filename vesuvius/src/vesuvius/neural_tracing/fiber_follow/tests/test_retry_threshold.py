"""--retry-threshold: training refinement passes follow their own threshold; selection keeps the operating one."""
import torch

from model_fixtures import coordinate_batch, coordinate_config
from vesuvius.neural_tracing.fiber_follow.models.model import build_model
from vesuvius.neural_tracing.fiber_follow.train.train import prepare_training, training_prediction


def test_retry_threshold_trains_more_retries_without_changing_selection():
    torch.manual_seed(115)
    model = build_model(coordinate_config(recurrent_refinement_steps=2))
    batch = coordinate_batch(model.cfg, 2)
    with torch.no_grad():
        confidence = model(batch['x'], batch['hist'], batch['hmask'])['refinement_confidence'][:, 0, -1]
    threshold = float(confidence.mean())  # the operating policy accepts one row's first proposal
    model = prepare_training(model, backend='eager')
    args = (model, batch['x'], batch['hist'], batch['hmask'], threshold)
    with torch.no_grad():
        policy = training_prediction(*args)
        retries = training_prediction(*args, retry_threshold=1.)
    assert policy['refinement_mask'][:, 1].sum() == 1  # only the rejected row retried
    assert retries['refinement_mask'][:, 1].all()  # both rows retried
    accepted = int(confidence.argmax())
    assert policy['selected_refinement'][accepted] == retries['selected_refinement'][accepted] == 0


def test_refinement_loss_all_keeps_initial_weight_and_reports_policy_attempts():
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
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
