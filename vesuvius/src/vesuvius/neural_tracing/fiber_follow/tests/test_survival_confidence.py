"""Survival confidence: conditional products, censored first-failure likelihoods and loss semantics."""
import math

import torch

from model_fixtures import config
from model_fixtures import coordinate_batch as memory_batch
from model_fixtures import proposal_output
from label_fixtures import set_terminal, set_unknown
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.models.survival_confidence import survival_loss, survival_predictions


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
