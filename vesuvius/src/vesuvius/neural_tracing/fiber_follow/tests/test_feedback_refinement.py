"""Adaptive absolute proposals, detached feedback and threshold-aware selection."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from model_fixtures import coordinate_batch, coordinate_config, fake_ct, proposal_output
from vesuvius.neural_tracing.fiber_follow.models.model import build_model, select_refinement
from vesuvius.neural_tracing.fiber_follow.train.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.evaluation.diagnostics import decision_rows
from vesuvius.neural_tracing.fiber_follow.data.observations import FiberTracer
from vesuvius.neural_tracing.fiber_follow.tracing.policy import commit_prefix
from vesuvius.neural_tracing.fiber_follow.tracing.trace import TraceParams


def policy_output():
    curves = torch.zeros(1, 3, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    curves[0, :, :, 0] = torch.tensor([0., 2., 4.])[:, None]
    confidence = torch.tensor([[[.99, .8, .7, .1], [.95, .9, .4, .3], [.9, .85, .75, .2]]])
    previous = torch.cat((torch.ones_like(confidence[..., :1]), confidence[..., :-1]), -1)
    return proposal_output(curves, torch.logit(1-confidence/previous))


def test_selection_uses_actual_threshold_horizon_and_matching_scores():
    output, c = policy_output(), coordinate_config()
    for threshold, window, expected in ((.5, 4, 2), (.5, 1, 0), (.79, 4, 1), (1., 4, 0)):
        selected = select_refinement(output, c, threshold, window)
        assert selected['selected_refinement'].item() == expected
        for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
            torch.testing.assert_close(selected[name], output['refinement_'+name][:, expected])
    stopped = select_refinement(output, c, 1., 4)
    assert commit_prefix(stopped['points'], stopped['confidence'], 1., 4)[0].item() == 0
    # Diagnostics label the proposal the policy chose at each threshold.
    rows = decision_rows(output, coordinate_batch(c, 1), c, n_commit=1, thresholds=(.5, 1.))
    assert rows[0]['gate_0.5']['accepted_wrong'] == 0
    assert rows[0]['gate_1.0']['false_stops'] == 1
    # An invalid connection cannot win even with the longest confident prefix.
    output['refinement_points'][:, 2, :, 0] = 20.
    assert select_refinement(output, c)['selected_refinement'].item() == 0
    # Equal prefix length and confidence retain the earlier proposal.
    output = policy_output()
    output['refinement_confidence'][:, 2] = output['refinement_confidence'][:, 0]
    assert select_refinement(output, c)['selected_refinement'].item() == 0


def hazards(model, monkeypatch, values):
    """Replace the hazard head's output: ``values(decoded, call)`` -> (B, P) logits."""
    calls = []
    def forward(decoded):
        calls.append(len(decoded))
        return values(decoded, len(calls)-1)[..., None]
    monkeypatch.setattr(model.hazard, 'forward', forward)
    return calls


def test_retries_run_for_every_slot_and_mask_attempts_after_acceptance(monkeypatch):
    model = build_model(coordinate_config(recurrent_refinement_steps=4))
    b = coordinate_batch(model.cfg, 3)
    accepted = []
    def values(decoded, call):
        out = decoded.new_full(decoded.shape[:2], 2.)
        out[accepted] = -12.
        return out
    calls = hazards(model, monkeypatch, values)
    # Every pass runs for every row (fixed slots); an accepted row keeps its first attempt.
    accepted[:] = [0, 1, 2]
    out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=.9)
    assert calls == [3]*5
    assert out['refinement_mask'].tolist() == [[True, False, False, False, False]]*3
    torch.testing.assert_close(out['points'], out['initial_points'])
    # Row i is accepted from pass i on.
    calls.clear()
    def values(decoded, call):
        out = decoded.new_full(decoded.shape[:2], 2.)
        out[:call+1] = -12.
        return out
    hazards(model, monkeypatch, values)
    out = model(b['x'], b['hist'], b['hmask'])
    assert out['refinement_mask'].tolist() == [[True, False, False, False, False], [True, True, False, False, False],
                                               [True, True, True, False, False]]
    assert out['selected_refinement'].tolist() == [0, 1, 2]
    terms = loss_terms(out, b, model.cfg)
    # Loss for each state agrees with evaluating only its actual attempts.
    for i, attempts in enumerate((1, 2, 3)):
        row = {k: v[i:i+1] for k, v in b.items() if torch.is_tensor(v)}
        prediction = {k: v[i:i+1] for k, v in out.items()}
        for key in ('refinement_points', 'refinement_hazard_logits', 'refinement_confidence_logits',
                    'refinement_confidence', 'refinement_mask'):
            prediction[key] = prediction[key][:, :attempts]
        expected = loss_terms(prediction, row, model.cfg)
        for key in ('geometry_per_state', 'confidence_per_state'):
            torch.testing.assert_close(terms[key][i:i+1], expected[key])
    terms['geometry_per_state'].sum().backward()
    assert model.coordinates.weight.grad.abs().sum() > 0


def test_partial_acceptance_uses_budget_and_absolute_head_can_replace_path(monkeypatch):
    model = build_model(coordinate_config(recurrent_refinement_steps=2))
    b = coordinate_batch(model.cfg, 1)
    calls = []
    def coordinates(decoded):
        calls.append(1)
        return decoded.new_full((*decoded.shape[:2], 2), (-1 if len(calls) == 1 else 1)*1.1)
    monkeypatch.setattr(model.coordinates, 'forward', coordinates)
    # First point accepted, later prefix rejected. A one-point commit does not
    # make the entire prediction acceptable and must not suppress retries.
    hazards(model, monkeypatch, lambda decoded, call: decoded.new_tensor([[-12., 2., 2., 2.]]))
    out = model(b['x'], b['hist'], b['hmask'], n_commit=1)
    assert len(calls) == 3 and out['refinement_mask'].all()
    curves = out['refinement_points']
    assert (curves[:, 1, 1:, :2]-curves[:, 0, 1:, :2]).norm(dim=-1).min() > 4.
    torch.testing.assert_close(curves[:, 1], curves[:, 2])  # Absolute, not accumulated offsets.
    assert curves[..., :2].abs().max() <= model.cfg.lateral_limit
    assert curves[:, :, 0].norm(dim=-1).max() <= model.cfg.max_recovery_distance+1e-5


def test_feedback_changes_new_proposal_but_geometry_cannot_train_scores(monkeypatch):
    torch.manual_seed(93)
    model = build_model(coordinate_config(recurrent_refinement_steps=1))
    b = coordinate_batch(model.cfg, 1)
    outputs, scores = [], []
    for value in (-1., 3.):
        def score(decoded, call, value=value):
            result = decoded.new_full(decoded.shape[:2], value).detach().requires_grad_()
            scores.append(result)
            return result
        hazards(model, monkeypatch, score)
        out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
        outputs.append(out)
        loss_terms(out, b, model.cfg)['geometry_per_state'].sum().backward()
    torch.testing.assert_close(outputs[0]['initial_points'], outputs[1]['initial_points'])
    assert not torch.equal(outputs[0]['refinement_points'][:, 1], outputs[1]['refinement_points'][:, 1])
    assert all(score.grad is None for score in scores)


def test_tracer_passes_operating_threshold_into_adaptive_model(monkeypatch):
    fake_ct(monkeypatch)
    model = build_model(coordinate_config(recurrent_refinement_steps=2)).eval()
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.data.observations.image_crop',
        lambda items, vol, crop, pool=None, **kw: torch.ones(len(items), 1, crop.depth, crop.width, crop.width)*.25)
    thresholds = []
    original = model.forward
    def forward(*args, **kwargs):
        thresholds.append((kwargs['confidence_threshold'], kwargs['n_commit']))
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', forward)
    # The prefix gate ranks proposals by the committed prefix; the full gate (default) by the full horizon,
    # because only the last plane decides acceptance.
    for gate, window in (('prefix', 1), ('full', model.cfg.n_future)):
        thresholds.clear()
        tracer = FiberTracer(model, SimpleNamespace(shape=(1000, 1000, 1000)), model.cfg.fine, model.cfg.n_history,
                             TraceParams(n_commit=1, max_len=1., confidence=0., gate=gate), device='cpu')
        try:
            tracer.trace(np.array([[100., 100., 400.]]), np.array([[0., 0., 1.]]))
        finally:
            tracer.close()
        assert thresholds and all(value == (0., window) for value in thresholds)


def test_recovery_evaluator_reruns_policy_per_threshold_on_float_inputs_and_observed_states(monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.ct_tensor',
                        lambda vol, pos: np.outer(np.array([1., 0., 0.]), np.array([1., 0., 0.])))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.tracing.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    from vesuvius.neural_tracing.fiber_follow.data.data import TracedFiber, SampleConfig
    from vesuvius.neural_tracing.fiber_follow.evaluation.recovery_fixtures import make_recovery_states, evaluate_recovery_states
    c = coordinate_config()
    model = build_model(c)
    sample = SampleConfig(crop=c.fine, n_history=c.n_history, recent_history_points=c.n_history, n_future=c.n_future)
    arc = np.arange(300, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    states = make_recovery_states([fiber], [dict(fiber=0, t=150., sign=1)], sample, dict(split='monitor', volume={}), None)
    inputs = coordinate_batch(c, 1)
    # The crop builder can return float16; floating inputs enter the model in float32.
    inputs['x'] = {k: v.half() if v.is_floating_point() else v for k, v in inputs['x'].items()}
    calls = []
    def forward(x, hist, hmask, *, confidence_threshold, n_commit):
        assert all(v.dtype != torch.float16 for v in x.values())
        calls.append((confidence_threshold, n_commit))
        return select_refinement(policy_output(), c, confidence_threshold, n_commit)
    monkeypatch.setattr(model, 'forward', forward)
    traced, closed, audited = [], [], []
    class Tracer:
        def __init__(self, *args, **kwargs):
            pass
        def trace(self, pos, heading, initial_states):
            traced.append(initial_states[0])
            return [pos], ['confidence']
        def close(self):
            closed.append(True)
    rows, predictions = evaluate_recovery_states(model, None, states, [fiber], sample, limit=1,
        thresholds=(.5, .79), n_commit=4, tracer_class=Tracer, batch_builder=lambda items, vol: inputs,
        on_prediction=lambda out, b: audited.append(out))
    assert calls == [(.5, 4), (.79, 4)]
    assert rows[0]['confidence4'] == pytest.approx(.2)
    assert rows[1]['confidence4'] == pytest.approx(.3)
    assert predictions.shape == (1, 4, 3) and len(audited) == 1 and len(closed) == len(traced) == 2
    for key in ('hist', 'hmask', 'frame'):
        np.testing.assert_array_equal(traced[0][key], getattr(states, key)[0])
    np.testing.assert_array_equal(traced[0]['observed_path'], states.observed_prefix(0))
