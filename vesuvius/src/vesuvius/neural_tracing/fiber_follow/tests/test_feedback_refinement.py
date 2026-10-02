"""Adaptive absolute proposals, detached feedback and threshold-aware selection."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_regression import proposal_output
from slab_fixtures import cfg, slab_batch as memory_batch
from vesuvius.neural_tracing.fiber_follow.regression.model import build_model, select_refinement
from vesuvius.neural_tracing.fiber_follow.regression.supervision import loss_terms
from vesuvius.neural_tracing.fiber_follow.regression.diagnostics import decision_rows
from vesuvius.neural_tracing.fiber_follow.regression.data import DirectTracer
from vesuvius.neural_tracing.fiber_follow.shared.policy import commit_prefix
from vesuvius.neural_tracing.fiber_follow.shared.trace import TraceParams


def policy_output():
    curves = torch.zeros(1, 3, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    curves[0, :, :, 0] = torch.tensor([0., 2., 4.])[:, None]
    confidence = torch.tensor([[[.99, .8, .7, .1], [.95, .9, .4, .3], [.9, .85, .75, .2]]])
    previous = torch.cat((torch.ones_like(confidence[..., :1]), confidence[..., :-1]), -1)
    return proposal_output(curves, torch.logit(1-confidence/previous))


def test_selection_uses_actual_threshold_horizon_and_matching_scores():
    output, c = policy_output(), cfg()
    for threshold, window, expected in ((.5, 4, 2), (.5, 1, 0), (.79, 4, 1), (1., 4, 0)):
        selected = select_refinement(output, c, threshold, window)
        assert selected['selected_refinement'].item() == expected
        for name in ('points', 'hazard_logits', 'confidence_logits', 'confidence'):
            torch.testing.assert_close(selected[name], output['refinement_'+name][:, expected])
    stopped = select_refinement(output, c, 1., 4)
    assert commit_prefix(stopped['points'], stopped['confidence'], 1., 4)[0].item() == 0
    # An invalid connection cannot win even with the longest confident prefix.
    output['refinement_points'][:, 2, :, 0] = 20.
    assert select_refinement(output, c)['selected_refinement'].item() == 0
    # Equal prefix length and confidence retain the earlier proposal.
    output = policy_output()
    output['refinement_confidence'][:, 2] = output['refinement_confidence'][:, 0]
    assert select_refinement(output, c)['selected_refinement'].item() == 0


@pytest.mark.parametrize('training', [False, True])
def test_full_acceptance_skips_all_additional_decoder_and_scorer_work(monkeypatch, training):
    model = build_model(cfg(recurrent_refinement_steps=3)).train(training)
    b = memory_batch(model.cfg, 2)
    calls = []
    def score(ctx, points):
        calls.append(len(points))
        return points.new_full(points.shape[:2], -12.)
    monkeypatch.setattr(model, 'hazard_logits', score)
    decoded = []
    hook = model.coordinates.register_forward_hook(lambda *args: decoded.append(1))
    out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=.9)
    hook.remove()
    assert calls == [2] and len(decoded) == 1
    assert out['refinement_mask'].tolist() == [[True], [True]]
    torch.testing.assert_close(out['points'], out['initial_points'])


def test_mixed_batch_retries_only_unaccepted_rows_and_masks_unused_attempts(monkeypatch):
    model = build_model(cfg(recurrent_refinement_steps=4))
    b = memory_batch(model.cfg, 3)
    calls, decoder_batches = [], []
    def score(ctx, points):
        calls.append(len(points))
        values = points.new_full(points.shape[:2], 2.)
        values[0] = -12.
        return values
    monkeypatch.setattr(model, 'hazard_logits', score)
    hook = model.coordinates.register_forward_hook(lambda m, args, out: decoder_batches.append(len(out)))
    out = model(b['x'], b['hist'], b['hmask'])
    hook.remove()
    assert calls == decoder_batches == [3, 2, 1]
    assert out['refinement_mask'].tolist() == [[True, False, False], [True, True, False], [True, True, True]]
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


def test_compiled_compacted_batch_matches_eager_outputs_and_gradients(monkeypatch):
    torch.manual_seed(94)
    eager = build_model(cfg(recurrent_refinement_steps=3))
    compiled_model = copy.deepcopy(eager)
    b = memory_batch(eager.cfg, 3)
    def score(ctx, points):
        values = points.new_full(points.shape[:2], 2.)
        values[0] = -12.
        return values
    for model in (eager, compiled_model):
        monkeypatch.setattr(model, 'hazard_logits', score)
    outputs = []
    for model in (eager, torch.compile(compiled_model, backend='eager')):
        out = model(b['x'], b['hist'], b['hmask'])
        loss_terms(out, b, eager.cfg)['geometry_per_state'].sum().backward()
        outputs.append(out)
    for key in ('points', 'refinement_points', 'refinement_mask', 'confidence'):
        torch.testing.assert_close(outputs[0][key], outputs[1][key])
    for (name, p), (_, q) in zip(eager.named_parameters(), compiled_model.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6, msg=name)


def test_partial_acceptance_uses_budget_and_absolute_head_can_replace_path(monkeypatch):
    model = build_model(cfg(recurrent_refinement_steps=2))
    b = memory_batch(model.cfg, 1)
    calls = []
    def coordinates(decoded):
        calls.append(1)
        return decoded.new_full((*decoded.shape[:2], 2), (-1 if len(calls) == 1 else 1)*1.1)
    monkeypatch.setattr(model.coordinates, 'forward', coordinates)
    # First point accepted, later prefix rejected. A one-point commit does not
    # make the entire prediction acceptable and must not suppress retries.
    monkeypatch.setattr(model, 'hazard_logits', lambda ctx, p: p.new_tensor([[-12., 2., 2., 2.]]))
    out = model(b['x'], b['hist'], b['hmask'], n_commit=1)
    assert len(calls) == 3 and out['refinement_mask'].all()
    curves = out['refinement_points']
    assert (curves[:, 1, 1:, :2]-curves[:, 0, 1:, :2]).norm(dim=-1).min() > 4.
    torch.testing.assert_close(curves[:, 1], curves[:, 2])  # Absolute, not accumulated offsets.
    assert curves[..., :2].abs().max() <= model.cfg.lateral_limit
    assert curves[:, :, 0].norm(dim=-1).max() <= model.cfg.max_recovery_distance+1e-5


def test_feedback_changes_new_proposal_but_geometry_cannot_train_scores(monkeypatch):
    torch.manual_seed(93)
    model = build_model(cfg(recurrent_refinement_steps=1))
    b = memory_batch(model.cfg, 1)
    outputs, scores = [], []
    for value in (-1., 3.):
        def score(ctx, points, value=value):
            result = points.new_full(points.shape[:2], value, requires_grad=True)
            scores.append(result)
            return result
        monkeypatch.setattr(model, 'hazard_logits', score)
        out = model(b['x'], b['hist'], b['hmask'], confidence_threshold=1.)
        outputs.append(out)
        loss_terms(out, b, model.cfg)['geometry_per_state'].sum().backward()
    torch.testing.assert_close(outputs[0]['initial_points'], outputs[1]['initial_points'])
    assert not torch.equal(outputs[0]['refinement_points'][:, 1], outputs[1]['refinement_points'][:, 1])
    assert all(score.grad is None for score in scores)


def test_all_attempts_get_their_own_first_failure_supervision():
    c = cfg(recurrent_refinement_steps=2)
    b = memory_batch(c, 1)
    curves = torch.zeros(1, 3, 4, 3)
    curves[..., 2] = torch.arange(1, 5)
    curves[:, 1, :, 0] = 3.  # Immediate failure.
    curves[:, 2, 2:, 0] = 3.  # Failure in the third segment.
    hazards = torch.zeros(1, 3, 4, requires_grad=True)
    out = proposal_output(curves, hazards, selected=0)
    loss_terms(out, b, c)['confidence_per_state'].sum().backward()
    assert (hazards.grad[0, 0] > 0).all()
    assert hazards.grad[0, 1, 0] < 0 and hazards.grad[0, 1, 1:].eq(0).all()
    assert (hazards.grad[0, 2, :2] > 0).all()
    assert hazards.grad[0, 2, 2] < 0 and hazards.grad[0, 2, 3] == 0


def test_tracer_passes_operating_threshold_into_adaptive_model(monkeypatch):
    from test_history_slabs import fake_ct
    fake_ct(monkeypatch)
    model = build_model(cfg(recurrent_refinement_steps=2)).eval()
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.regression.data.image_crop',
        lambda items, vol, crop, pool=None, **kw: torch.ones(len(items), 2, crop.depth, crop.width, crop.width)*.25)
    thresholds = []
    original = model.forward
    def forward(*args, **kwargs):
        thresholds.append((kwargs['confidence_threshold'], kwargs['n_commit']))
        return original(*args, **kwargs)
    monkeypatch.setattr(model, 'forward', forward)
    tracer = DirectTracer(model, SimpleNamespace(shape=(1000, 1000, 1000)), model.cfg.fine,
                          model.cfg.n_history, TraceParams(n_commit=1, max_len=1., confidence=0.), device='cpu')
    try:
        tracer.trace(np.array([[100., 100., 400.]]), np.array([[0., 0., 1.]]))
    finally:
        tracer.close()
    assert thresholds and all(value == (0., 1) for value in thresholds)


def test_threshold_diagnostics_label_the_chosen_proposal():
    output, c = policy_output(), cfg()
    b = memory_batch(c, 1)
    rows = decision_rows(output, b, c, n_commit=1, thresholds=(.5, 1.))
    assert rows[0]['gate_0.5']['accepted_wrong'] == 0
    assert rows[0]['gate_1.0']['false_stops'] == 1


def test_recovery_sweeps_rerun_the_adaptive_policy_at_each_threshold(monkeypatch):
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_tensor',
                        lambda vol,pos: np.outer(np.array([1., 0., 0.]), np.array([1., 0., 0.])))
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.oriented_seed_heading',
                        lambda vol, pos, family, direction: np.asarray(direction))
    from vesuvius.neural_tracing.fiber_follow.shared.data import TracedFiber, SampleConfig
    from vesuvius.neural_tracing.fiber_follow.shared.recovery import make_recovery_states, evaluate_recovery_states
    c = cfg()
    model = build_model(c)
    sample = SampleConfig(crop=c.fine, n_history=c.n_history, n_future=c.n_future)
    arc = np.arange(300, dtype=float)
    fiber = TracedFiber('line', np.c_[arc*0, arc*0, arc], arc, '')
    states = make_recovery_states([fiber], [dict(fiber=0, t=150., sign=1)], sample, dict(volume={}), None)
    calls = []
    def forward(*args, confidence_threshold, n_commit):
        calls.append((confidence_threshold, n_commit))
        return select_refinement(policy_output(), c, confidence_threshold, n_commit)
    monkeypatch.setattr(model, 'forward', forward)
    class Tracer:
        def __init__(self, *args, **kwargs):
            pass
        def trace(self, pos, heading, initial_states):
            return [pos], ['confidence']
        def close(self):
            pass
    rows, _ = evaluate_recovery_states(model, None, states, [fiber], sample, limit=1,
        thresholds=(.5, .79), n_commit=4, tracer_class=Tracer,
        batch_builder=lambda items, vol: memory_batch(c, 1))
    assert calls == [(.5, 4), (.79, 4)]
    assert rows[0]['confidence4'] == pytest.approx(.2)
    assert rows[1]['confidence4'] == pytest.approx(.3)
