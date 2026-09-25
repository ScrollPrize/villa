"""CUDA regressions for recovery limits and metrics matching rollout choices."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.model import FollowNetConfig
from vesuvius.neural_tracing.fiber_follow.policy import choose_candidate, recovery_allowed
from vesuvius.neural_tracing.fiber_follow.supervision import candidate_labels, loss_fn


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA required; no CPU fallback')
    return torch.device('cuda')


def paths(device, offsets):
    points = torch.zeros(1, len(offsets), 4, 3, device=device)
    points[..., 2] = torch.arange(1, 5, device=device)
    points[..., 0] = torch.tensor(offsets, device=device)[None, :, None]
    return points


def batch(device, offset=0.):
    hist = torch.zeros(1, 4, 3, device=device)
    hist[..., 2] = -torch.arange(1, 5, device=device)
    ab = torch.zeros(1, 13, 2, device=device)
    ab[..., 0] = offset
    return dict(dense_ab=ab, dense_mask=torch.ones(1, 13, device=device),
                plane_ab=ab[:, ::4], plane_mask=torch.ones(1, 4, device=device),
                offtrack=torch.zeros(1, device=device), endpoint_known=torch.zeros(1, device=device),
                end_local=torch.zeros(1, 3, device=device), hist=hist, hmask=torch.ones(1, 4, device=device),
                gt_history=torch.cat([hist.new_zeros(1, 1, 3), hist], 1),
                gt_history_mask=torch.ones(1, 5, device=device))


@pytest.mark.parametrize('offset,positive', [(3., True), (7., False)])
def test_first_connection_labels_allow_bounded_recovery(device, offset, positive):
    points = paths(device, [offset])
    labels, mask, _, error = candidate_labels(points, batch(device, offset))
    assert mask.bool().all()
    assert labels.bool().eq(positive).all()
    assert error.item() == 0  # Endpoint agreement cannot excuse a long bridge.


def test_unknown_annotation_does_not_hide_recovery_failure(device):
    target = batch(device)
    target['dense_mask'].zero_()
    target['dense_ab'].fill_(float('nan'))
    labels, mask, quality, error = candidate_labels(paths(device, [3., 7.]), target)
    assert not mask[0, 0].any()
    assert mask[0, 1].all() and not labels[0, 1].any()
    assert torch.isfinite(quality).all() and torch.isfinite(error).all()


def test_limit_checks_full_segment_length_and_finite_forward_motion(device):
    points = paths(device, [0.] * 6)
    points[0, :, 0] = torch.tensor([[0, 0, 6], [0, 0, 6.01], [6, 0, 1],
                                     [float('nan'), 0, 1], [0, 0, 0], [0, 0, -1]], device=device)
    assert recovery_allowed(points).tolist() == [[True, False, False, False, False, False]]


def test_choice_applies_confidence_recovery_and_prefix_gates(device):
    points = paths(device, [7., 0., 2.]).expand(3, -1, -1, -1).clone()
    ranks = torch.tensor([[100., 10., 1.]], device=device).expand(3, -1)
    conf = torch.full((3, 3, 4), .9, device=device)
    conf[0, 1] = .1
    conf[0, 2, 2] = .1  # Later confidence cannot reopen a closed prefix.
    conf[1] = .1
    points[2, :, :, 0] = 7.
    chosen, commit, allowed = choose_candidate(points, ranks, conf)
    assert chosen.tolist() == [2, 1, 0]
    assert commit.tolist() == [2, 0, 0]
    assert not allowed[2].any()


def metric_fixture(device):
    cfg = FollowNetConfig(flow_samples=2, n_future=4, future_step=1.)
    candidates = paths(device, [2., 3.])
    confidence = torch.tensor([[[.1]*4, [.9]*4]], device=device)
    out = dict(candidates=candidates, samples=candidates,
               ranks=torch.tensor([[10., 1.]], device=device), confidence=confidence,
               confidence_logits=torch.logit(confidence), candidate_support=torch.ones(1, 2, device=device),
               flow_loss=torch.zeros((), device=device), flow_known_fraction=torch.ones((), device=device),
               flow_censored_fraction=torch.zeros((), device=device))
    return cfg, out, batch(device)


def test_metrics_use_tracer_choice_over_all_samples(device):
    cfg, out, target = metric_fixture(device)
    _, metrics = loss_fn(out, target, cfg)
    assert metrics['selected_error'] == pytest.approx(3.)  # Raw rank would choose 2.
    assert metrics['accepted_selected_error'] == pytest.approx(3.)
    assert metrics['selected_accept_fraction'] == 1
    assert metrics['oracle_error'] == pytest.approx(2.)
    assert metrics['oracle_recall'] == 0
    _, metrics = loss_fn(out, target, cfg, confidence_threshold=.95)
    assert metrics['selected_error'] == pytest.approx(2.)
    assert metrics['selected_accept_fraction'] == 0 and metrics['selected_commit'] == 0


def test_oracle_recall_censors_unknowns(device):
    cfg, out, target = metric_fixture(device)
    out['candidates'][..., :2] = 0
    out['samples'][..., :2] = 0
    target['dense_mask'][:, 5:] = 0
    _, metrics = loss_fn(out, target, cfg)
    assert metrics['oracle_recall_known_fraction'] == 0


def test_gate_sweep_reselects_for_each_threshold(device):
    cfg, out, target = metric_fixture(device)
    out['candidates'][:, 0, :, 0] = 0.
    out['confidence'][:, 0] = .4
    out['confidence_logits'] = torch.logit(out['confidence'])
    _, metrics = loss_fn(out, target, cfg)
    assert metrics['gate0.3_first_false_go'] == 0
    assert metrics['gate0.7_first_false_go'] == 1


def test_teacher_candidates_do_not_improve_proposal_metrics(device):
    cfg, out, target = metric_fixture(device)
    out['candidates'] = torch.cat([out['candidates'], paths(device, [0.] * 5)], 1)
    out['ranks'] = torch.cat([out['ranks'], torch.full((1, 5), 100., device=device)], 1)
    out['confidence'] = torch.cat([out['confidence'], torch.full((1, 5, 4), .99, device=device)], 1)
    out['confidence_logits'] = torch.logit(out['confidence'])
    _, metrics = loss_fn(out, target, cfg)
    assert metrics['selected_error'] == pytest.approx(3.)
    assert metrics['oracle_error'] == pytest.approx(2.) and metrics['oracle_recall'] == 0


@pytest.mark.parametrize('blocked', [False, True])
def test_tracer_enforces_recovery_even_during_exploration(device, monkeypatch, blocked):
    from vesuvius.neural_tracing.fiber_follow import trace
    from vesuvius.neural_tracing.fiber_follow.geometry import CropSpec

    class Proposals(torch.nn.Module):
        cfg = FollowNetConfig(n_future=4, future_step=1., flow_samples=2)

        def forward(self, x, hist, mask, generator=None):
            assert x.is_cuda and hist.is_cuda
            return dict(candidates=paths(device, [7., 8. if blocked else 2.]),
                        ranks=torch.tensor([[10., 1.]], device=device),
                        confidence=torch.ones(1, 2, 4, device=device))

    monkeypatch.setattr(trace, 'read_blocks', lambda *a: (np.zeros((1, 1)), np.zeros((1, 3))))
    monkeypatch.setattr(trace, 'build_inputs', lambda *a, **k: torch.zeros(1, 1, 1, 1, 1, device=device))
    vol = SimpleNamespace(spec=SimpleNamespace(mode='fiber'), shape=(100, 100, 100))
    tracer = trace.ModelTracer(Proposals(), vol, CropSpec(depth=8, width=9, behind=2), n_history=4,
                               params=trace.TraceParams(confidence=0., stop_patience=5, explore_calls=8), device=device)
    decisions = []
    try:
        result, reasons = tracer.trace(np.array([[50., 50., 50.]]), np.array([[0., 0., 1.]]),
                                       abort=lambda *a: True,
                                       on_decision=lambda i, state: decisions.append(state))
    finally:
        tracer.close()
    assert decisions[0]['recovery_blocked'] == blocked
    if blocked:
        assert reasons == ['recovery_limit'] and len(result[0]) == 1
        assert decisions[0]['n_commit'] == 0
    else:
        assert decisions[0]['chosen'] == 1 and len(result[0]) > 1
        assert reasons == ['abort']


@pytest.mark.parametrize('limit', [0., float('inf'), float('nan'), .5])
def test_recovery_config_rejects_unusable_limits(limit):
    with pytest.raises(ValueError):
        FollowNetConfig(future_step=1., max_recovery_distance=limit)
