"""VC3D model-only normals and per-request confidence, without checkpoints or GPU."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from vesuvius.neural_tracing.tracers.fibers.tracing import service


def follower():
    result = service.FiberFollow.__new__(service.FiberFollow)
    result.base_um = 2.
    result.g = 4.
    result.vol = object()
    result.tracer = SimpleNamespace(frame_predictor=object(), p=SimpleNamespace(confidence=.5))
    result.policy = SimpleNamespace(confidence=.5)
    result.lock = threading.Lock()
    return result


def test_confidence_is_per_request_and_missing_uses_startup_default():
    model = follower()
    model.extrapolate = lambda request: model.tracer.p.confidence
    model.segment = lambda request: model.tracer.p.confidence
    assert model.handle(dict(op='extrapolate', confidence=.2)) == .2
    assert model.handle(dict(op='segment', confidence=.85)) == .85
    assert model.handle(dict(op='extrapolate')) == .5
    for value in [-.1, 1.1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='confidence'):
            model.handle(dict(op='extrapolate', confidence=value))


def test_normals_use_checkpoint_frames_in_trace_coordinates(monkeypatch):
    model = follower()
    points = np.array([[i*8., 0., 0.] for i in range(30)])
    calls = []

    def frames(predictor, vol, positions, priors, paths, families):
        assert predictor is model.tracer.frame_predictor
        assert vol is model.vol
        np.testing.assert_allclose(positions, points/4.)  # requested base_um=2
        assert all(f == 'H' for f in families)
        for p, path in zip(positions, paths):
            np.testing.assert_allclose(path[-1], p)
            assert np.linalg.norm(path[-1]-path[0]) <= service.HEADING_FIT
        calls.append(len(positions))
        result = np.repeat(np.eye(3)[None], len(positions), axis=0)
        result[:, :, 0] = [0., 2., 0.]
        result[1::2, :, 0] *= -1
        return result

    monkeypatch.setattr(service, 'predict_frames', frames)
    result = model.handle(dict(op='normals', points=points.tolist(), base_um=2.))
    np.testing.assert_allclose(result['normals'], np.tile([0., 1., 0.], (30, 1)))
    assert calls == [30]


def test_normals_support_checkpoint_ct_frame_policy(monkeypatch):
    model = follower()
    model.tracer.frame_predictor = None
    monkeypatch.setattr(service, 'ct_frame', lambda *args: np.eye(3))
    result = model.handle(dict(op='normals', points=[[0, 0, 0], [0, 0, 8]], base_um=2.))
    assert result['normals'] == [[1., 0., 0.], [1., 0., 0.]]


def test_bad_normals_fail_instead_of_creating_a_display_frame(monkeypatch):
    model = follower()
    monkeypatch.setattr(service, 'predict_frames', lambda *args: np.zeros((2, 3, 3)))
    with pytest.raises(ValueError, match='invalid sheet normals'):
        model.handle(dict(op='normals', points=[[0, 0, 0], [0, 0, 8]], base_um=2.))


@pytest.mark.parametrize('length', [.25, 5., 30.])
@pytest.mark.parametrize('sign', [-1., 1.])
def test_every_observed_history_resumes_in_travel_order(monkeypatch, length, sign):
    model = follower()
    model.tracer.n_history = 16
    model.tracer.frame_policy = 'learned_heading_normal_v1'
    path = np.array([[0., 0., 0.], [length*sign, 0., 0.]])
    opposite = np.array([-sign, 0., 0.])

    def frames(predictor, vol, positions, priors, paths, families):
        np.testing.assert_allclose(priors[0], [sign, 0., 0.])
        np.testing.assert_allclose(paths[0][-1], path[-1])
        return [np.column_stack([[0., 1., 0.], [0., 0., sign], [sign, 0., 0.]])]

    monkeypatch.setattr(service, 'predict_frames', frames)
    monkeypatch.setattr(service, 'ct_seed_heading', lambda *args: pytest.fail('history must not become a fresh seed'))
    pos, heading, state = model.start_state(path, opposite, 'H')
    np.testing.assert_allclose(heading, [sign, 0., 0.])
    assert state is not None
    np.testing.assert_allclose(state['observed_path'][0], path[0])
    np.testing.assert_allclose(state['observed_path'][-1], path[-1])
    assert state['seed_age'] == pytest.approx(length)


def test_target_behind_head_does_not_reverse_trusted_heading():
    model = follower()
    def run(path, direction, family, max_len, on_decision=None):
        np.testing.assert_allclose(direction, [1., 0., 0.])
        assert family == 'H'
        return np.array([path[-1], path[-1]+direction]), 'confidence', 1
    model.run = run
    result = model.segment(dict(history=[[0, 0, 0], [40, 0, 0]], target=[-40, 0, 0], base_um=2.))
    assert not result['accepted']
    assert result['points'][-1][0] > 40
