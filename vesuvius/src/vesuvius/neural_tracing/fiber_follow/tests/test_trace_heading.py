"""Crop frames after confidence failures and subsequent successful recovery."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec, frame_from_heading, normalize,
)
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams


class ScriptedModel(torch.nn.Module):
    def __init__(self, predictions):
        super().__init__()
        self.cfg = SimpleNamespace(n_future=2, max_recovery_distance=6.)
        self.predictions = iter(predictions)

    def forward(self, image, history, mask):
        points, confidence = next(self.predictions)
        return dict(points=torch.tensor([points], dtype=torch.float32),
                    confidence=torch.tensor([confidence], dtype=torch.float32))


class RecordingTracer(ModelTracer):
    def __init__(self, predictions, **policy):
        super().__init__(ScriptedModel(predictions), SimpleNamespace(shape=(1000,)*3),
                         CropSpec(depth=4, width=4, behind=1), n_history=4,
                         params=TraceParams(n_commit=2, **policy), device='cpu')
        self.inputs = []

    def build_inputs(self, pos, frames, hist, hmask):
        self.inputs.append((pos.copy(), frames.copy()))
        return torch.zeros(len(pos), 1)


@pytest.mark.parametrize('policy', [dict(explore_calls=5), dict(stop_patience=3)])
def test_failed_steps_preserve_crop_frame_until_a_confident_commit(policy):
    predictions = [
        ([[0., 0., 1.], [1., 0., 2.]], [.9, .9]),  # Establish a new trusted heading.
        ([[3., 0., 1.], [3., 0., 2.]], [.1, .1]),  # Forced lateral recovery step.
        ([[-2., 1., 1.], [-2., 1., 2.]], [.1, .1]),  # Another failure, same frame.
        ([[0., 2., 1.], [0., 5., 2.]], [.9, .1]),  # One confident point is enough.
        ([[0., 0., 1.], [0., 0., 2.]], [.1, .1]),
    ]
    tracer = RecordingTracer(predictions, **policy)
    states = []

    def capture(index, state):
        states.append(state)
        return len(states) < len(predictions)

    try:
        tracer.trace(np.array([[100., 100., 100.]]), np.array([[1., 1., 1.]]),
                     on_decision=capture)
    finally:
        tracer.close()

    assert [s['n_commit'] for s in states] == [2, 0, 0, 1, 0]
    original = states[0]['frame']
    trusted = frame_from_heading(normalize(original @ np.array([1., 0., 1.])), original[:, 0])
    np.testing.assert_allclose(states[1]['frame'], trusted, atol=1e-12)
    assert not np.allclose(trusted, original)  # Preserve the pre-failure frame, not the seed frame.
    for j in (1, 2):
        np.testing.assert_array_equal(states[j+1]['frame'], states[j]['frame'])
        np.testing.assert_allclose(states[j+1]['pos'],
                                   states[j]['pos'] + trusted @ predictions[j][0][0])
    recovered = frame_from_heading(normalize(trusted @ np.array([0., 2., 1.])), trusted[:, 0])
    np.testing.assert_allclose(states[4]['frame'], recovered, atol=1e-12)
    assert not np.allclose(recovered, trusted)
    # Verify the frame supplied to the next model observation, not just metadata.
    for (_, frames), state in zip(tracer.inputs, states):
        np.testing.assert_array_equal(frames[0], state['frame'])


@pytest.mark.parametrize('policy,first,reason', [
    ({}, [3., 0., 1.], 'confidence'),
    (dict(explore_calls=5), [7., 0., 1.], 'recovery_limit'),
])
def test_failed_steps_still_respect_stopping_and_recovery_limits(policy, first, reason):
    tracer = RecordingTracer([( [first, [0., 0., 2.]], [.1, .1])], **policy)
    seed = np.array([[100., 100., 100.]])
    try:
        paths, reasons = tracer.trace(seed, np.array([[0., 0., 1.]]))
    finally:
        tracer.close()
    assert reasons == [reason]
    np.testing.assert_array_equal(paths[0], seed)
    assert len(tracer.inputs) == 1
