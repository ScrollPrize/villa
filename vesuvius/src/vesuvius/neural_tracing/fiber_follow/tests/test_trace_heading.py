"""Crop frames after confidence failures and subsequent successful recovery."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec, frame_from_heading, normalize,
)
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.heading import normal_frame


@pytest.fixture(autouse=True)
def ct_sheet(monkeypatch):
    # Isolate commit policy from image I/O using an exact, constant sheet normal.
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_normal',
                        lambda vol,pos: np.array([0., -1., 0.]))


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
def test_failed_steps_preserve_crop_frame_and_short_recovery_cannot_steer(policy):
    predictions = [
        ([[0., 0., 1.], [1., 0., 2.]], [.9, .9]),
        ([[3., 0., 1.], [3., 0., 2.]], [.1, .1]),  # Forced lateral recovery step.
        ([[-2., 1., 1.], [-2., 1., 2.]], [.1, .1]),  # Another failure, same frame.
        ([[0., 2., 1.], [0., 5., 2.]], [.9, .1]),  # Too short to replace the retained heading.
        ([[0., 0., 1.], [0., 0., 2.]], [.1, .1]),
    ]
    tracer = RecordingTracer(predictions, **policy)
    states = []

    def capture(index, state):
        states.append(state)
        return len(states) < len(predictions)

    try:
        seed = np.array([100., 100., 100.])
        original_frame = normal_frame(np.array([1., 1., 1.]),np.array([0., -1., 0.]))
        direction = original_frame @ normalize(np.array([1., 0., 1.]))
        history = seed-np.arange(14., 0., -1)[:, None]*direction
        tracer.trace(seed[None], np.array([[1., 1., 1.]]), histories=[history], on_decision=capture)
    finally:
        tracer.close()

    assert [s['n_commit'] for s in states] == [2, 0, 0, 1, 0]
    original = states[0]['frame']
    trusted = states[1]['frame']
    assert not np.allclose(trusted, original)  # Preserve the pre-failure frame, not the seed frame.
    for j in (1, 2):
        np.testing.assert_array_equal(states[j+1]['frame'], states[j]['frame'])
        np.testing.assert_allclose(states[j+1]['pos'],
                                   states[j]['pos'] + trusted @ predictions[j][0][0])
    np.testing.assert_array_equal(states[4]['frame'], trusted)
    for j in (2, 3):
        assert states[j]['heading_start'] == len(states[j]['observed_path'])
    assert states[4]['heading_start'] == len(states[4]['observed_path'])-1
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


def record(predictions, *, histories=None, initial=None, **policy):
    tracer = RecordingTracer(predictions, **policy)
    rows = []
    try:
        tracer.trace(np.array([[100., 100., 100.]]) if initial is None else initial['pos'][None],
                     np.array([[0., 0., 1.]]) if initial is None else initial['frame'][None, :, 2],
                     histories=histories, initial_states=None if initial is None else [initial],
                     on_decision=lambda i,s: rows.append(s) or len(rows)<len(predictions))
    finally:
        tracer.close()
    return rows


def test_one_point_commits_accumulate_a_twelve_voxel_heading_baseline():
    step = ([[1., 0., 1.], [1., 0., 2.]], [.9, .1])
    rows = record([step]*10)
    for row in rows[:9]:
        np.testing.assert_array_equal(row['frame'], rows[0]['frame'])
    expected = rows[0]['frame'] @ normalize(np.array([1., 0., 1.]))
    np.testing.assert_allclose(rows[9]['frame'][:, 2], expected, atol=1e-12)


def test_single_lateral_correction_cannot_replace_an_established_axis():
    seed = np.array([100., 100., 100.])
    history = seed-np.arange(14., 0., -1)[:, None]*np.array([0., 0., 1.])
    step = ([[1.438, 0., 1.], [0., 0., 2.]], [.9, .1])
    rows = record([step]*2, histories=[history])
    turn = np.degrees(np.arccos(np.clip(rows[0]['frame'][:,2] @ rows[1]['frame'][:,2], -1, 1)))
    assert turn < 6.  # Last-segment replacement would turn by 55 degrees.


def test_forced_points_and_recovery_connector_never_enter_trusted_fit():
    fail = ([[3., 0., 1.], [3., 0., 2.]], [.1, .1])
    recover = ([[-3., 0., 1.], [-3., 0., 2.]], [.9, .1])
    step = ([[1., 0., 1.], [1., 0., 2.]], [.9, .1])
    rows = record([fail, fail, recover]+[step]*10, explore_calls=20)
    for row in rows[:12]:
        np.testing.assert_array_equal(row['frame'], rows[0]['frame'])
    expected = rows[0]['frame'] @ normalize(np.array([1., 0., 1.]))
    np.testing.assert_allclose(rows[12]['frame'][:,2], expected, atol=1e-12)
    start = rows[3]['heading_start']
    np.testing.assert_array_equal(rows[12]['observed_path'][start], rows[3]['pos'])


def test_resume_preserves_the_trusted_boundary_and_next_heading():
    fail = ([[3., 0., 1.], [3., 0., 2.]], [.1, .1])
    step = ([[1., 0., 1.], [1., 0., 2.]], [.9, .1])
    predictions = [fail]+[step]*12
    rows = record(predictions, stop_patience=2)
    resumed = record(predictions[7:], initial=rows[7], stop_patience=2)
    for expected, actual in zip(rows[7:], resumed):
        np.testing.assert_array_equal(actual['frame'], expected['frame'])
        np.testing.assert_array_equal(actual['pos'], expected['pos'])
        assert actual['heading_start'] == expected['heading_start']
