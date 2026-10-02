"""Crop frames after confidence failures and subsequent successful recovery."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vesuvius.neural_tracing.fiber_follow.shared.geometry import (
    CropSpec, frame_from_heading, normalize,
)
from vesuvius.neural_tracing.fiber_follow.shared.trace import ModelTracer, TraceParams
from vesuvius.neural_tracing.fiber_follow.shared.heading import transverse_frame


@pytest.fixture(autouse=True)
def ct_sheet(monkeypatch):
    # Isolate commit policy from image I/O using an exact, constant sheet normal.
    monkeypatch.setattr('vesuvius.neural_tracing.fiber_follow.shared.heading.ct_tensor',
                        lambda vol,pos: np.outer(np.array([0., -1., 0.]), np.array([0., -1., 0.])))


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


def test_rejected_decision_stops_immediately_and_is_observed_without_a_forced_commit():
    predictions = [
        ([[0., 0., 1.], [1., 0., 2.]], [.9, .9]),
        ([[3., 0., 1.], [3., 0., 2.]], [.1, .1]),  # Rejected after refinement: the trace ends here.
        ([[0., 0., 1.], [0., 0., 2.]], [.9, .9]),
    ]
    tracer = RecordingTracer(predictions)
    states = []
    try:
        seed = np.array([[100., 100., 100.]])
        paths, reasons = tracer.trace(seed, np.array([[1., 1., 1.]]), on_decision=lambda i, s: states.append(s))
    finally:
        tracer.close()
    assert reasons == ['confidence'] and len(states) == 2 and len(tracer.inputs) == 2
    assert [s['n_commit'] for s in states] == [2, 0] and states[1]['would_stop']
    # The rejected proposal is retained in the observed decision but never committed.
    np.testing.assert_array_equal(states[1]['points'], np.asarray(predictions[1][0], np.float32))
    np.testing.assert_array_equal(paths[0][-1], states[1]['pos'])
    assert not hasattr(TraceParams(), 'explore_calls') and not hasattr(TraceParams(), 'stop_patience')


@pytest.mark.parametrize('first,confidence,reason', [
    ([3., 0., 1.], .1, 'confidence'),
    ([7., 0., 1.], .9, 'recovery_limit'),
])
def test_rejections_respect_stopping_and_recovery_limits(first, confidence, reason):
    tracer = RecordingTracer([([first, [0., 0., 2.]], [confidence, confidence])])
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


def test_resume_preserves_the_trusted_boundary_and_next_heading():
    step = ([[1., 0., 1.], [1., 0., 2.]], [.9, .1])
    predictions = [step]*13
    rows = record(predictions)
    resumed = record(predictions[7:], initial=rows[7])
    for expected, actual in zip(rows[7:], resumed):
        np.testing.assert_array_equal(actual['frame'], expected['frame'])
        np.testing.assert_array_equal(actual['pos'], expected['pos'])
        assert actual['heading_start'] == expected['heading_start']
