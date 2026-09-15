"""Shared fixtures for the focused spiral tests."""

import threading
import fit_spiral
from fit_session import SessionState
from spiral_progress import NullProgressReporter
from spiral_runtime import InteractiveFitSession


class _FakeContext:
    """The fitter-thread context the runtime load commands drive."""

    def __init__(self, accepted=True, apply_error=None):
        self.accepted = accepted
        self.apply_error = apply_error
        self.inspected = []
        self.applied = []

    def inspect_checkpoint(self, checkpoint, source=""):
        self.inspected.append(source)
        return fit_spiral.CheckpointVerdict(
            self.accepted, () if self.accepted else ("z-domain differs",),
            completed_iterations=99, source=source)

    def apply_checkpoint(self, checkpoint, realign_lr=False):
        self.applied.append(realign_lr)
        if self.apply_error is not None:
            raise self.apply_error
        return 99


def _idle_session(completed=5):
    session = InteractiveFitSession.__new__(InteractiveFitSession)
    session._condition = threading.Condition()
    session._state = SessionState.Idle
    session._phase = "Idle"
    session._completed = completed
    session._target = completed
    session._pending = 0
    session._commands = []
    session._pending_checkpoint = None
    session.session_generation = 0
    session._config_revision = 0
    session._command_epoch = 0
    session._step_epoch = 0
    session._step_config_revision = 0
    session._stop_requested = False
    session._shutdown = False
    session._run_start_completed = completed
    session._latest_metrics = {"total_loss": 1.0}
    session.input_manifest = {}
    session.rank = 0
    session.world_size = 1
    session._status_callback = None
    session._event_callback = None
    session._progress_reporter = lambda: NullProgressReporter()
    session._publish_status = lambda: None
    return session
