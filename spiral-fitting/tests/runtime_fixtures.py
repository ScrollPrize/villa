"""Minimal resident sessions and distributed worker stand-ins."""

from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from fit_session import SessionState, SpiralInputPaths
from spiral_runtime import DistributedInteractiveFitSession, InteractiveFitSession


class _FakeWorker:
    """A worker process stand-in for the parent watchdog and fail-stop paths."""

    def __init__(self, rank):
        self.rank = rank
        self.alive = True
        self.exitcode = None
        self.terminated = 0
        self.killed = 0

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated += 1
        self.alive = False
        if self.exitcode is None:
            self.exitcode = -15

    def kill(self):
        self.killed += 1
        self.alive = False
        self.exitcode = -9

    def join(self, timeout=None):
        return None


class RuntimeFixture(unittest.TestCase):
    def _proxy(self, world_size=2, published=None):
        """A coordinator with fake workers and no spawned processes."""
        session = DistributedInteractiveFitSession.__new__(
            DistributedInteractiveFitSession)
        session._init_coordinator_state(
            tuple(range(world_size)),
            (published.append if published is not None else None),
            None)
        session._events = queue.Queue()
        session._commands = [queue.Queue() for _ in range(world_size)]
        session._processes = [_FakeWorker(rank) for rank in range(world_size)]
        return session

    def _wait_for(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def _idle_session(self, completed=0, rank=0, world_size=1):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Idle
        session._phase = "Idle"
        session._completed = completed
        session._target = completed
        session._pending = 0
        session._commands = []
        session.session_generation = 0
        session._config_revision = 0
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session.rank = 0
        session.world_size = 1
        session._command_epoch = 0
        session._step_epoch = 0
        session._step_config_revision = 0
        session._stop_requested = False
        session._shutdown = False
        session._run_start_completed = completed
        session._context = SimpleNamespace(
            configure_dt_loss_schedule=lambda *_: None)
        session.requested_config = {
            "optimizer_num_training_steps": 30_000}
        session._run_config = dict(session.requested_config)
        session._warnings = []
        session.rank = rank
        session.world_size = world_size
        session._status_callback = None
        session._event_callback = None
        return session

    def _paused_session(self, calls, output_path, autosave_on_pause=True):
        session = InteractiveFitSession.__new__(InteractiveFitSession)
        session._condition = threading.Condition()
        session._state = SessionState.Running
        session._completed = 9
        session._pending = 1
        session._target = 10
        session._stop_requested = False
        session._latest_metrics = {}
        session._output_path = str(output_path)
        session._status_callback = None
        session._autosave_on_pause = autosave_on_pause
        session.paths = SpiralInputPaths.from_mapping({
            "dataset_root": "/datasets/scroll1",
            "output_directory": str(output_path),
        })

        def save_checkpoint(path, *_):
            calls.append("save")
            Path(path).write_bytes(_zip_checkpoint_bytes())

        session._context = SimpleNamespace(
            clear_interactive_run_state=lambda: calls.append("finish"),
            save_checkpoint=save_checkpoint)
        session._publish_preview = lambda: calls.append("preview")
        return session

    def _mid_run_session(self, calls, output_path, completed,
                         autosave_on_pause=True):
        session = self._paused_session(
            calls, output_path, autosave_on_pause=autosave_on_pause)
        session._completed = completed - 1
        session._pending = 500
        session._target = completed + 499
        session._warnings = []
        return session


def _zip_checkpoint_bytes(payload=b"payload"):
    import io
    import zipfile
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("data.pkl", payload)
    return buffer.getvalue()
