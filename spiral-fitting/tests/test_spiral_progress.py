import io
import threading

import pytest

from spiral_progress import ProgressReporter


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_determinate_snapshot_has_stage_local_elapsed_and_eta():
    clock = FakeClock()
    reporter = ProgressReporter(clock=clock, heartbeat_interval=0)
    reporter.begin(
        "loading", "Loading patches",
        step=0, total_steps=10, unit="patches")
    clock.advance(4)
    reporter.update(2)

    snapshot = reporter.snapshot()

    assert snapshot == {
        "operation": "loading",
        "stage_name": "Loading patches",
        "detail": None,
        "step": 2,
        "total_steps": 10,
        "unit": "patches",
        "elapsed_seconds": pytest.approx(4.0),
        "eta_seconds": pytest.approx(16.0),
    }


def test_indeterminate_snapshot_has_elapsed_without_eta():
    clock = FakeClock()
    reporter = ProgressReporter(clock=clock, heartbeat_interval=0)
    reporter.begin("loading", "Building geometry index")
    clock.advance(75)

    snapshot = reporter.snapshot()

    assert snapshot["elapsed_seconds"] == pytest.approx(75)
    assert snapshot["step"] is None
    assert snapshot["total_steps"] is None
    assert snapshot["eta_seconds"] is None

    reporter.begin(
        "loading", "Discovering GPU bricks",
        step=0, total_steps=0, unit="bricks")
    clock.advance(10)
    assert reporter.snapshot()["eta_seconds"] is None


def test_publish_is_rate_limited_but_snapshot_keeps_latest_counter():
    clock = FakeClock()
    published = []
    reporter = ProgressReporter(
        published.append, clock=clock, publish_interval=1.0,
        heartbeat_interval=0)
    reporter.begin("loading", "Loading tracks", step=0, total_steps=10)
    reporter.update(1)
    reporter.update(2)

    assert len(published) == 1
    assert reporter.snapshot()["step"] == 2

    clock.advance(1)
    reporter.update(3)
    assert len(published) == 2
    assert published[-1]["step"] == 3


def test_clear_publishes_null_and_removes_progress():
    published = []
    reporter = ProgressReporter(published.append, heartbeat_interval=0)
    reporter.begin("saving_checkpoint", "Saving checkpoint")
    reporter.clear()

    assert published[-1] is None
    assert reporter.snapshot() is None


def test_non_tty_console_uses_stable_progress_lines():
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter(
        stream=stream, clock=clock, heartbeat_interval=0)
    reporter.begin(
        "optimizing", "Optimizing",
        step=0, total_steps=4, unit="iterations")
    clock.advance(5)
    reporter.update(1)
    reporter.finish()

    output = stream.getvalue()
    assert "PROGRESS Optimizing" in output
    assert "1/4 iterations (25.0%)" in output
    assert "0.2 it/s" in output
    assert "ETA 15s" in output
    assert "4/4 iterations (100.0%)" in output


def test_iteration_rate_is_console_only_and_requires_completed_work():
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter(
        stream=stream, clock=clock, heartbeat_interval=0)
    reporter.begin(
        "optimizing", "Optimizing",
        step=0, total_steps=100, unit="iterations")
    assert "it/s" not in stream.getvalue()

    clock.advance(5)
    reporter.update(10)

    assert "2.0 it/s" in stream.getvalue()
    assert "rate_per_second" not in reporter.snapshot()


def test_non_iteration_progress_does_not_show_iteration_rate():
    clock = FakeClock()
    stream = io.StringIO()
    reporter = ProgressReporter(
        stream=stream, clock=clock, heartbeat_interval=0)
    reporter.begin(
        "loading", "Loading patches",
        step=0, total_steps=100, unit="patches")
    clock.advance(5)
    reporter.update(20)

    assert "it/s" not in stream.getvalue()


def test_updates_and_snapshots_are_thread_safe():
    reporter = ProgressReporter(heartbeat_interval=0)
    reporter.begin("loading", "Loading", step=0, total_steps=1000)

    thread = threading.Thread(
        target=lambda: [reporter.update(index) for index in range(1001)])
    thread.start()
    while thread.is_alive():
        snapshot = reporter.snapshot()
        assert 0 <= snapshot["step"] <= 1000
    thread.join()
    assert reporter.snapshot()["step"] == 1000
