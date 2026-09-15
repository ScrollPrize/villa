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
