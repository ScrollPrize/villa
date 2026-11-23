"""Initialization waits distinguish the current run from old arrays without real sleeps."""
from types import SimpleNamespace

import pytest

from vesuvius.utils.io import zarr_utils


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, sleeps=[])

    def sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(zarr_utils.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(zarr_utils.time, "sleep", sleep)
    return clock


@pytest.mark.parametrize("old_marker", [False, True])
def test_wait_uses_requested_timeout_and_ignores_old_run(tmp_path, clock, old_marker):
    if old_marker:
        zarr_utils.mark_zarr_ready(tmp_path, "old-run")
    with pytest.raises(RuntimeError, match="run 'new-run'.*Waited 12 seconds"):
        zarr_utils.wait_for_zarr_creation(tmp_path, run_id="new-run", max_wait_time=12, verbose=False)
    assert clock.sleeps == [5, 5, 2]


def test_wait_still_supports_existing_staged_blending(tmp_path, clock, monkeypatch):
    monkeypatch.setattr(zarr_utils, "zarr_array_exists", lambda _: clock.now >= 10)
    zarr_utils.wait_for_zarr_creation(tmp_path, max_wait_time=12, verbose=False)
    assert clock.sleeps == [5, 5]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_wait_rejects_invalid_timeout(tmp_path, timeout):
    with pytest.raises(ValueError, match="max_wait_time"):
        zarr_utils.wait_for_zarr_creation(tmp_path, max_wait_time=timeout)
