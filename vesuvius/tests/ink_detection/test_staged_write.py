from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import pytest

from vesuvius.ink_detection.preprocessing.staged_write import (
    create_staged_output,
    discard_staged_output,
    publish_staged_output,
)


def _sharing_violation(winerror: int) -> PermissionError:
    """A PermissionError shaped like the one Windows raises, on any platform."""

    error = PermissionError(13, "held open")
    error.winerror = winerror
    return error


def test_staged_file_round_trips_through_create_and_publish(tmp_path: Path) -> None:
    output = tmp_path / "result.tif"
    staged = create_staged_output(output)
    staged.write_bytes(b"payload")

    publish_staged_output(staged, output)

    assert output.read_bytes() == b"payload"
    assert not staged.exists()


def test_publish_moves_a_staged_directory(tmp_path: Path) -> None:
    output = tmp_path / "volume.zarr"
    staged = output.with_name(output.name + ".partial")
    staged.mkdir()
    (staged / "0.0.0").write_bytes(b"chunk")

    publish_staged_output(staged, output)

    assert (output / "0.0.0").read_bytes() == b"chunk"
    assert not staged.exists()


@pytest.mark.parametrize("winerror", [5, 32])
def test_publish_retries_a_sharing_violation_until_it_clears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, winerror: int
) -> None:
    output = tmp_path / "volume.zarr"
    staged = output.with_name(output.name + ".partial")
    staged.mkdir()
    real_replace = Path.replace
    calls: list[float] = []
    slept: list[float] = []

    def flaky(self: Path, target) -> Path:
        calls.append(time.monotonic())
        if len(calls) < 3:
            raise _sharing_violation(winerror)
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky)
    monkeypatch.setattr(
        "vesuvius.ink_detection.preprocessing.staged_write.time.sleep", slept.append
    )

    publish_staged_output(staged, output)

    assert output.is_dir() and not staged.exists()
    assert len(calls) == 3
    assert slept == [0.5, 1.0]  # backoff doubles


def test_publish_raises_other_permission_errors_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """POSIX has no winerror, so its behaviour is unchanged: no retry, no delay."""

    output = tmp_path / "volume.zarr"
    staged = output.with_name(output.name + ".partial")
    staged.mkdir()
    calls: list[int] = []
    slept: list[float] = []

    def denied(self: Path, target) -> Path:
        calls.append(1)
        raise PermissionError(13, "permission denied")

    monkeypatch.setattr(Path, "replace", denied)
    monkeypatch.setattr(
        "vesuvius.ink_detection.preprocessing.staged_write.time.sleep", slept.append
    )

    with pytest.raises(PermissionError):
        publish_staged_output(staged, output)

    assert calls == [1]
    assert slept == []


def test_publish_gives_up_saying_the_stage_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "volume.zarr"
    staged = output.with_name(output.name + ".partial")
    staged.mkdir()
    (staged / "0.0.0").write_bytes(b"chunk")
    slept: list[float] = []

    def held(self: Path, target) -> Path:
        raise _sharing_violation(32)

    monkeypatch.setattr(Path, "replace", held)
    monkeypatch.setattr(
        "vesuvius.ink_detection.preprocessing.staged_write.time.sleep", slept.append
    )

    with pytest.raises(PermissionError) as raised:
        publish_staged_output(staged, output, attempts=3)

    note = "\n".join(raised.value.__notes__)
    assert "nothing needs recomputing" in note
    assert str(staged) in note and str(output) in note
    assert len(slept) == 2
    # the finished work is still on disk, under the staged name
    assert (staged / "0.0.0").read_bytes() == b"chunk"
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses the rename")
def test_publish_survives_a_real_open_handle_on_windows(tmp_path: Path) -> None:
    output = tmp_path / "volume.zarr"
    staged = output.with_name(output.name + ".partial")
    staged.mkdir()
    chunk = staged / "0.0.0"
    chunk.write_bytes(b"chunk")
    released = threading.Event()

    def hold() -> None:
        with chunk.open("rb"):
            time.sleep(0.4)
        released.set()

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        publish_staged_output(staged, output, retry_delay=0.2)
    finally:
        holder.join()

    assert released.is_set()
    assert (output / "0.0.0").read_bytes() == b"chunk"


def test_discard_removes_an_unpublished_stage(tmp_path: Path) -> None:
    output = tmp_path / "result.tif"
    staged = create_staged_output(output)

    discard_staged_output(staged)
    discard_staged_output(staged)  # idempotent

    assert not staged.exists()
