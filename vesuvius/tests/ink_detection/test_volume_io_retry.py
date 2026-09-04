"""Transient remote read failures must not abort ink-detection runs.

The flat inference path (``vesuvius.ink_detection.inference.infer``, issue
#1666) and the shared bbox reader both stream remote zarr volumes chunk by
chunk.  A single truncated download used to propagate out and kill the whole
multi-minute run.  These tests drive the real entry points — the bbox reader
and ``FlatPatchReader._read_raw`` — against a store that fails the way an
object store does, rather than testing the retry helper in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.data._transient_reads import (
    _is_transient_read_error,
    _read_array_with_retry,
)
from vesuvius.ink_detection.inference.infer import FlatPatchReader
from vesuvius.ink_detection.volume_io import read_bbox_with_padding


class _FlakyVolume:
    """Array-like volume that raises a given sequence of errors before serving."""

    def __init__(self, errors: list[BaseException]) -> None:
        self._errors = list(errors)
        self.shape = (3, 4, 4)
        self.dtype = np.dtype("uint8")
        self.reads = 0

    def __getitem__(self, idx):
        self.reads += 1
        if self._errors:
            raise self._errors.pop(0)
        return np.full((3, 4, 4), 7, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(
        "vesuvius.data._transient_reads.time.sleep", lambda _s: None
    )


def _bbox():
    return (0, 0, 0, 3, 4, 4)


def test_bbox_read_retries_transient_error_then_succeeds() -> None:
    volume = _FlakyVolume([
        OSError("Response payload is not completed: ContentLengthError"),
        OSError("[SSL: RECORD_LAYER_FAILURE] record layer failure"),
    ])
    crop, valid = read_bbox_with_padding(volume, _bbox())
    assert volume.reads == 3
    np.testing.assert_array_equal(crop, 7)


def test_bbox_read_recovers_from_wrapped_cause() -> None:
    def make():
        outer = RuntimeError("Failed to get item")
        outer.__cause__ = OSError("Connection reset by peer")
        return outer

    volume = _FlakyVolume([make(), make()])
    crop, _ = read_bbox_with_padding(volume, _bbox())
    assert volume.reads == 3
    np.testing.assert_array_equal(crop, 7)


def test_bbox_read_exhausts_attempts_and_raises() -> None:
    volume = _FlakyVolume([OSError("Connection reset by peer")] * 5)
    with pytest.raises(OSError):
        read_bbox_with_padding(volume, _bbox())
    assert volume.reads == 4, "must stop at the retry cap"


def test_bbox_read_deterministic_error_fails_fast() -> None:
    volume = _FlakyVolume([KeyError("no such array: 0")])
    with pytest.raises(KeyError):
        read_bbox_with_padding(volume, _bbox())
    assert volume.reads == 1, "a coding error must fail fast, not be retried"


def test_flat_reader_raw_retries_transient_error_then_succeeds() -> None:
    # This is the path reported in issue #1666: FlatPatchReader reads the
    # surface volume through _read_raw during flat inference.
    volume = _FlakyVolume([OSError("Response payload is not completed")])
    reader = FlatPatchReader(
        input_path="ignored.zarr",
        resolution="0",
        depth_axis_first=True,
        height=4,
        width=4,
        layer_indices=np.arange(3),
        output_depth=3,
        preprocessing="divide_255",
    )
    reader._array = volume
    patch = reader.read(0, 0, 4, 4)
    assert volume.reads == 2
    np.testing.assert_array_equal(patch, 7)


def test_flat_reader_raw_deterministic_error_fails_fast() -> None:
    volume = _FlakyVolume([IndexError("index 99 is out of bounds")])
    reader = FlatPatchReader(
        input_path="ignored.zarr",
        resolution="0",
        depth_axis_first=True,
        height=4,
        width=4,
        layer_indices=np.arange(3),
        output_depth=3,
        preprocessing="divide_255",
    )
    reader._array = volume
    with pytest.raises(IndexError):
        reader.read(0, 0, 4, 4)
    assert volume.reads == 1


def test_retries_disabled_with_one_attempt() -> None:
    volume = _FlakyVolume([OSError("Connection reset by peer")] * 3)
    with pytest.raises(OSError):
        _read_array_with_retry(volume, (slice(None),), retries=1)
    assert volume.reads == 1


@pytest.mark.parametrize(
    "message",
    [
        "Response payload is not completed: ContentLengthError: 400, message='Not enough data to satisfy content length header (received 173631 of 507920 bytes).'",
        "[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2713)",
        "Connection reset by peer",
        "read operation timed out",
        "An error occurred (503) when calling GetObject: SlowDown",
    ],
)
def test_transient_messages_detected(message: str) -> None:
    assert _is_transient_read_error(OSError(message))


@pytest.mark.parametrize(
    "exc",
    [
        IndexError("index 99 is out of bounds"),
        KeyError("no such array: 0"),
        ValueError("patch_size must be a tuple of 3 integers"),
    ],
)
def test_deterministic_errors_not_flagged(exc: BaseException) -> None:
    assert not _is_transient_read_error(exc)