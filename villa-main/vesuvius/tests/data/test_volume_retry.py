"""Transient remote read failures must not abort a long streaming run.

Reads go through Volume.__getitem__, so the retry lives there and every caller
of Volume gets it. These drive the real __getitem__ against a store that fails
the way an object store does, rather than testing the helper in isolation.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.data.volume import Volume, _is_transient_read_error


class _FlakyStore:
    """Array-like store that raises a given sequence of errors before serving."""

    def __init__(self, errors: list[BaseException]) -> None:
        self._errors = list(errors)
        self.ndim = 3
        self.shape = (16, 16, 16)
        self.dtype = np.dtype("uint8")
        self.reads = 0

    def __getitem__(self, idx):
        self.reads += 1
        if self._errors:
            raise self._errors.pop(0)
        return np.ones((4, 4, 4), dtype=np.uint8)


def _make_volume(store, read_retries: int = 4) -> Volume:
    """Build a Volume without __init__, which needs network and config."""
    vol = Volume.__new__(Volume)
    vol.type = "zarr"
    vol.data = [store]
    vol.dtype = store.dtype
    vol.normalization_scheme = "none"
    vol.global_mean = None
    vol.global_std = None
    vol.intensity_props = None
    vol.return_as_type = "none"
    vol.return_as_tensor = False
    vol.verbose = False
    vol.read_retries = read_retries
    return vol


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("vesuvius.data.volume.time.sleep", lambda _s: None)


def _idx():
    return (slice(0, 4), slice(0, 4), slice(0, 4))


def test_transient_error_is_retried_then_succeeds() -> None:
    store = _FlakyStore([
        OSError("Response payload is not completed"),
        OSError("[SSL: RECORD_LAYER_FAILURE] record layer failure"),
    ])
    result = _make_volume(store)[_idx()]
    assert store.reads == 3
    assert result.shape == (4, 4, 4)


def test_transient_error_exhausts_attempts_and_raises() -> None:
    store = _FlakyStore([OSError("connection reset by peer")] * 5)
    with pytest.raises(OSError):
        _make_volume(store, read_retries=3)[_idx()]
    assert store.reads == 3


def test_deterministic_error_is_not_retried() -> None:
    store = _FlakyStore([IndexError("index 99 is out of bounds")])
    with pytest.raises(IndexError):
        _make_volume(store)[_idx()]
    assert store.reads == 1, "a coding error must fail fast, not be retried"


def test_retries_disabled_with_one_attempt() -> None:
    store = _FlakyStore([OSError("connection reset by peer")] * 3)
    with pytest.raises(OSError):
        _make_volume(store, read_retries=1)[_idx()]
    assert store.reads == 1


@pytest.mark.parametrize(
    "message",
    [
        "Response payload is not completed: ContentLengthError",
        "Not enough data to satisfy content length header (received 1202432 of 2097152 bytes)",
        "[SSL: RECORD_LAYER_FAILURE] record layer failure",
        "Connection reset by peer",
        "botocore error: An error occurred (503) when calling GetObject: SlowDown",
        "read operation timed out",
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


def test_transient_cause_detected_through_wrapper() -> None:
    # zarr/fsspec wrap the underlying network error; the cause chain must be walked.
    try:
        try:
            raise OSError("Response payload is not completed")
        except OSError as inner:
            raise RuntimeError("Failed to get item 6 (pos 8064,3264,3200)") from inner
    except RuntimeError as wrapped:
        assert _is_transient_read_error(wrapped)
