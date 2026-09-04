"""A transient remote chunk read must not abort ink inference (issue #1666).

Streaming a published surface volume issues one read per patch. A single
truncated response (``ClientPayloadError: Response payload is not completed
... received 173631 of 507920 bytes``) propagated out of the DataLoader worker
and killed the whole run with exit 1. These tests drive the real readers
(``FlatPatchReader`` and ``read_bbox_with_padding``) against arrays that fail
the way an object store does, and check deterministic errors still fail fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.ink_detection import volume_io
from vesuvius.ink_detection.inference.infer import (
    FlatPatchReader,
    compute_nonempty_mask_from_lowres_array,
    parse_args,
)
from vesuvius.ink_detection.volume_io import read_bbox_with_padding, read_with_retry


class _TruncatedPayload(Exception):
    """Mimics aiohttp.ClientPayloadError as zarr/fsspec surface it."""

    def __init__(self, received: int = 173631, total: int = 507920) -> None:
        super().__init__(
            "Response payload is not completed: <ContentLengthError: 400, "
            f"message='Not enough data to satisfy content length header "
            f"(received {received} of {total} bytes).'>. "
            "SSLError(1, '[SSL: RECORD_LAYER_FAILURE] record layer failure (_ssl.c:2713)')"
        )


class _FlakyArray:
    """Array-like store that raises a queue of errors before serving data."""

    def __init__(self, data: np.ndarray, errors: list[BaseException]) -> None:
        self._data = data
        self._errors = list(errors)
        self.reads = 0
        self.shape = data.shape
        self.dtype = data.dtype
        self.ndim = data.ndim

    def __getitem__(self, idx):
        self.reads += 1
        if self._errors:
            raise self._errors.pop(0)
        return self._data[idx]


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(volume_io.time, "sleep", lambda _s: None)


def _reader(array, *, retries: int = 4) -> FlatPatchReader:
    reader = FlatPatchReader(
        input_path="s3://vesuvius-challenge-open-data/fake/surface.zarr",
        resolution="0",
        depth_axis_first=True,
        height=array.shape[1],
        width=array.shape[2],
        layer_indices=np.arange(array.shape[0]),
        output_depth=array.shape[0],
        preprocessing="divide_255",
        read_retries=retries,
    )
    reader._array = array  # skip open_volume; the store is what is under test
    return reader


def _volume() -> np.ndarray:
    return np.arange(3 * 8 * 8, dtype=np.uint8).reshape(3, 8, 8)


def test_flat_reader_survives_one_truncated_chunk():
    data = _volume()
    array = _FlakyArray(data, [_TruncatedPayload()])
    patch = _reader(array).read(0, 0, 4, 4)
    np.testing.assert_array_equal(patch, np.moveaxis(data[:, :4, :4], 0, -1))
    assert array.reads == 2


def test_flat_reader_survives_repeated_truncation_at_different_offsets():
    data = _volume()
    array = _FlakyArray(data, [_TruncatedPayload(173631), _TruncatedPayload(299585)])
    patch = _reader(array).read(2, 2, 4, 4)
    np.testing.assert_array_equal(patch, np.moveaxis(data[:, 2:6, 2:6], 0, -1))
    assert array.reads == 3


def test_flat_reader_gives_up_after_configured_attempts():
    array = _FlakyArray(_volume(), [_TruncatedPayload()] * 4)
    with pytest.raises(_TruncatedPayload):
        _reader(array, retries=4).read(0, 0, 4, 4)
    assert array.reads == 4


def test_flat_reader_retries_can_be_disabled():
    array = _FlakyArray(_volume(), [_TruncatedPayload()])
    with pytest.raises(_TruncatedPayload):
        _reader(array, retries=1).read(0, 0, 4, 4)
    assert array.reads == 1


def test_flat_reader_does_not_retry_deterministic_errors():
    array = _FlakyArray(_volume(), [KeyError("0/1/2")])
    with pytest.raises(KeyError):
        _reader(array).read(0, 0, 4, 4)
    assert array.reads == 1


def test_occupancy_scan_read_retries_transient_failures():
    """The low-res occupancy scan is read once, before the patch loop; a
    truncated response there killed the run in the proxied reproduction."""
    lowres = np.zeros((3, 8, 8), dtype=np.uint8)
    lowres[:, 2:4, 2:4] = 1
    array = _FlakyArray(lowres, [_TruncatedPayload(9346, 28040)])
    occupancy = compute_nonempty_mask_from_lowres_array(array)
    assert occupancy.shape == (8, 8)
    assert occupancy[2:4, 2:4].all() and occupancy.sum() == 4
    assert array.reads == 2


def test_read_bbox_with_padding_retries_transient_failures():
    data = _volume()
    array = _FlakyArray(data, [_TruncatedPayload()])
    crop, valid = read_bbox_with_padding(array, (0, 0, 0, 3, 4, 4), fill_value=0)
    np.testing.assert_array_equal(crop, data[:, :4, :4])
    assert valid is not None
    assert array.reads == 2


def test_read_bbox_with_padding_fails_fast_on_deterministic_errors():
    array = _FlakyArray(_volume(), [IndexError("index out of range")])
    with pytest.raises(IndexError):
        read_bbox_with_padding(array, (0, 0, 0, 3, 4, 4))
    assert array.reads == 1


def test_read_with_retry_backoff_is_bounded(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(volume_io.time, "sleep", delays.append)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 6:
            raise ConnectionResetError("Connection reset by peer")
        return "ok"

    assert read_with_retry(flaky, retries=6) == "ok"
    assert delays == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_cli_exposes_read_retries():
    args = parse_args(["in.zarr", "ckpt.pth", "out.tif", "--read-retries", "2"])
    assert args.read_retries == 2
    assert parse_args(["in.zarr", "ckpt.pth", "out.tif"]).read_retries == volume_io.DEFAULT_READ_RETRIES
    with pytest.raises(SystemExit):
        parse_args(["in.zarr", "ckpt.pth", "out.tif", "--read-retries", "0"])
