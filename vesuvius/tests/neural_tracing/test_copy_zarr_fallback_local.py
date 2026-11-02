"""Offline check of the zarr fallback reader in infer_rowcol_triplet_wraps.py (the path used when ``vc`` is not
installed), on a small OME-Zarr written to tmp_path: crops inside, across chunk edges, past both volume edges and over
a missing chunk match numpy, and a .zattrs that lists scale levels missing on disk (a partial local copy) still opens.
The network test next to it compares the same reader with the public PHerc0841 chunks."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vesuvius.neural_tracing.inference.infer_rowcol_triplet_wraps import _open_zarr_volume_level

SHAPE = (40, 44, 48)
CHUNKS = (16, 16, 16)
MISSING = (1, 0, 2)  # this chunk file is not written, so it reads as fill (0)


def _volume():
    z, y, x = np.indices(SHAPE)
    data = ((z * 7 + y * 3 + x) % 250 + 1).astype(np.uint8)  # never 0, so a fill read cannot pass for data
    expect = data.copy()
    expect[tuple(slice(k * c, (k + 1) * c) for k, c in zip(MISSING, CHUNKS))] = 0
    return data, expect


def _write_omezarr(root, data, levels_listed):
    root.mkdir()
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps(
        {"multiscales": [{"version": "0.4", "datasets": [{"path": str(n)} for n in levels_listed]}]}))
    level = root / "0"
    level.mkdir()
    (level / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": list(SHAPE), "chunks": list(CHUNKS), "dtype": "|u1", "fill_value": 0,
        "compressor": None, "filters": None, "order": "C", "dimension_separator": "/"}))
    grid = [range(-(-n // c)) for n, c in zip(SHAPE, CHUNKS)]
    for key in np.ndindex(*(len(g) for g in grid)):
        if key == MISSING:
            continue
        chunk = np.zeros(CHUNKS, dtype=np.uint8)
        part = data[tuple(slice(k * c, (k + 1) * c) for k, c in zip(key, CHUNKS))]
        chunk[tuple(slice(0, n) for n in part.shape)] = part  # edge chunks are stored padded to full size
        path = level.joinpath(*(str(k) for k in key))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(chunk.tobytes())


def _reference(expect, start, size):
    out = np.zeros(size, dtype=np.uint8)
    lo = [max(0, s) for s in start]
    hi = [min(n, s + d) for n, s, d in zip(SHAPE, start, size)]
    if all(h > l for l, h in zip(lo, hi)):
        out[tuple(slice(l - s, h - s) for l, h, s in zip(lo, hi, start))] = expect[
            tuple(slice(l, h) for l, h in zip(lo, hi))]
    return out


@pytest.fixture
def partial_copy(tmp_path):
    """Level 0 on disk, .zattrs listing levels 0, 1 and 2."""
    data, expect = _volume()
    _write_omezarr(tmp_path / "vol.zarr", data, levels_listed=(0, 1, 2))
    return tmp_path / "vol.zarr", expect


CROPS = {
    "inside": ((2, 3, 4), (8, 9, 10)),
    "chunk_corner": ((10, 12, 14), (12, 12, 12)),  # across a corner shared by 8 chunks
    "missing_chunk": ((14, 2, 30), (8, 8, 8)),  # half in the chunk with no file
    "below_zero": ((-5, -3, -7), (10, 10, 10)),
    "past_upper_edge": ((34, 38, 40), (10, 10, 12)),  # data inside, fill past the last voxel on every axis
    "fully_outside": ((50, 0, 0), (4, 4, 4)),
}


def test_levels_listed_but_missing_on_disk_still_open(partial_copy):
    path, _ = partial_copy
    volume, level = _open_zarr_volume_level(str(path), 0)
    assert level == 0
    assert volume.shape == SHAPE
    _, nearest = _open_zarr_volume_level(str(path), 2)  # asked for a missing level: the nearest one on disk
    assert nearest == 0


@pytest.mark.parametrize("name", sorted(CROPS))
def test_crop_matches_numpy(partial_copy, name):
    path, expect = partial_copy
    volume, _ = _open_zarr_volume_level(str(path), 0)
    start, size = CROPS[name]
    want = _reference(expect, start, size)
    if name == "past_upper_edge":
        assert want.any() and not want[-1, -1, -1]  # the crop has data and runs past the edge
    got = volume[tuple(slice(s, s + n) for s, n in zip(start, size))]  # the reads the Copy script makes
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got, want)
