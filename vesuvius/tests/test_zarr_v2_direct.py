"""DirectZarrV2Reader must match zarr-python slicing exactly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr
from numcodecs import Blosc, Zstd

from vesuvius.utils.zarr_v2_direct import DirectZarrV2Reader, UnsupportedZarrArray


def _write_v2(path: Path, data: np.ndarray, *, chunks, compressor, separator, fill_value=0):
    zarr.create_array(
        store=str(path),
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        zarr_format=2,
        compressors=compressor,
        chunk_key_encoding={"name": "v2", "separator": separator},
        fill_value=fill_value,
        overwrite=True,
    )[...] = data
    return zarr.open(str(path), mode="r")


@pytest.mark.parametrize("separator", ["/", "."])
@pytest.mark.parametrize("compressor", [None, Zstd(level=1), Blosc(cname="zstd", clevel=1)])
def test_direct_reader_matches_zarr_on_random_regions(tmp_path: Path, separator, compressor) -> None:
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, size=(5, 37, 53), dtype=np.uint8)
    data[:, 20:, 30:] = 0  # fully-zero chunks are still written; missing chunks tested below
    array = _write_v2(tmp_path / "a.zarr", data, chunks=(4, 16, 16), compressor=compressor, separator=separator)
    reader = DirectZarrV2Reader.open(tmp_path / "a.zarr")
    assert reader.shape == data.shape and reader.chunks == (4, 16, 16)
    np.testing.assert_array_equal(reader.read([(0, 5), (0, 37), (0, 53)]), data)
    for _ in range(25):
        z0, z1 = sorted(rng.integers(0, 6, size=2))
        y0, y1 = sorted(rng.integers(0, 38, size=2))
        x0, x1 = sorted(rng.integers(0, 54, size=2))
        got = reader.read([(z0, z1), (y0, y1), (x0, x1)])
        expected = np.asarray(array[z0:z1, y0:y1, x0:x1])
        np.testing.assert_array_equal(got, expected)
        assert got.flags.c_contiguous


def test_direct_reader_missing_chunks_use_fill_value(tmp_path: Path) -> None:
    data = np.full((2, 32, 32), 7, dtype=np.uint16)
    _write_v2(tmp_path / "a.zarr", data, chunks=(2, 16, 16), compressor=Zstd(level=1), separator="/", fill_value=3)
    (tmp_path / "a.zarr" / "0" / "1" / "1").unlink()
    reader = DirectZarrV2Reader.open(tmp_path / "a.zarr")
    expected = np.asarray(zarr.open(str(tmp_path / "a.zarr"), mode="r")[:])
    assert expected[0, 16:, 16:].min() == 3
    np.testing.assert_array_equal(reader.read([(0, 2), (0, 32), (0, 32)]), expected)


def test_direct_reader_float_nan_fill_and_4d(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    data = rng.standard_normal((1, 3, 20, 20)).astype(np.float32)
    _write_v2(tmp_path / "f.zarr", data, chunks=(1, 3, 8, 8), compressor=Zstd(level=1), separator="/", fill_value=np.nan)
    (tmp_path / "f.zarr" / "0" / "0" / "2" / "2").unlink()
    reader = DirectZarrV2Reader.open(tmp_path / "f.zarr")
    expected = np.asarray(zarr.open(str(tmp_path / "f.zarr"), mode="r")[:])
    assert np.isnan(expected[0, 0, 16, 16])
    np.testing.assert_array_equal(reader.read([(0, 1), (0, 3), (0, 20), (0, 20)]), expected)


def test_direct_reader_rejects_unsupported(tmp_path: Path) -> None:
    with pytest.raises(UnsupportedZarrArray):
        DirectZarrV2Reader.open(tmp_path)  # no .zarray
    zarr.create_array(store=str(tmp_path / "v3.zarr"), shape=(4, 4), chunks=(2, 2), dtype="u1", zarr_format=3)
    with pytest.raises(UnsupportedZarrArray):
        DirectZarrV2Reader.open(tmp_path / "v3.zarr")
    _write_v2(tmp_path / "b.zarr", np.zeros((4, 4), np.uint8), chunks=(2, 2), compressor=None, separator=".")
    reader = DirectZarrV2Reader.open(tmp_path / "b.zarr")
    with pytest.raises(ValueError):
        reader.read([(0, 5), (0, 4)])
