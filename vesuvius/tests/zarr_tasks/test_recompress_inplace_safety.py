"""In-place recompression must not lose the source array.

``--inplace`` copies each chunk into ``<array>_tmp`` and deletes the source
chunk as it goes, then swaps the temp array in. Two ways this destroyed data:
an input path with a trailing slash put the temp array inside the source
array, and a run that failed part-way removed the temp array that by then held
every chunk already copied.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks import recompress
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)

ZARR3 = int(zarr.__version__.split(".")[0]) >= 3

if ZARR3 and not hasattr(recompress, "open_zarr"):
    # On zarr 3 the in-place temp array cannot be created yet (#2069).
    pytest.skip(
        "in-place recompression needs the zarr 3 temp-array fix",
        allow_module_level=True,
    )


def _make_source(path: Path, data: np.ndarray, chunks) -> None:
    kwargs = dict(
        mode="w",
        shape=data.shape,
        chunks=chunks,
        dtype=data.dtype,
        dimension_separator="/",
    )
    if ZARR3:
        kwargs["zarr_format"] = 2
    zarr.open(str(path), **kwargs)[:] = data


def _run_inplace(path: str, skip_corrupt: bool = False) -> None:
    RecompressTask(
        RecompressConfig(
            input_zarr=path,
            output_zarr=None,
            num_workers=1,
            inplace=True,
            skip_corrupt=skip_corrupt,
        )
    ).run()


def _data(shape) -> np.ndarray:
    return (np.arange(np.prod(shape)) % 251 + 1).astype("u1").reshape(shape)


def test_trailing_slash_recompresses_the_array_in_place(tmp_path: Path) -> None:
    path = tmp_path / "vol.zarr"
    data = _data((8, 8, 8))
    _make_source(path, data, (4, 4, 4))

    _run_inplace(str(path) + "/")

    meta = json.loads((path / ".zarray").read_text())
    assert meta["compressor"]["cname"] == "zstd"
    assert np.array_equal(zarr.open(str(path), mode="r")[:], data)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["vol.zarr"]


def test_failed_run_leaves_every_chunk_in_the_source(tmp_path: Path) -> None:
    path = tmp_path / "vol.zarr"
    data = _data((12, 12, 12))
    _make_source(path, data, (4, 4, 4))
    # A truncated chunk, as an interrupted copy leaves it. With one worker the
    # chunks run in order, so 26 chunks are copied before this one fails.
    bad = path / "2" / "2" / "2"
    bad.write_bytes(bad.read_bytes()[:10])
    before = zarr.open(str(path), mode="r")[:8]

    with pytest.raises(RuntimeError, match="Recompression failed"):
        _run_inplace(str(path))

    files = [p for p in path.rglob("*") if p.is_file() and not p.name.startswith(".")]
    assert len(files) == 27
    assert np.array_equal(zarr.open(str(path), mode="r")[:8], before)
    assert not (tmp_path / "vol.zarr_tmp").exists()


def test_skip_corrupt_still_fills_the_bad_chunk_with_zeros(tmp_path: Path) -> None:
    path = tmp_path / "vol.zarr"
    data = _data((12, 12, 12))
    _make_source(path, data, (4, 4, 4))
    bad = path / "2" / "2" / "2"
    bad.write_bytes(bad.read_bytes()[:10])

    _run_inplace(str(path), skip_corrupt=True)

    out = zarr.open(str(path), mode="r")[:]
    assert not out[8:, 8:, 8:].any()
    expected = data.copy()
    expected[8:, 8:, 8:] = 0
    assert np.array_equal(out, expected)
