"""Regression coverage for zarr_tasks under both supported zarr-python majors.

The package CI runs this suite with zarr 2.18.7 and 3.2.1.  zarr_tasks writes
the Vesuvius Zarr-v2 on-disk layout in both environments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from vesuvius.data.utils import open_zarr
from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
)
from vesuvius.image_proc.run.zarr_tasks.tasks.threshold import (
    ThresholdConfig,
    ThresholdTask,
)


def _write_input(path: Path, data: np.ndarray):
    arr = open_zarr(
        path=str(path),
        mode="w",
        shape=data.shape,
        chunks=(1, 2, 2),
        dtype=data.dtype,
        compressor=Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE),
    )
    arr[...] = data
    return arr


def _assert_v2_array(path: Path) -> None:
    assert (path / ".zarray").exists()
    assert not (path / "zarr.json").exists()


def test_threshold_writes_v2_pyramid_with_zarr2_and_zarr3(tmp_path: Path) -> None:
    source = np.arange(2 * 4 * 4, dtype=np.uint8).reshape(2, 4, 4)
    input_path = tmp_path / "input.zarr"
    output_path = tmp_path / "threshold.zarr"
    _write_input(input_path, source)

    task = ThresholdTask(
        ThresholdConfig(
            input_zarr=str(input_path),
            output_zarr=str(output_path),
            num_workers=1,
            threshold=7,
            num_levels=2,
        )
    )
    task.run()

    output = zarr.open_group(str(output_path), mode="r")
    expected = np.where(source > 7, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(output["0"][:], expected)

    assert sorted(output.keys()) == ["0", "1"]
    assert output.attrs["multiscales"][0]["datasets"] == [
        {"path": "0"},
        {"path": "1"},
    ]
    assert (output_path / ".zgroup").exists()
    assert not (output_path / "zarr.json").exists()
    _assert_v2_array(output_path / "0")
    _assert_v2_array(output_path / "1")


def test_recompress_inplace_remains_v2_with_zarr2_and_zarr3(tmp_path: Path) -> None:
    source = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    input_path = tmp_path / "recompress.zarr"
    arr = _write_input(input_path, source)
    arr.attrs["note"] = "preserve-me"

    task = RecompressTask(
        RecompressConfig(
            input_zarr=str(input_path),
            output_zarr=None,
            num_workers=1,
            inplace=True,
            compression_level=1,
            num_levels=1,
        )
    )
    task.run()

    reopened = zarr.open(str(input_path), mode="r")
    np.testing.assert_array_equal(reopened[:], source)
    assert reopened.attrs["note"] == "preserve-me"
    _assert_v2_array(input_path)
