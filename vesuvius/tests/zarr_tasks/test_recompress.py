import zarr
import numpy as np
import pytest

from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
    _create_recompressed_array,
    _ZARR_MAJOR_VERSION,
)


def _make_source(tmp_path, **open_kwargs):
    data = np.arange(64, dtype="uint8").reshape(4, 4, 4) + 1
    path = tmp_path / "vol.zarr"
    arr = zarr.open(
        str(path), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="uint8", **open_kwargs
    )
    arr[:] = data
    return path, data


def test_run_inplace_recompresses_v2_source(tmp_path):
    path, data = _make_source(tmp_path)

    task = RecompressTask(
        RecompressConfig(input_zarr=str(path), output_zarr=None, num_workers=1, inplace=True)
    )
    task.prepare()
    task._run_inplace()

    out = zarr.open(str(path), mode="r")
    assert np.array_equal(out[:], data)


@pytest.mark.skipif(_ZARR_MAJOR_VERSION < 3, reason="zarr_format=3 arrays require zarr-python 3.x")
def test_run_inplace_recompresses_v3_source(tmp_path):
    path, data = _make_source(tmp_path, zarr_format=3)

    task = RecompressTask(
        RecompressConfig(input_zarr=str(path), output_zarr=None, num_workers=1, inplace=True)
    )
    task.prepare()
    task._run_inplace()

    out = zarr.open(str(path), mode="r")
    assert np.array_equal(out[:], data)
    assert out.metadata.zarr_format == 3


@pytest.mark.skipif(_ZARR_MAJOR_VERSION < 3, reason="zarr_format=3 arrays require zarr-python 3.x")
def test_create_recompressed_array_v3_uses_compressors_not_compressor(tmp_path):
    arr = _create_recompressed_array(
        str(tmp_path / "out.zarr"),
        shape=(4, 4, 4),
        chunks=(2, 2, 2),
        dtype=np.dtype("uint8"),
        compression_level=1,
        zarr_format=3,
    )
    assert arr.metadata.zarr_format == 3
