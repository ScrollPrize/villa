"""RecompressTask on an OME-Zarr group must hand the source LEVEL to its workers, not the group root.

With the group root, every worker failed on a group it could not slice, and --skip-corrupt then
filled every destination chunk with zeros and reported success (#2186).
"""
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.recompress import (
    RecompressConfig,
    RecompressTask,
    _recompress_worker,
)


def _ome_group(tmp_path):
    root = str(tmp_path / "input.zarr")
    data = (np.arange(6 * 6 * 6, dtype=np.uint8).reshape(6, 6, 6) % 250) + 1
    group = zarr.open_group(root, mode="w", zarr_format=2)
    arr = group.create_array("0", shape=data.shape, chunks=(4, 4, 4), dtype=data.dtype)
    arr[:] = data
    return root, data


@pytest.mark.parametrize("skip_corrupt", [False, True])
def test_ome_zarr_group_keeps_its_data(tmp_path, skip_corrupt):
    root, data = _ome_group(tmp_path)
    out = str(tmp_path / "output.zarr")
    RecompressTask(
        RecompressConfig(
            input_zarr=root,
            output_zarr=out,
            num_workers=1,
            skip_corrupt=skip_corrupt,
            num_levels=1,
        )
    ).run()
    np.testing.assert_array_equal(zarr.open(f"{out}/0", mode="r")[:], data)


def test_worker_does_not_zero_fill_a_group_input(tmp_path):
    root, _ = _ome_group(tmp_path)
    out = zarr.open(
        str(tmp_path / "out0"), mode="w", shape=(6, 6, 6), chunks=(4, 4, 4), dtype="u1"
    )
    coords = ((0, 4), (0, 4), (0, 4))
    with pytest.raises(TypeError):
        _recompress_worker((root, str(tmp_path / "out0"), coords, None, None, True))
    assert not out[:].any()
