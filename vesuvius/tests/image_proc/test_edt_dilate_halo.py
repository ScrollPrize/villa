import numpy as np
import pytest
import zarr

scipy_ndimage = pytest.importorskip("scipy.ndimage")

from vesuvius.image_proc.run.zarr_tasks.tasks import edt_dilate as m


@pytest.mark.parametrize("black_border", [False, True])
def test_dilation_crosses_processing_chunk_boundaries(tmp_path, black_border):
    shape, ch, dist = (32, 64, 64), (32, 32, 32), 5
    vol = np.zeros(shape, np.uint8)
    vol[16, 30, 30] = 1  # 2 voxels from the y/x boundary at 32
    zarr.open_group(str(tmp_path / "in.zarr"), mode="w").create_dataset("0", data=vol, chunks=ch)
    out = zarr.open_group(str(tmp_path / "out.zarr"), mode="w")
    out.create_dataset("0", shape=shape, chunks=ch, dtype=np.uint8, fill_value=0)
    m._WORKER_STATE.update(
        input_path=str(tmp_path / "in.zarr"), output_path=str(tmp_path / "out.zarr"), resolution="0"
    )
    for cy in range(2):
        for cx in range(2):
            bounds = (0, 32, cy * 32, cy * 32 + 32, cx * 32, cx * 32 + 32)
            m._edt_dilate_worker(((0, cy, cx), bounds, dist, black_border))
    expected = scipy_ndimage.distance_transform_edt(vol == 0) <= dist
    np.testing.assert_array_equal(out["0"][:].astype(bool), expected)
