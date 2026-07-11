from pathlib import Path

import numpy as np
import tifffile

from koine_machines.inference import infer_full3d_tifxyz as infer


def test_read_tifxyz_points_scales_native_coordinates(tmp_path):
    x = np.array([[0.0, 4.0], [8.0, np.nan]], dtype=np.float32)
    y = np.array([[4.0, 8.0], [12.0, np.nan]], dtype=np.float32)
    z = np.array([[8.0, 12.0], [16.0, np.nan]], dtype=np.float32)
    for name, values in (("x.tif", x), ("y.tif", y), ("z.tif", z)):
        tifffile.imwrite(tmp_path / name, values)

    scaled_x, scaled_y, scaled_z, valid = infer.read_tifxyz_points(
        tmp_path,
        native_coordinate_scale=0.25,
    )

    np.testing.assert_array_equal(valid, np.array([[True, True], [True, False]]))
    np.testing.assert_allclose(scaled_x[valid], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(scaled_y[valid], np.array([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(scaled_z[valid], np.array([2.0, 3.0, 4.0], dtype=np.float32))


def test_full3d_dataset_uses_full_resolution_tifxyz_and_level_two_transform(monkeypatch, tmp_path):
    class StubTifxyz:
        def __init__(self):
            self.used_full_resolution = False

        def use_full_resolution(self):
            self.used_full_resolution = True
            return self

    stub = StubTifxyz()
    monkeypatch.setattr(infer.tifxyz, "read_tifxyz", lambda *args, **kwargs: stub)

    dataset = infer.Full3DPatchDataset(
        tifxyz_dir=Path(tmp_path),
        volume_path="unused.zarr",
        resolution="2",
        patches=[],
        patch_size_zyx=(8, 8, 8),
        config={"mode": "full_3d_single_wrap"},
    )

    assert dataset._ensure_tifxyz() is stub
    assert stub.used_full_resolution
    assert dataset.native_downsample_factor == 4
    assert dataset.native_coordinate_scale == 0.25
    assert dataset.coarse_native_pad == 5
