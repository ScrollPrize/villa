from pathlib import Path

import numpy as np
import pytest
import tifffile

from vesuvius.neural_tracing.datasets.common import voxelize_surface_grid
from vesuvius.neural_tracing.inference import infer_rowcol_triplet_wraps as triplet_wraps


def test_select_middle_bbox_records_returns_centered_slice():
    records = [{"bbox_id": i} for i in range(8)]

    selected = triplet_wraps._select_middle_bbox_records(records, count=3)

    assert [item["bbox_id"] for item in selected] == [2, 3, 4]


def test_parse_args_accepts_save_tifs():
    args = triplet_wraps.parse_args(
        [
            "--tifxyz-path",
            "input",
            "--volume-path",
            "volume.zarr",
            "--checkpoint-path",
            "checkpoint.pth",
            "--save-tifs",
        ]
    )

    assert args.save_tifs is True


def test_parse_args_rejects_save_tifs_with_iterations():
    with pytest.raises(SystemExit):
        triplet_wraps.parse_args(
            [
                "--tifxyz-path",
                "input",
                "--volume-path",
                "volume.zarr",
                "--checkpoint-path",
                "checkpoint.pth",
                "--save-tifs",
                "--iterations",
                "2",
                "--iter-direction",
                "front",
            ]
        )


def test_save_selected_bbox_tifs_writes_expected_stacks(tmp_path):
    items = [
        {
            "bbox_id": 17,
            "min_corner": np.array([10, 20, 30], dtype=np.int32),
            "volume": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "cond_vox": np.ones((2, 3, 4), dtype=np.float32),
            "prior_unit_normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
    ]
    disp_pred_np = np.arange(6 * 2 * 3 * 4, dtype=np.float32).reshape(1, 6, 2, 3, 4)

    manifest = triplet_wraps._save_selected_bbox_tifs(
        out_dir=str(tmp_path),
        out_prefix="debug_run",
        items=items,
        disp_pred_np=disp_pred_np,
    )

    assert manifest["bbox_count"] == 1
    saved = manifest["saved_items"][0]
    assert Path(saved["input_crop_tif"]).is_file()
    assert Path(saved["voxelized_input_sheet_tif"]).is_file()
    assert Path(saved["displacement_field_tif"]).is_file()
    assert Path(saved["front_displacement_vector_tif"]).is_file()
    assert Path(saved["back_displacement_vector_tif"]).is_file()
    assert Path(saved["front_displacement_magnitude_tif"]).is_file()
    assert Path(saved["back_displacement_magnitude_tif"]).is_file()
    assert Path(saved["front_signed_displacement_along_prior_tif"]).is_file()
    assert Path(saved["back_signed_displacement_along_prior_tif"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()

    np.testing.assert_array_equal(tifffile.imread(saved["input_crop_tif"]), items[0]["volume"])
    np.testing.assert_array_equal(tifffile.imread(saved["voxelized_input_sheet_tif"]), items[0]["cond_vox"])
    np.testing.assert_array_equal(tifffile.imread(saved["displacement_field_tif"]), disp_pred_np[0])
    np.testing.assert_array_equal(tifffile.imread(saved["front_displacement_vector_tif"]), disp_pred_np[0, 0:3])
    np.testing.assert_array_equal(tifffile.imread(saved["back_displacement_vector_tif"]), disp_pred_np[0, 3:6])

    expected_front_mag = np.linalg.norm(disp_pred_np[0, 0:3], axis=0)
    expected_back_mag = np.linalg.norm(disp_pred_np[0, 3:6], axis=0)
    np.testing.assert_allclose(tifffile.imread(saved["front_displacement_magnitude_tif"]), expected_front_mag)
    np.testing.assert_allclose(tifffile.imread(saved["back_displacement_magnitude_tif"]), expected_back_mag)

    expected_front_signed = disp_pred_np[0, 2]
    expected_back_signed = -disp_pred_np[0, 5]
    np.testing.assert_allclose(tifffile.imread(saved["front_signed_displacement_along_prior_tif"]), expected_front_signed)
    np.testing.assert_allclose(tifffile.imread(saved["back_signed_displacement_along_prior_tif"]), expected_back_signed)


class _DummyFullResSurface:
    def __init__(self, full_grid, valid, scale):
        self._full_grid = np.asarray(full_grid, dtype=np.float32)
        self._valid = np.asarray(valid, dtype=bool)
        self._scale = tuple(float(v) for v in scale)
        self.full_resolution_shape = self._full_grid.shape[:2]
        self.resolution = "stored"

    def use_full_resolution(self):
        self.resolution = "full"
        return self

    def __getitem__(self, key):
        if self.resolution != "full":
            raise AssertionError("surface must be set to full resolution before slicing")
        sub = self._full_grid[key]
        valid = self._valid[key]
        return sub[..., 2], sub[..., 1], sub[..., 0], valid


def test_voxelize_local_surface_from_input_surface_uses_lazy_fullres_slice():
    yy, xx = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32), indexing="ij")
    full_grid = np.stack([np.full_like(xx, 1.0), yy + 1.0, xx + 1.0], axis=-1)
    full_valid = np.ones((4, 4), dtype=bool)
    surface = _DummyFullResSurface(full_grid, full_valid, scale=(0.5, 0.5))

    input_grid = full_grid[::2, ::2]
    input_valid = np.ones((2, 2), dtype=bool)
    min_corner = np.array([0, 0, 0], dtype=np.int32)
    crop_size = (4, 8, 8)

    cond_vox = triplet_wraps._voxelize_local_surface_from_input_surface(
        surface=surface,
        input_grid=input_grid,
        input_valid=input_valid,
        min_corner=min_corner,
        crop_size=crop_size,
    )

    expected = voxelize_surface_grid(full_grid.astype(np.float64, copy=False), crop_size).astype(np.float32, copy=False)
    np.testing.assert_array_equal(cond_vox, expected)
    assert surface.resolution == "full"


def test_estimate_local_unit_normal_uses_bbox_uv_points():
    input_normals = np.array(
        [
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    input_normals_valid = np.array(
        [
            [True, True],
            [True, True],
        ],
        dtype=bool,
    )
    uv_rc = np.array([[0, 1], [1, 1]], dtype=np.int32)
    fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    normal, count = triplet_wraps._estimate_local_unit_normal(
        input_normals=input_normals,
        input_normals_valid=input_normals_valid,
        uv_rc=uv_rc,
        fallback_unit_normal=fallback,
    )

    np.testing.assert_allclose(normal, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert count == 2


def test_build_triplet_model_batch_uses_per_item_normals():
    items = [
        {
            "volume": np.zeros((2, 2, 2), dtype=np.float32),
            "cond_vox": np.ones((2, 2, 2), dtype=np.float32),
            "prior_unit_normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        },
        {
            "volume": np.zeros((2, 2, 2), dtype=np.float32),
            "cond_vox": np.ones((2, 2, 2), dtype=np.float32),
            "prior_unit_normal": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        },
    ]

    batch = triplet_wraps._build_triplet_model_batch(items, crop_size=(2, 2, 2), mask_mode="cond")

    expected_front_0 = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float32)[:, None, None, None], (3, 2, 2, 2))
    expected_back_0 = np.broadcast_to(np.array([0.0, 0.0, -1.0], dtype=np.float32)[:, None, None, None], (3, 2, 2, 2))
    expected_front_1 = np.broadcast_to(np.array([0.0, 1.0, 0.0], dtype=np.float32)[:, None, None, None], (3, 2, 2, 2))
    expected_back_1 = np.broadcast_to(np.array([0.0, -1.0, 0.0], dtype=np.float32)[:, None, None, None], (3, 2, 2, 2))

    np.testing.assert_allclose(batch[0, 2:5], expected_front_0)
    np.testing.assert_allclose(batch[0, 5:8], expected_back_0)
    np.testing.assert_allclose(batch[1, 2:5], expected_front_1)
    np.testing.assert_allclose(batch[1, 5:8], expected_back_1)
