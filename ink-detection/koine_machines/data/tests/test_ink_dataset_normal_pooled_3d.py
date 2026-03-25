import numpy as np
import torch

from koine_machines.data.normal_pooled_sample import _build_normal_pooled_flat_metadata
from koine_machines.data.normal_pooled_sample import (
    _pack_normal_pooled_augmentation_data,
    _restore_normal_pooled_augmentation_data,
)
from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform


class StubPatchTifxyz:
    def __init__(self, normals_xyz):
        self._normals_xyz = normals_xyz

    def get_normals(self, row_start, row_end, col_start, col_end):
        del row_start, row_end, col_start, col_end
        nx = self._normals_xyz[..., 0]
        ny = self._normals_xyz[..., 1]
        nz = self._normals_xyz[..., 2]
        return nx, ny, nz


def test_build_normal_pooled_flat_metadata_returns_expected_contract():
    support_patch_zyxs = np.array(
        [
            [[10.0, 20.0, 30.0], [10.0, 20.0, 31.0]],
            [[11.0, 21.0, 30.0], [11.0, 21.0, 31.0]],
        ],
        dtype=np.float32,
    )
    support_valid = np.array(
        [
            [True, True],
            [True, False],
        ],
        dtype=bool,
    )
    normals_xyz = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
            [[0.0, 0.0, -3.0], [np.nan, np.nan, np.nan]],
        ],
        dtype=np.float32,
    )

    metadata = _build_normal_pooled_flat_metadata(
        patch_tifxyz=StubPatchTifxyz(normals_xyz),
        support_bbox=(100, 102, 200, 202),
        support_patch_zyxs=support_patch_zyxs,
        support_valid=support_valid,
        support_inklabels_flat_patch=np.array([[0, 255], [255, 0]], dtype=np.uint8),
        support_supervision_flat_patch=np.array([[255, 255], [0, 255]], dtype=np.uint8),
        crop_bbox=(10, 20, 30, 14, 24, 34),
    )

    assert metadata["flat_target"].shape == (1, 2, 2)
    assert metadata["flat_supervision"].shape == (1, 2, 2)
    assert metadata["flat_valid"].shape == (1, 2, 2)
    assert metadata["flat_points_local_zyx"].shape == (2, 2, 3)
    assert metadata["flat_normals_local_zyx"].shape == (2, 2, 3)

    np.testing.assert_array_equal(
        metadata["flat_target"].numpy(),
        np.array([[[0.0, 1.0], [1.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        metadata["flat_supervision"].numpy(),
        np.array([[[1.0, 1.0], [0.0, 1.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        metadata["flat_valid"].numpy(),
        np.array([[[1.0, 1.0], [1.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        metadata["flat_points_local_zyx"].numpy()[0, 1],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        metadata["flat_normals_local_zyx"].numpy()[0, 0],
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        metadata["flat_normals_local_zyx"].numpy()[0, 1],
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        metadata["flat_normals_local_zyx"].numpy()[1, 0],
        np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        metadata["flat_normals_local_zyx"].numpy()[1, 1],
        np.zeros((3,), dtype=np.float32),
    )


def test_build_normal_pooled_flat_metadata_marks_crop_edge_overshoot_invalid():
    metadata = _build_normal_pooled_flat_metadata(
        patch_tifxyz=StubPatchTifxyz(
            np.array(
                [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                dtype=np.float32,
            )
        ),
        support_bbox=(0, 1, 0, 2),
        support_patch_zyxs=np.array(
            [[[10.0, 20.0, 30.0], [10.0, 20.0, 31.2]]],
            dtype=np.float32,
        ),
        support_valid=np.array([[True, True]], dtype=bool),
        support_inklabels_flat_patch=np.array([[1, 1]], dtype=np.uint8),
        support_supervision_flat_patch=np.array([[1, 1]], dtype=np.uint8),
        crop_bbox=(10, 20, 30, 11, 21, 32),
    )

    np.testing.assert_array_equal(
        metadata["flat_valid"].numpy(),
        np.array([[[1.0, 0.0]]], dtype=np.float32),
    )


def test_normal_pooled_augmentation_pack_restore_with_rot90_keeps_flat_targets_fixed():
    data = {
        "image": torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3),
        "surface_mask": torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) + 100.0,
        "flat_target": torch.tensor([[[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
        "flat_supervision": torch.tensor([[[1.0, 1.0], [0.0, 1.0]]], dtype=torch.float32),
        "flat_valid": torch.tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32),
        "flat_points_local_zyx": torch.tensor(
            [
                [[0.0, 0.0, 1.0], [1.0, 2.0, 0.0]],
                [[1.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "flat_normals_local_zyx": torch.tensor(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
    }

    augmentation_data, flat_valid_mask = _pack_normal_pooled_augmentation_data(data)
    transform = Rot90Transform(num_axis_combinations=1, num_rot_per_combination=(1,), allowed_axes={1, 2})
    augmented = transform.apply(
        augmentation_data,
        num_rot_per_combination=[1],
        axis_combinations=[[2, 3]],
        crop_shape=(2, 3, 3),
    )
    restored = _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)

    expected_image = torch.rot90(data["image"], k=1, dims=(2, 3))
    expected_surface_mask = torch.rot90(data["surface_mask"], k=1, dims=(2, 3))
    assert torch.equal(restored["image"], expected_image)
    assert torch.equal(restored["surface_mask"], expected_surface_mask)
    assert torch.equal(restored["flat_target"], data["flat_target"])
    assert torch.equal(restored["flat_supervision"], data["flat_supervision"])
    assert torch.equal(restored["flat_valid"], data["flat_valid"])

    expected_points = torch.zeros_like(data["flat_points_local_zyx"])
    expected_points[0, 0] = torch.tensor([0.0, 1.0, 0.0])
    expected_points[0, 1] = torch.tensor([1.0, 2.0, 2.0])
    expected_points[1, 0] = torch.tensor([1.0, 0.0, 1.0])
    assert torch.equal(restored["flat_points_local_zyx"], expected_points)

    expected_normals = torch.zeros_like(data["flat_normals_local_zyx"])
    expected_normals[0, 0] = torch.tensor([0.0, 0.0, 1.0])
    expected_normals[0, 1] = torch.tensor([0.0, -1.0, 0.0])
    expected_normals[1, 0] = torch.tensor([1.0, 0.0, 0.0])
    assert torch.equal(restored["flat_normals_local_zyx"], expected_normals)


def test_normal_pooled_augmentation_pack_restore_with_transpose_reorders_geometry():
    data = {
        "image": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2),
        "surface_mask": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) + 10.0,
        "flat_target": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
        "flat_supervision": torch.tensor([[[1.0, 1.0]]], dtype=torch.float32),
        "flat_valid": torch.tensor([[[1.0, 1.0]]], dtype=torch.float32),
        "flat_points_local_zyx": torch.tensor(
            [[[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ),
        "flat_normals_local_zyx": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ),
    }

    augmentation_data, flat_valid_mask = _pack_normal_pooled_augmentation_data(data)
    transform = TransposeAxesTransform(allowed_axes={1, 2})
    augmented = transform.apply(
        augmentation_data,
        axis_order=[0, 1, 3, 2],
    )
    restored = _restore_normal_pooled_augmentation_data(augmented, data, flat_valid_mask)

    assert torch.equal(restored["image"], data["image"].permute(0, 1, 3, 2).contiguous())
    assert torch.equal(restored["surface_mask"], data["surface_mask"].permute(0, 1, 3, 2).contiguous())
    assert torch.equal(restored["flat_target"], data["flat_target"])
    assert torch.equal(restored["flat_supervision"], data["flat_supervision"])
    assert torch.equal(restored["flat_valid"], data["flat_valid"])
    assert torch.equal(
        restored["flat_points_local_zyx"],
        torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]], dtype=torch.float32),
    )
    assert torch.equal(
        restored["flat_normals_local_zyx"],
        torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32),
    )
