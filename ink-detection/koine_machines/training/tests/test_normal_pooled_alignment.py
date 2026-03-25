import numpy as np
import torch

from koine_machines.data.ink_dataset import _build_normal_pooled_flat_metadata
from koine_machines.training.normal_pooling import pool_logits_along_normals


class StubPatchTifxyz:
    def __init__(self, normals_xyz):
        self._normals_xyz = np.asarray(normals_xyz, dtype=np.float32)

    def get_normals(self, row_start, row_end, col_start, col_end):
        del row_start, row_end, col_start, col_end
        nx = self._normals_xyz[..., 0]
        ny = self._normals_xyz[..., 1]
        nz = self._normals_xyz[..., 2]
        return nx, ny, nz


def test_normal_pooled_targets_stay_aligned_with_crop_local_3d_points():
    crop_bbox = (10, 20, 30, 13, 23, 35)
    support_patch_zyxs = np.array(
        [
            [[10.0, 20.0, 31.0], [10.0, 20.0, 34.0]],
            [[11.0, 21.0, 32.0], [12.0, 22.0, 33.0]],
        ],
        dtype=np.float32,
    )
    support_valid = np.array(
        [
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )
    normals_xyz = np.zeros((2, 2, 3), dtype=np.float32)
    normals_xyz[..., 2] = 1.0

    metadata = _build_normal_pooled_flat_metadata(
        patch_tifxyz=StubPatchTifxyz(normals_xyz),
        support_bbox=(100, 102, 200, 202),
        support_patch_zyxs=support_patch_zyxs,
        support_valid=support_valid,
        support_inklabels_flat_patch=np.array([[0, 1], [1, 1]], dtype=np.uint8),
        support_supervision_flat_patch=np.array([[1, 1], [1, 0]], dtype=np.uint8),
        crop_bbox=crop_bbox,
    )

    logits = torch.zeros((1, 1, 3, 3, 5), dtype=torch.float32)
    logits[0, 0, 0, 0, 1] = 2.0
    logits[0, 0, 1, 1, 2] = 5.0
    logits[0, 0, 2, 2, 3] = 7.0

    pooled_logits, pooled_valid = pool_logits_along_normals(
        logits,
        metadata["flat_points_local_zyx"].unsqueeze(0),
        metadata["flat_normals_local_zyx"].unsqueeze(0),
        metadata["flat_valid"].unsqueeze(0),
        neg_dist=0.0,
        pos_dist=0.0,
        sample_step=1.0,
        align_corners=True,
    )

    assert torch.equal(
        metadata["flat_points_local_zyx"],
        torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 1.0, 2.0], [2.0, 2.0, 3.0]],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        pooled_logits,
        torch.tensor([[[[2.0, 0.0], [5.0, 7.0]]]], dtype=torch.float32),
    )
    assert torch.equal(
        pooled_valid,
        torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32),
    )

    ignore_mask = (
        (metadata["flat_supervision"].unsqueeze(0) <= 0)
        | (metadata["flat_valid"].unsqueeze(0) <= 0)
        | (pooled_valid <= 0)
    ).float()
    assert torch.equal(
        metadata["flat_target"].unsqueeze(0),
        torch.tensor([[[[0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32),
    )
    assert torch.equal(
        ignore_mask,
        torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], dtype=torch.float32),
    )
