import torch

from koine_machines.training.normal_pooling import (
    build_sample_offsets,
    collate_normal_pooled_batch,
    local_points_zyx_to_normalized_grid,
    pool_logits_along_normals,
)


def test_build_sample_offsets_includes_both_endpoints():
    offsets = build_sample_offsets(2.0, 2.0, 1.0, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(offsets, torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]))


def test_local_points_zyx_to_normalized_grid_maps_voxel_centers():
    points = torch.tensor(
        [
            [[[[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]]]],
        ],
        dtype=torch.float32,
    )
    grid = local_points_zyx_to_normalized_grid(points, (3, 5, 7))
    expected = torch.tensor(
        [
            [[[[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]]]],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(grid, expected)


def test_pool_logits_along_normals_max_pools_along_sample_line():
    logits = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
    logits[0, 0, 1, 1, 0] = 2.0
    logits[0, 0, 1, 1, 1] = 4.0
    logits[0, 0, 1, 1, 2] = 7.0

    flat_points = torch.tensor([[[[1.0, 1.0, 1.0]]]], dtype=torch.float32)
    flat_normals = torch.tensor([[[[0.0, 0.0, 1.0]]]], dtype=torch.float32)
    flat_valid = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    pooled_logits, pooled_valid = pool_logits_along_normals(
        logits,
        flat_points,
        flat_normals,
        flat_valid,
        neg_dist=1.0,
        pos_dist=1.0,
        sample_step=1.0,
    )

    assert pooled_logits.shape == (1, 1, 1, 1)
    assert pooled_valid.shape == (1, 1, 1, 1)
    assert torch.allclose(pooled_logits, torch.tensor([[[[7.0]]]]))
    assert torch.equal(pooled_valid, torch.ones_like(pooled_valid))


def test_pool_logits_along_normals_marks_all_oob_pixels_invalid():
    logits = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32)
    flat_points = torch.tensor([[[[-2.0, 0.0, 0.0]]]], dtype=torch.float32)
    flat_normals = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    flat_valid = torch.ones((1, 1, 1, 1), dtype=torch.float32)

    pooled_logits, pooled_valid = pool_logits_along_normals(
        logits,
        flat_points,
        flat_normals,
        flat_valid,
        neg_dist=0.0,
        pos_dist=0.0,
        sample_step=1.0,
    )

    assert torch.equal(pooled_logits, torch.zeros_like(pooled_logits))
    assert torch.equal(pooled_valid, torch.zeros_like(pooled_valid))


def test_collate_normal_pooled_batch_pads_to_batch_maximum():
    sample_a = {
        "image": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "surface_mask": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "flat_target": torch.ones((1, 2, 3), dtype=torch.float32),
        "flat_supervision": torch.ones((1, 2, 3), dtype=torch.float32),
        "flat_valid": torch.ones((1, 2, 3), dtype=torch.float32),
        "flat_points_local_zyx": torch.ones((2, 3, 3), dtype=torch.float32),
        "flat_normals_local_zyx": torch.ones((2, 3, 3), dtype=torch.float32),
    }
    sample_b = {
        "image": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "surface_mask": torch.zeros((1, 4, 4, 4), dtype=torch.float32),
        "flat_target": torch.ones((1, 4, 2), dtype=torch.float32),
        "flat_supervision": torch.ones((1, 4, 2), dtype=torch.float32),
        "flat_valid": torch.ones((1, 4, 2), dtype=torch.float32),
        "flat_points_local_zyx": torch.ones((4, 2, 3), dtype=torch.float32),
        "flat_normals_local_zyx": torch.ones((4, 2, 3), dtype=torch.float32),
    }

    batch = collate_normal_pooled_batch([sample_a, sample_b])

    assert batch["flat_target"].shape == (2, 1, 4, 3)
    assert batch["flat_supervision"].shape == (2, 1, 4, 3)
    assert batch["flat_valid"].shape == (2, 1, 4, 3)
    assert batch["flat_points_local_zyx"].shape == (2, 4, 3, 3)
    assert batch["flat_normals_local_zyx"].shape == (2, 4, 3, 3)
    assert torch.equal(batch["flat_valid"][0, 0, 2:, :], torch.zeros((2, 3)))
    assert torch.equal(batch["flat_valid"][1, 0, :, 2:], torch.zeros((4, 1)))
