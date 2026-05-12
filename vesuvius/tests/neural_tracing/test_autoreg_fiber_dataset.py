from __future__ import annotations

import numpy as np
import torch
import zarr

from vesuvius.neural_tracing.autoreg_fiber.dataset import (
    AutoregFiberDataset,
    FiberSamplePlan,
    autoreg_fiber_collate,
    split_indices_by_fiber_id,
)
from vesuvius.neural_tracing.autoreg_fiber.fiber_geometry import FiberPath, write_fiber_cache
from vesuvius.neural_tracing.autoreg_fiber.serialization import serialize_fiber_example


def _write_cache(
    tmp_path,
    *,
    annotation_id: str,
    tree_id: str,
    points_zyx: np.ndarray,
    target_volume: str = "PHerc0332",
    marker: str = "fibers_s3",
):
    fiber = FiberPath(
        annotation_id=annotation_id,
        tree_id=tree_id,
        target_volume=target_volume,
        marker=marker,
        source_points_xyz=points_zyx[:, ::-1].astype(np.float32),
        points_zyx=points_zyx.astype(np.float32),
        transform_checksum=f"checksum-{annotation_id}-{tree_id}",
        densify_step=None,
    )
    return write_fiber_cache(fiber, tmp_path)


def test_serialize_fiber_example_keeps_order_and_places_stop() -> None:
    points = np.asarray(
        [
            [2.0, 3.0, 4.0],
            [3.0, 3.0, 4.0],
            [4.0, 3.0, 4.0],
            [5.0, 3.0, 4.0],
            [6.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )

    serialized = serialize_fiber_example(
        points,
        prompt_length=2,
        target_length=3,
        volume_shape=(16, 16, 16),
        patch_size=(4, 4, 4),
        offset_num_bins=(4, 4, 4),
    )

    np.testing.assert_allclose(serialized["prompt_tokens"]["xyz"], points[:2])
    np.testing.assert_allclose(serialized["target_xyz"], points[2:])
    assert serialized["target_valid_mask"].tolist() == [True, True, True]
    assert serialized["target_stop"].tolist() == [0.0, 0.0, 1.0]


def test_dataset_builds_only_valid_crop_windows_and_localizes_points(tmp_path) -> None:
    inside = np.asarray(
        [
            [8.0, 8.0, 8.0],
            [9.0, 8.0, 8.0],
            [10.0, 8.0, 8.0],
            [11.0, 8.0, 8.0],
            [12.0, 8.0, 8.0],
            [13.0, 8.0, 8.0],
        ],
        dtype=np.float32,
    )
    outside = np.asarray(
        [
            [-20.0, 4.0, 4.0],
            [-19.0, 4.0, 4.0],
            [-18.0, 4.0, 4.0],
            [-17.0, 4.0, 4.0],
            [-16.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    inside_cache = _write_cache(tmp_path, annotation_id="ann-a", tree_id="tree-a", points_zyx=inside)
    outside_cache = _write_cache(tmp_path, annotation_id="ann-b", tree_id="tree-b", points_zyx=outside)
    volume = np.arange(32 * 32 * 32, dtype=np.float32).reshape(32, 32, 32)

    dataset = AutoregFiberDataset(
        {
            "fiber_cache_paths": [str(inside_cache), str(outside_cache)],
            "crop_size": (16, 16, 16),
            "patch_size": (4, 4, 4),
            "offset_num_bins": (4, 4, 4),
            "prompt_length": 2,
            "target_length": 3,
            "point_stride": 1,
        },
        volume_array=volume,
    )

    assert len(dataset) == 2
    assert {plan.fiber_id for plan in dataset.sample_plans} == {"ann-a:tree-a"}
    sample = dataset[0]
    min_corner = sample["min_corner"].numpy()
    expected_local = inside[:5] - min_corner
    assert sample["volume"].shape == (1, 16, 16, 16)
    torch.testing.assert_close(sample["prompt_tokens"]["xyz"], torch.from_numpy(expected_local[:2]))
    torch.testing.assert_close(sample["target_xyz"], torch.from_numpy(expected_local[2:]))
    assert sample["target_stop"].tolist() == [0.0, 0.0, 1.0]
    assert sample["fiber_metadata"]["fiber_id"] == "ann-a:tree-a"


def test_mixed_volume_dataset_routes_crops_by_fiber_target_volume(tmp_path) -> None:
    paris_points = np.asarray([[8 + idx, 8.0, 8.0] for idx in range(5)], dtype=np.float32)
    s3_points = np.asarray([[10 + idx, 10.0, 10.0] for idx in range(5)], dtype=np.float32)
    paris_cache = _write_cache(
        tmp_path,
        annotation_id="ann-paris",
        tree_id="tree-paris",
        points_zyx=paris_points,
        target_volume="PHercParis4",
        marker="fibers_s1a",
    )
    s3_cache = _write_cache(
        tmp_path,
        annotation_id="ann-s3",
        tree_id="tree-s3",
        points_zyx=s3_points,
        target_volume="PHerc0332",
        marker="fibers_s3",
    )
    paris_zarr = tmp_path / "paris.zarr"
    s3_zarr = tmp_path / "s3.zarr"
    zarr.save_array(str(paris_zarr), np.full((32, 32, 32), 11, dtype=np.uint8), chunks=(16, 16, 16))
    zarr.save_array(str(s3_zarr), np.full((32, 32, 32), 22, dtype=np.uint8), chunks=(16, 16, 16))

    dataset = AutoregFiberDataset(
        {
            "fiber_cache_paths": [str(paris_cache), str(s3_cache)],
            "volumes": {
                "PHercParis4": {"volume_zarr_url": str(paris_zarr), "volume_shape": [32, 32, 32]},
                "PHerc0332": {"volume_zarr_url": str(s3_zarr), "volume_shape": [32, 32, 32]},
            },
            "crop_size": (16, 16, 16),
            "patch_size": (4, 4, 4),
            "offset_num_bins": (4, 4, 4),
            "prompt_length": 2,
            "target_length": 3,
        },
    )

    by_fiber = {dataset[idx]["fiber_metadata"]["fiber_id"]: dataset[idx] for idx in range(len(dataset))}
    assert set(by_fiber) == {"ann-paris:tree-paris", "ann-s3:tree-s3"}
    assert by_fiber["ann-paris:tree-paris"]["fiber_metadata"]["target_volume"] == "PHercParis4"
    assert by_fiber["ann-s3:tree-s3"]["fiber_metadata"]["target_volume"] == "PHerc0332"
    assert float(by_fiber["ann-paris:tree-paris"]["volume"][0, 0, 0, 0]) == 11.0
    assert float(by_fiber["ann-s3:tree-s3"]["volume"][0, 0, 0, 0]) == 22.0


def test_collate_pads_fiber_batch(tmp_path) -> None:
    first = np.asarray([[8 + idx, 8.0, 8.0] for idx in range(5)], dtype=np.float32)
    second = np.asarray([[12 + idx, 12.0, 12.0] for idx in range(6)], dtype=np.float32)
    first_cache = _write_cache(tmp_path, annotation_id="ann-a", tree_id="tree-a", points_zyx=first)
    second_cache = _write_cache(tmp_path, annotation_id="ann-b", tree_id="tree-b", points_zyx=second)
    volume = np.zeros((32, 32, 32), dtype=np.float32)
    dataset = AutoregFiberDataset(
        {
            "fiber_cache_paths": [str(first_cache), str(second_cache)],
            "crop_size": (16, 16, 16),
            "patch_size": (4, 4, 4),
            "offset_num_bins": (4, 4, 4),
            "prompt_length": 2,
            "target_length": 3,
        },
        volume_array=volume,
    )

    batch = autoreg_fiber_collate([dataset[0], dataset[1]])

    assert batch["volume"].shape == (2, 1, 16, 16, 16)
    assert batch["prompt_tokens"]["coarse_ids"].shape == (2, 2)
    assert batch["target_coarse_ids"].shape == (2, 3)
    assert batch["target_mask"].all()
    assert batch["target_stop"][:, -1].tolist() == [1.0, 1.0]
    assert len(batch["fiber_metadata"]) == 2


def test_split_indices_by_fiber_id_has_no_train_val_leakage() -> None:
    plans = [
        FiberSamplePlan(fiber_index=0, fiber_id="fiber-a", point_start=0, min_corner=(0, 0, 0)),
        FiberSamplePlan(fiber_index=0, fiber_id="fiber-a", point_start=1, min_corner=(0, 0, 0)),
        FiberSamplePlan(fiber_index=1, fiber_id="fiber-b", point_start=0, min_corner=(0, 0, 0)),
        FiberSamplePlan(fiber_index=2, fiber_id="fiber-c", point_start=0, min_corner=(0, 0, 0)),
    ]

    train_indices, val_indices = split_indices_by_fiber_id(plans, val_fraction=0.34, seed=3)

    train_ids = {plans[idx].fiber_id for idx in train_indices}
    val_ids = {plans[idx].fiber_id for idx in val_indices}
    assert train_ids
    assert val_ids
    assert train_ids.isdisjoint(val_ids)
