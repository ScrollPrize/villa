from pathlib import Path

import numpy as np
import pytest
import torch

import vesuvius.neural_tracing.datasets.dataset_rowcol_cond as rowcol_dataset_module
from dataset_rowcol_cond_test_setup import (
    TEST_TIFXYZ_ROOT,
    build_rowcol_patchfinding_config,
    make_temp_test_volume_zarr,
)
from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches


REAL_SURFACE_COLLECTION = "PHerc0343p"


def _segment_wrap_ids(wrap: dict) -> tuple[int, ...]:
    seg = wrap["segment"]
    seg_path = getattr(seg, "path", None)
    if seg_path is None:
        raise AssertionError("Expected wrap segment to have a real filesystem path")
    wrap_ids = EdtSegDataset._extract_wrap_ids(Path(seg_path).name)
    if not wrap_ids:
        raise AssertionError(f"Could not parse wrap id from segment path: {seg_path}")
    return wrap_ids


def _has_adjacent_wrap_id(center_ids: tuple[int, ...], neighbor_ids: tuple[int, ...]) -> bool:
    neighbor_set = set(int(v) for v in neighbor_ids)
    for center_id in center_ids:
        if (center_id - 1) in neighbor_set or (center_id + 1) in neighbor_set:
            return True
    return False


def _load_overlap_mask_bool(mask_cache: dict[str, np.ndarray], wrap: dict) -> np.ndarray:
    seg = wrap["segment"]
    seg_path = getattr(seg, "path", None)
    if seg_path is None:
        raise AssertionError("Expected wrap segment to have a real filesystem path")
    mask_path = (Path(seg_path) / "overlap_mask.tif").resolve()
    key = str(mask_path)
    mask = mask_cache.get(key)
    if mask is None:
        loaded = rowcol_dataset_module.tifffile.imread(str(mask_path))
        loaded = np.asarray(loaded)
        if loaded.ndim > 2:
            loaded = np.squeeze(loaded)
        if loaded.ndim != 2:
            raise AssertionError(
                f"Expected 2D overlap mask for {mask_path}, got shape {tuple(loaded.shape)}"
            )
        mask = loaded > 0
        mask_cache[key] = mask
    return mask


@pytest.fixture(scope="module")
def triplet_real343p_dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rowcol_triplet_real343p")
    volume_path = make_temp_test_volume_zarr(tmp_path, shape=(128, 128, 128), scale=0)
    segments_path = TEST_TIFXYZ_ROOT / REAL_SURFACE_COLLECTION

    config = build_rowcol_patchfinding_config(
        volume_path=volume_path,
        segments_path=segments_path,
        volume_scale=0,
        require_all_valid_in_bbox=False,
        overrides={
            "use_triplet_wrap_displacement": True,
            "use_dense_displacement": True,
            "use_extrapolation": False,
            "use_other_wrap_cond": False,
            "use_sdt": False,
            "use_heatmap_targets": False,
            "use_segmentation": False,
            "validate_result_tensors": True,
            "triplet_warn_missing_overlap_masks": False,
            "triplet_close_check_enabled": False,
            "triplet_close_print": False,
            "force_recompute_patches": True,
        },
    )

    def _find_world_chunk_patches_no_cache(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["cache_dir"] = None
        return find_world_chunk_patches(*args, **kwargs)

    orig_fn = rowcol_dataset_module.find_world_chunk_patches
    rowcol_dataset_module.find_world_chunk_patches = _find_world_chunk_patches_no_cache
    try:
        ds = EdtSegDataset(config=config, apply_augmentation=False, patch_metadata=None)
    finally:
        rowcol_dataset_module.find_world_chunk_patches = orig_fn
    return ds


def test_triplet_real343p_overlap_filter_and_neighbor_lookup_invariants(triplet_real343p_dataset):
    ds = triplet_real343p_dataset

    assert len(ds.patches) > 0
    assert len(ds.sample_index) > 0

    stats = ds._triplet_overlap_filter_stats
    assert int(stats["chunks_total"]) > int(stats["chunks_kept"])
    assert int(stats["chunks_dropped_overlap"]) > 0

    mask_cache: dict[str, np.ndarray] = {}
    for patch in ds.patches:
        for wrap in patch.wraps:
            overlap_mask = _load_overlap_mask_bool(mask_cache, wrap)
            has_overlap = EdtSegDataset._wrap_bbox_has_overlap(overlap_mask, wrap["bbox_2d"])
            assert not has_overlap

    lookup = ds._triplet_neighbor_lookup
    assert len(lookup) > 0

    for (patch_idx, center_wrap_idx), triplet_meta in lookup.items():
        patch = ds.patches[int(patch_idx)]
        center_wrap = patch.wraps[int(center_wrap_idx)]
        behind_wrap = patch.wraps[int(triplet_meta["behind_wrap_idx"])]
        front_wrap = patch.wraps[int(triplet_meta["front_wrap_idx"])]

        center_ids = _segment_wrap_ids(center_wrap)
        behind_ids = _segment_wrap_ids(behind_wrap)
        front_ids = _segment_wrap_ids(front_wrap)

        assert int(triplet_meta["behind_wrap_idx"]) != int(triplet_meta["front_wrap_idx"])
        assert _has_adjacent_wrap_id(center_ids, behind_ids)
        assert _has_adjacent_wrap_id(center_ids, front_ids)


def test_triplet_real343p_generates_training_payloads_without_overlap_chunks(triplet_real343p_dataset):
    ds = triplet_real343p_dataset
    assert len(ds.sample_index) > 0

    max_samples = min(16, len(ds.sample_index))
    mask_cache: dict[str, np.ndarray] = {}

    for sample_idx in range(max_samples):
        patch_idx, _ = ds.sample_index[sample_idx]
        patch = ds.patches[int(patch_idx)]
        for wrap in patch.wraps:
            overlap_mask = _load_overlap_mask_bool(mask_cache, wrap)
            has_overlap = EdtSegDataset._wrap_bbox_has_overlap(overlap_mask, wrap["bbox_2d"])
            assert not has_overlap

        sample = ds[sample_idx]
        required_keys = {
            "vol",
            "cond",
            "dense_gt_displacement",
            "dense_loss_weight",
            "triplet_channel_order",
        }
        assert required_keys.issubset(sample.keys())
        if ds.use_triplet_direction_priors:
            assert "dir_priors" in sample

        vol = sample["vol"]
        cond = sample["cond"]
        dense_gt = sample["dense_gt_displacement"]
        dense_weight = sample["dense_loss_weight"]
        channel_order = sample["triplet_channel_order"]

        assert isinstance(vol, torch.Tensor)
        assert isinstance(cond, torch.Tensor)
        assert isinstance(dense_gt, torch.Tensor)
        assert isinstance(dense_weight, torch.Tensor)
        assert isinstance(channel_order, torch.Tensor)

        expected_shape = tuple(int(v) for v in ds.crop_size)
        assert tuple(vol.shape) == expected_shape
        assert tuple(cond.shape) == expected_shape
        assert tuple(dense_gt.shape) == (6, *expected_shape)
        assert tuple(dense_weight.shape) == (1, *expected_shape)
        assert tuple(channel_order.shape) == (2,)

        assert vol.dtype == torch.float32
        assert cond.dtype == torch.float32
        assert dense_gt.dtype == torch.float32
        assert dense_weight.dtype == torch.float32
        assert channel_order.dtype == torch.int64

        assert bool(torch.isfinite(vol).all())
        assert bool(torch.isfinite(cond).all())
        assert bool(torch.isfinite(dense_gt).all())
        assert bool(torch.isfinite(dense_weight).all())
        assert bool(torch.isfinite(channel_order).all())
        assert float(cond.sum().item()) > 0.0
        assert float(dense_weight.sum().item()) > 0.0
        assert set(int(v) for v in channel_order.tolist()) == {0, 1}
