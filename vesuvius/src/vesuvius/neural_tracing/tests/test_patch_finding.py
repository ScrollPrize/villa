import copy

import numpy as np
import pytest

import vesuvius.neural_tracing.datasets.dataset_rowcol_cond as rowcol_dataset_module
from dataset_rowcol_cond_test_setup import (
    PATCH_SIZE,
    TEST_TIFXYZ_ROOT,
    build_rowcol_patchfinding_config,
    load_real_test_segments,
    make_temp_test_volume_zarr,
)
from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.patch_finding import find_world_chunk_patches
from vesuvius.tifxyz import Tifxyz


REAL_SURFACE_COLLECTION = "PHerc0343p"


def _run_patch_finding_with_real_surfaces(*, require_all_valid_in_bbox: bool):
    segments = load_real_test_segments(
        num_segments=3,
        collection=REAL_SURFACE_COLLECTION,
    )
    results = find_world_chunk_patches(
        segments=segments,
        target_size=PATCH_SIZE,
        overlap_fraction=0.0,
        min_span_ratio=0.1,
        edge_touch_frac=0.01,
        edge_touch_min_count=1,
        edge_touch_pad=0,
        min_points_per_wrap=8,
        bbox_pad_2d=0,
        require_all_valid_in_bbox=bool(require_all_valid_in_bbox),
        skip_chunk_if_any_invalid=False,
        inner_bbox_fraction=1.0,
        cache_dir=None,
        force_recompute=True,
        verbose=False,
        chunk_pad=0.0,
    )
    return segments, results


def _assert_chunk_result_invariants(results, *, num_segments: int):
    assert isinstance(results, list)
    assert len(results) > 0
    for chunk in results:
        chunk_id = tuple(chunk["chunk_id"])
        assert len(chunk_id) == 3
        assert all(int(v) >= 0 for v in chunk_id)

        bbox_3d = np.asarray(chunk["bbox_3d"], dtype=np.float64)
        assert bbox_3d.shape == (6,)
        assert bool(np.isfinite(bbox_3d).all())
        z_min, z_max, y_min, y_max, x_min, x_max = bbox_3d.tolist()
        assert z_min < z_max
        assert y_min < y_max
        assert x_min < x_max

        wraps = list(chunk["wraps"])
        assert int(chunk["wrap_count"]) == len(wraps)
        assert bool(chunk["has_multiple_wraps"]) == (len(wraps) > 1)
        assert len(wraps) > 0

        for wrap in wraps:
            seg_idx = int(wrap["segment_idx"])
            assert 0 <= seg_idx < num_segments
            assert int(wrap["wrap_id"]) >= 0
            bbox_2d = tuple(int(v) for v in wrap["bbox_2d"])
            assert len(bbox_2d) == 4
            r_min, r_max, c_min, c_max = bbox_2d
            assert r_min <= r_max
            assert c_min <= c_max


@pytest.fixture(scope="module")
def real_surface_dataset(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("rowcol_patchfinding")
    volume_path = make_temp_test_volume_zarr(tmp_path, shape=(16, 16, 16), scale=0)
    segments_path = TEST_TIFXYZ_ROOT / REAL_SURFACE_COLLECTION
    config = build_rowcol_patchfinding_config(
        volume_path=volume_path,
        segments_path=segments_path,
        volume_scale=0,
        require_all_valid_in_bbox=False,
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


def test_find_world_chunk_patches_returns_valid_real_surface_chunks():
    segments, results = _run_patch_finding_with_real_surfaces(
        require_all_valid_in_bbox=False,
    )
    _assert_chunk_result_invariants(results, num_segments=len(segments))
    assert any(bool(chunk["has_multiple_wraps"]) for chunk in results)


def test_find_world_chunk_patches_strict_validity_filter_reduces_or_equals_chunks():
    segments, baseline_results = _run_patch_finding_with_real_surfaces(
        require_all_valid_in_bbox=False,
    )
    _, strict_results = _run_patch_finding_with_real_surfaces(
        require_all_valid_in_bbox=True,
    )

    _assert_chunk_result_invariants(baseline_results, num_segments=len(segments))
    _assert_chunk_result_invariants(strict_results, num_segments=len(segments))

    baseline_wrap_count = sum(int(chunk["wrap_count"]) for chunk in baseline_results)
    strict_wrap_count = sum(int(chunk["wrap_count"]) for chunk in strict_results)
    assert len(strict_results) <= len(baseline_results)
    assert strict_wrap_count <= baseline_wrap_count


def test_dataset_rowcol_cond_builds_patches_from_real_surfaces_end_to_end(real_surface_dataset):
    ds = real_surface_dataset

    assert len(ds.patches) > 0
    assert str(ds.sample_mode).lower() == "wrap"

    expected_samples = sum(len(patch.wraps) for patch in ds.patches)
    assert len(ds.sample_index) == expected_samples

    for patch in ds.patches[:128]:
        assert len(patch.wraps) > 0
        bbox_3d = np.asarray(patch.world_bbox, dtype=np.float64)
        assert bbox_3d.shape == (6,)
        assert bool(np.isfinite(bbox_3d).all())
        z_min, z_max, y_min, y_max, x_min, x_max = bbox_3d.tolist()
        assert z_min < z_max
        assert y_min < y_max
        assert x_min < x_max
        for wrap in patch.wraps:
            assert isinstance(wrap["segment"], Tifxyz)

    for patch_idx, wrap_idx in ds.sample_index[:512]:
        assert 0 <= int(patch_idx) < len(ds.patches)
        assert wrap_idx is not None
        assert 0 <= int(wrap_idx) < len(ds.patches[int(patch_idx)].wraps)


def test_dataset_patch_metadata_roundtrip_consistency_for_patchfinding_outputs(real_surface_dataset):
    base_ds = real_surface_dataset
    metadata = base_ds.export_patch_metadata()
    config = copy.deepcopy(base_ds.config)

    ds_from_metadata = EdtSegDataset(
        config=config,
        apply_augmentation=False,
        patch_metadata=metadata,
    )

    assert len(ds_from_metadata.patches) == len(base_ds.patches)
    assert ds_from_metadata.sample_index == base_ds.sample_index
    assert ds_from_metadata._triplet_neighbor_lookup == base_ds._triplet_neighbor_lookup
    assert ds_from_metadata._triplet_lookup_stats == base_ds._triplet_lookup_stats
    assert ds_from_metadata._triplet_overlap_kept_indices == tuple(
        int(i) for i in metadata.get("triplet_overlap_kept_indices", tuple())
    )
