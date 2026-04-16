from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt
import torch

from koine_machines.augmentation.translation import (
    count_points_within_crop,
    maybe_translate_normal_pooled_crop_bbox,
)
import koine_machines.data.ink_dataset as ink_dataset_module
from koine_machines.data.ink_dataset import (
    InkDataset,
    _is_native_3d_mode,
    _maybe_select_flat_pixels_for_native_crop_via_stored_resolution,
    _normalize_image_crop,
    _project_flat_labels_and_supervision_to_native_crop,
    _project_flat_patch_to_native_crop,
    _project_valid_surface_mask_to_native_crop,
    _read_flat_surface_patch,
    _select_flat_pixels_for_native_crop,
    _select_flat_pixels_for_native_crop_via_stored_resolution,
    _uses_surface_mask_channel,
)


class StubRng:
    def __init__(self, *, random_values=(), randint_values=(), shuffle_orders=()):
        self._random_values = list(random_values)
        self._randint_values = list(randint_values)
        self._shuffle_orders = [tuple(order) for order in shuffle_orders]

    def random(self):
        if not self._random_values:
            raise AssertionError("StubRng.random was called more times than expected")
        return self._random_values.pop(0)

    def randint(self, low, high=None):
        if high is None:
            high = low
            low = 0
        if not self._randint_values:
            raise AssertionError("StubRng.randint was called more times than expected")
        value = int(self._randint_values.pop(0))
        assert low <= value < high, (low, high, value)
        return value

    def shuffle(self, values):
        if not self._shuffle_orders:
            return
        order = self._shuffle_orders.pop(0)
        values[:] = np.asarray(order, dtype=values.dtype)


class StubTifxyz:
    def __init__(self, stored_zyxs, *, full_zyxs=None, full_resolution_shape=None, normals_xyz=None):
        self._stored_zyxs = np.asarray(stored_zyxs, dtype=np.float32)
        self._full_zyxs = (
            self._stored_zyxs
            if full_zyxs is None
            else np.asarray(full_zyxs, dtype=np.float32)
        )
        self._normals_xyz = (
            np.zeros((*self._full_zyxs.shape[:2], 3), dtype=np.float32)
            if normals_xyz is None
            else np.asarray(normals_xyz, dtype=np.float32)
        )
        if normals_xyz is None:
            self._normals_xyz[..., 2] = 1.0
        self.shape = self._stored_zyxs.shape[:2]
        self.full_resolution_shape = (
            tuple(int(v) for v in full_resolution_shape)
            if full_resolution_shape is not None
            else self._full_zyxs.shape[:2]
        )

    def get_zyxs(self, *, stored_resolution=None):
        assert stored_resolution is True
        return self._stored_zyxs

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(f"Expected (rows, cols) key, got {key!r}")
        rows, cols = key
        patch = self._full_zyxs[rows, cols]
        return patch[..., 2], patch[..., 1], patch[..., 0], np.isfinite(patch).all(axis=-1) & (patch >= 0).all(axis=-1)

    def get_normals(self, row_start, row_end, col_start, col_end):
        normals = self._normals_xyz[row_start:row_end, col_start:col_end]
        return normals[..., 0], normals[..., 1], normals[..., 2]


def test_normalize_image_crop_robust_percentile_span_scales_by_clipped_percentile_span():
    image = np.array([0, 19, 42, 161, 255], dtype=np.uint8)

    normalized = _normalize_image_crop(
        image,
        {
            "image_normalization": {
                "mode": "robust_percentile_span",
                "percentile_lower": 0,
                "percentile_upper": 100,
            }
        },
    )

    expected = (image.astype(np.float32) - 42.0) / ((255.0 - 0.0) * 0.5)
    np.testing.assert_allclose(normalized, expected, rtol=1e-6, atol=1e-6)


def test_normalize_image_crop_rejects_invalid_mode():
    with pytest.raises(ValueError, match="Unsupported image_normalization mode"):
        _normalize_image_crop(np.array([1, 2, 3], dtype=np.uint8), {"image_normalization": "wat"})


def test_read_flat_surface_patch_uses_middle_slice_with_padding():
    volume = np.zeros((5, 4, 4), dtype=np.uint8)
    volume[2, 1:3, 1:4] = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.uint8,
    )

    patch = _read_flat_surface_patch(volume, y0=0, y1=3, x0=2, x1=5)

    assert patch.shape == (3, 3)
    np.testing.assert_array_equal(
        patch,
        np.array(
            [
                [0, 0, 0],
                [2, 3, 0],
                [5, 6, 0],
            ],
            dtype=np.uint8,
        ),
    )


def test_project_flat_patch_to_native_crop_scatter_positives():
    flat_patch = np.array(
        [
            [0, 7, 0],
            [3, 0, 5],
        ],
        dtype=np.uint8,
    )
    patch_zyxs = np.array(
        [
            [[10, 20, 30], [10, 21, 31], [10, 22, 32]],
            [[11, 20, 30], [11, 21, 31], [11, 22, 32]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [True, True, False],
            [True, False, True],
        ]
    )

    projected = _project_flat_patch_to_native_crop(
        flat_patch,
        patch_zyxs,
        valid_mask,
        crop_bbox=(10, 20, 30, 13, 24, 35),
    )

    expected = np.zeros((3, 4, 5), dtype=np.uint8)
    expected[0, 1, 1] = 7
    expected[1, 0, 0] = 3
    expected[1, 2, 2] = 5
    np.testing.assert_array_equal(projected, expected)


def test_project_flat_labels_and_supervision_to_native_crop_adds_normal_thickness():
    support_patch_zyxs = np.array(
        [
            [[2, 1, 1], [2, 1, 2]],
        ],
        dtype=np.float32,
    )
    support_normals_local_zyx = np.array(
        [
            [[1, 0, 0], [1, 0, 0]],
        ],
        dtype=np.float32,
    )
    support_valid = np.ones((1, 2), dtype=bool)
    support_inklabels_flat_patch = np.array([[1, 0]], dtype=np.uint8)
    support_supervision_flat_patch = np.array([[1, 1]], dtype=np.uint8)

    inklabels_crop, supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
        support_patch_zyxs=support_patch_zyxs,
        support_valid=support_valid,
        support_inklabels_flat_patch=support_inklabels_flat_patch,
        support_supervision_flat_patch=support_supervision_flat_patch,
        crop_bbox=(0, 0, 0, 5, 3, 4),
        support_normals_local_zyx=support_normals_local_zyx,
        label_projection_half_thickness=1.0,
        background_projection_half_thickness=1.0,
    )

    expected_ink = np.zeros((5, 3, 4), dtype=np.float32)
    expected_ink[1:4, 1, 1] = 1.0
    expected_supervision = np.zeros((5, 3, 4), dtype=np.float32)
    expected_supervision[1:4, 1, 1] = 1.0
    expected_supervision[1:4, 1, 2] = 1.0
    np.testing.assert_array_equal(inklabels_crop, expected_ink)
    np.testing.assert_array_equal(supervision_crop, expected_supervision)


def test_project_flat_labels_and_supervision_to_native_crop_interpolates_between_normal_positions():
    support_patch_zyxs = np.array(
        [
            [[2, 1, 1], [2, 1, 4]],
        ],
        dtype=np.float32,
    )
    support_normals_local_zyx = np.array(
        [
            [[1, 0, 0], [1, 0, 0]],
        ],
        dtype=np.float32,
    )
    support_valid = np.ones((1, 2), dtype=bool)
    support_inklabels_flat_patch = np.array([[1, 1]], dtype=np.uint8)
    support_supervision_flat_patch = np.array([[1, 1]], dtype=np.uint8)

    inklabels_crop, supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
        support_patch_zyxs=support_patch_zyxs,
        support_valid=support_valid,
        support_inklabels_flat_patch=support_inklabels_flat_patch,
        support_supervision_flat_patch=support_supervision_flat_patch,
        crop_bbox=(0, 0, 0, 5, 3, 6),
        support_normals_local_zyx=support_normals_local_zyx,
        label_projection_half_thickness=1.0,
        background_projection_half_thickness=1.0,
    )

    expected = np.zeros((5, 3, 6), dtype=np.float32)
    expected[1:4, 1, 1:5] = 1.0
    np.testing.assert_array_equal(inklabels_crop, expected)
    np.testing.assert_array_equal(supervision_crop, expected)


def test_project_flat_labels_and_supervision_to_native_crop_fills_interpolated_normal_cells():
    support_patch_zyxs = np.array(
        [
            [[2, 1, 1], [2, 1, 4]],
            [[2, 4, 1], [2, 4, 4]],
        ],
        dtype=np.float32,
    )
    support_normals_local_zyx = np.zeros((2, 2, 3), dtype=np.float32)
    support_normals_local_zyx[..., 0] = 1.0
    support_valid = np.ones((2, 2), dtype=bool)
    support_inklabels_flat_patch = np.ones((2, 2), dtype=np.uint8)
    support_supervision_flat_patch = np.ones((2, 2), dtype=np.uint8)

    inklabels_crop, supervision_crop = _project_flat_labels_and_supervision_to_native_crop(
        support_patch_zyxs=support_patch_zyxs,
        support_valid=support_valid,
        support_inklabels_flat_patch=support_inklabels_flat_patch,
        support_supervision_flat_patch=support_supervision_flat_patch,
        crop_bbox=(0, 0, 0, 5, 6, 6),
        support_normals_local_zyx=support_normals_local_zyx,
        label_projection_half_thickness=1.0,
        background_projection_half_thickness=1.0,
    )

    expected = np.zeros((5, 6, 6), dtype=np.float32)
    expected[1:4, 1:5, 1:5] = 1.0
    np.testing.assert_array_equal(inklabels_crop, expected)
    np.testing.assert_array_equal(supervision_crop, expected)


def test_project_valid_surface_mask_to_native_crop_marks_all_valid_points():
    patch_zyxs = np.array(
        [
            [[10, 20, 30], [10, 21, 31], [10, 22, 32]],
            [[11, 20, 30], [11, 21, 31], [11, 22, 32]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array(
        [
            [True, True, False],
            [True, False, True],
        ]
    )

    projected = _project_valid_surface_mask_to_native_crop(
        patch_zyxs,
        valid_mask,
        crop_bbox=(10, 20, 30, 13, 24, 35),
    )

    occupancy = np.zeros((3, 4, 5), dtype=bool)
    occupancy[0, 0, 0] = True
    occupancy[0, 1, 1] = True
    occupancy[1, 0, 0] = True
    occupancy[1, 2, 2] = True
    expected = np.clip(1.0 - (distance_transform_edt(~occupancy) / 10.0), 0.0, 1.0).astype(np.float32)
    np.testing.assert_array_equal(projected, expected)


def test_full_3d_mode_names_surface_mask_contract():
    assert _is_native_3d_mode("full_3d")
    assert _is_native_3d_mode("full_3d_single_wrap")
    assert not _uses_surface_mask_channel("full_3d")
    assert _uses_surface_mask_channel("full_3d_single_wrap")


def test_gather_segments_preserves_remote_native_volume_path(tmp_path):
    segment_dir = tmp_path / "segment-a"
    segment_dir.mkdir()
    (segment_dir / "x.tif").touch()
    (segment_dir / "segment-a_inklabels.zarr").mkdir()
    (segment_dir / "segment-a_supervision_mask.zarr").mkdir()
    volume_path = "s3://vesuvius-challenge-open-data/example-volume.zarr/"

    dataset = object.__new__(InkDataset)
    dataset.config = {"patch_size": [8, 8, 8]}
    dataset.datasets = [
        {
            "volume_path": volume_path,
            "volume_scale": 0,
            "segments_path": str(tmp_path),
        }
    ]
    dataset.mode = "full_3d"
    dataset.discovery_mode = "labeled"
    dataset.debug = False

    segments = list(dataset._gather_segments())

    assert len(segments) == 1
    assert segments[0].image_volume == volume_path


def test_patch_finding_worker_error_names_segment(monkeypatch, tmp_path):
    class FailingSegment:
        dataset_idx = 2
        segment_relpath = "segment-b"
        segment_name = "segment-b"
        segment_dir = tmp_path / "segment-b"
        image_volume = "s3://bucket/example-volume.zarr/"
        scale = 0
        supervision_mask = tmp_path / "segment-b_supervision_mask.zarr"
        inklabels = tmp_path / "segment-b_inklabels.zarr"
        validation_mask = None
        cache_key = (
            dataset_idx,
            segment_relpath,
            scale,
            str(inklabels),
            str(supervision_mask),
            "",
        )

        def _find_patches(self):
            raise RuntimeError("missing resolution")

    monkeypatch.setattr(
        InkDataset,
        "_gather_segments",
        lambda self: [FailingSegment()],
    )

    with pytest.raises(RuntimeError) as exc_info:
        InkDataset(
            {
                "patch_size": [8, 8, 8],
                "datasets": [{"segments_path": str(tmp_path)}],
                "dataloader_workers": 1,
                "out_dir": str(tmp_path),
            },
            do_augmentations=False,
        )

    message = str(exc_info.value)
    assert "dataset_idx=2" in message
    assert "segment='segment-b'" in message
    assert "image_volume='s3://bucket/example-volume.zarr/'" in message
    assert "missing resolution" in str(exc_info.value.__cause__)


def test_full_3d_merge_intersecting_segments_inserts_supervised_background():
    base_segment = SimpleNamespace(
        dataset_idx=0,
        image_volume="volume-a",
        scale=0,
        segment_relpath="base",
        segment_dir="base",
        inklabels="base_ink",
        supervision_mask="base_supervision",
        validation_mask=None,
    )
    other_segment = SimpleNamespace(
        dataset_idx=0,
        image_volume="volume-a",
        scale=0,
        segment_relpath="other",
        segment_dir="other",
        inklabels="other_ink",
        supervision_mask="other_supervision",
        validation_mask=None,
    )
    unrelated_segment = SimpleNamespace(
        dataset_idx=1,
        image_volume="volume-b",
        scale=0,
        segment_relpath="unrelated",
        segment_dir="unrelated",
        inklabels="unrelated_ink",
        supervision_mask="unrelated_supervision",
        validation_mask=None,
    )

    rows = np.arange(3, dtype=np.float32)[:, None]
    cols = np.arange(4, dtype=np.float32)[None, :]
    other_zyxs = np.stack(
        [
            np.full((3, 4), 5.0, dtype=np.float32),
            (10.0 + rows).repeat(4, axis=1),
            (20.0 + cols).repeat(3, axis=0),
        ],
        axis=-1,
    )
    tifxyz_by_dir = {
        "other": StubTifxyz(other_zyxs),
        "unrelated": StubTifxyz(other_zyxs),
    }

    other_supervision = np.zeros((3, 3, 4), dtype=np.uint8)
    other_supervision[1, 1, 1] = 1
    other_supervision[1, 2, 2] = 1
    other_ink = np.zeros((3, 3, 4), dtype=np.uint8)
    other_ink[1, 2, 2] = 1
    arrays_by_path = {
        "other_supervision": other_supervision,
        "other_ink": other_ink,
    }

    dataset = object.__new__(InkDataset)
    dataset._register_segments([base_segment, other_segment, unrelated_segment])
    dataset._get_cached_tifxyz = lambda segment_dir: tifxyz_by_dir[str(segment_dir)]
    dataset._get_cached_stored_resolution_zyxs = lambda segment_dir, *, patch_tifxyz=None: (
        patch_tifxyz.get_zyxs(stored_resolution=True),
        np.ones(patch_tifxyz.shape, dtype=bool),
    )
    dataset._get_cached_zarr = lambda path, *, resolution: arrays_by_path[str(path)]

    inklabels_crop, supervision_crop = dataset._merge_intersecting_segment_labels_into_crop(
        patch=SimpleNamespace(segment=base_segment, is_validation=False),
        crop_bbox=(5, 10, 20, 6, 13, 24),
        inklabels_crop=np.zeros((1, 3, 4), dtype=np.float32),
        supervision_crop=np.zeros((1, 3, 4), dtype=np.float32),
    )

    expected_supervision = np.zeros((1, 3, 4), dtype=np.float32)
    expected_supervision[0, 1, 1] = 1.0
    expected_supervision[0, 2, 2] = 1.0
    expected_ink = np.zeros((1, 3, 4), dtype=np.float32)
    expected_ink[0, 2, 2] = 1.0
    np.testing.assert_array_equal(supervision_crop, expected_supervision)
    np.testing.assert_array_equal(inklabels_crop, expected_ink)


def test_getitem_resamples_empty_native_patch_before_using_result(monkeypatch):
    dataset = object.__new__(InkDataset)
    dataset.config = {"seed": 0}
    dataset.patch_size = (1, 2, 2)
    dataset.mode = "normal_pooled_3d"
    dataset.do_augmentations = False
    dataset.augmentations = None
    dataset.unlabeled_segments = []

    empty_segment = SimpleNamespace(scale=0, validation_mask=None, segment_name="empty")
    valid_segment = SimpleNamespace(scale=0, validation_mask=None, segment_name="valid")
    dataset.patches = [
        SimpleNamespace(
            bbox=(0, 0, 0, 1, 2, 2),
            image_volume="empty_image",
            supervision_mask="empty_supervision",
            inklabels="empty_ink",
            segment=empty_segment,
            segment_dir="empty",
            is_validation=False,
            is_unlabeled=False,
        ),
        SimpleNamespace(
            bbox=(0, 0, 0, 1, 2, 2),
            image_volume="valid_image",
            supervision_mask="valid_supervision",
            inklabels="valid_ink",
            segment=valid_segment,
            segment_dir="valid",
            is_validation=False,
            is_unlabeled=False,
        ),
    ]

    image = np.arange(4, dtype=np.float32).reshape(1, 2, 2)
    supervision = np.ones((1, 2, 2), dtype=np.uint8)
    ink = np.zeros((1, 2, 2), dtype=np.uint8)
    arrays_by_path = {
        "empty_image": image,
        "empty_supervision": supervision,
        "empty_ink": ink,
        "valid_image": image,
        "valid_supervision": supervision,
        "valid_ink": ink,
    }
    dataset._get_cached_zarr = lambda path, *, resolution: arrays_by_path[path]

    empty_zyxs = np.full((2, 2, 3), np.nan, dtype=np.float32)
    valid_zyxs = np.array(
        [
            [[0, 0, 0], [0, 0, 1]],
            [[0, 1, 0], [0, 1, 1]],
        ],
        dtype=np.float32,
    )
    tifxyz_by_dir = {
        "empty": StubTifxyz(empty_zyxs),
        "valid": StubTifxyz(valid_zyxs),
    }
    dataset._get_cached_tifxyz = lambda segment_dir: tifxyz_by_dir[str(segment_dir)]
    dataset._get_cached_stored_resolution_zyxs = lambda segment_dir, *, patch_tifxyz=None: (
        patch_tifxyz.get_zyxs(stored_resolution=True),
        np.isfinite(patch_tifxyz.get_zyxs(stored_resolution=True)).all(axis=-1),
    )
    dataset._choose_replacement_patch_index = lambda *, current_idx: 1

    support_valid = np.ones((2, 2), dtype=bool)
    trim_slices = (slice(None), slice(None))
    monkeypatch.setattr(
        ink_dataset_module,
        "_select_flat_pixels_for_native_crop_via_stored_resolution",
        lambda *args, **kwargs: ((0, 2, 0, 2), valid_zyxs, support_valid, valid_zyxs, support_valid, trim_slices),
    )
    monkeypatch.setattr(
        ink_dataset_module,
        "_filter_support_components_by_active_supervision",
        lambda **kwargs: (
            kwargs["support_bbox"],
            kwargs["support_patch_zyxs"],
            kwargs["support_valid"],
            kwargs["support_inklabels_flat_patch"],
            kwargs["support_supervision_flat_patch"],
        ),
    )
    monkeypatch.setattr(
        ink_dataset_module,
        "_slice_support_halo_for_subwindow",
        lambda support_patch_zyxs_halo, support_valid_halo, trim_slices, base_support_bbox, support_bbox: (
            support_patch_zyxs_halo,
            support_valid_halo,
            trim_slices,
        ),
    )
    monkeypatch.setattr(
        ink_dataset_module,
        "_project_valid_surface_mask_to_native_crop",
        lambda support_patch_zyxs, support_valid, crop_bbox: np.ones((1, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        ink_dataset_module,
        "_build_normal_pooled_flat_metadata",
        lambda **kwargs: {
            "flat_target": torch.zeros((1, 2, 2), dtype=torch.float32),
            "flat_supervision": torch.ones((1, 2, 2), dtype=torch.float32),
            "flat_valid": torch.ones((1, 2, 2), dtype=torch.float32),
        },
    )

    with pytest.warns(RuntimeWarning, match="no valid tifxyz points"):
        result = dataset[0]

    assert result["image"].shape == (1, 1, 2, 2)
    assert result["flat_target"].shape == (1, 2, 2)
    assert result["is_unlabeled"].item() is False


def test_select_flat_pixels_for_native_crop_returns_exact_support_window():
    support_bbox, support_patch_zyxs, support_valid = _select_flat_pixels_for_native_crop(
        patch_zyxs=np.array(
            [
                [[5, 99, 98], [5, 99, 100], [5, 99, 105]],
                [[5, 100, 99], [5, 100, 100], [5, 100, 103]],
                [[5, 101, 97], [5, 101, 101], [5, 101, 104]],
            ],
            dtype=np.float32,
        ),
        valid_mask=np.array(
            [
                [True, True, True],
                [True, True, True],
                [False, True, True],
            ],
            dtype=bool,
        ),
        crop_bbox=(5, 99, 100, 6, 102, 104),
    )

    assert support_bbox == (0, 3, 1, 3)
    np.testing.assert_array_equal(
        support_patch_zyxs,
        np.array(
            [
                [[5, 99, 100], [5, 99, 105]],
                [[5, 100, 100], [5, 100, 103]],
                [[5, 101, 101], [5, 101, 104]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        support_valid,
        np.array(
            [
                [True, False],
                [True, True],
                [True, False],
            ],
            dtype=bool,
        ),
    )


def test_select_flat_pixels_for_native_crop_via_stored_resolution_matches_exact_full_scan():
    full_h, full_w = 6, 8
    rows = np.arange(full_h, dtype=np.float32)[:, None]
    cols = np.arange(full_w, dtype=np.float32)[None, :]
    full_patch_zyxs = np.stack(
        [
            np.full((full_h, full_w), 5.0, dtype=np.float32),
            rows.repeat(full_w, axis=1),
            cols.repeat(full_h, axis=0),
        ],
        axis=-1,
    )
    full_valid = np.ones((full_h, full_w), dtype=bool)

    direct = _select_flat_pixels_for_native_crop(
        full_patch_zyxs,
        full_valid,
        crop_bbox=(5, 1, 1, 6, 5, 6),
    )

    stored_patch_zyxs = full_patch_zyxs[::2, ::2]
    via_stored = _select_flat_pixels_for_native_crop_via_stored_resolution(
        StubTifxyz(
            stored_patch_zyxs,
            full_zyxs=full_patch_zyxs,
            full_resolution_shape=(full_h, full_w),
        ),
        crop_bbox=(5, 1, 1, 6, 5, 6),
        coarse_native_pad=1,
    )

    assert via_stored[0] == direct[0]
    np.testing.assert_array_equal(via_stored[1], direct[1])
    np.testing.assert_array_equal(via_stored[2], direct[2])


def test_maybe_select_flat_pixels_for_native_crop_via_stored_resolution_returns_none_without_intersection():
    patch_zyxs = np.array(
        [
            [[5, 10, 20], [5, 10, 21]],
            [[5, 11, 20], [5, 11, 21]],
        ],
        dtype=np.float32,
    )

    selected = _maybe_select_flat_pixels_for_native_crop_via_stored_resolution(
        StubTifxyz(patch_zyxs),
        crop_bbox=(5, 100, 200, 6, 102, 202),
        coarse_native_pad=1,
    )

    assert selected is None


def test_project_flat_patch_to_native_crop_includes_points_from_expanded_flat_support():
    support_flat_patch = np.array(
        [
            [0, 0, 0, 0],
            [0, 4, 0, 6],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    support_patch_zyxs = np.array(
        [
            [[5, 99, 99], [5, 99, 100], [5, 99, 101], [5, 99, 102]],
            [[5, 100, 99], [5, 100, 100], [5, 100, 101], [5, 100, 103]],
            [[5, 101, 99], [5, 101, 100], [5, 101, 101], [5, 101, 102]],
        ],
        dtype=np.float32,
    )
    support_valid = np.ones((3, 4), dtype=bool)

    projected = _project_flat_patch_to_native_crop(
        support_flat_patch,
        support_patch_zyxs,
        support_valid,
        crop_bbox=(5, 99, 100, 6, 102, 104),
    )

    expected = np.zeros((1, 3, 4), dtype=np.uint8)
    expected[0, 1, 0] = 4
    expected[0, 1, 3] = 6
    np.testing.assert_array_equal(projected, expected)


def test_translate_normal_pooled_crop_bbox_retries_until_one_third_is_kept():
    crop_bbox = (0, 0, 0, 20, 20, 60)
    patch_zyxs = np.array(
        [
            [[5, 5, 5], [5, 5, 15], [5, 5, 25], [5, 5, 55]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, True, True, True]])
    supervision_flat_patch = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    rng = StubRng(
        random_values=[0.0, 0.9, 0.9, 0.9, 0.9],
        randint_values=[40, 25],
        shuffle_orders=[(2, 1), (2, 1)],
    )

    translated = maybe_translate_normal_pooled_crop_bbox(
        crop_bbox,
        patch_zyxs,
        valid_mask,
        supervision_flat_patch,
        rng=rng,
    )

    assert translated == (0, 0, 25, 20, 20, 85)
    assert count_points_within_crop(patch_zyxs[valid_mask].astype(np.int64), translated) == 2


def test_translate_normal_pooled_crop_bbox_supports_two_axis_shift():
    crop_bbox = (100, 200, 300, 120, 220, 340)
    patch_zyxs = np.array(
        [
            [[110, 208, 305], [111, 209, 315], [110, 208, 325]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, True, True]])
    supervision_flat_patch = np.array([[1, 1, 1]], dtype=np.uint8)
    rng = StubRng(
        random_values=[0.0, 0.0, 0.0, 0.0],
        randint_values=[10, 10],
        shuffle_orders=[(1, 2)],
    )

    translated = maybe_translate_normal_pooled_crop_bbox(
        crop_bbox,
        patch_zyxs,
        valid_mask,
        supervision_flat_patch,
        rng=rng,
    )

    assert translated == (100, 190, 290, 120, 210, 330)
    assert count_points_within_crop(patch_zyxs[valid_mask].astype(np.int64), translated) == 3


def test_translate_normal_pooled_crop_bbox_falls_back_when_it_cannot_keep_one_third():
    crop_bbox = (0, 0, 0, 20, 20, 60)
    patch_zyxs = np.array(
        [
            [[5, 5, 5], [5, 5, 15], [5, 5, 25], [5, 5, 55]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, True, True, True]])
    supervision_flat_patch = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    rng = StubRng(
        random_values=[0.0, 0.9, 0.9],
        randint_values=[40],
        shuffle_orders=[(2, 1)],
    )

    translated = maybe_translate_normal_pooled_crop_bbox(
        crop_bbox,
        patch_zyxs,
        valid_mask,
        supervision_flat_patch,
        rng=rng,
        max_attempts=1,
    )

    assert translated == crop_bbox


def test_translate_normal_pooled_crop_bbox_ignores_label_only_points():
    crop_bbox = (0, 0, 0, 20, 20, 60)
    patch_zyxs = np.array(
        [
            [[5, 5, 5], [5, 5, 15], [5, 5, 25], [5, 5, 55]],
        ],
        dtype=np.float32,
    )
    valid_mask = np.array([[True, True, True, True]])
    supervision_flat_patch = np.array([[0, 0, 0, 1]], dtype=np.uint8)
    rng = StubRng(
        random_values=[0.0, 0.9, 0.9],
        randint_values=[40],
        shuffle_orders=[(2, 1)],
    )

    translated = maybe_translate_normal_pooled_crop_bbox(
        crop_bbox,
        patch_zyxs,
        valid_mask,
        supervision_flat_patch,
        rng=rng,
        max_attempts=1,
    )

    assert translated == (0, 0, 40, 20, 20, 100)
