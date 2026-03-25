import numpy as np
from scipy.ndimage import distance_transform_edt

from koine_machines.augmentation.translation import (
    count_points_within_crop,
    maybe_translate_normal_pooled_crop_bbox,
)
from koine_machines.data.ink_dataset import (
    _project_flat_patch_to_native_crop,
    _project_valid_surface_mask_to_native_crop,
    _read_flat_surface_patch,
    _select_flat_pixels_for_native_crop,
    _select_flat_pixels_for_native_crop_via_stored_resolution,
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
    def __init__(self, stored_zyxs, *, full_zyxs=None, full_resolution_shape=None):
        self._stored_zyxs = np.asarray(stored_zyxs, dtype=np.float32)
        self._full_zyxs = (
            self._stored_zyxs
            if full_zyxs is None
            else np.asarray(full_zyxs, dtype=np.float32)
        )
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
