import numpy as np
from scipy.ndimage import distance_transform_edt

from vesuvius.models.datasets.zarr_dataset import ZarrDataset


def _reference_dilate(values: np.ndarray, distance: float, ignore_label=None) -> np.ndarray:
    arr = np.array(values, copy=True)
    ignore_mask = np.zeros(arr.shape, dtype=bool) if ignore_label is None else (arr == ignore_label)
    source_mask = (arr != 0) & ~ignore_mask
    fill_mask = arr == 0
    if not np.any(source_mask) or not np.any(fill_mask):
        return arr

    distances, nearest_indices = distance_transform_edt(
        ~source_mask,
        return_indices=True,
    )
    fill_mask &= distances <= float(distance)
    arr[fill_mask] = arr[tuple(nearest_indices[axis][fill_mask] for axis in range(arr.ndim))]
    return arr


def test_roi_dilate_matches_full_patch_result():
    values = np.zeros((12, 12, 12), dtype=np.float32)
    values[4:6, 4:6, 4:6] = 1.0
    values[5, 5, 7] = 2.0

    expected = _reference_dilate(values, distance=2.0)
    result, roi_slices = ZarrDataset._dilate_label_patch(
        values,
        distance=2.0,
        ignore_label=None,
        original_shape=values.shape,
    )

    np.testing.assert_array_equal(result, expected)
    assert roi_slices is not None


def test_binary_fast_path_matches_full_patch_result():
    values = np.zeros((16, 16, 16), dtype=np.float32)
    values[6:9, 6:9, 6:9] = 1.0
    values[7, 7, 10] = 2.0

    expected = _reference_dilate(values, distance=2.0, ignore_label=2.0)
    result, roi_slices = ZarrDataset._dilate_label_patch(
        values,
        distance=2.0,
        ignore_label=2.0,
        original_shape=values.shape,
        binary_fast_path=True,
    )

    np.testing.assert_array_equal(result, expected)
    assert roi_slices is not None
