"""Regression tests for label image shape boundaries in create_label_zarrs.

``_normalize_to_2d`` and ``_normalized_2d_shape`` removed every 1-sized
dimension, so a single-row or single-column label TIFF was rejected as
non-2D and a one-row RGB image's height was dropped. The fix squeezes only
singleton axes the TIFF series axes codes mark as non-spatial, so legacy
leading-singleton files keep their interpretation and degenerate spatial
images stay 2D.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from vesuvius.ink_detection.preprocessing.create_label_zarrs import (
    convert_image,
    load_image,
)


def _level_zero(zarr_path: Path) -> zarr.Array:
    group = zarr.open_group(str(zarr_path), mode="r")
    arrays = [
        group[key] for key in sorted(group.array_keys(), key=lambda k: int(k))
    ]
    return arrays[0]


def test_single_row_label_tiff_converts(tmp_path):
    label_path = tmp_path / "segment-a_ink.tif"
    image = np.array([[1, 0, 1, 1]], dtype=np.uint8)
    tifffile.imwrite(label_path, image)

    result = convert_image(label_path, levels=1)

    assert result["status"] == "written"
    level0 = _level_zero(label_path.with_suffix(".zarr"))
    assert level0.shape == (65, 1, 4)
    assert np.array_equal(level0[32], image)


def test_single_column_label_tiff_converts(tmp_path):
    label_path = tmp_path / "segment-b_ink.tif"
    image = np.array([[1], [0], [1]], dtype=np.uint8)
    tifffile.imwrite(label_path, image)

    result = convert_image(label_path, levels=1)

    assert result["status"] == "written"
    level0 = _level_zero(label_path.with_suffix(".zarr"))
    assert level0.shape == (65, 3, 1)
    assert np.array_equal(level0[32], image)


def test_single_pixel_label_tiff_converts(tmp_path):
    label_path = tmp_path / "segment-c_ink.tif"
    image = np.array([[5]], dtype=np.uint8)
    tifffile.imwrite(label_path, image)

    result = convert_image(label_path, levels=1)

    assert result["status"] == "written"
    level0 = _level_zero(label_path.with_suffix(".zarr"))
    assert level0.shape == (65, 1, 1)
    assert level0[32, 0, 0] == 5


def test_one_row_rgb_label_tiff_keeps_height(tmp_path):
    label_path = tmp_path / "segment-f_ink.tif"
    image = np.zeros((1, 5, 3), dtype=np.uint8)
    image[0, 0, 0] = 1
    image[0, 3, 0] = 2
    tifffile.imwrite(label_path, image)

    loaded = load_image(label_path)
    assert loaded.shape == (1, 5)
    assert np.array_equal(loaded, image[..., 0])

    result = convert_image(label_path, levels=1)
    assert result["status"] == "written"
    level0 = _level_zero(label_path.with_suffix(".zarr"))
    assert level0.shape == (65, 1, 5)
    assert np.array_equal(level0[32], image[..., 0])


def test_one_column_rgb_label_tiff_keeps_width(tmp_path):
    label_path = tmp_path / "segment-g_ink.tif"
    image = np.zeros((4, 1, 3), dtype=np.uint8)
    image[0, 0, 0] = 1
    image[2, 0, 0] = 2
    tifffile.imwrite(label_path, image)

    np.testing.assert_array_equal(load_image(label_path), image[..., 0])


def test_one_row_rgb_label_png_keeps_height(tmp_path):
    import cv2

    label_path = tmp_path / "segment-h_ink.png"
    image = np.zeros((1, 5, 3), dtype=np.uint8)
    image[0, 2, 0] = 1
    assert cv2.imwrite(str(label_path), image)

    loaded = load_image(label_path)

    assert loaded.shape == (1, 5)
    assert np.array_equal(loaded, image[..., 0])


def test_one_column_rgb_label_png_keeps_width(tmp_path):
    import cv2

    label_path = tmp_path / "segment-i_ink.png"
    image = np.zeros((4, 1, 3), dtype=np.uint8)
    image[1, 0, 0] = 1
    assert cv2.imwrite(str(label_path), image)

    loaded = load_image(label_path)

    assert loaded.shape == (4, 1)
    assert np.array_equal(loaded, image[..., 0])


def test_multi_sample_tiff_streams_first_sample(tmp_path):
    label_path = tmp_path / "segment-d_ink.tif"
    image = np.zeros((3, 2, 3), dtype=np.uint8)
    image[0, 1, 0] = 1
    image[2, 0, 0] = 1
    tifffile.imwrite(label_path, image)

    result = convert_image(label_path, levels=1)

    assert result["status"] == "written"
    level0 = _level_zero(label_path.with_suffix(".zarr"))
    assert level0.shape == (65, 3, 2)
    assert np.array_equal(level0[32], image[..., 0])


def test_multi_sample_tiff_load_image_first_channel(tmp_path):
    label_path = tmp_path / "segment-e_ink.tif"
    image = np.zeros((3, 2, 3), dtype=np.uint8)
    image[1, 0, 0] = 1
    tifffile.imwrite(label_path, image)

    loaded = load_image(label_path)

    assert loaded.shape == (3, 2)
    assert np.array_equal(loaded, image[..., 0])


@pytest.mark.parametrize("shape", [(2, 3, 4, 5), (3, 4, 5, 6)])
def test_non_flat_label_image_is_rejected(tmp_path, shape):
    source = tmp_path / "label.tif"
    tifffile.imwrite(source, np.zeros(shape, dtype=np.uint8), photometric="minisblack")
    with pytest.raises(ValueError, match="Expected a 2D image"):
        load_image(source)
