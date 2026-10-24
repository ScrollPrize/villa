"""Regression tests for grayscale mask TIFF shape boundaries.

``load_grayscale_mask`` used ``np.squeeze`` before checking dimensionality,
which collapsed a length-1 height or width axis of a valid grayscale mask:
a single-row ``(1, W)`` TIFF became ``(W,)`` and was rejected as non-2D.
The fix squeezes only singleton axes the TIFF series axes metadata marks as
non-spatial, so legacy leading-singleton masks (``(1, 8, 9)`` QYX,
``(1, 2, 3, 4)`` QYXS) keep their prior interpretation while degenerate
spatial masks stay 2D. These tests also pin first-channel semantics for
multi-sample TIFFs and top-left crop/pad.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from vesuvius.ink_detection.inference.infer import load_grayscale_mask


def test_single_row_mask_keeps_2d_shape(tmp_path):
    mask_path = tmp_path / "row_mask.tif"
    tifffile.imwrite(mask_path, np.array([[1, 0, 1]], dtype=np.uint8))

    mask = load_grayscale_mask(mask_path, (4, 3))

    assert mask.shape == (4, 3)
    assert mask.dtype == np.dtype(bool)
    assert mask[0].tolist() == [True, False, True]
    assert not mask[1:].any()


def test_single_column_mask_keeps_2d_shape(tmp_path):
    mask_path = tmp_path / "column_mask.tif"
    tifffile.imwrite(mask_path, np.array([[1], [0], [1]], dtype=np.uint8))

    mask = load_grayscale_mask(mask_path, (3, 4))

    assert mask.shape == (3, 4)
    assert mask[:, 0].tolist() == [True, False, True]
    assert not mask[:, 1:].any()


def test_single_pixel_mask_keeps_2d_shape(tmp_path):
    mask_path = tmp_path / "pixel_mask.tif"
    tifffile.imwrite(mask_path, np.array([[5]], dtype=np.uint8))

    mask = load_grayscale_mask(mask_path, (3, 2))

    assert mask.shape == (3, 2)
    assert mask.sum() == 1
    assert mask[0, 0]


def test_trailing_singleton_channel_is_folded(tmp_path):
    mask_path = tmp_path / "channel_mask.tif"
    array = np.zeros((3, 2, 1), dtype=np.uint8)
    array[0, 0, 0] = 1
    array[2, 1, 0] = 1
    tifffile.imwrite(mask_path, array)

    mask = load_grayscale_mask(mask_path, (3, 2))

    assert mask.shape == (3, 2)
    assert mask.sum() == 2
    assert mask[0, 0] and mask[2, 1]


def test_single_row_with_singleton_channel_keeps_2d_shape(tmp_path):
    mask_path = tmp_path / "row_channel_mask.tif"
    tifffile.imwrite(mask_path, np.ones((1, 4, 1), dtype=np.uint8))

    mask = load_grayscale_mask(mask_path, (4, 3))

    assert mask.shape == (4, 3)
    assert mask[0].all()
    assert not mask[1:].any()


@pytest.mark.parametrize(
    "samples",
    [pytest.param(3, id="rgb"), pytest.param(4, id="rgba")],
)
def test_multisample_mask_uses_first_channel_only(tmp_path, samples):
    mask_path = tmp_path / "multisample_mask.tif"
    array = np.zeros((3, 2, samples), dtype=np.uint8)
    array[0, 1, 0] = 7
    array[1, 0, 1] = 99
    if samples == 4:
        array[2, 0, 3] = 255
    tifffile.imwrite(mask_path, array)

    mask = load_grayscale_mask(mask_path, (3, 2))

    assert mask.shape == (3, 2)
    assert mask.sum() == 1
    assert mask[0, 1]


def test_larger_mask_is_top_left_cropped(tmp_path):
    mask_path = tmp_path / "large_mask.tif"
    tifffile.imwrite(mask_path, np.ones((5, 6), dtype=np.uint8))

    mask = load_grayscale_mask(mask_path, (2, 3))

    assert mask.shape == (2, 3)
    assert mask.all()


def test_leading_singleton_frame_mask_is_unsqueezed(tmp_path):
    mask_path = tmp_path / "leading_frame_mask.tif"
    array = np.zeros((1, 8, 9), dtype=np.uint8)
    array[0, 2, 3] = 1
    array[0, 5, 7] = 1
    tifffile.imwrite(mask_path, array)

    mask = load_grayscale_mask(mask_path, (8, 9))

    assert mask.shape == (8, 9)
    assert mask.sum() == 2
    assert mask[2, 3] and mask[5, 7]


def test_leading_singleton_with_samples_is_unsqueezed_first_channel(
    tmp_path,
):
    mask_path = tmp_path / "leading_frame_samples_mask.tif"
    array = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    array[0, 1, 2, 0] = 1
    array[0, 1, 2, 2] = 1
    tifffile.imwrite(mask_path, array)

    mask = load_grayscale_mask(mask_path, (2, 3))

    assert mask.shape == (2, 3)
    assert mask.sum() == 1
    assert mask[1, 2]
