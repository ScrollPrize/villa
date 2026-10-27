"""`normalize_image`'s robust_mad mode must stay bit-identical after the fast path.

`normalize_image` is the per-crop normaliser of the ink pipeline: ink training
(`data/dataset.py`, `data/coord_patch_dataset.py`) and native 3D inference
(`inference/infer_full3d_tifxyz.py`) all hand it a uint8 CT crop widened to float32 and the
default `NormalizationConfig` (`robust_mad`, percentiles 1.0/99.0). It now routes that mode
through `normalize_robust_histogram`, so the contract is the same as flat inference's:
element-for-element identical output, with every case the fast path cannot prove delegated
back to `normalize_robust`.
"""
from __future__ import annotations

import numpy as np
import pytest

from vesuvius.image_proc.intensity.normalization import normalize_robust
from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.ink_detection.data.normalization import normalize_image

# Crops the ink datasets produce: depth x patch x patch and a single 2D slice.
CROP_SHAPES = [(17, 128, 128), (4, 32, 32), (128, 128), (7, 64, 64)]


def _uint8_crop(shape, seed):
    rng = np.random.default_rng(seed)
    # Real CT crops are mostly a smooth background with sparse structure, and they
    # saturate at 0 and 255 far more often than a uniform sample would.
    crop = rng.normal(96.0, 40.0, size=shape).round().clip(0, 255).astype(np.uint8)
    flat = crop.reshape(-1)
    flat[rng.random(flat.size) < 0.05] = 0
    flat[rng.random(flat.size) < 0.02] = 255
    return crop


@pytest.mark.parametrize("shape", CROP_SHAPES)
def test_robust_mad_is_bit_identical_for_integer_crops(shape):
    config = NormalizationConfig()
    assert config.mode == "robust_mad" and config.percentile_lower == 1.0

    for seed in range(8):
        crop = _uint8_crop(shape, seed)
        expected = normalize_robust(
            crop.astype(np.float32, copy=False),
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
        )
        got = normalize_image(crop, config)
        assert got.dtype == np.float32
        np.testing.assert_array_equal(got, expected)


def test_non_contiguous_view_still_matches_normalize_robust():
    """The datasets normalise `image[valid_slices]` -- a view into a padded crop."""
    config = NormalizationConfig()
    padded = np.zeros((9, 64, 64), dtype=np.uint8)
    padded[1:9, 2:62, 3:61] = _uint8_crop((8, 60, 58), seed=21)
    view = padded[1:9, 2:62, 3:61]
    assert not view.flags["C_CONTIGUOUS"]

    expected = normalize_robust(
        view.astype(np.float32, copy=False),
        percentile_lower=config.percentile_lower,
        percentile_upper=config.percentile_upper,
    )
    np.testing.assert_array_equal(normalize_image(view, config), expected)


def test_non_integer_crops_still_match_normalize_robust():
    config = NormalizationConfig()
    rng = np.random.default_rng(11)
    crop = (rng.normal(0.3, 0.02, size=(8, 48, 48)).astype(np.float32) + 0.5)  # not integer-valued
    assert not np.array_equal(crop, np.rint(crop))

    expected = normalize_robust(
        crop, percentile_lower=config.percentile_lower, percentile_upper=config.percentile_upper
    )
    np.testing.assert_array_equal(normalize_image(crop, config), expected)


@pytest.mark.parametrize("lower,upper", [(0.5, 99.5), (0.0, 100.0), (2.0, 98.0)])
def test_non_default_percentiles_still_match_normalize_robust(lower, upper):
    config = NormalizationConfig(percentile_lower=lower, percentile_upper=upper)
    crop = _uint8_crop((8, 48, 48), seed=3)

    expected = normalize_robust(
        crop.astype(np.float32, copy=False), percentile_lower=lower, percentile_upper=upper
    )
    np.testing.assert_array_equal(normalize_image(crop, config), expected)


def test_other_modes_are_untouched():
    crop = _uint8_crop((4, 16, 16), seed=5)

    none_cfg = NormalizationConfig(mode="none")
    np.testing.assert_array_equal(
        normalize_image(crop, none_cfg), crop.astype(np.float32, copy=False)
    )

    divide_cfg = NormalizationConfig(mode="divide", divisor=255.0)
    np.testing.assert_allclose(
        normalize_image(crop, divide_cfg), crop.astype(np.float32) / 255.0, rtol=0, atol=0
    )
