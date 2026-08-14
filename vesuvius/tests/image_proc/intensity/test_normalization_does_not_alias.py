"""Normalisers must not write through to the array they are handed.

They rescale with in-place arithmetic on whatever `_prepare_image` returns, and
that used `astype(..., copy=False)`: for input already in the target dtype it
returned the input itself, so the caller's buffer was normalised in place. The
aliasing was dtype-dependent -- float32 in place, anything else copied -- which
is why it went unnoticed.

It bites through views. `neural_tracing.infer.CropCache` slices its cached
supercrops (`_extract_subcrop` returns `supercrop[z0:z0+d, ...]`) and the caller
passes `crop.numpy()` straight to `normalize_zscore`, so normalising one crop
rewrote the cached supercrop, and the next crop drawn from that entry -- the
whole point of caching a supercrop is that neighbouring crops reuse it -- was
normalised twice over.
"""

from __future__ import annotations

import numpy as np
import pytest

from vesuvius.image_proc.intensity.normalization import (
    normalize_ct,
    normalize_minmax,
    normalize_robust,
    normalize_zscore,
)

CT_PROPERTIES = {
    "mean": 10.0,
    "std": 5.0,
    "percentile_00_5": 1.0,
    "percentile_99_5": 20.0,
}

NORMALISERS = [
    pytest.param(normalize_minmax, id="minmax"),
    pytest.param(normalize_zscore, id="zscore"),
    pytest.param(lambda image: normalize_ct(image, intensity_properties=CT_PROPERTIES),
                 id="ct"),
    pytest.param(normalize_robust, id="robust"),
]


def _volume(dtype=np.float32):
    return np.arange(64, dtype=dtype).reshape(4, 4, 4)


@pytest.mark.parametrize("normalise", NORMALISERS)
def test_float32_input_is_left_alone(normalise):
    """float32 is the dtype that used to be normalised in place."""
    image = _volume()
    original = image.copy()

    normalise(image)

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("normalise", NORMALISERS)
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float64])
def test_other_dtypes_are_left_alone_too(normalise, dtype):
    image = _volume(dtype)
    original = image.copy()

    normalise(image)

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("normalise", NORMALISERS)
def test_a_view_does_not_corrupt_its_base(normalise):
    """The crop-cache pattern: normalise a slice, the backing buffer must survive."""
    supercrop = np.arange(512, dtype=np.float32).reshape(8, 8, 8)
    original = supercrop.copy()

    normalise(supercrop[2:6, 2:6, 2:6])

    np.testing.assert_array_equal(supercrop, original)


def test_overlapping_crops_normalise_independently():
    """Two crops sharing a cached supercrop must give the same result either order."""
    supercrop = np.arange(512, dtype=np.float32).reshape(8, 8, 8)

    second_alone = normalize_zscore(supercrop[2:6, 2:6, 2:6].copy())
    normalize_zscore(supercrop[0:4, 0:4, 0:4])          # overlaps the crop above
    second_after_first = normalize_zscore(supercrop[2:6, 2:6, 2:6])

    np.testing.assert_allclose(second_after_first, second_alone)


@pytest.mark.parametrize("normalise", NORMALISERS)
def test_result_is_still_normalised(normalise):
    """The copy must not change what the functions compute."""
    result = normalise(_volume())

    assert result.dtype == np.float32
    assert result.shape == (4, 4, 4)
    assert np.isfinite(result).all()


def test_zscore_result_is_standardised():
    result = normalize_zscore(_volume())

    assert float(np.mean(result)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.std(result)) == pytest.approx(1.0, abs=1e-6)
