"""median_filter_3d must match scipy's nearest-border median exactly."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from vesuvius.utils.median3d import median_filter_3d


@pytest.mark.parametrize("radius", [1, 2, 3])
@pytest.mark.parametrize("kind", ["noise", "sparse", "constant"])
def test_uint8_kernel_matches_scipy(radius: int, kind: str) -> None:
    rng = np.random.default_rng(radius)
    shape = (7, 40, 45)
    if kind == "noise":
        volume = rng.integers(0, 256, size=shape, dtype=np.uint8)
    elif kind == "sparse":
        volume = (rng.random(shape) < 0.05).astype(np.uint8) * rng.integers(1, 256, size=shape, dtype=np.uint8)
    else:
        volume = np.full(shape, 200, dtype=np.uint8)
        volume[3, 20, 22] = 0
    expected = ndimage.median_filter(volume, size=2 * radius + 1, mode="nearest")
    got = median_filter_3d(volume, radius)
    assert got.dtype == np.uint8 and got.flags.c_contiguous
    np.testing.assert_array_equal(got, expected)


def test_non_uint8_falls_back_to_scipy_and_radius_zero_is_identity() -> None:
    volume = np.random.default_rng(0).standard_normal((3, 8, 9)).astype(np.float32)
    np.testing.assert_array_equal(median_filter_3d(volume, 1), ndimage.median_filter(volume, size=3, mode="nearest"))
    np.testing.assert_array_equal(median_filter_3d(volume, 0), volume)
    with pytest.raises(ValueError):
        median_filter_3d(volume, -1)
    with pytest.raises(ValueError):
        median_filter_3d(volume[0], 1)
