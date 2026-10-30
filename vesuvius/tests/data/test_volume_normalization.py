"""Volume's per-instance normalizations must treat a 1D or 2D read as one instance.

Before the fix only 3D reads got the temporary channel axis, so the per-channel
loops in Volume.__getitem__ iterated over dim 0 of a 2D slice: volume[z, :, :]
was normalized one row at a time, which removes every intensity difference
between rows (instance_zscore gives every row mean 0).
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from vesuvius.data.volume import Volume

INSTANCE_SCHEMES = ("instance_zscore", "instance_minmax", "percentile_minmax")


@pytest.fixture
def gradient_zarr(tmp_path):
    """A uint16 volume whose intensity rises along y, plus noise."""
    rng = np.random.default_rng(0)
    z, y, x = 4, 32, 24
    data = (np.arange(y)[None, :, None] * 100 + rng.integers(0, 50, (z, y, x))).astype(np.uint16)
    path = str(tmp_path / "gradient.zarr")
    arr = zarr.open(path, mode="w", shape=data.shape, chunks=(4, 16, 16), dtype="uint16")
    arr[:] = data
    return path, data


def _reference(scheme: str, block: np.ndarray) -> np.ndarray:
    """The documented per-instance formula, applied to the whole block at once."""
    block = block.astype(np.float32)
    if scheme == "instance_zscore":
        return (block - block.mean()) / max(block.std(), 1e-8)
    if scheme == "instance_minmax":
        return (block - block.min()) / max(block.max() - block.min(), 1e-8)
    lower, upper = np.percentile(block, (1.0, 99.0))
    return (np.clip(block, lower, upper) - lower) / float(upper - lower)


@pytest.mark.parametrize("scheme", INSTANCE_SCHEMES)
def test_2d_read_is_normalized_as_one_instance(gradient_zarr, scheme):
    path, data = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme=scheme)

    plane = vol[1, :, :]
    same_plane_as_3d = vol[1:2, :, :][0]

    assert plane.shape == (32, 24)
    np.testing.assert_allclose(plane, same_plane_as_3d, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(plane, _reference(scheme, data[1]), rtol=1e-5, atol=1e-5)
    # The y gradient survives: the first row is darker than the last.
    assert plane[0].mean() < plane[-1].mean()


@pytest.mark.parametrize("scheme", INSTANCE_SCHEMES)
def test_1d_read_is_normalized_as_one_instance(gradient_zarr, scheme):
    path, data = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme=scheme)

    line = vol[2, :, 5]

    assert line.shape == (32,)
    np.testing.assert_allclose(line, _reference(scheme, data[2, :, 5]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("scheme", INSTANCE_SCHEMES)
def test_3d_read_is_unchanged(gradient_zarr, scheme):
    path, data = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme=scheme)

    block = vol[0:3, 4:20, :]

    assert block.shape == (3, 16, 24)
    np.testing.assert_allclose(block, _reference(scheme, data[0:3, 4:20, :]), rtol=1e-5, atol=1e-5)


def test_single_voxel_read_with_instance_scheme(gradient_zarr):
    path, _ = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")

    # Used to raise IndexError: a 0-d read has no dim 0 to loop over.
    assert float(vol[1, 2, 3]) == 0.0


def test_global_zscore_2d_read(gradient_zarr):
    path, data = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme="global_zscore",
                 global_mean=1000.0, global_std=500.0)

    np.testing.assert_allclose(vol[1, :, :], (data[1].astype(np.float32) - 1000.0) / 500.0, rtol=1e-6)


def test_no_normalization_2d_read_returns_raw_values(gradient_zarr):
    path, data = gradient_zarr
    vol = Volume(type="zarr", path=path, normalization_scheme="none")

    plane = vol[1, :, :]

    assert plane.dtype == np.uint16
    np.testing.assert_array_equal(plane, data[1])
