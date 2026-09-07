"""Instance normalization must treat a read without a channel axis as one
channel, whatever its dimensionality.

Regression: the pseudo-channel axis was only inserted when a read returned
exactly 3 dimensions, so a 2D read (``vol[z, y0:y1, x0:x1]``) fell through to
``for c in range(data_slice.shape[0])`` with the first *spatial* axis standing
in for channels. Every row was then normalized against its own statistics
instead of the slice's, silently and with no error. A 1D read collapsed
further: each voxel was normalized against itself, returning all zeros.
"""

import numpy as np
import pytest
import zarr

_ZARR_V3 = int(zarr.__version__.split('.', 1)[0]) >= 3


def _write(path, array):
    """Write ``array`` to a zarr array at ``path`` with either zarr API."""
    if _ZARR_V3:
        z = zarr.open_array(str(path), mode="w", shape=array.shape,
                            chunks=array.shape, dtype=array.dtype, zarr_format=2)
    else:
        z = zarr.open(str(path), mode="w", shape=array.shape,
                      chunks=array.shape, dtype=array.dtype)
    z[:] = array
    return str(path)


@pytest.fixture
def gradient_volume(tmp_path):
    """A volume with a strong per-row gradient, so per-row normalization is
    numerically distinguishable from whole-slice normalization."""
    rng = np.random.default_rng(0)
    vol = (np.arange(16)[None, :, None] * 400
           + rng.integers(0, 300, size=(8, 16, 16))).astype("uint16")
    return _write(tmp_path / "gradient.zarr", vol), vol


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["instance_zscore", "instance_minmax"])
def test_2d_read_normalizes_over_whole_slice(gradient_volume, scheme):
    from vesuvius.data.volume import Volume

    path, vol = gradient_volume
    ref = vol[4].astype(np.float32)
    if scheme == "instance_zscore":
        expected = (ref - ref.mean()) / max(ref.std(), 1e-8)
    else:
        expected = (ref - ref.min()) / max(ref.max() - ref.min(), 1e-8)

    v = Volume(type="zarr", path=path, normalization_scheme=scheme)
    got = np.asarray(v[4, 0:16, 0:16])

    assert got.shape == ref.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["instance_zscore", "instance_minmax"])
def test_2d_and_3d_reads_agree_on_same_voxels(gradient_volume, scheme):
    from vesuvius.data.volume import Volume

    path, _ = gradient_volume
    v = Volume(type="zarr", path=path, normalization_scheme=scheme)
    two_d = np.asarray(v[4, 0:16, 0:16])
    three_d = np.asarray(v[4:5, 0:16, 0:16])[0]
    np.testing.assert_allclose(two_d, three_d, atol=1e-5)


@pytest.mark.unit
def test_1d_read_is_not_collapsed_to_zeros(gradient_volume):
    from vesuvius.data.volume import Volume

    path, vol = gradient_volume
    ref = vol[4, 7].astype(np.float32)
    expected = (ref - ref.mean()) / max(ref.std(), 1e-8)

    v = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")
    got = np.asarray(v[4, 7, 0:16])

    assert got.shape == ref.shape
    assert not np.allclose(got, 0.0), "1D read collapsed to zeros"
    np.testing.assert_allclose(got, expected, atol=1e-5)


@pytest.mark.unit
def test_channel_first_reads_still_normalize_per_channel(tmp_path):
    """Guard the intended behaviour: a 4D read keeps per-channel statistics."""
    from vesuvius.data.volume import Volume

    rng = np.random.default_rng(1)
    arr = np.stack([
        rng.integers(0, 100, size=(4, 8, 8)),
        rng.integers(500, 900, size=(4, 8, 8)),
    ]).astype("uint16")
    path = _write(tmp_path / "channels.zarr", arr)

    v = Volume(type="zarr", path=path, normalization_scheme="instance_zscore")
    got = np.asarray(v[0:2, 0:4, 0:8, 0:8])

    expected = np.stack([
        (arr[c].astype(np.float32) - arr[c].mean()) / max(arr[c].std(), 1e-8)
        for c in range(arr.shape[0])
    ])
    assert got.shape == arr.shape
    np.testing.assert_allclose(got, expected, atol=1e-5)
