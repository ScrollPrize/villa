"""merge's pyramid downsampling rounds the 2x2x2 mean for integer volumes.

Casting the float mean straight to the input dtype truncated it, so every
integer level was on average just under half a grey level darker than the
level it came from, and the bias added up with each level.
"""
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.merge import _downsample_chunk_worker


def _run(tmp_path, level0):
    path = str(tmp_path / "vol.zarr")
    root = zarr.open_group(path, mode="w")
    root.zeros(name="0", shape=level0.shape, chunks=level0.shape, dtype=level0.dtype)
    root["0"][...] = level0
    out_shape = tuple((s + 1) // 2 for s in level0.shape)
    root.zeros(name="1", shape=out_shape, chunks=out_shape, dtype=level0.dtype)
    status = _downsample_chunk_worker(
        ((0, 0, 0), path, 0, 1, level0.shape, out_shape, out_shape, "average"))
    assert status == "written"
    return zarr.open_group(path, mode="r")["1"][...]


def _blocks(dtype, low, high):
    # Each 2x2x2 block holds seven `low` voxels and one `high` voxel.
    vol = np.full((4, 4, 4), low, dtype=dtype)
    vol[::2, ::2, ::2] = high
    return vol


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_average_rounds_integer_mean(tmp_path, dtype):
    # (7 * 10 + 15) / 8 = 10.625 -> 11 (truncation gave 10)
    out = _run(tmp_path, _blocks(dtype, 10, 15))
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, np.full((2, 2, 2), 11, dtype=dtype))


def test_average_rounds_down_below_half(tmp_path):
    # (7 * 10 + 13) / 8 = 10.375 -> 10
    out = _run(tmp_path, _blocks(np.uint8, 10, 13))
    np.testing.assert_array_equal(out, np.full((2, 2, 2), 10, dtype=np.uint8))


@pytest.mark.parametrize(
    "dtype, low, expected",
    [
        (np.uint8, 10, 10),       # mean 10.5: tie between 10 (even) and 11 -> 10
        (np.uint8, 11, 12),       # mean 11.5: tie between 11 and 12 (even) -> 12
        (np.uint8, 250, 250),     # mean 250.5 -> 250 (even), near the top of the range
        (np.uint8, 251, 252),     # mean 251.5 -> 252 (even)
        (np.uint16, 1000, 1000),  # mean 1000.5 -> 1000 (even)
        (np.uint16, 1001, 1002),  # mean 1001.5 -> 1002 (even)
        (np.uint16, 65530, 65530),  # mean 65530.5 -> 65530 (even)
        (np.uint16, 65531, 65532),  # mean 65531.5 -> 65532 (even)
    ],
)
def test_average_exact_half_ties_round_to_even(tmp_path, dtype, low, expected):
    # Seven `low` voxels and one `low + 4`: block sum 8 * low + 4, an exact .5 mean.
    # np.rint rounds half to even, so the tie goes to whichever neighbour is even.
    out = _run(tmp_path, _blocks(dtype, low, low + 4))
    np.testing.assert_array_equal(out, np.full((2, 2, 2), expected, dtype=dtype))


def test_average_saturated_uint8_stays_255(tmp_path):
    out = _run(tmp_path, np.full((4, 4, 4), 255, dtype=np.uint8))
    np.testing.assert_array_equal(out, np.full((2, 2, 2), 255, dtype=np.uint8))


def test_average_float_volume_is_not_rounded(tmp_path):
    # (7 * 1.0 + 1.5) / 8 = 1.0625 stays exact for float data
    out = _run(tmp_path, _blocks(np.float32, 1.0, 1.5))
    np.testing.assert_allclose(out, np.full((2, 2, 2), 1.0625, dtype=np.float32))
