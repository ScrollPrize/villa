"""Chunked resize must match the whole-volume cv2 result with 1-voxel-thick chunks.

cv2 computes source positions as floor(x * scale) in floating point, which is
not translation-invariant (90 * 0.7 = 62.999..., but 10 * 0.7 + 56 = 63), so a
chunk cannot be computed by calling cv2.resize on a cropped window.
"""
import cv2
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks import resize as R

ALL = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
       cv2.INTER_LANCZOS4]


def chunked(tmp, src, target, chunks, interp):
    tmp.mkdir(parents=True, exist_ok=True)
    pin, pout = str(tmp / "i.zarr"), str(tmp / "o.zarr")
    zarr.open_group(pin, mode="w").zeros(name="0", shape=src.shape, chunks=src.shape,
                                         dtype=src.dtype)[...] = src
    zarr.open_group(pout, mode="w").zeros(name="0", shape=target, chunks=chunks,
                                          dtype=src.dtype)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                R._resize_chunk_worker((pin, pout, "0", oz, oy, ox, chunks, src.shape,
                                        target, interp, None))
    return zarr.open_group(pout, mode="r")["0"][...]


def one_axis(n_in, n_out, axis, other=2):
    shape, target, chunks = [other] * 3, [other] * 3, [other] * 3
    shape[axis], target[axis], chunks[axis] = n_in, n_out, 1
    return tuple(shape), tuple(target), tuple(chunks)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("n_in,n_out", [(70, 100), (140, 200)])
def test_nearest_witnesses_1_thick(tmp_path, axis, n_in, n_out):
    shape, target, chunks = one_axis(n_in, n_out, axis)
    src = np.broadcast_to(
        np.arange(1, n_in + 1, dtype=np.uint16).reshape([-1 if a == axis else 1
                                                         for a in range(3)]), shape).copy()
    ref = R._resize_chunk_3d(src, target, cv2.INTER_NEAREST)
    out = chunked(tmp_path, src, target, chunks, cv2.INTER_NEAREST)
    np.testing.assert_array_equal(out, ref)


# (n_in, n_out, mode, max error vs whole volume on uint8 random data)
CASES = [
    (70, 100, cv2.INTER_NEAREST, 0), (140, 200, cv2.INTER_NEAREST, 0),
    (70, 100, cv2.INTER_AREA, 1),    # area upscale, failed by 149 on 1-thick chunks
    (208, 254, cv2.INTER_AREA, 1),   # area upscale, cv2 rounds its offset in float32
    (116, 300, cv2.INTER_LINEAR, 1),
    (70, 47, cv2.INTER_AREA, 1),     # true area shrink (box filter table)
    (300, 93, cv2.INTER_AREA, 1),
]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("case", CASES)
def test_harness_witnesses_1_thick(tmp_path, axis, case):
    n_in, n_out, mode, tol = case
    shape, target, chunks = one_axis(n_in, n_out, axis)
    src = np.random.default_rng(n_in * 1000 + n_out).integers(1, 256, shape).astype(np.uint8)
    ref = R._resize_chunk_3d(src, target, mode).astype(int)
    out = chunked(tmp_path, src, target, chunks, mode).astype(int)
    assert np.abs(out - ref).max() <= tol


@pytest.mark.parametrize("mode", ALL)
def test_one_thick_equals_whole_chunk(tmp_path, mode):
    """Grid independence: 1-thick chunks == a single chunk covering everything."""
    rng = np.random.default_rng(mode)
    src = rng.integers(1, 256, (5, 70, 23)).astype(np.uint8)
    target = (7, 100, 31)
    a = chunked(tmp_path / "a", src, target, (1, 1, 31), mode)
    b = chunked(tmp_path / "b", src, target, target, mode)
    np.testing.assert_array_equal(a, b)


def test_nearest_taps_sweep_vs_cv2():
    rng = np.random.default_rng(0)
    for n_in in range(1, 301, 7):
        for n_out in range(1, 301, 11):
            row = np.arange(n_in, dtype=np.float32)[None, :]
            ref = cv2.resize(np.repeat(row, 2, 0), (n_out, 2),
                             interpolation=cv2.INTER_NEAREST)[0]
            for o0 in {0, int(rng.integers(0, n_out)), n_out - 1}:
                idx, _ = R._axis_taps(o0, n_out, n_in, n_out, cv2.INTER_NEAREST, True)
                np.testing.assert_array_equal(row[0][idx[:, 0]], ref[o0:])


@pytest.mark.parametrize("rows_out", [1, 3])
def test_area_taps_sweep_vs_cv2(rows_out):
    """Both cv2 INTER_AREA regimes (box table when both axes shrink, else upscale)."""
    rng = np.random.default_rng(rows_out)
    for n_in in range(1, 301, 13):
        for n_out in range(1, 301, 7):
            row = rng.random((1, n_in)).astype(np.float32)
            ref = cv2.resize(np.repeat(row, 2, 0), (n_out, rows_out),
                             interpolation=cv2.INTER_AREA)[0]
            true_area = rows_out == 1 and n_in >= n_out
            idx, w = R._axis_taps(0, n_out, n_in, n_out, cv2.INTER_AREA, true_area)
            got = (row[0][idx].astype(np.float64) * w).sum(axis=1)
            np.testing.assert_allclose(got, ref, atol=1e-4, rtol=0)


# float32 INTER_LINEAR witnesses: a ratio-aligned local cv2 window differs from
# the whole-axis cv2 result by ~1e-3, so these must not use a local cv2 window.
# (name, n_in, n_out, output index, nonzero input indices, values)
LINEAR_F32 = [
    ("L07", 285, 280, 253, [257, 258], [-59, 53]),
    ("L08", 260, 255, 254, [258, 259], [-101, 1]),
    ("L09", 300, 138, 136, [296, 297], [253, 0]),
    ("L10", 264, 102, 101, [262, 263], [0, 255]),
]


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("case", LINEAR_F32, ids=[c[0] for c in LINEAR_F32])
def test_linear_float32_witnesses(tmp_path, axis, case):
    _, n_in, n_out, o, idx, vals = case
    shape, target, chunks = one_axis(n_in, n_out, axis)
    line = np.zeros(n_in, np.float32)
    line[idx] = vals
    src = np.broadcast_to(line.reshape([-1 if a == axis else 1 for a in range(3)]),
                          shape).copy()
    thin = chunked(tmp_path / "a", src, target, chunks, cv2.INTER_LINEAR)
    whole = chunked(tmp_path / "b", src, target, target, cv2.INTER_LINEAR)
    # Grid independence is exact.
    np.testing.assert_array_equal(thin, whole)
    # Close to cv2 on the whole axis (cv2 uses float32 coefficients).
    ref = R._resize_chunk_3d(src, target, cv2.INTER_LINEAR)
    np.testing.assert_allclose(thin, ref, atol=1e-3, rtol=0)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_linear_float32_92_300_1_thick(tmp_path, axis):
    """92 -> 300 (reduced 23:75) float32 linear was chunk-dependent by ~1.5e-5."""
    shape, target, chunks = one_axis(92, 300, axis)
    src = (np.random.default_rng(7).random(shape) * 510 - 255).astype(np.float32)
    thin = chunked(tmp_path / "a", src, target, chunks, cv2.INTER_LINEAR)
    whole = chunked(tmp_path / "b", src, target, target, cv2.INTER_LINEAR)
    np.testing.assert_array_equal(thin, whole)
