"""Deterministic chunk-seam cases for zarr_tasks resize (fast; no hypothesis)."""
import cv2
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks import resize as R

CASES = [
    ((16, 20, 18), (8, 10, 9)),     # 2x down
    ((8, 12, 10), (8, 30, 25)),     # 2.5x up
    ((9, 17, 13), (6, 23, 29)),     # coprime but small blocks
    ((5, 67, 13), (5, 23, 29)),     # y block 67 > 64: tap path, y shrinks, x grows
    ((7, 13, 67), (11, 29, 23)),    # tap path, z grows, x shrinks
    ((67, 9, 9), (9, 9, 9)),        # tap path, z-only shrink
    ((10, 12, 12), (10, 7, 18)),    # mixed shrink/grow in one 2D call
]


def _chunked(tmp, src, target, chunks, interp, tag):
    pin, pout = str(tmp / f"i{tag}.zarr"), str(tmp / f"o{tag}.zarr")
    gi = zarr.open_group(pin, mode="w")
    gi.zeros(name="0", shape=src.shape, chunks=(4, 4, 4), dtype=src.dtype)
    gi["0"][...] = src
    go = zarr.open_group(pout, mode="w")
    go.zeros(name="0", shape=target, chunks=chunks, dtype=src.dtype)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                R._resize_chunk_worker((pin, pout, "0", oz, oy, ox, chunks,
                                        src.shape, target, interp, None))
    return zarr.open_group(pout, mode="r")["0"][...]


@pytest.mark.parametrize("interp", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_cases(tmp_path, interp, case):
    shape, target = CASES[case]
    rng = np.random.default_rng(case)
    src = rng.integers(1, 256, shape).astype(np.uint8)
    ref = R._resize_chunk_3d(src, target, interp)
    a = _chunked(tmp_path, src, target, (3, 4, 5), interp, "a")
    b = _chunked(tmp_path, src, target, (5, 7, 2), interp, "b")
    np.testing.assert_array_equal(a, b)
    err = np.abs(a.astype(int) - ref)
    assert err.max() <= (0 if interp == 0 else 2)
    assert (err > 0).mean() <= (0.5 if interp == 4 else 0.25)


def test_all_zero_input_chunk_is_skipped(tmp_path):
    src = np.zeros((8, 8, 8), np.uint8)
    src[0, 0, 0] = 9
    out = _chunked(tmp_path, src, (16, 16, 16), (4, 4, 4), cv2.INTER_LINEAR, "z")
    assert out[8:, 8:, 8:].sum() == 0 and out[0, 0, 0] > 0


def test_output_dtype_cast(tmp_path):
    src = np.full((4, 4, 4), 300, np.uint16)
    pin, pout = str(tmp_path / "i.zarr"), str(tmp_path / "o.zarr")
    zarr.open_group(pin, mode="w").zeros(name="0", shape=src.shape, dtype=src.dtype)[...] = src
    zarr.open_group(pout, mode="w").zeros(name="0", shape=(2, 2, 2), dtype=np.uint8)
    R._resize_chunk_worker((pin, pout, "0", 0, 0, 0, (2, 2, 2), src.shape, (2, 2, 2),
                            cv2.INTER_LINEAR, np.dtype(np.uint8)))
    assert (zarr.open_group(pout, mode="r")["0"][...] == 255).all()


@pytest.mark.parametrize("interp", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_in,n_out", [(16, 7), (16, 8), (13, 29), (10, 25), (9, 9),
                                        (31, 10), (5, 2)])
@pytest.mark.parametrize("rows_out", [1, 3])
def test_axis_taps_match_cv2_1d(interp, n_in, n_out, rows_out):
    """Tap weights reproduce cv2.resize along x (float32).

    Resizing 2 identical rows to 1 row (shrink) or 3 rows (grow) exercises both
    of cv2's INTER_AREA modes, which depend on the other axis.
    """
    rng = np.random.default_rng(n_in * 100 + n_out)
    row = (rng.random((1, n_in)) * 255).astype(np.float32)
    ref = cv2.resize(np.repeat(row, 2, axis=0), (n_out, rows_out),
                     interpolation=interp)[0]
    true_area = rows_out == 1 and n_in >= n_out
    idx, w = R._axis_taps(0, n_out, n_in, n_out, interp, true_area)
    got = (row[0][idx].astype(np.float64) * w).sum(axis=1)
    np.testing.assert_allclose(got, ref, atol=2e-3, rtol=0)
    # A sub-range of outputs gets exactly the whole-axis taps.
    hi = min(5, n_out)
    i2, w2 = R._axis_taps(2, hi, n_in, n_out, interp, true_area)
    np.testing.assert_array_equal(i2, idx[2:hi])
    np.testing.assert_array_equal(w2, w[2:hi])
