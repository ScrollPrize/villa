"""Gaps found by an independent cv2 oracle: int16/float64, extreme ratios,
dims of 1, non-contiguous inputs and cubic/lanczos on partial border chunks.

Reference is the whole-volume two-pass cv2 resize. Tolerance as elsewhere:
integers <= 2 grey levels; floats 3e-5 * max|input| + 4 ulp of that magnitude.
"""
import cv2
import numpy as np
import pytest
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks import resize as R

ALL = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
       cv2.INTER_LANCZOS4]
MODE_IDS = ["nearest", "linear", "cubic", "area", "lanczos"]
DTYPES = [np.uint8, np.uint16, np.int16, np.float32, np.float64]


def _data(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(max(info.min, -30000), min(info.max, 30000), shape,
                            endpoint=True).astype(dtype)
    return (rng.random(shape) * 2000 - 1000).astype(dtype)


def _check(out, src, ref, mode):
    assert out.shape == ref.shape and out.dtype == ref.dtype
    if mode == cv2.INTER_NEAREST:
        np.testing.assert_array_equal(out, ref)
    elif np.issubdtype(src.dtype, np.integer):
        assert np.abs(out.astype(np.int64) - ref.astype(np.int64)).max() <= 2
    else:
        amax = np.abs(src).max()
        np.testing.assert_allclose(out, ref, rtol=0,
                                   atol=3e-5 * amax + 4 * np.spacing(amax))


def _chunked(tmp, src, target, chunks, mode):
    pin, pout = str(tmp / "i.zarr"), str(tmp / "o.zarr")
    zarr.open_group(pin, mode="w").zeros(name="0", shape=src.shape, chunks=src.shape,
                                         dtype=src.dtype)[...] = src
    zarr.open_group(pout, mode="w").zeros(name="0", shape=target, chunks=chunks,
                                          dtype=src.dtype)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                R._resize_chunk_worker((pin, pout, "0", oz, oy, ox, chunks, src.shape,
                                        target, mode, None))
    return zarr.open_group(pout, mode="r")["0"][...]


def _run(tmp, src, target, chunks, mode):
    ref = R._resize_chunk_3d(src, target, mode)
    _check(_chunked(tmp, src, target, chunks, mode), src, ref, mode)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("mode", ALL, ids=MODE_IDS)
def test_dtypes(tmp_path, dtype, mode):
    src = _data((11, 13, 17), dtype, 1)
    _run(tmp_path, src, (7, 20, 9), (3, 6, 4), mode)


# Ratios well beyond 1:101, one axis at a time (others 3 voxels).
EXTREME = [(1, 1000), (1000, 1), (2, 512), (512, 3), (997, 2)]


@pytest.mark.parametrize("dtype", [np.uint16, np.float64], ids=["uint16", "float64"])
@pytest.mark.parametrize("axis", [0, 2])
@pytest.mark.parametrize("n_in,n_out", EXTREME, ids=[f"{a}to{b}" for a, b in EXTREME])
@pytest.mark.parametrize("mode", ALL, ids=MODE_IDS)
def test_extreme_ratios(tmp_path, dtype, axis, n_in, n_out, mode):
    shape, target, chunks = [3] * 3, [3] * 3, [3] * 3
    shape[axis], target[axis], chunks[axis] = n_in, n_out, 97
    src = _data(tuple(shape), dtype, n_in * 7 + n_out)
    _run(tmp_path, src, tuple(target), tuple(chunks), mode)


DIM1 = [((1, 9, 11), (1, 17, 6)), ((9, 1, 11), (17, 1, 6)), ((9, 11, 1), (17, 6, 1)),
        ((1, 1, 1), (5, 7, 3)), ((7, 1, 1), (1, 1, 1)), ((6, 8, 10), (1, 1, 4))]


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32],
                         ids=["uint8", "int16", "float32"])
@pytest.mark.parametrize("shape,target", DIM1, ids=[f"{a}to{b}" for a, b in DIM1])
@pytest.mark.parametrize("mode", ALL, ids=MODE_IDS)
def test_dims_of_one(tmp_path, dtype, shape, target, mode):
    src = _data(shape, dtype, 3)
    _run(tmp_path, src, target, tuple(max(1, t // 2) for t in target), mode)


VIEWS = {
    "transpose": lambda a: a.transpose(2, 1, 0),
    "rev_z": lambda a: a[::-1],
    "rev_all": lambda a: a[::-1, ::-1, ::-1],
    "step2_y": lambda a: a[:, ::2],
}


@pytest.mark.parametrize("dtype", [np.uint8, np.float64], ids=["uint8", "float64"])
@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("mode", ALL, ids=MODE_IDS)
def test_non_contiguous_input(dtype, view, mode):
    base = _data((19, 34, 13), dtype, 4)
    src = VIEWS[view](base)
    assert not src.flags.c_contiguous
    target, chunks = (21, 9, 31), (5, 4, 7)
    ref = R._resize_chunk_3d(np.ascontiguousarray(src), target, mode)
    out = np.empty_like(ref)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                s = (oz, oy, ox)
                e = tuple(min(a + c, n) for a, c, n in zip(s, chunks, target))
                r = R._resize_chunk_taps(src, s, e, src.shape, target, mode)
                out[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = r
    _check(out, src, ref, mode)


# Odd chunk grids so the last chunk along every axis is partial.
BORDER = [((40, 37, 41), (23, 59, 67), (8, 16, 16)),
          ((33, 64, 65), (66, 32, 31), (7, 9, 30))]


@pytest.mark.parametrize("dtype", [np.uint16, np.float32], ids=["uint16", "float32"])
@pytest.mark.parametrize("case", BORDER, ids=["b0", "b1"])
@pytest.mark.parametrize("mode", [cv2.INTER_CUBIC, cv2.INTER_LANCZOS4],
                         ids=["cubic", "lanczos"])
def test_cubic_lanczos_border_chunks(tmp_path, dtype, case, mode):
    shape, target, chunks = case
    assert all(t % c for t, c in zip(target, chunks))
    _run(tmp_path, _data(shape, dtype, 5), target, chunks, mode)


# Window path (power-of-two ratios, nearest/linear/area): same views as above.
WINDOW = [((16, 24, 20), (32, 12, 40)), ((8, 32, 16), (8, 16, 64))]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("view", list(VIEWS))
@pytest.mark.parametrize("case", WINDOW, ids=["w0", "w1"])
@pytest.mark.parametrize("mode", [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA],
                         ids=["nearest", "linear", "area"])
def test_non_contiguous_input_window_path(dtype, view, case, mode):
    shape, target = case
    assert all(R._is_dyadic(a, b) for a, b in zip(shape, target))
    pre = {"transpose": lambda s: s[::-1], "step2_y": lambda s: (s[0], 2 * s[1], s[2])}
    src = VIEWS[view](_data(pre.get(view, lambda s: s)(shape), dtype, 6))
    assert src.shape == shape and not src.flags.c_contiguous
    ref = R._resize_chunk_3d(np.ascontiguousarray(src), target, mode)
    chunks = (5, 7, 9)
    out = np.empty_like(ref)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                s = (oz, oy, ox)
                e = tuple(min(a + c, n) for a, c, n in zip(s, chunks, target))
                out[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = R._resize_chunk_window(
                    src, s, e, src.shape, target, mode)
    np.testing.assert_array_equal(out, ref)  # window path is bit-identical to cv2


class _FakeZarr:
    """Stands in for zarr.open: the input array hands out non-contiguous views."""

    def __init__(self, arrays):
        self.arrays = arrays

    def __call__(self, path, mode="r"):
        return self.arrays[path]


@pytest.mark.parametrize("view", ["transpose", "rev_all"])
@pytest.mark.parametrize("case", [((16, 24, 20), (32, 12, 40)),    # window path
                                  ((13, 17, 19), (21, 9, 31))],   # tap path
                         ids=["window", "taps"])
@pytest.mark.parametrize("mode", [cv2.INTER_LINEAR, cv2.INTER_CUBIC],
                         ids=["linear", "cubic"])
def test_worker_fake_store_non_contiguous(monkeypatch, view, case, mode):
    shape, target = case
    src = VIEWS[view](_data(shape[::-1] if view == "transpose" else shape, np.uint16, 7))
    assert src.shape == shape and not src.flags.c_contiguous
    out = np.zeros(target, np.uint16)
    monkeypatch.setattr(R.zarr, "open", _FakeZarr({"in": {"0": src}, "out": {"0": out}}))
    chunks = (6, 5, 8)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                R._resize_chunk_worker(("in", "out", "0", oz, oy, ox, chunks, shape,
                                        target, mode, None))
    _check(out, src, R._resize_chunk_3d(np.ascontiguousarray(src), target, mode), mode)
