"""zarr_tasks resize must not depend on the output chunk grid.

The oracle is the module's own whole-volume ``_resize_chunk_3d`` (cv2.resize
per YX slice, then per ZX plane). The chunked result must be identical across
chunkings, bit-identical to the oracle for nearest, and within 2 grey levels
(uint8) of it for other modes.
"""
import cv2
import numpy as np
import pytest
import zarr

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from vesuvius.image_proc.run.zarr_tasks.tasks import resize as R

MODES = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
         cv2.INTER_LANCZOS4]


def chunked(tmp, src, target, chunks, interp, tag):
    pin, pout = str(tmp / f"in{tag}.zarr"), str(tmp / f"out{tag}.zarr")
    gi = zarr.open_group(pin, mode="w")
    gi.zeros(name="0", shape=src.shape, chunks=src.shape, dtype=src.dtype)
    gi["0"][...] = src
    go = zarr.open_group(pout, mode="w")
    go.zeros(name="0", shape=target, chunks=chunks, dtype=src.dtype)
    for oz in range(0, target[0], chunks[0]):
        for oy in range(0, target[1], chunks[1]):
            for ox in range(0, target[2], chunks[2]):
                R._resize_chunk_worker((pin, pout, "0", oz, oy, ox, chunks,
                                        src.shape, target, interp, None))
    return zarr.open_group(pout, mode="r")["0"][...]




@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
@pytest.mark.parametrize("interp", MODES)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_chunk_invariant(tmp_path, interp, dtype, data):
    shape = tuple(data.draw(st.integers(3, 24)) for _ in range(3))
    target = tuple(data.draw(st.integers(2, 40)) for _ in range(3))
    c1 = tuple(data.draw(st.integers(1, 16)) for _ in range(3))
    c2 = tuple(data.draw(st.integers(1, 16)) for _ in range(3))
    seed = data.draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    src = (rng.integers(0, 256, shape) if dtype == np.uint8
           else rng.random(shape) * 100).astype(dtype)
    ref = R._resize_chunk_3d(src, target, interp)
    a = chunked(tmp_path, src, target, c1, interp, "a")
    b = chunked(tmp_path, src, target, c2, interp, "b")
    np.testing.assert_array_equal(a, b)
    if interp == cv2.INTER_NEAREST:
        np.testing.assert_array_equal(a, ref)
    else:
        err = np.abs(a.astype(np.float64) - ref)
        if dtype == np.uint8:
            assert err.max() <= 2
        else:
            assert err.max() <= 1e-5 * (1 + np.abs(ref).max())


@pytest.mark.parametrize("interp", MODES)
def test_real_style_odd_ratio_taps_path(tmp_path, interp):
    """3943->7888-like coprime ratio takes the tap path; still seam-free."""
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (8, 67, 71)).astype(np.uint8)
    target = (8, 134, 141)  # 67/134 = 1/2 exact on y, 71/141 coprime on x
    a = chunked(tmp_path, src, target, (8, 32, 32), interp, "a")
    b = chunked(tmp_path, src, target, (8, 141, 141), interp, "b")
    np.testing.assert_array_equal(a, b)
