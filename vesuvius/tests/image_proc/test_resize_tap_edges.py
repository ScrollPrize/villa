"""Edge cases for the zarr_tasks resize tap path (axis boundaries, tiny axes)."""
import cv2
import numpy as np
import pytest

from vesuvius.image_proc.run.zarr_tasks.tasks import resize as R

MODES = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA,
         cv2.INTER_LANCZOS4]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("n_in,n_out", [(1, 1), (1, 5), (1, 67), (2, 1), (5, 1),
                                        (67, 101), (101, 67), (71, 23), (65, 64)])
def test_axis_taps_indices_in_range_at_both_ends(mode, n_in, n_out):
    for o0, o1 in [(0, n_out), (0, 1), (n_out - 1, n_out)]:
        idx, w = R._axis_taps(o0, o1, n_in, n_out, mode, True)
        assert idx.shape[0] == o1 - o0 == w.shape[0]
        assert idx.min() >= 0 and idx.max() < n_in


def test_axis_taps_random_sizes_in_range():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n_in, n_out = (int(v) for v in rng.integers(1, 400, 2))
        for mode in MODES:
            idx, _ = R._axis_taps(max(n_out - 3, 0), n_out, n_in, n_out, mode, True)
            assert idx.min() >= 0 and idx.max() < n_in


@pytest.mark.parametrize("n_in,n_out", [(67, 101), (101, 67), (1, 101), (2, 1)])
@pytest.mark.parametrize("last_only", [False, True])
def test_linear_taps_last_output_matches_cv2(n_in, n_out, last_only):
    src = (1 + np.arange(n_in) * 254 // max(n_in - 1, 1)).astype(np.uint8)[None, None, :]
    o0 = n_out - 1 if last_only else 0
    out = R._resize_chunk_taps(src, (0, 0, o0), (1, 1, n_out), src.shape,
                               (1, 1, n_out), cv2.INTER_LINEAR)
    ref = cv2.resize(src[0], (n_out, 1), interpolation=cv2.INTER_LINEAR)[0, o0:]
    assert out.shape == (1, 1, n_out - o0)
    assert np.abs(out[0, 0].astype(int) - ref).max() <= 1


@pytest.mark.parametrize("n_in,n_out", [(30011, 29989), (99991, 100003)])
def test_nearest_taps_match_cv2_on_large_axes(n_in, n_out):
    row = np.arange(n_in, dtype=np.float32)[None, :]
    ref = cv2.resize(np.repeat(row, 2, axis=0), (n_out, 2),
                     interpolation=cv2.INTER_NEAREST)[0]
    idx, _ = R._axis_taps(0, n_out, n_in, n_out, cv2.INTER_NEAREST, True)
    np.testing.assert_array_equal(row[0][idx[:, 0]], ref)
