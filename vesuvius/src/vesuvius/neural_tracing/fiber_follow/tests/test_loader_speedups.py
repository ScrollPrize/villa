"""Loader speedups that must not change values: CT read hints, vectorized dense_line, heading-state draws."""
import json

import numpy as np

from vesuvius.neural_tracing.fiber_follow.regression.neighbor_mining import dense_line
from vesuvius.neural_tracing.fiber_follow.shared.data import SampleConfig, TracedFiber, make_sample, simulated_trace, traversal_curve
from vesuvius.neural_tracing.fiber_follow.shared.geometry import arclength
from vesuvius.neural_tracing.fiber_follow.shared.volume import ChunkedArray


def loop_dense_line(points, step):
    pieces = []
    for a, b in zip(points[:-1], points[1:]):
        length = np.linalg.norm(b-a)
        if length > 1e-9:
            n = max(1, int(np.ceil(length/step)))
            pieces.append(a + np.arange(n)[:, None]/n*(b-a))
    return np.concatenate([*pieces, points[-1:]])


def test_dense_line_matches_the_segment_loop_bit_for_bit():
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(scale=3., size=(400, 3)), axis=0)
    axis = np.array([[0., 0., 0.], [0., 0., 4.], [0., 3., 4.], [0., 3., 4.], [2., 3., 4.], [2., 3., 4.+5e-10]])
    for points in (walk, axis, walk*1e-3+1e4):
        for step in (1., .5, .3):
            np.testing.assert_array_equal(dense_line(points, step), loop_dense_line(points, step))


def test_hinted_block_reads_equal_direct_slices(tmp_path):
    shape, chunks = (40, 300, 260), (16, 128, 128)  # 16 KB planes, rows narrower than a page
    data = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    root = tmp_path/'ct'
    root.mkdir()
    (root/'.zarray').write_text(json.dumps(dict(shape=shape, chunks=chunks, dtype='|u1', fill_value=0, order='C',
                                                filters=None, compressor=None, zarr_format=2, dimension_separator='.')))
    padded = np.zeros([-(-n//c)*c for n, c in zip(shape, chunks)], np.uint8)
    padded[:shape[0], :shape[1], :shape[2]] = data
    for key in np.ndindex(*(p//c for p, c in zip(padded.shape, chunks))):
        if key == (1, 1, 1):
            continue  # missing chunk reads as fill
        block = padded[tuple(slice(k*c, (k+1)*c) for k, c in zip(key, chunks))]
        (root/'.'.join(map(str, key))).write_bytes(block.tobytes())
    expected = padded.copy()
    expected[16:32, 128:256, 128:256] = 0
    array = ChunkedArray(root, cache_bytes=1 << 20)
    for start, size in (((3, 100, 120), (20, 60, 30)),     # narrow rows: one hint per slice
                        ((0, 0, 0), (40, 300, 260)),       # whole planes: one hint per piece
                        ((-5, 250, 240), (30, 80, 40)),    # partly outside the array
                        ((10, 120, 120), (12, 20, 20))):   # touches the missing chunk
        got = array.read(np.array(start), np.array(size))
        want = np.zeros(size, np.uint8)
        lo, hi = np.maximum(start, 0), np.minimum(np.add(start, size), shape)
        want[tuple(slice(a-s, b-s) for a, b, s in zip(lo, hi, start))] = expected[tuple(slice(a, b) for a, b in zip(lo, hi))]
        np.testing.assert_array_equal(got, want)


def test_simulated_trace_is_make_samples_path_with_the_same_draws():
    arc = np.arange(0., 300.)
    p = np.c_[10*np.sin(arc/40), arc, 5*np.cos(arc/25)]
    fiber = TracedFiber('curve', p, arclength(p), 'V')
    cfg = SampleConfig()
    for seed in range(12):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        item = make_sample(fiber, 150., bool(seed % 2), cfg, a)
        path = simulated_trace(*traversal_curve(fiber, bool(seed % 2)), 150., cfg, b)[0]
        np.testing.assert_array_equal(path, item['observed_path'])
        assert a.random() == b.random()


def test_cached_cdf_draws_equal_generator_choice():
    weights = np.random.default_rng(2).random(5000)
    weights /= weights.sum()
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    assert [int(a.choice(len(weights), p=weights)) for _ in range(2000)] == \
           [int(cdf.searchsorted(b.random(), side='right')) for _ in range(2000)]
