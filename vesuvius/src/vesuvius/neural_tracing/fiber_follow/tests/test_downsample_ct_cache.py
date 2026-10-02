"""Local CT pyramid level from cached chunks: remote rounding rule, complete-children-only writes."""
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'/'downsample_ct_cache.py'
spec = importlib.util.spec_from_file_location('downsample_ct_cache', SCRIPT)
downsample = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downsample)


def test_block_mean_rounds_half_to_even_like_the_remote_pyramid():
    fine = np.random.default_rng(0).integers(0, 256, (8, 8, 8)).astype(np.uint8)
    total = fine.reshape(4, 2, 4, 2, 4, 2).sum((1, 3, 5)).astype(np.float64)
    np.testing.assert_array_equal(downsample.block_mean(fine), np.round(total/8))


def test_builds_only_coarse_chunks_with_every_child_cached(tmp_path):
    from vesuvius.neural_tracing.fiber_follow.shared.volume import RemoteChunkedArray
    url, chunks = 's3://bucket/volume.zarr', (8, 8, 8)
    fine_root, coarse_root = (RemoteChunkedArray.cache_path(url, level, tmp_path) for level in (0, 1))
    volume = np.random.default_rng(1).integers(0, 256, (32, 32, 32)).astype(np.uint8)
    for root, shape in ((fine_root, 32), (coarse_root, 16)):
        root.mkdir(parents=True)
        (root/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=[shape]*3, chunks=list(chunks), dtype='|u1',
                                                    fill_value=0, compressor=None, filters=None, order='C',
                                                    dimension_separator='/')))
    for key in np.ndindex(4, 4, 4):
        if key != (3, 3, 3):  # coarse chunk (1, 1, 1) misses one child
            path = downsample.chunk_path(fine_root, key, '/')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(volume[tuple(slice(8*k, 8*k+8) for k in key)].tobytes())
    tasks, touched, _, _ = downsample.plan(url, 0, tmp_path)
    assert touched == 8 and sorted(t[4] for t in tasks) == [k for k in np.ndindex(2, 2, 2) if k != (1, 1, 1)]
    assert {downsample.build(t) for t in tasks} == {'built'} and {downsample.build(t) for t in tasks} == {'existing'}
    expected = downsample.block_mean(volume)
    for key in np.ndindex(2, 2, 2):
        path = downsample.chunk_path(coarse_root, key, '/')
        if key == (1, 1, 1):
            assert not path.exists()
        else:
            np.testing.assert_array_equal(np.fromfile(path, np.uint8).reshape(chunks),
                                          expected[tuple(slice(8*k, 8*k+8) for k in key)])
    assert all(downsample.verify(t) for t in tasks)
