"""Affine CT sampling across dense boxes, sparse grids and source boundaries."""
import itertools
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import cuda
from vc_reader.reader import ZarrReader
from vc_reader.reader._sampling import affine32


def reference(values, shape, origin, axes, fill):
    shape, base, origin, axes = affine32(shape, origin, axes)
    indices = np.indices(shape, dtype=np.float32).reshape(3, -1).T
    local = np.broadcast_to(origin, indices.shape).copy()
    for a in range(3):
        local += indices[:, a:a+1]*axes[:, a]
    floor = np.floor(local).astype(np.int64)
    fraction = local-floor
    floor += base
    out = np.zeros(len(indices), np.float64)
    for corner in itertools.product((0, 1), repeat=3):
        xyz = floor+corner
        inside = ((xyz >= 0) & (xyz < values.shape)).all(1)
        sampled = np.full(len(indices), fill, np.float64)
        sampled[inside] = values[tuple(xyz[inside].T)]
        weights = np.prod(np.where(corner, fraction, 1-fraction), axis=1)
        out += weights*sampled
    return out.reshape(shape)


@pytest.mark.parametrize('shape,origin,axes', [
    ((24, 27, 29), (5.25, 7.5, 9.75), np.eye(3)),
    ((24, 27, 29), (12.25, 9.5, 10.75), [[.9, -.3, .1], [.3, .9, -.2], [.1, .2, 1.]]),
    ((24, 27, 29), (28.25, 31.5, 40.75), -np.eye(3)),
    ((24, 27, 29), (-1.25, -2.5, -3.75), np.eye(3)),
    ((24, 27, 29), (51.25, 53.5, 55.75), np.eye(3)),
    ((24, 27, 29), (-100., 200., 300.), np.eye(3)),
    ((24, 27, 29), (2., 3., 4.), np.zeros((3, 3))),
    ((24, 27, 29), (0., 0., 0.), np.eye(3)*4),
    ((1, 80, 80), (2., 3., 4.), [[0., .3, .2], [0., .4, .3], [0., -.2, .4]]),
    ((0, 27, 29), (1., 2., 3.), np.eye(3)),
])
@pytest.mark.parametrize('threads', [1, 4])
def test_affine_sampling(tmp_path, shape, origin, axes, threads):
    rng = np.random.default_rng(794)
    values = rng.integers(0, 256, (63, 65, 67), dtype=np.uint8)
    chunks = (17, 19, 23)
    fill = 19
    path = tmp_path/'ct'
    path.mkdir()
    (path/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=values.shape,
        chunks=chunks, dtype='|u1', compressor=None, filters=None, fill_value=fill, order='C')))
    for index in itertools.product(*(range((s+c-1)//c) for s, c in zip(values.shape, chunks))):
        region = tuple(slice(i*c, min((i+1)*c, s)) for i, c, s in zip(index, chunks, values.shape))
        if index == (1, 1, 1):
            values[region] = fill  # An absent source chunk.
            continue
        block = np.full(chunks, fill, np.uint8)
        part = values[region]
        block[tuple(slice(n) for n in part.shape)] = part
        (path/'.'.join(map(str, index))).write_bytes(block.tobytes())
    expected = reference(values, shape, origin, axes, fill)
    with ZarrReader.open(path, threads=threads) as reader:
        output = np.empty(shape, np.float32)
        assert reader.read_affine(shape, origin, axes, out=output) is output
        np.testing.assert_allclose(output, expected, atol=6e-5, rtol=3e-7)
        if cuda.is_available():
            with reader.cuda() as device:
                actual = device.read_affine(shape, origin, axes, source='chunks').numpy()
                np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=3e-7)


def test_zscore_float_extremes_and_cancellation():
    from vc_reader import augment as a
    rng = np.random.default_rng(391)
    inputs = [rng.normal(size=(1, 31, 33, 35)).astype(np.float32),
        np.full((1, 17, 19), 1e6, np.float32), np.zeros((1, 9, 13), np.float32),
        np.full((1, 9, 13), np.finfo(np.float32).max, np.float32),
        np.resize(np.array([-1, 0, 1], np.float32)*np.finfo(np.float32).max, (1, 13, 17)),
        np.resize(np.array([0, 1, 2], np.float32)*np.nextafter(np.float32(0), np.float32(1)), (1, 13, 17))]
    inputs[1].ravel()[::3] += .125
    for image in inputs:
        wide = image.astype(np.float64)
        for eps in (0., 1e-6):
            divisor = max(wide.std(), eps)
            expected = (wide-wide.mean())/divisor if divisor else np.zeros_like(wide)
            for backend in (('cpu', 'cuda') if cuda.is_available() else ('cpu',)):
                pipe = a.Pipeline([a.Normalize('zscore', nonzero=False, eps=eps)], backend=backend)
                actual = pipe({'image': image}, seed=0)['image']
                np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    for invalid in (np.nan, np.inf, -np.inf):
        image = np.zeros((1, 9, 13), np.float32)
        image[0, 3, 2] = invalid
        with pytest.raises((ValueError, RuntimeError)):
            a.Pipeline([a.Normalize('zscore', nonzero=False)])({'image': image}, seed=0)


@pytest.mark.skipif(os.environ.get('VC_READER_LARGE_TESTS') != '1', reason='Requires a 2 GiB CPU array')
def test_zscore_large_constant():
    from vc_reader import augment as a
    # A nonbinary constant and odd count make rounding of an unsegmented
    # double sum observable as a false nonzero standard deviation.
    value = np.nextafter(np.float32(1), np.float32(0))
    image = np.full((1, 1, (1 << 29)+35), value, np.float32)
    pipe = a.Pipeline([a.Normalize('zscore', nonzero=False)], backend='cpu')
    actual = pipe({'image': image}, seed=0, inplace=True)['image']
    assert np.count_nonzero(actual) == 0
