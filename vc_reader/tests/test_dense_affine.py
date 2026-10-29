"""Affine CT sampling across dense boxes, sparse grids and source boundaries."""
import itertools
import json

import numpy as np
import pytest

from build_support import load_build
load_build()

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
