"""Metal (Apple GPU) affine crops against the CPU sampler: rotations, source boundaries, absent chunks, evictions."""
import itertools
import json

import numpy as np
import pytest

from build_support import load_build
load_build()

from vc_reader.reader import ZarrReader
from vc_reader.reader import metal

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not metal.is_available(), reason='needs PyTorch with MPS')

FILL = 19
CHUNKS = (17, 19, 23)


@pytest.fixture
def volume(tmp_path):
    """A (63, 65, 67) uint8 zarr with edge chunks, one absent chunk and a non-zero fill value."""
    rng = np.random.default_rng(794)
    values = rng.integers(0, 256, (63, 65, 67), dtype=np.uint8)
    path = tmp_path/'ct'
    path.mkdir()
    (path/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=values.shape, chunks=CHUNKS, dtype='|u1',
                                                compressor=None, filters=None, fill_value=FILL, order='C')))
    for index in itertools.product(*(range((s+c-1)//c) for s, c in zip(values.shape, CHUNKS))):
        region = tuple(slice(i*c, min((i+1)*c, s)) for i, c, s in zip(index, CHUNKS, values.shape))
        if index == (1, 1, 1):
            values[region] = FILL  # an absent source chunk
            continue
        block = np.full(CHUNKS, FILL, np.uint8)
        part = values[region]
        block[tuple(slice(n) for n in part.shape)] = part
        (path/'.'.join(map(str, index))).write_bytes(block.tobytes())
    return path


GRIDS = [
    ((5.25, 7.5, 9.75), np.eye(3)),
    ((12.25, 9.5, 10.75), [[.9, -.3, .1], [.3, .9, -.2], [.1, .2, 1.]]),
    ((28.25, 31.5, 40.75), -np.eye(3)),
    ((-1.25, -2.5, -3.75), np.eye(3)),
    ((51.25, 53.5, 55.75), np.eye(3)),
    ((-100., 200., 300.), np.eye(3)),
    ((2., 3., 4.), np.zeros((3, 3))),
    ((0., 0., 0.), np.eye(3)*4),
]


def test_metal_matches_cpu(volume):
    shape = (24, 27, 29)
    origins = np.array([o for o, _ in GRIDS], np.float64)
    axes = np.array([np.asarray(a, np.float64) for _, a in GRIDS])
    with ZarrReader.open(volume) as reader:
        expected = np.stack(reader.read_affine_many([(shape, o, a) for o, a in zip(origins, axes)]))
        device = reader.metal(cache_bytes=64*int(np.prod(CHUNKS)))
        actual = device.read_affine_batch(shape, origins, axes).cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=6e-5, rtol=3e-7)


def test_metal_evicts_and_reuses(volume):
    """A cache of a few chunks serves crops across the array one batch at a time, evicting as it goes."""
    shape = (8, 9, 10)
    rng = np.random.default_rng(3)
    with ZarrReader.open(volume) as reader:
        device = reader.metal(cache_bytes=8*int(np.prod(CHUNKS)))
        out = torch.full((1, *shape), float('nan'), device='mps')
        for _ in range(40):
            origin = rng.uniform(-4, 60, (1, 3))
            angle = rng.uniform(0, 2*np.pi)
            axes = np.array([[[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]]])
            expected = reader.read_affine(shape, origin[0], axes[0])
            device.read_affine_batch(shape, origin, axes, out=out)
            np.testing.assert_allclose(out.cpu().numpy()[0], expected, atol=6e-5, rtol=3e-7)
        assert device.stats['uploads'] > device.slots  # chunks were evicted and uploaded again
        with pytest.raises(ValueError, match='touches'):
            device.read_affine_batch((60, 60, 60), np.zeros((1, 3)), np.eye(3)[None])
