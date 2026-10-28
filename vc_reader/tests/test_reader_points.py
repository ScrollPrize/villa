"""Point sampling precision across compact and dispersed coordinate batches."""
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


@pytest.mark.parametrize('order', [0, 1])
@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int64])
def test_dispersed_points(tmp_path, order, dtype):
    # Two small stored chunks separated by more than float32's integer range.
    far = 2**25
    values = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    (tmp_path/'.zarray').write_text(json.dumps(dict(zarr_format=2, shape=(4, 4, far+4),
        chunks=(4, 4, 4), dtype='|u1', compressor=None, filters=None, fill_value=99, order='C')))
    (tmp_path/'0.0.0').write_bytes(values.tobytes())
    (tmp_path/f'0.0.{far//4}').write_bytes((values+50).tobytes())
    points = np.array([[1e9, 0, 0], [1, 1, 1], [1.25, 1.5, far+1.25],
        [1.25, 1.5, 1.25], [1, 1, far+1.5], [-1e9, 0, 0],
        [1, 1, 1.5-1e-8], [1, 1, 1.5+1e-8], [-.25, 1, 1]], dtype=dtype)
    if np.issubdtype(dtype, np.floating):
        points = np.vstack([points, [np.nan, 0, 0], [0, np.inf, 0]])

    def voxel(at):
        z, y, x = at
        if not (0 <= z < 4 and 0 <= y < 4):
            return 99
        if 0 <= x < 4:
            return int(values[z, y, x])
        if far <= x < far+4:
            return int(values[z, y, x-far])+50
        return 99

    expected, valid = [], []
    for p in points.astype(np.float64):
        if not np.isfinite(p).all():
            expected.append(np.nan if order else 99)
            valid.append(False)
            continue
        low = np.floor(p).astype(np.int64) if order else np.rint(p).astype(np.int64)
        fraction = p-low if order else np.zeros(3)
        value, inside = 0., True
        for corner in itertools.product((0, 1), repeat=3):
            weight = np.prod(np.where(corner, fraction, 1-fraction))
            if weight:
                at = low+corner
                value += weight*voxel(at)
                inside &= bool(((at >= 0) & (at < [4, 4, far+4])).all())
        expected.append(value)
        valid.append(inside)
    expected, valid = np.asarray(expected), np.asarray(valid)
    with ZarrReader.open(tmp_path, threads=2) as reader:
        for permutation in (np.arange(len(points)), np.arange(len(points))[::-1],
                            np.random.default_rng(41).permutation(len(points)), np.array([2, 4])):
            batch = points[permutation]
            batch.setflags(write=False)
            np.testing.assert_allclose(reader.sample(batch, order=order), expected[permutation], atol=2e-5)
            if cuda.is_available():
                with reader.cuda() as device:
                    plan = device.prepare_points(batch, order=order)
                    for route in ('chunks', 'cpu', 'auto'):
                        for actual, mask in (device.sample(batch, order=order, source=route, valid=True),
                                             plan.read(source=route, valid=True)):
                            np.testing.assert_allclose(actual.numpy(), expected[permutation], atol=2e-5)
                            np.testing.assert_array_equal(mask.numpy(), valid[permutation])


@pytest.mark.parametrize('order', [0, 1])
def test_compact_empty_and_nonfinite_points(tmp_path, order):
    from test_source_storage import array
    values = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
    array(tmp_path, values, fill=99)
    batches = [np.empty((0, 3)), np.array([[np.nan, 0, 0], [0, np.inf, 0]]),
               np.array([[1, 1, 1], [2.5, 1, 1], [1.5, 1, 1]])]
    expected = [[], [np.nan, np.nan] if order else [99, 99], [21, 45 if order else 37, 29 if order else 37]]
    with ZarrReader.open(tmp_path) as reader:
        for points, wanted in zip(batches, expected):
            np.testing.assert_array_equal(reader.sample(points, order=order), wanted)
            if cuda.is_available():
                with reader.cuda() as device:
                    np.testing.assert_array_equal(device.sample(points, order=order, source='chunks').numpy(), wanted)
