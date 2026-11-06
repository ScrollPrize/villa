"""Sparse metadata mapping matches native dense CPU/CUDA geometry."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import transforms as A


class PointMappingTests(unittest.TestCase):
    def test_matches_dense_geometry(self):
        for shape in ((7, 9), (5, 7, 9)):
            order = 'yx' if len(shape) == 2 else 'zyx'
            points = np.array([[1] * len(shape), [3] * len(shape)], dtype=np.float32)
            image = np.zeros((1, *shape), dtype=np.float32)
            operations = [A.Flip(p_axis=.5), A.Rotate90(planes=[order[-2:]], resize=True),
                          A.Transpose(order=order[::-1]), A.CropPad(shape=shape, random=True),
                          A.Affine(scale=(.8, 1.2), shape=shape)]
            for backend in ('cpu', 'cuda') if A._transforms.cuda_available() else ('cpu',):
                pipeline = A.Pipeline(operations, targets={'image': A.Image(valid=None),
                    'points': A.Points(order=order)}, backend=backend, compact=False)
                for seed in range(10):
                    result = pipeline.map_points({'points': points}, shape=shape, seed=seed)
                    expected = pipeline({'image': image, 'points': points}, seed=seed)
                    np.testing.assert_allclose(result['points'], expected['points'], atol=1e-6)
                np.testing.assert_array_equal(points, [[1] * len(shape), [3] * len(shape)])

    def test_shape_only_empty_and_outside(self):
        pipeline = A.Pipeline([A.Flip(axes='z', p_axis=1)], targets={'points': A.Points()})
        # An enormous shape must not allocate any voxels.
        points = np.array([[1, 2, 3]], dtype=np.float32)
        result = pipeline.map_points({'points': points}, shape=(100000,)*3, seed=0)
        np.testing.assert_array_equal(result['points'], [[99998, 2, 3]])
        empty = pipeline.map_points({'points': points[:0]}, shape=(8,)*3, seed=0)
        self.assertEqual(empty['points'].shape, (0, 3))
        drop = A.Pipeline([A.CropPad((2,)*3)], targets={'points': A.Points(outside='drop')})
        self.assertEqual(drop.map_points({'points': points}, shape=(8,)*3, seed=0)['points'].shape, (0, 3))

    def test_validation(self):
        points = np.zeros((2, 3), dtype=np.float32)
        pipeline = A.Pipeline([A.Gamma()], targets={'points': A.Points()})
        with self.assertRaisesRegex(ValueError, 'supports'):
            pipeline.map_points({'points': points}, shape=(8,)*3, seed=0)
        pipeline = A.Pipeline([], targets={'points': A.Points()})
        with self.assertRaisesRegex(ValueError, 'expected'):
            pipeline.map_points({'points': points[:, :2]}, shape=(8,)*3, seed=0)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            pipeline.map_points({'points': points}, shape=(0, 8, 8), seed=0)


if __name__ == '__main__':
    unittest.main()
