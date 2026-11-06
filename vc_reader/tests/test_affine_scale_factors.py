"""Per-axis affine scales share voxel-center geometry on CPU and CUDA."""
import itertools
import os
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np
if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import transforms as A


def resize_reference(image, target):
    coords = [(np.arange(out, dtype=np.float32) + .5) * (size / out) - .5
              for size, out in zip(image.shape[1:], target)]
    coords = np.meshgrid(*(np.clip(q, 0, size - 1) for q, size in zip(coords, image.shape[1:])), indexing='ij')
    lower = [q.astype(np.int32) for q in coords]
    fraction = [q - lo.astype(np.float32) for q, lo in zip(coords, lower)]
    result = np.zeros((1, *target), dtype=np.float32)
    for corner in itertools.product((0, 1), repeat=len(target)):
        indices = tuple(np.minimum(lo + offset, size - 1) for lo, offset, size in zip(lower, corner, image.shape[1:]))
        weight = np.ones(target, dtype=np.float32)
        for f, offset in zip(fraction, corner):
            weight *= f if offset else 1 - f
        result += image[(slice(None), *indices)] * weight
    return result


class AffineScaleFactorsTests(unittest.TestCase):
    def backends(self):
        return ('cpu', 'cuda') if A._transforms.cuda_available() else ('cpu',)

    def test_resize_geometry_and_replay(self):
        rng = np.random.default_rng(123)
        for shape, target in (((7, 11), (12, 6)), ((9, 13, 17), (12, 8, 20))):
            raw = rng.random((1, *shape), dtype=np.float32)
            expected = resize_reference(raw, target)
            transform = A.Affine(shape=target, scale_factors=[b / a for a, b in zip(shape, target)], border=True)
            for backend in self.backends():
                with self.subTest(shape=shape, backend=backend):
                    pipeline = A.Pipeline([transform], backend=backend, compact=False)
                    result = pipeline({'image': raw}, seed=5)['image']
                    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-5)
                    plan = pickle.loads(pickle.dumps(pipeline.sample({'image': raw}, seed=5)))
                    np.testing.assert_array_equal(plan.apply({'image': raw})['image'], result)
                    if backend == 'cuda':
                        try:
                            import torch
                        except ImportError:
                            continue
                        batched = pipeline.batch({'image': torch.from_numpy(np.stack([raw, raw]))}, seeds=[5, 6])['image']
                        self.assertTrue(batched.is_cuda)
                        np.testing.assert_allclose(batched.cpu().numpy(), np.stack([expected, expected]), atol=2e-6, rtol=1e-5)

    def test_validation_and_composition(self):
        for factors in ((0, 1, 1), (-1, 1, 1), (float('nan'), 1, 1), (float('inf'), 1, 1)):
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                A.Pipeline([A.Affine(scale_factors=factors)])
        image = np.arange(7 * 9, dtype=np.float32).reshape(1, 7, 9)
        for backend in self.backends():
            with self.assertRaises(ValueError):
                A.Pipeline([A.Affine(scale_factors=[1, 2, 3])], backend=backend)({'image': image}, seed=0)
            result = A.Pipeline([A.Affine(scale=2, scale_factors=[.5, .5])], backend=backend)({'image': image}, seed=0)['image']
            np.testing.assert_array_equal(result, image)


if __name__ == '__main__':
    unittest.main()
