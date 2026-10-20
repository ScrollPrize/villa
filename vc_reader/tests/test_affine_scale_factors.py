"""Per-axis affine scales share voxel-center geometry on CPU and CUDA."""
import pickle
import unittest

import numpy as np
from vc_reader import transforms as A


class AffineScaleFactorsTests(unittest.TestCase):
    def test_resize_geometry_and_replay(self):
        import torch
        from torch.nn.functional import interpolate
        rng = np.random.default_rng(123)
        for shape, target in (((7, 11), (12, 6)), ((9, 13, 17), (12, 8, 20))):
            raw = rng.random((1, *shape), dtype=np.float32)
            expected = interpolate(torch.from_numpy(raw)[None], size=target,
                mode='bilinear' if len(shape) == 2 else 'trilinear', align_corners=False)[0].numpy()
            transform = A.Affine(shape=target, scale_factors=[b / a for a, b in zip(shape, target)], border=True)
            for backend in ('cpu', 'cuda'):
                with self.subTest(shape=shape, backend=backend):
                    pipeline = A.Pipeline([transform], backend=backend, compact=False)
                    result = pipeline({'image': raw}, seed=5)['image']
                    np.testing.assert_allclose(result, expected, atol=2e-6, rtol=1e-5)
                    plan = pickle.loads(pickle.dumps(pipeline.sample({'image': raw}, seed=5)))
                    np.testing.assert_array_equal(plan.apply({'image': raw})['image'], result)
                    if backend == 'cuda':
                        batched = pipeline.batch({'image': torch.from_numpy(np.stack([raw, raw]))}, seeds=[5, 6])['image']
                        self.assertTrue(batched.is_cuda)
                        np.testing.assert_allclose(batched.cpu().numpy(), np.stack([expected, expected]), atol=2e-6, rtol=1e-5)

    def test_validation_and_composition(self):
        for factors in ((0, 1, 1), (-1, 1, 1), (float('nan'), 1, 1), (float('inf'), 1, 1)):
            with self.subTest(factors=factors), self.assertRaises(ValueError):
                A.Pipeline([A.Affine(scale_factors=factors)])
        image = np.arange(7 * 9, dtype=np.float32).reshape(1, 7, 9)
        for backend in ('cpu', 'cuda'):
            with self.assertRaises(ValueError):
                A.Pipeline([A.Affine(scale_factors=[1, 2, 3])], backend=backend)({'image': image}, seed=0)
            result = A.Pipeline([A.Affine(scale=2, scale_factors=[.5, .5])], backend=backend)({'image': image}, seed=0)['image']
            np.testing.assert_array_equal(result, image)


if __name__ == '__main__':
    unittest.main()
