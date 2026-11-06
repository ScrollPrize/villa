"""FDA/PDA reference targets: CPU/CUDA parity, replay, and device-resident batches."""
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


class FDAReferenceTests(unittest.TestCase):
    def backends(self):
        return ('cpu', 'cuda') if A._transforms.cuda_available() else ('cpu',)

    def transform(self, reference):
        return A.FDA(reference, blend=.3, fraction=.2, on='image')

    def pipeline(self, backend):
        return A.Pipeline([self.transform('style')],
            targets={'image': A.Image(valid=None), 'style': A.ScalarField()}, backend=backend, compact=False)

    def test_named_reference_matches_fixed_and_replays(self):
        rng = np.random.default_rng(7)
        for shape in ((2, 9, 7), (2, 8, 8, 8)):
            image = rng.normal(size=shape).astype(np.float32)
            style = rng.normal(size=(1, *shape[1:])).astype(np.float32)
            outputs = {}
            for backend in self.backends():
                with self.subTest(shape=shape, backend=backend):
                    dynamic = self.pipeline(backend)
                    fixed = A.Pipeline([self.transform(style)],
                        targets={'image': A.Image(valid=None)}, backend=backend, compact=False)
                    sample = dict(image=image, style=style)
                    result = dynamic(sample, seed=10)
                    np.testing.assert_allclose(result['image'], fixed({'image': image}, seed=10)['image'], atol=2e-6, rtol=1e-5)
                    np.testing.assert_array_equal(result['style'], style)
                    plan = pickle.loads(pickle.dumps(dynamic.sample(sample, seed=10)))
                    np.testing.assert_array_equal(plan.apply(sample)['image'], result['image'])
                    self.assertGreater(float(np.abs(result['image'] - image).max()), .01)
                    outputs[backend] = result['image']
            if 'cuda' in outputs:
                np.testing.assert_allclose(outputs['cpu'], outputs['cuda'], atol=2e-5, rtol=1e-5)

    def test_cuda_batch_uses_each_reference(self):
        if not A._transforms.cuda_available():
            self.skipTest('CUDA build/device required')
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch integration is optional')
        rng = np.random.default_rng(9)
        images = rng.normal(size=(3, 1, 8, 8, 8)).astype(np.float32)
        styles = rng.normal(size=images.shape).astype(np.float32) * np.array([1, 2, 3], dtype=np.float32)[:, None, None, None, None]
        pipeline = self.pipeline('cuda')
        gpu_image, gpu_style = torch.from_numpy(images).cuda(), torch.from_numpy(styles).cuda()
        result = pipeline.batch(dict(image=gpu_image, style=gpu_style), seeds=[1, 2, 3])
        self.assertTrue(result['image'].is_cuda)
        self.assertTrue(result['style'].is_cuda)
        expected = np.stack([self.pipeline('cpu')(dict(image=x, style=y), seed=i+1)['image']
                             for i, (x, y) in enumerate(zip(images, styles))])
        np.testing.assert_allclose(result['image'].cpu().numpy(), expected, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(gpu_style, result['style'], rtol=0, atol=0)
        torch.testing.assert_close(gpu_image.cpu(), torch.from_numpy(images), rtol=0, atol=0)

    def test_reference_validation(self):
        image = np.ones((1, 8, 8, 8), dtype=np.float32)
        for backend in self.backends():
            pipeline = self.pipeline(backend)
            with self.assertRaisesRegex(ValueError, 'missing'):
                pipeline({'image': image}, seed=1)
            with self.assertRaisesRegex(ValueError, 'spatial shape'):
                pipeline(dict(image=image, style=image[:, :4]), seed=1)
            with self.assertRaisesRegex(ValueError, 'channels'):
                pipeline(dict(image=image, style=np.repeat(image, 2, axis=0)), seed=1)


class PDAReferenceTests(FDAReferenceTests):
    def transform(self, reference):
        return A.PixelDistributionAdaptation(reference, blend=.3, mode='standard', on='image')


class PDAMinmaxReferenceTests(PDAReferenceTests):
    def transform(self, reference):
        return A.PixelDistributionAdaptation(reference, blend=.3, mode='minmax', on='image')


if __name__ == '__main__':
    unittest.main()
