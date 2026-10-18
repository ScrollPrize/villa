"""Numerical contracts for float32 execution and justified wider arithmetic."""
import os
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

if os.environ.get('VC_READER_NATIVE_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from bench_pipeline import load_extensions
    load_extensions(Path(os.environ['VC_READER_NATIVE_DIR']))

from vc_reader import augment as A


class PrecisionTests(unittest.TestCase):
    def backends(self):
        return ('cpu', 'cuda') if A._native.cuda_available() else ('cpu',)

    def test_transforms_and_replay(self):
        rng = np.random.default_rng(124)
        reference = rng.random((1, 5, 7, 9), dtype=np.float32)
        transforms = [
            A.Elastic(sigma=3, magnitude=2, border=True), A.SheetCompression(strength=.2),
            A.Affine(degrees=7, scale=.9, border=True), A.LowResolution(.7), A.ThickSlice(.7),
            A.OrientedBlur(sigma_along=.7, sigma_across=.4, axis=(1, 2, 3)),
            A.BrightnessMultiply(1.2), A.BrightnessAdd(.1), A.Contrast(.8), A.Invert(),
            A.Gamma(.8), A.GaussianNoise(.1), A.RicianNoise(.1), A.ClipOutliers(),
            A.GaussianBlur(.7), A.MedianFilter(3), A.Sharpen(.1),
            A.BrightnessGradient(scale=3, loc=.5, strength=.2),
            A.LocalContrast(scale=3, loc=.5, factor=1.1),
            A.LocalGamma(scale=3, loc=.5, gamma=.8),
            A.LocalSmoothing(scale=3, loc=.5, strength=.5, sigma=.7),
            A.BlankRectangle(size=2, count=2, value='mean'),
            A.BlankRectangle(size=2, count=2, value='median'),
            A.SliceIllumination(), A.Smear(shift=(1, 0)),
            A.BiasField(), A.PlasmaBrightnessContrast(), A.PlasmaShadow(),
            A.Illumination(), A.Vignetting(), A.RandomToneCurve(),
            A.Equalize(bins=16), A.CLAHE(bins=16, tile_shape=(3, 4, 5)),
            A.HistogramMatching(reference, bins=16), A.PixelDistributionAdaptation(reference),
            A.FDA(reference), A.MultiplicativeNoise(), A.ShotNoise(), A.AdditiveNoise(),
            A.SaltAndPepper(), A.MotionBlur(), A.Defocus(radius=1), A.ZoomBlur(),
            A.UnsharpMask(), A.GibbsRinging(), A.RingingOvershoot(),
            A.GridDropout(unit_size=(3, 3, 3)), A.StripeMasking(unit_size=(3, 3, 3)),
            A.ForegroundDropout(size=(2, 2, 2)), A.ObjectDropout(),
        ]
        for dtype in (np.float32, np.float64):
            image = reference.astype(dtype)
            mask = rng.integers(0, 3, image.shape, dtype=np.int32)
            for transform in transforms:
                results = {}
                for backend in self.backends():
                    with self.subTest(transform=repr(transform), dtype=dtype, backend=backend):
                        pipeline = A.Pipeline([transform], backend=backend, compact=False)
                        sample = {'image': image, 'mask': mask}
                        out = pipeline(sample, seed=456)
                        self.assertEqual(out['image'].dtype, image.dtype)
                        self.assertTrue(np.isfinite(out['image']).all())
                        plan = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=456)))
                        replay = plan.apply(sample)
                        for key in sample:
                            self.assertEqual(out[key].tobytes(), replay[key].tobytes())
                        results[backend] = out['image']
                if len(results) == 2:
                    with self.subTest(transform=repr(transform), dtype=dtype, comparison='cpu/cuda'):
                        np.testing.assert_allclose(results['cuda'], results['cpu'], atol=5e-5, rtol=1e-5)

    def test_derived_fields(self):
        rng = np.random.default_rng(99)
        image = rng.random((1, 7, 9, 11), dtype=np.float32)
        mask = (image > .6).astype(np.uint8)
        transforms = [A.DistanceTransform(normals='_normal'),
                      A.StructureTensor(field='components'),
                      A.StructureTensor(field='normal', confidence='confidence'),
                      A.StructureTensor(at=(3, 4, 5))]
        for transform in transforms:
            results = {}
            for backend in self.backends():
                pipeline = A.Pipeline([transform], backend=backend, compact=False)
                sample = {'image': image, 'mask': mask}
                out = pipeline(sample, seed=17)
                replay = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=17))).apply(sample)
                for key in out:
                    np.testing.assert_array_equal(out[key], replay[key])
                    self.assertTrue(np.isfinite(out[key]).all())
                    if key not in sample:
                        self.assertEqual(out[key].dtype, np.float32)
                results[backend] = out
            if len(results) == 2:
                for key in results['cpu']:
                    np.testing.assert_allclose(results['cpu'][key], results['cuda'][key], atol=5e-5, rtol=1e-5)

    def test_large_coordinates_keep_subvoxel_translation(self):
        # At 4096, float32 cannot represent this translation. Wider coordinate
        # arithmetic is required even though the image/output are float32.
        image = np.broadcast_to((np.arange(8193) % 2).astype(np.float32), (1, 3, 8193)).copy()
        delta = 2.**-13
        for backend in self.backends():
            pipeline = A.Pipeline([A.Affine(translate=delta, border=True)], backend=backend)
            result = pipeline({'image': image}, seed=0)['image']
            self.assertEqual(result[0, 1, 4096], delta)

    def test_zscore_large_baseline_small_deviation(self):
        image = np.full((1, 17, 19), 1e6, dtype=np.float32)
        image.ravel()[::3] += .125
        reference = image.astype(np.float64)
        reference = (reference-reference.mean())/reference.std()
        for backend in self.backends():
            pipeline = A.Pipeline([A.Normalize('zscore', nonzero=False)], backend=backend)
            result = pipeline({'image': image}, seed=0)['image']
            np.testing.assert_allclose(result, reference, atol=2e-6, rtol=0)

    def test_extreme_float32_intermediates_remain_finite(self):
        image = np.full((1, 5, 7, 9), np.finfo(np.float32).max, dtype=np.float32)
        for backend in self.backends():
            for transform in (A.Affine(degrees=7, border=True), A.Elastic(sigma=3, magnitude=2, border=True),
                              A.LowResolution(.7), A.OrientedBlur(.7, .4, axis=(1, 2, 3)),
                              A.Defocus(radius=1), A.GibbsRinging(.8), A.Gamma(.8)):
                with self.subTest(backend=backend, transform=repr(transform)):
                    out = A.Pipeline([transform], backend=backend, compact=False)({'image': image}, seed=17)['image']
                    self.assertTrue(np.isfinite(out).all())
                    np.testing.assert_allclose(out, image, rtol=3e-6, atol=0)
            # Squaring this input would overflow float32. hypot avoids promotion.
            large = np.full_like(image, 1e30)
            out = A.Pipeline([A.RicianNoise(.1)], backend=backend)({'image': large}, seed=17)['image']
            np.testing.assert_array_equal(out, large)

    def test_explicit_float64_data_and_component_indices(self):
        image = np.full((1, 5, 7, 9), 1+2.**-40, dtype=np.float64)
        mask = np.zeros_like(image, dtype=np.uint8)
        mask[0, 1, 2, 3] = mask[0, 3, 4, 5] = 1
        for backend in self.backends():
            pipeline = A.Pipeline([A.BrightnessMultiply(1), A.RemoveComponents(on='mask', max_fraction=.5, p_channel=1, p_fill=0)], backend=backend)
            sample = {'image': image, 'mask': mask}
            result = pipeline(sample, seed=1)
            np.testing.assert_array_equal(result['image'], image)
            self.assertEqual(result['mask'].sum(), 1)
            replay = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=1))).apply(sample)
            np.testing.assert_array_equal(result['mask'], replay['mask'])


if __name__ == '__main__':
    unittest.main()
