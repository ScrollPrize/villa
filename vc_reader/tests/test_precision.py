"""Numerical contracts for float32 execution and justified wider arithmetic."""
import os
import io
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


def elastic_field(plan):
    """Read the field from a single Elastic VCA2 record for independent checks."""
    stream = io.BytesIO(plan._native.dump())
    assert stream.read(4) == b'VCA2'

    def number():
        return int.from_bytes(stream.read(8), 'little')

    def skip_vector(width):
        stream.seek(number()*width, 1)

    for _ in range(number()):  # Input names, shapes, kinds and item sizes.
        skip_vector(1)
        skip_vector(8)
        stream.read(16)
    assert number() == 1
    skip_vector(8)  # Transform path.
    assert number() == 0  # Elastic has no per-channel parameters.
    shape = tuple(number() for _ in range(number()))
    skip_vector(8)  # Matrix.
    skip_vector(8)  # Offset.
    return np.frombuffer(stream.read(number()*4), dtype='<f4').reshape(*shape, len(shape)).astype(np.float64)


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
            A.AdditiveNoise(distribution="gaussian"), A.AdditiveNoise(distribution="laplace"),
            A.AdditiveNoise(distribution="beta"),
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

    def test_elastic_field_strength_and_gradient(self):
        # Include tiny grids, singleton axes, noninteger knot spacing and a
        # volume large enough to exercise the capped CUDA reduction grid.
        for shape in ((1, 1), (1, 7), (7, 9), (1, 1, 1), (1, 7, 9),
                      (5, 1, 9), (9, 11, 13), (65, 67, 69)):
            sample = {'image': np.zeros((1, *shape), dtype=np.float32)}
            for sigma, magnitude in ((.5, .2), (2.3, 20), (12, 2)):
                fields = {}
                for backend in self.backends():
                    with self.subTest(shape=shape, sigma=sigma, magnitude=magnitude, backend=backend):
                        pipeline = A.Pipeline([A.Elastic(sigma, magnitude)], backend=backend)
                        plan = pipeline.sample(sample, seed=173)
                        field = elastic_field(plan)
                        self.assertTrue(np.isfinite(field).all())
                        np.testing.assert_allclose(field.mean(axis=tuple(range(len(shape)))), 0, atol=2e-5)
                        rms = np.sqrt(np.mean(np.sum(field*field, axis=-1)))
                        self.assertLessEqual(rms, magnitude+2e-5)
                        row_bounds = np.zeros(len(shape))
                        for axis, extent in enumerate(shape):
                            if extent > 1:
                                row_bounds += np.abs(np.diff(field, axis=axis)).reshape(-1, len(shape)).max(axis=0)
                        self.assertLessEqual(row_bounds.max(), .8+2e-5)
                        # At least one of the strength/gradient limits is active
                        # except for a single voxel, whose centered field is zero.
                        if np.prod(shape) > 1:
                            self.assertTrue(abs(rms-magnitude) < 2e-5 or abs(row_bounds.max()-.8) < 2e-5)
                        fields[backend] = field
                if len(fields) == 2:
                    np.testing.assert_allclose(fields['cuda'], fields['cpu'], atol=3e-5, rtol=1e-5)

    def test_geometry_multilinear_and_boundaries(self):
        for shape in ((7, 9), (5, 7, 9), (1, 7, 9), (5, 1, 9)):
            grid = np.indices(shape, dtype=np.float64)
            scale = np.maximum(np.array(shape)-1, 1).reshape(-1, *([1]*len(shape)))
            polynomial = np.prod(1+grid/scale, axis=0)
            image = np.stack([polynomial, -2*polynomial]).astype(np.float32)
            label = np.arange(np.prod(shape), dtype=np.int64).reshape(1, *shape)+(1 << 40)
            valid = np.ones((1, *shape), dtype=np.uint8)
            valid.reshape(-1)[::11] = 0
            for border in (False, True):
                for translation in (0, .5, -.375, 20):
                    outputs = {}
                    for backend in self.backends():
                        with self.subTest(shape=shape, border=border, translation=translation, backend=backend):
                            transform = A.Affine(translate=translation, border=border)
                            pipeline = A.Pipeline([transform], backend=backend,
                                                  targets={'image': A.Image(fill=-7, valid=None),
                                                           'mask': A.Label(fill=-9), 'padding_mask': A.ValidMask()})
                            sample = {'image': image, 'mask': label, 'padding_mask': valid}
                            out = pipeline(sample, seed=1)
                            replay = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=1))).apply(sample)
                            for key in sample:
                                np.testing.assert_array_equal(out[key], replay[key])
                            if border:
                                q = np.clip(grid-translation, 0, np.array(shape).reshape(scale.shape)-1)
                                expected = np.prod(1+q/scale, axis=0)
                                np.testing.assert_allclose(out['image'], np.stack([expected, -2*expected]), atol=2e-6, rtol=1e-6)
                            outputs[backend] = out
                    if len(outputs) == 2:
                        np.testing.assert_allclose(outputs['cuda']['image'], outputs['cpu']['image'], atol=2e-6, rtol=1e-6)
                        for key in ('mask', 'padding_mask'):
                            np.testing.assert_array_equal(outputs['cuda'][key], outputs['cpu'][key])

    def test_resize_exact_half_voxel_ties(self):
        # 128 -> 109 maps output x=54 to source 63.5 exactly. Floating
        # affine evaluation used to disagree across CPU/CUDA at that plane.
        image = np.broadcast_to((np.arange(128) % 2).astype(np.float32), (1, 3, 128)).copy()
        indices = ((2*np.arange(109)+1)*128)//(2*109)
        reduced = image[..., indices]
        q = np.clip((np.arange(128)+.5)*109/128-.5, 0, 108)
        lo = np.floor(q).astype(int)
        fraction = (q-lo).astype(np.float32)
        reference = reduced[..., lo]*(1-fraction)+reduced[..., np.minimum(lo+1,108)]*fraction
        for backend in self.backends():
            out = A.Pipeline([A.LowResolution(.85, axes='x')], backend=backend)({'image': image}, seed=0)['image']
            np.testing.assert_allclose(out, reference, atol=1e-5, rtol=0)

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
