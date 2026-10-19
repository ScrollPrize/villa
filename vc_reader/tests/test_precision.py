"""Numerical contracts for float32 execution and justified wider arithmetic."""
import os
import io
import math
from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

if os.environ.get('VC_READER_BUILD_DIR'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from extension_build import load_extensions
    load_extensions(Path(os.environ['VC_READER_BUILD_DIR']))

from vc_reader import augment as A


def elastic_field(plan):
    """Read the field from a single Elastic VCA2 record for independent checks."""
    stream = io.BytesIO(plan._plan.dump())
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
        return ('cpu', 'cuda') if A._augment.cuda_available() else ('cpu',)

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

    def test_median_neighborhoods(self):
        rng = np.random.default_rng(2026)
        # Cross row, warp and block boundaries, including partially active warps.
        shapes = [(1, 1), (1, 33), (7, 1), (9, 31), (7, 32), (9, 33), (3, 257),
                  (1, 1, 1), (1, 7, 33), (5, 1, 33), (5, 7, 1), (3, 5, 31),
                  (3, 5, 32), (3, 5, 33), (7, 9, 65)]
        for shape in shapes:
            for dtype in (np.float32, np.float64):
                image = rng.normal(size=(2, *shape)).astype(dtype)
                for quantize in (False, True):
                    source = image.copy()
                    if quantize:
                        # An exact discrete palette should not be requantized.
                        source = np.round(source*4).astype(dtype)
                        source[source == 0] = 0  # One zero sign for the exact palette.
                    padded = np.pad(source, [(0, 0)]+[(1, 1)]*len(shape), mode='symmetric')
                    windows = np.lib.stride_tricks.sliding_window_view(
                        padded, (3,)*len(shape), axis=tuple(range(1, source.ndim)))
                    expected = np.median(windows, axis=tuple(range(-len(shape), 0)))
                    for channels in ([], [1]):
                        reference = expected.copy()
                        if channels:
                            reference[0] = source[0]
                        for backend in self.backends():
                            with self.subTest(shape=shape, dtype=dtype, quantize=quantize,
                                              channels=channels, backend=backend):
                                pipe = A.Pipeline([A.MedianFilter(3, quantize=quantize, channels=channels)],
                                                  backend=backend, compact=False)
                                sample = {'image': source.copy()}
                                actual = pipe(sample, seed=19)['image']
                                np.testing.assert_array_equal(actual, reference)
                                np.testing.assert_array_equal(sample['image'], source)
                                replay = pickle.loads(pickle.dumps(pipe.sample(sample, seed=19)))
                                np.testing.assert_array_equal(replay.apply(sample)['image'], actual)

        # Finite extremes and signed zeros must remain exact selected values.
        for dtype in (np.float32, np.float64):
            values = np.array([-np.finfo(dtype).max, -1, -0., 0., 1, np.finfo(dtype).max], dtype=dtype)
            image = np.resize(values, (1, 3, 5, 33))
            padded = np.pad(image, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='symmetric')
            windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3, 3), axis=(1, 2, 3))
            reference = np.median(windows, axis=(-3, -2, -1))
            valid = np.ones_like(image, dtype=np.uint8)
            valid[..., ::11] = 0
            reference[valid == 0] = -7
            for backend in self.backends():
                pipe = A.Pipeline([A.MedianFilter(3)], backend=backend, compact=False,
                                  targets={'image': A.Image(fill=-7), 'padding_mask': A.ValidMask()})
                actual = pipe({'image': image, 'padding_mask': valid}, seed=0)
                np.testing.assert_array_equal(actual['image'], reference)
                np.testing.assert_array_equal(actual['padding_mask'], valid)

    def test_median_general_and_quantized(self):
        rng = np.random.default_rng(54)
        for shape in ((5, 7), (3, 5, 7)):
            image = rng.normal(size=(1, *shape)).astype(np.float32)
            for size in (1, 2, 4, 5):
                # Even windows select the upper middle rank, not their average.
                padded = np.pad(image, [(0, 0)]+[(size//2, (size-1)//2)]*len(shape), mode='symmetric')
                windows = np.lib.stride_tricks.sliding_window_view(
                    padded, (size,)*len(shape), axis=tuple(range(1, image.ndim)))
                windows = windows.reshape(*image.shape, -1)
                rank = windows.shape[-1]//2
                reference = np.partition(windows, rank, axis=-1)[..., rank]
                for backend in self.backends():
                    pipe = A.Pipeline([A.MedianFilter(size)], backend=backend, compact=False)
                    np.testing.assert_array_equal(pipe({'image': image}, seed=0)['image'], reference)
        for dtype in (np.float32, np.float64):
            image = rng.random((1, 9, 11, 33)).astype(dtype)
            low, high = float(image.min()), float(image.max())
            code = np.floor((image.astype(np.float64)-low)/(high-low)*255+.5)
            quantized = (low+(high-low)*code/255).astype(dtype)
            padded = np.pad(quantized, ((0, 0), (1, 1), (1, 1), (1, 1)), mode='symmetric')
            windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3, 3), axis=(1, 2, 3))
            reference = np.median(windows, axis=(-3, -2, -1))
            for backend in self.backends():
                pipe = A.Pipeline([A.MedianFilter(3, quantize=True)], backend=backend, compact=False)
                np.testing.assert_allclose(pipe({'image': image}, seed=0)['image'], reference, atol=1e-7, rtol=1e-7)

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

    def test_low_resolution_composed_sampling(self):
        # Independent reference materializes the nearest-reduced image, then
        # reconstructs one axis at a time using NumPy linear interpolation.
        def reference(image, scale, axes):
            shape = image.shape[1:]
            names = 'zyx'[-len(shape):]
            reduced = tuple(max(1, round(n*scale)) if a in axes else n
                            for n, a in zip(shape, names))
            result = image.astype(np.float64)
            for a, (n, m) in enumerate(zip(shape, reduced), 1):
                result = np.take(result, ((2*np.arange(m)+1)*n)//(2*m), axis=a)
            for a, (n, m) in enumerate(zip(shape, reduced), 1):
                q = np.clip((np.arange(n)+.5)*m/n-.5, 0, m-1)
                lo = np.floor(q).astype(int)
                weights = (q-lo).reshape([n if j == a else 1 for j in range(image.ndim)])
                result = (np.take(result, lo, axis=a)*(1-weights)
                          + np.take(result, np.minimum(lo+1, m-1), axis=a)*weights)
            return result.astype(image.dtype)

        rng = np.random.default_rng(571)
        for shape in ((1, 1), (1, 19), (13, 17), (1, 1, 1), (1, 7, 11),
                      (7, 1, 11), (7, 11, 1), (9, 11, 13), (3, 5, 128)):
            for dtype in (np.float32, np.float64):
                image = rng.uniform(-2, 3, (2, *shape)).astype(dtype)
                original = image.copy()
                mask = (image > 0).astype(np.uint8)
                for scale, axes in ((.01, 'zyx'), (.5, 'zyx'), (.85, 'x'),
                                    (.7, 'zy'), (1., 'zyx')):
                    expected = reference(image, scale, axes)
                    for backend in self.backends():
                        for channels in ([], [1]):
                            with self.subTest(shape=shape, dtype=dtype, scale=scale,
                                              axes=axes, backend=backend, channels=channels):
                                pipe = A.Pipeline([A.LowResolution(scale, axes=axes, channels=channels)],
                                                  backend=backend, compact=False)
                                sample = {'image': image, 'mask': mask}
                                actual = pipe(sample, seed=5)
                                wanted = expected.copy()
                                if channels:
                                    wanted[0] = image[0]
                                np.testing.assert_allclose(actual['image'], wanted, atol=2e-6, rtol=2e-6)
                                np.testing.assert_array_equal(actual['mask'], mask)
                                np.testing.assert_array_equal(image, original)
                                replay = pickle.loads(pickle.dumps(pipe.sample(sample, seed=5))).apply(sample)
                                np.testing.assert_array_equal(actual['image'], replay['image'])

    def test_low_resolution_extremes_and_validity(self):
        for dtype in (np.float32, np.float64):
            limit = np.finfo(dtype).max
            image = np.full((1, 5, 7, 9), limit, dtype=dtype)
            image[..., ::2] = -limit
            valid = np.ones(image.shape, dtype=np.uint8)
            valid[..., 0] = 0
            for backend in self.backends():
                pipe = A.Pipeline([A.LowResolution(.7)], backend=backend, compact=False)
                out = pipe({'image': image, 'padding_mask': valid}, seed=0)
                self.assertTrue(np.isfinite(out['image']).all())
                self.assertTrue((np.abs(out['image']) <= limit).all())
                np.testing.assert_array_equal(out['image'][valid == 0], 0)
                np.testing.assert_array_equal(out['padding_mask'], valid)
                # A scale that rounds back to the input shape must preserve bits.
                identity = A.Pipeline([A.LowResolution(.999)], backend=backend, compact=False)
                zeros = np.zeros_like(image)
                zeros[..., ::2] = -0.
                result = identity({'image': zeros}, seed=0)['image']
                self.assertEqual(result.tobytes(), zeros.tobytes())

    def test_gaussian_tiles_and_boundaries(self):
        # Independent separable reference: partial CUDA tiles, multiple rows,
        # singleton axes, identity taps, and radii on either side of the tiled
        # dispatch limit. No optional SciPy dependency is needed.
        rng = np.random.default_rng(418)
        for shape in ((1, 1), (9, 35), (2, 9, 257), (3, 1, 7)):
            for dtype in (np.float32, np.float64):
                image = rng.standard_normal((2, *shape)).astype(dtype)
                for sigma in (0, .1, .7, 10.75, 11):
                    extent = sigma * 6 + .5
                    size = round(extent)
                    if size % 2 == 0:
                        size += 1 if extent >= size else -1
                    radius = max(0, (size - 1) // 2)
                    weights = np.exp(-.5 * (np.arange(-radius, radius+1) / sigma)**2) if sigma else np.ones(1)
                    weights = (weights / weights.sum()).astype(dtype)
                    for edge in ('mirror', 'reflect', 'nearest'):
                        expected = image.copy()
                        for axis, n in enumerate(shape, start=1):
                            result = np.zeros_like(expected)
                            for j, weight in enumerate(weights):
                                indices = np.arange(n) + j - radius
                                if edge == 'nearest' or n == 1:
                                    indices = np.clip(indices, 0, n-1)
                                else:
                                    period = 2 * (n-1) if edge == 'mirror' else 2*n
                                    indices %= period
                                    indices = np.where(indices < n, indices, period-indices-(edge == 'reflect'))
                                result += weight * np.take(expected, indices, axis=axis)
                            expected = result
                        for backend in self.backends():
                            with self.subTest(shape=shape, dtype=dtype, sigma=sigma, edge=edge, backend=backend):
                                pipeline = A.Pipeline([A.GaussianBlur(sigma, boundary=edge)], backend=backend, compact=False)
                                out = pipeline({'image': image}, seed=31)['image']
                                tolerance = 2e-6 if dtype == np.float32 else 1e-13
                                np.testing.assert_allclose(out, expected, atol=tolerance, rtol=tolerance)

    def test_separate_blur_output(self):
        # Local smoothing retains the source for blending. Exercise no passes,
        # even/odd pass counts, and shared-memory/general convolution paths.
        rng = np.random.default_rng(419)
        for shape in ((9, 35), (3, 9, 35)):
            for dtype in (np.float32, np.float64):
                image = rng.random((2, *shape)).astype(dtype)
                original = image.copy()
                for sigma in (0, .01, .7, 8.25):
                    transforms = [A.LocalSmoothing(scale=3, loc=.5, strength=.7, sigma=sigma)]
                    if len(shape) == 3 and sigma > 0:
                        transforms.append(A.UnsharpMask(sigma=sigma))
                    for transform in transforms:
                        expected = A.Pipeline([transform], backend='cpu', compact=False)({'image': image}, seed=41)['image']
                        for backend in self.backends():
                            with self.subTest(shape=shape, dtype=dtype, transform=repr(transform), backend=backend):
                                pipeline = A.Pipeline([transform], backend=backend, compact=False)
                                out = pipeline({'image': image}, seed=41)['image']
                                np.testing.assert_allclose(out, expected, atol=2e-6, rtol=2e-6)
                                np.testing.assert_array_equal(image, original)

    def test_separable_local_fields(self):
        # Independent dense construction checks the product, normalization and
        # centering semantics, including narrow and almost constant fields.
        rng = np.random.default_rng(431)
        for shape in ((1, 1), (1, 9), (7, 35), (1, 7, 9), (3, 5, 257)):
            for scale, loc in ((.2, .5), (3., .5), (1000., .5), (1., -10.)):
                axes = []
                for n in shape:
                    axis = np.array([.5 * (math.erf((x+.5-loc*n)/(scale*math.sqrt(2))) -
                                          math.erf((x-.5-loc*n)/(scale*math.sqrt(2))))
                                     for x in range(n)], dtype=np.float32)
                    if axis.max() > 0:
                        axis /= axis.max()
                    axes.append(axis)
                weight = axes[0]
                for axis in axes[1:]:
                    weight = np.multiply.outer(weight, axis)
                span = weight.max() - weight.min()
                weight = (weight-weight.min())/span if span else np.zeros(shape, dtype=np.float32)
                centered = weight - np.float32(weight.mean(dtype=np.float64))
                if np.abs(centered).max() > 1e-8:
                    centered /= np.abs(centered).max()
                for dtype in (np.float32, np.float64):
                    image = rng.random((2, *shape)).astype(dtype)*3-1
                    sample = {'image': image}
                    original = image.copy()
                    low = image.min(axis=tuple(range(1, image.ndim)), keepdims=True)
                    high = image.max(axis=tuple(range(1, image.ndim)), keepdims=True)
                    w = weight.astype(dtype)
                    mean = (image.astype(np.float64)*w).sum(axis=tuple(range(1, image.ndim)), keepdims=True)/(w.sum(dtype=np.float64)+1e-8)
                    mean = mean.astype(dtype)
                    norm = (image-low)/np.maximum(high-low, dtype(1e-8))
                    transforms = [
                        (A.BrightnessGradient(scale, loc=loc, strength=.3), image+centered*dtype(.3)),
                        (A.BrightnessGradient(scale, loc=loc, strength=.3, mean_centered=False, clip=True),
                         np.clip(image+w*dtype(.3), low, high)),
                        (A.LocalContrast(scale, loc=loc, factor=1.3), image*(1-w)+((image-mean)*dtype(1.3)+mean)*w),
                        (A.LocalGamma(scale, loc=loc, gamma=.7), (norm*(1-w)+norm**dtype(.7)*w)*np.maximum(high-low, dtype(1e-8))+low),
                    ]
                    taps = np.exp(-.5*(np.arange(-3, 4)/.7)**2)
                    taps = (taps/taps.sum()).astype(dtype)
                    smooth = image.copy()
                    for axis, n in enumerate(shape, start=1):
                        filtered = np.zeros_like(smooth)
                        for j, tap in enumerate(taps):
                            indices = (np.arange(n)+j-3) % (2*n)
                            indices = np.where(indices < n, indices, 2*n-indices-1)
                            filtered += tap*np.take(smooth, indices, axis=axis)
                        smooth = filtered
                    blend = w*dtype(.6)
                    transforms.append((A.LocalSmoothing(scale, loc=loc, strength=.6, sigma=.7),
                                       image*(1-blend)+smooth*blend))
                    for transform, expected in transforms:
                        for backend in self.backends():
                            with self.subTest(shape=shape, scale=scale, loc=loc, dtype=dtype,
                                              transform=repr(transform), backend=backend):
                                pipeline = A.Pipeline([transform], backend=backend, compact=False)
                                actual = pipeline(sample, seed=8)['image']
                                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=2e-6)
                                replay = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=8)))
                                np.testing.assert_array_equal(replay.apply(sample)['image'], actual)
                                np.testing.assert_array_equal(image, original)

    def test_local_plan_storage_scales_with_axes(self):
        sample = {'image': np.zeros((1, 128, 128, 128), dtype=np.float32)}
        for transform in (A.BrightnessGradient(12), A.LocalContrast(12),
                          A.LocalGamma(12), A.LocalSmoothing(12)):
            for backend in self.backends():
                with self.subTest(transform=repr(transform), backend=backend):
                    pipeline = A.Pipeline([transform], backend=backend, compact=False)
                    plan = pipeline.sample(sample, seed=0)
                    self.assertLess(len(plan._plan.dump()), 2048)

    def test_zscore_large_baseline_small_deviation(self):
        image = np.full((1, 17, 19), 1e6, dtype=np.float32)
        image.ravel()[::3] += .125
        reference = image.astype(np.float64)
        reference = (reference-reference.mean())/reference.std()
        for backend in self.backends():
            pipeline = A.Pipeline([A.Normalize('zscore', nonzero=False)], backend=backend)
            result = pipeline({'image': image}, seed=0)['image']
            np.testing.assert_allclose(result, reference, atol=2e-6, rtol=0)

    def test_gamma_and_sharpen_statistics(self):
        rng = np.random.default_rng(917)
        for shape in ((1, 1), (7, 37), (3, 7, 37)):
            for dtype in (np.float32, np.float64):
                image = rng.uniform(-3, 5, (2, *shape)).astype(dtype)
                original = image.copy()
                cases = []
                for invert in (False, True):
                    for retain in (False, True):
                        expected = image.astype(np.float64).copy()
                        for channel in expected:
                            before = -channel.copy() if invert else channel.copy()
                            low, high = before.min(), before.max()
                            result = ((before-low)/max(high-low, 1e-7))**.8*(high-low)+low
                            if retain:
                                sd = lambda a: a.std(ddof=1) if a.size > 1 else 0.
                                result = (result-result.mean())*(sd(before)/max(sd(result), 1e-7))+before.mean()
                            channel[:] = -result if invert else result
                        cases.append((A.Gamma(.8, p_invert=int(invert), p_retain_stats=int(retain)), expected))
                for clip in (False, True):
                    source = image.astype(np.float64)
                    laplace = 2*len(shape)*source
                    for axis, n in enumerate(shape, 1):
                        laplace -= np.take(source, np.maximum(np.arange(n)-1, 0), axis=axis)
                        laplace -= np.take(source, np.minimum(np.arange(n)+1, n-1), axis=axis)
                    expected = source+.25*laplace
                    if clip:
                        axes = tuple(range(1, image.ndim))
                        expected = np.clip(expected, source.min(axis=axes, keepdims=True),
                                           source.max(axis=axes, keepdims=True))
                    cases.append((A.Sharpen(.25, p_clamp=int(clip)), expected))
                for transform, expected in cases:
                    for backend in self.backends():
                        with self.subTest(shape=shape, dtype=dtype, transform=repr(transform), backend=backend):
                            pipeline = A.Pipeline([transform], backend=backend, compact=False)
                            output = pipeline({'image': image}, seed=0)['image']
                            np.testing.assert_allclose(output, expected, rtol=2e-6, atol=3e-6)
                            plan = pickle.loads(pickle.dumps(pipeline.sample({'image': image}, seed=0)))
                            np.testing.assert_array_equal(plan.apply({'image': image})['image'], output)
                            np.testing.assert_array_equal(image, original)

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
