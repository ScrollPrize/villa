"""Planar augmentation, spatial parameter validation and dimension semantics."""
import pickle
import unittest

import numpy as np

from build_support import load_build
load_build()

from vc_reader import transforms as A


def image_transforms(reference, **options):
    """Every image augmentation whose implementation gained planar execution."""
    return [
        A.BiasField(std=.2, scale=.15, **options),
        A.PlasmaBrightnessContrast(scale=.15, octaves=3, **options),
        A.PlasmaShadow(scale=.15, octaves=3, **options),
        A.Illumination(**options), A.Vignetting(**options), A.RandomToneCurve(**options),
        A.MultiplicativeNoise(**options), A.ShotNoise(**options),
        *(A.AdditiveNoise(distribution=d, **options) for d in ('uniform', 'gaussian', 'laplace', 'beta')),
        A.SaltAndPepper(amount=.2, **options), A.MotionBlur(length=3, **options),
        A.Defocus(radius=1.5, **options), A.ZoomBlur(samples=3, **options),
        A.Equalize(bins=16, **options), A.CLAHE(bins=16, tile_shape=(3, 4), **options),
        A.HistogramMatching(reference, bins=16, **options), A.PixelDistributionAdaptation(reference, **options),
        A.FDA(reference, **options), A.UnsharpMask(**options), A.GibbsRinging(**options),
        A.RingingOvershoot(radius=2, **options), A.ChannelMisalignment(**options),
        A.OrientedBlur(sigma_along=.7, sigma_across=.4, **options),
    ]


def bilinear(image, y, x):
    y = np.clip(y, 0, image.shape[-2]-1)
    x = np.clip(x, 0, image.shape[-1]-1)
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    y1, x1 = np.minimum(y0+1, image.shape[-2]-1), np.minimum(x0+1, image.shape[-1]-1)
    fy, fx = y-y0, x-x0
    return ((1-fy)*((1-fx)*image[..., y0, x0]+fx*image[..., y0, x1])+
            fy*((1-fx)*image[..., y1, x0]+fx*image[..., y1, x1]))


class DimensionTests(unittest.TestCase):
    def backends(self):
        return ('cpu', 'cuda') if A._transforms.cuda_available() else ('cpu',)

    def check_transform(self, transform, sample, targets=None, atol=6e-5):
        results = {}
        original = {k: v.copy() for k, v in sample.items()}
        for backend in self.backends():
            with self.subTest(transform=repr(transform), backend=backend,
                              shape=next(iter(sample.values())).shape):
                pipeline = A.Pipeline([transform], backend=backend, compact=False, targets=targets)
                output = pipeline(sample, seed=97)
                replay = pickle.loads(pickle.dumps(pipeline.sample(sample, seed=97))).apply(sample)
                for key, value in output.items():
                    self.assertTrue(np.isfinite(value).all())
                    self.assertEqual(value.dtype, sample[key].dtype)
                    np.testing.assert_array_equal(value, replay[key])
                for key in sample:
                    np.testing.assert_array_equal(sample[key], original[key])
                results[backend] = output
        if len(results) == 2:
            for key, value in results['cpu'].items():
                if value.dtype.kind == 'f':
                    np.testing.assert_allclose(results['cuda'][key], value, atol=atol, rtol=1e-5)
                else:
                    np.testing.assert_array_equal(results['cuda'][key], value)
        return results

    def test_planar_images_replay_channels_and_singletons(self):
        rng = np.random.default_rng(54)
        for shape in ((1, 1), (1, 9), (7, 1), (9, 11)):
            for dtype in (np.float32, np.float64):
                image = (.1+.8*rng.random((2, *shape))).astype(dtype)
                reference = (.1+.8*rng.random((1, *shape))).astype(dtype)
                for selected in ([], [1]):
                    for transform in image_transforms(reference, channels=selected):
                        results = self.check_transform(transform, {'image': image})
                        for output in results.values():
                            self.assertEqual(output['image'].shape, image.shape)
                            if selected:
                                np.testing.assert_array_equal(output['image'][0], image[0])

    def test_planar_defaults_and_masked_outputs(self):
        rng = np.random.default_rng(5)
        image = rng.random((1, 13, 17), dtype=np.float32)
        mask = np.zeros(image.shape, dtype=np.int64)
        mask[:, 2:6, 3:8] = (1 << 40)+1
        mask[:, 8:11, 9:14] = (1 << 40)+2
        valid = np.ones_like(mask, dtype=np.uint8)
        valid[:, 0] = 0
        sample = {'image': image, 'mask': mask, 'padding_mask': valid}
        targets = {'image': A.Image(fill=-3), 'mask': A.Label(), 'padding_mask': A.ValidMask()}
        transforms = [A.CLAHE(bins=16), A.GridDropout(), A.StripeMasking(),
                      A.ForegroundDropout(), A.ObjectDropout(),
                      A.GridDropout(unit_size=(3, 4), axes='yx', fill_mask=0),
                      A.StripeMasking(unit_size=(3, 4), axes='x', fill_mask=0),
                      A.ForegroundDropout(size=(3, 4), count=3, fill_mask=0)]
        for transform in transforms:
            results = self.check_transform(transform, sample, targets)
            for output in results.values():
                np.testing.assert_array_equal(output['image'][valid == 0], -3)
                np.testing.assert_array_equal(output['padding_mask'], valid)
                self.assertTrue(set(np.unique(output['mask'])).issubset(set(np.unique(mask))))
        for shape in ((5, 7), (19, 21)):
            for labels in (mask, np.zeros_like(mask)):
                results = self.check_transform(A.ForegroundCrop(shape), {'image': image, 'mask': labels})
                for output in results.values():
                    self.assertEqual(output['image'].shape, (1, *shape))
                    if labels.any():
                        self.assertTrue(output['mask'].any())
        # Face connectivity: diagonally adjacent objects do not merge.
        diagonal = np.eye(9, dtype=np.int64)[None]
        transform = A.ObjectDropout(drop_probability=.5)
        self.check_transform(transform, {'image': np.ones_like(diagonal, dtype=np.float32), 'mask': diagonal})

    def test_planar_filter_references(self):
        rng = np.random.default_rng(88)
        for dtype in (np.float32, np.float64):
            image = rng.random((1, 9, 11)).astype(dtype)
            y, x = np.indices(image.shape[1:], dtype=np.float64)
            radius = 1.5
            taps = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy*dy+dx*dx <= radius*radius]
            defocused = sum(image[:, np.clip(y.astype(int)+dy, 0, 8), np.clip(x.astype(int)+dx, 0, 10)]
                            for dy, dx in taps)/len(taps)
            direction = np.array([1., 2.])/np.sqrt(5)
            motion = sum(bilinear(image, y+direction[0]*u, x+direction[1]*u)
                         for u in np.linspace(-1.5, 1.5, 5))/5
            center = np.array([.3, .7])*np.array([8, 10])
            zoom = sum(bilinear(image, center[0]+(y-center[0])/(1+.2*j/4),
                               center[1]+(x-center[1])/(1+.2*j/4)) for j in range(5))/5
            cases = [(A.Defocus(radius), defocused), (A.MotionBlur(length=3, direction=(1, 2), samples=5), motion),
                     (A.ZoomBlur(strength=.2, center=(.3, .7), samples=5), zoom)]
            offsets = np.arange(-2, 3)
            weights = .6*np.sinc(.6*offsets)*(.5+.5*np.cos(np.pi*offsets/3))
            weights /= weights.sum()
            ringing = image.astype(np.float64)
            for direction in ((1, 0), (0, 1)):
                ringing = sum(w*bilinear(ringing, y+k*direction[0], x+k*direction[1])
                              for k, w in zip(offsets, weights))
            cases.append((A.RingingOvershoot(cutoff=.6, radius=2), ringing))
            for axis in ((1., 0.), (0., 1.), (1., 2.)):
                u = np.array(axis)/np.linalg.norm(axis)
                frame = (u, np.array([-u[1], u[0]]))
                oriented = image.astype(np.float64)
                for sigma, direction in zip((.7, .4), frame):
                    r = max(1, int(np.ceil(3*sigma)))
                    offsets = np.arange(-r, r+1)
                    weights = np.exp(-.5*(offsets/sigma)**2)
                    weights /= weights.sum()
                    oriented = sum(w*bilinear(oriented, y+k*direction[0], x+k*direction[1])
                                   for k, w in zip(offsets, weights))
                cases.append((A.OrientedBlur(.7, .4, axis=axis), oriented))
            for transform, reference in cases:
                for output in self.check_transform(transform, {'image': image}).values():
                    np.testing.assert_allclose(output['image'], reference, atol=2e-6, rtol=2e-6)

    def test_planar_frequency_and_intensity_references(self):
        rng = np.random.default_rng(19)
        image = rng.random((1, 9, 11), dtype=np.float32)
        reference = rng.random(image.shape, dtype=np.float32)
        spectrum, target = np.fft.fftn(image[0]), np.fft.fftn(reference[0])
        yy, xx = np.indices(image.shape[1:])
        frequency = np.maximum(np.minimum(yy, 9-yy)/(9/2), np.minimum(xx, 11-xx)/(11/2))
        truncated = spectrum.copy()
        truncated[frequency > .6] = 0
        adapted = spectrum.copy()
        selected = frequency <= .4
        amplitude = np.abs(spectrum)
        adapted[selected] *= (.7*amplitude[selected]+.3*np.abs(target[selected]))/amplitude[selected]
        y, x = yy/8, xx/10
        distance = (y-.3)**2+(x-.7)**2
        cases = [
            (A.GibbsRinging(cutoff=.6), np.fft.ifftn(truncated).real[None]),
            (A.FDA(reference, blend=.3, fraction=.4), np.fft.ifftn(adapted).real[None]),
            (A.Vignetting(strength=.4, radius=.8, center=(.3, .7)), image*(1-.4*np.clip(distance/.8**2, 0, 1))),
            (A.Illumination(strength=.3, direction=(1, 2)), image*(1+.3*((2*y-1)+2*(2*x-1))/3)),
            (A.Illumination(strength=.3, mode='corner', center=(0, 1)), image*(1+.3*(1-np.sqrt((y*y+(x-1)**2)/2)))),
            (A.Illumination(strength=.3, mode='gaussian', radius=.4, center=(.3, .7)), image*(1+.3*np.exp(-distance/(2*.4**2)))),
        ]
        for transform, expected in cases:
            for output in self.check_transform(transform, {'image': image}).values():
                np.testing.assert_allclose(output['image'], expected, atol=3e-6, rtol=3e-6)

    def test_dimension_validation(self):
        image = np.ones((1, 7, 9), dtype=np.float32)
        transforms = [A.MotionBlur(direction=(1, 0, 0)), A.ZoomBlur(center=(.5, .5, .5)),
                      A.Vignetting(center=(.5, .5, .5)), A.Illumination(direction=(1, 0, 0)),
                      A.OrientedBlur(axis=(1, 0, 0)), A.CLAHE(tile_shape=(3, 3, 3)),
                      A.GridDropout(unit_size=(3, 3, 3)), A.StripeMasking(axes='z'),
                      A.ForegroundDropout(size=(3, 3, 3)), A.ForegroundCrop((3, 3, 3)),
                      A.FDA(image[:, None])]
        for backend in self.backends():
            for transform in transforms:
                with self.subTest(transform=repr(transform), backend=backend):
                    with self.assertRaises(ValueError):
                        A.Pipeline([transform], backend=backend)({'image': image, 'mask': image.astype(np.int32)}, seed=2)

    def test_oriented_vector_target(self):
        image = np.random.default_rng(7).random((1, 11, 13), dtype=np.float32)
        for order, vector in (('yx', (1., 2.)), ('xy', (2., 1.))):
            sample = {'image': image, 'direction': np.array([vector], dtype=np.float32)}
            targets = {'image': A.Image(), 'direction': A.Vectors(components=order)}
            results = self.check_transform(A.OrientedBlur(.7, .4, axis='direction'), sample, targets)
            for backend, output in results.items():
                expected = A.Pipeline([A.OrientedBlur(.7, .4, axis=(1, 2))], backend=backend)(
                    {'image': image}, seed=97)
                np.testing.assert_allclose(output['image'], expected['image'], atol=2e-6, rtol=2e-6)

    def test_uint8_images_match_float32_values(self):
        # uint8 images are intensities 0..255: every path returns float32, bit-identical to float32 input.
        rng = np.random.default_rng(4)
        u8 = rng.integers(20, 200, (2, 10, 12, 14), dtype=np.uint8)
        f32 = u8.astype(np.float32)
        valid = (u8[:1] > 60).astype(np.uint8)
        reference = rng.random((1, 10, 12, 14)).astype(np.float32)
        for chain in ([A.MedianFilter(size=3)], [A.MedianFilter(size=5)], [A.GaussianBlur(sigma=1)], [A.Equalize(bins=32)],
                      [A.CLAHE(clip_limit=2, bins=32, tile_shape=(5, 6, 7))], [A.HistogramMatching(reference=reference, bins=32)],
                      [A.UnsharpMask(strength=.5, sigma=1)], [A.Gamma(gamma=.8), A.MedianFilter(size=3), A.Contrast(factor=1.2)],
                      [A.Flip(axes='x', p_axis=1)], []):
            for compact in (True, False):
                with self.subTest(chain=repr(chain), compact=compact):
                    p = A.Pipeline(chain, compact=compact)
                    for extra in ({}, {'padding_mask': valid}):
                        out = p({'image': u8, **extra}, seed=2)['image']
                        self.assertEqual(out.dtype, np.float32)
                        np.testing.assert_array_equal(out.view(np.uint32), p({'image': f32, **extra}, seed=2)['image'].view(np.uint32))
        for op in (A.MedianFilter(size=3), A.Equalize(bins=32), A.GaussianBlur(sigma=1), A.Normalize('minmax')):
            with self.subTest(batch=repr(op)):
                planes = u8[0][:, None]
                out = op.batch(planes)
                self.assertEqual(out.dtype, np.float32)
                np.testing.assert_array_equal(out.view(np.uint32), op.batch(planes.astype(np.float32)).view(np.uint32))

    def test_host_batches_match_single_calls(self):
        # Direct CPU .batch runs natively over (B, ...) and must equal per-sample calls bitwise.
        rng = np.random.default_rng(9)
        image = (rng.integers(0, 256, (4, 1, 9, 13)) / 255).astype(np.float32)
        image[1, 0, :, ::3] = 0
        image[1, 0, ::2, ::3] = -0.0  # both signed zeros: the general median selector
        mask = (image > .5).astype(np.uint8)
        for op, x in ((A.MedianFilter(size=3), image), (A.MedianFilter(size=5), image), (A.GaussianBlur(sigma=1), image),
                      (A.Equalize(bins=16), image), (A.CLAHE(bins=16, tile_shape=(3, 4)), image),
                      (A.Morphology(operation='close', radius=1, footprint='cross'), mask),
                      (A.Morphology(operation='dilate', radius=2, footprint='ball'), mask),
                      (A.DistanceTransform(mode='outside'), mask), (A.CropPad((5, 6)), image)):
            with self.subTest(transform=repr(op)):
                batch = op.batch(x)
                single = np.stack([op(v) for v in x])
                self.assertEqual(batch.dtype, single.dtype)
                np.testing.assert_array_equal(batch.view(np.uint8), single.view(np.uint8))
        rotated = A.Rotate90(planes=['yx'], turns=[1], resize=True).batch(image)
        self.assertEqual(rotated.shape, (4, 1, 13, 9))
        bad = image.copy(); bad[2, 0, 1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite input'):
            A.GaussianBlur(sigma=1).batch(bad)
        np.testing.assert_array_equal(image[1, 0, ::2, ::3].view(np.uint32), np.float32(-0.0).view(np.uint32))
        try:
            import torch
        except ImportError:
            return
        result = A.MedianFilter(size=3).batch(torch.from_numpy(image))
        self.assertIsInstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), A.MedianFilter(size=3).batch(image))


if __name__ == '__main__':
    unittest.main()
