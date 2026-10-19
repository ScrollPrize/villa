"""Batch independence and native CUDA stage coverage."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('VC_READER_NATIVE_DIR'):
    sys.path.insert(0, str(ROOT / 'benchmarks'))
    from bench_pipeline import load_extensions
    load_extensions(Path(os.environ['VC_READER_NATIVE_DIR']))

from vc_reader import augment as A, cuda


@unittest.skipUnless(torch.cuda.is_available() and A._native.cuda_available(), 'CUDA build and device required')
class NativeBatchTests(unittest.TestCase):
    def compare(self, transform, sample, *, targets=None, seeds=(3, 19, 107), params=None):
        pipeline = A.Pipeline([transform], targets=targets, backend='cuda', compact=False)
        source = {key: torch.from_numpy(value.copy()).cuda() for key, value in sample.items()}
        saved = {key: value.clone() for key, value in source.items()}
        result = pipeline.batch(source, seeds=seeds, params=params)
        for b, seed in enumerate(seeds):
            named = None if params is None else {k: values[b] for k, values in params.items()}
            single = pipeline({key: cuda.wrap(value[b]) for key, value in source.items()}, seed=seed, params=named)
            self.assertEqual(result.keys(), single.keys())
            for key, value in single.items():
                torch.testing.assert_close(result[key][b], torch.from_dlpack(value), atol=6e-5, rtol=1e-5)
        # Repartitioning a batch must not couple RNG or statistics across samples.
        tail_params = None if params is None else {k: values[1:] for k, values in params.items()}
        tail = pipeline.batch({key: value[1:] for key, value in source.items()}, seeds=seeds[1:], params=tail_params)
        for key in result:
            torch.testing.assert_close(result[key][1:], tail[key], atol=6e-5, rtol=1e-5)
        for key in source:
            torch.testing.assert_close(source[key], saved[key], atol=0, rtol=0)
        return result

    def test_pointwise_filters_geometry_and_statistics(self):
        from test_dimensions import image_transforms
        rng = np.random.default_rng(42)
        for shape in ((7, 11), (5, 7, 11)):
            image = rng.uniform(.1, .9, (3, 2, *shape)).astype(np.float32)
            reference = rng.uniform(.1, .9, (1, *shape)).astype(np.float32)
            transforms = image_transforms(reference)
            if len(shape) == 3:
                transforms = [t for t in transforms if t._constructor.__name__ != 'CLAHE']
                transforms.append(A.CLAHE(tile_shape=(3, 4, 5), bins=16))
            transforms += [
                A.BrightnessAdd((-.2, .2)), A.BrightnessMultiply((.8, 1.2)),
                A.Contrast(), A.Gamma(), A.Invert(), A.GaussianNoise(.03), A.RicianNoise(.03),
                A.GaussianBlur((.2, 1.2)), A.MedianFilter(3), A.MedianFilter(3, quantize=True),
                A.Sharpen(), A.LocalContrast(), A.LocalGamma(), A.LocalSmoothing(),
                A.BrightnessGradient(), A.BlankRectangle(size=3, value='median'),
                A.ClipOutliers(), A.SliceIllumination(axis='y'),
                A.Smear(axis='y', shift=(1,) if len(shape)==2 else (1, 0)),
                A.Flip(), A.Affine(degrees=7), A.Elastic(magnitude=1), A.LowResolution(.7),
                A.Rotate90(planes=['yx'], turns=[2]), A.CropPad(shape), A.SheetCompression(),
                A.Normalize('minmax'), A.Normalize('zscore'), A.Normalize('percentile'),
                A.GridDropout(), A.StripeMasking(), A.ForegroundDropout(), A.ObjectDropout(),
            ]
            for transform in transforms:
                with self.subTest(shape=shape, transform=repr(transform)):
                    self.compare(transform, {'image': image, 'mask': (image[:, :1]>.5).astype(np.int16)})

    def test_variable_parameters_and_skipped_stages(self):
        image = np.random.default_rng(9).random((3, 2, 9, 13), dtype=np.float32)
        self.compare(A.GaussianBlur(A.Param('sigma')), {'image': image}, params={'sigma': [0., .3, 2.]})
        self.compare(A.BrightnessAdd(.1, p=.5, p_channel=.5), {'image': image})
        self.compare(A.GaussianNoise(.1, p=.5, p_channel=.5), {'image': image})

    def test_data_dependent_selection_and_empty_foreground(self):
        image = np.random.default_rng(10).random((3, 1, 11, 13), dtype=np.float32)
        mask = np.zeros_like(image, dtype=np.int16)
        mask[1:, :, 1:3, 2:5] = 1
        mask[2, :, 7:9, 8:11] = 1
        self.compare(A.ForegroundCrop((7, 9)), {'image': image, 'mask': mask})
        channels = np.concatenate([mask, np.zeros_like(mask)], axis=1)
        result = self.compare(A.RemoveComponents(on='mask', p_fill=1), {'mask': channels})
        self.assertTrue(torch.all(result['mask'][0] == 0))
        yy, xx = np.indices((11, 13), dtype=np.float32)
        surface = np.stack([np.zeros_like(xx), yy, xx])
        surface = np.repeat(surface[None], 3, axis=0)
        surface[0] = 0
        self.compare(A.SurfacePerturb(), {'surface': surface}, targets={'surface': A.CoordinateField()})

    def test_derived_targets_and_multichannel_masks(self):
        image = np.random.default_rng(21).random((3, 1, 7, 9, 11), dtype=np.float32)
        mask = (image > .6).astype(np.int16)
        for transform in (A.OneHot([0, 1]), A.Regions([[1]]), A.ReplaceLabel(), A.MaskImage(), A.DistanceTransform(),
                          A.StructureTensor(), A.Morphology(), A.MedialSurface()):
            with self.subTest(transform=repr(transform)):
                self.compare(transform, {'image': image, 'mask': mask})

    def test_one_pointwise_launch_per_batch(self):
        pipeline = A.Pipeline([A.BrightnessAdd(.1)], backend='cuda')
        for count in (4, 16, 65):
            image = torch.zeros((count, 1, 17, 19), device='cuda')
            pipeline.batch({'image': image}, seeds=list(range(count)))
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                result = pipeline.batch({'image': image}, seeds=list(range(count)))
            kernels = [e for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA
                       and 'pointwise_kernel_launch' in e.name]
            self.assertEqual(len(kernels), 1, [e.name for e in kernels])
            copies = [e.name for e in profile.events() if e.device_type == torch.autograd.DeviceType.CUDA
                      and 'copy_bytes_launch' in e.name]
            self.assertEqual(copies, [])
            torch.testing.assert_close(result['image'], image+.1)

    def test_large_scans_and_signed_order_statistics(self):
        rng = np.random.default_rng(78)
        image = rng.normal(size=(3, 1, 257, 269)).astype(np.float32)
        mask = (image > 1).astype(np.int16)
        self.compare(A.ForegroundCrop((13, 17)), {'image': image, 'mask': mask})
        for transform in (A.Normalize('percentile'), A.Normalize('robust'),
                          A.ClipOutliers(), A.BlankRectangle(size=129, value='median')):
            with self.subTest(transform=repr(transform)):
                self.compare(transform, {'image': image})

    def test_failed_parameter_sampling_releases_pending_work(self):
        pipeline = A.Pipeline([A.GaussianBlur(A.Param('sigma'))], backend='cuda')
        image = torch.ones((3, 1, 11, 13), device='cuda')
        with self.assertRaises(ValueError):
            pipeline.batch({'image': image}, seeds=[1, 2, 3], params={'sigma': [.7, 1., -1.]})
        result = pipeline.batch({'image': image}, seeds=[1, 2, 3], params={'sigma': [.7, 1., 2.]})
        torch.testing.assert_close(result['image'], image)


if __name__ == '__main__':
    unittest.main()
