"""Run with unittest; VC_READER_NATIVE_DIR selects an existing native build."""
import gc
import os
from pathlib import Path
import subprocess
import sys
import unittest

if os.environ.get("VC_READER_NATIVE_DIR"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from bench_pipeline import load_extensions
    load_extensions(Path(os.environ["VC_READER_NATIVE_DIR"]))

import numpy as np
import torch
from vc_reader import augment as A, cuda
from vc_reader.torch import CudaLoader


class Samples(torch.utils.data.Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        image = np.arange(24, dtype=np.uint8).reshape(1, 2, 3, 4) + index
        return {"image": image, "label": image.astype(np.int16), "seed": index + 9, "id": index}


class WorkerMisuse(torch.utils.data.Dataset):
    def __len__(self): return 1

    def __getitem__(self, index):
        from vc_reader.torch import _main_process
        _main_process()


def image_batch(count=3):
    return torch.arange(count * 24, dtype=torch.uint8).reshape(count, 1, 2, 3, 4)


class ConfigurationTests(unittest.TestCase):
    def test_cpu_pipeline_rejected(self):
        pipeline = A.Pipeline([])
        with self.assertRaisesRegex(ValueError, "CUDA pipeline"):
            pipeline.batch({"image": image_batch()}, seeds=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, "CUDA pipeline"):
            CudaLoader([], pipeline)

    def test_import_does_not_require_torch(self):
        # The catalog and ordinary augmentation APIs remain independent of Torch.
        code = """
import sys, os
if os.environ.get('VC_READER_NATIVE_DIR'):
    sys.path.insert(0, 'benchmarks')
    from pathlib import Path
    from bench_pipeline import load_extensions
    load_extensions(Path(os.environ['VC_READER_NATIVE_DIR']))
import vc_reader
from vc_reader import augment
assert 'torch' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, timeout=30,
                       cwd=Path(__file__).resolve().parents[1])


@unittest.skipUnless(torch.cuda.is_available() and A._native.cuda_available(), "CUDA build and device required")
class BatchTests(unittest.TestCase):
    def assertSame(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_samples_labels_metadata_and_inputs(self):
        pipeline = A.Pipeline([A.Normalize('minmax'), A.Flip(axes='x', p_axis=1), A.BrightnessMultiply((.5, 1.5))],
                              targets={'image': A.Image(), 'label': A.Label()}, backend='cuda')
        x = image_batch()
        label = x.to(torch.int64) + (1 << 40)
        ids = torch.tensor([3, 4, 5])
        batch = {'image': x, 'label': label, 'id': ids, 'note': 'metadata'}
        result = pipeline.batch(batch, seeds=[11, 12, 13])
        self.assertIs(result['id'], ids)
        self.assertEqual(result['note'], 'metadata')
        self.assertSame(x, image_batch())
        self.assertEqual(result['label'].dtype, torch.int64)
        for i, seed in enumerate([11, 12, 13]):
            ref = pipeline({'image': cuda.asarray(x[i].numpy()).astype(np.float32),
                            'label': cuda.asarray(label[i].numpy())}, seed=seed)
            self.assertSame(result['image'][i], torch.from_dlpack(ref['image']))
            self.assertSame(result['label'][i], torch.from_dlpack(ref['label']))

    def test_resident_noncontiguous_and_float64(self):
        pipeline = A.Pipeline([A.Flip(axes='x', p_axis=1)], backend='cuda')
        x = image_batch().to('cuda', dtype=torch.float64).transpose(-1, -2)
        result = pipeline.batch({'image': x}, seeds=[1, 2, 3])['image']
        self.assertEqual(result.dtype, torch.float64)
        self.assertSame(result, x.flip(-1))
        native = cuda.asarray(image_batch().numpy())
        self.assertSame(pipeline.batch({'image': native}, seeds=[1, 2, 3])['image'],
                        image_batch().to('cuda', dtype=torch.float32).flip(-1))

    def test_changed_shape_generated_target_and_removed_source(self):
        p = A.Pipeline([A.CropPad((3, 4, 5)), A.OneHot([0, 1], on='mask', remove_source=True)], backend='cuda')
        x = image_batch().numpy()
        batch = {'image': x, 'mask': (x % 2).astype(np.int16)}
        result = p.batch(batch, seeds=[1, 2, 3])
        self.assertNotIn('mask', result)
        self.assertEqual(result['image'].shape, (3, 1, 3, 4, 5))
        self.assertEqual(result['onehot'].shape, (3, 2, 3, 4, 5))
        self.assertIsInstance(result['onehot'], torch.Tensor)

    def test_ragged_and_pyramid_outputs(self):
        x = image_batch(12).to('cuda', dtype=torch.float32)
        p = A.Pipeline([A.Rotate90(planes=['yx'], turns=[0, 1], resize=True)], backend='cuda')
        result = p.batch({'image': x}, seeds=list(range(12)))['image']
        self.assertIsInstance(result, list)
        self.assertGreater(len({tuple(v.shape) for v in result}), 1)
        for i in range(12):
            ref = p({'image': cuda.wrap(x[i])}, seed=i)['image']
            self.assertSame(result[i], torch.from_dlpack(ref))
        p = A.Pipeline([A.DeepSupervision([1., .5])], backend='cuda')
        out = p.batch({'image': x, 'mask': x.to(torch.int16)}, seeds=list(range(12)))
        self.assertEqual(len(out['mask']), 2)
        self.assertEqual(out['mask'][0].shape, x.shape)
        self.assertTrue(all(isinstance(level, torch.Tensor) and level.is_cuda for level in out['mask']))

    def test_errors_before_consumption_and_recovery(self):
        p = A.Pipeline([A.Normalize('minmax')], backend='cuda')
        bad = image_batch().float()
        bad[0, 0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex((ValueError, RuntimeError), 'nonfinite'):
            p.batch({'image': bad}, seeds=[1, 2, 3])
        with self.assertRaisesRegex((ValueError, RuntimeError), 'zero or one'):
            p.batch({'image': image_batch(), 'padding_mask': torch.full_like(image_batch(), 2)}, seeds=[1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'one integer per sample'):
            p.batch({'image': image_batch()}, seeds=[1])
        with self.assertRaisesRegex(ValueError, 'unsigned 64-bit'):
            p.batch({'image': image_batch()}, seeds=[-1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'autograd'):
            p.batch({'image': image_batch().float().requires_grad_()}, seeds=[1, 2, 3])
        with self.assertRaisesRegex(TypeError, 'float32 or float64'):
            p.batch({'image': image_batch().half()}, seeds=[1, 2, 3])
        self.assertTrue(torch.isfinite(p.batch({'image': image_batch()}, seeds=[1, 2, 3])['image']).all())

    def test_streams_and_training_backward(self):
        p = A.Pipeline([A.Flip(axes='x', p_axis=1)], backend='cuda')
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = image_batch().to('cuda', dtype=torch.float32) + 1
            y = p.batch({'image': x}, seeds=[1, 2, 3])['image']
            expected = x.flip(-1)
            weight = torch.ones((), device='cuda', requires_grad=True)
            loss = (y * weight).sum()
            loss.backward()
        stream.synchronize()
        self.assertSame(y, expected)
        self.assertSame(weight.grad, expected.sum())
        # A batch call must not leave vc_reader using a destroyed side stream.
        del stream
        gc.collect()
        ref = p({'image': cuda.asarray(np.ones((1, 2, 3, 4), np.float32))}, seed=1)
        self.assertTrue(np.all(ref['image'].numpy() == 1))

    def test_low_resolution_stream_and_output_ownership(self):
        transforms = [A.LowResolution(.7), A.BrightnessAdd(.25), A.LowResolution(.85)]
        cpu = A.Pipeline(transforms, compact=False)
        gpu = A.Pipeline(transforms, backend='cuda', compact=False)
        for channels in (1, 2):
            for dtype in (torch.float32, torch.float64):
                host = torch.arange(3*channels*5*7*9, dtype=dtype).reshape(3, channels, 5, 7, 9)/100
                expected = torch.from_numpy(np.stack([
                    cpu({'image': image.numpy()}, seed=i)['image'] for i, image in enumerate(host)]))
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    source = host.to('cuda')
                    first = gpu.batch({'image': source}, seeds=[0, 1, 2])['image']
                    # Reuse the pool while retaining the first result.
                    second = gpu.batch({'image': source}, seeds=[0, 1, 2])['image']
                    weight = torch.ones((), device='cuda', requires_grad=True)
                    (first*weight).sum().backward()
                stream.synchronize()
                torch.testing.assert_close(first.cpu(), expected, rtol=2e-6, atol=2e-6)
                self.assertSame(first, second)
                self.assertSame(source.cpu(), host)
                self.assertSame(weight.grad, first.sum().to(weight.dtype))

    def test_full_uint64_seeds_after_default_collation(self):
        seeds = [(1 << 64) - 1, 1 << 63, 7]
        p = A.Pipeline([A.BrightnessMultiply((.5, 1.5))], backend='cuda')
        batch = torch.utils.data.default_collate([
            {'image': x, 'seed': np.asarray(seed, dtype=np.uint64)}
            for x, seed in zip(image_batch(), seeds)])
        expected = p.batch(batch, seeds=seeds)['image']
        self.assertSame(p.batch(batch, seeds=batch['seed'])['image'], expected)
        self.assertSame(next(iter(CudaLoader([batch], p, seed_key='seed')))['image'], expected)


@unittest.skipUnless(torch.cuda.is_available() and A._native.cuda_available(), "CUDA build and device required")
class LoaderTests(unittest.TestCase):
    def pipeline(self):
        return A.Pipeline([A.Flip(axes='x', p_axis=.5), A.BrightnessMultiply((.5, 1.5))],
                          targets={'image': A.Image(), 'label': A.Label()}, backend='cuda')

    def test_spawn_workers_partial_batch_and_explicit_seeds(self):
        p = self.pipeline()
        cpu = torch.utils.data.DataLoader(Samples(), batch_size=2, num_workers=2,
                                         multiprocessing_context='spawn', pin_memory=True)
        loader = CudaLoader(cpu, p, seed_key='seed')
        self.assertEqual(len(loader), 3)
        actual = list(loader)
        self.assertEqual([v['image'].shape[0] for v in actual], [2, 2, 1])
        reference = list(CudaLoader(torch.utils.data.DataLoader(Samples(), batch_size=3), p, seed_key='seed'))
        torch.testing.assert_close(torch.cat([x['image'] for x in actual]),
                                   torch.cat([x['image'] for x in reference]), rtol=0, atol=0)
        self.assertTrue(all(x['id'].device.type == 'cpu' for x in actual))

    def test_epoch_prefetch_and_retained_outputs(self):
        p = self.pipeline()
        cpu = torch.utils.data.DataLoader(Samples(), batch_size=2)
        loader = CudaLoader(cpu, p, seed=123)
        before = torch.cat([b['image'] for b in loader])
        repeated = torch.cat([b['image'] for b in loader])
        torch.testing.assert_close(before, repeated, rtol=0, atol=0)
        no_prefetch = torch.cat([b['image'] for b in CudaLoader(cpu, p, seed=123, prefetch=False)])
        torch.testing.assert_close(before, no_prefetch, rtol=0, atol=0)
        loader.set_epoch(1)
        self.assertFalse(torch.equal(before, torch.cat([b['image'] for b in loader])))
        snapshot = iter(loader)
        loader.set_epoch(2)
        a = torch.cat([b['image'] for b in snapshot])
        loader.set_epoch(1)
        torch.testing.assert_close(a, torch.cat([b['image'] for b in loader]), rtol=0, atol=0)

    def test_early_close_empty_and_exception_cleanup(self):
        loader = CudaLoader(torch.utils.data.DataLoader(Samples(), batch_size=2), self.pipeline())
        iterator = iter(loader)
        out = next(iterator)
        saved = out['image'].clone()
        iterator.close()
        gc.collect()
        torch.testing.assert_close(out['image'], saved, rtol=0, atol=0)
        self.assertEqual(list(CudaLoader([], self.pipeline())), [])
        with self.assertRaises(TypeError):
            list(CudaLoader([None], self.pipeline()))
        bad = image_batch().float()
        bad.fill_(float('nan'))
        with self.assertRaisesRegex((ValueError, RuntimeError), 'nonfinite'):
            next(iter(CudaLoader([{'image': bad}], self.pipeline())))

    def test_worker_guard(self):
        cpu = torch.utils.data.DataLoader(WorkerMisuse(), num_workers=1, multiprocessing_context='spawn')
        with self.assertRaisesRegex(RuntimeError, 'training process'):
            next(iter(cpu))

    def test_consumer_stream_and_resident_inputs(self):
        p = self.pipeline()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            x = image_batch().to('cuda', dtype=torch.float32)
            loader = CudaLoader([{'image': x, 'seed': [4, 5, 6]}], p, seed_key='seed')
            out = next(iter(loader))['image']
            result = out.square()
            del out, loader
            gc.collect()
            torch.empty((100000,), device='cuda').fill_(17)
            ref = p.batch({'image': x}, seeds=[4, 5, 6])['image'].square()
        stream.synchronize()
        torch.testing.assert_close(result, ref, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available() and A._native.cuda_available(), "CUDA build and device required")
class GeometryTests(unittest.TestCase):
    def test_exact_reflections_all_axes_shapes_and_dtypes(self):
        import itertools
        for shape in ((1, 7), (5, 9), (1, 5, 9), (3, 5, 7)):
            names = 'zyx'[-len(shape):]
            size = 2 * int(np.prod(shape))
            for dtype in (np.float32, np.float64, np.int8, np.uint16, np.int64, np.uint64):
                values = np.arange(size).reshape(2, *shape).astype(dtype)
                if np.issubdtype(dtype, np.floating):
                    values.reshape(-1)[::3] = -0.0
                    role = A.Image(valid=None)
                else:
                    if np.dtype(dtype).itemsize == 8:
                        values += np.array(1 << 54, dtype=dtype)
                    role = A.Label(fill=1000)
                for flags in itertools.product((False, True), repeat=len(shape)):
                    selected = ''.join(a for a, use in zip(names, flags) if use)
                    p = A.Pipeline([A.Flip(axes=selected or names, p_axis=int(bool(selected)))],
                                   targets={'data': role}, backend='cuda')
                    actual = p({'data': cuda.asarray(values)}, seed=19)['data'].numpy()
                    axes = tuple(i+1 for i, use in enumerate(flags) if use)
                    expected = np.flip(values, axes)
                    self.assertEqual(actual.tobytes(), expected.tobytes(), (shape, dtype, flags))

    def test_reflection_validity_and_directional_routing(self):
        image = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
        valid = np.ones_like(image)
        valid[0, 0, ::2] = -0.0
        sample = {'image': image, 'valid': valid,
                  'points': np.array([[0., 0.], [2., 3.]], dtype=np.float64),
                  'field': np.broadcast_to(np.array([1., 2.])[:, None, None], (2, 4, 6)).copy()}
        targets = {'image': A.Image(valid='valid'), 'valid': A.ValidMask(),
                   'points': A.Points(order='yx'), 'field': A.VectorField(components='yx')}
        transforms = [A.Flip(axes='yx', p_axis=1)]
        cpu = A.Pipeline(transforms, targets=targets, backend='cpu')(sample, seed=19)
        gpu = A.Pipeline(transforms, targets=targets, backend='cuda')(
            {k: cuda.asarray(v) for k, v in sample.items()}, seed=19)
        for key in sample:
            np.testing.assert_array_equal(gpu[key].numpy(), cpu[key])
        actual = gpu['valid'].numpy()
        self.assertFalse(np.signbit(actual[actual == 0]).any())

    def test_affine_fallback_and_elastic_replay(self):
        image = np.random.default_rng(42).random((2, 9, 13), dtype=np.float32)
        sample = {'image': cuda.asarray(image)}
        for transform in (A.Affine(degrees=7, rotation_axes='yx'),
                          A.Elastic(sigma=3, magnitude=1, border=True)):
            p = A.Pipeline([transform], backend='cuda')
            direct = p(sample, seed=9)['image'].numpy()
            plan = p.sample(sample, seed=9)
            replay = plan.apply(sample)['image'].numpy()
            self.assertEqual(direct.tobytes(), replay.tobytes())

    def test_float32_elastic_matches_cpu_and_preserves_labels(self):
        rng = np.random.default_rng(19)
        for shape in ((1, 1), (1, 9), (9, 13), (1, 5, 7), (17, 19, 23)):
            image = rng.random((1, *shape), dtype=np.float32)
            mask = rng.integers(0, 5, image.shape, dtype=np.int64) + (1 << 54)
            sample = {'image': cuda.asarray(image), 'mask': cuda.asarray(mask)}
            for sigma in (0, 3, 12):
                for magnitude in (0, 2, 100):
                    with self.subTest(shape=shape, sigma=sigma, magnitude=magnitude):
                        transforms = [A.Elastic(sigma=sigma, magnitude=magnitude, border=True)]
                        cpu = A.Pipeline(transforms, backend='cpu')({'image': image}, seed=456)['image']
                        pipeline = A.Pipeline(transforms, backend='cuda')
                        actual = pipeline(sample, seed=456)
                        np.testing.assert_allclose(actual['image'].numpy(), cpu, atol=2e-5, rtol=0)
                        self.assertTrue(np.isin(actual['mask'].numpy(), mask).all())
                        repeated = pipeline(sample, seed=456)
                        for key in sample:
                            self.assertEqual(actual[key].numpy().tobytes(), repeated[key].numpy().tobytes())


if __name__ == '__main__':
    unittest.main()
