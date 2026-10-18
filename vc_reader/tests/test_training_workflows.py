"""Exercise example-owned integration, without adding a package Dataset API."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from training_workflows import (Array, TrainingDataset, augment_cpu_sample, cpu_batches, data_loader,
                                image_arrays, load_native, pipeline, restore_layers, select_arrays)
load_native()
import numpy as np
import torch
from vc_reader import Affine, ScrollSource, ValueFrame
from vc_reader import augment as A
from vc_reader.cuda import is_available
from training_inputs import build_workflow


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, name='custom'):
        root = self.root / name
        specs, requests, arrays, dimensions, _ = build_workflow(name, root)
        dataset = TrainingDataset(specs, requests, arrays, seeds=range(len(requests)), memory_cache_bytes=1 << 20)
        self.addCleanup(dataset.close)
        return dataset, dimensions

    def test_default_collation_spawn_persistence_epochs_and_close(self):
        dataset, dims = self.dataset()
        # Parent reads must never serialize a live handle into a worker.
        saved = dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._runtime)
        self.assertIsNone(restored._finalizer)
        restored.close()
        with data_loader(dataset, workers=2, batch_size=2) as loader:
            first = list(loader)
            workers = loader._iterator._workers
            pids = [worker.pid for worker in workers]
            second = list(loader)
            self.assertEqual(pids, [worker.pid for worker in loader._iterator._workers])
            for a, b in zip(first, second):
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            iterator = iter(loader)
            retained = next(iterator)
            snapshot = retained['image'].clone()
            del iterator
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertIsNone(dataset._runtime)
        torch.testing.assert_close(retained['image'], snapshot, rtol=0, atol=0)
        np.testing.assert_array_equal(saved['image'], first[0]['image'][0].numpy())
        self.assertEqual(restore_layers(first[0]).shape, (2, 2, 3, 4, 5))
        self.assertEqual(restore_layers(first[0], 'image_valid').shape, (2, 2, 3, 4, 5))

    def test_explicit_overlay_json_and_pixel_affine(self):
        dataset, _ = self.dataset()
        label = dataset.specs[0].reference
        view = ScrollSource.from_dict(json.loads(json.dumps(label.scroll.to_dict())))
        label = view.labels[label.id]
        other = replace(label.source, id='user:shifted-label')
        view = view.with_source(other, kind='label', volume='a', image=label.images.zarr().id,
                                pixel_affine=((1, 0, 2), (0, 1, 0)))
        shifted = view.labels[other.id]
        spec = replace(dataset.specs[0], reference=shifted, image=shifted.images.zarr())
        dataset.specs = (*dataset.specs, spec)
        dataset.requests = ((1, (0, 0), (4, 5)), *dataset.requests[1:])
        result = dataset[0]
        np.testing.assert_array_equal(result['label_valid'][..., :2], False)
        np.testing.assert_array_equal(result['label'][0, :, 2:], np.tile([0, 1, 2], (4, 1)))
        self.assertIs(dataset.reader(0)._options['memory_cache'], dataset.reader(1)._options['memory_cache'])
        self.assertLessEqual(dataset._runtime[2].nbytes, dataset.memory_cache_bytes)
        with data_loader(dataset, workers=1, batch_size=1) as loader:
            actual = next(iter(loader))
            np.testing.assert_array_equal(actual['label'][0].numpy(), result['label'])

    def test_per_layer_validity_geometry_and_categorical_values(self):
        dataset, dims = self.dataset()
        sample = dataset.read_sample(0)
        # Different holes at each depth prove that flattening never collapses D.
        sample.image.valid[0, 0, 0, 0] = False
        sample.image.valid[0, 1, 1, 1] = False
        selected = select_arrays(sample, dataset.arrays)
        p = pipeline(dataset.arrays, spatial_dims=dims, backend='cpu',
                     transforms=[A.Flip(axes='x', p_axis=1), A.BrightnessMultiply(2)])
        actual = p(selected, seed=7)
        np.testing.assert_array_equal(actual['image_valid'], selected['image_valid'][..., ::-1])
        expected = selected['image'][..., ::-1].astype(np.float32) * 2
        expected[~selected['image_valid'][..., ::-1]] = 0
        np.testing.assert_array_equal(actual['image'], expected)
        for key in ('label', 'label_valid', 'label_supervision'):
            np.testing.assert_array_equal(actual[key], selected[key][..., ::-1])
        self.assertEqual(set(np.unique(actual['label_supervision'])), {128, 255})
        if torch.cuda.is_available() and is_available():
            gpu = pipeline(dataset.arrays, spatial_dims=dims, backend='cuda',
                           transforms=[A.Flip(axes='x', p_axis=1), A.BrightnessMultiply(2)])
            batch = torch.utils.data.default_collate([selected])
            result = gpu.batch(batch, seeds=[7])
            for key in dataset.arrays:
                np.testing.assert_array_equal(result[key][0].cpu().numpy(), actual[key])
        with self.assertRaisesRegex(ValueError, '2D'):
            pipeline(dataset.arrays, spatial_dims=3, backend='cpu')

    def test_single_layer_broadcast_and_byte_order_for_default_collation(self):
        dataset, _ = self.dataset()
        dataset.specs = (replace(dataset.specs[0], layers=slice(0, 1)),)
        sample = dataset.read_sample(0)
        sample.label.data = sample.label.data.astype('>u2') + np.uint16(300)
        # Make byte order explicit after arithmetic, which may produce native endian.
        sample.label.data = sample.label.data.astype('>u2')
        selected = select_arrays(sample, dataset.arrays)
        for key in dataset.arrays:
            self.assertTrue(selected[key].flags.writeable)
            self.assertTrue(selected[key].dtype.isnative)
        actual = torch.utils.data.default_collate([selected])
        np.testing.assert_array_equal(actual['label'][0].numpy(), sample.label.data)
        self.assertEqual(actual['image_valid'].shape, (1, 2, 4, 5))

    def test_registered_transfer_independent_values_frames_and_coverage(self):
        for name in ('registered', 'transfer'):
            with self.subTest(name=name):
                dataset, dims = self.dataset(name)
                sample = dataset.read_sample(0)
                start, shape = dataset.requests[0][1:]
                z, y, x = np.indices(shape, dtype=np.float32)
                z, y, x = z + start[0], y + start[1], x + start[2]
                shift = 2 if name == 'transfer' else 0
                expected = ((x - shift) % 3).astype(np.uint8)
                valid = x >= shift
                expected[~valid] = 0
                np.testing.assert_array_equal(sample.label.data[0], expected)
                np.testing.assert_array_equal(sample.label.valid[0], valid)
                np.testing.assert_array_equal(sample.image.data[0], z*100 + y*10 + x + (100 if shift else 0))
                expected_vector = np.broadcast_to(np.array([-2, 1, 3], np.float32)[:, None, None, None], (3, *shape)).copy()
                expected_vector[:, ~valid] = 0
                np.testing.assert_array_equal(sample['features'].data, expected_vector)
                self.assertEqual(sample.metadata['value_frames']['features']['volume_id'], 'b' if shift else 'a')
                with data_loader(dataset, workers=0) as loader:
                    result = list(cpu_batches(loader, pipeline(dataset.arrays, spatial_dims=dims, backend='cpu')))
                    self.assertEqual(len(result), 3)
                # Empty coverage is explicit, not a valid all-background target.
                dataset.requests = ((0, (0, 0, -8), (2, 2, 2)),)
                sample = dataset.read_sample(0)
                self.assertFalse(sample.label.valid.any())
                self.assertFalse(sample.label.data.any())

    def test_surface_layers_projection_ownership_and_pipeline(self):
        for name in ('local-render', 'local-project'):
            dataset, dims = self.dataset(name)
            first = dataset.read_sample(0)
            values = {key: value.copy() for key, value in dataset[0].items() if isinstance(value, np.ndarray)}
            dataset[1]
            dataset.close()
            if name == 'local-render':
                self.assertEqual(first.image.data.shape, (1, 3, 3, 3))
                self.assertEqual(first.label.data.shape, (1, 3, 3))
                y, x = np.indices((3, 3), dtype=np.float32)
                expected = np.stack([z*100 + (y+3)*10 + x+3 for z in (4, 5, 6)])[None]
                np.testing.assert_array_equal(first.image.data, expected)
                np.testing.assert_array_equal(first.label.data[0], np.tile([0, 1, 2], (3, 1)))
                self.assertEqual(first.metadata['surface']['offsets_um'], (-9.6, 0., 9.6))
            else:
                self.assertTrue(first.label.valid.any())
                self.assertIn('conflict', first.label.masks)
                self.assertEqual(first.metadata['projection']['thickness_um'], 9.6)
            for key, value in values.items():
                np.testing.assert_array_equal(dataset[0][key], value)
            with data_loader(dataset, workers=1, batch_size=1) as loader:
                result = list(cpu_batches(loader, pipeline(dataset.arrays, spatial_dims=dims, backend='cpu')))
                self.assertEqual(len(result), 6)

    def test_registration_failures_cleanup_and_missing_mask(self):
        dataset, _ = self.dataset('registered')
        with self.assertRaises(TypeError):
            TrainingDataset(dataset.specs, dataset.requests, dataset.arrays, seeds=[1.5] * len(dataset))
        spec = dataset.specs[0]
        bad = replace(spec.targets['label'].source, registration=None)
        dataset.specs = (replace(spec, targets={'label': bad}),)
        with self.assertRaises(ValueError):
            dataset[0]
        self.assertEqual(dataset._runtime[3], {})
        dataset.close()
        dataset.specs = (spec,)
        dataset.arrays['absent'] = Array('label', 'masks/absent', 'label')
        with self.assertRaises(KeyError):
            dataset[0]
        dataset.close()

    def test_uint8_images_and_all_field_roles(self):
        from torch.utils.data import default_collate
        arrays = {**image_arrays(), 'scalar': Array('scalar', role='scalar'),
                  'vector': Array('vector', role='vector'), 'normal': Array('normal', role='normal'),
                  'coordinates': Array('coordinates', role='coordinate')}
        shape = (1, 2, 3, 4)
        image = np.arange(24, dtype=np.uint8).reshape(shape)
        field = np.broadcast_to(np.array([1, 2, 3], np.float32)[:, None, None, None], (3, 2, 3, 4)).copy()
        values = dict(image=image, image_valid=np.ones(shape, bool), scalar=image.copy(),
                      vector=field, normal=field, coordinates=field, seed=np.asarray(7, np.uint64))
        transforms = [A.Flip(axes='x', p_axis=1), A.BrightnessMultiply(2)]
        cpu = pipeline(arrays, spatial_dims=3, backend='cpu', transforms=transforms)
        result = augment_cpu_sample(cpu, values, seed=7)
        np.testing.assert_array_equal(result['image'], image[..., ::-1].astype(np.float32)*2)
        np.testing.assert_array_equal(result['scalar'], image[..., ::-1])
        np.testing.assert_array_equal(result['coordinates'], field[..., ::-1])
        np.testing.assert_array_equal(result['vector'][0], -1)
        np.testing.assert_allclose(np.linalg.norm(result['normal'], axis=0), 1, atol=1e-6)
        if torch.cuda.is_available() and is_available():
            gpu = pipeline(arrays, spatial_dims=3, backend='cuda', transforms=transforms)
            actual = gpu.batch(default_collate([values]), seeds=[7])
            for key in arrays:
                np.testing.assert_allclose(actual[key][0].cpu().numpy(), result[key], rtol=1e-6, atol=1e-6)

    def test_catalog_and_example_import_without_torch_numpy(self):
        code = "import sys; import vc_reader; import training_workflows; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        environment = {**os.environ, 'PYTHONPATH': os.pathsep.join(['src', 'scripts'])}
        subprocess.run([sys.executable, '-c', code], env=environment, check=True, timeout=30,
                       cwd=Path(__file__).resolve().parents[1])

    @unittest.skipUnless(torch.cuda.is_available() and is_available(), 'CUDA build and device required')
    def test_cuda_spawn_repeated_epochs_early_close_and_cpu_equivalence(self):
        from vc_reader.torch import CudaLoader
        for name in ('custom', 'registered', 'transfer', 'local-render', 'local-project'):
            dataset, dims = self.dataset(name)
            cpu = pipeline(dataset.arrays, spatial_dims=dims, backend='cpu')
            expected = [cpu({k: dataset[i][k] for k in dataset.arrays}, seed=dataset.seeds[i]) for i in range(len(dataset))]
            gpu = pipeline(dataset.arrays, spatial_dims=dims, backend='cuda')
            with data_loader(dataset, workers=2, batch_size=1, pin_memory=True) as loader:
                augmented = CudaLoader(loader, gpu, seed_key='seed')
                for epoch in range(2):
                    for i, actual in enumerate(augmented):
                        for key in dataset.arrays:
                            np.testing.assert_allclose(actual[key][0].cpu().numpy(), expected[i][key], rtol=2e-5, atol=2e-5)
                iterator = iter(augmented)
                retained = next(iterator)
                snapshot = retained['image'].clone()
                iterator.close()
            torch.testing.assert_close(retained['image'], snapshot, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()


class PublishedTransferRecipeTests(unittest.TestCase):
    def test_discovery_and_serialization_preserve_volume_mapping(self):
        fixture = Path(__file__).resolve().parents[1] / 'benchmarks/fixtures/training/catalog.json'
        specs, requests, arrays, dimensions, provenance = build_workflow('published-transfer', Path('/tmp'), fixture)
        spec = pickle.loads(pickle.dumps(specs[0]))
        self.assertEqual(spec.reference.id, '20250521125136')
        self.assertEqual(spec.targets['label'].volume.id, '20250820154339')
        self.assertEqual(spec.targets['scan_a'].registration.volume_id, '20250820154339')
        self.assertEqual(spec.mode, 'project')
        self.assertEqual(dimensions, 3)
        self.assertEqual(len(requests), 6)
        self.assertEqual(arrays['label_conflict'].role, 'label')
        self.assertEqual(provenance['required_operations'], ('surface_projection', 'volume_transform'))
        self.assertEqual(spec.targets['label'].source.registration.pixel_to_uv, ((1, 0, .5), (0, 1, .5)))
        self.assertNotIn('registration.pixel_to_uv', spec.targets['label'].source.unresolved_fields)
