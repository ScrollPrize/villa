"""Validation and README updates for the augmentation benchmark harness."""
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from bench_augmentations import compare, update_readme
from augmentation_cases import CASES, input_sample
from bench_pipeline import load_extensions


class AugmentationBenchmarkTests(unittest.TestCase):
    def test_cases_run_on_planar_and_volume_inputs(self):
        directory = os.environ.get('VC_READER_NATIVE_DIR')
        if directory:
            load_extensions(Path(directory))
        from vc_reader import augment as A, cuda
        backends = ['cpu', 'cuda'] if cuda.is_available() else ['cpu']
        for dims in (2, 3):
            image = np.random.default_rng(123).random((1,) + (16,) * dims, dtype=np.float32)
            reference = np.ascontiguousarray(image[..., ::-1])
            for case in CASES:
                for backend in backends:
                    with self.subTest(case=case.name, dims=dims, backend=backend):
                        source = input_sample(image, case.sample)
                        original = {key: value.copy() for key, value in source.items()}
                        transform = case.construct(A, reference, dims)
                        pipeline = A.Pipeline([transform], backend=backend,
                            targets=dict(image=A.Image(), mask=A.Label(), surface=A.CoordinateField()))
                        output = pipeline(source, seed=123)
                        self.assertEqual(output.keys(), source.keys())
                        for key, value in output.items():
                            self.assertEqual(value.ndim, source[key].ndim)
                            self.assertEqual(value.dtype, source[key].dtype)
                            self.assertTrue(np.isfinite(value).all())
                            np.testing.assert_array_equal(source[key], original[key])

    def test_native_override_is_idempotent(self):
        import importlib.machinery
        with tempfile.TemporaryDirectory() as directory:
            modules = {}
            for name in ('_native', '_augment'):
                path = Path(directory) / (name+importlib.machinery.EXTENSION_SUFFIXES[0])
                path.touch()
                modules['vc_reader.'+name] = SimpleNamespace(__file__=str(path))
            with patch.dict(sys.modules, modules), patch('importlib.util.spec_from_file_location') as load:
                load_extensions(Path(directory))
                load.assert_not_called()

    def test_numerical_validation_and_exact_large_labels(self):
        source = {'image': np.array([.25, .5], dtype=np.float32)}
        rounded = {'image': source['image']+1e-7}
        self.assertEqual(compare(rounded, source)['image']['mismatched'], 0)
        self.assertEqual(compare(rounded, source, atol=0, rtol=0)['image']['mismatched'], 2)
        labels = {'mask': np.array([2**60], dtype=np.int64)}
        self.assertEqual(compare({'mask': labels['mask']+1}, labels)['mask']['mismatched'], 1)
        for bad in ({}, {'image': np.array([.25], dtype=np.float32)},
                    {'image': source['image'].astype(np.float64)},
                    {'image': np.array([.25, np.nan], dtype=np.float32)}):
            with self.assertRaises(AssertionError):
                compare(bad, source)

    def test_input_routing_and_image_allocation(self):
        for shape in ((1, 4, 6), (1, 3, 4, 6)):
            image = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
            # Image-only cases must not allocate labels, duplicated channels or surfaces.
            with patch('numpy.indices', side_effect=AssertionError('surface allocated')), \
                 patch('numpy.concatenate', side_effect=AssertionError('channels allocated')):
                self.assertIs(input_sample(image, 'image')['image'], image)
            mask = input_sample(image, 'both')['mask']
            self.assertEqual(mask.dtype, np.int16)
            np.testing.assert_array_equal(mask, image > .5)
            np.testing.assert_array_equal(input_sample(image, 'labels')['mask'], mask)
            binary = input_sample(image, 'binary')['image']
            self.assertEqual(binary.dtype, np.float32)
            np.testing.assert_array_equal(binary, mask)
            channels = input_sample(image, 'channels')['image']
            np.testing.assert_array_equal(channels, np.concatenate([image, image]))
            plane = image if image.ndim == 3 else image[:, 1]
            np.testing.assert_array_equal(input_sample(image, 'strip')['image'], plane)
            surface = input_sample(image, 'surface')['surface']
            self.assertEqual(surface.dtype, np.float32)
            np.testing.assert_array_equal(surface[0], plane[0] * 2)
            np.testing.assert_array_equal(surface[1], np.broadcast_to(np.arange(4)[:, None], (4, 6)))
            np.testing.assert_array_equal(surface[2], np.broadcast_to(np.arange(6), (4, 6)))
        with self.assertRaises(ValueError):
            input_sample(image, 'unknown')

    def test_timing_update_preserves_recipes_and_images(self):
        text = '''<details>
<summary><strong>Example</strong></summary>
```python
A.Example()
```
| CPU | CUDA uploaded | CUDA resident | Uploaded speedup | Resident speedup |
| ---: | ---: | ---: | ---: | ---: |
| old timing |
![preview](example.webp)
</details>
'''
        row = {'timings': {m: {'p50_ms': v} for m, v in
                          [('cpu', 10), ('cuda_host', 2), ('cuda_resident', 1)]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'README.md'
            path.write_text(text)
            update_readme(path, {'Example': row})
            expected = text.replace('| old timing |', '| 10.000 ms | 2.000 ms | 1.000 ms | 5.0× | 10.0× |')
            self.assertEqual(path.read_text(), expected)
            with self.assertRaises(ValueError):
                update_readme(path, {'Missing': row})
            self.assertEqual(path.read_text(), expected)


if __name__ == '__main__':
    unittest.main()
