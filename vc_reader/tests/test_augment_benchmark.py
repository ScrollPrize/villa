"""Validation and README updates for the augmentation benchmark harness."""
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
from bench_augmentations import compare, constructor, inputs, planar_recipe, recipes, update_readme, workload
from bench_pipeline import load_extensions


class AugmentationBenchmarkTests(unittest.TestCase):
    def test_planar_recipes_and_targets(self):
        image = np.ones((1, 7, 9), dtype=np.float32)
        sample = inputs(image)
        self.assertEqual(sample['image']['image'].shape, (1, 7, 9))
        self.assertEqual(sample['both']['mask'].shape, (1, 7, 9))
        self.assertEqual(sample['channels']['image'].shape, (2, 7, 9))
        self.assertEqual(sample['surface']['surface'].shape, (3, 7, 9))
        self.assertEqual(planar_recipe('CLAHE', 'A.CLAHE(tile_shape=(3, 4, 5))'),
                         'A.CLAHE(tile_shape=(4, 5))')
        self.assertEqual(planar_recipe('OrientedBlur', 'A.OrientedBlur(axis=(1, 0, 0))'),
                         'A.OrientedBlur(axis=(1, 0))')
        self.assertEqual(planar_recipe('StripeMasking', "A.StripeMasking(axes='zyx')"),
                         "A.StripeMasking(axes='yx')")

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

    def test_recipes_and_input_routing(self):
        readme = Path(__file__).resolve().parents[1] / 'README.md'
        selected = recipes(readme)
        image = np.arange(64, dtype=np.float32).reshape(1, 4, 4, 4)/63
        samples = inputs(image)
        for name in selected:
            self.assertIn(workload(name).sample, samples)
        np.testing.assert_array_equal(samples['both']['mask'], image > .5)
        np.testing.assert_array_equal(samples['strip']['image'], image[:, 2])
        self.assertEqual(samples['channels']['image'].shape, (2, 4, 4, 4))
        self.assertEqual(samples['surface']['surface'].shape, (3, 4, 4))
        augment = SimpleNamespace(__all__=['Example'], Example=lambda *a, **k: (a, k))
        self.assertEqual(constructor('Example', 'A.Example(reference, strength=.5)', 'ref', augment),
                         (('ref',), {'strength': .5}))
        for unsafe in ('other.Example()', 'A.Example(open("file"))', 'A.Example(**{})'):
            with self.assertRaises(ValueError):
                constructor('Example', unsafe, 'ref', augment)

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
