"""0063 standalone generated-store null-value lifecycle; PBS execution only."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches import generate

MISSING = object()


def input_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [root / 'images', root / 'labels']
            for p in folder.rglob('*') if p.is_file()}


class NullValueCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = np.arange(4096, dtype=np.uint16).reshape(16, 16, 16)
        self.ink = np.zeros((16, 16, 16), dtype=np.uint8); self.ink[:8, :8, :8] = 1
        self.surface = np.zeros((16, 16, 16), dtype=np.uint8); self.surface[8:16, :8, :8] = 2
        for name, values in [('images/vol.zarr', self.image), ('labels/vol_ink.zarr', self.ink),
                             ('labels/vol_surface.zarr', self.surface)]:
            array = create_level_dataset(str(self.root / name), '0', values.shape, (8, 8, 8), values.dtype, None)
            array[:] = values
        self.before = input_tree(self.root)

    def run_lifecycle(self, *, primary=None, secondary=True, fallback=1, expected_value=1, expected_count=1):
        first = {} if primary is MISSING else {'valid_patch_value': primary}
        targets = {'ink': first}
        if secondary:
            targets['surface'] = {'valid_patch_value': 2}
        dataset = {} if fallback is MISSING else {'valid_patch_value': fallback}
        mgr = SimpleNamespace(data_path=self.root, train_patch_size=(8, 8, 8),
            targets=targets, dataset_config=dataset, min_labeled_ratio=.01,
            min_bbox_percent=.01, valid_patch_find_resolution=0,
            bg_sampling_enabled=False, unlabeled_foreground_enabled=False,
            unlabeled_foreground_threshold=.05, unlabeled_foreground_bbox_threshold=.15,
            normalization_scheme='none', skip_intensity_sampling=True,
            ome_zarr_resolution=0, load_config=lambda path: None)
        with mock.patch.object(generate, 'ConfigManager', return_value=mgr):
            result = generate.generate_patch_caches(self.root / 'supplied-config.yaml')
        self.assertEqual((result.written_caches, result.total_fg_patches), (1, expected_count))
        files = list((self.root / '.patches_cache').glob('*.json')); self.assertEqual(len(files), 1)
        payload = json.loads(files[0].read_text())
        self.assertEqual(payload['metadata']['cache_params']['valid_patch_value'], expected_value)
        data = ZarrDataset(mgr, is_training=False)
        self.assertIsNotNone(data._try_load_cache())
        self.assertEqual((len(data), data.n_fg, data.n_unlabeled_fg), (expected_count, expected_count, 0))
        if expected_count:
            self.assertEqual([tuple(p.position) for p in data.valid_patches], [(0, 0, 0)])
            item = data[0]
            np.testing.assert_array_equal(item['image'].numpy()[0], self.image[:8, :8, :8].astype(np.float32))
            np.testing.assert_array_equal(item['ink'].numpy()[0], np.ones((8, 8, 8), dtype=np.float32))
            if secondary:
                np.testing.assert_array_equal(item['surface'].numpy()[0], np.zeros((8, 8, 8), dtype=np.float32))
        self.assertEqual(input_tree(self.root), self.before)

    def test_null_primary_uses_dataset_fallback_before_secondary(self):
        self.run_lifecycle()

    def test_null_primary_without_dataset_fallback_stays_unfiltered(self):
        self.run_lifecycle(fallback=MISSING, expected_value=None)

    def test_explicit_primary_keeps_its_precedence(self):
        self.run_lifecycle(primary=1, fallback=3)

    def test_single_null_target_keeps_dataset_fallback(self):
        self.run_lifecycle(secondary=False)

    def test_missing_key_keeps_secondary_inheritance(self):
        self.run_lifecycle(primary=MISSING, expected_value=2, expected_count=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
