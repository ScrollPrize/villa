"""0058 generated-store lifecycle controls; execute only in a frozen PBS job.

The0055 target repair is common to both arms. Config loading supplies metadata;
discovery, finder, cache serialization, ZarrDataset and item reads are production.
No CT, real annotations, trainer, model or full held-out-isolation claim.
"""
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
from vesuvius.models.preprocessing.patches import cache, generate


FG = (0, 0, 0)
UF = (8, 0, 0)


def tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [root / 'images', root / 'labels']
            for p in folder.rglob('*') if p.is_file()}


def manager(root, selected=None, *, enabled=True):
    return SimpleNamespace(data_path=root, train_patch_size=(8, 8, 8),
        targets={'ink': {'valid_patch_value': 1, 'ignore_label': 2}},
        dataset_config={'valid_patch_value': 1}, min_labeled_ratio=.01,
        min_bbox_percent=.01, valid_patch_find_resolution=0,
        bg_sampling_enabled=False, unlabeled_foreground_enabled=enabled,
        unlabeled_foreground_volumes=selected,
        unlabeled_foreground_threshold=.05, unlabeled_foreground_bbox_threshold=.15,
        normalization_scheme='none', skip_intensity_sampling=True,
        ome_zarr_resolution=0, load_config=lambda path: None)


def produce(mgr, *, force=False):
    with mock.patch.object(generate, 'ConfigManager', return_value=mgr):
        return generate.generate_patch_caches(mgr.data_path / 'supplied-config.yaml', force=force)


class UnlabeledAllowlistCacheTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name); self.images = {}; self.labels = {}
        for i, name in enumerate(['a', 'b']):
            image = np.zeros((16, 16, 16), dtype=np.uint16)
            labels = np.full((16, 16, 16), 2, dtype=np.uint8)
            for origin in [FG, UF]:
                slices = tuple(slice(v, v + 8) for v in origin)
                image[slices] = (np.arange(512, dtype=np.uint16).reshape(8, 8, 8)
                                 + (i + 1) * 1000)
            labels[:8, :8, :8] = 1
            labels[8:16, :8, :8] = 0
            self.images[name] = image; self.labels[name] = labels
            for path, values in [(self.root / 'images' / f'{name}.zarr', image),
                                 (self.root / 'labels' / f'{name}_ink.zarr', labels)]:
                a = create_level_dataset(str(path), '0', values.shape, (16, 8, 8), values.dtype, None)
                a[:] = values
        self.before = tree(self.root)

    def assert_items(self, selected, wanted, *, enabled=True):
        data = ZarrDataset(manager(self.root, selected, enabled=enabled), is_training=False)
        actual = sorted((p.volume_name, tuple(p.position), p.is_unlabeled_fg) for p in data.valid_patches)
        expected = sorted([(name, FG, False) for name in ['a', 'b']]
                          + [(name, UF, True) for name in wanted])
        self.assertEqual(actual, expected)
        self.assertEqual((data.n_fg, data.n_unlabeled_fg), (2, len(wanted)))
        for i, patch in enumerate(data.valid_patches):
            item = data[i]; slices = tuple(slice(v, v + 8) for v in patch.position)
            np.testing.assert_array_equal(item['image'].numpy()[0],
                                          self.images[patch.volume_name][slices].astype(np.float32))
            np.testing.assert_array_equal(item['ink'].numpy()[0],
                                          self.labels[patch.volume_name][slices].astype(np.float32))
            self.assertEqual(bool(item['is_unlabeled']), patch.is_unlabeled_fg)
        self.assertEqual(tree(self.root), self.before)
        return data

    def test_narrowing_allowlist_recomputes_the_pool(self):
        first = produce(manager(self.root))
        second = produce(manager(self.root, ['b']))
        self.assertEqual((first.written_caches, second.written_caches), (1, 1))
        self.assertEqual((first.total_unlabeled_fg_patches, second.total_unlabeled_fg_patches), (2, 1))
        self.assert_items(['b'], ['b'])

    def test_distinct_nonempty_allowlists_have_distinct_pools(self):
        first = produce(manager(self.root, ['a']))
        second = produce(manager(self.root, ['b']))
        self.assertEqual((first.written_caches, second.written_caches), (1, 1))
        self.assertEqual((first.total_unlabeled_fg_patches, second.total_unlabeled_fg_patches), (1, 1))
        self.assert_items(['a'], ['a']); self.assert_items(['b'], ['b'])

    def test_force_other_selection_preserves_the_first_cache(self):
        produce(manager(self.root, ['a']))
        produce(manager(self.root, ['b']), force=True)
        self.assert_items(['a'], ['a'])
        self.assert_items(['b'], ['b'])

    def test_loader_lookup_misses_a_different_selection(self):
        produce(manager(self.root))
        data = self.assert_items(None, ['a', 'b'])
        data.unlabeled_fg_volume_ids = {'b'}
        # Isolate the actual cache lookup; no-cache fallback is a separate contract.
        self.assertIsNone(data._try_load_cache())
        self.assertEqual(tree(self.root), self.before)

    def test_order_and_duplicates_reuse_the_same_cache(self):
        first = produce(manager(self.root, ['b', 'a', 'b']))
        with mock.patch.object(generate, 'find_valid_patches', wraps=generate.find_valid_patches) as finder:
            second = produce(manager(self.root, ['a', 'b']))
        self.assertEqual((first.written_caches, second.written_caches, finder.call_count), (1, 0, 0))
        self.assert_items(['a', 'b'], ['a', 'b'])

    def test_none_and_empty_keep_unrestricted_reuse(self):
        first = produce(manager(self.root, None))
        with mock.patch.object(generate, 'find_valid_patches', wraps=generate.find_valid_patches) as finder:
            second = produce(manager(self.root, []))
        self.assertEqual((first.written_caches, second.written_caches, finder.call_count), (1, 0, 0))
        self.assert_items(None, ['a', 'b']); self.assert_items([], ['a', 'b'])

    def test_disabled_discovery_keeps_its_old_key_and_labeled_pool(self):
        first = produce(manager(self.root, ['a'], enabled=False))
        with mock.patch.object(generate, 'find_valid_patches', wraps=generate.find_valid_patches) as finder:
            second = produce(manager(self.root, ['b'], enabled=False))
        self.assertEqual((first.written_caches, second.written_caches, finder.call_count), (1, 0, 0))
        paths = list((self.root / '.patches_cache').glob('*.json')); self.assertEqual(len(paths), 1)
        self.assertNotIn('unlabeled_fg_volume_ids', json.loads(paths[0].read_text())['metadata']['cache_params'])
        self.assert_items(['b'], [], enabled=False)

    def test_unknown_selection_recomputes_labeled_only(self):
        produce(manager(self.root))
        second = produce(manager(self.root, ['unknown']))
        self.assertEqual((second.written_caches, second.total_fg_patches,
                          second.total_unlabeled_fg_patches), (1, 2, 0))
        self.assert_items(['unknown'], [])

    def test_legacy_enabled_cache_without_selection_is_not_reused(self):
        produce(manager(self.root, ['a']))
        folder = self.root / '.patches_cache'; current, = folder.glob('*.json')
        legacy = json.loads(current.read_text())
        params = legacy['metadata']['cache_params']; params.pop('unlabeled_fg_volume_ids', None)
        old_path = folder / f'patches_v{cache.SCHEMA_VERSION}_{cache._hash_config(params)}.json'
        old_path.write_text(json.dumps(legacy))
        if old_path != current:
            current.unlink()
        # The old payload contains only a, but its key cannot tell us that.
        second = produce(manager(self.root, None))
        self.assertEqual((second.written_caches, second.total_unlabeled_fg_patches), (1, 2))
        self.assertTrue(old_path.is_file())
        self.assert_items(None, ['a', 'b'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
