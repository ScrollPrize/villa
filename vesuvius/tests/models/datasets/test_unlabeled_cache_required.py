"""0060 PBS-only guard integration, with0055+0058 common to both arms."""
import json
import unittest

from vesuvius.models.datasets.zarr_dataset import ZarrDataset
import test_cache_unlabeled_allowlist as fixtures


class UnlabeledCacheRequiredTests(unittest.TestCase):
    def setUp(self):
        fixtures.UnlabeledAllowlistCacheTests.setUp(self)

    def test_default_missing_cache_rejects_unlabeled_sampling(self):
        with self.assertRaisesRegex(ValueError, 'matching patch cache'):
            ZarrDataset(fixtures.manager(self.root, ['b']), is_training=False)
        self.assertEqual(fixtures.tree(self.root), self.before)

    def test_invalid_mapping_missing_cache_rejects_unlabeled_sampling(self):
        mapping = self.root / 'samples_mapping.json'
        payload = json.dumps({'layout': {'sample_shape': []}, 'samples': []}).encode()
        mapping.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, 'matching patch cache'):
            ZarrDataset(fixtures.manager(self.root, ['b']), is_training=False)
        self.assertEqual(mapping.read_bytes(), payload)
        self.assertEqual(fixtures.tree(self.root), self.before)

    def test_matching_selected_cache_retains_exact_items(self):
        fixtures.produce(fixtures.manager(self.root, ['b']))
        fixtures.UnlabeledAllowlistCacheTests.assert_items(self, ['b'], ['b'])

    def test_disabled_sampling_keeps_existing_unrestricted_fallback(self):
        data = ZarrDataset(fixtures.manager(self.root, ['b'], enabled=False), is_training=False)
        self.assertEqual((len(data), data.n_fg, data.n_unlabeled_fg), (16, 16, 0))
        self.assertEqual(sorted(set(p.volume_name for p in data.valid_patches)), ['a', 'b'])
        self.assertFalse(any(p.is_unlabeled_fg for p in data.valid_patches))
        self.assertEqual(fixtures.tree(self.root), self.before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
