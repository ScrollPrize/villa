"""0065 guard compatibility only; actual scientific/store execution in PBS."""
import json
import unittest
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
import test_cache_unlabeled_allowlist as fixtures


class UnlabeledSkipCompatibilityTests(unittest.TestCase):
    def setUp(self):
        fixtures.UnlabeledAllowlistCacheTests.setUp(self)

    def manager(self, *, skip):
        mgr = fixtures.manager(self.root, ['b'])
        mgr.skip_patch_validation = skip
        return mgr

    def assert_enumeration(self, data):
        self.assertEqual((len(data), data.n_fg, data.n_unlabeled_fg), (16, 16, 0))
        self.assertEqual(sorted(set(p.volume_name for p in data.valid_patches)), ['a', 'b'])
        self.assertEqual(fixtures.tree(self.root), self.before)

    def test_explicit_skip_without_cache_keeps_enumeration(self):
        error = None; data = None
        try:
            data = ZarrDataset(self.manager(skip=True), is_training=False)
        except ValueError as caught:
            error = str(caught)
        self.assertIsNone(error, 'Explicit skip was incorrectly rejected')
        self.assert_enumeration(data)

    def test_default_validation_still_requires_matching_cache(self):
        with self.assertRaisesRegex(ValueError, 'matching patch cache'):
            ZarrDataset(self.manager(skip=False), is_training=False)
        self.assertEqual(fixtures.tree(self.root), self.before)

    def test_invalid_mapping_with_explicit_skip_keeps_enumeration(self):
        mapping = self.root / 'samples_mapping.json'
        raw = json.dumps({'layout': {'sample_shape': []}, 'samples': []}).encode()
        mapping.write_bytes(raw)
        self.assert_enumeration(ZarrDataset(self.manager(skip=True), is_training=False))
        self.assertEqual(mapping.read_bytes(), raw)

    def test_explicit_skip_with_matching_cache_keeps_selected_items(self):
        mgr = self.manager(skip=True); fixtures.produce(mgr)
        data = ZarrDataset(mgr, is_training=False)
        self.assertEqual(sorted((p.volume_name, tuple(p.position), p.is_unlabeled_fg)
                               for p in data.valid_patches),
                         [('a', fixtures.FG, False), ('b', fixtures.FG, False), ('b', fixtures.UF, True)])
        for i, patch in enumerate(data.valid_patches):
            item = data[i]; slices = tuple(slice(v, v + 8) for v in patch.position)
            fixtures.np.testing.assert_array_equal(item['image'].numpy()[0], self.images[patch.volume_name][slices].astype('float32'))
            fixtures.np.testing.assert_array_equal(item['ink'].numpy()[0], self.labels[patch.volume_name][slices].astype('float32'))
        self.assertEqual(fixtures.tree(self.root), self.before)


if __name__ == '__main__':
    unittest.main(verbosity=2)
