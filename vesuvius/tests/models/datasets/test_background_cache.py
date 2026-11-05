"""PBS-only production cache/generator/dataset controls; labels are engineered.

Only ConfigManager construction/loading is replaced with an explicit metadata
fixture. The actual discovery, patch finder, serialization and dataset run.
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
import zarr

from vesuvius.image_proc.run.zarr_tasks.utils import create_level_dataset
from vesuvius.models.datasets.zarr_dataset import ZarrDataset
from vesuvius.models.preprocessing.patches import cache, generate


FG = [(0, 0, 0), (0, 8, 0)]
BG = [(0, 0, 8), (8, 0, 0)]


def fixture(root, *, image=None, enabled=True, chunk=8, ignore_alias="ignore_label"):
    root = Path(root)
    root.mkdir(parents=True)
    image = (np.arange(4096, dtype=np.uint16).reshape(16, 16, 16)
             if image is None else np.ascontiguousarray(image))
    assert image.shape == (16, 16, 16)
    labels = np.full((16, 16, 16), 2, dtype=np.uint8)
    for origin in FG:
        slices = tuple(slice(p, p+8) for p in origin)
        labels[slices] = 1
        labels[origin] = 2
    for z, y, x in BG:
        labels[z:z+8, y:y+8, x:x+8] = 0
        labels[z:z+4, y:y+8, x:x+8] = 2
    for path, values in [(root / "images/vol.zarr", image),
                         (root / "labels/vol_surface.zarr", labels)]:
        array = create_level_dataset(str(path), "0", values.shape, (chunk,)*3, values.dtype, None)
        array[:] = values
    mgr = SimpleNamespace(data_path=root, train_patch_size=(8, 8, 8),
        targets={"surface": {ignore_alias: 2, "valid_patch_value": 1}},
        dataset_config={"valid_patch_value": 1}, bg_sampling_enabled=enabled,
        min_labeled_ratio=.01, min_bbox_percent=.01, valid_patch_find_resolution=0,
        unlabeled_foreground_enabled=False, unlabeled_foreground_threshold=.05,
        unlabeled_foreground_bbox_threshold=.15, normalization_scheme="none",
        skip_intensity_sampling=True, ome_zarr_resolution=0,
        load_config=lambda path: None)
    return SimpleNamespace(root=root, image=image, labels=labels, mgr=mgr)


def generate_fixture(value, *, force=True):
    with mock.patch.object(generate, "ConfigManager", return_value=value.mgr):
        return generate.generate_patch_caches(value.root / "fixture-config.yaml", force=force)


def cached_background(value):
    paths = list((value.root / ".patches_cache").glob("*.json"))
    assert len(paths) == 1
    path = paths[0]
    payload = json.loads(path.read_text())
    payload["bg_patches"] = [{"vol_idx": 0, "vol_name": "vol", "pos": list(pos)} for pos in BG]
    payload["metadata"]["counts"]["bg"] = len(BG)
    path.write_text(json.dumps(payload))
    return path


def dataset(value):
    return ZarrDataset(value.mgr, is_training=False)


def positions(data):
    return [tuple(item.position) for item in data.valid_patches]


def exact_items(value, data):
    assert data.normalizer is None and data.transforms is None
    for index, position in enumerate(positions(data)):
        item = data[index]
        slices = tuple(slice(p, p+8) for p in position)
        np.testing.assert_array_equal(item["image"].numpy()[0], value.image[slices].astype(np.float32))
        np.testing.assert_array_equal(item["surface"].numpy()[0], value.labels[slices].astype(np.float32))
        assert not item["is_unlabeled"]


class BackgroundCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make(self, name="data", **kwargs):
        return fixture(self.root / name, **kwargs)

    def test_generator_collects_requested_background(self):
        value = self.make()
        result = generate_fixture(value)
        self.assertEqual(result.total_fg_patches, 2)
        self.assertEqual(result.total_bg_patches, 2)

    def test_generator_passes_scanned_ignore_label_aliases(self):
        for alias in ("ignore_label", "ignore_index", "ignore_value"):
            value = self.make(alias, ignore_alias=alias)
            with mock.patch.object(generate, "find_valid_patches", wraps=generate.find_valid_patches) as finder:
                result = generate_fixture(value)
            self.assertTrue(finder.call_args.kwargs.get("collect_bg_only", False))
            self.assertEqual(finder.call_args.kwargs.get("ignore_labels"), [2.0])
            self.assertEqual(result.total_bg_patches, 2)

    def test_enabled_request_avoids_old_foreground_only_cache(self):
        value = self.make(enabled=False)
        self.assertEqual(generate_fixture(value).total_bg_patches, 0)
        old = next((value.root / ".patches_cache").glob("*.json"))
        digest = hashlib.sha256(old.read_bytes()).hexdigest()
        value.mgr.bg_sampling_enabled = True
        result = generate_fixture(value, force=False)
        self.assertEqual(result.written_caches, 1)
        self.assertEqual(result.total_bg_patches, 2)
        self.assertEqual(len(list(old.parent.glob("*.json"))), 2)
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), digest)

    def test_enabled_cache_roundtrip_restores_background_pool(self):
        value = self.make()
        generate_fixture(value)
        data = dataset(value)
        self.assertEqual(positions(data), FG + BG)
        self.assertEqual(data.n_fg, 2)
        exact_items(value, data)

    def test_preexisting_background_cache_is_consumed(self):
        value = self.make()
        generate_fixture(value)
        cached_background(value)
        data = dataset(value)
        self.assertEqual(positions(data), FG + BG)
        self.assertEqual((data.n_fg, data.n_unlabeled_fg), (2, 0))
        exact_items(value, data)

    def test_enabled_missing_cache_rejects_unvalidated_fallback(self):
        value = self.make()
        with self.assertRaisesRegex(ValueError, "Background sampling needs a matching patch cache"):
            dataset(value)

    def rejection_before_finder(self, value):
        with mock.patch.object(generate, "find_valid_patches", wraps=generate.find_valid_patches) as finder:
            with self.assertRaisesRegex(ValueError, "Background"):
                generate_fixture(value)
            finder.assert_not_called()
        self.assertEqual(list((value.root / ".patches_cache").glob("*.json")), [])

    def test_missing_scanned_ignore_rejects_before_finder(self):
        value = self.make()
        del value.mgr.targets["surface"]["ignore_label"]
        value.mgr.targets["other"] = {"ignore_label": 2}
        self.rejection_before_finder(value)

    def test_missing_foreground_rejects_before_finder(self):
        value = self.make()
        del value.mgr.targets["surface"]["valid_patch_value"]
        value.mgr.dataset_config.clear()
        self.rejection_before_finder(value)

    def test_equal_labels_reject_before_finder(self):
        value = self.make()
        value.mgr.targets["surface"]["valid_patch_value"] = 2
        self.rejection_before_finder(value)

    def test_nonfinite_ignore_rejects_before_finder(self):
        value = self.make()
        value.mgr.targets["surface"]["ignore_label"] = float("nan")
        self.rejection_before_finder(value)

    def test_ignore_change_invalidates_enabled_cache(self):
        value = self.make()
        generate_fixture(value)
        old = next((value.root / ".patches_cache").glob("*.json"))
        digest = hashlib.sha256(old.read_bytes()).hexdigest()
        value.mgr.targets["surface"]["ignore_label"] = 3
        self.assertEqual(generate_fixture(value, force=False).written_caches, 1)
        self.assertEqual(len(list(old.parent.glob("*.json"))), 2)
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), digest)

    def test_loaded_background_keeps_labeled_foreground_boundary(self):
        value = self.make()
        generate_fixture(value)
        cached_background(value)
        data = dataset(value)
        # The existing trainer splits only range(n_fg) for validation and adds
        # non-FG indices to training. No trainer/DDP execution is claimed here.
        self.assertEqual(positions(data)[:data.n_fg], FG)
        self.assertEqual(positions(data)[data.n_fg:], BG)
        self.assertEqual(data.get_labeled_unlabeled_patch_indices(), ([0, 1, 2, 3], []))

    def test_disabled_cache_schema_and_name_control(self):
        value = self.make(enabled=False)
        result = generate_fixture(value)
        self.assertEqual((result.total_fg_patches, result.total_bg_patches), (2, 0))
        path = next((value.root / ".patches_cache").glob("*.json"))
        expected = {"schema_version": 3, "data_path": str(value.root.resolve()),
            "volume_ids": ["vol"], "patch_size": [8, 8, 8], "min_labeled_ratio": .01,
            "bbox_threshold": .01, "valid_patch_find_resolution": 0, "valid_patch_value": 1,
            "unlabeled_fg_enabled": False, "unlabeled_fg_threshold": .05,
            "unlabeled_fg_bbox_threshold": .15}
        payload = json.loads(path.read_text())
        self.assertEqual(payload["metadata"]["cache_params"], expected)
        name = hashlib.md5(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:12]
        self.assertEqual(path.name, "patches_v3_" + name + ".json")

    def test_disabled_dataset_excludes_preexisting_background_control(self):
        value = self.make(enabled=False)
        generate_fixture(value)
        cached_background(value)
        data = dataset(value)
        self.assertEqual(positions(data), FG)
        exact_items(value, data)

    def test_disabled_missing_cache_keeps_enumeration_control(self):
        data = dataset(self.make(enabled=False))
        self.assertEqual((len(data), data.n_fg), (8, 8))

    def test_empty_background_cache_is_allowed_control(self):
        value = self.make()
        labels = np.ones((16, 16, 16), dtype=np.uint8)
        labels[::8, ::8, ::8] = 2
        zarr.open(str(value.root / "labels/vol_surface.zarr/0"), mode="r+")[:] = labels
        value.labels = labels
        result = generate_fixture(value)
        self.assertEqual((result.total_fg_patches, result.total_bg_patches), (8, 0))
        data = dataset(value)
        self.assertEqual((len(data), data.n_fg), (8, 8))
        exact_items(value, data)

    def test_foreground_unlabeled_and_background_serialization_control(self):
        params = cache.build_cache_params(data_path=self.root, volume_ids=["v"], patch_size=(8,)*3,
            min_labeled_ratio=.1, bbox_threshold=.1, valid_patch_find_resolution=0)
        entry = lambda pos: cache.PatchEntry(0, "v", pos)
        data = cache.PatchCacheData([entry((0, 0, 0))], [entry((8, 0, 0))], [entry((0, 8, 0))])
        cache.save_patch_cache(self.root / "cache", params, data)
        self.assertEqual(cache.load_patch_cache(self.root / "cache", params), data)

    def test_large_storage_chunk_remains_unrepaired_negative_control(self):
        value = self.make(chunk=16)
        result = generate_fixture(value)
        # Existing finder checks BG over a storage block containing several
        # candidates. FG anywhere in it suppresses BG elsewhere.0041 does not
        # repair that separate scanner limitation; freeze and report it.
        self.assertEqual((result.total_fg_patches, result.total_bg_patches), (2, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
