"""A reader exposes what its store's sidecar documents say, read once at open."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from vc_reader import _catalog
from vc_reader.reader import ZarrReader, open_group, store_metadata
from vc_reader.reader.auth import Access


def write_array(path: Path) -> None:
    path.mkdir(parents=True)
    (path / ".zarray").write_text(json.dumps({
        "zarr_format": 2, "shape": [2, 2, 2], "chunks": [2, 2, 2], "dtype": "|u1", "compressor": None,
        "filters": None, "fill_value": 0, "order": "C"}))
    (path / "0.0.0").write_bytes(np.zeros(8, np.uint8).tobytes())


DOCUMENT = {
    "zarr_export": {"target_window_f32_min": -0.03, "target_window_f32_max": 0.21, "z_crop_start": 10, "z_crop_end": 20},
    "scan": {"tomo": {"acquisition": {
        "scanRadix": "SCAN", "date": "2025-12-05", "beamline": "bm18", "sampleDetectorDistance": 220.0,
        "energy": 78.0, "detector": {"samplePixelSize": 0.002399}}}},
}


class StoreMetadataTests(unittest.TestCase):
    def test_voxel_size_and_metadata_at_each_level(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp, "store.zarr")
            write_array(store / "0")
            write_array(store / "1")
            self.assertIsNone(ZarrReader.open(store, level=1).voxel_size)
            (store / "meta.json").write_text("not json")
            (store / "metadata.json").write_text(json.dumps(DOCUMENT))

            level0 = ZarrReader.open(store, level=0)
            level1 = ZarrReader.open(store, level=1)
            self.assertAlmostEqual(level0.voxel_size, 2.399)
            self.assertAlmostEqual(level1.voxel_size, 2 * 2.399)
            metadata = level1.metadata
            self.assertAlmostEqual(metadata.voxel_size_um, 2.399)
            self.assertEqual(metadata.energy_kev, 78.0)
            self.assertEqual(metadata.detector_distance_mm, 220.0)
            self.assertEqual((metadata.beamline, metadata.scan_name, metadata.acquisition_date),
                             ("bm18", "SCAN", "2025-12-05"))
            self.assertEqual(metadata.value_window, (-0.03, 0.21))
            self.assertEqual(metadata.z_crop, (10, 20))
            self.assertEqual(metadata.documents["metadata.json"], DOCUMENT)

            # Read at open: a worker's copy does not read it again.
            (store / "metadata.json").unlink()
            self.assertAlmostEqual(pickle.loads(pickle.dumps(level1)).voxel_size, 2 * 2.399)

    def test_declared_level_scales_and_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Path(tmp, "store.zarr")
            write_array(store / "0")
            write_array(store / "1")
            (store / ".zgroup").write_text('{"zarr_format": 2}')
            (store / ".zattrs").write_text(json.dumps({"multiscales": [{"datasets": [
                {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [3.0, 1.0, 1.0]}]},
                {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [3.0, 2.0, 2.0]}]},
            ]}]}))
            (store / "meta.json").write_text(json.dumps({"voxelsize": 2.0}))
            group = open_group(store)
            self.assertEqual(group.voxel_size, 2.0)
            coarse = group.array("1")
            self.assertEqual(coarse.voxel_size_zyx, (2.0, 4.0, 4.0))
            self.assertIsNone(coarse.voxel_size)  # not cubic
            self.assertEqual(ZarrReader.open(store, level=1).voxel_size_zyx, (2.0, 4.0, 4.0))


CATALOG = {"samples": {"S1": {"volumes": {"V1": {
    "id": "V1", "scan_id": "SCAN1", "suffix": "V1-2.4um.zarr",
    "properties": {"pixel_size_um": 2.4, "energy_keV": 70.0, "detector_distance_mm": 150.0},
    "data": [{"type": "zarr", "origins": [{"path": "S1/volumes/v1.zarr", "access_roots": [
        {"type": "https", "url": "https://bucket.example/", "usage": "public-read"}]}]}]}}}}}
URL = "https://bucket.example/S1/volumes/v1.zarr"


class CatalogMergeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(store_metadata._known.clear)
        store_metadata._known.clear()
        self.reads, self.catalogs = [], []
        self.documents = {}

        def read_node_file(location, name, access):
            self.reads.append(name)
            text = self.documents.get(name)
            return None if text is None else text.encode()

        def catalog_entry(location, offline):
            self.catalogs.append(location)
            return _catalog.parse(json.dumps(CATALOG)).volume_at(location), True

        for name, value in (("read_node_file", read_node_file), ("_catalog_entry", catalog_entry)):
            patcher = mock.patch.object(store_metadata, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.spec = {"cache_dirs": [self.tmp.name], "cache_writes": True, "offline": False}

    def read(self):
        return store_metadata.read_store_metadata(URL, Access("anonymous"), self.spec)

    def test_both_sources_merge_and_the_store_wins(self):
        self.documents["metadata.json"] = json.dumps({"tomo": {"acquisition": {
            "energy": 78.0, "beamline": "bm18", "detector": {"samplePixelSize": 0.002399}}}})
        metadata = self.read()
        self.assertAlmostEqual(metadata.voxel_size_um, 2.399)  # the store's own
        self.assertEqual(metadata.energy_kev, 78.0)
        self.assertEqual(metadata.detector_distance_mm, 150.0)  # only the catalog has it
        self.assertEqual(metadata.beamline, "bm18")
        self.assertEqual((metadata.sample_id, metadata.volume_id, metadata.scan_id), ("S1", "V1", "SCAN1"))
        self.assertEqual(sorted(metadata.documents), ["catalog", "metadata.json"])

    def test_catalog_fills_a_store_that_says_nothing(self):
        self.assertEqual(self.read().voxel_size_um, 2.4)

    def test_sources_are_consulted_only_the_first_time(self):
        first = self.read()
        self.assertEqual(len(self.catalogs), 1)
        reads = len(self.reads)
        self.assertEqual(self.read(), first)  # this process
        store_metadata._known.clear()
        self.assertEqual(self.read(), first)  # a later process: the disk cache
        self.assertEqual(len(self.catalogs), 1)
        self.assertEqual(len(self.reads), reads)

    def test_without_a_disk_cache_the_process_remembers(self):
        self.spec = {"cache_dirs": [], "cache_writes": True, "offline": False}
        self.read()
        self.read()
        self.assertEqual(len(self.catalogs), 1)

    def test_an_unreachable_catalog_is_tried_again(self):
        with mock.patch.object(store_metadata, "_catalog_entry", lambda location, offline: (None, False)):
            self.assertIsNone(self.read().voxel_size_um)
        self.assertEqual(self.read().voxel_size_um, 2.4)


if __name__ == "__main__":
    unittest.main()
