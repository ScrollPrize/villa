"""A reader exposes what its store's sidecar documents say, read once at open."""
import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from vc_reader.reader import ZarrReader, open_group


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


if __name__ == "__main__":
    unittest.main()
