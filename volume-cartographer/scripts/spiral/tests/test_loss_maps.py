import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from loss_maps import (LossMapRecorder, attach_loss_maps_to_manifest,
                       capture_loss_maps, record_loss_samples)


class LossMapRecorderTests(unittest.TestCase):
    def test_bins_weights_and_publishes_surface_aligned_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = root / "preview"
            surface.mkdir()
            Image.fromarray(np.zeros((4, 8), dtype=np.float32)).save(surface / "x.tif")
            manifest = {
                "schema_version": 1,
                "kind": "spiral_combined_preview",
                "surface_id": "preview",
                "components": [[0, 8]],
                "winding_ids": [10],
            }
            (root / "manifest.json").write_text(json.dumps(manifest))

            recorder = LossMapRecorder(
                manifest, root, z0=0, grid_spacing=4,
                dr_per_winding=10, weights={"patch_radius": 2},
            )
            # theta=0, radius=100 -> winding 10; z=4 -> preview row 1.
            points = np.array([[4, 0, 100], [4, 0, 100]], dtype=np.float32)
            with capture_loss_maps(recorder):
                record_loss_samples(
                    "patch_radius", points,
                    np.array([1, 3], dtype=np.float32),
                )
            entries = recorder.finish()
            published = attach_loss_maps_to_manifest(manifest, root, entries)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["name"], "patch_radius")
            self.assertEqual(entries[0]["sample_count"], 2)
            self.assertEqual(entries[0]["eligible_sample_count"], 2)
            self.assertEqual(entries[0]["projected_sample_count"], 2)
            self.assertEqual(entries[0]["off_surface_sample_count"], 0)
            self.assertEqual(entries[0]["omitted_sample_count"], 0)
            self.assertEqual(entries[0]["supported_pixels"], 1)
            self.assertEqual(entries[0]["p50"], 4.0)  # mean residual 2 * weight 2
            image = np.asarray(Image.open(root / entries[0]["path"]))
            self.assertEqual(image.shape, (4, 8, 4))
            self.assertGreater(image[1, 0, 3], 0)
            self.assertEqual(image[3, 7, 3], 0)
            self.assertEqual(published["loss_maps"][0]["name"], "patch_radius")
            on_disk = json.loads((root / "manifest.json").read_text())
            self.assertEqual(on_disk["loss_maps"], published["loss_maps"])

    def test_explicit_display_coordinates_and_invalid_surface_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface = root / "preview"
            surface.mkdir()
            x = np.zeros((4, 8), dtype=np.float32)
            x[2, 0] = -1.0
            Image.fromarray(x).save(surface / "x.tif")
            manifest = {
                "schema_version": 1,
                "kind": "spiral_combined_preview",
                "surface_id": "preview",
                "components": [[0, 8]],
                "winding_ids": [10],
            }

            recorder = LossMapRecorder(
                manifest, root, z0=0, grid_spacing=4,
                dr_per_winding=10, weights={"patch_dt": 2},
            )
            # All source samples sit on winding 9.  Explicit display points put
            # two on winding 10 (one valid and one invalid preview vertex); the
            # third remains on an unavailable winding and is omitted.
            source = np.array(
                [[4, 0, 90], [8, 0, 90], [4, 0, 90]], dtype=np.float32)
            display = np.array(
                [[4, 0, 100], [8, 0, 100], [4, 0, 90]], dtype=np.float32)
            with capture_loss_maps(recorder):
                record_loss_samples(
                    "patch_dt", source,
                    np.array([1, 2, 3], dtype=np.float32),
                    display_spiral_zyx=display,
                )

            entries = recorder.finish()
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["eligible_sample_count"], 3)
            self.assertEqual(entry["projected_sample_count"], 2)
            self.assertEqual(entry["off_surface_sample_count"], 1)
            self.assertEqual(entry["omitted_sample_count"], 1)
            self.assertEqual(entry["sample_count"], 1)
            self.assertEqual(entry["supported_pixels"], 1)

            image = np.asarray(Image.open(root / entry["path"]))
            self.assertGreater(image[1, 0, 3], 0)
            self.assertEqual(image[2, 0, 3], 0)


if __name__ == "__main__":
    unittest.main()
