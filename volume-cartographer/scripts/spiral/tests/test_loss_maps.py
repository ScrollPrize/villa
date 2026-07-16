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
            self.assertEqual(entries[0]["supported_pixels"], 1)
            self.assertEqual(entries[0]["p50"], 4.0)  # mean residual 2 * weight 2
            image = np.asarray(Image.open(root / entries[0]["path"]))
            self.assertEqual(image.shape, (4, 8, 4))
            self.assertGreater(image[1, 0, 3], 0)
            self.assertEqual(image[3, 7, 3], 0)
            self.assertEqual(published["loss_maps"][0]["name"], "patch_radius")
            on_disk = json.loads((root / "manifest.json").read_text())
            self.assertEqual(on_disk["loss_maps"], published["loss_maps"])


if __name__ == "__main__":
    unittest.main()
