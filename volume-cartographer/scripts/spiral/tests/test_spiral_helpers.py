import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from spiral_helpers import (
    load_fiber_point_collection,
    patch_dir_may_intersect_z_roi,
    patch_intersects_z_roi,
)
from tifxyz import load_tifxyz


class FiberPointCollectionTests(unittest.TestCase):
    def _write_fiber(self, directory, data):
        path = Path(directory) / "fiber.json"
        path.write_text(json.dumps(data))
        return path

    def test_loads_control_points_instead_of_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "control_points": [[4, 8, 12], [20, 24, 28]],
                "line_points": [[400, 800, 1200]],
            })

            collection = load_fiber_point_collection(
                path, collection_id=7, min_point_spacing=0)

            points = [point["p"] for point in collection["points"].values()]
            np.testing.assert_array_equal(points, [[1, 2, 3], [5, 6, 7]])

    def test_does_not_fall_back_to_line_points(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self._write_fiber(temporary, {
                "line_points": [[4, 8, 12]],
            })

            collection = load_fiber_point_collection(path, collection_id=7)

            self.assertIsNone(collection)


class TifxyzMetadataTests(unittest.TestCase):
    def _write_patch(self, root, metadata):
        (root / "meta.json").write_text(json.dumps(metadata))
        values = np.ones((2, 2), dtype=np.float32)
        for coordinate in "zyx":
            Image.fromarray(values).save(root / f"{coordinate}.tif")

    def test_patch_can_override_configured_erosion_with_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_patch(root, {
                "format": "tifxyz",
                "scale": [1.0, 1.0],
                "spiral_patch_erode_cells": 0,
            })

            patch = load_tifxyz(root)

            self.assertEqual(patch.erosion_cells(7), 0)

    def test_ordinary_patch_uses_configured_erosion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_patch(root, {"format": "tifxyz", "scale": [1.0, 1.0]})

            patch = load_tifxyz(root)

            self.assertEqual(patch.erosion_cells(7), 7)


class PatchDirZRoiPrefilterTests(unittest.TestCase):
    def _write_patch(self, root, zs, bbox=None, write_meta=True):
        """A 2x2 patch whose z grid is `zs`; bbox defaults to the true extent."""
        root.mkdir(parents=True, exist_ok=True)
        z_grid = np.asarray(zs, dtype=np.float32)
        for coordinate, values in (
            ("z", z_grid),
            ("y", np.zeros_like(z_grid)),
            ("x", np.zeros_like(z_grid)),
        ):
            Image.fromarray(values).save(root / f"{coordinate}.tif")
        if write_meta:
            if bbox is None:
                bbox = [[0.0, 0.0, float(z_grid.min())], [0.0, 0.0, float(z_grid.max())]]
            (root / "meta.json").write_text(json.dumps({
                "format": "tifxyz", "scale": [1.0, 1.0], "bbox": bbox,
            }))
        return root

    def test_excludes_patch_entirely_outside_the_roi(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = self._write_patch(Path(temporary) / "p", [[500, 501], [502, 503]])
            self.assertFalse(patch_dir_may_intersect_z_roi(root, 100, 200))
            # and the authoritative check agrees
            self.assertFalse(patch_intersects_z_roi(load_tifxyz(root), 100, 200))

    def test_keeps_overlapping_and_boundary_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            inside = self._write_patch(base / "inside", [[120, 130], [140, 150]])
            straddling = self._write_patch(base / "straddling", [[80, 90], [100, 110]])
            for root in (inside, straddling):
                self.assertTrue(patch_dir_may_intersect_z_roi(root, 100, 200))
                self.assertTrue(patch_intersects_z_roi(load_tifxyz(root), 100, 200))

    def test_conservative_when_metadata_is_unusable(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            no_meta = self._write_patch(base / "no_meta", [[500, 501], [502, 503]],
                                        write_meta=False)
            no_bbox = base / "no_bbox"
            no_bbox.mkdir()
            (no_bbox / "meta.json").write_text(json.dumps({"scale": [1.0, 1.0]}))
            sentinel = self._write_patch(
                base / "sentinel", [[500, 501], [502, 503]],
                bbox=[[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
            broken = base / "broken"
            broken.mkdir()
            (broken / "meta.json").write_text("{not json")

            for root in (no_meta, no_bbox, sentinel, broken):
                self.assertTrue(patch_dir_may_intersect_z_roi(root, 100, 200))


if __name__ == "__main__":
    unittest.main()
