# SPDX-License-Identifier: MIT
"""Generated-data regression checks for paired TIFF export coverage.

Run with the intended vesuvius/src on PYTHONPATH. Real-CT evidence is separate.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import tifffile
import zarr

from vesuvius.image_proc.run.zarr_tasks.tasks.export_tifs import ExportTifsConfig, ExportTifsTask


class PairedExportBounds(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.values = np.arange(512, dtype=np.uint16).reshape(8, 8, 8) + 1
        self.first = self.store("first", self.values)
        self.second = self.store("second", self.values[:6])

    def store(self, name, values):
        path = self.root / (name + ".zarr")
        kw = {"zarr_format": 2} if int(zarr.__version__.split(".")[0]) >= 3 else {}
        zarr.open_group(str(path), mode="w", **kw)
        downsampled = values[(slice(None, None, 2),) * min(3, values.ndim)]
        for level, data in (("0", values), ("1", downsampled)):
            a = zarr.open_array(str(path / level), mode="w", shape=data.shape,
                                chunks=tuple(min(4, d) for d in data.shape), dtype=data.dtype, **kw)
            a[:] = data
        return path

    def task(self, *, second=None, bbox=None, resolution="0"):
        return ExportTifsTask(ExportTifsConfig(
            input_zarr=str(self.first), output_zarr=str(self.root / "images"),
            zarr2_path=str(second or self.second), out2_dir=str(self.root / "targets"),
            num_workers=2, chunk_size=4, bbox=bbox, resolution=resolution))

    def assert_no_output(self):
        self.assertFalse((self.root / "images").exists())
        self.assertFalse((self.root / "targets").exists())

    def test_full_coverage_exact(self):
        self.task(second=self.first).run()
        paths = sorted((self.root / "images").glob("*.tif"))
        self.assertEqual(len(paths), 8)
        for path in paths:
            np.testing.assert_array_equal(tifffile.imread(path), tifffile.imread(self.root / "targets" / path.name))

    def test_smaller_secondary_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, "does not cover requested export"):
            self.task().prepare()
        self.assert_no_output()

    def test_common_bbox_allowed(self):
        self.task(bbox="1:5,1:7,1:7").run()
        paths = sorted((self.root / "images").glob("*.tif"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            np.testing.assert_array_equal(tifffile.imread(path), tifffile.imread(self.root / "targets" / path.name))

    def test_incomplete_bbox_preserves_existing_output(self):
        for name in ("images", "targets"):
            folder = self.root / name; folder.mkdir()
            (folder / "keep.txt").write_text("existing output\n")
        with self.assertRaisesRegex(ValueError, "does not cover requested export"):
            self.task(bbox="4:8,0:8,0:8").prepare()
        for name in ("images", "targets"):
            folder = self.root / name
            self.assertEqual([p.name for p in folder.iterdir()], ["keep.txt"])
            self.assertEqual((folder / "keep.txt").read_text(), "existing output\n")

    def test_nonzero_resolution_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not cover requested export"):
            self.task(resolution="1").prepare()
        self.assert_no_output()

    def test_non3d_primary_rejected(self):
        self.first = self.store("flat", self.values[0])
        with self.assertRaisesRegex(ValueError, "3D input array"):
            self.task().prepare()
        self.assert_no_output()

    def test_non3d_secondary_rejected(self):
        extra_axis = self.store("channels", np.stack([self.values, self.values], axis=-1))
        with self.assertRaisesRegex(ValueError, "3D second array"):
            self.task(second=extra_axis).prepare()
        self.assert_no_output()


if __name__ == "__main__":
    unittest.main(verbosity=2)
