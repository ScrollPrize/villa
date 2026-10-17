import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def sixteen_bit_layer(seed):
    """A layer in the value range of a 16-bit surface-volume render."""
    rng = np.random.default_rng(seed)
    layer = rng.integers(20000, 60000, size=(48, 64), dtype=np.uint16)
    layer[:8, :] = 0  # background outside the segment mask
    return layer


class PrepareLayerReadingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def write_tif(self, name, data):
        path = os.path.join(self.tmp, name)
        tifffile.imwrite(path, data)
        return path

    def test_sixteen_bit_tif_keeps_its_signal(self):
        layer = sixteen_bit_layer(0)
        path = self.write_tif("00.tif", layer)

        image = processing._read_gray_any(path)

        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, (layer >> 8).astype(np.uint8))
        self.assertGreater(len(np.unique(image)), 100)

    def test_sixteen_bit_tif_matches_the_opencv_reader(self):
        path = self.write_tif("00.tif", sixteen_bit_layer(1))

        image = processing._read_gray_any(path)

        np.testing.assert_array_equal(image, cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    def test_eight_bit_tif_is_unchanged(self):
        layer = (sixteen_bit_layer(2) >> 8).astype(np.uint8)
        path = self.write_tif("00.tif", layer)

        np.testing.assert_array_equal(processing._read_gray_any(path), layer)

    def test_surface_volume_from_sixteen_bit_layers_is_not_saturated(self):
        layers = [sixteen_bit_layer(seed) for seed in (3, 4)]
        paths = [self.write_tif(f"{i:02d}.tif", layer) for i, layer in enumerate(layers)]
        out = os.path.join(self.tmp, "surface_volume.zarr")

        processing.create_surface_volume_zarr(paths, out, chunk_size=32, max_workers=1)

        volume = np.asarray(zarr.open(out, mode="r")[:])
        expected = np.stack([(layer >> 8).astype(np.uint8) for layer in layers], axis=2)
        np.testing.assert_array_equal(volume, expected)


if __name__ == "__main__":
    unittest.main()
