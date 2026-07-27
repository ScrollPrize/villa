from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lasagna_mmap import prepare_lasagna_mmap


class ArrayFixture:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape
        self.chunks = (2, 2, 2)
        self.dtype = array.dtype
        self.path = "4"
        self.store = None

    def __getitem__(self, item):
        return self.array[item]


class LasagnaMmapTests(unittest.TestCase):
    def make_fixture(self, root):
        paths = {}
        arrays = []
        base = np.arange(4 * 3 * 2, dtype=np.uint8).reshape(4, 3, 2)
        for name, offset in (("normal_x", 1), ("normal_y", 2), ("gradient_magnitude", 3)):
            path = root / name
            path.mkdir()
            paths[name] = str(path)
            arrays.append(ArrayFixture(base + offset))
        return paths, arrays

    def test_channel_last_cache_and_order_preserving_gather(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, arrays = self.make_fixture(root)
            store = prepare_lasagna_mmap(
                nx_array=arrays[0], ny_array=arrays[1], grad_mag_array=arrays[2],
                source_paths=paths, group="4", z_lo=1, z_hi=4, lasagna_scale=4,
                cache_directory=root / "cache", slab_depth=1, workers=2)
            try:
                normals, gradient = store.gather_pair(
                    torch.tensor([[2, 2, 1], [0, 0, 0]]),
                    torch.tensor([[[1, 0, 0], [0, 1, 1]]]), "cpu")
                self.assertEqual(normals.tolist(), [[24, 25], [7, 8]])
                self.assertEqual(gradient.tolist(), [15, 12])
                self.assertEqual(store.normals.shape, (3, 3, 2, 2))
                self.assertIn("gather_seconds", store.last_timings)
            finally:
                store.close()

    def test_probe_change_rebuilds_without_unlinking_old_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, arrays = self.make_fixture(root)
            kwargs = dict(
                nx_array=arrays[0], ny_array=arrays[1], grad_mag_array=arrays[2],
                source_paths=paths, group="4", z_lo=1, z_hi=4, lasagna_scale=4,
                cache_directory=root / "cache", slab_depth=1, workers=2)
            first = prepare_lasagna_mmap(**kwargs)
            destination = first.directory
            first.close()
            arrays[0].array[1, 0, 0] = 199
            second = prepare_lasagna_mmap(**kwargs)
            try:
                self.assertEqual(int(second.normals[0, 0, 0, 0]), 199)
                retired = list(destination.parent.glob(f".{destination.name}.retired-*"))
                self.assertEqual(len(retired), 1)
                self.assertTrue((retired[0] / "manifest.json").is_file())
            finally:
                second.close()


if __name__ == "__main__":
    unittest.main()
