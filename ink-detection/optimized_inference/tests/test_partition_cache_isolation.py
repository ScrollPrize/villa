import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


class PartitionCacheIsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        # No create=True: without the constant this fails here instead of using the real /tmp.
        self._cache = mock.patch.object(
            processing, "PARTITION_CACHE_DIR", os.path.join(self.tmp, "partition_cache")
        )
        self._cache.start()

    def tearDown(self):
        self._cache.stop()
        self._tmp.cleanup()

    def write_partition(self, name, pred, part_id=0):
        """Write one partition like inference.py does: zarr v2, empty chunks not stored."""
        out_dir = os.path.join(self.tmp, name)
        for kind, data in (("pred", pred), ("count", np.ones_like(pred))):
            array = zarr.open(
                os.path.join(out_dir, f"mask_{kind}_part_{part_id:03d}.zarr"), mode="w",
                shape=data.shape, chunks=(4, 4), dtype=np.float32, compressor=None, zarr_format=2,
                config={"write_empty_chunks": False},
            )
            array[:] = data
        return out_dir

    def reduce(self, out_dir, num_parts=1):
        tiles, _shape = processing.reduce_partitions(out_dir, num_parts, (8, 8), tile_size=8)
        (tile,) = tiles
        return tile

    def test_second_output_dir_is_not_served_from_first_runs_cache(self):
        first = self.write_partition("run_a", np.full((8, 8), 0.25, dtype=np.float32))
        second = self.write_partition("run_b", np.full((8, 8), 0.75, dtype=np.float32))

        np.testing.assert_array_equal(self.reduce(first), np.full((8, 8), 63, dtype=np.uint8))
        np.testing.assert_array_equal(self.reduce(second), np.full((8, 8), 191, dtype=np.uint8))

    def test_rewritten_output_dir_is_read_again(self):
        # Every inference run rewrites the default ZARR_OUTPUT_DIR (/tmp/partitions) in place.
        out_dir = self.write_partition("partitions", np.full((8, 8), 0.75, dtype=np.float32))
        self.reduce(out_dir)

        pred = np.zeros((8, 8), dtype=np.float32)
        pred[:4, :4] = 0.25  # the other three pred chunks are all zero, so zarr does not store them
        self.write_partition("partitions", pred)

        expected = np.zeros((8, 8), dtype=np.uint8)
        expected[:4, :4] = 63
        np.testing.assert_array_equal(self.reduce(out_dir), expected)

    def test_more_partitions_are_not_blended_with_last_runs(self):
        out_dir = self.write_partition("partitions", np.full((8, 8), 0.25, dtype=np.float32))
        self.reduce(out_dir)

        for part_id in (0, 1):
            self.write_partition("partitions", np.full((8, 8), 0.75, dtype=np.float32), part_id)

        np.testing.assert_array_equal(
            self.reduce(out_dir, num_parts=2), np.full((8, 8), 191, dtype=np.uint8)
        )


if __name__ == "__main__":
    unittest.main()
