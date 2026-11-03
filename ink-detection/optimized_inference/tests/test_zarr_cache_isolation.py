import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import zarr
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper
from fsspec.implementations.local import LocalFileSystem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def local_filesystem_as_s3(protocol, **_kwargs):
    """Serve "s3:///abs/path" URLs from the local disk so no bucket is needed."""
    assert protocol == "s3"
    return AsyncFileSystemWrapper(LocalFileSystem(), asynchronous=True)


class ZarrCacheIsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self._env = mock.patch.dict(
            os.environ,
            {"ZARR_CACHE_DIR": os.path.join(self.tmp, "zarr_cache"), "ZARR_MEM_CACHE_MB": "0"},
        )
        self._fs = mock.patch.object(processing.fsspec, "filesystem", local_filesystem_as_s3)
        self._env.start()
        self._fs.start()

    def tearDown(self):
        self._fs.stop()
        self._env.stop()
        self._tmp.cleanup()

    def write_volume(self, name, data):
        path = os.path.join(self.tmp, name)
        array = zarr.open(
            path, mode="w", shape=data.shape, chunks=(4, 4, 2), dtype=np.uint8,
            compressor=None, zarr_format=2,
        )
        array[:] = data
        return "s3://" + path

    def read_through_cache(self, url):
        return np.asarray(zarr.open(processing.get_cached_zarr_store(url), mode="r")[:])

    def test_second_volume_is_not_served_from_first_volumes_cache(self):
        first = np.full((8, 8, 2), 10, dtype=np.uint8)
        second = np.full((8, 8, 2), 200, dtype=np.uint8)
        first_url = self.write_volume("a.zarr", first)
        second_url = self.write_volume("b.zarr", second)

        np.testing.assert_array_equal(self.read_through_cache(first_url), first)
        np.testing.assert_array_equal(self.read_through_cache(second_url), second)

    def test_second_volume_keeps_its_own_shape(self):
        first_url = self.write_volume("a.zarr", np.full((8, 8, 2), 10, dtype=np.uint8))
        second = np.full((12, 16, 2), 200, dtype=np.uint8)
        second_url = self.write_volume("b.zarr", second)

        self.read_through_cache(first_url)
        np.testing.assert_array_equal(self.read_through_cache(second_url), second)

    def test_same_volume_is_served_from_its_cache_on_the_next_open(self):
        data = np.arange(8 * 8 * 2, dtype=np.uint8).reshape(8, 8, 2)
        url = self.write_volume("a.zarr", data)

        np.testing.assert_array_equal(self.read_through_cache(url), data)
        cached_files = sum(len(files) for _, _, files in os.walk(os.environ["ZARR_CACHE_DIR"]))
        self.assertGreater(cached_files, 0)
        np.testing.assert_array_equal(self.read_through_cache(url), data)
        self.assertEqual(
            cached_files,
            sum(len(files) for _, _, files in os.walk(os.environ["ZARR_CACHE_DIR"])),
        )


if __name__ == "__main__":
    unittest.main()
