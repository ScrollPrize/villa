import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import processing  # noqa: E402


def no_credentials():
    session = mock.Mock()
    session.get_credentials.return_value = None
    return mock.patch("botocore.session.get_session", return_value=session)


def with_credentials():
    session = mock.Mock()
    session.get_credentials.return_value = mock.Mock(name="credentials")
    return mock.patch("botocore.session.get_session", return_value=session)


class S3AnonymousTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._env = mock.patch.dict(os.environ, {"ZARR_CACHE_DIR": os.path.join(self._tmp.name, "zarr_cache"), "ZARR_MEM_CACHE_MB": "0"})
        self._env.start()
        processing.s3_anonymous.cache_clear()

    def tearDown(self):
        processing.s3_anonymous.cache_clear()
        self._env.stop()
        self._tmp.cleanup()

    def cached_store_anon(self):
        with mock.patch.object(processing.fsspec, "filesystem") as filesystem, \
                mock.patch.object(processing, "FsspecStore"), mock.patch.object(processing, "CacheStore"), \
                mock.patch.object(processing, "LocalStore"):
            processing.get_cached_zarr_store("s3://bucket/volume.zarr")
        return filesystem.call_args.kwargs["anon"]

    def test_cached_store_reads_anonymously_without_credentials(self):
        with no_credentials():
            self.assertTrue(self.cached_store_anon())

    def test_cached_store_keeps_using_credentials_when_present(self):
        with with_credentials():
            self.assertFalse(self.cached_store_anon())

    def test_path_exists_reads_anonymously_without_credentials(self):
        s3fs = mock.Mock()
        s3fs.S3FileSystem.return_value.exists.return_value = True
        with no_credentials(), mock.patch.dict(sys.modules, {"s3fs": s3fs}):
            self.assertTrue(processing.path_exists("s3://bucket/volume.zarr/.zarray"))
        self.assertTrue(s3fs.S3FileSystem.call_args.kwargs["anon"])


if __name__ == "__main__":
    unittest.main()
