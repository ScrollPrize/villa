from __future__ import annotations

from functools import partial
import http.server
import threading

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter

from lasagna.input_level import InputLevel, split_level_location


def _sparse_level(path):
    """An (8,8,8) uint8 level with chunks only in its z < 4 half."""
    data = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    array = ZarrWriter.create(path, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="|u1", compressor=None, separator="/")
    array[0:4, 0:8, 0:8] = data[:4]
    return data


@pytest.fixture
def served(tmp_path):
    """A local sparse level, also served over plain HTTP (no listing)."""
    remote = tmp_path / "remote"
    data = _sparse_level(remote / "volume.zarr" / "1")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_Quiet, directory=str(remote)),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield remote / "volume.zarr" / "1", f"http://127.0.0.1:{server.server_port}/volume.zarr/1", data
    server.shutdown()
    server.server_close()


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


def test_split_level_location_handles_paths_and_urls():
    assert split_level_location("/data/volume.zarr/2/") == ("/data/volume.zarr", 2)
    assert split_level_location("s3://bucket/volume.zarr/0") == ("s3://bucket/volume.zarr", 0)
    assert split_level_location("https://host/volume.zarr") == ("https://host/volume.zarr", None)


def test_local_level_lists_its_chunks_and_reads_nothing_for_empty_regions(served):
    level, _url, data = served
    with InputLevel(level, read_ahead_tiles=2, connections=2) as source:
        assert source.remote is False and source.level == 1
        assert source.region_has_chunks((0, 4, 0, 8, 0, 8)) is True
        assert source.region_has_chunks((4, 8, 0, 8, 0, 8)) is False
        assert source.region_has_chunks((0, 5, 0, 1, 0, 1)) is True
        assert source.request_region((4, 8, 0, 8, 0, 8)).result() is False
        assert source.request_region((0, 4, 0, 8, 0, 8)).result() is True
        np.testing.assert_array_equal(source.reader[0:4, 0:8, 0:8], data[:4])
    snapshot = source.snapshot()
    assert snapshot["listed_planes"] == 2
    assert snapshot["listed_chunks"] == 4 and snapshot["missing_chunks"] == 4
    assert snapshot["materialized_regions"] == 2


def test_remote_level_streams_through_the_disk_cache(served, tmp_path):
    _level, url, data = served
    cache = tmp_path / "cache"
    with InputLevel(url, cache_dir=cache, cache_max_bytes=1 << 20, read_ahead_tiles=2, connections=2) as source:
        assert source.remote is True
        # A plain HTTP server cannot be listed: every region counts as stored.
        assert source.region_has_chunks((4, 8, 0, 8, 0, 8)) is True
        assert source.request_region((0, 4, 0, 8, 0, 8)).result() is True
        np.testing.assert_array_equal(source.reader[0:4, 0:8, 0:8], data[:4])
    first = source.snapshot()
    assert first["fetched_chunks"] == 4
    assert cache.is_dir()

    with InputLevel(url, cache_dir=cache, read_ahead_tiles=2, connections=2) as source:
        np.testing.assert_array_equal(source.reader[0:4, 0:8, 0:8], data[:4])
    second = source.snapshot()
    assert second["fetched_chunks"] == 0
    assert second["disk_hits"] == 4


def test_closed_level_refuses_requests_and_keeps_its_counters(served):
    level, _url, _data = served
    source = InputLevel(level, read_ahead_tiles=2, connections=2)
    source.reader[0:4, 0:4, 0:4]
    source.close()
    source.close()
    assert source.snapshot()["requested_regions"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        source.request_region((0, 4, 0, 4, 0, 4))


def test_rejects_non_3d_input(tmp_path):
    ZarrWriter.create(tmp_path / "flat.zarr", shape=(8, 8), chunks=(4, 4), dtype="|u1", compressor=None)
    with pytest.raises(ValueError, match="3-D"):
        InputLevel(tmp_path / "flat.zarr")
