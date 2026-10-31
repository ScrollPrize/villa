from __future__ import annotations

from functools import partial
import http.server
import json
from pathlib import Path
import threading
import types

import numpy as np
import pytest
from vc_reader.reader import ZarrWriter

from lasagna import live_omezarr_cache as live


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A 3-D source served over HTTP behind a local level with `_download` metadata."""
    remote = tmp_path / "remote"
    source = remote / "volume.zarr" / "1"
    data = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    array = ZarrWriter.create(source, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="|u1", compressor=None)
    array[0:4, 0:8, 0:8] = data[:4]  # the z >= 4 half is sparse
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(remote)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/volume.zarr"

    def read_json(_bucket, key, _anon, **_kwargs):
        path = remote / key
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def iter_objects(_bucket, prefix, _anon, **_kwargs):
        return [str(p.relative_to(remote)) for p in sorted((remote / prefix).rglob("*")) if p.is_file()]

    monkeypatch.setattr(live, "_downloader", lambda: types.SimpleNamespace(
        _parse_s3_uri=lambda _uri: ("bucket", "volume.zarr"),
        _s3_read_json=read_json,
        _s3_location=lambda location, _anon, _region: location.replace("s3://bucket/volume.zarr", url),
        _inventory_prefix=lambda level_prefix, _separator, iz: f"{level_prefix}/{iz}.",
        _s3_iter_objects=lambda bucket, prefix, anon, **kw: iter_objects(bucket, prefix.rpartition("/")[0], anon),
        _chunk_key_from_s3_object=lambda key, level_prefix, _separator: key.removeprefix(f"{level_prefix}/"),
    ))
    root = tmp_path / "local.zarr"
    root.mkdir()
    (root / ".zattrs").write_text(json.dumps({"_download": {"source": "s3://bucket/volume.zarr", "anon": True}}))
    yield root / "1", data
    server.shutdown()
    server.server_close()


def test_reads_through_the_disk_cache_and_writes_only_metadata_in_the_level(served):
    level, data = served
    with live.LiveOmeZarrCache(level, max_bytes=1 << 20, lookahead_tiles=2, workers=2) as cache:
        reader = cache.open_reader()
        assert cache.request_region((0, 4, 0, 8, 0, 8)).result() is True
        np.testing.assert_array_equal(reader[0:4, 0:8, 0:8], data[:4])
    snapshot = cache.snapshot()
    assert snapshot["downloaded_chunks"] == 4
    assert snapshot["missing_chunks"] == 0
    assert [p.name for p in level.iterdir() if p.is_file()] == [".zarray"]

    # A second run reads the cache.
    with live.LiveOmeZarrCache(level, max_bytes=1 << 20, lookahead_tiles=2, workers=2) as cache:
        np.testing.assert_array_equal(cache.open_reader()[0:4, 0:8, 0:8], data[:4])
    assert cache.snapshot()["downloaded_chunks"] == 0
    assert cache.snapshot()["disk_hits"] == 4


def test_read_ahead_skips_a_region_the_listing_says_is_empty(served):
    level, _data = served
    with live.LiveOmeZarrCache(level, max_bytes=1 << 20, lookahead_tiles=2, workers=2) as cache:
        cache.open_reader()
        bounds = (4, 8, 0, 8, 0, 8)
        assert cache.region_has_remote_chunks(bounds) is False
        assert cache.request_region(bounds).result() is False
    assert cache.snapshot()["downloaded_chunks"] == 0


def test_cache_stays_within_its_target(served):
    level, _data = served
    with live.LiveOmeZarrCache(level, max_bytes=2 * 64, lookahead_tiles=2, workers=2) as cache:
        reader = cache.open_reader()
        for iy in range(2):
            for ix in range(2):
                reader[0:4, iy * 4:iy * 4 + 4, ix * 4:ix * 4 + 4]
    chunks = [p for p in (level.parent / ".live_cache").rglob("*") if p.is_file() and p.stat().st_size == 64]
    assert len(chunks) <= 2


def test_closed_cache_refuses_requests_and_keeps_its_counters(served):
    level, _data = served
    cache = live.LiveOmeZarrCache(level, max_bytes=1 << 20, lookahead_tiles=2, workers=2)
    cache.open_reader()[0:4, 0:4, 0:4]
    cache.close()
    cache.close()
    assert cache.snapshot()["downloaded_chunks"] == 1
    with pytest.raises(RuntimeError, match="closed"):
        cache.request_region((0, 4, 0, 4, 0, 4))


def test_selected_level_lock_rejects_mutator_while_reader_is_active(tmp_path):
    root = tmp_path / "volume.zarr"
    root.mkdir()
    with live.SelectedLevelLock(root, 1, exclusive=False):
        with pytest.raises(RuntimeError, match="locked by another"):
            with live.SelectedLevelLock(root, 1, exclusive=True):
                pass


def test_zarray_compatibility_includes_codec_and_layout(tmp_path):
    local = {
        "shape": [8, 8, 8], "chunks": [2, 2, 2], "dtype": "|u1", "order": "C",
        "compressor": None, "filters": None, "fill_value": 0, "dimension_separator": "/",
    }
    remote = {**local, "compressor": {"id": "zstd", "level": 3}}
    with pytest.raises(ValueError, match="compressor"):
        live._validate_zarray_compatible(local, remote, local_path=tmp_path / ".zarray")
