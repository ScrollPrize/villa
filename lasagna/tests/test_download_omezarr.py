import queue
import threading

from lasagna.scripts import download_omezarr as dl


def _drain_chunk_keys(q: queue.Queue) -> list[str]:
    keys: list[str] = []
    while not q.empty():
        item = q.get_nowait()
        keys.append(item[3])
    return keys


def test_scanner_remote_inventory_filters_missing_chunks(tmp_path, monkeypatch):
    local_root = tmp_path / "vol.zarr"
    local_level = local_root / "0"
    local_level.mkdir(parents=True)
    (local_level / "0.0.0").write_bytes(b"cached")

    levels_meta = {0: {"shape": (4, 4, 4), "chunks": (2, 2, 2), "dim_sep": "."}}
    local_keys = {0: dl._list_local_chunk_keys(str(local_level), ".")}
    stats = dl.Stats()
    q: queue.Queue = queue.Queue()
    listed_prefixes: list[str] = []

    def fake_iter(_bucket, prefix, _anon):
        listed_prefixes.append(prefix)
        if prefix.endswith("/0."):
            yield "vol.zarr/0/0.0.1"
            assert q.qsize() == 1
            assert list(q.queue)[0][3] == "0.0.1"
            yield "vol.zarr/0/0.1.0"

    monkeypatch.setattr(dl, "_s3_iter_objects", fake_iter)

    dl._scanner(
        "bucket",
        "vol.zarr",
        str(local_root),
        levels_meta,
        None,
        {0: [1.0, 1.0, 1.0]},
        local_keys,
        True,
        True,
        "z",
        q,
        stats,
        threading.Event(),
    )

    snap = stats.snapshot()
    assert snap["total"] == 8
    assert snap["scanned"] == 8
    assert snap["local"] == 1
    assert snap["remote"] == 2
    assert snap["missing_remote"] == 5
    assert snap["inventory_total"] == 2
    assert snap["inventory_done"] == 2
    assert snap["inventory_objects"] == 2
    assert _drain_chunk_keys(q) == ["0.0.1", "0.1.0"]
    assert listed_prefixes == ["vol.zarr/0/0.", "vol.zarr/0/1."]


def test_scanner_skips_remote_inventory_when_all_chunks_accounted(tmp_path, monkeypatch):
    local_root = tmp_path / "vol.zarr"
    local_level = local_root / "0"
    local_level.mkdir(parents=True)
    (local_level / "0.0.0").write_bytes(b"cached")

    def fail_iter(*_args, **_kwargs):
        raise AssertionError("remote inventory should be skipped")

    monkeypatch.setattr(dl, "_s3_iter_objects", fail_iter)

    levels_meta = {0: {"shape": (2, 4, 4), "chunks": (2, 2, 2), "dim_sep": "."}}
    stats = dl.Stats()
    stats.noremote_keys = {0: {"0.0.1", "0.1.0", "0.1.1"}}
    q: queue.Queue = queue.Queue()

    dl._scanner(
        "bucket",
        "vol.zarr",
        str(local_root),
        levels_meta,
        None,
        {0: [1.0, 1.0, 1.0]},
        {0: dl._list_local_chunk_keys(str(local_level), ".")},
        True,
        True,
        "z",
        q,
        stats,
        threading.Event(),
    )

    snap = stats.snapshot()
    assert snap["inventory_total"] == 0
    assert snap["local"] == 1
    assert snap["missing_remote"] == 3
    assert q.empty()


def test_scanner_uses_cached_missing_only_without_remote_inventory(tmp_path):
    local_root = tmp_path / "vol.zarr"
    local_level = local_root / "0"
    local_level.mkdir(parents=True)

    levels_meta = {0: {"shape": (4, 4, 4), "chunks": (2, 2, 2), "dim_sep": "."}}
    stats = dl.Stats()
    stats.noremote_keys = {0: {"0.0.1"}}
    q: queue.Queue = queue.Queue()

    dl._scanner(
        "bucket",
        "vol.zarr",
        str(local_root),
        levels_meta,
        None,
        {0: [1.0, 1.0, 1.0]},
        {0: set()},
        True,
        False,
        "z",
        q,
        stats,
        threading.Event(),
    )

    snap = stats.snapshot()
    assert snap["remote"] == 7
    assert snap["missing_remote"] == 1
    assert "0.0.1" not in _drain_chunk_keys(q)


def test_remaining_download_estimate_counts_queued_404s_as_resolved():
    remaining, estimated = dl._remaining_download_estimate(
        {
            "scan_done": True,
            "total": 10,
            "scanned": 10,
            "remote": 5,
            "downloaded": 2,
            "download_missing": 2,
            "failed": 0,
        }
    )

    assert remaining == 1
    assert estimated is False


def test_remote_chunk_listing_ignores_metadata_and_non_chunk_keys(monkeypatch):
    def fake_list(_bucket, _prefix, _anon):
        return [
            "root/0/.zarray",
            "root/0/.zattrs",
            "root/0/0.0.0",
            "root/0/0.0.1.tmp",
            "root/0/1.2.3",
        ]

    monkeypatch.setattr(dl, "_s3_list_objects", fake_list)

    assert dl._list_remote_chunk_keys("bucket", "root/0", ".", True) == {
        "0.0.0",
        "1.2.3",
    }


def test_remote_chunk_listing_can_be_limited_to_expected_z_prefixes(monkeypatch):
    calls: list[str] = []

    def fake_list(_bucket, prefix, _anon):
        calls.append(prefix)
        if prefix.endswith("/1."):
            return ["root/0/1.0.0", "root/0/1.9.9"]
        if prefix.endswith("/3."):
            return ["root/0/3.0.0"]
        return []

    monkeypatch.setattr(dl, "_s3_list_objects", fake_list)

    assert dl._list_remote_chunk_keys(
        "bucket",
        "root/0",
        ".",
        True,
        expected_indices=[(1, 0, 0), (3, 0, 0)],
    ) == {
        "1.0.0",
        "3.0.0",
    }
    assert calls == ["root/0/1.", "root/0/3."]
