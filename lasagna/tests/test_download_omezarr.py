import queue
import threading
import json
from pathlib import Path

import pytest

from lasagna.scripts import download_omezarr as dl


@pytest.mark.parametrize("payload", ("", "{", "{}", "null", "[1]", '["ok", 2]'))
def test_load_noremote_ignores_malformed_or_schema_invalid_cache(tmp_path, capsys, payload):
    cache = tmp_path / ".dl_cache" / "2.noremote.json"
    cache.parent.mkdir()
    cache.write_text(payload, encoding="utf-8")

    assert dl._load_noremote(str(tmp_path), [2]) == {2: set()}
    warning = capsys.readouterr().err
    assert "ignoring invalid noremote cache for level 2" in warning
    assert str(cache) in warning


def test_load_noremote_ignores_unreadable_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / ".dl_cache" / "1.noremote.json"
    cache.parent.mkdir()
    cache.write_text('[]\n', encoding="utf-8")
    real_open = open

    def fail_cache_open(path, *args, **kwargs):
        if Path(path) == cache:
            raise OSError("injected read failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fail_cache_open)
    assert dl._load_noremote(str(tmp_path), [1]) == {1: set()}
    assert "injected read failure" in capsys.readouterr().err


def test_noremote_cache_valid_load_and_atomic_empty_save(tmp_path):
    cache = tmp_path / ".dl_cache" / "0.noremote.json"
    cache.parent.mkdir()
    cache.write_text('["0.0.1", "0.0.2"]\n', encoding="utf-8")
    assert dl._load_noremote(str(tmp_path), [0]) == {0: {"0.0.1", "0.0.2"}}

    dl._save_noremote(str(tmp_path), {0: set()})
    assert json.loads(cache.read_text(encoding="utf-8")) == []
    assert list(cache.parent.glob("0.noremote.json.tmp.*")) == []


def test_noremote_atomic_replace_failure_keeps_old_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / ".dl_cache" / "0.noremote.json"
    cache.parent.mkdir()
    cache.write_text('["old"]\n', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(dl.os, "replace", fail_replace)
    dl._save_noremote(str(tmp_path), {0: {"new"}})
    assert json.loads(cache.read_text(encoding="utf-8")) == ["old"]
    assert list(cache.parent.glob("0.noremote.json.tmp.*")) == []
    assert "could not save advisory noremote cache" in capsys.readouterr().err


@pytest.mark.parametrize("failure_stage", ("open", "dump"))
def test_noremote_atomic_write_failure_keeps_old_cache(
    tmp_path, monkeypatch, capsys, failure_stage,
):
    cache = tmp_path / ".dl_cache" / "0.noremote.json"
    cache.parent.mkdir()
    cache.write_text('["old"]\n', encoding="utf-8")
    if failure_stage == "open":
        real_open = open

        def fail_temp_open(path, *args, **kwargs):
            if ".tmp." in str(path):
                raise OSError("injected temp open failure")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", fail_temp_open)
    else:
        monkeypatch.setattr(
            dl.json, "dump",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("injected dump failure")),
        )

    dl._save_noremote(str(tmp_path), {0: {"new"}})
    assert json.loads(cache.read_text(encoding="utf-8")) == ["old"]
    assert list(cache.parent.glob("0.noremote.json.tmp.*")) == []
    assert "could not save advisory noremote cache" in capsys.readouterr().err


def test_download_rejects_non_positive_worker_count():
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        dl.download("s3://bucket/volume.zarr", "volume.zarr", workers=0)


def test_s3_client_uses_bounded_standard_retries(monkeypatch):
    captured = {}

    class FakeSession:
        def client(self, service, *, config, region_name):
            captured.update(service=service, config=config, region=region_name)
            return object()

    monkeypatch.setattr(dl.boto3, "Session", FakeSession)
    dl._thread_local.s3_client_cache = None
    dl._get_s3_client(True, "us-east-1")

    assert captured["service"] == "s3"
    assert captured["region"] == "us-east-1"
    assert captured["config"].retries == {
        "mode": "standard",
        "total_max_attempts": 5,
    }


def test_stats_snapshots_noremote_sets_by_value():
    stats = dl.Stats()
    stats.noremote_keys = {0: {"0.0.0"}}
    snapshot = stats.snapshot_noremote()
    stats.add_noremote(0, "0.0.1")
    assert snapshot == {0: {"0.0.0"}}


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


def test_scanner_transport_failure_is_reported_to_coordinator(tmp_path, monkeypatch):
    def fail_iter(_bucket, _prefix, _anon):
        raise dl.botocore.exceptions.ReadTimeoutError(endpoint_url="https://example.invalid")

    monkeypatch.setattr(dl, "_s3_iter_objects", fail_iter)
    stats = dl.Stats()
    q: queue.Queue = queue.Queue()

    dl._guarded_scanner(
        "bucket",
        "vol.zarr",
        str(tmp_path / "vol.zarr"),
        {0: {"shape": (2, 2, 2), "chunks": (2, 2, 2), "dim_sep": "."}},
        None,
        {0: [1.0, 1.0, 1.0]},
        {0: set()},
        True,
        True,
        "z",
        q,
        stats,
        threading.Event(),
    )

    snapshot = stats.snapshot()
    assert snapshot["scan_done"] is True
    assert snapshot["inventory_done"] == 0
    assert snapshot["inventory_failed"] == 1
    assert "Read timeout" in snapshot["scan_error"]
    assert "s3://bucket/vol.zarr/0/0." in snapshot["scan_error"]
    with pytest.raises(dl._ScannerFailure, match="Read timeout"):
        dl._drain_download_queue(
            q, stats, bucket="bucket", anon=True, workers=2,
            local_root=str(tmp_path / "vol.zarr"), levels=[0],
        )


def test_progress_reporter_is_silent_while_idle_and_writes_history_for_progress():
    stats = dl.Stats()
    reporter = dl._ProgressReporter(history_interval=60.0)

    assert reporter.update(stats.snapshot(), 0.0) == ("[starting scan...]", False)
    assert reporter.update(stats.snapshot(), 30.0) is None
    assert reporter.update(stats.snapshot(), 61.0) is None

    stats.inc(total_chunks=10, scanned=1, local=1)
    live = reporter.update(stats.snapshot(), 62.0)
    assert live is not None and live[1] is True
    assert "scan 10.0%" in live[0]

    assert reporter.update(stats.snapshot(), 123.0) is None
    stats.inc(scanned=1, remote=1)
    assert reporter.update(stats.snapshot(), 124.0)[1] is True
    assert reporter.update(stats.snapshot(), 185.0) is None


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
