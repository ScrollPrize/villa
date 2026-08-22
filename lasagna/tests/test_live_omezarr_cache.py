from __future__ import annotations

import json
from pathlib import Path
import types

import pytest

from lasagna import live_omezarr_cache as live


def _source(root: Path, *, separator: str = "/") -> live.SelectedLevelSource:
    level_path = root / "1"
    level_path.mkdir(parents=True, exist_ok=True)
    zarray = {
        "zarr_format": 2,
        "shape": [8, 8, 8],
        "chunks": [2, 2, 2],
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "dimension_separator": separator,
    }
    (level_path / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")
    return live.SelectedLevelSource(
        group_root=root,
        level_path=level_path,
        level=1,
        source_uri="s3://bucket/volume.zarr",
        bucket="bucket",
        prefix="volume.zarr",
        anon=True,
        region=None,
        shape=(8, 8, 8),
        chunks=(2, 2, 2),
        dimension_separator=separator,
        zarray=zarray,
    )


def _write_chunk(source: live.SelectedLevelSource, iz: int, iy: int, ix: int, data: bytes) -> Path:
    key = source.dimension_separator.join(str(value) for value in (iz, iy, ix))
    path = source.level_path / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize("separator", ["/", "."])
def test_inventory_counts_only_regular_selected_level_chunks(tmp_path, separator):
    source = _source(tmp_path / "volume.zarr", separator=separator)
    _write_chunk(source, 0, 0, 0, b"abc")
    _write_chunk(source, 1, 0, 0, b"defgh")
    (source.level_path / "not-a-chunk").write_bytes(b"ignored")
    (source.group_root / "2").mkdir()
    (source.group_root / "2" / "0.0.0").write_bytes(b"other-scale")
    symlink = source.level_path / ("2/0/0" if separator == "/" else "2.0.0")
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to(source.group_root / "2" / "0.0.0")

    plane_bytes, plane_counts = live.inventory_selected_level(source)

    assert plane_bytes == {0: 3, 1: 5}
    assert plane_counts == {0: 1, 1: 1}


def test_live_cache_evicts_only_complete_safe_z_planes(tmp_path, monkeypatch):
    source = _source(tmp_path / "volume.zarr")
    first = _write_chunk(source, 0, 0, 0, b"aaaa")
    second = _write_chunk(source, 1, 0, 0, b"bbbb")
    protected = _write_chunk(source, 2, 0, 0, b"cccc")
    other_scale = source.group_root / "2" / "0" / "0" / "0"
    other_scale.parent.mkdir(parents=True)
    other_scale.write_bytes(b"other")
    monkeypatch.setattr(live, "prepare_selected_level_source", lambda _path: source)

    with live.LiveOmeZarrCache(source.level_path, max_bytes=3, lookahead_tiles=2, workers=1) as cache:
        cache.advance_safe_boundary(2)
        snapshot = cache.snapshot()
        assert not first.exists()
        assert second.exists()
        assert protected.exists()
        assert snapshot["resident_bytes"] == 8
        assert snapshot["over_target_events"] == 1

        cache.advance_safe_boundary(4)
        snapshot = cache.snapshot()
        assert not second.exists()
        assert protected.exists()
        assert snapshot["resident_bytes"] == 4

        cache.advance_safe_boundary(6)
        assert not protected.exists()
        assert cache.snapshot()["resident_bytes"] == 0

    assert other_scale.read_bytes() == b"other"
    assert (source.level_path / ".zarray").is_file()


def test_projected_or_inflight_state_does_not_trigger_eviction(tmp_path, monkeypatch):
    source = _source(tmp_path / "volume.zarr")
    chunk = _write_chunk(source, 0, 0, 0, b"aaaa")
    monkeypatch.setattr(live, "prepare_selected_level_source", lambda _path: source)

    with live.LiveOmeZarrCache(source.level_path, max_bytes=8, lookahead_tiles=2, workers=1) as cache:
        with cache._lock:
            cache._stats["projected_bytes"] = 1000
        cache.advance_safe_boundary(2)
        assert chunk.exists()
        assert cache.snapshot()["resident_bytes"] == 4


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


def test_materialization_lists_authoritatively_deduplicates_and_keeps_missing_sparse(
    tmp_path, monkeypatch,
):
    source = _source(tmp_path / "volume.zarr")
    noremote = source.group_root / ".dl_cache" / "1.noremote.json"
    noremote.parent.mkdir()
    noremote.write_text('["0/0/0"]\n', encoding="utf-8")
    monkeypatch.setattr(live, "prepare_selected_level_source", lambda _path: source)
    downloaded = []

    def iter_objects(_bucket, prefix, _anon, **_kwargs):
        assert prefix == "volume.zarr/1/0/"
        yield "volume.zarr/1/0/0/0"

    def download_atomic(_bucket, key, local_path, _anon, **_kwargs):
        downloaded.append(key)
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"chunk")
        return 5

    fake_downloader = types.SimpleNamespace(
        _inventory_prefix=lambda level_prefix, separator, iz: f"{level_prefix}/{iz}/",
        _s3_iter_objects=iter_objects,
        _chunk_key_from_s3_object=lambda key, level_prefix, _separator: key.removeprefix(f"{level_prefix}/"),
        _download_chunk_atomic=download_atomic,
    )
    monkeypatch.setattr(live, "_downloader", lambda: fake_downloader)

    with live.LiveOmeZarrCache(source.level_path, max_bytes=100, lookahead_tiles=2, workers=2) as cache:
        bounds = (0, 2, 0, 2, 0, 4)
        first = cache.request_region(bounds)
        second = cache.request_region(bounds)
        assert first.result() is True
        assert second.result() is True
        assert cache.region_has_remote_chunks(bounds) is True
        snapshot = cache.snapshot()

    assert downloaded == ["volume.zarr/1/0/0/0"]
    assert snapshot["downloaded_chunks"] == 1
    assert snapshot["resident_bytes"] == 5
    assert snapshot["missing_chunks"] == 15
    assert not (source.level_path / "0" / "0" / "1").exists()
    assert noremote.read_text(encoding="utf-8") == '["0/0/0"]\n'
