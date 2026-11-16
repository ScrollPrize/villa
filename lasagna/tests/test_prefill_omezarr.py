from __future__ import annotations

from functools import partial
import http.server
import json
import threading

import numpy as np
import pytest
from vc_reader.reader import ZarrReader, ZarrWriter, create_group

from lasagna.scripts import prefill_omezarr as prefill
from lasagna.scripts import prefill_volume_list as batch


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def served(tmp_path):
    """An OME-Zarr group with levels 0 (8³, chunks 4³, sparse) and 1 (4³), served over HTTP.

    The multiscale metadata states physical voxel spacing (2.4 µm at level 0), as
    the open-data volumes do; placement normalizes it to level-0 voxels.
    """
    remote = tmp_path / "remote"
    group = remote / "volume.zarr"
    create_group(group).attrs["multiscales"] = [{"version": "0.4", "axes": [
        {"name": a, "type": "space", "unit": "micrometer"} for a in "zyx"], "datasets": [
        {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.4, 2.4, 2.4]}]},
        {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [4.8, 4.8, 4.8]}]},
    ]}]
    data = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
    level0 = ZarrWriter.create(group / "0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="|u1", compressor=None, separator="/")
    level0[0:4, 0:8, 0:8] = data[:4]  # only the z < 4 half is stored
    level1 = ZarrWriter.create(group / "1", shape=(4, 4, 4), chunks=(4, 4, 4), dtype="|u1", compressor=None, separator="/")
    level1[:] = data[::2, ::2, ::2]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), partial(_Quiet, directory=str(remote)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/volume.zarr", data
    server.shutdown()
    server.server_close()


def test_prefill_level_fills_the_cache_once(served, tmp_path):
    url, data = served
    cache = tmp_path / "cache"
    progress = []
    result = prefill.prefill_level(prefill.level_location(url, 0), str(cache), anon=True, connections=2,
                                   progress=lambda done, total: progress.append((done, total)))
    assert result.total_chunks == 8
    assert result.fetched_chunks == 8  # every chunk was asked for once ...
    assert (result.data_chunks, result.empty_chunks) == (4, 4)  # ... half exist, half are remembered absent
    assert progress and progress[-1] == (8, 8)
    assert cache.is_dir()
    # A later reader is served from the cache alone.
    reader = ZarrReader.open(f"{url}/0", cache_dir=str(cache), auth=None, connections=2)
    try:
        np.testing.assert_array_equal(reader[0:4, 0:8, 0:8], data[:4])
        assert reader[4:8, 0:8, 0:8].max() == 0
        stats = reader.stats()
    finally:
        reader.close()
    assert stats["remote_fetches"] == 0 and stats["remote_missing"] == 0
    assert stats["disk_hits"] >= 4
    again = prefill.prefill_level(prefill.level_location(url, 0), str(cache), anon=True, connections=2)
    assert again.fetched_chunks == 0


def test_prefill_level_rejects_bad_connections(served, tmp_path):
    url, _data = served
    with pytest.raises(ValueError, match="connections"):
        prefill.prefill_level(prefill.level_location(url, 0), str(tmp_path / "c"), connections=0)
    with pytest.raises(ValueError, match="scale"):
        prefill.level_location(url, -1)


def test_region_maps_level0_voxels_to_each_level(served):
    url, _data = served
    # Physical 2.4/4.8 µm metadata places level 1 at factor 2 in level-0 voxels.
    assert prefill.level_placement(url, 0, anon=True) == ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    factor, offset = prefill.level_placement(url, 1, anon=True)
    assert factor == (2.0, 2.0, 2.0) and all(abs(v) < 1 for v in offset)
    unit = ((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
    region = (0, 5, 9, 4, 8, 16)  # z0,y0,x0,z1,y1,x1 in level-0 voxels
    assert prefill.level_chunks_in_region((8, 8, 16), (4, 4, 4), region, unit) == [(0, 1, 2), (0, 1, 3)]
    assert prefill.level_chunks_in_region((2, 2, 4), (2, 2, 2), region, ((4, 4, 4), (0, 0, 0))) == [(0, 0, 1)]
    assert prefill.level_chunks_in_region((8, 8, 8), (4, 4, 4), (0, 0, 20, 8, 8, 24), unit) == []
    with pytest.raises(ValueError, match="factor"):
        prefill.level_chunks_in_region((8, 8, 8), (4, 4, 4), region, ((0, 1, 1), (0, 0, 0)))


def test_main_prefills_a_region_of_the_requested_levels(served, tmp_path, capsys):
    url, _data = served
    cache = tmp_path / "cache"
    assert prefill.main([url, str(cache), "--scales", "0,1", "--anon", "--connections", "2",
                         "--bbox", "0,0,0,8,8,2"]) == 0
    err = capsys.readouterr().err
    assert "Level 0: 4 chunks in region" in err
    assert "Level 1: 1 chunks in region" in err
    assert "Level 0: done, 4 fetched" in err and "of 4 chunks" in err
    assert "Level 1: done, 1 fetched (1 with data, 0 absent) of 1 chunks" in err
    with pytest.raises(SystemExit, match="mutually exclusive"):
        prefill.main([url, str(cache), "--bbox", "0,0,0,1,1,1", "--zrange", "0,1"])
    with pytest.raises(SystemExit, match="requires --radius"):
        prefill.main([url, str(cache), "--center", "1,1,1"])
    with pytest.raises(SystemExit, match="positive extent"):
        prefill.main([url, str(cache), "--zrange", "4,4"])


def test_volume_list_parses_and_dry_runs(tmp_path, capsys, monkeypatch):
    listing = tmp_path / "volumes.txt"
    listing.write_text(
        "# comment\n"
        "s3://vesuvius-challenge-open-data/PHerc0125/volumes/a.zarr/\n"
        "\n"
        "s3://vesuvius-challenge-open-data/PHerc0001/volumes/b.zarr\n",
        encoding="utf-8",
    )
    volumes = batch.read_volume_list(listing)
    assert [v.name for v in volumes] == ["a.zarr", "b.zarr"]
    assert volumes[0].cached_level(tmp_path / "cache") == (
        tmp_path / "cache/s3/vesuvius-challenge-open-data/PHerc0125/volumes/a.zarr/0")
    for bad in ("https://example.org/x.zarr", "s3://vesuvius-challenge-open-data/x.zarr",
                "s3://vesuvius-challenge-open-data/PHerc0125/volumes/.zarr"):
        with pytest.raises(ValueError):
            batch.parse_volume_uri(bad, line_number=1)
    assert batch.main([str(listing), str(tmp_path / "cache"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[2/2] s3://vesuvius-challenge-open-data/PHerc0001/volumes/b.zarr" in out
    assert "PHerc0001/volumes/b.zarr/0" in out

    calls = []
    def fake_prefill(location, cache_dir, *, anon, connections, chunks=None, progress=None):
        calls.append((location, cache_dir, anon, connections))
        if "PHerc0125" in location:
            raise RuntimeError("boom")
        class Result:
            fetched_chunks = 3
            total_chunks = 3
        return Result()
    monkeypatch.setattr(batch, "prefill_level", fake_prefill)
    assert batch.main([str(listing), str(tmp_path / "cache"), "--connections", "5"]) == 1
    assert len(calls) == 1  # stops at the first failure
    assert calls[0] == ("s3://vesuvius-challenge-open-data/PHerc0125/volumes/a.zarr/0",
                        str((tmp_path / "cache").resolve()), True, 5)
    assert batch.main([str(listing), str(tmp_path / "cache"), "--keep-going"]) == 1
    assert len(calls) == 3


