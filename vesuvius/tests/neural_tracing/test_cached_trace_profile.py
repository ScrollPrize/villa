import json

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import PublishedStore, prefix_points


def test_prefix_is_deterministic_and_handles_duplicate_points():
    points = [[0, 0, 0], [0, 0, 0], [3, 0, 0], [3, 4, 0]]
    assert prefix_points(points, 5) == [[0, 0, 0], [3, 0, 0], [3, 2, 0]]
    assert prefix_points(points, 100)[-1] == [3, 4, 0]
    with pytest.raises(ValueError):
        prefix_points(points, 0)


def test_published_store_missing_sibling_fails_and_never_infers(tmp_path, monkeypatch):
    groups = {}
    for name in ("presence", "nx", "ny"):
        level = tmp_path / f"{name}.zarr" / "3"
        level.mkdir(parents=True)
        (level / ".zarray").write_text(json.dumps({"shape": [64] * 3, "chunks": [32] * 3, "dtype": "|u1"}))
        groups[name] = {"zarr": f"{name}.zarr/3"}
    manifest = tmp_path / "inference.lasagna.json"
    manifest.write_text(json.dumps({"groups": groups}))
    store = PublishedStore(manifest)
    assert store.pending_tile_key() is None
    calls = []
    def complete(*args):
        calls.append(args)
        return len(calls) > 1
    monkeypatch.setattr("lasagna.tiled_predict3d._omezarr_chunk_group_complete", complete)
    with pytest.raises(RuntimeError, match="inference is disabled"):
        store.request([0, 0, 0])
    assert not store.ready((0, 0, 0))
    assert store.request([0, 0, 0]) == (0, 0, 0)
    assert store.request([0, 0, 0]) == (0, 0, 0)
    assert len(calls) == 2
    assert store.stats == {"inferred_tiles": 0, "published_chunks": 0, "warm_chunks": 1}
    assert store.priorities == store.tiles == {}
    assert store.pending_tile_key() is None
    with pytest.raises(ValueError):
        store.request([2, 0, 0])
    with pytest.raises(ValueError):
        store.request([True, 0, 0])
