import json

import pytest


def test_cached_extension_reuses_shared_store_and_validates_source(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached, live_evaluate as live
    identity = {"source": {"locator": "volume", "level": 0, "shape": [64]*3,
                           "base_shape": [64]*3, "metadata": {}},
                "model": {"checkpoint_sha256": "sha", "implementation": {}, "config": {}}}
    (tmp_path / "cache_identity.json").write_text(json.dumps({"identity": identity,
        "geometry": {"shape": [64]*3, "tile":32, "overlap":8, "border":4, "scale":2, "chunk":8}}))
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "/cache")
    monkeypatch.setattr(live, "implementation_identity", lambda: {})
    monkeypatch.setattr(cached, "sha256_file", lambda p: "sha")
    volume = SimpleNamespace(shape_at=lambda level: [64]*3, dtype="uint8")
    monkeypatch.setattr(live, "Vc3dCoordinateSampler", lambda *a, **k: SimpleNamespace(volume=volume))
    monkeypatch.setattr(live, "input_root_attributes", lambda v: {})
    monkeypatch.setattr(live, "validate_input_lattice", lambda *a: None)
    monkeypatch.setattr(live, "NativeTileReader", lambda *a: "reader")
    calls = []
    @contextmanager
    def store(*args):
        calls.append("open")
        yield SimpleNamespace(products=[])
        calls.append("closed")
    monkeypatch.setattr(live, "create_demand_store", store)
    monkeypatch.setattr(cached, "coordinate_native", lambda *a, **k: calls.append("replay"))
    evaluation = {"inference": {}, "sources": [{"id": "source", "fiber_manifest": str(tmp_path / "manifest.json")}]}
    cached.extend_cached_replay(evaluation, tmp_path / "binary", tmp_path / "checkpoint")
    assert calls == ["open", "replay", "closed"]
    volume.shape_at = lambda level: [32]*3
    with pytest.raises(ValueError, match="source metadata changed"):
        cached.extend_cached_replay(evaluation, tmp_path / "binary", tmp_path / "checkpoint")

from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import PublishedStore, prefix_points


def test_prefix_is_deterministic_and_handles_duplicate_points():
    points = [[0, 0, 0], [0, 0, 0], [3, 0, 0], [3, 4, 0]]
    assert prefix_points(points, 5) == [[0, 0, 0], [3, 0, 0], [3, 2, 0]]
    assert prefix_points(points, 100)[-1] == [3, 4, 0]
    with pytest.raises(ValueError):
        prefix_points(points, 0)


@pytest.mark.parametrize("dtype", ["|u1", "<f4"])
def test_published_store_missing_sibling_fails_and_never_infers(tmp_path, monkeypatch, dtype):
    groups = {}
    for name in ("presence", "nx", "ny"):
        level = tmp_path / f"{name}.zarr" / "3"
        level.mkdir(parents=True)
        (level / ".zarray").write_text(json.dumps({"shape": [64] * 3, "chunks": [32] * 3, "dtype": dtype}))
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
