import json

import pytest


def test_cancelled_replay_never_launches_native(monkeypatch):
    from concurrent.futures import CancelledError
    from threading import Event
    from vesuvius.neural_tracing.fiber_trace_3d import live_evaluate as live
    event = Event(); event.set()
    monkeypatch.setattr(live, 'implementation_identity', lambda: {})
    monkeypatch.setattr(live.subprocess, 'Popen', lambda *a, **k: pytest.fail('launched cancelled work'))
    with pytest.raises(CancelledError):
        live.coordinate_native({},None,{},{},{},{},{},(),{},cancel_event=event)


@pytest.mark.parametrize('duplicate_source', [False, True])
def test_cached_extension_reuses_shared_store_and_validates_source(tmp_path, monkeypatch, duplicate_source):
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
    def coordinate(*args, **kwargs):
        assert [str(device) for device in args[-2]] == [f'cuda:{i}' for i in range(8)]
        assert 'demand_gpu_manager' not in args[-1]
        calls.append('replay')
    monkeypatch.setattr(cached, "coordinate_native", coordinate)
    evaluation = {"inference": {'demand_gpu_manager': {'enabled': True}}, "sources": [{"id": "source", "fiber_manifest": str(tmp_path / "manifest.json")}]}
    if duplicate_source:
        evaluation['sources'].append(dict(evaluation['sources'][0], id='second_collection'))
    cached.extend_cached_replay(evaluation, tmp_path / "binary", tmp_path / "checkpoint", devices='0,1,2,3,4,5,6,7')
    assert calls == ["open", "replay", "closed"]
    volume.shape_at = lambda level: [32]*3
    with pytest.raises(ValueError, match="source metadata changed"):
        cached.extend_cached_replay(evaluation, tmp_path / "binary", tmp_path / "checkpoint")


def test_retained_session_reuses_pool_and_closes_before_stores(monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    events = []
    @contextmanager
    def resources(*args, **kwargs):
        events.append('open stores')
        yield ({},{},{},{},(),{})
        events.append('close stores')
    monkeypatch.setattr(cached,'_cached_replay_resources',resources)
    def coordinate(*args, **kwargs):
        session=kwargs['inference_session']
        if session.pool is None:
            events.append('open pool')
            session.pool=SimpleNamespace(close=lambda:events.append('close pool'))
        events.append('run')
    monkeypatch.setattr(cached,'coordinate_native',coordinate)
    with cached.CachedReplaySession({'sources':[]},'binary','checkpoint') as session:
        session.run({'sources':[]})
        session.run({'sources':[]})
        assert events==['open stores','open pool','run','run']
        session.pool.busy=False
        session.pool.stop_gpu=lambda:events.append('stop gpu')
        session.pool.start_gpu=lambda devices:events.append('start gpu')
        session.pool.authorize_gpu=lambda:events.append('authorize gpu')
        session.pause_gpu()
        session.pause_gpu()
        session.run({'sources':[]})
    session.close()
    assert events==['open stores','open pool','run','run','stop gpu','start gpu',
                   'authorize gpu','run','close pool','close stores']


def test_worker_evicts_overlapping_idle_session_and_reuses_matching(monkeypatch,tmp_path):
    from threading import Event
    from collections import OrderedDict
    from vesuvius.neural_tracing.fiber_trace_3d import cached_trace_profile as cached
    monkeypatch.setattr(cached,'_trial_cancel_event',Event())
    monkeypatch.setattr(cached,'_trial_sessions',OrderedDict())
    monkeypatch.setattr(cached,'_trial_session_limit',2)
    events=[]
    class Session:
        def __init__(self,e,b,c,**kwargs):
            self.closed=False
            self.tag=e['inference']['tag']
            events.append(('open',self.tag))
        def run(self,e): events.append(('run',self.tag))
        def close(self):
            if not self.closed: events.append(('close',self.tag))
            self.closed=True
    monkeypatch.setattr(cached,'CachedReplaySession',Session)
    evaluation=dict(sources=[{'id':'s','fiber_manifest':str(tmp_path/'cache'/'manifest.json')}],inference={'tag':'a'})
    cached.execute_trial_replay(evaluation,'binary','snapshot')
    cached.execute_trial_replay(evaluation,'binary')
    changed=dict(evaluation,inference={'tag':'b'})
    cached.execute_trial_replay(changed,'binary','snapshot')
    assert events==[('open','a'),('run','a'),('run','a'),('close','a'),('open','b'),('run','b')]
    cached.close_trial_worker()
    assert events[-1]==('close','b')

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
