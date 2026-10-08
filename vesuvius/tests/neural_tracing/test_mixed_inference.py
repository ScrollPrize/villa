import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lasagna.inference_provenance import atomic_write
from vesuvius.neural_tracing.fiber_trace_3d.mixed_inference import (
    ModelWorkerPools, partition_devices, request_store_key, write_mixed_manifest)


def manifest(root, *, embeddings=2, source="ct", level=0):
    groups = {}
    for name in ["presence", "nx", "ny"] + [f"embedding_{i:03d}" for i in range(embeddings)]:
        groups[name] = {"zarr": f"{name}/{level}", "scaledown": level, "channels": [name]}
    atomic_write(root / "inference.lasagna.json", {"version": 2, "source_to_base": 1,
        "base_shape_zyx": [32]*3, "groups": groups})
    atomic_write(root / "cache_identity.json", {"identity": {"source": {"locator": source},
        "model": {"checkpoint_sha256": root.name}}})
    return root / "inference.lasagna.json"


def test_composition_preserves_paths_and_replaces_embeddings(tmp_path):
    a, b = manifest(tmp_path / "a"), manifest(tmp_path / "b")
    before = [p.read_bytes() for p in (a, b)]
    mixed = write_mixed_manifest(tmp_path / "views", a, b)
    data = json.loads(mixed.read_text())
    assert len(data["groups"]) == 5
    for name, group in data["groups"].items():
        assert group["zarr"] == str((b if name.startswith("embedding") else a).parent / name / "0")
    assert [p.read_bytes() for p in (a, b)] == before
    assert write_mixed_manifest(tmp_path / "views", a, b) == mixed
    assert write_mixed_manifest(tmp_path / "views", a, a) != mixed


@pytest.mark.parametrize("kwargs,error", [
    ({"source": "other"}, "identical physical source"),
    ({"level": 1}, "output lattice"),
    ({"embeddings": 0}, "contiguous"),
])
def test_incompatible_composition_rejected(tmp_path, kwargs, error):
    with pytest.raises(ValueError, match=error):
        write_mixed_manifest(tmp_path / "views", manifest(tmp_path / "a"), manifest(tmp_path / "b", **kwargs))


def test_routing_is_per_source_and_array():
    channels = {"base": {"/a": None}, "embed": {"/b": None}, "foreign": {"/c": None}}
    assert request_store_key(["base", "embed"], "/a", channels) == "base"
    assert request_store_key(["base", "embed"], "/b", channels) == "embed"
    assert request_store_key(["base", "base"], "/a", channels) == "base"
    assert request_store_key("base", "/a", channels) == "base"
    with pytest.raises(ValueError, match="escapes"):
        request_store_key(["base", "embed"], "/c", channels)


def test_model_pools_partition_budget_lifecycle(monkeypatch):
    from lasagna import demand_predict3d as demand
    made = []
    class Pool:
        def __init__(self, stores, readers, devices, **options):
            self.devices, self.options = devices, options
            self.active, self.loading, self.input_owners = {}, {}, {}
            self.processes = []
            self.stats = {"max_batch_size": options["max_batch_size"], "input_slots": options["input_slots"]}
            self.busy = False
            self.closed = self.pumped = False
            made.append(self)
        def pump(self): self.pumped = True
        def close(self): self.closed = True
        def start_gpu(self, devices, **kwargs): self.start = (devices, kwargs)
        def authorize_gpu(self): self.gpu_authorized = True
        def queue_snapshot(self): return {"queued": 2}
        def poll_stop_gpu(self): self.stopped = True; return True
    monkeypatch.setattr(demand, "DemandWorkerPool", Pool)
    pools = ModelWorkerPools({"a": 1, "b": 2}, {"a": 3, "b": 4}, list(range(8)),
        groups={"A": {"keys": ["a"], "settings": {"max_batch_size": 8}},
                "B": {"keys": ["b"], "settings": {"max_batch_size": 4}}},
        input_slots=256, input_workers=128, result_slots=128, deferred_gpu=True)
    assert [p.devices for p in made] == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert all(p.options["input_slots"] == 128 for p in made)
    assert [p.options["max_batch_size"] for p in made] == [8, 4]
    pools.start_gpu(["one"], vram_budget=40 << 30)
    assert all(p.start[1]["vram_budget"] == int((40 << 30) * .45) for p in made)
    monkeypatch.setenv("FIBER_INFERENCE_VRAM_BUDGET_BYTES", str(20 << 30))
    pools.start_gpu(["one"])
    assert all(p.start[1]["vram_budget"] == int((20 << 30) * .45) for p in made)
    pools.authorize_gpu()
    pools.gpu_authorized = False
    assert not any(p.gpu_authorized for p in made)
    assert pools.queue_snapshot() == {"queued": 4}
    pools.pump()
    assert all(p.pumped for p in made)
    assert pools.poll_stop_gpu() and all(p.stopped for p in made)
    pools.close()
    assert all(p.closed for p in made)
    assert partition_devices([0], 2) == [[0], [0]]


@pytest.mark.parametrize("embedding_chunk", [4, 8])
@pytest.mark.parametrize("high_descriptors", [False, True])
def test_native_mixed_replay_and_readiness(tmp_path, embedding_chunk, high_descriptors):
    """Real native sampler/control loop reads base bytes and alternate float embeddings."""
    binary = os.environ.get("VC_REPLAY_TEST_BINARY")
    if not binary:
        pytest.skip("requires native replay binary")
    import subprocess
    from lasagna.tiled_predict3d import _create_omezarr
    from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import PublishedStore
    from vesuvius.neural_tracing.fiber_trace_3d.live_evaluate import coordinate_native
    a, b = manifest(tmp_path / "a"), manifest(tmp_path / "b")
    for path in (a, b):
        for name in json.loads(path.read_text())["groups"]:
            embedding = name.startswith("embedding")
            array = _create_omezarr(str(path.parent / name), (32,)*3, 0, 1,
                embedding_chunk if path == b else 8, name,
                dtype=np.float32 if embedding else np.uint8)["0"]
            # Base and alternate deliberately disagree in both heads.
            value = (1 if path == b else -1) if embedding else (
                {"presence": 255, "nx": 255, "ny": 128}[name] if path == a else 17)
            array[:] = value
    mixed = write_mixed_manifest(tmp_path / "views", a, b)
    points = tmp_path / "points.json"
    atomic_write(points, [[8., 16., 16.]])
    query = subprocess.run([binary, "sample-embeddings", str(mixed), str(points)],
        capture_output=True, text=True, check=True)
    row, = json.loads(query.stdout)
    assert row["presence"] == 1
    np.testing.assert_allclose(row["embedding"], [2**-.5]*2, atol=1e-6)
    groups = {}
    for name, value in (("nx", 128), ("ny", 128), ("grad_mag", 255)):
        array = _create_omezarr(str(tmp_path / "normal" / name), (32,)*3, 0, 1, 8, name)["0"]
        array[:] = value
        groups[name] = {"zarr": f"{name}/0", "scaledown": 0, "channels": [name]}
    normal = tmp_path / "normal/manifest.json"
    atomic_write(normal, {"version": 2, "source_to_base": 1, "base_shape_zyx": [32]*3,
                         "grad_mag_encode_scale": 255., "grad_mag_factor": 1., "groups": groups})
    annotation = tmp_path / "fiber.json"
    atomic_write(annotation, {"type": "vc3d_fiber", "version": 1,
        "optimization_mode": "native_fiber_trace3d",
        "line_points": [[8,16,16], [24,16,16]], "control_points": [[8,16,16], [24,16,16]]})
    class DelayedStore(PublishedStore):
        polls = 0
        def ready(self, origin):
            self.polls += 1
            return self.polls > 5 and super().ready(origin)
    stores = {"a": PublishedStore(a), "b": DelayedStore(b)}
    evaluation = {"version": 1, "output": str(tmp_path / "output"), "workers": 2,
        "trace_args": ["--cone-angle-degrees", "0", "--step-voxels", "2", "--embedding-weight", "0.1"],
        "sources": [{"id": "test", "fiber_manifest": str(mixed), "normal_manifest": str(normal),
            "inference_scaledown_power": 0, "base_voxel_size_um": 1, "fibers": [str(annotation)]}]}
    from contextlib import ExitStack
    with ExitStack() as handles:
        if high_descriptors:
            import resource
            if resource.getrlimit(resource.RLIMIT_NOFILE)[0] < 1200:
                pytest.skip('high-descriptor regression requires fd limit>=1200')
            while handles.enter_context(open(os.devnull, 'rb')).fileno() < 1100:
                pass
        coordinate_native(evaluation, Path(binary), stores, {"test": ["a", "b"]},
            {k: s.channels for k, s in stores.items()}, {"a": set(), "b": set()}, {}, (), {},
            allow_inference=False)
    assert all(s.stats["warm_chunks"] > 0 for s in stores.values())
    assert stores["b"].polls > 5
    summary = json.loads((tmp_path / "output/summary.json").read_text())
    assert summary["completed"] and summary["cases"] == 2
    if not high_descriptors:
        # Borrowed inference workers survive multiple native replay connections;
        # only per-batch control/outputs are recreated. No GPU is needed here.
        from types import SimpleNamespace
        class IdlePool:
            busy = False
            stats = dict(shared_slot_bytes=0,input_slots=1,input_workers=1,result_slots=1)
            closed = 0
            def pump(self): pass
            def close(self): self.closed+=1
            def queue_snapshot(self):
                return {k:0 for k in ('requested_chunks_pending','tiles_queued','read_queued',
                    'read_running','read_done','gpu_submitted_uncollected','input_slots_free',
                    'input_slots_capacity','tiles_accumulating','publications_queued','publications_submitted')}
        pool=IdlePool()
        session=SimpleNamespace(pool=pool,gpu_lease=None,close=pool.close)
        for i in range(2):
            repeat=dict(evaluation,output=str(tmp_path/f'repeat-{i}'))
            coordinate_native(repeat,Path(binary),stores,{'test':['a','b']},
                {k:s.channels for k,s in stores.items()},{'a':set(),'b':set()},{},(),{},
                allow_inference=False,inference_session=session)
            assert pool.closed==0
            result=json.loads((Path(repeat['output'])/'inference_summary.json').read_text())
            assert result['reused_worker_pool'] is True
            assert all(v['warm_chunks']==0 for v in result['sources'].values())
            assert json.loads((Path(repeat['output'])/'summary.json').read_text())['failures']==summary['failures']
        # A poisoned replay invalidates/closes borrowed resources exactly once.
        original_request=stores['a'].request
        def fail(*args,**kwargs): raise RuntimeError('synthetic demand failure')
        stores['a'].request=fail
        with pytest.raises(RuntimeError,match='synthetic demand failure'):
            coordinate_native(dict(evaluation,output=str(tmp_path/'failed')),Path(binary),stores,
                {'test':['a','b']},{k:s.channels for k,s in stores.items()},
                {'a':set(),'b':set()},{},(),{},allow_inference=False,inference_session=session)
        stores['a'].request=original_request
        assert pool.closed==1


def test_cached_helpers_reject_composite_explicitly(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.cached_trace_profile import PublishedStore, require_single_model_profile
    mixed = write_mixed_manifest(tmp_path / "views", manifest(tmp_path / "a"), manifest(tmp_path / "b"))
    with pytest.raises(ValueError, match="underlying ordinary cache"):
        PublishedStore(mixed)
    with pytest.raises(ValueError, match="original embedding_inference"):
        require_single_model_profile({"embedding_inference": {"checkpoint": "b"}, "sources": []})
