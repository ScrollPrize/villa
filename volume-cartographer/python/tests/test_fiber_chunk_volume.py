"""Whole-volume resume, sparse inventory and geometry-preserving annotations."""
import json
from pathlib import Path

import numpy as np
import pytest

from test_fiber_chunk_trace import inputs, handles, chunk_config, trace_config
from test_fiber_trace import fiber_trace as ft
from vc.fiber_chunk_volume import sparse_keys, output_lock, execute


def test_queued_loader_configuration(monkeypatch):
    from vc import fiber_chunk_volume as workflow
    calls = []
    monkeypatch.setattr(workflow.ft, "configure_chunk_loader",
                        lambda *args: calls.append(args) or '{"backend":"queued"}')
    cfg = dict(cache_bytes=10 << 30, loader=dict(backend="queued", queued_chunk_window=512))
    assert workflow.configure_loader(cfg, 36) == {"backend": "queued"}
    assert calls == [(36, 10 << 30, "queued", 512)]
    for loader in [dict(backend="unknown"), dict(queued_chunk_window=1),
                   dict(backend="queued", queued_chunk_window=-1),
                   dict(backend="queued", queued_chunk_window=True),
                   dict(backend="queued", resource_scaled=False)]:
        with pytest.raises(ValueError):
            workflow.configure_loader(dict(loader=loader), 36)
    with pytest.raises(ValueError, match="separate backends"):
        workflow.configure_loader(dict(loader=dict(backend="queued"),
                                       chunks=dict(worker_cache_bytes=1 << 30)), 36)
    assert len(calls) == 1


def test_gpu_default_uses_manager_inventory_without_nvidia_smi(monkeypatch):
    from vc import fiber_chunk_volume as workflow
    def unexpected(*args, **kwargs):
        raise AssertionError("default GPU eligibility must not query physical indices")
    monkeypatch.setattr(workflow.subprocess, "check_output", unexpected)
    assert workflow.eligible_gpu_uuids(None, {"gpus": {"GPU-7": 80, "GPU-6": 80}}) == ["GPU-6", "GPU-7"]
    with pytest.raises(ValueError, match="no GPUs"):
        workflow.eligible_gpu_uuids(None, {"gpus": {}})


@pytest.mark.parametrize("selection", ["", "wrong", "6,6", "9", "0"])
def test_explicit_gpu_restriction_validation(selection, monkeypatch):
    from vc import fiber_chunk_volume as workflow
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **kw: "0, GPU-0\n6, GPU-6\n7, GPU-7\n")
    inventory = {"gpus": {"GPU-6": 80, "GPU-7": 80}}
    assert workflow.eligible_gpu_uuids("7,6", inventory) == ["GPU-7", "GPU-6"]
    with pytest.raises(ValueError):
        workflow.eligible_gpu_uuids(selection, inventory)


def test_cli_default_submits_all_manager_gpus(tmp_path, monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from vc import fiber_chunk_volume as workflow
    inventory = {"cpu_cores": 224, "gpus": {"GPU-6": 80 << 30, "GPU-7": 80 << 30}}
    observed = {}
    def submit(command, **kwargs):
        observed.update(kwargs)
        return 0
    monkeypatch.setitem(sys.modules, "villa_gpu_manager", ModuleType("villa_gpu_manager"))
    monkeypatch.setitem(sys.modules, "villa_gpu_manager.bootstrap", SimpleNamespace(ensure_manager=lambda: ("mock", inventory)))
    monkeypatch.setitem(sys.modules, "villa_gpu_manager.exclusive", SimpleNamespace(run_exclusive=submit))
    monkeypatch.setattr(workflow, "load_config", lambda path: {})
    monkeypatch.setattr(workflow, "configure_loader", lambda *args: pytest.fail("pools created before grant"))
    monkeypatch.setattr(workflow, "inspect_outputs", lambda *args: (1, 1, True))
    monkeypatch.setattr(workflow, "open_inputs", lambda cfg: (None, None, None, SimpleNamespace(gpu_library="cuda", cache_only=False)))
    monkeypatch.setattr(workflow, "job_keys", lambda *args: [(0, 0, 0)])
    monkeypatch.setattr(workflow.subprocess, "check_output", lambda *a, **kw: pytest.fail("unexpected nvidia-smi"))
    assert workflow.main([str(tmp_path / "config"), str(tmp_path / "output")]) == 0
    assert observed["gpu_uuids"] == ["GPU-6", "GPU-7"]
    assert observed["accept_partial"] is True
    assert observed["exclusive"] is False


def run(root, chunks, keys=((0, 0, 0), (1, 0, 0)), identity="volume-test", workers=4, control=None):
    field, normals = handles(root)
    control = control or ft.EvaluationControl()
    ft.prepare_chunk_volume(field, normals, trace_config(), chunks, str(root / "output"),
                            identity, keys, control, workers, 2)
    return control.status()


def test_resume_flags_and_legacy(inputs):
    c = chunk_config()
    c.conflict_normal_base, c.conflict_tangent_base = 10, 20
    assert run(inputs, c)["built"] == 2
    files = sorted((inputs / "output").glob("*.msgpack.zst"))
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    first = ft.load_chunk_paths(str(files[0]))
    assert len(first.lines_base) > 0
    assert len(first.conflict_flags) == len(first.lines_base)
    field, normals = handles(inputs)
    _, report = ft.filter_chunk_paths(first, c, normals, 1, 2)
    assert json.loads(report)["rejected"] == [i for i, flag in enumerate(first.conflict_flags) if flag]
    assert run(inputs, c, workers=1)["reused"] == 2
    assert before == [(p.read_bytes(), p.stat().st_mtime_ns) for p in files]
    with pytest.raises(RuntimeError, match="identity or configuration mismatch"):
        run(inputs, c, identity="different")
    # Filter-only change reannotates, but keeps geometry and construction times.
    c.conflict_agreement_base = 80
    assert run(inputs, c)["annotated"] == 2
    second = ft.load_chunk_paths(str(files[0]))
    for a, b in zip(first.lines_base, second.lines_base, strict=True):
        np.testing.assert_array_equal(a, b)
    assert first.end_reasons == second.end_reasons
    assert json.loads(first.timings_json)["construction_seconds"] == json.loads(second.timings_json)["construction_seconds"]
    # Missing annotation on old raw construction must not be silently clean.
    c.conflict_normal_base = c.conflict_tangent_base = 0
    assert run(inputs, c, keys=((2, 0, 0),))["built"] == 1
    c.conflict_normal_base, c.conflict_tangent_base = 10, 20
    assert run(inputs, c, keys=((2, 0, 0),))["annotated"] == 1
    files[0].write_bytes(b"invalid cache")
    with pytest.raises(RuntimeError, match="chunk cache frame"):
        run(inputs, c)
    assert files[0].read_bytes() == b"invalid cache"


def test_volume_cancel_duplicates_and_empty(inputs):
    c = chunk_config()
    control = ft.EvaluationControl()
    control.cancel("test cancel")
    with pytest.raises(RuntimeError, match="test cancel"):
        run(inputs, c, control=control)
    assert not (inputs / "output").exists()
    with pytest.raises(ValueError, match="duplicate"):
        run(inputs, c, keys=((0, 0, 0), (0, 0, 0)))
    assert run(inputs, c, keys=())["completed"] == 0


@pytest.mark.parametrize("separator", ["/", "."])
def test_sparse_packed_inventory(tmp_path, separator):
    metadata = dict(zarr_format=2, shape=[7, 70, 128, 128], chunks=[2, 64, 64, 64],
                    fill_value=0, dimension_separator=separator)
    (tmp_path / ".zarray").write_text(json.dumps(metadata))
    for idx in [(0, 0, 0, 0), (1, 0, 0, 1), (1, 1, 1, 1)]:
        path = tmp_path / separator.join(map(str, idx))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"payload not decoded")
    (tmp_path / "junk.tmp").write_text("ignored")
    assert sparse_keys(tmp_path, 2, 4., [280, 512, 512], 256) == [(1, 0, 0), (1, 1, 1)]
    # Larger tracing cores deduplicate several stored chunks.
    assert sparse_keys(tmp_path, 2, 4., [280, 512, 512], 512) == [(0, 0, 0)]
    metadata["fill_value"] = 1
    (tmp_path / ".zarray").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="zero fill"):
        sparse_keys(tmp_path, 2, 4., [280, 512, 512], 512)


def test_writer_lock(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another writer"):
            with output_lock(tmp_path):
                pass
    with output_lock(tmp_path):
        pass


def test_native_pool_inventory(tmp_path):
    source, pooled = tmp_path / "source", tmp_path / "pooled"
    source.mkdir(); pooled.mkdir()
    meta = dict(zarr_format=2, shape=[64, 64, 128], chunks=[64, 64, 64], fill_value=0)
    (source / ".zarray").write_text(json.dumps(meta))
    (source / "0.0.1").write_bytes(b"present")
    meta.update(shape=[32, 32, 64], chunks=[32, 32, 32])
    (pooled / ".zarray").write_text(json.dumps(meta))
    (pooled / ".native_pool.json").write_text(json.dumps(dict(revision=1, factor=2,
        mode="top2_spatial_comp", channel="presence", source=str(tmp_path), channels={"presence": "source"})))
    assert sparse_keys(pooled, 0, 4, [128, 128, 256], 128) == [(1, 0, 0)]


def test_command_callable_cpu(inputs, monkeypatch):
    cfg = dict(prediction_manifest=str(inputs / "pred.json"), normal_manifest=str(inputs / "norm.json"),
               source_identity="callable", scaledown_power=0, enable_width=True,loader={"resource_scaled":False},
               trace=trace_config().to_dict(), chunks=dict(core_base=64, halo_base=16, grid_base=4,
               nms_base=4, tube_half_base=8, tube_step_base=4, rim_normal_base=4, rim_tangent_base=4,
               conflict_normal_base=10, conflict_tangent_base=20), keys_xyz=[[0, 0, 0], [1, 0, 0]])
    output = inputs / "command"
    from vc.fiber_chunk_volume import inspect_outputs
    monkeypatch.setenv("VC_CHUNK_VOLUME_LAUNCHER", str(Path(__file__).resolve().parents[2] / "scripts/run_fiber_chunk_volume.py"))
    config = inputs / "config.json"
    config.write_text(json.dumps(cfg))
    assert inspect_outputs(config, output) == (2, 2, False)
    first = execute(cfg, output, 4, 2)
    second = execute(cfg, output, 4, 2)
    assert first["built"] == 2 and second["reused"] == 2
    assert second["built"] == second["annotated"] == 0


def test_partial_grant_caps_cpu_even_on_resume(inputs, monkeypatch):
    from vc import fiber_chunk_volume as workflow
    field, normal = handles(inputs)
    chunks = chunk_config()
    chunks.gpu_library = "unused-library-on-resume"
    chunks.gpu_devices = 1
    monkeypatch.setenv("VILLA_GPU_MANAGER_GRANT", json.dumps(dict(cpu_cores=2, gpus={})))
    monkeypatch.setattr(workflow, "open_inputs", lambda cfg: (field, normal, trace_config(), chunks))
    monkeypatch.setattr(workflow, "job_keys", lambda *args: [])
    observed = []
    loader_cores = []
    monkeypatch.setattr(workflow, "configure_loader", lambda cfg, cores: loader_cores.append(cores) or {})
    def prepare(*args):
        observed.append(args[-2])
    monkeypatch.setattr(workflow.ft, "prepare_chunk_volume", prepare)
    result = workflow.execute(dict(source_identity="test"), inputs / "unused", 224, 8)
    assert observed == [2]
    assert loader_cores == [2]
    assert result["workers"] == 2


def test_packed_normal_alias_preserves_indices(inputs):
    # Four packed channels: unrelated original nx, then three normal heads.
    from test_fiber_trace import _write_u8_zarr
    root = inputs / "packed"
    root.mkdir()
    shape = (32, 32, 256)
    values = np.stack([np.full(shape, v, np.uint8) for v in (1, 128, 255, 255)])
    metadata = json.loads((inputs / "nx" / ".zarray").read_text())
    metadata["shape"] = metadata["chunks"] = list(values.shape)
    (root / ".zarray").write_text(json.dumps(metadata))
    (root / "0.0.0.0").write_bytes(values.tobytes())
    manifest = json.loads((inputs / "pred.json").read_text())
    manifest["version"] = 3
    manifest["groups"]["normal_heads"] = dict(zarr="packed", scaledown=0,
        channels=["unused", "sheet_nx", "sheet_ny", "sheet_grad_mag"])
    (inputs / "pred.json").write_text(json.dumps(manifest))
    field, _ = handles(inputs)
    sampler = ft.open_fiber_normal_sampler(field)
    # A native construction/filter call exercises decoded packed normals.
    c = chunk_config()
    paths = ft.build_chunk_paths(field, sampler, trace_config(), c, [0, 0, 0], 2)
    assert paths.seeds > 0
    c.conflict_normal_base, c.conflict_tangent_base = 10, 20
    _, report = ft.filter_chunk_paths(paths, c, sampler, 1, 2)
    assert json.loads(report)["invalid_normal_samples"] == 0
