import json

import pytest

from vesuvius.neural_tracing.fiber_trace_3d.reviewed_failure_test import (
    directional_restart_metrics,
    materialize_failure_union,
    validate_failure_union,
)
from vesuvius.neural_tracing.fiber_trace_3d.training_trace_benchmark import build_workload
from vesuvius.neural_tracing.fiber_trace_3d.train import _reviewed_trace_test_config
from vesuvius.neural_tracing.fiber_trace_3d import train


def _case(direction, length, failures=()):
    return {"direction": direction, "reference_length_base_voxels": length,
            "evaluated_through_base_voxels": length, "evaluation_complete": True,
            "failures": [{"directional_reference_arc_base_voxels": value} for value in failures]}


def _summary(paths, failed_indices=()):
    cases = []
    for index in range(len(paths)):
        failures = (20.0,) if index in failed_indices else ()
        cases.extend((_case("forward", 100.0, failures), _case("reverse", 100.0)))
        cases[-2]["source_index"] = index
        cases[-1]["source_index"] = index
    return {"completed": True, "infrastructure_failures": 0, "cases": len(cases),
            "evaluated_cases": len(cases), "sources": [{"source_id": "S",
                "backend_config": {"fibers": [str(path) for path in paths]},
                "summary": {"base_voxel_size_um": 2.5}, "cases": cases}]}


def test_materialize_union_and_validate(tmp_path):
    source = tmp_path / "source" / "S"
    source.mkdir(parents=True)
    paths = []
    for index in range(3):
        path = source / f"{index}.json"
        path.write_text(json.dumps({"index": index}))
        paths.append(path)
    summaries = []
    for ordinal, failed in enumerate(((0,), (2,))):
        path = tmp_path / f"summary{ordinal}.json"
        path.write_text(json.dumps(_summary(paths, failed)))
        summaries.append(path)
    destination = tmp_path / "locked"
    manifest = materialize_failure_union(summaries, destination)
    assert manifest["expected_count"] == 2
    validate_failure_union(destination, expected_count=2)
    (destination / "test/S/0.json").write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        validate_failure_union(destination, expected_count=2)


def test_directional_restart_metrics():
    summary = _summary(["a", "b"], failed_indices=(0,))
    metrics = directional_restart_metrics(summary, expected_pieces=2)
    assert metrics["restarts_forward"] == 1
    assert metrics["restarts_reverse"] == 0
    assert metrics["restarts_combined"] == 1
    assert metrics["reference_mm_forward"] == pytest.approx(0.5)
    assert metrics["reference_mm_reverse"] == pytest.approx(0.5)
    assert metrics["implied_mean_mm_combined"] == pytest.approx(0.5)


def test_directional_restart_metrics_rejects_incomplete():
    summary = _summary(["a"], failed_indices=())
    summary["completed"] = False
    with pytest.raises(ValueError, match="complete"):
        directional_restart_metrics(summary, expected_pieces=1)


def test_build_workload_locks_sources_and_runtime(tmp_path):
    split = tmp_path / "split"
    rows = []
    for collection, count in (("A", 11), ("B", 6), ("C", 15)):
        directory = split / "test" / collection
        directory.mkdir(parents=True)
        for index in range(count):
            path = directory / f"{index}.json"
            path.write_text("{}")
            import hashlib
            rows.append({"source": collection, "split": "test",
                         "path": f"{collection}/{index}.json",
                         "sha256": hashlib.sha256(b"{}").hexdigest()})
    (split / "manifest.json").write_text(json.dumps({
        "version": 1, "expected_count": 32, "files": rows,
    }))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    binary = tmp_path / "native"
    binary.write_bytes(b"binary")
    template = {
        "sources": [{"id": name, "expected_test_count": 99} for name in ("A", "B", "C", "unused")],
        "evaluation": {"render": True, "workers": 192,
                       "gpu_manager": {"enabled": True},
                       "inference": {"input_workers": 192, "flush_workers": 192,
                                     "accumulator_workers": 192,
                                     "max_batch_size": 4}},
    }
    result = build_workload(template, checkpoint=checkpoint, output_root=tmp_path / "out",
                            split_root=split, devices="0,1", native_binary=binary)
    assert [(row["id"], row["expected_test_count"]) for row in result["sources"]] == [
        ("A", 11), ("B", 6), ("C", 15)]
    assert result["evaluation"]["gpu_manager"]["enabled"] is False
    assert result["evaluation"]["workers"] == 128
    assert result["evaluation"]["inference"]["max_batch_size"] == 4
    assert result["devices"] == "0,1"

def test_reviewed_trace_config_validation():
    value = {"interval": 100, "template": "a", "split_root": "b",
             "native_binary": "c", "runner": "d"}
    assert _reviewed_trace_test_config({"reviewed_trace_test": value}) == value
    with pytest.raises(ValueError, match="missing"):
        _reviewed_trace_test_config({"reviewed_trace_test": {"interval": 100}})


def test_training_trace_subprocess_completion(monkeypatch, tmp_path):
    checkpoint = tmp_path / "snapshots/current.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    for name in ("template.json", "native", "runner.py"):
        (tmp_path / name).write_text("x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")

    class Process:
        def __init__(self, command, **_kwargs):
            job = json.loads((tmp_path / "run/reviewed_trace_test/step_00000001/job.json").read_text())
            assert job["samples_seen"] == 128
            output = tmp_path / "run/reviewed_trace_test/step_00000001"
            values = {key: 1.0 for key in (
                "restarts_forward", "restarts_reverse", "restarts_combined",
                "reference_mm_forward", "reference_mm_reverse", "reference_mm_combined",
                "implied_mean_mm_forward", "implied_mean_mm_reverse", "implied_mean_mm_combined",
                "wall_seconds")}
            (output / "training_metrics.json").write_text(json.dumps(values))

        def poll(self):
            return 0

    monkeypatch.setattr(train.subprocess, "Popen", Process)
    distributed = train._distributed_config_from_env(train.torch.device("cpu"), env={})
    metrics = train._run_reviewed_trace_test(
        settings={"template": str(tmp_path / "template.json"), "split_root": str(tmp_path),
                  "native_binary": str(tmp_path / "native"), "runner": str(tmp_path / "runner.py")},
        step=1, run_dir=tmp_path / "run", snapshot_dir=checkpoint.parent,
        distributed=distributed, writer=None, samples_seen=128,
    )
    assert metrics["restarts_combined"] == 1.0


def test_training_trace_subprocess_start_failure_is_reported(monkeypatch, tmp_path):
    checkpoint = tmp_path / "snapshots/current.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    for name in ("template.json", "native", "runner.py"):
        (tmp_path / name).write_text("x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")
    monkeypatch.setattr(train.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    distributed = train._distributed_config_from_env(train.torch.device("cpu"), env={})
    with pytest.raises(RuntimeError, match="failed to start.*boom"):
        train._run_reviewed_trace_test(
            settings={"template": str(tmp_path / "template.json"), "split_root": str(tmp_path),
                      "native_binary": str(tmp_path / "native"), "runner": str(tmp_path / "runner.py")},
            step=1, run_dir=tmp_path / "run", snapshot_dir=checkpoint.parent,
            distributed=distributed, writer=None, samples_seen=128,
        )


def test_training_trace_uses_startup_best_snapshot(monkeypatch, tmp_path):
    checkpoint = tmp_path / "snapshots/best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    for name in ("template.json", "native", "runner.py"):
        (tmp_path / name).write_text("x")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-test")

    class Process:
        def __init__(self, command, **_kwargs):
            job = json.loads((tmp_path / "run/reviewed_trace_test/step_00000001/job.json").read_text())
            assert job["checkpoint"] == str(checkpoint)
            assert job["samples_seen"] == 128
            output = tmp_path / "run/reviewed_trace_test/step_00000001"
            values = {key: 1.0 for key in (
                "restarts_forward", "restarts_reverse", "restarts_combined",
                "reference_mm_forward", "reference_mm_reverse", "reference_mm_combined",
                "implied_mean_mm_forward", "implied_mean_mm_reverse", "implied_mean_mm_combined",
                "wall_seconds")}
            (output / "training_metrics.json").write_text(json.dumps(values))

        def poll(self):
            return 0

    monkeypatch.setattr(train.subprocess, "Popen", Process)
    distributed = train._distributed_config_from_env(train.torch.device("cpu"), env={})
    train._run_reviewed_trace_test(
        settings={"template": str(tmp_path / "template.json"), "split_root": str(tmp_path),
                  "native_binary": str(tmp_path / "native"), "runner": str(tmp_path / "runner.py")},
        step=1, run_dir=tmp_path / "run", snapshot_dir=checkpoint.parent,
        distributed=distributed, writer=None, samples_seen=128,
    )
