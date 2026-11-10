import json

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.reviewed_failure_test import (
    directional_restart_metrics,
    materialize_failure_union,
    validate_failure_union,
)
from vesuvius.neural_tracing.fiber_trace_3d.training_trace_benchmark import build_workload
from vesuvius.neural_tracing.fiber_trace_3d import training_trace_benchmark
from vesuvius.neural_tracing.fiber_trace_3d.train import _reviewed_trace_test_config
from vesuvius.neural_tracing.fiber_trace_3d import train


def _write_training_snapshot(path, *, step=1, samples_seen=128):
    torch.save({"model": {"weight": torch.tensor([1.0])}, "step": step,
                "samples_seen": samples_seen, "metric": 1.0,
                "metric_name": "test/loss_total"}, path)


def _stub_trace_selection(monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import reviewed_trace_selection

    def finalize(output, _snapshot_dir, **_kwargs):
        metrics = json.loads((output / "training_metrics.json").read_text())
        return metrics, {"best_restarts_combined": int(metrics["restarts_combined"]),
                         "selected": True}, output

    monkeypatch.setattr(reviewed_trace_selection, "finalize_attempt", finalize)


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
                       "trace_args": ["--replay-presence-cutoff", "0", "--replay-presence-mode", "raw"],
                       "gpu_manager": {"enabled": True},
                       "inference": {"input_workers": 192, "flush_workers": 192,
                                     "accumulator_workers": 192,
                                     "max_batch_size": 4}},
    }
    result = build_workload(template, checkpoint=checkpoint, output_root=tmp_path / "out",
                            split_root=split, devices="0,1", native_binary=binary, presence_cutoff=.25)
    assert [(row["id"], row["expected_test_count"]) for row in result["sources"]] == [
        ("A", 11), ("B", 6), ("C", 15)]
    assert result["evaluation"]["gpu_manager"]["enabled"] is False
    assert result["evaluation"]["workers"] == 128
    assert result["evaluation"]["inference"]["max_batch_size"] == 4
    assert result["devices"] == "0,1"
    args = result['evaluation']['trace_args']
    assert args[args.index('--replay-presence-cutoff') + 1] == '0.25'
    assert args[args.index('--replay-presence-mode') + 1] == 'modulated'
    assert template['evaluation']['trace_args'][1] == '0'


def test_training_trace_job_rejects_gpu_binding_outside_inherited_grant(monkeypatch, tmp_path):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "inherited_manager_grant": True,
        "gpu_grant": {"GPU-a": 10, "GPU-b": 20},
        "devices": "0,1",
    }))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-c")
    with pytest.raises(ValueError, match="does not match inherited grant"):
        training_trace_benchmark.run_job(job)

def test_reviewed_trace_config_validation():
    value = {"interval": 100, "template": "a", "split_root": "b",
             "native_binary": "c", "runner": "d"}
    assert _reviewed_trace_test_config({"reviewed_trace_test": value}) == value
    with pytest.raises(ValueError, match="missing"):
        _reviewed_trace_test_config({"reviewed_trace_test": {"interval": 100}})


def test_unmanaged_trace_does_not_spawn_a_second_model():
    with pytest.raises(ValueError, match="requires managed training"):
        train._run_reviewed_trace_test()


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, float("nan"), float("inf"), True, "0.25", None])
def test_reviewed_trace_rejects_invalid_presence_cutoff(cutoff):
    value = {"interval": 100, "template": "a", "split_root": "b",
             "native_binary": "c", "presence_cutoff": cutoff}
    with pytest.raises(ValueError, match="presence_cutoff"):
        _reviewed_trace_test_config({"reviewed_trace_test": value})


@pytest.mark.parametrize("weights", ["raw", "ema_0p999"])
def test_reviewed_trace_accepts_weight_selection(weights):
    value = {"interval": 100, "template": "a", "split_root": "b",
             "native_binary": "c", "weights": weights}
    assert _reviewed_trace_test_config({"reviewed_trace_test": value}) == value


def test_reviewed_trace_rejects_unsupported_weights():
    value = {"interval": 100, "template": "a", "split_root": "b",
             "native_binary": "c", "weights": "ema_typo"}
    with pytest.raises(ValueError, match="weights"):
        _reviewed_trace_test_config({"reviewed_trace_test": value})


@pytest.mark.parametrize("weights", ["raw", "ema_0p999"])
def test_resumed_initial_trace_publishes_exact_selected_checkpoint(tmp_path, monkeypatch, weights):
    from types import SimpleNamespace
    from tests.neural_tracing.test_fiber_trace_3d import _loader_with_mapping
    from tests.neural_tracing.test_training_queue_integration import _Writer
    from vesuvius.neural_tracing.fiber_trace_3d.ema import ModelEMA
    from vesuvius.neural_tracing.fiber_trace_3d.reviewed_trace_selection import freeze_trace_checkpoint

    loader = _loader_with_mapping({"patch_shape_zyx": [8, 8, 8], "batch_size": 2,
                                   "augment_enabled": False})
    model = torch.nn.Sequential(torch.nn.Conv3d(1, 7, 1), torch.nn.Sigmoid())
    ema = ModelEMA(model, [.999])
    for value in ema.tracks["ema_0p999"]["model"].values():
        value.add_(.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    raw = {"batch_size": 2, "patch_shape_zyx": [8, 8, 8],
           "datasets": [{"base_volume_path": "unused"}], "test_datasets": [{"base_volume_path": "unused"}],
           "training": {"device": "cpu", "max_steps": 8, "run_path": str(tmp_path / "runs"),
                        "run_name": "initial-trace", "run_datestr": "fixed", "learning_rate": .01,
                        "bounded_gradient_accumulation": True, "optimizer_global_batch_size": 2,
                        "ddp_broadcast_buffers": False, "ema_decays": [.999],
                        "test_interval": 2, "checkpoint_interval": 2, "kept_snapshot_interval": 0,
                        "test_control_points": 2, "test_trace2cp_enabled": False,
                        "sample_vis_interval": 0, "test_sample_vis_interval": 0,
                        "tensorboard_enabled": False,
                        "reviewed_trace_test": {"interval": 2, "template": "unused",
                            "split_root": "unused", "native_binary": "unused", "weights": weights}}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(raw))
    resume = tmp_path / "resume.pt"
    train._save_snapshot(resume, model=model, optimizer=optimizer, step=7, config=raw,
                         metric=1., ema=ema, reviewed_trace_unlocked=True)
    monkeypatch.setattr(train, "build_fiber_trace_3d_model", lambda config: model)
    # This test targets publication after the dense test, not angle-gate quality.
    monkeypatch.setattr(train, "_reviewed_trace_unlocked", lambda previous, losses: True)

    class Batches:
        def begin_test(self, step):
            pass
        def end_test(self, step):
            pass
        def test_batches(self, *, global_batch_indices, **kwargs):
            return (loader.load_batch(index * 2) for index in global_batch_indices)

    metadata = SimpleNamespace(for_config=lambda config: loader)
    units = train.training_units(config, resume_checkpoint=resume, physical_microbatch_sizes=(2,),
        yield_initialized=True, batch_source=Batches(), dataset_metadata={"train": metadata, "test": metadata},
        summary_writer_factory=lambda *args, **kwargs: _Writer())
    try:
        for boundary in units:
            if boundary.phase == "reviewed_trace":
                assert boundary.step == 7  # Not a regular checkpoint interval.
                suffix = "" if weights == "raw" else "_ema_0p999"
                checkpoint, = (tmp_path / "runs").glob(f"**/snapshots/current{suffix}.pt")
                freeze_trace_checkpoint(checkpoint, tmp_path / "frozen.pt", step=boundary.step,
                                        samples_seen=boundary.samples_seen, weights=weights)
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                expected = model.state_dict() if weights == "raw" else ema.tracks[weights]["model"]
                for key, value in expected.items():
                    torch.testing.assert_close(saved["model"][key], value, rtol=0, atol=0)
                break
        else:
            pytest.fail("no initial reviewed-trace boundary")
    finally:
        units.close()
