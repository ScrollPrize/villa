import json
from pathlib import Path

import pytest
import torch

from vesuvius.neural_tracing.fiber_trace_3d.reviewed_trace_selection import (
    BEST_FILENAME,
    METRIC_NAME,
    benchmark_identity,
    finalize_attempt,
    freeze_raw_checkpoint,
    reconcile_completed_attempts,
    select_attempt,
    sha256_file,
    write_benchmark_identity,
    write_completion,
)


def _snapshot(path: Path, *, step: int, samples: int, value: float,
              ema: bool = False) -> None:
    payload = {
        "model": {"weight": torch.tensor([value])},
        "step": step,
        "samples_seen": samples,
        "metric": value,
        "metric_name": "test/loss_total",
    }
    if ema:
        payload.update(snapshot_kind="ema", ema_metadata={"decay": 0.99})
    torch.save(payload, path)


def _attempt(tmp_path: Path, name: str, *, step: int, restarts: int,
             identity_salt: str = "same") -> Path:
    output = tmp_path / name
    output.mkdir()
    source = tmp_path / f"source-{name}.pt"
    _snapshot(source, step=step, samples=step * 128, value=float(step))
    frozen = output / "evaluated_raw.pt"
    digest = freeze_raw_checkpoint(source, frozen, step=step, samples_seen=step * 128)
    template = tmp_path / f"template-{identity_salt}.json"
    template.write_text(json.dumps({"salt": identity_salt}))
    split = tmp_path / f"split-{identity_salt}"
    split.mkdir(exist_ok=True)
    (split / "manifest.json").write_text(json.dumps({"salt": identity_salt}))
    native = tmp_path / f"native-{identity_salt}"
    native.write_bytes(identity_salt.encode())
    runner = tmp_path / f"runner-{identity_salt}.py"
    runner.write_text(identity_salt)
    job = {
        "step": step,
        "samples_seen": step * 128,
        "checkpoint": str(frozen),
        "checkpoint_sha256": digest,
        "output_root": str(output),
        "template": str(template),
        "split_root": str(split),
        "native_binary": str(native),
        "runner": str(runner),
    }
    workload = {
        "checkpoint": str(frozen),
        "checkpoint_sha256": digest,
        "output_root": str(output),
        "split_root": str(split),
        "devices": "0",
        "pilot_devices": "0",
        "native_binary": str(native),
        "evaluation": {"mode": identity_salt},
    }
    metrics = {
        "restarts_forward": restarts,
        "restarts_reverse": 0,
        "restarts_combined": restarts,
        "samples_seen": step * 128,
    }
    (output / "job.json").write_text(json.dumps(job))
    (output / "workload.json").write_text(json.dumps(workload))
    (output / "training_metrics.json").write_text(json.dumps(metrics))
    identity = write_benchmark_identity(output, job, workload)
    write_completion(output, metrics=metrics, checkpoint_sha256=digest, identity=identity)
    return output


def test_restart_selected_raw_checkpoint_strictly_improves_and_ties_hold(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    dense_best = snapshots / "best.pt"
    ema_best = snapshots / "best_ema_0p99.pt"
    dense_best.write_bytes(b"dense-best")
    ema_best.write_bytes(b"ema-best")

    first = select_attempt(_attempt(tmp_path, "first", step=10, restarts=7), snapshots)
    assert first["updated"] and first["selected"]
    best = torch.load(snapshots / BEST_FILENAME, map_location="cpu", weights_only=True)
    assert best["metric"] == 7
    assert best["metric_name"] == METRIC_NAME
    assert best["step"] == 10

    tie = select_attempt(_attempt(tmp_path, "tie", step=11, restarts=7), snapshots)
    assert not tie["updated"] and not tie["selected"]
    worse = select_attempt(_attempt(tmp_path, "worse", step=12, restarts=9), snapshots)
    assert not worse["updated"] and not worse["selected"]
    better = select_attempt(_attempt(tmp_path, "better", step=13, restarts=4), snapshots)
    assert better["updated"] and better["selected"]
    best = torch.load(snapshots / BEST_FILENAME, map_location="cpu", weights_only=True)
    assert best["metric"] == 4 and best["step"] == 13
    assert dense_best.read_bytes() == b"dense-best"
    assert ema_best.read_bytes() == b"ema-best"


def test_selection_is_idempotent_and_finalize_removes_transient_checkpoint(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    attempt = _attempt(tmp_path, "attempt", step=5, restarts=3)
    first = select_attempt(attempt, snapshots)
    second = select_attempt(attempt, snapshots)
    assert second == first
    metrics, selection, output = finalize_attempt(attempt, snapshots)
    assert output == attempt and metrics["restarts_combined"] == 3
    assert selection == first
    assert not (attempt / "evaluated_raw.pt").exists()


def test_older_selection_record_remains_valid_after_later_improvement(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    older = _attempt(tmp_path, "older", step=5, restarts=6)
    recorded = select_attempt(older, snapshots)
    select_attempt(_attempt(tmp_path, "newer", step=6, restarts=2), snapshots)
    assert select_attempt(older, snapshots) == recorded


def test_selection_rejects_corrupt_completed_record(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    attempt = _attempt(tmp_path, "attempt", step=5, restarts=3)
    select_attempt(attempt, snapshots)
    record = json.loads((attempt / "checkpoint_selection.json").read_text())
    record["best_restarts_combined"] = 999
    (attempt / "checkpoint_selection.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="disagrees"):
        select_attempt(attempt, snapshots)


def test_reconcile_promotes_completed_attempt_after_interrupted_finalization(tmp_path):
    run = tmp_path / "run"
    root = run / "reviewed_trace_test"
    root.mkdir(parents=True)
    attempt = _attempt(root, ".step_00000009.attempt-1-1", step=9, restarts=2)
    snapshots = run / "snapshots"
    snapshots.mkdir()
    paths = reconcile_completed_attempts(run, snapshots)
    canonical = root / "step_00000009"
    assert paths == [canonical]
    assert canonical.is_dir()
    assert (canonical / "checkpoint_selection.json").is_file()
    assert not (canonical / "evaluated_raw.pt").exists()
    assert (snapshots / BEST_FILENAME).is_file()


def test_reconcile_ignores_incomplete_and_revoked_attempts(tmp_path):
    run = tmp_path / "run"
    root = run / "reviewed_trace_test"
    root.mkdir(parents=True)
    incomplete = root / ".step_00000007.attempt-1-1"
    incomplete.mkdir()
    revoked = _attempt(root, ".step_00000008.attempt-1-1", step=8, restarts=1)
    revoked.rename(root / f"{revoked.name}.revoked")
    snapshots = run / "snapshots"
    snapshots.mkdir()

    assert reconcile_completed_attempts(run, snapshots) == []
    assert not (snapshots / BEST_FILENAME).exists()
    assert incomplete.is_dir()


def test_selection_rejects_benchmark_identity_change(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    select_attempt(_attempt(tmp_path, "first", step=1, restarts=5,
                            identity_salt="one"), snapshots)
    with pytest.raises(ValueError, match="benchmark identity changed"):
        select_attempt(_attempt(tmp_path, "second", step=2, restarts=4,
                                identity_salt="two"), snapshots)


@pytest.mark.parametrize("value", [True, -1, 1.5, float("inf"), float("nan")])
def test_completion_rejects_invalid_restart_metric(tmp_path, value):
    with pytest.raises(ValueError, match="nonnegative integer"):
        write_completion(
            tmp_path,
            metrics={"restarts_combined": value, "samples_seen": 1},
            checkpoint_sha256="a" * 64,
            identity={"sha256": "b" * 64},
        )


def test_selection_rejects_checkpoint_mutation_and_ema_source(tmp_path):
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    attempt = _attempt(tmp_path, "mutated", step=3, restarts=1)
    with (attempt / "evaluated_raw.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="digest mismatch"):
        select_attempt(attempt, snapshots)

    source = tmp_path / "ema.pt"
    _snapshot(source, step=1, samples=128, value=1.0, ema=True)
    with pytest.raises(ValueError, match="raw model weights"):
        freeze_raw_checkpoint(source, tmp_path / "ema-frozen.pt", step=1, samples_seen=128)


def test_benchmark_identity_ignores_checkpoint_and_output_but_not_workload(tmp_path):
    attempt = _attempt(tmp_path, "identity", step=1, restarts=1)
    job = json.loads((attempt / "job.json").read_text())
    workload = json.loads((attempt / "workload.json").read_text())
    original = benchmark_identity(job, workload)
    workload.update(checkpoint="elsewhere", checkpoint_sha256="0" * 64,
                    output_root="elsewhere", devices="0,1")
    assert benchmark_identity(job, workload) == original
    workload["evaluation"]["mode"] = "changed"
    assert benchmark_identity(job, workload) != original
