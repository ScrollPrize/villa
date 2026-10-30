"""Crash-safe raw checkpoint selection from reviewed-trace results."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import torch


BEST_FILENAME = "best_reviewed_trace.pt"
LAST_BEST_FILENAME = "best_reviewed_trace_last.pt"
METRIC_NAME = "reviewed_trace/restarts_combined"
SCHEMA_VERSION = 1


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _raw_snapshot_metadata(path: Path, *, expected_step: int,
                           expected_samples: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"reviewed-trace raw checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict):
        raise ValueError(f"reviewed-trace checkpoint is not a mapping: {path}")
    if payload.get("snapshot_kind") not in (None, "raw") or "ema_metadata" in payload:
        raise ValueError(f"reviewed-trace checkpoint must contain raw model weights: {path}")
    if int(payload.get("step", -1)) != int(expected_step):
        raise ValueError(
            f"reviewed-trace checkpoint step mismatch: expected {expected_step}, "
            f"found {payload.get('step')}"
        )
    if int(payload.get("samples_seen", -1)) != int(expected_samples):
        raise ValueError(
            f"reviewed-trace checkpoint sample mismatch: expected {expected_samples}, "
            f"found {payload.get('samples_seen')}"
        )
    return payload


def freeze_raw_checkpoint(source: Path, destination: Path, *, step: int,
                          samples_seen: int) -> str:
    """Copy and verify the immutable checkpoint that a trace attempt evaluates."""
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _raw_snapshot_metadata(
            destination, expected_step=step, expected_samples=samples_seen
        )
        return sha256_file(destination)
    _raw_snapshot_metadata(source, expected_step=step, expected_samples=samples_seen)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        source_digest = sha256_file(source)
        frozen_digest = sha256_file(temporary)
        if source_digest != frozen_digest:
            raise RuntimeError("raw checkpoint changed while being frozen for reviewed tracing")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    _raw_snapshot_metadata(
        destination, expected_step=step, expected_samples=samples_seen
    )
    return frozen_digest


def benchmark_identity(job: Mapping[str, Any], workload: Mapping[str, Any]) -> dict[str, Any]:
    """Identify the benchmark independently of checkpoint, output, and GPU grant."""
    effective = deepcopy(dict(workload))
    for key in (
        "checkpoint", "checkpoint_sha256", "output_root", "split_root",
        "devices", "pilot_devices", "native_binary",
    ):
        effective.pop(key, None)
    split_manifest = Path(str(job["split_root"])) / "manifest.json"
    identity = {
        "version": SCHEMA_VERSION,
        "effective_workload_sha256": _canonical_digest(effective),
        "template_sha256": sha256_file(Path(str(job["template"]))),
        "split_manifest_sha256": sha256_file(split_manifest),
        "native_binary_sha256": sha256_file(Path(str(job["native_binary"]))),
        "runner_sha256": sha256_file(Path(str(job["runner"]))),
    }
    identity["sha256"] = _canonical_digest(identity)
    return identity


def write_benchmark_identity(output: Path, job: Mapping[str, Any],
                             workload: Mapping[str, Any]) -> dict[str, Any]:
    identity = benchmark_identity(job, workload)
    _atomic_json(output / "benchmark_identity.json", identity)
    return identity


def _restart_count(metrics: Mapping[str, Any]) -> int:
    value = metrics.get("restarts_combined")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("reviewed trace restarts_combined must be a nonnegative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("reviewed trace restarts_combined must be a nonnegative integer")
    return int(numeric)


def write_completion(output: Path, *, metrics: Mapping[str, Any],
                     checkpoint_sha256: str,
                     identity: Mapping[str, Any]) -> None:
    restarts = _restart_count(metrics)
    samples = metrics.get("samples_seen")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 0:
        raise ValueError("reviewed trace samples_seen must be a nonnegative integer")
    _atomic_json(output / "completion.json", {
        "version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "benchmark_identity_sha256": identity["sha256"],
        "restarts_combined": restarts,
        "samples_seen": samples,
    })


def _validated_attempt(output: Path) -> tuple[dict[str, Any], dict[str, Any],
                                              dict[str, Any], Path, str, int, int]:
    required = {
        name: output / name for name in (
            "job.json", "workload.json", "training_metrics.json",
            "benchmark_identity.json", "completion.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise ValueError(f"reviewed trace is incomplete; missing {missing}")
    job = json.loads(required["job.json"].read_text())
    workload = json.loads(required["workload.json"].read_text())
    metrics = json.loads(required["training_metrics.json"].read_text())
    identity = json.loads(required["benchmark_identity.json"].read_text())
    completion = json.loads(required["completion.json"].read_text())
    if benchmark_identity(job, workload) != identity:
        raise ValueError("reviewed trace benchmark identity does not match its inputs")
    checkpoint = Path(str(job["checkpoint"])).resolve()
    if checkpoint.parent != output.resolve():
        raise ValueError("reviewed trace checkpoint is not owned by its attempt directory")
    checkpoint_digest = sha256_file(checkpoint)
    if (workload.get("checkpoint_sha256") != checkpoint_digest
            or completion.get("checkpoint_sha256") != checkpoint_digest):
        raise ValueError("reviewed trace checkpoint digest mismatch")
    if completion.get("benchmark_identity_sha256") != identity["sha256"]:
        raise ValueError("reviewed trace completion benchmark identity mismatch")
    restarts = _restart_count(metrics)
    if completion.get("restarts_combined") != restarts:
        raise ValueError("reviewed trace completion metric mismatch")
    step = job.get("step")
    samples = job.get("samples_seen")
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or isinstance(samples, bool) or not isinstance(samples, int) or samples < 0
            or metrics.get("samples_seen") != samples
            or completion.get("samples_seen") != samples):
        raise ValueError("reviewed trace step/sample provenance is invalid")
    _raw_snapshot_metadata(checkpoint, expected_step=step, expected_samples=samples)
    return job, metrics, identity, checkpoint, checkpoint_digest, restarts, samples


def _load_existing_best(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (not isinstance(payload, dict)
            or payload.get("snapshot_kind") not in (None, "raw")
            or "ema_metadata" in payload
            or payload.get("metric_name") != METRIC_NAME):
        raise ValueError(f"invalid reviewed-trace best checkpoint: {path}")
    _restart_count({"restarts_combined": payload.get("metric")})
    selection = payload.get("reviewed_trace_selection")
    if not isinstance(selection, dict):
        raise ValueError(f"reviewed-trace best checkpoint lacks selection metadata: {path}")
    return payload


def _save_checkpoint(destination: Path, payload: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def select_attempt(output: Path, snapshot_dir: Path) -> dict[str, Any]:
    """Idempotently consider one completed trace attempt for the raw best."""
    marker = output / "checkpoint_selection.json"
    latest_path = snapshot_dir / LAST_BEST_FILENAME
    # Legacy runs already kept the first minimum. Seed the new companion without
    # changing that historical checkpoint or requiring its deleted attempt file.
    if not latest_path.exists():
        existing = _load_existing_best(snapshot_dir / BEST_FILENAME)
        if existing is not None:
            _save_checkpoint(latest_path, existing)
    if marker.is_file():
        result = json.loads(marker.read_text())
        if not isinstance(result, dict) or result.get("version") != SCHEMA_VERSION:
            raise ValueError(f"invalid reviewed-trace selection record: {marker}")
        job = json.loads((output / "job.json").read_text())
        metrics = json.loads((output / "training_metrics.json").read_text())
        identity = json.loads((output / "benchmark_identity.json").read_text())
        completion = json.loads((output / "completion.json").read_text())
        candidate = {
            "candidate_step": job.get("step"),
            "candidate_samples_seen": job.get("samples_seen"),
            "candidate_restarts_combined": _restart_count(metrics),
            "candidate_checkpoint_sha256": completion.get("checkpoint_sha256"),
            "benchmark_identity_sha256": identity.get("sha256"),
        }
        if any(result.get(key) != value for key, value in candidate.items()):
            raise ValueError("reviewed-trace selection record disagrees with its result")
        best = _load_existing_best(snapshot_dir / BEST_FILENAME)
        if best is None:
            raise ValueError("reviewed-trace selection record has no best checkpoint")
        best_selection = best["reviewed_trace_selection"]
        if result["benchmark_identity_sha256"] != best_selection.get(
                "benchmark_identity_sha256"):
            raise ValueError("reviewed-trace selection record benchmark identity changed")
        if best_selection.get("evaluated_checkpoint_sha256") == result[
                "candidate_checkpoint_sha256"]:
            if (result.get("best_step") != best_selection.get("step")
                    or result.get("best_samples_seen") != best_selection.get("samples_seen")
                    or result.get("best_restarts_combined") != best.get("metric")):
                raise ValueError("reviewed-trace selection record disagrees with best checkpoint")
        elif int(best["metric"]) > result["candidate_restarts_combined"]:
            raise ValueError("reviewed-trace best checkpoint regressed after recorded selection")
        return result
    job, metrics, identity, checkpoint, digest, restarts, samples = _validated_attempt(output)
    destination = snapshot_dir / BEST_FILENAME
    previous = _load_existing_best(destination)
    previous_restarts = math.inf if previous is None else int(previous["metric"])
    if previous is not None:
        previous_identity = previous["reviewed_trace_selection"].get("benchmark_identity_sha256")
        if previous_identity != identity["sha256"]:
            raise ValueError(
                "reviewed trace benchmark identity changed within one training run; "
                "start a new run before comparing restart counts"
            )
    same_candidate = bool(previous is not None and previous["reviewed_trace_selection"].get(
        "evaluated_checkpoint_sha256") == digest)
    updated = restarts < previous_restarts
    selected = updated or (restarts == previous_restarts and same_candidate)
    latest = _load_existing_best(latest_path)
    if latest is not None and latest['reviewed_trace_selection'].get('benchmark_identity_sha256') != identity['sha256']:
        raise ValueError('reviewed trace latest-best benchmark identity changed')
    update_latest = (latest is None or restarts < int(latest['metric'])
                     or (restarts == int(latest['metric'])
                         and samples > int(latest['reviewed_trace_selection']['samples_seen'])))
    if updated or update_latest:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        payload["snapshot_kind"] = "raw"
        payload["metric"] = restarts
        payload["metric_name"] = METRIC_NAME
        payload["reviewed_trace_metrics"] = dict(metrics)
        payload["reviewed_trace_selection"] = {
            "version": SCHEMA_VERSION,
            "step": int(job["step"]),
            "samples_seen": samples,
            "evaluated_checkpoint_sha256": digest,
            "benchmark_identity_sha256": identity["sha256"],
        }
        if updated:
            _save_checkpoint(destination, payload)
        if update_latest:
            _save_checkpoint(latest_path, payload)
    best = _load_existing_best(destination)
    latest = _load_existing_best(latest_path)
    result = {
        "version": SCHEMA_VERSION,
        "candidate_step": int(job["step"]),
        "candidate_samples_seen": samples,
        "candidate_restarts_combined": restarts,
        "candidate_checkpoint_sha256": digest,
        "benchmark_identity_sha256": identity["sha256"],
        "selected": selected,
        "updated": updated,
        "best_step": int(best["reviewed_trace_selection"]["step"]),
        "best_samples_seen": int(best["reviewed_trace_selection"]["samples_seen"]),
        "best_restarts_combined": int(best["metric"]),
        "best_checkpoint": str(destination),
        "latest_updated": update_latest,
        "latest_best_checkpoint": str(latest_path),
        "latest_best_samples_seen": int(latest['reviewed_trace_selection']['samples_seen']),
    }
    _atomic_json(marker, result)
    return result


def finalize_attempt(output: Path, snapshot_dir: Path, *, canonical: Path | None = None
                     ) -> tuple[dict[str, Any], dict[str, Any], Path]:
    selection = select_attempt(output, snapshot_dir)
    metrics = json.loads((output / "training_metrics.json").read_text())
    destination = output
    if canonical is not None and output.resolve() != canonical.resolve():
        if canonical.exists():
            raise RuntimeError(f"refusing to overwrite reviewed trace result: {canonical}")
        output.rename(canonical)
        destination = canonical
    evaluated = destination / "evaluated_raw.pt"
    evaluated.unlink(missing_ok=True)
    return metrics, selection, destination


def reconcile_completed_attempts(run_dir: Path, snapshot_dir: Path) -> list[Path]:
    """Finish successful transactions interrupted before canonical promotion."""
    root = run_dir / "reviewed_trace_test"
    if not root.is_dir():
        return []
    reconciled = []
    for attempt in sorted(root.glob(".step_*.attempt-*")):
        if ".revoked" in attempt.name or not (attempt / "completion.json").is_file():
            continue
        job = json.loads((attempt / "job.json").read_text())
        canonical = root / f"step_{int(job['step']):08d}"
        _, _, promoted = finalize_attempt(attempt, snapshot_dir, canonical=canonical)
        reconciled.append(promoted)
    for canonical in sorted(root.glob("step_*")):
        if not canonical.is_dir() or not (canonical / "completion.json").is_file():
            continue
        if not (canonical / "checkpoint_selection.json").is_file():
            finalize_attempt(canonical, snapshot_dir)
            reconciled.append(canonical)
        else:
            (canonical / "evaluated_raw.pt").unlink(missing_ok=True)
    return reconciled
