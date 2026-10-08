"""Run and report the locked reviewed-failure trace test for a training step."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .reviewed_failure_test import directional_restart_metrics, validate_failure_union
from .evaluation_workload import prepare, run_prepared
from .reviewed_trace_selection import (
    sha256_file,
    write_benchmark_identity,
    write_completion,
)


def build_workload(template: dict, *, checkpoint: Path, output_root: Path,
                   split_root: Path, devices: str, native_binary: Path) -> dict:
    from lasagna.inference_provenance import sha256_file
    manifest = validate_failure_union(split_root, expected_count=32)
    counts: dict[str, int] = {}
    for row in manifest["files"]:
        counts[row["source"]] = counts.get(row["source"], 0) + 1
    result = deepcopy(template)
    result.update(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
                  output_root=str(output_root.resolve()), split_root=str(split_root.resolve()),
                  devices=devices, pilot_devices=devices, native_binary=str(native_binary.resolve()))
    result.pop("pilot_result", None)
    result["sources"] = [dict(source, expected_test_count=counts[source["id"]])
                         for source in result["sources"] if source["id"] in counts]
    if {source["id"] for source in result["sources"]} != set(counts):
        raise ValueError("trace workload template does not cover the locked failure cohort")
    evaluation = result["evaluation"]
    evaluation["render"] = False
    evaluation["workers"] = min(int(evaluation["workers"]), 128)
    for key in ("input_workers", "flush_workers", "accumulator_workers"):
        evaluation["inference"][key] = min(int(evaluation["inference"][key]), 128)
    evaluation["inference_cache_namespace"] = "training-reviewed-failure-v1"
    # The benchmark is launched as a descendant of the training supervisor and
    # uses that task's already-held grant.  Registering it as another manager
    # task would overlap the same process tree and deadlock the handoff.
    evaluation["gpu_manager"]["enabled"] = False
    result["training_cleanup_prediction_cache"] = True
    return result


def cleanup_prediction_cache(workload: dict, output: Path) -> None:
    if workload.get("training_cleanup_prediction_cache") is not True:
        raise ValueError("training trace workload must explicitly clean its transient prediction cache")
    binary = Path(workload["native_binary"])
    cache_root = Path(subprocess.check_output(
        [str(binary.with_name("vc_cache_root"))], text=True,
    ).strip()).resolve()
    namespace = workload["evaluation"]["inference_cache_namespace"]
    target = (cache_root / "live_inference" / "_namespaces" / namespace).resolve()
    expected_parent = (cache_root / "live_inference" / "_namespaces").resolve()
    if target.parent != expected_parent or target.name != "training-reviewed-failure-v1":
        raise ValueError(f"refusing unsafe training prediction cleanup: {target}")
    existed = target.exists()
    if existed:
        shutil.rmtree(target)
    (output / "cache_cleanup.json").write_text(json.dumps({
        "cache_namespace": namespace, "path": str(target),
        "existed": existed, "removed": not target.exists(),
    }, indent=2) + "\n")


def run_job(job_path: Path, *, worker_factory=None, cancel_event=None, on_process_start=None) -> dict[str, float]:
    job = json.loads(job_path.read_text())
    if job.get("inherited_manager_grant") is not True:
        raise ValueError("training trace benchmark requires an inherited manager grant")
    grant = job.get("gpu_grant")
    if (not isinstance(grant, dict) or not grant
            or any(not isinstance(gpu, str) or not gpu.startswith("GPU-")
                   or type(vram) is not int or vram <= 0
                   for gpu, vram in grant.items())):
        raise ValueError("training trace benchmark requires the exact inherited GPU grant")
    visible = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    expected_uuids = sorted(grant)
    expected_devices = ",".join(str(index) for index in range(len(expected_uuids)))
    if (worker_factory is None and visible != expected_uuids) or job.get("devices") != expected_devices:
        raise ValueError(
            f"training trace GPU binding does not match inherited grant: "
            f"visible={visible} devices={job.get('devices')!r} grant={expected_uuids}"
        )
    output = Path(job["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(job["checkpoint"])
    checkpoint_sha256 = sha256_file(checkpoint)
    if job.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("training trace job checkpoint digest mismatch")
    workload = build_workload(
        json.loads(Path(job["template"]).read_text()),
        checkpoint=checkpoint, output_root=output,
        split_root=Path(job["split_root"]), devices=job["devices"],
        native_binary=Path(job["native_binary"]),
    )
    workload_path = output / "workload.json"
    workload_path.write_text(json.dumps(workload, indent=2) + "\n")
    identity = write_benchmark_identity(output, job, workload)
    from concurrent.futures import CancelledError
    started = time.monotonic()
    metrics = None
    try:
        prepare(workload)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("training trace allocation revoked during preparation")
        run_prepared(workload, devices=job["devices"], worker_factory=worker_factory,
                     cancel_event=cancel_event, on_process_start=on_process_start)
        summary_path = output / "full" / "summary.json"
        metrics = directional_restart_metrics(json.loads(summary_path.read_text()), expected_pieces=32)
        metrics.update(samples_seen=int(job["samples_seen"]), pieces=32, directions=64,
                       wall_seconds=time.monotonic() - started)
        (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print("training reviewed trace: " + " ".join(f"{key}={value}" for key, value in metrics.items()), flush=True)
    finally:
        cleanup_prediction_cache(workload, output)
    write_completion(
        output,
        metrics=metrics,
        checkpoint_sha256=checkpoint_sha256,
        identity=identity,
    )
    return metrics


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.executable} -m {__package__}.training_trace_benchmark JOB.json")
    run_job(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
