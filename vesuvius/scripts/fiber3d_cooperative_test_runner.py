"""Fixed-entry test caller; edit adjacent JSON instead of the approved command.

Only named test suites run. CUDA suites have small allocator caps in their tests
(4 GiB numerical oracle; 6 GiB/cohort worker + 4 GiB driver reserve). No unbounded
reference benchmark is launched by this caller. GPU suites first reserve the
whole manager inventory exclusively and non-preemptibly. Suites run sequentially.
"""

from datetime import datetime, timezone
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
# Reuse the manager's nonblocking driver parser; do not create another one.
sys.path.insert(0, str(ROOT / "gpu_manager/src"))
from villa_gpu_manager.processes import NvidiaMonitor
CPU_TESTS = [
    "tests/neural_tracing/test_fiber_trace_3d.py",
    "tests/neural_tracing/test_gradient_accumulation.py",
    "tests/neural_tracing/test_training_state.py",
    "tests/neural_tracing/test_memory_calibration.py",
    "tests/neural_tracing/test_cooperative_resources.py",
    "tests/neural_tracing/test_sample_broker.py",
    "tests/neural_tracing/test_ordered_sample_pipeline.py",
    "tests/neural_tracing/test_cooperative_integration.py",
    "tests/neural_tracing/test_cooperative_reference.py",
]
SUITES = {
    "reproduction": ["tests/neural_tracing/test_reproduction.py", "../gpu_manager/tests/test_stop_guard.py", "../gpu_manager/tests/test_exclusive.py"],
    "loss-regression": ["tests/neural_tracing/test_fiber_trace_3d.py", "tests/neural_tracing/test_fiber_embeddings.py", "-k", "loss or bounded_gradients or two_branch_positive_supervision"],
    "stop-guard": ["../gpu_manager/tests/test_stop_guard.py"],
    "shutdown": ["../gpu_manager/tests/test_stop_guard.py", "../gpu_manager/tests/test_supervisor.py", "tests/neural_tracing/test_ordered_sample_pipeline.py", "tests/neural_tracing/test_sample_broker.py", "tests/neural_tracing/test_cooperative_integration.py"],
    "single-backward": ["tests/neural_tracing/test_single_backward.py", "tests/neural_tracing/test_gradient_accumulation.py", "tests/neural_tracing/test_cooperative_integration.py"],
    "retrain-merge": ["../lasagna/tests/test_demand_predict3d.py", "../lasagna/tests/test_dynamic_inference_batching.py", "../lasagna/tests/test_inference_profiling.py", "tests/neural_tracing/test_replay_metrics.py", "tests/neural_tracing/test_managed_evaluation.py", "tests/neural_tracing/test_cached_trace_profile.py", "tests/neural_tracing/test_live_inference_profile.py", "tests/neural_tracing/test_native_bbox_read.py"],
    "cuda-arc": ["tests/neural_tracing/test_arc_sampling.py", "tests/neural_tracing/test_single_backward.py", "-k", "cuda_dense", "-s"],
    "arc-sampling": ["tests/neural_tracing/test_arc_sampling.py", "tests/neural_tracing/test_collection_training_config.py"],
    "sampling-regression": ["tests/neural_tracing/test_fiber_trace_3d.py", "tests/neural_tracing/test_queued_batches.py", "tests/neural_tracing/test_evaluation_pipeline.py", "tests/neural_tracing/test_fiber_embeddings.py"],
    "visualization": ["tests/neural_tracing/test_forward_visualization.py", "tests/neural_tracing/test_evaluation_pipeline.py", "tests/neural_tracing/test_sample_broker.py", "tests/neural_tracing/test_cooperative_integration.py"],
    "unified-loading": ["tests/neural_tracing/test_queued_batches.py", "tests/neural_tracing/test_training_state.py", "tests/neural_tracing/test_evaluation_pipeline.py", "tests/neural_tracing/test_gradient_accumulation.py"],
    "unified-managed": ["tests/neural_tracing/test_cooperative_integration.py", "tests/neural_tracing/test_startup_calibration.py", "tests/neural_tracing/test_cooperative_setup.py", "tests/neural_tracing/test_sample_broker.py", "../gpu_manager/tests/test_supervisor.py"],
    "cpu": CPU_TESTS,
    "manager": ["../gpu_manager/tests"],
    "reference-unit": ["tests/neural_tracing/test_cooperative_reference.py"],
    "cuda-amp": ["tests/neural_tracing/test_gradient_accumulation.py", "-k", "cuda_amp"],
    "cuda-cohort": ["tests/neural_tracing/test_cooperative_integration.py", "-k", "different_rank_count and cuda"],
    "cuda-reference": [],
    "native-build": [],
    "native-configure": [],
    "native-tests": [],
}


def test_budget_seconds(control):
    timeout = control["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("test timeout_seconds must be an integer in [1,300] for the entire invocation")
    if "cuda-reference" in control["suites"] and control.get("reference", {}).get("phase") == "benchmark":
        raise ValueError("long benchmark sweeps are not tests; use a short reference lifecycle or profile")
    return timeout


def main():
    control = json.loads((SCRIPT_DIR / "fiber3d_cooperative_tests.json").read_text())
    if set(control) - {"suites", "gpu_devices", "timeout_seconds", "output_root", "reference", "native_build"}:
        raise ValueError("unexpected/missing test control fields")
    suites, devices = control["suites"], control["gpu_devices"]
    if not isinstance(suites, list) or not suites or any(s not in SUITES for s in suites):
        raise ValueError(f"suites must select from {list(SUITES)}")
    if not isinstance(devices, list) or any(not isinstance(d, str) or not d or "," in d for d in devices):
        raise ValueError("gpu_devices must be a list of individual device IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("duplicate GPU device")
    if any(s in suites for s in ("cuda-amp", "cuda-arc")) and not devices or "cuda-cohort" in suites and len(devices) != 2:
        raise ValueError("CUDA oracle needs a device; cohort test needs exactly two")
    timeout = test_budget_seconds(control)
    output = Path(control["output_root"]).expanduser() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "control.json").write_text(json.dumps(control, indent=2) + "\n")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    (output / "source.json").write_text(json.dumps({"commit": revision, "worktree_status": diff}, indent=2) + "\n")
    results = []
    invocation_started = time.monotonic()
    for suite in suites:
        env = dict(os.environ, AGENTS_AGENT_MODE="1", CUDA_VISIBLE_DEVICES="",
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   GPU_MANAGER_SCHEMA_PYTHON="/usr/bin/python3",
                   PYTHONPATH=f"{ROOT / 'vesuvius/src'}:{ROOT / 'gpu_manager/src'}:{ROOT}")
        env.pop("FIBER_COOPERATIVE_CUDA_TEST", None)
        env.pop("FIBER_COOPERATIVE_TEST_GPUS", None)
        if control.get("native_build", False):
            env["FIBER_COOPERATIVE_NATIVE_BUILD"] = str(ROOT / 'volume-cartographer/build/dev-quickbuild-gcc/python')
            env["PYTHONPATH"] = f"{ROOT / 'vesuvius/scripts/cooperative_test_support'}:" + env["PYTHONPATH"]
        if suite in ("cuda-amp", "cuda-arc"):
            env.update(CUDA_VISIBLE_DEVICES=devices[0], FIBER_COOPERATIVE_CUDA_TEST="1")
        elif suite in ("cuda-cohort", "cuda-reference"):
            env["FIBER_COOPERATIVE_TEST_GPUS"] = ",".join(devices)
        command = [sys.executable, "-m", "pytest", "-q", "-o", "faulthandler_timeout=60", *SUITES[suite]]
        if suite == "cuda-reference":
            env.update(FIBER_COOPERATIVE_TEST_CONTROL=str(output / "control.json"),
                       FIBER_COOPERATIVE_TEST_OUTPUT=str(output))
            command = [sys.executable, "-m", "vesuvius.neural_tracing.fiber_trace_3d.cooperative_reference_validation"]
        elif suite == "native-configure":
            command = ["cmake", "-S", str(ROOT / "volume-cartographer"),
                       "-B", str(ROOT / "volume-cartographer/build/dev-quickbuild-gcc"),
                       "-DCMAKE_AUTOGEN_PARALLEL=1", "-DCMAKE_BUILD_TYPE=QuickBuild",
                       "-DVC_QUICKBUILD_OPT_LEVEL=3"]
        elif suite == "native-build":
            env.update(CCACHE_DIR="/tmp/ccache", CCACHE_TEMPDIR="/tmp/ccache-tmp")
            command = ["cmake", "--build", str(ROOT / "volume-cartographer/build/dev-quickbuild-gcc"),
                       "--target", "vc_volume", "test_chunk_cache", "test_fiber_json",
                       "test_fiberlet_crop_trace", "-j", "4"]
            # GCC -flto=auto otherwise starts a host-sized pool inside each
            # linker, independently of Ninja's job limit. Bound this test build
            # without changing optimized production compiler flags.
            if hasattr(os, "sched_getaffinity"):
                cores = sorted(os.sched_getaffinity(0))[:4]
                command = ["taskset", "--cpu-list", ",".join(map(str, cores)), *command]
        elif suite == "native-tests":
            env["VC_TEST_TRACE"] = "1"
            command = ["ctest", "--test-dir", str(ROOT / "volume-cartographer/build/dev-quickbuild-gcc"),
                       "--output-on-failure", "--timeout", "120", "-R", "^(test_chunk_cache|test_fiber_json|test_fiberlet_crop_trace)$"]
        log = output / f"{suite}.log"
        print(f"[{suite}] log={log}", flush=True)
        started = time.monotonic()
        monitor = None
        peaks = {}
        if suite.startswith("cuda-"):
            selected = devices[:1] if suite in ("cuda-amp", "cuda-arc") else devices
            uuids = subprocess.check_output(["nvidia-smi", f"--id={','.join(selected)}",
                "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).splitlines()
            peaks = {uuid.strip(): 0 for uuid in uuids}
            monitor = NvidiaMonitor({"cpu_cores": 1, "gpus": {gpu: 20 * 1024**3 for gpu in peaks}},
                                    0, interval=0.25)
        with ExitStack() as resources, log.open("w") as stream:
            managed_command = None
            if suite.startswith("cuda-"):
                from villa_gpu_manager.bootstrap import ensure_manager
                from villa_gpu_manager.exclusive import ExclusiveTask
                from villa_gpu_manager.exclusive_command import ExclusiveCommand
                wait_started = time.monotonic()
                manager_socket, inventory = ensure_manager()
                lease = resources.enter_context(ExclusiveTask(manager_socket, inventory,
                    label=f"fiber-tests-{suite}", priority=100))
                # Resource waiting is not test execution time.
                invocation_started += time.monotonic() - wait_started
                started = time.monotonic()
                env["FIBER_COOPERATIVE_TEST_GPUS"] = ",".join(peaks)
                if suite in ("cuda-amp", "cuda-arc"):
                    env["CUDA_VISIBLE_DEVICES"] = next(iter(peaks))
                managed_command = ExclusiveCommand(command, cwd=ROOT / "vesuvius", env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                lease.cohort = managed_command
                lease.authorize()
                process = managed_command.process
            else:
                process = subprocess.Popen(command, cwd=ROOT / "vesuvius", env=env,
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                known_pids = {process.pid}
                while process.poll() is None:
                    if managed_command is not None:
                        managed_command.observe()
                    if time.monotonic() - invocation_started > timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    if monitor is not None:
                        try:
                            known_pids.update(child.pid for child in psutil.Process(process.pid).children(recursive=True))
                        except psutil.NoSuchProcess:
                            pass
                        monitor.poll(known_pids)
                        if time.monotonic() - (monitor.last_snapshot_at or started) > 20:
                            raise RuntimeError("test driver memory monitoring unavailable")
                        if monitor.last_snapshot is not None:
                            for gpu in peaks:
                                used = sum(size for pid, size in monitor.last_snapshot[gpu]["processes"].items()
                                           if pid in known_pids)
                                peaks[gpu] = max(peaks[gpu], used)
                                if used > 20 * 1024**3:
                                    raise RuntimeError(f"test exceeded 20 GiB on {gpu}; aborting only test processes")
                    time.sleep(0.1)
                code = process.returncode
            except BaseException:
                if managed_command is not None:
                    managed_command.abort()
                    raise
                # Kill only this test's process group, never the user's training.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            finally:
                if monitor is not None:
                    monitor.close()
        results.append({"suite": suite, "command": command, "exit_code": code,
                        "elapsed_seconds": time.monotonic() - started, "log": str(log),
                        "owned_driver_peak_bytes": peaks})
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(log.read_text(), end="", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
