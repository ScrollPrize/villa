"""Reference-data lifecycle oracle using the production broker/cohort/supervisor.

No alternative training loop: compare a fixed grant with pause and rank changes,
including the trainer's canonical evaluation and CPU handoff state.
"""

import json
import math
from contextlib import ExitStack
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time


class _PauseDeadline:
    """A driver-observation deadline belongs to one cohort, not the whole run."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.started = None

    def check(self, now):
        if self.started is None:
            self.started = now
        if now - self.started > 20:
            raise AssertionError("paused GPU ownership did not disappear from driver inventory")
        return self.started


def validate_benchmark_transitions(saved):
    """Check applied resize evidence, including retrospectively saved lifecycle JSON.

    A factory record alone is insufficient: its cohort must complete an update
    before the next factory record and before the next scheduled resize.
    """
    if isinstance(saved, (str, Path)):
        saved = json.loads(Path(saved).read_text())
    result = saved.get("dynamic", saved)
    cohorts = [row for row in result["transitions"] if "grant" in row]
    assert cohorts, "no cohort allocation evidence"
    full = cohorts[0]["grant"]["gpus"]
    assert len(full) == 2, "reference requires two initial GPUs"
    initial_shrinks = [row for row in cohorts if 128 <= row["frontier"] < 4 * 128
                       and len(row["grant"]["gpus"]) == 1]
    assert initial_shrinks, "missing initial one-GPU reference allocation"
    minimum = next(iter(initial_shrinks[0]["grant"]["gpus"].values()))
    assert all(minimum < value for value in full.values()), "no actual partial-VRAM range"
    evidence = []
    for start, stop, kind in ((20, 40, "partial"), (40, 60, "expand"),
                              (60, 80, "shrink"), (80, math.inf, "expand")):
        matches = []
        for index, cohort in enumerate(cohorts):
            frontier = cohort["frontier"] // 128
            if not start <= frontier < stop:
                continue
            gpus = cohort["grant"]["gpus"]
            if kind == "expand":
                expected = gpus == full
            elif kind == "partial":
                expected = set(gpus) == set(full) and all(v == minimum for v in gpus.values())
            else:
                expected = len(gpus) == 1 and set(gpus) <= set(full) and all(v == minimum for v in gpus.values())
            if not expected:
                continue
            next_frontier = cohorts[index + 1]["frontier"] // 128 if index + 1 < len(cohorts) else math.inf
            completed = [row for row in result["units"] if row["update_completed"]
                         and frontier < row["step"] <= min(next_frontier, stop)
                         and row["gpus"] == len(gpus)]
            if completed:
                matches.append({"kind": kind, "requested_after_step": start,
                                "frontier": cohort["frontier"], "grant": cohort["grant"],
                                "completed_step": completed[0]["step"]})
        assert matches, f"missing successfully applied benchmark {kind} after step {start}"
        evidence.append(matches[0])
    return evidence


def describe_state_differences(output, *, fixed_reference=None, payloads=None):
    """CPU-only diagnosis of saved reference states; does not relax assertions."""
    import torch
    files = ({"fixed": fixed_reference / "fixed-validation-state.pt",
              "dynamic": output / "fixed-validation-state.pt"} if fixed_reference is not None else
             {mode: output / f"{mode}-validation-state.pt" for mode in ("fixed", "dynamic")})
    if payloads is None:
        payloads = {mode: torch.load(files[mode],
                                    map_location="cpu", weights_only=False)
                    for mode in ("fixed", "dynamic")}
    report = {}
    for component in ("model", "optimizer"):
        rows = []

        def visit(a, b, path):
            if isinstance(a, torch.Tensor):
                assert a.shape == b.shape and a.dtype == b.dtype
                if not a.is_floating_point() or path.endswith("/step"):
                    assert torch.equal(a, b), path
                    return
                assert torch.isfinite(a).all() and torch.isfinite(b).all(), path
                delta = a.double() - b.double()
                rows.append({"path": path, "elements": a.numel(),
                    "max_abs": delta.abs().max().item() if a.numel() else 0,
                    "squared_difference": delta.square().sum().item(),
                    "squared_reference": b.double().square().sum().item(),
                    "strict_mismatches": int((delta.abs() > 2e-5 + 2e-3 * b.double().abs()).sum())})
            elif isinstance(a, dict):
                assert a.keys() == b.keys(), path
                for key in a:
                    visit(a[key], b[key], f"{path}/{key}")
            elif isinstance(a, (list, tuple)):
                assert len(a) == len(b), path
                for key, (x, y) in enumerate(zip(a, b)):
                    visit(x, y, f"{path}/{key}")
            else:
                assert a == b, path

        visit(payloads["fixed"][component], payloads["dynamic"][component], component)
        report[component] = {
            "elements": sum(row["elements"] for row in rows),
            "strict_mismatches": sum(row["strict_mismatches"] for row in rows),
            "relative_l2": (sum(row["squared_difference"] for row in rows) /
                            max(sum(row["squared_reference"] for row in rows), 1e-300)) ** 0.5,
            "max_abs": max(row["max_abs"] for row in rows),
            "largest_differences": sorted(rows, key=lambda row: row["max_abs"], reverse=True)[:5]}
    report["progress"] = {mode: {key: payloads[mode]["progress"][key]
                                for key in ("step", "sample_frontier", "best_metric", "pending_test")}
                          for mode in payloads}
    report["scaler_equal"] = payloads["fixed"].get("grad_scaler") == payloads["dynamic"].get("grad_scaler")
    (output / "numerical-differences.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def _serve(path, inventory, stop):
    from villa_gpu_manager.server import Server
    with Server(path, inventory) as server:
        while not stop.is_set():
            server.step(0.02)


class _DirectCohortRunner:
    """Test baseline: identical production cohort, without manager RPC/policy.

    This controls admission only; it does not implement any training operations.
    """
    def __init__(self, factory, grant):
        self.cohort = factory(grant, None)
        self.cohort.authorize()
        self.state = self.error = None
        self.finished = self.stopping = False

    def tick(self):
        self.cohort.poll()
        if self.cohort.done and not self.stopping:
            self.stopping = True
            self.cohort.drain()
        if self.cohort.closed:
            self.state = self.cohort.state
            self.cohort = None
            self.finished = True
        elif self.cohort.ready and self.cohort.idle and not self.stopping:
            self.cohort.start_unit()


def validate(config, calibration_path, output, devices, *, steps=6, modes=("fixed", "dynamic"),
             resize=True, fixed_reference=None, benchmark=False, timeout=1800, profile=False):
    import torch
    from villa_gpu_manager.client import Client
    from villa_gpu_manager.supervisor import Supervisor
    from villa_gpu_manager.processes import NvidiaMonitor
    from .cooperative_cohort import FiberCohort
    from .cooperative_logging import WriterHub
    from .cooperative_resources import MemoryCalibration
    from .loader import load_config
    from .sample_broker import CanonicalSampleBroker
    from .train import _load_raw_config, _make_summary_writer, _training_cuda_value_augmentation
    from .training_state import config_fingerprint

    raw = _load_raw_config(config)
    measured = json.loads(Path(calibration_path).read_text())
    if measured["config_fingerprint"] != config_fingerprint(raw):
        raise ValueError("reference lifecycle config differs from calibration")
    estimate = dict(measured["estimate"])
    estimate["measured_peaks"] = tuple(map(tuple, estimate["measured_peaks"]))
    calibration = MemoryCalibration(**estimate)
    allowance = 16 * 1024**3
    minimum = calibration.estimated_bytes(calibration.measured_peaks[0][0])
    required_steps = 6 if "dynamic" in modes else 2
    if minimum > allowance or len(devices) != 2 or steps < required_steps:
        raise ValueError(f"reference needs two GPUs, {required_steps} updates, and a fitting calibration")
    inventory = {"cpu_cores": int(raw["training"]["sample_pipeline_workers"]),
                 "gpus": {gpu: allowance for gpu in devices}}
    request = dict(max_cpu_cores=inventory["cpu_cores"], max_gpus=2,
        min_vram_bytes_per_gpu=minimum, max_vram_bytes_per_gpu=allowance,
        allowed_gpu_uuids=list(devices), priority=0, exclusive=False)
    results = {}
    payloads = {}
    if fixed_reference is not None:
        payloads["fixed"] = torch.load(fixed_reference / "fixed-validation-state.pt",
                                       map_location="cpu", weights_only=False)
        assert payloads["fixed"]["progress"]["config_fingerprint"] == config_fingerprint(raw)
        results["fixed"] = {"reused_reference": str(fixed_reference)}
    for mode in modes:
        context = mp.get_context("spawn")
        stop = context.Event()
        temporary = tempfile.TemporaryDirectory(prefix="fiber-lifecycle-")
        path = str(Path(temporary.name) / "manager.sock")
        server = context.Process(target=_serve, args=(path, inventory, stop))
        if mode != "unmanaged":
            server.start()
        client = Client(path)
        hub = WriterHub(_make_summary_writer)
        broker = None
        supervisor = None
        changes = []
        timings = []
        last_frontier = 0
        phase = 0
        paused_at = None
        pause_deadline = _PauseDeadline()
        owned_gpu_pids = set()
        completed_boundaries = []
        unit_timings = []
        additional_event = 0
        additional_pending = None
        additional_requested_at = None
        monitor = NvidiaMonitor(inventory, 0, interval=0.25)
        started = time.monotonic()

        class ObservedCohort(FiberCohort):
            def start_unit(self):
                self._reference_unit_started = time.monotonic()
                return super().start_unit()

            def _completed_unit(self, rows):
                nonlocal phase, additional_event, additional_pending, additional_requested_at
                before = broker.frontier // 128
                done = super()._completed_unit(rows)
                completed_boundaries.append(dict(rows[0]))
                unit_timings.append(dict(rows[0], seconds=time.monotonic() - self._reference_unit_started,
                    update_completed=rows[0]["step"] > before, gpus=len(self.grant["gpus"]),
                    microbatches=[calibration.microbatch_for(v) for v in self.grant["gpus"].values()]))
                # Inject control changes at the boundary, before tick() can
                # admit another unit (especially the final test-wave commit).
                if mode == "dynamic":
                    if phase == 0 and rows[0].get("phase") == "test":
                        client.pause(True)
                        phase = 1
                    elif phase == 3 and rows[0]["step"] >= 1:
                        client.pause(True)
                        phase = 4
                    elif phase == 6 and rows[0]["step"] >= 4:
                        client.set_request(request)
                        phase = 7
                    elif benchmark and phase == 7 and additional_event < 4 and rows[0]["step"] >= (20, 40, 60, 80)[additional_event]:
                        # Repeated pause, two-GPU partial VRAM, shrink and expansion.
                        if additional_event in (0, 2):
                            client.pause(True)
                            additional_pending = (dict(request, max_vram_bytes_per_gpu=minimum)
                                if additional_event == 0 else dict(request, max_gpus=1, max_vram_bytes_per_gpu=minimum))
                            additional_requested_at = time.monotonic()
                        else:
                            client.set_request(request)
                        additional_event += 1
                return done

        def factory(grant, state):
            nonlocal broker
            pause_deadline.reset()
            if broker is None:
                broker = CanonicalSampleBroker.create(config_path=config,
                    loader_config=load_config(config), start_step=0, max_steps=steps,
                    effective_batch_size=128, workers=inventory["cpu_cores"],
                    slot_count=128, initial_in_flight=8,
                    defer_cuda_value_augmentation=_training_cuda_value_augmentation(raw["training"]))
            changes.append({"seconds": time.monotonic() - started, "grant": grant,
                            "frontier": broker.frontier})
            print(f"reference {mode}: starting cohort gpus={len(grant['gpus'])} "
                  f"frontier={broker.frontier}", flush=True)
            cohort = ObservedCohort(config_path=config, grant=grant, state=state,
                broker=broker, calibration=calibration, writer_factory=hub.proxy_factory,
                calibration_observations=measured["observations"],
                training_overrides={"max_steps": steps, "run_path": str(output / mode),
                                    "run_datestr": "reference", **({"test_interval": steps,
                                    "checkpoint_interval": steps, "sample_vis_interval": steps} if benchmark else {}),
                                    **({"profile_enabled": True, "profile_trace_start": 2,
                                        "profile_trace_steps": 2} if profile else {})})
            owned_gpu_pids.update(row["pid"] for row in cohort.gpu.processes)
            return cohort

        def verify_pause(*, during_test):
            nonlocal paused_at
            assert supervisor.cohort is None
            assert not torch.cuda.is_initialized()
            paused_at = pause_deadline.check(time.monotonic())
            if monitor.last_snapshot_at is None or monitor.last_snapshot_at <= paused_at:
                return False
            assert monitor.last_snapshot is not None
            if set(monitor.last_snapshot) != set(devices):
                return False
            active = {pid for row in monitor.last_snapshot.values() for pid in row["processes"]
                      if pid in owned_gpu_pids}
            if active:
                return False
            pending = supervisor.state.load()["progress"]["pending_test"]
            if during_test:
                assert pending is not None and pending["rows"], "test-wave handoff was not exercised"
            changes.append({"paused_at_step": step, "gpu_workers_joined": True,
                            "driver_owned_processes": 0, "pending_test": pending is not None})
            print(f"reference {mode}: paused step={step} owned_GPU_processes=0 "
                  f"pending_test={pending is not None}", flush=True)
            return True

        try:
            while mode != "unmanaged" and not Path(path).exists():
                if not server.is_alive() or time.monotonic() - started > 15:
                    raise RuntimeError("reference manager failed to start")
                time.sleep(0.01)
            supervisor = (_DirectCohortRunner(factory, inventory) if mode == "unmanaged" else
                Supervisor(client, label=mode, request=request,
                    cohort_factory=factory, dispose_state=lambda state: state.close()))
            last_step_time = time.monotonic()
            while not supervisor.finished:
                if time.monotonic() - started > timeout:
                    raise TimeoutError("reference lifecycle exceeded configured timeout")
                hub.poll()
                supervisor.tick()
                monitor.poll(owned_gpu_pids)
                if supervisor.error is not None:
                    raise RuntimeError("reference lifecycle failed") from supervisor.error
                step = broker.frontier // 128 if broker is not None else 0
                if step != last_frontier:
                    now = time.monotonic()
                    timings.append({"step": step, "seconds": now - last_step_time})
                    last_frontier, last_step_time = step, now
                    print(f"reference {mode}: step={step}/{steps}", flush=True)
                if mode == "dynamic" and not supervisor.stopping:
                    if additional_pending is not None and supervisor.status == "paused":
                        if verify_pause(during_test=False):
                            changes[-1]["pause_latency_seconds"] = time.monotonic() - additional_requested_at
                            client.set_request(additional_pending)
                            client.pause(False)
                            additional_pending = None
                            paused_at = None
                    elif phase == 1 and supervisor.status == "paused":
                        if verify_pause(during_test=True):
                            phase = 2
                    elif phase == 2 and time.monotonic() - paused_at >= 1:
                        client.pause(False)
                        paused_at = None
                        phase = 3
                    elif phase == 4 and supervisor.status == "paused":
                        if verify_pause(during_test=False):
                            phase = 5
                    elif phase == 5 and time.monotonic() - paused_at >= 1:
                        client.set_request(dict(request, max_gpus=1, max_vram_bytes_per_gpu=minimum)
                                           if resize else request)
                        client.pause(False)
                        paused_at = None
                        phase = 6
                time.sleep(0.005)
            if mode == "dynamic" and (phase != 7 or len(changes) < (6 if resize else 5)):
                raise AssertionError("reference finished without all lifecycle transitions")
            if mode == "dynamic" and benchmark and (additional_event != 4 or additional_pending is not None):
                raise AssertionError("reference did not exercise every repeated resize/pause event")
            payloads[mode] = supervisor.state.load()
            results[mode] = {"seconds": time.monotonic() - started,
                             "updates": timings, "units": unit_timings, "transitions": changes}
            # Preserve evidence even if the subsequent numerical oracle fails.
            torch.save(payloads[mode], output / f"{mode}-validation-state.pt")
            (output / "lifecycle.json").write_text(json.dumps(results, indent=2) + "\n")
            if mode == "dynamic" and benchmark:
                validate_benchmark_transitions(results)
            print(f"reference {mode}: completed in {results[mode]['seconds']:.1f}s", flush=True)
        finally:
            def stop_server():
                if mode == "unmanaged":
                    return
                stop.set()
                server.join(10)
                if server.is_alive():
                    server.terminate()
                    server.join(10)

            with ExitStack() as cleanup:
                cleanup.callback(temporary.cleanup)
                cleanup.callback(stop_server)
                cleanup.callback(client.close)
                cleanup.callback(hub.close)
                cleanup.callback(monitor.close)
                if broker is not None:
                    cleanup.callback(broker.close)
                if supervisor is not None and supervisor.state is not None:
                    cleanup.callback(supervisor.state.close)
                if supervisor is not None and supervisor.cohort is not None:
                    cleanup.callback(supervisor.cohort.abort)
    if len(modes) == 1 and fixed_reference is None:
        return
    if resize:
        # Do not hide the failed old pointwise criterion by increasing epsilon.
        # Changing rank partition changes FP32 sum order; post-Adam trajectory
        # differences are diagnostic. Gradient semantics have separate analytical
        # and frozen-update roundoff oracles. This also checks structure, finite
        # tensors, exact optimizer counters and unchanged hyperparameters.
        describe_state_differences(output, payloads=payloads)
    else:
        for key in ("model", "optimizer"):
            torch.testing.assert_close(payloads["fixed"][key], payloads["dynamic"][key], rtol=0, atol=0)
    assert payloads["fixed"].get("grad_scaler") == payloads["dynamic"].get("grad_scaler")
    assert payloads["fixed"]["progress"]["step"] == payloads["dynamic"]["progress"]["step"] == steps
    for mode in payloads:
        assert payloads[mode]["progress"]["sample_frontier"] == steps * 128
        assert payloads[mode]["progress"]["pending_test"] is None
        assert math.isfinite(payloads[mode]["progress"]["best_metric"])
    if not resize:
        assert payloads["fixed"]["progress"]["best_metric"] == payloads["dynamic"]["progress"]["best_metric"]
    results["comparison"] = {"steps": steps, "effective_batch": 128,
                             "trajectory_gate": "diagnostic only; separate gradient oracle" if resize else "byte exact",
                             "original_pointwise_rtol": 2e-3, "original_pointwise_atol": 2e-5}
    (output / "lifecycle.json").write_text(json.dumps(results, indent=2) + "\n")
