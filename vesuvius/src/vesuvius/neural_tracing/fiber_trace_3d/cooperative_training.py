"""Default CPU supervisor entry point for cooperative Fiber training.

The manager grants metadata reservations; only gated child processes initialize
CUDA. This module does not create a second training algorithm.
"""

from dataclasses import dataclass
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from copy import deepcopy
import json
import os
import signal
import subprocess
import sys
import time

from .cooperative_resources import MemoryCalibration
from .training_state import checkpoint_sources, config_fingerprint


@dataclass(frozen=True)
class LaunchSettings:
    socket: str
    request: dict
    calibration: MemoryCalibration | None
    batch_size: int
    max_steps: int
    workers: int
    slots: int
    initial: int
    queue_depth: int


def launch_settings(raw, *, socket=None):
    """Validate without querying a GPU or opening any dataset."""
    training = raw.get("training", {})
    settings = training.get("cooperative", {})
    if not isinstance(settings, dict):
        raise ValueError("training.cooperative must be an object")
    unknown = set(settings) - {"socket", "request", "memory_calibration", "input_queue_depth", "priority",
                               "startup_calibration", "startup_inventory"}
    if unknown:
        raise ValueError(f"unknown cooperative settings: {sorted(unknown)}")
    from villa_gpu_manager.paths import default_socket_path
    if socket is None:
        socket = settings["socket"] if "socket" in settings else default_socket_path()
    if not isinstance(socket, str) or not socket:
        raise ValueError("cooperative training requires a filesystem socket path")
    request = settings.get("request")
    if not isinstance(request, dict):
        raise ValueError("cooperative training requires a complete resource request")
    observation = settings.get("memory_calibration", {})
    from .cooperative_setup import validate_calibration
    startup = settings.get("startup_calibration", False)
    calibration = None if startup else validate_calibration(raw, observation)

    def positive(value, name):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    batch = positive(training.get("optimizer_global_batch_size"), "optimizer_global_batch_size")
    max_steps = positive(training.get("max_steps"), "max_steps")
    workers = positive(training.get("sample_pipeline_workers"), "sample_pipeline_workers")
    slots = positive(training.get("sample_pipeline_slots", 512), "sample_pipeline_slots")
    initial = positive(training.get("sample_pipeline_initial_in_flight", 16), "sample_pipeline_initial_in_flight")
    queue_depth = positive(settings.get("input_queue_depth", 2), "input_queue_depth")
    if slots < batch:
        raise ValueError("sample_pipeline_slots must fit one effective batch")
    workers = min(workers, positive(request.get("max_cpu_cores"), "max_cpu_cores"))
    gpu_count = positive(request.get("max_gpus"), "max_gpus")
    for size, _ in (() if calibration is None else calibration.measured_peaks):
        if size <= calibration.max_microbatch and not any(
            row["microbatch_size"] == size and row["world_size"] >= gpu_count
            for row in observation["observations"]
        ):
            raise ValueError("memory calibration must cover the requested distributed GPU count at every usable microbatch size")
    if calibration is not None and calibration.microbatch_for(positive(request.get("min_vram_bytes_per_gpu"), "min_vram_bytes_per_gpu")) < 1:
        raise ValueError("minimum VRAM request cannot fit a measured microbatch")
    if training.get("sync_batchnorm", False) or training.get("ddp_broadcast_buffers", False):
        raise ValueError("cooperative training requires sync_batchnorm=false and ddp_broadcast_buffers=false")
    if training.get("profile_replay_batch") or training.get("profile_replay_cpu_batch"):
        raise ValueError("cooperative training does not accept synthetic replay input")
    if str(training.get("device", "auto")).split(":")[0] not in ("auto", "cuda"):
        raise ValueError("cooperative training requires CUDA, not a CPU training config")
    return LaunchSettings(str(Path(socket).expanduser()), dict(request), calibration,
                          batch, max_steps, workers, slots, min(initial, slots), queue_depth)


def run_cooperative_training(config_path, *, socket=None, resume_checkpoint=None,
                             test_hang_diagnostics=False):
    import torch
    from villa_gpu_manager.client import Client
    from villa_gpu_manager.supervisor import Supervisor
    from villa_gpu_manager.stop_guard import StopGuard
    from .cooperative_cohort import FiberCohort
    from .cooperative_logging import WriterHub
    from .cooperative_maintenance import MaintenanceMailbox
    from .sample_broker import CanonicalSampleBroker
    from .loader import load_config
    from .train import _load_raw_config, _make_summary_writer, _training_cuda_value_augmentation

    if torch.cuda.is_initialized():
        raise RuntimeError("cooperative supervisor must start before CUDA initialization")
    config_path = Path(config_path).resolve()
    raw = _load_raw_config(config_path)
    original_raw = deepcopy(raw)
    from .cooperative_setup import prepare_settings
    raw = prepare_settings(config_path, raw, socket=socket)
    settings = launch_settings(raw, socket=socket)
    fingerprint = config_fingerprint(raw)
    training = raw.get("training", {})
    resume, _initialize_from = checkpoint_sources(raw, resume_checkpoint)
    start_step = 0
    if resume is not None:
        # Same trusted checkpoint assumption as the ordinary trainer. Do not
        # construct a model/optimizer (or CUDA context) in the CPU supervisor.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        start_step = int(checkpoint["step"])
        del checkpoint
    if not 0 <= start_step < settings.max_steps:
        raise ValueError("resume step must precede max_steps")
    overrides = {
        "bounded_gradient_accumulation": True, "sync_batchnorm": False,
        "ddp_broadcast_buffers": False,
        "device": "cuda:0",
        "run_datestr": str(training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")),
        "test_hang_diagnostics_enabled": bool(test_hang_diagnostics or training.get("test_hang_diagnostics_enabled", False)),
    }
    from .train import _resolve_run_layout
    from .reviewed_trace_selection import reconcile_completed_attempts
    run_dir, snapshot_dir = _resolve_run_layout(
        raw, date_str_override=overrides["run_datestr"]
    )
    reconciled = reconcile_completed_attempts(run_dir, snapshot_dir)
    for path in reconciled:
        print(f"fiber cooperative: reconciled completed reviewed trace {path}", flush=True)
    broker = None
    startup = None
    startup_complete = False
    startup_run_done = False
    if settings.calibration is None:
        from .startup_calibration import StartupMemoryCalibration
        target = min(int(raw.get("batch_size", training.get("batch_size", 16))),
                     max(1, settings.batch_size // settings.request["max_gpus"]))
        group_size = 2 if raw.get("embedding", {}).get("paired_patches", False) else 1
        target = max(group_size, target // group_size * group_size)
        startup = StartupMemoryCalibration(fingerprint, target_microbatch=target, sample_group_size=group_size)
        overrides["evaluation_microbatch_size"] = group_size
    elif any(row.get("evaluation_microbatch_size") == 1
             for row in training["cooperative"]["memory_calibration"]["observations"]):
        overrides["evaluation_microbatch_size"] = 1
    elif raw.get("embedding", {}).get("paired_patches", False):
        overrides["evaluation_microbatch_size"] = 2
    hub = WriterHub(_make_summary_writer)
    maintenance_mailbox = MaintenanceMailbox()

    def observe_startup(rows, progress):
        nonlocal startup_complete, startup_run_done
        startup.observe(rows, progress)
        if not startup.ready:
            return
        measured_size = startup.microbatch_size
        next_size = startup.complete_size(supervisor.grant)
        print(f"fiber cooperative: measured real startup microbatch={measured_size} "
              f"step={progress['step']} next_microbatch={next_size}", flush=True)
        if next_size is not None and not progress["done"]:
            supervisor.restart_cohort()
        else:
            from .memory_calibration import calibration_report_path
            report = startup.report()
            target = calibration_report_path(config_path, original_raw)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(target)
            print(f"fiber cooperative: measured memory profile saved={target}", flush=True)
            startup_complete = True
            startup_run_done = bool(progress["done"])
            supervisor.stop("startup calibrated")

    def cohort_factory(grant, state):
        nonlocal broker
        if config_fingerprint(_load_raw_config(config_path)) != fingerprint:
            raise ValueError("training config changed while the cooperative task was running")
        if broker is None:
            broker = CanonicalSampleBroker.create(
                config_path=config_path, loader_config=load_config(config_path),
                start_step=start_step, max_steps=settings.max_steps,
                effective_batch_size=settings.batch_size, workers=settings.workers,
                slot_count=settings.slots, initial_in_flight=settings.initial,
                sample_index_limit=training.get("max_sample_index", 0),
                defer_cuda_value_augmentation=_training_cuda_value_augmentation(training),
                training_overrides=overrides,
                on_process_start=stop_guard.register,
            )
        cohort = FiberCohort(config_path=config_path, grant=grant, state=state, broker=broker,
            calibration=settings.calibration, queue_depth=settings.queue_depth,
            writer_factory=hub.proxy_factory, resume_checkpoint=resume, training_overrides=overrides,
            calibration_observations=training["cooperative"].get("memory_calibration", {}).get("observations", ()),
            startup_microbatch_size=startup.microbatch_size if startup is not None else None,
            startup_reserve_bytes=startup.reserve_bytes if startup is not None else 4 * 1024**3,
            on_startup_observation=observe_startup if startup is not None else None,
            on_process_start=stop_guard.register,
            maintenance_mailbox=maintenance_mailbox)
        return cohort

    client = Client(settings.socket)
    supervisor = Supervisor(client, label=str(training.get("run_name", config_path.stem)),
        request=settings.request, cohort_factory=cohort_factory, dispose_state=lambda state: state.close(),
        pausable=startup is None, adaptable=startup is None,
        disconnect_grace_seconds=60 if startup is None else None)
    interrupted = False
    stop_guard = StopGuard()

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        stop_guard.stop()
        supervisor.stop("interrupted")

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_status = None
    maintenance_process = None
    maintenance_log = None
    maintenance_group = None
    maintenance_owner = None
    maintenance_token = None
    maintenance_attempt = None
    maintenance_attempt_index = 0

    def grant_key(grant):
        return (
            int(grant["cpu_cores"]),
            tuple(sorted((str(gpu), int(vram)) for gpu, vram in grant["gpus"].items())),
        )

    def applied_ownership_token():
        task = client.task
        if (not supervisor.connected or supervisor.cohort is None
                or supervisor.status != "running" or not isinstance(task, dict)
                or task.get("phase") != "applied"):
            return None
        return client.epoch, supervisor.generation, grant_key(supervisor.grant)

    def owner_is_current():
        return (
            maintenance_owner is not None
            and supervisor.cohort is maintenance_owner
            and maintenance_owner.maintenance_ready
            and maintenance_token == (
                client.epoch, supervisor.generation, grant_key(supervisor.grant)
            )
        )

    def start_reviewed_trace(request, token):
        nonlocal maintenance_process, maintenance_log, maintenance_group
        nonlocal maintenance_owner, maintenance_token, maintenance_attempt
        nonlocal maintenance_attempt_index
        from .train import _reviewed_trace_test_config
        from .reviewed_trace_selection import freeze_raw_checkpoint
        from villa_gpu_manager.exclusive import OwnedProcessGroup
        from villa_gpu_manager.processes import process_identity
        trace = _reviewed_trace_test_config(training)
        if trace is None:
            raise RuntimeError("reviewed trace boundary has no configured test")
        step = int(request["step"])
        checkpoint = snapshot_dir / "current.pt"
        if not checkpoint.is_file():
            checkpoint = snapshot_dir / "best.pt"
        if not checkpoint.is_file():
            raise RuntimeError(f"reviewed trace checkpoint is missing: {checkpoint}")
        root = run_dir / "reviewed_trace_test"
        output = root / f"step_{step:08d}"
        if output.exists():
            raise RuntimeError(f"reviewed trace result already exists: {output}")
        maintenance_attempt_index += 1
        attempt = root / f".step_{step:08d}.attempt-{os.getpid()}-{maintenance_attempt_index}"
        attempt.mkdir(parents=True, exist_ok=False)
        frozen = attempt / "evaluated_raw.pt"
        checkpoint_sha256 = freeze_raw_checkpoint(
            checkpoint, frozen, step=step,
            samples_seen=int(request["samples_seen"]),
        )
        job = {
            "step": step,
            "samples_seen": int(request["samples_seen"]),
            "checkpoint": str(frozen),
            "checkpoint_sha256": checkpoint_sha256,
            "output_root": str(attempt),
            "template": str(Path(trace["template"]).expanduser().resolve()),
            "split_root": str(Path(trace["split_root"]).expanduser().resolve()),
            "native_binary": str(Path(trace["native_binary"]).expanduser().resolve()),
            "runner": str(Path(trace["runner"]).expanduser().resolve()),
            "devices": ",".join(str(index) for index in range(len(supervisor.grant["gpus"]))),
            "gpu_grant": dict(sorted(supervisor.grant["gpus"].items())),
            # This process remains inside the existing manager task's process
            # tree and uses its unchanged grant. It must not register a second
            # overlapping task.
            "inherited_manager_grant": True,
        }
        job_path = attempt / "job.json"
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        environment = dict(os.environ)
        for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(supervisor.grant["gpus"]))
        maintenance_log = (attempt / "runner.log").open("w")
        maintenance_mailbox.begin_attempt(request, token)
        try:
            maintenance_process = subprocess.Popen(
                [sys.executable, "-m",
                 "vesuvius.neural_tracing.fiber_trace_3d.training_trace_benchmark",
                 str(job_path)],
                cwd=Path(__file__).resolve().parents[5], env=environment,
                stdout=maintenance_log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            maintenance_group = OwnedProcessGroup(process_identity(maintenance_process.pid))
        except BaseException:
            if maintenance_process is not None and maintenance_process.poll() is None:
                try:
                    os.killpg(maintenance_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                maintenance_process.wait(timeout=5)
            (attempt / "evaluated_raw.pt").unlink(missing_ok=True)
            maintenance_mailbox.end_attempt(token)
            maintenance_log.close()
            maintenance_log = None
            maintenance_process = None
            maintenance_group = None
            raise
        maintenance_owner = supervisor.cohort
        maintenance_token = token
        maintenance_attempt = attempt
        stop_guard.register([maintenance_process.pid])
        print(
            f"fiber cooperative: training model unloaded; reviewed trace started "
            f"step={step} pid={maintenance_process.pid} output={attempt} "
            f"generation={token[1]} gpus={','.join(sorted(supervisor.grant['gpus']))}",
            flush=True,
        )

    def finish_reviewed_trace(request, returncode):
        nonlocal maintenance_process, maintenance_log, maintenance_group
        nonlocal maintenance_attempt
        if maintenance_log is not None:
            maintenance_log.flush()
            maintenance_log.close()
            maintenance_log = None
        output = maintenance_attempt
        if output is None:
            raise RuntimeError("reviewed trace attempt path is missing")
        if returncode == 0:
            from .reviewed_trace_selection import finalize_attempt
            canonical = output.parent / f"step_{int(request['step']):08d}"
            metrics, selection, output = finalize_attempt(
                output, snapshot_dir, canonical=canonical
            )
            if hub.writer is not None:
                for key in (
                    "restarts_forward", "restarts_reverse", "restarts_combined",
                    "reference_mm_forward", "reference_mm_reverse", "reference_mm_combined",
                    "implied_mean_mm_forward", "implied_mean_mm_reverse",
                    "implied_mean_mm_combined", "wall_seconds",
                ):
                    hub.writer.add_scalar(
                        f"reviewed_trace/{key}", metrics[key], int(request["samples_seen"])
                    )
                hub.writer.flush()
            print(
                "fiber cooperative: reviewed trace complete "
                f"step={request['step']} combined_restarts={metrics['restarts_combined']} "
                f"combined_implied_mm={metrics['implied_mean_mm_combined']:.6f} "
                f"best_reviewed_trace={selection['best_restarts_combined']} "
                f"selected={selection['selected']}",
                flush=True,
            )
        else:
            tail = (output / "runner.log").read_text(errors="replace")[-8000:]
            raise RuntimeError(
                f"reviewed trace failed step={request['step']} exit={returncode}\n{tail}"
            )
        maintenance_process = None
        maintenance_group = None
        maintenance_attempt = None
        maintenance_mailbox.end_attempt(maintenance_token)
        return metrics

    def cancel_reviewed_trace(reason):
        nonlocal maintenance_process, maintenance_log, maintenance_group
        nonlocal maintenance_owner, maintenance_token, maintenance_attempt
        if maintenance_process is None:
            return
        attempt = maintenance_attempt
        if maintenance_process.poll() is None:
            maintenance_group.close(reap=maintenance_process.poll)
        maintenance_process.wait(timeout=1)
        if maintenance_log is not None:
            maintenance_log.flush()
            maintenance_log.close()
            maintenance_log = None
        if attempt is not None and attempt.exists():
            workload_path = attempt / "workload.json"
            if workload_path.is_file():
                from .training_trace_benchmark import cleanup_prediction_cache
                cleanup_prediction_cache(json.loads(workload_path.read_text()), attempt)
            (attempt / "evaluated_raw.pt").unlink(missing_ok=True)
            aborted = attempt.with_name(attempt.name + f".revoked-{time.time_ns()}")
            attempt.rename(aborted)
        print(
            f"fiber cooperative: reviewed trace deferred reason={reason}; "
            "child process group reaped and request remains pending",
            flush=True,
        )
        maintenance_process = None
        maintenance_group = None
        maintenance_mailbox.end_attempt(maintenance_token)
        maintenance_owner = None
        maintenance_token = None
        maintenance_attempt = None
    try:
        while True:
            stop_guard.check()
            if supervisor.error is None:
                try:
                    hub.poll()
                except Exception as error:
                    supervisor.fail(error)
            supervisor.tick()
            request = maintenance_mailbox.request
            if maintenance_process is not None:
                returncode = maintenance_process.poll()
                ownership_current = owner_is_current()
                if returncode is not None and returncode == 0:
                    old = maintenance_owner
                    finish_reviewed_trace(request, returncode)
                    state = old.maintenance_state() if ownership_current else None
                    maintenance_mailbox.complete(request)
                    maintenance_owner = None
                    maintenance_token = None
                    if ownership_current:
                        if supervisor.state is not None and supervisor.state is not state:
                            supervisor.dispose_state(supervisor.state)
                        supervisor.state = state
                        supervisor.cohort = cohort_factory(supervisor.grant, state)
                        print(
                            "fiber cooperative: reviewed trace workers exited; "
                            "reinitializing training under retained allocation",
                            flush=True,
                        )
                    else:
                        print(
                            "fiber cooperative: reviewed trace completed while allocation changed; "
                            "normal supervisor reconstruction will resume training",
                            flush=True,
                        )
                elif not ownership_current:
                    cancel_reviewed_trace("manager allocation changed")
                elif returncode is not None:
                    finish_reviewed_trace(request, returncode)
            request = maintenance_mailbox.request
            if (request is not None and maintenance_process is None
                    and supervisor.cohort is not None):
                if supervisor.cohort.begin_maintenance():
                    print(
                        f"fiber cooperative: reviewed trace boundary step={request['step']}; "
                        "unloading training model while retaining allocation",
                        flush=True,
                    )
                token = applied_ownership_token()
                if supervisor.cohort.maintenance_ready and token is not None:
                    start_reviewed_trace(request, token)
            status = (supervisor.status, supervisor.generation)
            if status != last_status:
                reason = (client.task or {}).get("reason")
                print(f"fiber cooperative: task={client.task_id} phase={status[0]} generation={status[1]} "
                      f"allocation={supervisor.grant} reason={reason}", flush=True)
                last_status = status
            if supervisor.finished:
                if startup_complete and not startup_run_done and not interrupted and supervisor.error is None:
                    if supervisor.cohort is not None:
                        raise RuntimeError("startup workers must join before adaptive registration")
                    # Protocol v1 capabilities are immutable. Retain this real
                    # run's RAM state and broker, finish the exclusive task,
                    # then register adaptive ownership as the same training run.
                    state = supervisor.state
                    next_raw = deepcopy(original_raw)
                    next_raw.setdefault("training", {}).setdefault("cooperative", {})["memory_calibration"] = startup.report()
                    raw = prepare_settings(config_path, next_raw, socket=socket)
                    settings = launch_settings(raw, socket=socket)
                    training = raw["training"]
                    startup = None
                    startup_complete = False
                    client = Client(settings.socket)
                    next_supervisor = Supervisor(client, label=str(training.get("run_name", config_path.stem)),
                        request=settings.request, cohort_factory=cohort_factory, state=state,
                        dispose_state=lambda value: value.close(), disconnect_grace_seconds=60)
                    supervisor.state = None
                    supervisor = next_supervisor
                    last_status = None
                    print("fiber cooperative: startup workers joined; registering adaptive continuation of the same run", flush=True)
                else:
                    break
            time.sleep(0.005)
        if supervisor.error is not None:
            raise RuntimeError("cooperative training failed") from supervisor.error
        if interrupted:
            raise KeyboardInterrupt
    finally:
        # Fatal parent-side errors must not strand owned compute children. No
        # release is claimed until actual joins have succeeded.
        # Execute every cleanup even if a writer/queue close raises. Register
        # in reverse order: workers must join before their CPU resources close.
        if not stop_guard.stopping:
            stop_guard.stop()
        if maintenance_process is not None:
            cancel_reviewed_trace("training supervisor stopping")
        with ExitStack() as cleanup:
            if maintenance_log is not None:
                cleanup.callback(maintenance_log.close)
            if maintenance_process is not None and maintenance_process.poll() is None:
                cleanup.callback(maintenance_process.wait)
            cleanup.callback(stop_guard.close)
            for sig, handler in previous.items():
                cleanup.callback(signal.signal, sig, handler)
            if supervisor.state is not None:
                cleanup.callback(supervisor.state.close)
            cleanup.callback(client.close)
            cleanup.callback(hub.close)
            if broker is not None:
                cleanup.callback(broker.close)
            if supervisor.cohort is not None:
                cleanup.callback(supervisor.cohort.abort)
