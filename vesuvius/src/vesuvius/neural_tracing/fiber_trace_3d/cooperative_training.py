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
import time

from .cooperative_resources import MemoryCalibration
from .training_state import config_fingerprint


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
    from .cooperative_cohort import FiberCohort
    from .cooperative_logging import WriterHub
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
    resume = resume_checkpoint or training.get("resume") or raw.get("resume")
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
    broker = None
    startup = None
    startup_complete = False
    startup_run_done = False
    if settings.calibration is None:
        from .startup_calibration import StartupMemoryCalibration
        target = min(int(raw.get("batch_size", training.get("batch_size", 16))),
                     max(1, settings.batch_size // settings.request["max_gpus"]))
        startup = StartupMemoryCalibration(fingerprint, target_microbatch=target)
        overrides["evaluation_microbatch_size"] = 1
    elif any(row.get("evaluation_microbatch_size") == 1
             for row in training["cooperative"]["memory_calibration"]["observations"]):
        overrides["evaluation_microbatch_size"] = 1
    hub = WriterHub(_make_summary_writer)

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
            )
        return FiberCohort(config_path=config_path, grant=grant, state=state, broker=broker,
            calibration=settings.calibration, queue_depth=settings.queue_depth,
            writer_factory=hub.proxy_factory, resume_checkpoint=resume, training_overrides=overrides,
            calibration_observations=training["cooperative"].get("memory_calibration", {}).get("observations", ()),
            startup_microbatch_size=startup.microbatch_size if startup is not None else None,
            startup_reserve_bytes=startup.reserve_bytes if startup is not None else 4 * 1024**3,
            on_startup_observation=observe_startup if startup is not None else None)

    client = Client(settings.socket)
    supervisor = Supervisor(client, label=str(training.get("run_name", config_path.stem)),
        request=settings.request, cohort_factory=cohort_factory, dispose_state=lambda state: state.close(),
        pausable=startup is None, adaptable=startup is None,
        disconnect_grace_seconds=60 if startup is None else None)
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        supervisor.stop("interrupted")

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_status = None
    try:
        while True:
            if supervisor.error is None:
                try:
                    hub.poll()
                except Exception as error:
                    supervisor.fail(error)
            supervisor.tick()
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
        with ExitStack() as cleanup:
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
