"""Default CPU supervisor entry point for cooperative Fiber training.

The manager grants metadata reservations; only gated child processes initialize
CUDA. This module does not create a second training algorithm.
"""

from dataclasses import dataclass
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import signal
import time

from .cooperative_resources import MemoryCalibration
from .training_state import config_fingerprint


@dataclass(frozen=True)
class LaunchSettings:
    socket: str
    request: dict
    calibration: MemoryCalibration
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
    unknown = set(settings) - {"socket", "request", "memory_calibration", "input_queue_depth"}
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
    if not isinstance(observation, dict) or observation.get("config_fingerprint") != config_fingerprint(raw):
        raise ValueError("memory calibration is missing or belongs to a different training config")
    if set(observation) != {"config_fingerprint", "estimate", "observations"} or not observation["observations"]:
        raise ValueError("memory calibration requires its measured observations and estimate")
    estimate = dict(observation["estimate"])
    estimate["measured_peaks"] = tuple(tuple(row) for row in estimate.get("measured_peaks", ()))
    calibration = MemoryCalibration(**estimate)
    from .memory_calibration import calibration_from_observations
    measured = calibration_from_observations(config_fingerprint(raw), observation["observations"],
                                             reserve_bytes=calibration.reserve_bytes)
    if tuple(tuple(row) for row in measured["estimate"]["measured_peaks"]) != calibration.measured_peaks:
        raise ValueError("memory estimate peaks differ from its recorded observations")

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
    if workers > positive(request.get("max_cpu_cores"), "max_cpu_cores"):
        raise ValueError("sample_pipeline_workers exceeds requested maximum CPU cores")
    gpu_count = positive(request.get("max_gpus"), "max_gpus")
    for size, _ in calibration.measured_peaks:
        if size <= calibration.max_microbatch and not any(
            row["microbatch_size"] == size and row["world_size"] >= gpu_count
            for row in observation["observations"]
        ):
            raise ValueError("memory calibration must cover the requested distributed GPU count at every usable microbatch size")
    if calibration.microbatch_for(positive(request.get("min_vram_bytes_per_gpu"), "min_vram_bytes_per_gpu")) < 1:
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
    hub = WriterHub(_make_summary_writer)

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
            )
        return FiberCohort(config_path=config_path, grant=grant, state=state, broker=broker,
            calibration=settings.calibration, queue_depth=settings.queue_depth,
            writer_factory=hub.proxy_factory, resume_checkpoint=resume, training_overrides=overrides,
            calibration_observations=training["cooperative"]["memory_calibration"]["observations"])

    client = Client(settings.socket)
    supervisor = Supervisor(client, label=str(training.get("run_name", config_path.stem)),
        request=settings.request, cohort_factory=cohort_factory, dispose_state=lambda state: state.close())
    interrupted = False

    def stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True
        supervisor.stop("interrupted")

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    last_status = None
    try:
        while not supervisor.finished:
            if supervisor.error is None:
                try:
                    hub.poll()
                except Exception as error:
                    supervisor.fail(error)
            supervisor.tick()
            status = (supervisor.status, supervisor.generation)
            if status != last_status:
                print(f"fiber cooperative: task={client.task_id} phase={status[0]} generation={status[1]}", flush=True)
                last_status = status
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
