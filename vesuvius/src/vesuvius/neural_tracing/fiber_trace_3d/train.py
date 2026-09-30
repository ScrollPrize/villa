from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import signal
import socket
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from vesuvius.neural_tracing.fiber_trace_3d.loss_statistics import LossStatistics as _LossStatistics
from vesuvius.neural_tracing.fiber_trace_3d.embedding import embedding_statistics
from vesuvius.neural_tracing.fiber_trace_3d.ema import ModelEMA
from vesuvius.neural_tracing.fiber_trace_3d.historical_evaluation import (
    historical_test_config, historical_presence_mask,
)
from vesuvius.neural_tracing.fiber_trace_3d.gradient_accumulation import (
    loss_components,
    NormalizedGradientAccumulator,
    validate_target_only_loss_config, validate_sample_independent_model,
)
from vesuvius.neural_tracing.fiber_trace_3d.training_state import (
    TrainingBoundary, capture_handoff, restore_handoff,
    model_optimizer_payload, restore_model_optimizer,
    checkpoint_sources as _checkpoint_sources, config_fingerprint,
)
from vesuvius.neural_tracing.fiber_trace_3d.cooperative_resources import EffectiveBatchSampleIndex
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from vesuvius.neural_tracing.fiber_trace_3d.training_profile import (
    TrainingProfile, compile_training_model, loader_profile_enabled,
    replay_batch_enabled, replay_cpu_batch_enabled,
)

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    DEFAULT_VOLUME_CACHE_MEMORY_MIB,
    FiberTrace3DBatch,
    FiberTrace3DLoader,
    _normalize_image,
    _read_raw_block,
    config_from_mapping,
    load_config,
    collate_fiber_trace_3d_samples,
)
from vesuvius.neural_tracing.fiber_trace_3d.sample_pipeline import (
    MacroBatchTargetSource,
    FiberTrace3DCudaBatchPipeline,
    FiberTrace3DTrainingBatchPipeline,
    VirtualBatchSampleIndex,
    EvaluationSampleIndex,
    FiberTrace3DSampleQueue,
    SampleQueueSource,
    apply_deferred_batch_augmentation,
)
from vesuvius.neural_tracing.fiber_trace_3d.queued_batches import (
    QueuedBatchView, TrainingBatchIndices, test_batches as queued_test_batches,
)
from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
    encode_lasagna_direction_3x2,
)
from .refinement import RefinementConfig, RefinementObserver, refinement_session, active_store, log_refinement_scalars, empty_refinement_statistics
from vesuvius.neural_tracing.fiber_trace_3d.model import (
    build_fiber_trace_3d_model,
    direction_output,
    direction_outputs,
    presence_output,
    presence_outputs,
)
from vesuvius.neural_tracing.fiber_trace_3d.targets import (
    materialize_targets,
    require_materialized_targets,
)
from vesuvius.neural_tracing.fiber_trace_3d.trace2cp_bridge import (
    Trace2Cp3DProjectedFields,
    project_3d_output_to_trace2cp_fields,
    score_trace2cp_projected_fields,
)


@dataclass(frozen=True)
class _Trace2Cp3DConfig:
    enabled: bool
    control_points: int
    start_sample_index: int
    sample_mode: str
    step_px: float
    rf_margin_px: float
    presence_enabled: bool
    patch_shape_hw: tuple[int, int]
    strip_z_offset_count: int
    strip_z_offset_step: float
    tile_shape_hw: tuple[int, int]
    block_context_voxels: int
    loader_config_path: Path | None


@dataclass(frozen=True)
class _Trace2Cp3DMetricEvalResult:
    error_mean: float
    raw_y_error_mean_px: float
    segments: int
    skipped_segments: int
    first_skip_reason: str


@dataclass(frozen=True)
class _MixedPrecisionConfig:
    mode: str
    enabled: bool
    device_type: str
    dtype: torch.dtype | None
    use_grad_scaler: bool


@dataclass(frozen=True)
class _LrRangeTestConfig:
    warmup_steps: int
    range_steps: int
    start_lr: float
    end_lr: float


@dataclass(frozen=True)
class _DistributedConfig:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    backend: str
    device: torch.device


class _TrainingHangDiagnostics:
    """Best-effort persistent diagnostics for long rank-local stalls."""

    def __init__(
        self,
        run_dir: Path,
        *,
        rank: int,
        device: torch.device,
        watchdog_seconds: float = 480.0,
        automatic_watchdog: bool = False,
        enabled: bool = True,
    ) -> None:
        self.rank = int(rank)
        self.device = device
        self.watchdog_seconds = float(watchdog_seconds)
        self.automatic_watchdog = bool(automatic_watchdog)
        self.enabled = bool(enabled)
        self.path = Path(run_dir) / f"hang_diagnostics_rank_{self.rank}.log"
        self._file: Any | None = None
        self._watchdog_armed = False
        self._signal_registered = False
        self._closed = False
        if not self.enabled:
            return
        try:
            self._file = self.path.open("a", encoding="utf-8", buffering=1)
            self.marker(
                "diagnostics-start",
                pid=os.getpid(),
                host=socket.gethostname(),
                watchdog_seconds=self.watchdog_seconds,
                automatic=int(self.automatic_watchdog),
            )
            dump_signal = getattr(signal, "SIGUSR2", None)
            if dump_signal is not None:
                try:
                    faulthandler.register(
                        dump_signal,
                        file=self._file,
                        all_threads=True,
                        chain=False,
                    )
                    self._signal_registered = True
                    self.marker("manual-dump-registered", signal="SIGUSR2")
                except (OSError, RuntimeError, ValueError):
                    self.marker("manual-dump-unavailable", signal="SIGUSR2")
        except (OSError, RuntimeError, ValueError):
            self._file = None

    def _resource_fields(self) -> dict[str, Any]:
        if not self.enabled:
            return {}
        fields: dict[str, Any] = {
            "rss_mib": "unavailable",
            "threads": "unavailable",
            "fds": "unavailable",
        }
        try:
            status: dict[str, str] = {}
            with Path("/proc/self/status").open("r", encoding="utf-8") as handle:
                for line in handle:
                    key, separator, value = line.partition(":")
                    if separator:
                        status[key] = value.strip()
            rss_tokens = status.get("VmRSS", "").split()
            if rss_tokens:
                fields["rss_mib"] = f"{int(rss_tokens[0]) / 1024.0:.1f}"
            if status.get("Threads"):
                fields["threads"] = int(status["Threads"])
        except (OSError, ValueError):
            pass
        try:
            fields["fds"] = sum(1 for _ in Path("/proc/self/fd").iterdir())
        except OSError:
            pass
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                fields["cuda_allocated_mib"] = (
                    f"{torch.cuda.memory_allocated(self.device) / (1024.0 ** 2):.1f}"
                )
                fields["cuda_reserved_mib"] = (
                    f"{torch.cuda.memory_reserved(self.device) / (1024.0 ** 2):.1f}"
                )
            except (RuntimeError, ValueError):
                fields["cuda_allocated_mib"] = "unavailable"
                fields["cuda_reserved_mib"] = "unavailable"
        return fields

    def marker(self, phase: str, **fields: Any) -> None:
        if self._file is None or self._closed:
            return
        try:
            record = {
                "time_utc": datetime.now().astimezone().isoformat(),
                "monotonic": f"{time.monotonic():.6f}",
                "rank": self.rank,
                "phase": str(phase),
                **self._resource_fields(),
                **fields,
            }
            self._file.write(" ".join(f"{key}={value}" for key, value in record.items()) + "\n")
            self._file.flush()
        except (OSError, RuntimeError, ValueError):
            pass

    def arm_test_watchdog(self, step: int) -> None:
        if (
            not self.automatic_watchdog
            or self._file is None
            or self._closed
            or self.watchdog_seconds <= 0.0
        ):
            return
        self.cancel_test_watchdog(step=step, reason="rearm")
        try:
            faulthandler.dump_traceback_later(
                self.watchdog_seconds,
                repeat=False,
                file=self._file,
                exit=False,
            )
            self._watchdog_armed = True
            self.marker("test-watchdog-arm", step=int(step), timeout_s=self.watchdog_seconds)
        except (OSError, RuntimeError, ValueError):
            self.marker("test-watchdog-unavailable", step=int(step))

    def cancel_test_watchdog(self, *, step: int | None = None, reason: str = "complete") -> None:
        if not self._watchdog_armed:
            return
        try:
            faulthandler.cancel_dump_traceback_later()
        except (OSError, RuntimeError, ValueError):
            pass
        self._watchdog_armed = False
        self.marker("test-watchdog-cancel", step=step, reason=reason)

    def close(self) -> None:
        if self._closed:
            return
        self.cancel_test_watchdog(reason="close")
        if self._signal_registered:
            dump_signal = getattr(signal, "SIGUSR2", None)
            if dump_signal is not None:
                try:
                    faulthandler.unregister(dump_signal)
                except (OSError, RuntimeError, ValueError):
                    pass
            self._signal_registered = False
        self.marker("diagnostics-close")
        self._closed = True
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None


def _diagnostic_cuda_synchronize(
    diagnostics: _TrainingHangDiagnostics | None,
    *,
    phase: str,
    step: int,
    device: torch.device,
    **fields: Any,
) -> None:
    if diagnostics is None or not diagnostics.enabled or device.type != "cuda":
        return
    diagnostics.marker(f"{phase}-begin", step=step, **fields)
    torch.cuda.synchronize(device)
    diagnostics.marker(f"{phase}-end", step=step, **fields)


def _resolve_test_hang_diagnostics_enabled(
    training: dict[str, Any],
    *,
    cli_enabled: bool,
) -> tuple[bool, bool, bool]:
    config_value = training.get("test_hang_diagnostics_enabled", False)
    if not isinstance(config_value, bool):
        raise ValueError("training.test_hang_diagnostics_enabled must be a JSON boolean")
    cli_value = bool(cli_enabled)
    return bool(config_value), cli_value, bool(config_value or cli_value)


def _resolve_test_watchdog_seconds(
    training: dict[str, Any],
    *,
    diagnostics_enabled: bool,
) -> float:
    default_seconds = 480.0
    if not diagnostics_enabled:
        return default_seconds
    seconds = float(training.get("test_watchdog_seconds", default_seconds))
    if seconds <= 0.0 or seconds >= 600.0:
        raise ValueError("training.test_watchdog_seconds must be > 0 and < 600")
    return seconds


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    config.setdefault("_config_dir", str(config_path.parent))
    return config


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sanitize_run_name(value: Any) -> str:
    name = str(value or "fiber_trace_3d").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("._-") or "fiber_trace_3d"


def _resolve_run_layout(
    config: dict[str, Any],
    *,
    date_str_override: str | None = None,
) -> tuple[Path, Path]:
    training = dict(config.get("training", {}))
    run_path = Path(str(training.get("run_path", config.get("run_path", "runs/fiber_trace_3d"))))
    run_name = _sanitize_run_name(training.get("run_name", config.get("run_name", "fiber_trace_3d")))
    date_str = str(
        date_str_override
        if date_str_override is not None
        else training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_dir = run_path / f"{run_name}_{date_str}"
    snapshot_dir = run_dir / "snapshots"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, snapshot_dir


def _make_summary_writer(log_dir: Path, *, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires tensorboard; install it or set "
            "training.tensorboard_enabled=false"
        ) from exc
    return SummaryWriter(log_dir=str(log_dir))


def _device_from_training(training: dict[str, Any]) -> torch.device:
    raw = str(training.get("device", "auto"))
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def _env_int(env: dict[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _distributed_config_from_env(
    base_device: torch.device,
    *,
    env: dict[str, str] | None = None,
) -> _DistributedConfig:
    env_map = os.environ if env is None else env
    world_size = _env_int(env_map, "WORLD_SIZE")
    if world_size is None:
        return _DistributedConfig(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            backend="",
            device=base_device,
        )
    if int(world_size) <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if int(world_size) == 1:
        return _DistributedConfig(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main=True,
            backend="",
            device=base_device,
        )
    rank = _env_int(env_map, "RANK")
    local_rank = _env_int(env_map, "LOCAL_RANK")
    if rank is None or local_rank is None:
        raise ValueError(
            "DDP launch requires WORLD_SIZE, RANK, and LOCAL_RANK. "
            "Use torchrun for multi-process training."
        )
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")
    if int(rank) < 0 or int(rank) >= int(world_size):
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got rank={rank} world_size={world_size}")
    if int(local_rank) < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")

    if base_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP CUDA launch requested but CUDA is not available")
        device_count = int(torch.cuda.device_count())
        if device_count > 0 and int(local_rank) >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count {device_count}"
            )
        device = torch.device("cuda", int(local_rank))
        backend = "nccl"
    elif base_device.type == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError(f"DDP supports only cpu/cuda training devices, got {base_device}")
    return _DistributedConfig(
        enabled=True,
        rank=int(rank),
        local_rank=int(local_rank),
        world_size=int(world_size),
        is_main=int(rank) == 0,
        backend=backend,
        device=device,
    )


def _torchrun_world_size_from_env(env: dict[str, str] | None = None) -> int:
    env_map = os.environ if env is None else env
    world_size = _env_int(env_map, "WORLD_SIZE")
    return 1 if world_size is None else int(world_size)


def _require_single_process_cli_mode(mode_name: str) -> None:
    if _torchrun_world_size_from_env() > 1:
        raise SystemExit(
            f"{mode_name} is single-process only. Launch without torchrun/WORLD_SIZE>1; "
            "normal training's manager supervisor launches GPU workers. "
            "Unmanaged distributed testing requires the explicit --unmanaged flag."
        )


def _distributed_init(config: _DistributedConfig, *, init_method: str | None = None) -> None:
    if not config.enabled:
        return
    if config.device.type == "cuda":
        torch.cuda.set_device(config.local_rank)
    if not dist.is_initialized():
        options = {} if init_method is None else {
            "init_method": init_method, "rank": config.rank, "world_size": config.world_size,
        }
        dist.init_process_group(backend=config.backend, **options)


def _distributed_barrier(config: _DistributedConfig) -> None:
    if config.enabled and dist.is_initialized():
        dist.barrier()


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _distributed_max_float(value: float, config: _DistributedConfig) -> float:
    if not config.enabled:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=config.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _shutdown_dataloader_iterator(iterator: Any | None) -> None:
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
        return
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


def _replayed_training_batch(
    batch: FiberTrace3DBatch,
) -> tuple[FiberTrace3DBatch, float, float, float, float]:
    return batch, 0.0, 0.0, 0.0, 0.0


def _replayed_cpu_training_batch(
    batch: FiberTrace3DBatch,
    *,
    loader: FiberTrace3DLoader,
    device: torch.device,
    output_stride: int,
) -> tuple[FiberTrace3DBatch, float, float, float, float]:
    to_device_start = time.perf_counter()
    prepared = batch.to(device)
    _synchronize_device(device)
    to_device_ms = (time.perf_counter() - to_device_start) * 1000.0
    target_start = time.perf_counter()
    prepared = materialize_targets(
        prepared,
        loader.config,
        profile=True,
        output_stride=output_stride,
    )
    _synchronize_device(device)
    target_ms = (time.perf_counter() - target_start) * 1000.0
    return prepared, to_device_ms + target_ms, 0.0, to_device_ms, target_ms


def _batch_tensor_sha256(batch: FiberTrace3DBatch) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(vars(batch).items()):
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


_MATERIALIZED_TARGET_TENSOR_FIELDS = frozenset(
    {
        "direction_target",
        "direction_weight",
        "direction_mask",
        "direction_indices_bzyx",
        "direction_target_sparse",
        "direction_weight_sparse",
        "direction_tangent_sparse_zyx",
        "presence_target",
        "presence_mask",
    }
)


def _source_batch_tensor_sha256(batch: FiberTrace3DBatch) -> str:
    """Hash captured source tensors independently of output-lattice targets."""

    digest = hashlib.sha256()
    for name, value in sorted(vars(batch).items()):
        if name in _MATERIALIZED_TARGET_TENSOR_FIELDS or not isinstance(value, torch.Tensor):
            continue
        tensor = value.detach().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _normalization_module_counts(model: torch.nn.Module) -> dict[str, int]:
    classes = (torch.nn.BatchNorm3d, torch.nn.SyncBatchNorm, torch.nn.GroupNorm)
    return {
        cls.__name__: sum(isinstance(module, cls) for module in model.modules())
        for cls in classes
    }


def _validate_profile_normalization_modules(
    requested: str,
    *,
    sync_batchnorm: bool,
    counts: dict[str, int],
) -> None:
    requested = str(requested).lower().replace("_", "")
    batch = int(counts.get("BatchNorm3d", 0))
    sync = int(counts.get("SyncBatchNorm", 0))
    group = int(counts.get("GroupNorm", 0))
    if requested in {"batch", "batchnorm"}:
        valid = sync > 0 and batch == 0 and group == 0 if sync_batchnorm else (
            batch > 0 and sync == 0 and group == 0
        )
    elif requested in {"group", "groupnorm"}:
        valid = group > 0 and batch == 0 and sync == 0
    else:
        return
    if not valid:
        raise RuntimeError(
            "profile normalization modules do not match requested mode: "
            f"requested={requested} sync_batchnorm={sync_batchnorm} counts={counts}"
        )


def _distributed_cleanup(config: _DistributedConfig) -> None:
    if config.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_broadcast_object(value: Any, config: _DistributedConfig, *, src: int = 0) -> Any:
    if not config.enabled:
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def _distributed_should_use_sync_batchnorm(
    config: _DistributedConfig,
    *,
    enabled: bool = True,
) -> bool:
    if not isinstance(enabled, bool):
        raise ValueError("training.sync_batchnorm must be a JSON boolean")
    return bool(enabled and config.enabled and config.device.type == "cuda")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wraps the DDP module; callers still need its raw semantics.
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if isinstance(model, (DistributedDataParallel, torch.nn.DataParallel)):
        return model.module
    return model


def _wrap_distributed_model(
    model: torch.nn.Module,
    config: _DistributedConfig,
    *,
    broadcast_buffers: bool = True,
) -> torch.nn.Module:
    if not isinstance(broadcast_buffers, bool):
        raise ValueError("training.ddp_broadcast_buffers must be a JSON boolean")
    if not config.enabled:
        return model
    if config.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            broadcast_buffers=broadcast_buffers,
        )
    return DistributedDataParallel(model, broadcast_buffers=broadcast_buffers)


def _normalize_mixed_precision_mode(raw: Any) -> str:
    if isinstance(raw, bool):
        mode = "bf16" if raw else "off"
    else:
        mode = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "none": "off",
        "false": "off",
        "0": "off",
        "no": "off",
        "float32": "off",
        "fp32": "off",
        "true": "bf16",
        "1": "bf16",
        "yes": "bf16",
        "bfloat16": "bf16",
        "amp_bf16": "bf16",
        "float16": "fp16",
        "half": "fp16",
        "amp_fp16": "fp16",
    }
    mode = aliases.get(mode, mode)
    valid_modes = {"off", "bf16", "fp16", "auto"}
    if mode not in valid_modes:
        raise ValueError(
            "training.mixed_precision must be one of off, bf16, fp16, auto; "
            f"got {raw!r}"
        )
    return mode


def _mixed_precision_config_from_training(
    training: dict[str, Any],
    device: torch.device,
) -> _MixedPrecisionConfig:
    mode = _normalize_mixed_precision_mode(training.get("mixed_precision", "off"))
    device_type = str(device.type)
    if mode == "auto":
        if device_type == "cuda":
            supports_bf16 = bool(
                getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            mode = "bf16" if supports_bf16 else "fp16"
        else:
            mode = "off"
    if mode == "off":
        return _MixedPrecisionConfig(
            mode="off",
            enabled=False,
            device_type=device_type,
            dtype=None,
            use_grad_scaler=False,
        )
    if mode == "bf16":
        if device_type not in {"cuda", "cpu"}:
            raise ValueError(
                "training.mixed_precision='bf16' is supported only on cuda or cpu devices, "
                f"got {device_type!r}"
            )
        if device_type == "cuda":
            supports_bf16 = bool(
                getattr(torch.cuda, "is_bf16_supported", lambda: False)()
            )
            if not supports_bf16:
                raise ValueError(
                    "training.mixed_precision='bf16' requested, but CUDA BF16 is not supported "
                    "on this device. Use 'fp16', 'auto', or 'off'."
                )
        return _MixedPrecisionConfig(
            mode="bf16",
            enabled=True,
            device_type=device_type,
            dtype=torch.bfloat16,
            use_grad_scaler=False,
        )
    if device_type != "cuda":
        raise ValueError(
            "training.mixed_precision='fp16' requires a CUDA device; "
            f"got {device_type!r}"
        )
    return _MixedPrecisionConfig(
        mode="fp16",
        enabled=True,
        device_type=device_type,
        dtype=torch.float16,
        use_grad_scaler=True,
    )


def _autocast_context(precision: _MixedPrecisionConfig | None):
    if precision is None or not precision.enabled:
        return nullcontext()
    assert precision.dtype is not None
    return torch.autocast(
        device_type=precision.device_type,
        dtype=precision.dtype,
        enabled=True,
    )


def _make_grad_scaler(precision: _MixedPrecisionConfig) -> torch.amp.GradScaler | None:
    if not precision.use_grad_scaler:
        return None
    return torch.amp.GradScaler("cuda", enabled=True)


def _grad_scaler_enabled(grad_scaler: torch.amp.GradScaler | None) -> bool:
    return bool(grad_scaler is not None and grad_scaler.is_enabled())


def _training_sample_index_limit(training: dict[str, Any], sample_count: int) -> int:
    limit = int(training.get("max_sample_index", 0))
    if limit < 0:
        raise ValueError("training.max_sample_index must be >= 0")
    if limit == 0:
        return 0
    if limit > int(sample_count):
        raise ValueError(
            "training.max_sample_index must be <= configured sample count "
            f"({sample_count}), got {limit}"
        )
    return limit


def _bounded_training_sample_count(training: dict[str, Any], sample_count: int) -> int:
    limit = _training_sample_index_limit(training, sample_count)
    return int(sample_count) if limit <= 0 else int(limit)


def _make_test_loader_raw_config(raw_config: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    test_raw = dict(raw_config)
    test_raw["_refinement_split"] = "test"
    test_raw["datasets"] = raw_config["test_datasets"]
    test_raw.pop("test_datasets", None)
    if "test_batch_size" in training:
        size = training["test_batch_size"]
        if type(size) is not int or size < 1:
            raise ValueError("training.test_batch_size must be a positive integer")
        test_raw["batch_size"] = size
    if "test_sampling_strategy" in training:
        test_raw["sampling_strategy"] = training["test_sampling_strategy"]
    if not bool(training.get("test_augment_enabled", False)):
        test_raw["augment_enabled"] = False
    if test_raw.get("embedding", {}).get("paired_patches", False):
        if int(test_raw.get("batch_size", 1)) % 2 or int(training.get("test_start_sample_index", 0)) % 2:
            raise ValueError("paired evaluation requires even batch size and start index")
    return test_raw


def _resolve_prefetch_sample_count(
    *,
    training: dict[str, Any],
    loader_sample_count: int,
    batch_size: int,
    prefetch_steps: int | None,
) -> int:
    bounded_count = _bounded_training_sample_count(training, loader_sample_count)
    if prefetch_steps is None:
        max_steps = int(training.get("max_steps", 1))
        if max_steps < 0:
            raise ValueError("training.max_steps must be >= 0")
        if max_steps == 0:
            return bounded_count
        return min(int(max_steps) * int(batch_size), bounded_count)
    explicit = int(prefetch_steps)
    if explicit < 0:
        raise ValueError("--prefetch-steps must be >= 0")
    if explicit == 0:
        return bounded_count
    return min(explicit * int(batch_size), bounded_count)


def _validate_snapshot_intervals(
    *,
    checkpoint_interval: int,
    kept_snapshot_interval: int,
    test_interval: int,
) -> None:
    if checkpoint_interval <= 0:
        raise ValueError("training.checkpoint_interval must be > 0")
    if kept_snapshot_interval < 0:
        raise ValueError("training.kept_snapshot_interval must be >= 0")
    if test_interval <= 0:
        raise ValueError("training.test_interval must be > 0 when snapshots are enabled")
    if checkpoint_interval % test_interval != 0:
        raise ValueError(
            "training.checkpoint_interval must be a multiple of training.test_interval"
        )
    if kept_snapshot_interval > 0 and kept_snapshot_interval % test_interval != 0:
        raise ValueError(
            "training.kept_snapshot_interval must be 0 or a multiple of "
            "training.test_interval"
        )


def _resolve_dense_test_selection(
    training: dict[str, Any],
    *,
    loader_sample_count: int,
    default_count: int,
) -> tuple[int, int, str]:
    raw_count = int(training.get("test_control_points", int(default_count)))
    if raw_count <= 0:
        return int(loader_sample_count), 0, "random"
    return raw_count, int(training.get("test_start_sample_index", 0)), "random"


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (value * mask_f).sum() / denom


def _safe_probability_bce(
    input_prob: torch.Tensor,
    target: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    """Compute BCE on sigmoid probabilities outside autocast.

    PyTorch marks probability-space BCE unsafe under autocast. The model
    currently emits sigmoid probabilities, so keep that contract and run the BCE
    kernel in float32 instead of switching this path to logits.
    """
    device_type = str(input_prob.device.type)
    context = (
        torch.autocast(device_type=device_type, enabled=False)
        if device_type in {"cpu", "cuda"}
        else nullcontext()
    )
    with context:
        return F.binary_cross_entropy(
            input_prob.float(),
            target.to(device=input_prob.device, dtype=torch.float32),
            reduction=reduction,
        )


def _nonempty_group_statistics(
    group_sum: torch.Tensor,
    group_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = group_count > 0.0
    minimum = torch.finfo(group_count.dtype).tiny
    group_mean = torch.where(
        valid,
        group_sum / group_count.clamp_min(minimum),
        torch.zeros_like(group_sum),
    )
    return group_mean.sum(), valid.to(dtype=group_sum.dtype).sum()


def _finalize_loss_statistics(
    statistics: _LossStatistics,
    *,
    direction_weight: float,
    presence_weight: float,
    positive_query_weight: float = 1.0,
    negative_query_weight: float = 1.0,
    distributed: "_DistributedConfig | None" = None,
) -> dict[str, torch.Tensor]:
    names = (
        "direction",
        "positive_presence",
        "negative_presence",
        "distribution_presence",
        "angle",
        "branch0",
        "branch1",
        "selected_score",
        "positive_embedding",
        "negative_embedding",
        "contrastive_embedding",
        "short_embedding",
        "long_embedding",
    )
    local_pairs = {
        "contrastive_embedding": (statistics.contrastive_embedding_sum, statistics.contrastive_embedding_count),
        "short_embedding": (statistics.short_embedding_sum, statistics.short_embedding_count),
        "long_embedding": (statistics.long_embedding_sum, statistics.long_embedding_count),
        "positive_embedding": (statistics.positive_embedding_sum, statistics.positive_embedding_count),
        "negative_embedding": (statistics.negative_embedding_sum, statistics.negative_embedding_count),
        "direction": (statistics.direction_sum, statistics.direction_count),
        "positive_presence": (
            statistics.positive_presence_sum,
            statistics.positive_presence_count,
        ),
        "negative_presence": (
            statistics.negative_presence_sum,
            statistics.negative_presence_count,
        ),
        "distribution_presence": (
            statistics.distribution_presence_sum,
            statistics.distribution_presence_count,
        ),
        "angle": (statistics.angle_sum, statistics.angle_count),
        "branch0": (statistics.branch0_sum, statistics.branch_count),
        "branch1": (statistics.branch1_sum, statistics.branch_count),
        "selected_score": (
            statistics.selected_score_sum,
            statistics.selected_score_count,
        ),
    }
    global_values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    world_size = 1
    if distributed is not None and distributed.enabled:
        world_size = int(distributed.world_size)
        packed = torch.stack(
            [
                value.detach().to(dtype=torch.float32).reshape(())
                for name in names
                for value in local_pairs[name]
            ]
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        global_values = {
            name: (packed[index * 2], packed[index * 2 + 1])
            for index, name in enumerate(names)
        }

    def ratio(name: str) -> torch.Tensor:
        local_sum, local_count = local_pairs[name]
        if not global_values:
            return local_sum / local_count.clamp_min(1.0)
        global_sum, global_count = global_values[name]
        denominator = global_count.clamp_min(1.0)
        differentiable = local_sum * (float(world_size) / denominator)
        value = global_sum / denominator
        return differentiable + (
            value.to(dtype=differentiable.dtype) - differentiable.detach()
        )

    direction_loss = ratio("direction")
    positive_presence_loss = ratio("positive_presence")
    negative_presence_loss = ratio("negative_presence")
    distribution_presence_loss = ratio("distribution_presence")
    positive_count = (
        global_values["positive_presence"][1]
        if global_values
        else statistics.positive_presence_count
    )
    negative_count = (
        global_values["negative_presence"][1]
        if global_values
        else statistics.negative_presence_count
    )
    distribution_count = (
        global_values["distribution_presence"][1]
        if global_values
        else statistics.distribution_presence_count
    )
    distribution_active = (distribution_count > 0.0).to(
        dtype=positive_presence_loss.dtype
    )
    # Binary targets retain their historical 1:1 positive/background scale.
    # Gaussian targets decompose the same total positive/negative pressure as
    # 1.0 ridge-positive + 0.5 within-support distribution-negative + 0.5
    # external background-negative (conceptual shares 50% / 25% / 25%).
    presence_loss = (
        float(positive_query_weight)
        * positive_presence_loss
        * (positive_count > 0.0).to(dtype=positive_presence_loss.dtype)
        + float(negative_query_weight)
        * (
            (1.0 - 0.5 * distribution_active)
            * negative_presence_loss
            * (negative_count > 0.0).to(dtype=negative_presence_loss.dtype)
            + 0.5
            * distribution_active
            * distribution_presence_loss
        )
    )
    split_embedding = (local_pairs["short_embedding"][1] > 0) | (local_pairs["long_embedding"][1] > 0)
    if global_values:
        split_embedding = ((global_values["short_embedding"][1] > 0) |
                           (global_values["long_embedding"][1] > 0))
    embedding_loss = torch.where(
        split_embedding,
        0.5 * (ratio("short_embedding") + ratio("long_embedding")),
        0.5 * (ratio("positive_embedding") + ratio("negative_embedding")) + ratio("contrastive_embedding"),
    )
    total = float(direction_weight) * direction_loss + float(presence_weight) * presence_loss + embedding_loss
    return {
        "total": total,
        "direction": direction_loss,
        "presence": presence_loss,
        "presence_ridge": positive_presence_loss,
        "presence_distribution": distribution_presence_loss,
        "presence_background": negative_presence_loss,
        "embedding": embedding_loss,
        "embedding_contrastive": ratio("contrastive_embedding"),
        "embedding_anchors": global_values["contrastive_embedding"][1] if global_values else statistics.contrastive_embedding_count,
        "embedding_short": ratio("short_embedding"),
        "embedding_short_anchors": global_values["short_embedding"][1] if global_values else statistics.short_embedding_count,
        "embedding_long": ratio("long_embedding"),
        "embedding_long_anchors": global_values["long_embedding"][1] if global_values else statistics.long_embedding_count,
        "embedding_positive": ratio("positive_embedding"),
        "embedding_negative": ratio("negative_embedding"),
        "embedding_negative_pairs": global_values["negative_embedding"][1] if global_values else statistics.negative_embedding_count,
        "angle_mean_deg": ratio("angle"),
        "branch0_fraction": ratio("branch0"),
        "branch1_fraction": ratio("branch1"),
        "selected_score_mean": ratio("selected_score"),
    }


def _enforce_two_branch_min_fraction(
    selected_branch: torch.Tensor,
    score: torch.Tensor,
    *,
    min_fraction: float = 0.10,
) -> torch.Tensor:
    """Repair two-branch positive routing when one branch falls below quota."""
    if int(score.shape[1]) != 2 or int(selected_branch.numel()) == 0:
        return selected_branch
    sample_count = int(selected_branch.numel())
    min_count = min(sample_count // 2, int(math.ceil(float(sample_count) * float(min_fraction))))
    if min_count <= 0:
        return selected_branch

    score_detached = score.detach()
    min_count_t = torch.as_tensor(min_count, dtype=torch.long, device=selected_branch.device)
    branch0_count = (selected_branch == 0).sum()
    branch1_count = (selected_branch == 1).sum()
    under0 = branch0_count < min_count_t
    under1 = branch1_count < min_count_t

    def top_missing_mask(branch: int, missing: torch.Tensor) -> torch.Tensor:
        candidate_mask = selected_branch != branch
        branch_score = score_detached[:, branch]
        masked_score = torch.where(
            candidate_mask,
            branch_score,
            torch.full_like(branch_score, -torch.inf),
        )
        order = torch.argsort(masked_score, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank[order] = torch.arange(sample_count, dtype=order.dtype, device=order.device)
        return rank < missing.clamp_min(0)

    repair0 = under0 & top_missing_mask(0, min_count_t - branch0_count)
    repair1 = (~under0) & under1 & top_missing_mask(1, min_count_t - branch1_count)
    return torch.where(
        repair0,
        torch.zeros_like(selected_branch),
        torch.where(repair1, torch.ones_like(selected_branch), selected_branch),
    )


def _select_branch_by_chunked_min_fraction(
    score: torch.Tensor,
    indices_bzyx: torch.Tensor,
    *,
    stream_indices: torch.Tensor | None = None,
    chunk_size: int = 2,
    min_fraction: float = 0.10,
) -> torch.Tensor:
    if int(score.shape[1]) != 2 or int(score.shape[0]) == 0:
        return torch.argmax(score.detach(), dim=1)
    if int(chunk_size) <= 1:
        selected_branch = torch.argmax(score.detach(), dim=1)
        return _enforce_two_branch_min_fraction(
            selected_branch,
            score,
            min_fraction=min_fraction,
        )

    coarse = indices_bzyx.clone()
    if stream_indices is not None:
        offsets = _branch_choice_grid_offsets(
            stream_indices.to(device=indices_bzyx.device),
            chunk_size=int(chunk_size),
        )
        coarse[:, 1:] = coarse[:, 1:] + offsets[coarse[:, 0]]
    coarse[:, 1:] = torch.div(coarse[:, 1:], int(chunk_size), rounding_mode="floor")
    unique_chunks, inverse = torch.unique(coarse, dim=0, sorted=True, return_inverse=True)
    chunk_count = int(unique_chunks.shape[0])
    score_detached = score.detach()
    chunk_sum = torch.zeros(
        (chunk_count, 2),
        dtype=score_detached.dtype,
        device=score_detached.device,
    )
    chunk_items = torch.zeros((chunk_count,), dtype=score_detached.dtype, device=score_detached.device)
    chunk_sum.scatter_add_(0, inverse.view(-1, 1).expand(-1, 2), score_detached)
    chunk_items.scatter_add_(
        0,
        inverse,
        torch.ones((int(inverse.numel()),), dtype=score_detached.dtype, device=score_detached.device),
    )
    chunk_score = chunk_sum / chunk_items.clamp_min(1.0).view(-1, 1)
    selected_chunk_branch = torch.argmax(chunk_score, dim=1)
    # Enforce the quota independently for each stable stream/sample. A quota
    # over the rank-local batch changes routing when the same optimizer batch
    # is partitioned differently across DDP ranks.
    stream_by_batch = (
        stream_indices.to(device=indices_bzyx.device, dtype=torch.long)
        if stream_indices is not None
        else torch.arange(
            int(indices_bzyx[:, 0].max().item()) + 1,
            dtype=torch.long,
            device=indices_bzyx.device,
        )
    )
    chunk_stream = stream_by_batch[unique_chunks[:, 0]]
    for stream in torch.unique(chunk_stream, sorted=True):
        stream_mask = chunk_stream == stream
        selected_chunk_branch[stream_mask] = _enforce_two_branch_min_fraction(
            selected_chunk_branch[stream_mask],
            chunk_score[stream_mask],
            min_fraction=min_fraction,
        )
    return selected_chunk_branch[inverse]


def _branch_choice_grid_offsets(
    stream_indices: torch.Tensor,
    *,
    chunk_size: int = 2,
) -> torch.Tensor:
    chunk = int(chunk_size)
    if chunk <= 1:
        return torch.zeros(
            (int(stream_indices.numel()), 3),
            dtype=torch.long,
            device=stream_indices.device,
        )
    stream = stream_indices.to(dtype=torch.long).view(-1)
    hashed = (stream * 1_103_515_245 + 12_345) & 0x7FFFFFFF
    return torch.stack(
        [
            torch.remainder(hashed, chunk),
            torch.remainder(torch.div(hashed, chunk, rounding_mode="floor"), chunk),
            torch.remainder(torch.div(hashed, chunk * chunk, rounding_mode="floor"), chunk),
        ],
        dim=1,
    ).to(dtype=torch.long)


def _model_uses_conditioned_decoder(model: torch.nn.Module) -> bool:
    return bool(getattr(_unwrap_model(model), "conditioned_decoder_enabled", False))


def _model_output_stride(model: torch.nn.Module) -> int:
    stride = int(getattr(_unwrap_model(model), "output_stride", 1))
    if stride <= 0:
        raise ValueError(f"model output_stride must be > 0, got {stride}")
    return stride


def _conditioned_loss_options_from_training(training: dict[str, Any]) -> dict[str, float]:
    return {
        "perpendicular_jitter_degrees": float(
            training.get("conditioned_perpendicular_jitter_degrees", 45.0)
        ),
        "positive_query_weight": float(
            training.get("conditioned_positive_query_weight", 1.0)
        ),
        "negative_query_weight": float(
            training.get("conditioned_negative_query_weight", 1.0)
        ),
    }


def _deterministic_uniform01_from_keys(
    keys: torch.Tensor,
    *,
    salt: float,
) -> torch.Tensor:
    values = keys.to(dtype=torch.float32)
    coeff = torch.as_tensor(
        [12.9898, 78.233, 37.719, 15.173, 53.927],
        dtype=torch.float32,
        device=values.device,
    )
    mixed = torch.sum(values * coeff[: int(values.shape[-1])], dim=-1) + float(salt)
    hashed = torch.sin(mixed) * 43758.5453123
    return hashed - torch.floor(hashed)


def _deterministic_unit_vectors_xyz_from_keys(
    keys: torch.Tensor,
    *,
    salt: float,
) -> torch.Tensor:
    x = _deterministic_uniform01_from_keys(keys, salt=float(salt) + 0.17) * 2.0 - 1.0
    y = _deterministic_uniform01_from_keys(keys, salt=float(salt) + 11.31) * 2.0 - 1.0
    z = _deterministic_uniform01_from_keys(keys, salt=float(salt) + 23.79) * 2.0 - 1.0
    vectors = torch.stack([x, y, z], dim=-1).to(dtype=torch.float32)
    norm = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vectors)
    fallback[..., 0] = 1.0
    return torch.where(norm > 1.0e-6, vectors / norm.clamp_min(1.0e-6), fallback)


def _sparse_conditioned_keys(
    batch: FiberTrace3DBatch,
    indices_bzyx: torch.Tensor,
) -> torch.Tensor:
    indices = indices_bzyx.to(dtype=torch.long)
    stream_indices = batch.stream_indices.to(device=indices.device, dtype=torch.long)
    stream = stream_indices[indices[:, 0]]
    return torch.stack(
        [stream, indices[:, 1], indices[:, 2], indices[:, 3]],
        dim=1,
    )


def _perpendicular_basis_xyz(
    target_axis_xyz: torch.Tensor,
    seed_axis_xyz: torch.Tensor,
) -> torch.Tensor:
    target = F.normalize(target_axis_xyz.to(dtype=torch.float32), p=2.0, dim=-1, eps=1.0e-12)
    seed = F.normalize(seed_axis_xyz.to(dtype=torch.float32), p=2.0, dim=-1, eps=1.0e-12)
    perp = seed - torch.sum(seed * target, dim=-1, keepdim=True) * target
    perp_norm = torch.linalg.vector_norm(perp, dim=-1, keepdim=True)
    basis = torch.zeros_like(target)
    least_parallel = torch.argmin(torch.abs(target), dim=-1, keepdim=True)
    basis.scatter_(-1, least_parallel, 1.0)
    fallback = basis - torch.sum(basis * target, dim=-1, keepdim=True) * target
    fallback = F.normalize(fallback, p=2.0, dim=-1, eps=1.0e-12)
    return torch.where(perp_norm > 1.0e-6, perp / perp_norm.clamp_min(1.0e-6), fallback)


def _conditioned_perpendicular_queries(
    batch: FiberTrace3DBatch,
    indices_bzyx: torch.Tensor,
    target_axis_xyz: torch.Tensor,
    *,
    jitter_degrees: float,
) -> torch.Tensor:
    if int(indices_bzyx.shape[0]) == 0:
        return torch.zeros(
            (0, 6),
            dtype=torch.float32,
            device=target_axis_xyz.device,
        )
    keys = _sparse_conditioned_keys(batch, indices_bzyx).to(device=target_axis_xyz.device)
    seed_axis = _deterministic_unit_vectors_xyz_from_keys(keys, salt=19.0)
    perp = _perpendicular_basis_xyz(target_axis_xyz, seed_axis)
    jitter = float(jitter_degrees)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("training.conditioned_perpendicular_jitter_degrees must be >= 0")
    if jitter > 0.0:
        unit = _deterministic_uniform01_from_keys(keys, salt=41.0) * 2.0 - 1.0
        angle = unit.to(dtype=perp.dtype) * math.radians(jitter)
        query_axis = (
            torch.cos(angle)[:, None] * perp
            + torch.sin(angle)[:, None] * F.normalize(
                target_axis_xyz,
                p=2.0,
                dim=-1,
                eps=1.0e-12,
            )
        )
    else:
        query_axis = perp
    query_axis = F.normalize(query_axis, p=2.0, dim=-1, eps=1.0e-12)
    return encode_lasagna_direction_3x2(query_axis).to(dtype=torch.float32)


def _conditioned_random_query_b6(
    batch: FiberTrace3DBatch,
    *,
    device: torch.device,
) -> torch.Tensor:
    streams = batch.stream_indices.to(device=device, dtype=torch.long)
    zeros = torch.zeros_like(streams)
    keys = torch.stack([streams, zeros, zeros, zeros], dim=1)
    axis = _deterministic_unit_vectors_xyz_from_keys(keys, salt=83.0)
    return encode_lasagna_direction_3x2(axis).to(dtype=torch.float32)


def _gather_output_at_indices(output: torch.Tensor, indices_bzyx: torch.Tensor) -> torch.Tensor:
    indices = indices_bzyx.to(dtype=torch.long, device=output.device)
    return output[
        indices[:, 0],
        :,
        indices[:, 1],
        indices[:, 2],
        indices[:, 3],
    ]


def compute_losses(
    output: torch.Tensor,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    branch_selection_mode: str = "eval_voxel",
    embeddings: torch.Tensor | None = None,
    embedding_diagnostics: list | None = None,
    _return_statistics: bool = False,
    presence_loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor] | _LossStatistics:
    # TODO: replace prediction-dependent global counts before re-enabling these modes.
    if output.shape[1] != 7:
        raise ValueError("Ordinary multi-branch losses are temporarily unsupported: "
                         "fix target-only normalization so other samples' predictions cannot change loss weights.")
    require_materialized_targets(batch)
    assert batch.direction_indices_bzyx is not None
    assert batch.direction_target_sparse is not None
    assert batch.direction_weight_sparse is not None
    assert batch.presence_target is not None
    output_shape = tuple(int(v) for v in output.shape[-3:])
    target_shape = tuple(int(v) for v in batch.presence_target.shape[-3:])
    if output_shape != target_shape:
        raise ValueError(
            f"model output shape {output_shape} does not match target shape {target_shape}"
        )
    pred_dirs = direction_outputs(output)
    pred_presences = presence_outputs(output)
    branch_count = int(pred_dirs.shape[1])
    indices = batch.direction_indices_bzyx.to(dtype=torch.long)
    if int(indices.shape[0]) > 0:
        pred_sparse = pred_dirs[
            indices[:, 0],
            :,
            :,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        pred_presence_sparse = pred_presences[
            indices[:, 0],
            :,
            0,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        pred_axis = decode_lasagna_direction_3x2_analytic(
            pred_sparse.reshape(-1, 6)
        ).reshape(int(indices.shape[0]), branch_count, 3)
        target_axis = decode_lasagna_direction_3x2_analytic(
            batch.direction_target_sparse
        )
        score = torch.abs(
            torch.sum(pred_axis * target_axis[:, None, :], dim=-1)
        ).clamp(0.0, 1.0) * pred_presence_sparse.clamp(0.0, 1.0)
        if branch_selection_mode == "eval_voxel":
            selected_branch = torch.argmax(score.detach(), dim=1)
        elif branch_selection_mode == "train_offset_grid_min_fraction":
            selected_branch = _select_branch_by_chunked_min_fraction(
                score,
                indices,
                stream_indices=batch.stream_indices,
                chunk_size=2,
                min_fraction=0.10,
            )
        else:
            raise ValueError(
                "branch_selection_mode must be eval_voxel or "
                f"train_offset_grid_min_fraction, got {branch_selection_mode!r}"
            )
        row_index = torch.arange(
            int(indices.shape[0]),
            dtype=torch.long,
            device=indices.device,
        )
        pred_selected = pred_sparse[row_index, selected_branch]
        direction_error = (pred_selected - batch.direction_target_sparse) ** 2
        direction_error = direction_error * batch.direction_weight_sparse
        direction_sum = direction_error.sum()
        if batch.presence_target_mode == "anisotropic_gaussian":
            direction_falloff = batch.presence_target[
                indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]
            ].to(dtype=direction_error.dtype)
            direction_count = direction_falloff.sum() * int(direction_error.shape[-1])
        else:
            direction_falloff = None
            direction_count = direction_error.new_tensor(direction_error.numel())
        selected_axis = pred_axis[row_index, selected_branch]
        agreement = torch.abs(torch.sum(selected_axis * target_axis, dim=-1)).clamp(0.0, 1.0)
        angle_values = torch.rad2deg(torch.acos(agreement))
        if direction_falloff is None:
            angle_sum = angle_values.sum()
            angle_count = angle_values.new_tensor(angle_values.numel())
        else:
            angle_sum = (angle_values * direction_falloff).sum()
            angle_count = direction_falloff.sum()
        selected_presence = pred_presence_sparse[row_index, selected_branch]
        positive_presence_bce = _safe_probability_bce(
            selected_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.ones_like(selected_presence),
            reduction="none",
        )
        branch_patch_count = int(pred_presences.shape[0]) * branch_count
        positive_group_index = indices[:, 0] * branch_count + selected_branch
        positive_group_sum = torch.zeros(
            (branch_patch_count,),
            dtype=positive_presence_bce.dtype,
            device=positive_presence_bce.device,
        )
        positive_group_count = torch.zeros_like(positive_group_sum)
        positive_group_sum.scatter_add_(
            0,
            positive_group_index,
            positive_presence_bce if presence_loss_mask is None else positive_presence_bce * presence_loss_mask[
                indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]],
        )
        positive_group_count.scatter_add_(
            0,
            positive_group_index,
            torch.ones_like(positive_presence_bce) if presence_loss_mask is None else presence_loss_mask[
                indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]].to(positive_presence_bce.dtype),
        )
        positive_presence_sum, positive_presence_group_count = _nonempty_group_statistics(
            positive_group_sum,
            positive_group_count,
        )
        selected_scores = score[row_index, selected_branch]
        selected_score_sum = selected_scores.sum()
        selected_score_count = selected_scores.new_tensor(selected_scores.numel())
        branch_count_stat = selected_scores.new_tensor(selected_branch.numel())
        branch0_sum = (selected_branch == 0).to(dtype=selected_scores.dtype).sum()
        branch1_sum = (selected_branch == 1).to(dtype=selected_scores.dtype).sum()
    else:
        zero = pred_dirs.sum() * 0.0
        direction_sum = zero
        direction_count = zero.detach()
        angle_sum = zero
        angle_count = zero.detach()
        positive_presence_sum = pred_presences.sum() * 0.0
        positive_presence_group_count = torch.zeros(
            (),
            dtype=pred_presences.dtype,
            device=pred_presences.device,
        )
        selected_score_sum = pred_presences.sum() * 0.0
        selected_score_count = zero.detach()
        branch0_sum = pred_presences.sum() * 0.0
        branch1_sum = pred_presences.sum() * 0.0
        branch_count_stat = zero.detach()

    presence_target = batch.presence_target.unsqueeze(1).expand_as(pred_presences)
    if batch.presence_target_mode == "anisotropic_gaussian":
        positive_mask = presence_target > 0.0
        if presence_loss_mask is not None:
            positive_mask = positive_mask & presence_loss_mask.unsqueeze(1)
        positive_presence_bce = _safe_probability_bce(
            pred_presences.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.ones_like(pred_presences),
            reduction="none",
        )
        ridge_weight = presence_target.to(dtype=positive_presence_bce.dtype) * positive_mask
        positive_group_sum = (positive_presence_bce * ridge_weight).sum(
            dim=(2, 3, 4, 5)
        ).reshape(-1)
        positive_group_count = ridge_weight.sum(dim=(2, 3, 4, 5)).reshape(-1)
        positive_presence_sum, positive_presence_group_count = _nonempty_group_statistics(
            positive_group_sum,
            positive_group_count,
        )
        distribution_presence_bce = _safe_probability_bce(
            pred_presences.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.zeros_like(pred_presences),
            reduction="none",
        )
        distribution_weight = (
            (1.0 - presence_target).to(dtype=distribution_presence_bce.dtype)
            * positive_mask
        )
        distribution_group_sum = (distribution_presence_bce * distribution_weight).sum(
            dim=(2, 3, 4, 5)
        ).reshape(-1)
        distribution_group_count = distribution_weight.sum(
            dim=(2, 3, 4, 5)
        ).reshape(-1)
        distribution_presence_sum, distribution_presence_group_count = (
            _nonempty_group_statistics(
                distribution_group_sum,
                distribution_group_count,
            )
        )
        negative_mask = presence_target == 0.0
    else:
        distribution_presence_sum = pred_presences.sum() * 0.0
        distribution_presence_group_count = torch.zeros(
            (), dtype=pred_presences.dtype, device=pred_presences.device
        )
        negative_mask = presence_target <= 0.5
    if presence_loss_mask is not None:
        negative_mask = negative_mask & presence_loss_mask.unsqueeze(1)
    negative_presence_bce = _safe_probability_bce(
        pred_presences.clamp(1.0e-6, 1.0 - 1.0e-6),
        torch.zeros_like(pred_presences),
        reduction="none",
    )
    negative_mask_f = negative_mask.to(dtype=negative_presence_bce.dtype)
    negative_group_sum = (negative_presence_bce * negative_mask_f).sum(
        dim=(2, 3, 4, 5)
    ).reshape(-1)
    negative_group_count = negative_mask_f.sum(dim=(2, 3, 4, 5)).reshape(-1)
    negative_presence_sum, negative_presence_group_count = _nonempty_group_statistics(
        negative_group_sum,
        negative_group_count,
    )
    statistics = _LossStatistics(
        direction_sum=direction_sum,
        direction_count=direction_count,
        positive_presence_sum=positive_presence_sum,
        positive_presence_count=positive_presence_group_count,
        negative_presence_sum=negative_presence_sum,
        negative_presence_count=negative_presence_group_count,
        distribution_presence_sum=distribution_presence_sum,
        distribution_presence_count=distribution_presence_group_count,
        angle_sum=angle_sum,
        angle_count=angle_count,
        branch0_sum=branch0_sum,
        branch1_sum=branch1_sum,
        branch_count=branch_count_stat,
        selected_score_sum=selected_score_sum,
        selected_score_count=selected_score_count,
    )
    if embeddings is not None:
        embedding_values = embedding_statistics(embeddings, pred_presences.amax(dim=1), batch, embedding_diagnostics)
        pos, pc, neg, nc = embedding_values[:4]
        statistics = replace(statistics, positive_embedding_sum=pos, positive_embedding_count=pc,
                             negative_embedding_sum=neg, negative_embedding_count=nc)
        if len(embedding_values) == 6:
            statistics = replace(statistics, contrastive_embedding_sum=embedding_values[4],
                                 contrastive_embedding_count=embedding_values[5])
        elif len(embedding_values) == 10:
            statistics = replace(statistics,
                contrastive_embedding_sum=embedding_values[4],
                contrastive_embedding_count=embedding_values[5],
                short_embedding_sum=embedding_values[6],
                short_embedding_count=embedding_values[7],
                long_embedding_sum=embedding_values[8],
                long_embedding_count=embedding_values[9])
    if _return_statistics:
        return statistics
    return _finalize_loss_statistics(
        statistics,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
    )


def compute_conditioned_losses(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    perpendicular_jitter_degrees: float = 45.0,
    positive_query_weight: float = 1.0,
    negative_query_weight: float = 1.0,
    _return_statistics: bool = False,
    prediction_observer: Any | None = None,
) -> dict[str, torch.Tensor] | _LossStatistics:
    require_materialized_targets(batch)
    assert batch.direction_indices_bzyx is not None
    assert batch.direction_target_sparse is not None
    assert batch.direction_weight_sparse is not None
    assert batch.presence_target is not None
    if not _model_uses_conditioned_decoder(model):
        raise ValueError("compute_conditioned_losses requires conditioned decoder model mode")
    if float(positive_query_weight) < 0.0 or not math.isfinite(float(positive_query_weight)):
        raise ValueError("conditioned_positive_query_weight must be finite and non-negative")
    if float(negative_query_weight) < 0.0 or not math.isfinite(float(negative_query_weight)):
        raise ValueError("conditioned_negative_query_weight must be finite and non-negative")

    forward_device = batch.volume.device
    indices = batch.direction_indices_bzyx.to(dtype=torch.long, device=forward_device)
    target_axis = decode_lasagna_direction_3x2_analytic(
        batch.direction_target_sparse.to(device=forward_device)
    )
    random_query = _conditioned_random_query_b6(batch, device=forward_device).to(
        dtype=batch.volume.dtype
    )
    perp_query = (
        _conditioned_perpendicular_queries(
            batch,
            indices,
            target_axis,
            jitter_degrees=perpendicular_jitter_degrees,
        ).to(dtype=batch.volume.dtype, device=forward_device)
        if int(indices.shape[0]) > 0
        else None
    )
    components = model(
        batch.volume,
        conditioned_random_query=random_query,
        conditioned_point_indices_bzyx=indices if int(indices.shape[0]) > 0 else None,
        conditioned_point_query_n6=perp_query,
        return_conditioned_components=True,
    )
    if not isinstance(components, dict):
        raise RuntimeError("conditioned model forward did not return component outputs")
    zero_output = components["zero_output"]
    random_output = components["random_output"]
    perp_sparse = components["point_output"]
    if zero_output is None or random_output is None:
        raise RuntimeError("conditioned model forward did not return dense outputs")
    if prediction_observer is not None:
        prediction_observer(batch, torch.cat((zero_output.detach(), random_output.detach()), dim=1))
    batch_size, _channels, depth, height, width = (int(v) for v in zero_output.shape)

    positive_query_count = 2
    if int(indices.shape[0]) > 0:
        zero_sparse = _gather_output_at_indices(zero_output, indices)
        if perp_sparse is None:
            raise RuntimeError("conditioned model forward did not return sparse point output")
        positive_pred = torch.stack([zero_sparse, perp_sparse], dim=1)
        pred_dirs = positive_pred[:, :, :6]
        pred_presence = positive_pred[:, :, 6]
        direction_error = (
            pred_dirs
            - batch.direction_target_sparse.to(dtype=pred_dirs.dtype, device=pred_dirs.device)[:, None, :]
        ) ** 2
        direction_error = direction_error * batch.direction_weight_sparse.to(
            dtype=direction_error.dtype,
            device=direction_error.device,
        )[:, None, :]
        direction_sum = direction_error.sum()
        if batch.presence_target_mode == "anisotropic_gaussian":
            direction_falloff = batch.presence_target[
                indices[:, 0], 0, indices[:, 1], indices[:, 2], indices[:, 3]
            ].to(dtype=direction_error.dtype)
            direction_count = (
                direction_falloff.sum()
                * int(direction_error.shape[1])
                * int(direction_error.shape[2])
            )
        else:
            direction_falloff = None
            direction_count = direction_error.new_tensor(direction_error.numel())
        pred_axis = decode_lasagna_direction_3x2_analytic(pred_dirs.reshape(-1, 6)).reshape(
            int(indices.shape[0]),
            positive_query_count,
            3,
        )
        agreement = torch.abs(torch.sum(pred_axis * target_axis[:, None, :], dim=-1)).clamp(
            0.0,
            1.0,
        )
        angle_values = torch.rad2deg(torch.acos(agreement))
        if direction_falloff is None:
            angle_sum = angle_values.sum()
            angle_count = angle_values.new_tensor(angle_values.numel())
        else:
            angle_sum = (angle_values * direction_falloff[:, None]).sum()
            angle_count = direction_falloff.sum() * int(angle_values.shape[1])
        positive_presence_bce = _safe_probability_bce(
            pred_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.ones_like(pred_presence),
            reduction="none",
        )
        group_index = (
            indices[:, 0, None] * positive_query_count
            + torch.arange(
                positive_query_count,
                dtype=torch.long,
                device=indices.device,
            ).view(1, positive_query_count)
        )
        positive_group_sum = torch.zeros(
            (batch_size * positive_query_count,),
            dtype=positive_presence_bce.dtype,
            device=positive_presence_bce.device,
        )
        positive_group_count = torch.zeros_like(positive_group_sum)
        positive_group_sum.scatter_add_(
            0,
            group_index.reshape(-1),
            positive_presence_bce.reshape(-1),
        )
        positive_group_count.scatter_add_(
            0,
            group_index.reshape(-1),
            torch.ones_like(positive_presence_bce).reshape(-1),
        )
        positive_presence_sum, positive_presence_group_count = _nonempty_group_statistics(
            positive_group_sum,
            positive_group_count,
        )
        selected_scores = agreement * pred_presence.clamp(0.0, 1.0)
        selected_score_sum = selected_scores.sum()
        selected_score_count = selected_scores.new_tensor(selected_scores.numel())
        branch_count_stat = selected_scores.new_tensor(selected_scores.numel())
        branch0_sum = branch_count_stat * 0.5
        branch1_sum = branch_count_stat * 0.5
    else:
        zero = zero_output.sum() * 0.0
        direction_sum = zero
        direction_count = zero.detach()
        angle_sum = zero
        angle_count = zero.detach()
        positive_presence_sum = zero
        positive_presence_group_count = torch.zeros(
            (),
            dtype=zero_output.dtype,
            device=zero_output.device,
        )
        selected_score_sum = zero
        selected_score_count = zero.detach()
        branch0_sum = zero
        branch1_sum = zero
        branch_count_stat = zero.detach()

    negative_presence = torch.stack(
        [zero_output[:, 6:7], random_output[:, 6:7]],
        dim=1,
    )
    expected_spatial = (batch_size, 2, 1, depth, height, width)
    if tuple(int(v) for v in negative_presence.shape) != expected_spatial:
        raise ValueError(
            "conditioned negative presence output shape mismatch: "
            f"{tuple(int(v) for v in negative_presence.shape)} != {expected_spatial}"
        )
    negative_presence_bce = _safe_probability_bce(
        negative_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
        torch.zeros_like(negative_presence),
        reduction="none",
    )
    negative_group_sum = negative_presence_bce.sum(
        dim=(2, 3, 4, 5)
    ).reshape(-1)
    negative_group_count = torch.full_like(
        negative_group_sum,
        float(depth * height * width),
    )
    negative_presence_sum, negative_presence_group_count = _nonempty_group_statistics(
        negative_group_sum,
        negative_group_count,
    )
    statistics = _LossStatistics(
        direction_sum=direction_sum,
        direction_count=direction_count,
        positive_presence_sum=positive_presence_sum,
        positive_presence_count=positive_presence_group_count,
        negative_presence_sum=negative_presence_sum,
        negative_presence_count=negative_presence_group_count,
        angle_sum=angle_sum,
        angle_count=angle_count,
        branch0_sum=branch0_sum,
        branch1_sum=branch1_sum,
        branch_count=branch_count_stat,
        selected_score_sum=selected_score_sum,
        selected_score_count=selected_score_count,
    )
    if _return_statistics:
        return statistics
    return _finalize_loss_statistics(
        statistics,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
        positive_query_weight=positive_query_weight,
        negative_query_weight=negative_query_weight,
    )


@dataclass(frozen=True)
class _SampleProgress:
    start_update: int
    start_samples: int
    global_batch_size: int

    def at(self, update: int) -> int:
        return self.start_samples + (int(update) - self.start_update) * self.global_batch_size


class _SampleAxisWriter:
    """Translate update-based logging calls at the TensorBoard boundary only."""

    def __init__(self, writer: Any, progress: _SampleProgress):
        self.writer = writer
        self.progress = progress

    def add_scalar(self, tag, value, global_step, **kwargs):
        return self.writer.add_scalar(tag, value, self.progress.at(global_step), **kwargs)

    def add_image(self, tag, value, global_step, **kwargs):
        return self.writer.add_image(tag, value, self.progress.at(global_step), **kwargs)

    def __getattr__(self, name):
        return getattr(self.writer, name)


def _resume_sample_count(metadata: dict[str, Any], update: int, override: Any = None) -> int:
    if override is not None:
        count = int(override)
    elif metadata.get("samples_seen") is not None:
        count = int(metadata["samples_seen"])
    elif int(update) == 0:
        count = 0
    else:
        batch = metadata.get("optimizer_global_batch_size")
        if batch is None:
            batch = metadata.get("config", {}).get("training", {}).get("optimizer_global_batch_size")
        if batch is None or int(batch) <= 0:
            raise ValueError(
                "Legacy checkpoint has no global sample count/batch size; set "
                "training.resume_samples_seen to its cumulative training sample count"
            )
        count = int(update) * int(batch)
    if count < 0:
        raise ValueError("resumed sample count must be nonnegative")
    return count


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_scaler: torch.amp.GradScaler | None = None,
    step: int,
    config: dict[str, Any],
    metric: float | None,
    metric_name: str | None = None,
    sample_progress: _SampleProgress | None = None,
    ema: ModelEMA | None = None,
    ema_name: str | None = None,
    refinement_state: dict | None = None,
) -> None:
    snapshot_model = _unwrap_model(model)
    payload = {
        **model_optimizer_payload(snapshot_model, optimizer, grad_scaler, ema),
        "step": int(step),
        "config": _json_safe(config),
        "metric": metric,
        "metric_name": metric_name,
    }
    if refinement_state is not None:
        payload["refinement_state"] = refinement_state
    if sample_progress is not None:
        payload["samples_seen"] = sample_progress.at(step)
        payload["optimizer_global_batch_size"] = sample_progress.global_batch_size
    if ema_name is not None:
        track = ema.tracks[ema_name]
        payload["model"] = track["model"]
        payload["snapshot_kind"] = "ema"
        payload["ema_metadata"] = {key: value for key, value in track.items() if key != "model"}
        for key in ("optimizer", "grad_scaler", "ema"):
            payload.pop(key, None)
    torch.save(payload, path)


def _optimizer_hparams_from_training(training: dict[str, Any]) -> dict[str, float]:
    return {
        "lr": float(training.get("learning_rate", 1.0e-3)),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }


def _apply_optimizer_hparams(
    optimizer: torch.optim.Optimizer,
    hparams: dict[str, float],
) -> None:
    for group in optimizer.param_groups:
        for key, value in hparams.items():
            group[key] = float(value)


def _load_snapshot(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    optimizer_hparams: dict[str, float] | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    map_location: torch.device | str = "cpu",
    progress_metadata: dict[str, Any] | None = None,
) -> int:
    payload = torch.load(path, map_location=map_location)
    if progress_metadata is not None and isinstance(payload, dict):
        progress_metadata.update({
            key: payload[key] for key in ("samples_seen", "optimizer_global_batch_size", "config", "ema")
            if key in payload
        })
    restore_model_optimizer(payload, model=model, optimizer=optimizer, grad_scaler=grad_scaler)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        if optimizer_hparams is not None:
            _apply_optimizer_hparams(optimizer, optimizer_hparams)
    return int(payload.get("step", 0)) if isinstance(payload, dict) else 0


def _as_hw(value: Any, *, key: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{key} must be a length-2 sequence")
    height, width = (int(v) for v in value)
    if height <= 0 or width <= 0:
        raise ValueError(f"{key} values must be positive")
    return height, width


def _model_depth_for_trace_margin(raw_config: dict[str, Any]) -> int:
    model_cfg = dict(raw_config.get("model_3d", raw_config.get("model", {})))
    if "unet_depth" in model_cfg:
        return max(1, int(model_cfg["unet_depth"]))
    features = model_cfg.get("features_per_stage")
    if isinstance(features, (list, tuple)) and features:
        return max(1, len(features))
    return 4


def _resolve_path_relative(path: str | Path, raw_config: dict[str, Any]) -> Path:
    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return path_obj
    config_dir = raw_config.get("_config_dir")
    if config_dir is not None:
        return (Path(str(config_dir)) / path_obj).resolve()
    return (Path.cwd() / path_obj).resolve()


def _trace2cp_3d_config(raw_config: dict[str, Any]) -> _Trace2Cp3DConfig:
    training = dict(raw_config.get("training", {}))
    has_tests = bool(raw_config.get("test_datasets"))
    enabled = bool(training.get("test_trace2cp_enabled", has_tests))
    control_points = int(
        training.get("test_trace2cp_control_points", training.get("test_control_points", 0))
    )
    sample_mode = "flat" if control_points == 0 else "random"
    rf_margin_raw = training.get("test_trace2cp_rf_margin_px")
    rf_margin = (
        float(_model_depth_for_trace_margin(raw_config))
        if rf_margin_raw is None
        else float(rf_margin_raw)
    )
    if not math.isfinite(rf_margin) or rf_margin < 0.0:
        raise ValueError("training.test_trace2cp_rf_margin_px must be non-negative and finite")
    step_px = float(training.get("test_trace2cp_step_px", 4.0))
    if not math.isfinite(step_px) or step_px <= 0.0:
        raise ValueError("training.test_trace2cp_step_px must be positive and finite")
    loader_config_raw = training.get("test_trace2cp_loader_config")
    loader_config = None if loader_config_raw is None else _resolve_path_relative(loader_config_raw, raw_config)
    patch_key = "test_trace2cp_patch_shape_hw"
    if enabled and loader_config is None and patch_key not in training:
        raise ValueError(
            f"training.{patch_key} is required when test_trace2cp_enabled=true "
            "and no training.test_trace2cp_loader_config is provided"
        )
    patch_shape = _as_hw(training.get(patch_key, [128, 128]), key=f"training.{patch_key}")
    tile_shape = _as_hw(
        training.get("test_trace2cp_tile_shape_hw", [128, 128]),
        key="training.test_trace2cp_tile_shape_hw",
    )
    context = int(
        training.get(
            "test_trace2cp_block_context_voxels",
            max(1, _model_depth_for_trace_margin(raw_config)),
        )
    )
    if context < 0:
        raise ValueError("training.test_trace2cp_block_context_voxels must be >= 0")
    return _Trace2Cp3DConfig(
        enabled=enabled,
        control_points=control_points,
        start_sample_index=int(
            training.get(
                "test_trace2cp_start_sample_index",
                training.get("test_start_sample_index", 0),
            )
        ),
        sample_mode=sample_mode,
        step_px=step_px,
        rf_margin_px=rf_margin,
        presence_enabled=bool(training.get("test_trace2cp_presence_enabled", True)),
        patch_shape_hw=patch_shape,
        strip_z_offset_count=int(training.get("test_trace2cp_strip_z_offset_count", 1)),
        strip_z_offset_step=float(training.get("test_trace2cp_strip_z_offset_step", 1.0)),
        tile_shape_hw=tile_shape,
        block_context_voxels=context,
        loader_config_path=loader_config,
    )


def _make_trace2cp_geometry_loader(raw_config: dict[str, Any], cfg: _Trace2Cp3DConfig):
    from vesuvius.neural_tracing.fiber_trace_2d.loader import (
        FiberStrip2DConfig,
        FiberStrip2DLoader,
        load_config as load_config_2d,
    )

    if cfg.loader_config_path is not None:
        return FiberStrip2DLoader(load_config_2d(cfg.loader_config_path))
    datasets = raw_config.get("test_datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("3D Trace2CP metric requires test_datasets or test_trace2cp_loader_config")
    return FiberStrip2DLoader(
        FiberStrip2DConfig(
            datasets=tuple(dict(entry) for entry in datasets),
            batch_size=1,
            patch_shape_hw=cfg.patch_shape_hw,
            strip_z_offset_count=int(cfg.strip_z_offset_count),
            strip_z_offset_step=float(cfg.strip_z_offset_step),
            seed=int(raw_config.get("seed", 1)),
            prefetch_workers=int(raw_config.get("prefetch_workers", 16)),
            prefetch_sampler_workers=2,
            loader_workers=1,
            volume_cache_dir=(
                None
                if raw_config.get("volume_cache_dir") is None
                else str(raw_config.get("volume_cache_dir"))
            ),
            volume_cache_memory_mib=(
                DEFAULT_VOLUME_CACHE_MEMORY_MIB
                if raw_config.get("volume_cache_memory_mib") is None
                else raw_config.get("volume_cache_memory_mib")
            ),
            volume_io_threads=(
                None
                if raw_config.get("volume_io_threads") is None
                else int(raw_config.get("volume_io_threads"))
            ),
            volume_cache_offline=bool(raw_config.get("volume_cache_offline", False)),
            volume_cache_retry_seconds=float(raw_config.get("volume_cache_retry_seconds", 0.0)),
            config_dir=(
                None
                if raw_config.get("_config_dir") is None
                else Path(str(raw_config.get("_config_dir")))
            ),
            suppress_record_warnings=True,
        )
    )


def _trace2cp_frame_axes_xyz(source: Any) -> tuple[np.ndarray, np.ndarray]:
    coords = source.grid.coords_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    valid = source.grid.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    height, width = int(coords.shape[0]), int(coords.shape[1])
    x_axis = np.zeros((height, width, 3), dtype=np.float32)
    if width == 1:
        tangent = np.asarray(source.grid.frame.tangent_xyz, dtype=np.float32)
        x_axis[...] = tangent.reshape(1, 1, 3)
    else:
        x_axis[:, 1:-1] = coords[:, 2:] - coords[:, :-2]
        x_axis[:, 0] = coords[:, 1] - coords[:, 0]
        x_axis[:, -1] = coords[:, -1] - coords[:, -2]
    x_norm = np.linalg.norm(x_axis, axis=-1, keepdims=True)
    fallback_x = np.asarray(source.grid.frame.tangent_xyz, dtype=np.float32).reshape(1, 1, 3)
    x_axis = np.where(x_norm > 1.0e-6, x_axis / np.maximum(x_norm, 1.0e-6), fallback_x)

    if source.grid.offset_axis_xyz is not None:
        y_axis = source.grid.offset_axis_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    elif height == 1:
        y_axis = np.asarray(source.grid.frame.mesh_normal_xyz, dtype=np.float32).reshape(1, 1, 3)
        y_axis = np.broadcast_to(y_axis, (height, width, 3)).copy()
    else:
        y_axis = np.zeros_like(x_axis)
        y_axis[1:-1] = coords[2:] - coords[:-2]
        y_axis[0] = coords[1] - coords[0]
        y_axis[-1] = coords[-1] - coords[-2]
    y_norm = np.linalg.norm(y_axis, axis=-1, keepdims=True)
    fallback_y = np.asarray(source.grid.frame.mesh_normal_xyz, dtype=np.float32).reshape(1, 1, 3)
    y_axis = np.where(y_norm > 1.0e-6, y_axis / np.maximum(y_norm, 1.0e-6), fallback_y)
    x_axis = np.where(valid[..., None], x_axis, fallback_x)
    y_axis = np.where(valid[..., None], y_axis, fallback_y)
    return x_axis.astype(np.float32, copy=False), y_axis.astype(np.float32, copy=False)


def _valid_block_mask(shape: tuple[int, int, int], start: np.ndarray, volume_shape: tuple[int, int, int]) -> torch.Tensor:
    block_shape = tuple(int(v) for v in shape)
    zz, yy, xx = np.meshgrid(
        np.arange(block_shape[0], dtype=np.int64) + int(start[0]),
        np.arange(block_shape[1], dtype=np.int64) + int(start[1]),
        np.arange(block_shape[2], dtype=np.int64) + int(start[2]),
        indexing="ij",
    )
    valid = (
        (zz >= 0)
        & (zz < int(volume_shape[0]))
        & (yy >= 0)
        & (yy < int(volume_shape[1]))
        & (xx >= 0)
        & (xx < int(volume_shape[2]))
    )
    return torch.as_tensor(valid, dtype=torch.bool)


@torch.no_grad()
def _infer_trace2cp_fields_3d(
    model: torch.nn.Module,
    source: Any,
    *,
    image_normalization: str,
    cfg: _Trace2Cp3DConfig,
    device: torch.device,
    mixed_precision: _MixedPrecisionConfig | None = None,
) -> Trace2Cp3DProjectedFields:
    coords_xyz = source.grid.coords_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    valid_mask = source.grid.valid_mask.detach().cpu().numpy().astype(bool, copy=False)
    coords_zyx = coords_xyz[..., (2, 1, 0)] / np.float32(source.record.volume_spacing_base)
    frame_x, frame_y = _trace2cp_frame_axes_xyz(source)
    height, width = int(coords_zyx.shape[0]), int(coords_zyx.shape[1])
    direction = np.zeros((height, width, 2), dtype=np.float32)
    presence = np.zeros((height, width), dtype=np.float32) if cfg.presence_enabled else None
    projected_valid = np.zeros((height, width), dtype=bool)
    tile_h, tile_w = cfg.tile_shape_hw
    context = int(cfg.block_context_voxels)
    volume_shape = tuple(int(v) for v in getattr(source.record.volume, "shape"))

    was_training = model.training
    model.eval()
    for y0 in range(0, height, tile_h):
        y1 = min(height, y0 + tile_h)
        for x0 in range(0, width, tile_w):
            x1 = min(width, x0 + tile_w)
            tile_valid = valid_mask[y0:y1, x0:x1] & np.isfinite(coords_zyx[y0:y1, x0:x1]).all(axis=-1)
            if not bool(tile_valid.any()):
                continue
            tile_coords = coords_zyx[y0:y1, x0:x1]
            used = tile_coords[tile_valid]
            start = np.floor(np.min(used, axis=0) - float(context) - 1.0).astype(np.int64)
            end = np.ceil(np.max(used, axis=0) + float(context) + 2.0).astype(np.int64)
            if not bool(np.all(end > start)):
                continue
            block = _read_raw_block(source.record.volume, start, end)
            if block.size == 0:
                continue
            block_t = torch.as_tensor(block, dtype=torch.float32, device=device)
            block_valid = _valid_block_mask(tuple(block.shape), start, volume_shape).to(device)
            block_t = _normalize_image(block_t, block_valid, image_normalization)
            with _autocast_context(mixed_precision):
                output = model(block_t.view(1, 1, *block.shape))[0]
            output = output.float()
            fields = project_3d_output_to_trace2cp_fields(
                output,
                tile_coords - start.reshape(1, 1, 3).astype(np.float32),
                tile_valid,
                frame_x_xyz=frame_x[y0:y1, x0:x1],
                frame_y_xyz=frame_y[y0:y1, x0:x1],
            )
            direction[y0:y1, x0:x1] = fields.direction_xy
            projected_valid[y0:y1, x0:x1] |= fields.valid_mask
            if presence is not None and fields.presence_hw is not None:
                presence[y0:y1, x0:x1] = fields.presence_hw
    if was_training:
        model.train()
    return Trace2Cp3DProjectedFields(
        direction_xy=direction,
        valid_mask=projected_valid,
        presence_hw=presence,
    )


def _evaluate_trace2cp_metric_fixed_set_3d(
    model: torch.nn.Module,
    geometry_loader: Any,
    *,
    image_normalization: str,
    cfg: _Trace2Cp3DConfig,
    device: torch.device,
    mixed_precision: _MixedPrecisionConfig | None = None,
) -> _Trace2Cp3DMetricEvalResult:
    rows = list(_iter_trace2cp_metric_rows(model, geometry_loader,
        image_normalization=image_normalization, cfg=cfg, device=device,
        mixed_precision=mixed_precision))
    return _aggregate_trace2cp_metric_rows(rows)


def _iter_trace2cp_metric_rows(model, geometry_loader, *, image_normalization,
                             cfg, device, mixed_precision=None, start_offset=0):
    if int(cfg.control_points) == 0:
        sample_count = int(geometry_loader.sample_count)
        start_sample_index = 0
        sample_mode = "flat"
    else:
        sample_count = int(cfg.control_points)
        start_sample_index = int(cfg.start_sample_index)
        sample_mode = str(cfg.sample_mode)
    for offset in range(start_offset, max(1, sample_count)):
        sample_index = start_sample_index + int(offset)
        try:
            source = geometry_loader.build_trace2cp_segment_source(
                sample_index,
                target_offset=1,
                rf_margin_px=cfg.rf_margin_px,
                device=torch.device("cpu"),
                sample_mode=sample_mode,
            )
            fields = _infer_trace2cp_fields_3d(
                model,
                source,
                image_normalization=image_normalization,
                cfg=cfg,
                device=device,
                mixed_precision=mixed_precision,
            )
            score = score_trace2cp_projected_fields(
                fields,
                start_xy=np.asarray(source.start_control_point_xy, dtype=np.float32),
                target_xy=np.asarray(source.target_control_point_xy, dtype=np.float32),
                step_px=cfg.step_px,
                rf_margin_px=cfg.rf_margin_px,
            )
        except ValueError as exc:
            yield (offset, None, None, " ".join(str(exc).split()))
            continue
        yield (offset, float(score.trace2cp_error), float(score.raw_y_error_px), "")


def _aggregate_trace2cp_metric_rows(rows):
    errors = [row[1] for row in rows if row[1] is not None]
    raw_errors = [row[2] for row in rows if row[1] is not None]
    skipped = len(rows) - len(errors)
    first_skip = next((row[3] for row in rows if row[1] is None), "")
    if not errors:
        raise ValueError(
            "3D test Trace2CP metric found no valid CP-to-next-CP segments: "
            f"sample_count={len(rows)} "
            f"skipped={skipped} first_skip='{first_skip}'"
        )
    return _Trace2Cp3DMetricEvalResult(
        error_mean=float(np.mean(np.asarray(errors, dtype=np.float64))),
        raw_y_error_mean_px=float(np.mean(np.asarray(raw_errors, dtype=np.float64))),
        segments=len(errors),
        skipped_segments=int(skipped),
        first_skip_reason=first_skip,
    )


def _distributed_test_batch_indices(
    *,
    sample_count: int,
    batch_size: int,
    config: _DistributedConfig,
) -> tuple[int, ...]:
    if int(sample_count) < 0:
        raise ValueError("sample_count must be >= 0")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    batch_count = (int(sample_count) + int(batch_size) - 1) // int(batch_size)
    return tuple(range(int(config.rank), batch_count, int(config.world_size)))


def _distributed_evaluation_wave_indices(
    remaining: Sequence[int],
    *,
    start: int,
    wave_size: int,
    rank: int,
    world_size: int,
    total_batch_count: int,
    pad_empty_ranks: bool,
) -> tuple[list[int], bool]:
    """Assign one evaluation wave while preserving DDP forward parity.

    Managed evaluation yields between world-sized waves.  Its final wave can
    contain fewer batches than ranks, but every DDP rank must still execute the
    same number of forwards before the following collective.  Empty ranks load
    a deterministic duplicate solely to preserve that sequencing; callers
    discard its row and disable stateful prediction observers.
    """
    wave = list(remaining[int(start):int(start) + int(wave_size)])
    indices = wave[int(rank)::int(world_size)]
    if indices or not pad_empty_ranks or not wave:
        return indices, False
    # Prefer a distinct, already-defined test batch for every padding rank.
    # Reusing one live shared-memory ticket across processes is not a safe
    # transport primitive, even though all readers are logically read-only.
    occupied = set(wave)
    wanted = int(rank) - len(wave)
    for candidate in range(int(total_batch_count)):
        if candidate in occupied:
            continue
        if wanted == 0:
            return [candidate], True
        wanted -= 1
    # Very small test sets can have fewer batches than ranks. The broker's
    # reference-counted lease remains the bounded fallback for that case.
    return [wave[int(rank) % len(wave)]], True


def _test_batch_take(*, sample_count: int, batch_size: int, global_batch_index: int) -> int:
    remaining = int(sample_count) - int(global_batch_index) * int(batch_size)
    if remaining <= 0:
        return 0
    return min(int(batch_size), remaining)


def _mean_dense_test_rows(
    indexed_rows: Sequence[tuple[int, dict[str, float]]],
) -> dict[str, float]:
    ordered = sorted(indexed_rows, key=lambda item: int(item[0]))
    if not ordered:
        return {
            "total": math.inf,
            "embedding": math.inf,
            "embedding_positive": math.inf,
            "embedding_negative": math.inf,
            "embedding_negative_pairs": 0.0,
            "direction": math.inf,
            "presence": math.inf,
            "angle_mean_deg": math.inf,
            "branch0_fraction": math.inf,
            "branch1_fraction": math.inf,
            "selected_score_mean": math.inf,
        }
    rows = [row for _batch_index, row in ordered]
    result = {
        key: float(sum(row[key] for row in rows) / len(rows))
        for key in rows[0] if 'embedding_similarity/' not in key and 'embedding_similarity_count/' not in key
    }
    for key in {k for row in rows for k in row if 'embedding_similarity/' in k}:
        count_key = key.replace('embedding_similarity/', 'embedding_similarity_count/')
        count = sum(row.get(count_key, 0) for row in rows)
        if count:
            result[key] = sum(row.get(key, 0)*row.get(count_key, 0) for row in rows)/count
            result[count_key] = count
    return result


def _distributed_gather_dense_test_rows(
    local_rows: list[tuple[int, dict[str, float]]],
    config: _DistributedConfig,
) -> list[tuple[int, dict[str, float]]]:
    if not config.enabled:
        return list(local_rows)
    gathered: list[Any] | None = [None] * int(config.world_size) if config.is_main else None
    dist.gather_object(local_rows, gathered, dst=0)
    if not config.is_main:
        return []
    assert gathered is not None
    return [item for rank_rows in gathered for item in rank_rows]


class _FullResolutionEvaluation(torch.nn.Module):
    """Evaluate low-resolution predictions against the historical full-grid targets."""
    output_stride = 1

    def __init__(self, model, mode):
        super().__init__()
        if mode not in {"trilinear", "nearest"}:
            raise ValueError("test_output_upsample must be trilinear or nearest")
        if _model_uses_conditioned_decoder(model):
            raise ValueError("historical full-grid evaluation only supports ordinary models")
        self.model, self.mode = model, mode

    def forward(self, volume):
        output = self.model(volume)
        if output.shape[2:] == volume.shape[2:]:
            return output
        kwargs = {"align_corners": False} if self.mode == "trilinear" else {}
        return F.interpolate(output, size=volume.shape[2:], mode=self.mode, **kwargs)


@torch.no_grad()
def _evaluate_loaded_batch(model, batch, config, *, device, microbatch_size,
                           direction_weight, presence_weight, mixed_precision,
                           conditioned_loss_options=None, prediction_observer=None,
                           historical=False):
    """Shared per-logical-batch evaluation, including bounded physical splits."""
    take = int(batch.volume.shape[0])
    size = take if microbatch_size is None else microbatch_size
    if config.embedding.paired_patches:
        if take % 2 or size < 2:
            raise ValueError("paired evaluation requires complete pairs and microbatch >=2")
        size = size // 2 * 2
    statistics = None
    embedding_diagnostics = []
    for offset in range(0, take, size):
        physical = _slice_batch(batch, offset, min(take, offset + size)).to(device)
        physical = materialize_targets(physical, config, output_stride=_model_output_stride(model))
        if historical:
            physical = replace(physical, presence_mask=historical_presence_mask(physical, config))
        with _autocast_context(mixed_precision):
            part = _forward_loss_statistics(model, physical,
                conditioned_loss_options=conditioned_loss_options, training_loss=False,
                prediction_observer=prediction_observer, historical_presence_mask=historical,
                embedding_diagnostics=embedding_diagnostics)
            # Diagnostics reuse the forward pair samples; no extra inference.
        statistics = part if statistics is None else statistics + part
    options = conditioned_loss_options or {}
    result = _finalize_loss_statistics(statistics,
        direction_weight=direction_weight, presence_weight=presence_weight,
        positive_query_weight=options.get("positive_query_weight", 1.0),
        negative_query_weight=options.get("negative_query_weight", 1.0))
    for key in {k for row in embedding_diagnostics for k in row}:
        values = [row[key] for row in embedding_diagnostics if key in row]
        result['embedding_similarity/'+key] = torch.stack(values).mean()
        result['embedding_similarity_count/'+key] = torch.tensor(len(values), device=device)
    return result


@torch.no_grad()
def _evaluate_dense_loss_rows(
    model: torch.nn.Module,
    loader: FiberTrace3DLoader,
    *,
    device: torch.device,
    start_sample_index: int,
    sample_count: int,
    sample_mode: str = "random",
    sample_index_limit: int | None = None,
    direction_weight: float,
    presence_weight: float,
    conditioned_loss_options: dict[str, float] | None = None,
    mixed_precision: _MixedPrecisionConfig | None = None,
    diagnostics: _TrainingHangDiagnostics | None = None,
    diagnostic_step: int = 0,
    global_batch_indices: Sequence[int] | None = None,
    batch_iterator: Any | None = None,
    prediction_observer: Any | None = None,
    microbatch_size: int | None = None,
    profile_times: dict[str, list[float]] | None = None,
    output_upsample: str | None = None,
    ema: ModelEMA | None = None,
    historical: bool = False,
) -> list[tuple[int, dict[str, float]]]:
    model.eval()
    raw_model = model
    if output_upsample is not None:
        model = _FullResolutionEvaluation(model, output_upsample)
    batch_size = int(loader.config.batch_size)
    if microbatch_size is not None and (type(microbatch_size) is not int or microbatch_size < 1):
        raise ValueError("test microbatch_size must be positive")
    load_device = torch.device("cpu") if microbatch_size is not None else device
    if global_batch_indices is None:
        global_batch_indices = range((int(sample_count) + batch_size - 1) // batch_size)
    indexed_rows: list[tuple[int, dict[str, float]]] = []
    def timed_stage(name, started):
        if profile_times is not None:
            _synchronize_device(device)
            profile_times.setdefault(name, []).append(time.monotonic() - started)

    for local_batch_index, global_batch_index_raw in enumerate(global_batch_indices):
        global_batch_index = int(global_batch_index_raw)
        consumed = global_batch_index * batch_size
        if consumed >= int(sample_count):
            raise ValueError(
                f"global test batch {global_batch_index} begins beyond sample_count={sample_count}"
            )
        batch_started = time.monotonic()
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-load-begin",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
                sample_count=sample_count,
            )
        if batch_iterator is None:
            batch = loader.load_batch(
                start_sample_index + consumed,
                sample_mode=sample_mode,
                sample_index_limit=sample_index_limit,
                device=load_device,
            )
        else:
            batch = next(batch_iterator).to(load_device)
        timed_stage("load", batch_started)
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-load-end",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
                elapsed_s=f"{time.monotonic() - batch_started:.3f}",
            )
            diagnostics.marker(
                "dense-test-targets-begin",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
            )
        targets_started = time.monotonic()
        if microbatch_size is None:
            batch = materialize_targets(
                batch, loader.config, output_stride=_model_output_stride(model),
            )
        timed_stage("targets", targets_started)
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-targets-end",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
                elapsed_s=f"{time.monotonic() - targets_started:.3f}",
            )
        take = _test_batch_take(
            sample_count=sample_count,
            batch_size=batch_size,
            global_batch_index=global_batch_index,
        )
        if int(batch.volume.shape[0]) < take:
            raise ValueError(
                f"test batch {global_batch_index} has {int(batch.volume.shape[0])} samples; "
                f"expected at least {take}"
            )
        if take < int(batch.volume.shape[0]):
            batch = _slice_batch(batch, 0, take)
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-forward-begin",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
                take=take,
            )
        forward_started = time.monotonic()
        batch_options = dict(device=device, microbatch_size=microbatch_size,
            direction_weight=direction_weight, presence_weight=presence_weight,
            mixed_precision=mixed_precision, conditioned_loss_options=conditioned_loss_options,
            historical=historical)
        loss_tensors = _evaluate_loaded_batch(model, batch, loader.config,
            **batch_options, prediction_observer=prediction_observer)
        ema_rows = {}
        for name in (() if ema is None else ema.tracks):
            with ema.apply(raw_model, name):
                values = _evaluate_loaded_batch(model, batch, loader.config, **batch_options)
                ema_rows.update({f"{name}/{key}": value for key, value in _losses_to_float_dict(values).items()})
        _diagnostic_cuda_synchronize(
            diagnostics,
            phase="dense-test-cuda-sync",
            step=diagnostic_step,
            device=device,
            batch=global_batch_index,
            consumed=consumed,
        )
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-forward-end",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
                elapsed_s=f"{time.monotonic() - forward_started:.3f}",
            )
            diagnostics.marker(
                "dense-test-to-cpu-begin",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
            )
        rows = _losses_to_float_dict(loss_tensors)
        rows.update(ema_rows)
        timed_stage("forward", forward_started)
        if diagnostics is not None:
            diagnostics.marker(
                "dense-test-to-cpu-end",
                step=diagnostic_step,
                batch=global_batch_index,
                local_batch=local_batch_index,
                consumed=consumed,
            )
        indexed_rows.append((global_batch_index, rows))
        if batch_iterator is not None and hasattr(batch_iterator, "release_current"):
            # Loss-to-CPU above completes device consumption before slot reuse.
            batch_iterator.release_current()
    model.train()
    return indexed_rows


def _make_test_sample_pipeline(config, *, raw_config, global_batch_indices,
                               start_sample_index=0, sample_mode="sequential"):
    """Reuse training's bounded sample pool for exact logical test batches."""
    indices = tuple(int(i) for i in global_batch_indices)
    if not indices:
        return None
    training = raw_config.get("training", {})
    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    workers = max(2, int(training.get("sample_pipeline_workers", 128)) // world)
    batch = config.batch_size
    slots = max(batch, min(int(training.get("sample_pipeline_slots", 512)) // world,
                           batch * len(indices)))
    slots = max(batch, (slots // batch) * batch)
    return FiberTrace3DTrainingBatchPipeline(
        config_path=config, index_map=EvaluationSampleIndex(batch, indices, start_sample_index),
        batch_size=batch, patch_shape_zyx=config.patch_shape_zyx, batch_count=len(indices),
        workers=workers, load_affinity_workers=max(1, min(workers-1, int(workers*0.75))),
        slot_count=slots, initial_in_flight=min(slots, max(batch, workers)),
        sample_index_limit=None, sample_mode=sample_mode, profile=False,
        single_core_load=True,
        embedding_stride=config.embedding.output_stride if config.embedding.enabled else 0)


@torch.no_grad()
def evaluate_dense_loss(
    model: torch.nn.Module,
    loader: FiberTrace3DLoader,
    **kwargs: Any,
) -> dict[str, float]:
    return _mean_dense_test_rows(_evaluate_dense_loss_rows(model, loader, **kwargs))


def _report_test_timing(
    *, step: int, total_seconds: float, writer: Any | None,
    samples_seen: int | None = None,
) -> None:
    counter = f"step={int(step)}" if samples_seen is None else f"samples={samples_seen} update={step}"
    print(
        f"test_timing {counter} total_seconds={float(total_seconds):.3f}",
        flush=True,
    )
    if writer is not None:
        writer.add_scalar("timing/test_total_seconds", float(total_seconds), int(step))


def _slice_batch(batch: FiberTrace3DBatch, start: int, stop: int) -> FiberTrace3DBatch:
    segment_counts = batch.target_segment_counts[start:stop]
    if int(segment_counts.numel()) > 0:
        source_offsets = batch.target_segment_offsets[start:stop]
        first_segment = int(source_offsets[0])
        segment_total = int(segment_counts.sum())
        segment_start = first_segment
        segment_stop = first_segment + segment_total
        new_offsets = torch.cumsum(
            torch.cat(
                [
                    torch.zeros((1,), dtype=segment_counts.dtype, device=segment_counts.device),
                    segment_counts[:-1],
                ],
                dim=0,
            ),
            dim=0,
        )
    else:
        segment_start = 0
        segment_stop = 0
        new_offsets = batch.target_segment_offsets[start:stop]
    sparse_indices = batch.direction_indices_bzyx
    sparse_target = batch.direction_target_sparse
    sparse_weight = batch.direction_weight_sparse
    sparse_tangent = batch.direction_tangent_sparse_zyx
    if sparse_indices is not None:
        sparse_mask = (sparse_indices[:, 0] >= int(start)) & (sparse_indices[:, 0] < int(stop))
        sparse_indices = sparse_indices[sparse_mask].clone()
        sparse_indices[:, 0] -= int(start)
        if sparse_target is not None:
            sparse_target = sparse_target[sparse_mask]
        if sparse_weight is not None:
            sparse_weight = sparse_weight[sparse_mask]
        if sparse_tangent is not None:
            sparse_tangent = sparse_tangent[sparse_mask]
    return FiberTrace3DBatch(
        presence_target_mode=batch.presence_target_mode,
        embedding_geometry=None if batch.embedding_geometry is None else batch.embedding_geometry[start:stop],
        refinement_geometry=None if batch.refinement_geometry is None else batch.refinement_geometry[start:stop],
        volume=batch.volume[start:stop],
        valid_mask=None if batch.valid_mask is None else batch.valid_mask[start:stop],
        cp_local_zyx=batch.cp_local_zyx[start:stop],
        crop_origin_zyx=batch.crop_origin_zyx[start:stop],
        stream_indices=batch.stream_indices[start:stop],
        data_indices=batch.data_indices[start:stop],
        record_indices=batch.record_indices[start:stop],
        control_point_indices=batch.control_point_indices[start:stop],
        fiber_paths=batch.fiber_paths[start:stop],
        target_modes=batch.target_modes[start:stop],
        target_segment_offsets=new_offsets,
        target_segment_counts=segment_counts,
        target_segment_starts_zyx=batch.target_segment_starts_zyx[segment_start:segment_stop],
        target_segment_ends_zyx=batch.target_segment_ends_zyx[segment_start:segment_stop],
        target_segment_bbox_lo_zyx=batch.target_segment_bbox_lo_zyx[segment_start:segment_stop],
        target_segment_bbox_hi_zyx=batch.target_segment_bbox_hi_zyx[segment_start:segment_stop],
        target_tangent_zyx=batch.target_tangent_zyx[start:stop],
        target_segment_normals_zyx=None
        if batch.target_segment_normals_zyx is None
        else batch.target_segment_normals_zyx[segment_start:segment_stop],
        direction_target=None
        if batch.direction_target is None
        else batch.direction_target[start:stop],
        direction_weight=None
        if batch.direction_weight is None
        else batch.direction_weight[start:stop],
        direction_mask=None if batch.direction_mask is None else batch.direction_mask[start:stop],
        direction_indices_bzyx=sparse_indices,
        direction_target_sparse=sparse_target,
        direction_weight_sparse=sparse_weight,
        direction_tangent_sparse_zyx=sparse_tangent,
        presence_target=None
        if batch.presence_target is None
        else batch.presence_target[start:stop],
        presence_mask=None if batch.presence_mask is None else batch.presence_mask[start:stop],
        profile_timings_ms=batch.profile_timings_ms,
    )


def _combine_prediction_observers(*observers):
    active = tuple(observer for observer in observers if observer is not None)
    if not active:
        return None
    def observe(batch, output):
        for observer in active:
            observer(batch, output)
    return observe


def _forward_loss_tensors(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    conditioned_loss_options: dict[str, float] | None = None,
    training_loss: bool,
    prediction_observer: Any | None = None,
    historical_presence_mask: bool = False,
) -> dict[str, torch.Tensor]:
    if _model_uses_conditioned_decoder(model):
        return compute_conditioned_losses(
            model,
            batch,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
            prediction_observer=prediction_observer,
            **dict(conditioned_loss_options or {}),
        )
    use_embeddings = bool(getattr(_unwrap_model(model), "embedding_channels", 0))
    result = model(batch.volume, return_embeddings=True) if use_embeddings else model(batch.volume)
    output = result["prediction"] if use_embeddings else result
    if prediction_observer is not None:
        prediction_observer(batch, output)
    return compute_losses(
        output,
        batch,
        embeddings=result["embeddings"] if use_embeddings else None,
        presence_loss_mask=batch.presence_mask if historical_presence_mask else None,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
        branch_selection_mode=(
            "train_offset_grid_min_fraction" if training_loss else "eval_voxel"
        ),
    )


def _forward_loss_statistics(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    conditioned_loss_options: dict[str, float] | None = None,
    training_loss: bool = True,
    prediction_observer: Any | None = None,
    historical_presence_mask: bool = False,
    embedding_diagnostics: list | None = None,
) -> _LossStatistics:
    if _model_uses_conditioned_decoder(model):
        result = compute_conditioned_losses(
            model,
            batch,
            direction_weight=1.0,
            presence_weight=1.0,
            **dict(conditioned_loss_options or {}),
            _return_statistics=True,
            prediction_observer=prediction_observer,
        )
    else:
        use_embeddings = bool(getattr(_unwrap_model(model), "embedding_channels", 0))
        prediction = model(batch.volume, return_embeddings=True) if use_embeddings else model(batch.volume)
        output = prediction["prediction"] if use_embeddings else prediction
        if prediction_observer is not None:
            prediction_observer(batch, output)
        result = compute_losses(
            output,
            batch,
            embeddings=prediction["embeddings"] if use_embeddings else None,
            embedding_diagnostics=embedding_diagnostics,
            presence_loss_mask=batch.presence_mask if historical_presence_mask else None,
            direction_weight=1.0,
            presence_weight=1.0,
            branch_selection_mode="train_offset_grid_min_fraction" if training_loss else "eval_voxel",
            _return_statistics=True,
        )
    if not isinstance(result, _LossStatistics):
        raise TypeError("training loss did not return sufficient statistics")
    return result


def _embedding_log(losses, model):
    if not getattr(_unwrap_model(model), "embedding_channels", 0):
        return ""
    extra = ''
    for key in ('positive_mean', 'negative_mean', 'negative_above_0p9', 'hardest_negative_mean', 'positive_hardest_gap',
                'short_positive_mean', 'short_negative_mean', 'short_positive_hardest_gap',
                'long_positive_mean', 'long_negative_mean', 'long_positive_hardest_gap'):
        if 'embedding_similarity/'+key in losses:
            extra += f"emb_{key}={losses['embedding_similarity/'+key]:.4f} "
    if getattr(_unwrap_model(model), 'embedding_short_range_channels', 0):
        budget = (f"embedding_short={losses.get('embedding_short', 0):.6f} "
                  f"embedding_long={losses.get('embedding_long', 0):.6f} "
                  f"embedding_bank_anchors={losses.get('embedding_short_anchors', 0):.0f} ")
    else:
        budget = (f"embedding_anchors={losses.get('embedding_anchors', 0):.0f} "
              if getattr(_unwrap_model(model), 'embedding_objective', 'pairwise') == 'local_contrastive'
              else f"embedding_negative_pairs={losses['embedding_negative_pairs']:.0f} ")
    return f"loss_embedding={losses['embedding']:.6f} " + budget + extra


def _log_embedding_scalars(writer, prefix, losses, model, step):
    if getattr(_unwrap_model(model), "embedding_channels", 0):
        for key in ("embedding", "embedding_positive", "embedding_negative", "embedding_negative_pairs", "embedding_contrastive", "embedding_anchors",
                    "embedding_short", "embedding_short_anchors", "embedding_long", "embedding_long_anchors"):
            if key not in losses:
                continue
            writer.add_scalar(f"{prefix}/{key}", losses[key], step)
        for key, value in losses.items():
            if key.startswith(('embedding_similarity/', 'embedding_similarity_count/')):
                writer.add_scalar(f"{prefix}/{key}", value, step)


def _log_test_scalars(writer, prefix, losses, model, step):
    """Raw, EMA and historical observations use the same component tags."""
    for key in ("total", "direction", "presence", "angle_mean_deg",
                "branch0_fraction", "branch1_fraction", "selected_score_mean"):
        if key in losses:
            tag = f"loss_{key}" if key in {"total", "direction", "presence"} else key
            writer.add_scalar(f"{prefix}/{tag}", losses[key], step)
    _log_embedding_scalars(writer, prefix, losses, model, step)


def _losses_to_float_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def _distributed_mean_loss_tensors(
    losses: dict[str, torch.Tensor],
    config: _DistributedConfig,
) -> dict[str, torch.Tensor]:
    if not config.enabled:
        return losses
    keys = list(losses.keys())
    values = [
        losses[key].detach().to(dtype=torch.float32).reshape(())
        for key in keys
    ]
    packed = torch.stack(values)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    packed = packed / float(config.world_size)
    return {
        key: packed[index].to(device=losses[key].device, dtype=losses[key].dtype)
        for index, key in enumerate(keys)
    }


def _training_loss_metrics_if_due(
    losses: dict[str, torch.Tensor],
    config: _DistributedConfig,
    *,
    due: bool,
    globally_reduced: bool = False,
) -> dict[str, float] | None:
    """Materialize globally averaged training metrics only for real consumers."""
    if not due:
        return None
    reduced = losses if globally_reduced else _distributed_mean_loss_tensors(losses, config)
    return _losses_to_float_dict(reduced)


def _distributed_training_batch_index(step: int, config: _DistributedConfig) -> int:
    if int(step) <= 0:
        raise ValueError("training step must be positive")
    return (int(step) - 1) * int(config.world_size) + int(config.rank)


def _distributed_training_sample_index(
    step: int,
    *,
    batch_size: int,
    config: _DistributedConfig,
) -> int:
    return _distributed_training_batch_index(step, config) * int(batch_size)


def _forward_loss(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    conditioned_loss_options: dict[str, float] | None = None,
    backward: bool,
) -> dict[str, float]:
    losses = _forward_loss_tensors(
        model,
        batch,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
        conditioned_loss_options=conditioned_loss_options,
        training_loss=backward,
    )
    if backward:
        losses["total"].backward()
    return _losses_to_float_dict(losses)


@refinement_session
def run_training(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    test_hang_diagnostics: bool = False,
    manager_socket: str | None = None,
    unmanaged: bool = False,
) -> None:
    if unmanaged and manager_socket is not None:
        raise ValueError("--unmanaged cannot be combined with --manager-socket")
    validate_target_only_loss_config(_load_raw_config(config_path))
    if not unmanaged:
        _require_single_process_cli_mode("cooperative training (do not use torchrun)")
        from .cooperative_training import run_cooperative_training
        return run_cooperative_training(config_path, socket=manager_socket,
            resume_checkpoint=resume_checkpoint, test_hang_diagnostics=test_hang_diagnostics)
    # The normal launcher and cooperative cohort must execute the same training
    # implementation. Exhaustion preserves the original uninterrupted behavior.
    units = training_units(config_path, resume_checkpoint=resume_checkpoint,
                           test_hang_diagnostics=test_hang_diagnostics)
    try:
        for _boundary in units:
            pass
    finally:
        units.close()


def training_units(
    config_path: str | Path,
    *,
    resume_checkpoint: str | Path | None = None,
    test_hang_diagnostics: bool = False,
    handoff: dict[str, Any] | None = None,
    physical_microbatch_sizes: tuple[int, ...] | None = None,
    batch_source: Any | None = None,
    summary_writer_factory: Any | None = None,
    distributed_init_method: str | None = None,
    training_overrides: dict[str, Any] | None = None,
):
    """Shared training loop, yielding only after completed logical work.

    Closing the iterator runs the same loader/writer/process-group cleanup as
    normal completion. Cooperative workers advance one logical work unit, then
    wait for permission; ordinary training exhausts this same iterator.
    """
    raw_config = _load_raw_config(config_path)
    validate_target_only_loss_config(raw_config)
    source_fingerprint = config_fingerprint(raw_config)
    if training_overrides is not None:
        raw_config = {**raw_config, "training": {**raw_config.get("training", {}), **training_overrides}}
    loader_config = load_config(config_path)
    training = dict(raw_config.get("training", {}))
    refinement = RefinementObserver(RefinementConfig.parse(raw_config))
    if refinement.config.enabled and raw_config.get("historical_test_datasets"):
        raise ValueError("disable historical_test_datasets when refining train/test ground truth")
    if refinement.config.enabled and training.get("test_output_upsample"):
        raise ValueError("refinement requires native output-lattice testing (no test_output_upsample)")
    evaluation_microbatch_size = training.get("evaluation_microbatch_size")
    if evaluation_microbatch_size is not None and (
        type(evaluation_microbatch_size) is not int or evaluation_microbatch_size < 1
    ):
        raise ValueError("training.evaluation_microbatch_size must be a positive integer")
    bounded_accumulation = training.get("bounded_gradient_accumulation", False)
    if type(bounded_accumulation) is not bool:
        raise ValueError("training.bounded_gradient_accumulation must be a boolean")
    if raw_config.get("embedding", {}).get("paired_patches", False) and not bounded_accumulation:
        raise ValueError("paired embedding training requires bounded_gradient_accumulation to keep pairs together")
    profile_loader = loader_profile_enabled(training)
    profile_replay_batch = replay_batch_enabled(training)
    profile_replay_cpu_batch = replay_cpu_batch_enabled(training)
    if profile_replay_batch and profile_replay_cpu_batch:
        raise ValueError(
            "profile_replay_batch and profile_replay_cpu_batch are mutually exclusive"
        )
    cuda_batch_depth = _training_cuda_batch_pipeline_depth(training)
    cuda_value_augmentation = _training_cuda_value_augmentation(training)
    if cuda_value_augmentation is not None and cuda_batch_depth <= 0:
        raise ValueError(
            "sample_pipeline_cuda_value_augmentation requires "
            "sample_pipeline_cuda_prefetch_depth > 0"
        )
    if cuda_batch_depth > 0:
        if profile_replay_batch or profile_replay_cpu_batch:
            raise ValueError("CUDA batch preparation is incompatible with replay modes")
        if _training_sample_pipeline_workers(training) <= 0:
            raise ValueError(
                "CUDA batch preparation requires training.sample_pipeline_workers > 0"
            )
    if (
        profile_loader
        and _loader_worker_count(raw_config) <= 0
        and _training_sample_pipeline_workers(training) <= 0
    ):
        raise ValueError(
            "profile_loader requires loader_workers or sample_pipeline_workers > 0"
        )
    (
        test_hang_diagnostics_config,
        test_hang_diagnostics_cli,
        test_hang_diagnostics_enabled,
    ) = _resolve_test_hang_diagnostics_enabled(
        training,
        cli_enabled=test_hang_diagnostics,
    )
    base_device = _device_from_training(training)
    distributed = _distributed_config_from_env(base_device)
    _distributed_init(distributed, init_method=distributed_init_method)
    device = distributed.device
    if physical_microbatch_sizes is not None:
        if not bounded_accumulation or len(physical_microbatch_sizes) != distributed.world_size:
            raise ValueError("physical microbatches require bounded accumulation and one size per rank")
        if any(type(size) is not int or size < 1 for size in physical_microbatch_sizes):
            raise ValueError("physical microbatch sizes must be positive integers")
        loader_config = replace(loader_config, batch_size=physical_microbatch_sizes[distributed.rank],
                                volume_io_threads=1, volume_decode_threads=1, volume_sample_threads=1)
    if cuda_batch_depth > 0 and device.type != "cuda":
        raise ValueError("CUDA batch preparation requires a CUDA training device")
    writer = None
    diagnostics: _TrainingHangDiagnostics | None = None
    train_dataloader = None
    train_iterator = None
    test_dataloader = None
    loader = FiberTrace3DLoader(loader_config)
    test_loader = None
    if raw_config.get("test_datasets"):
        test_raw = _make_test_loader_raw_config(raw_config, training)
        test_config = config_from_mapping(
            test_raw,
            config_dir=Path(config_path).resolve().parent,
        )
        if physical_microbatch_sizes is not None:
            test_config = replace(test_config, volume_io_threads=1, volume_decode_threads=1, volume_sample_threads=1)
        test_loader = FiberTrace3DLoader(test_config)
    historical_loader = None
    historical_raw = historical_test_config(raw_config)
    if historical_raw is not None:
        if test_loader is None:
            raise ValueError("historical tests require the regular test_datasets as well")
        historical_loader = FiberTrace3DLoader(config_from_mapping(
            historical_raw, config_dir=Path(config_path).resolve().parent))
    trace2cp_cfg = _trace2cp_3d_config(raw_config)
    trace2cp_loader = (
        _make_trace2cp_geometry_loader(raw_config, trace2cp_cfg)
        if distributed.is_main and trace2cp_cfg.enabled
        else None
    )

    if "initialization_seed" in training:
        torch.manual_seed(int(training["initialization_seed"]))
    raw_model = build_fiber_trace_3d_model(raw_config)
    if (
        not _model_uses_conditioned_decoder(raw_model)
        and int(getattr(raw_model, "direction_branch_count", 1)) != 1
    ):
        raise ValueError("Ordinary multi-branch training is temporarily unsupported; "
                         "TODO: implement target-only loss normalization before re-enabling them.")
    sync_batchnorm = _distributed_should_use_sync_batchnorm(
        distributed,
        enabled=training.get("sync_batchnorm", True),
    )
    if sync_batchnorm:
        raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
    normalization_module_counts = _normalization_module_counts(raw_model)
    if profile_replay_batch or profile_replay_cpu_batch:
        _validate_profile_normalization_modules(
            raw_config.get("model_3d", raw_config.get("model", {})).get(
                "normalization", "batch"
            ),
            sync_batchnorm=sync_batchnorm,
            counts=normalization_module_counts,
        )
    raw_model = raw_model.to(device)
    mixed_precision = _mixed_precision_config_from_training(training, device)
    grad_scaler = _make_grad_scaler(mixed_precision)
    optimizer_hparams = _optimizer_hparams_from_training(training)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        **optimizer_hparams,
    )
    resume, initialize_from = _checkpoint_sources(raw_config, resume_checkpoint)
    start_step = 0
    progress_metadata: dict[str, Any] = {}
    handoff_progress = None
    if handoff is not None:
        if not bounded_accumulation:
            raise ValueError("RAM handoff requires bounded_gradient_accumulation")
        handoff_progress = restore_handoff(
            handoff, model=raw_model, optimizer=optimizer, grad_scaler=grad_scaler,
            cuda_device=device if device.type == "cuda" else None,
        )
        if handoff_progress.get("config_fingerprint") != source_fingerprint:
            raise ValueError("training config changed since the captured cohort state")
        start_step = int(handoff_progress["step"])
        progress_metadata = dict(handoff_progress)
    elif resume:
        start_step = _load_snapshot(
            resume,
            model=raw_model,
            optimizer=optimizer,
            optimizer_hparams=optimizer_hparams,
            grad_scaler=grad_scaler,
            map_location=device,
            progress_metadata=progress_metadata,
        )
    elif initialize_from:
        _load_snapshot(
            initialize_from,
            model=raw_model,
            optimizer=None,
            map_location=device,
        )
        if distributed.is_main:
            print(
                "fiber_trace_3d initialization: "
                f"checkpoint={initialize_from} weights_only=true start_step=0",
                flush=True,
            )
    optimizer_lr_group_scales = _optimizer_learning_rate_scales(optimizer)

    ddp_broadcast_buffers = training.get("ddp_broadcast_buffers", True)
    if bounded_accumulation:
        validate_sample_independent_model(raw_model)
        if ddp_broadcast_buffers:
            raise ValueError("bounded accumulation requires training.ddp_broadcast_buffers=false")
        distributed_model = raw_model
        # No DDP constructor/hooks: synchronize initial model only, then use
        # the component SUM reductions at each effective-update boundary.
        if distributed.enabled:
            for tensor in raw_model.state_dict().values():
                dist.broadcast(tensor, src=0)
    else:
        distributed_model = _wrap_distributed_model(
            raw_model,
            distributed,
            broadcast_buffers=ddp_broadcast_buffers,
        )
    train_model: torch.nn.Module | None = None
    ema = ModelEMA(raw_model, training.get("ema_decays", ()), start_step=start_step)
    saved_ema = (handoff or {}).get("ema", progress_metadata.pop("ema", None))
    if saved_ema is not None:
        ema.load_state_dict(saved_ema)
    elif ema.tracks and distributed.is_main:
        print(f"fiber EMA: initialized at update={start_step} decays={training['ema_decays']} (no earlier EMA history)", flush=True)
    del saved_ema
    run_datestr = str(
        (handoff_progress or {}).get("run_datestr")
        or training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_datestr = _distributed_broadcast_object(
        run_datestr if distributed.is_main else None,
        distributed,
    )
    run_dir, snapshot_dir = _resolve_run_layout(
        raw_config,
        date_str_override=str(run_datestr),
    )
    watchdog_seconds = _resolve_test_watchdog_seconds(
        training,
        diagnostics_enabled=test_hang_diagnostics_enabled,
    )
    diagnostics = _TrainingHangDiagnostics(
        run_dir,
        rank=distributed.rank,
        device=device,
        watchdog_seconds=watchdog_seconds,
        automatic_watchdog=distributed.is_main,
        enabled=test_hang_diagnostics_enabled,
    )
    if distributed.is_main:
        if test_hang_diagnostics_enabled:
            print(
                "fiber_trace_3d diagnostics: enabled=true "
                f"pid={os.getpid()} log={diagnostics.path} "
                f"test_watchdog_seconds={watchdog_seconds:g} "
                "manual_dump='kill -USR2 <pid>'",
                flush=True,
            )
        else:
            print("fiber_trace_3d diagnostics: enabled=false", flush=True)
    _distributed_barrier(distributed)
    effective_config = _json_safe(raw_config)
    effective_training = effective_config.setdefault("training", {})
    if historical_loader is not None:
        effective_training["historical_evaluation_contract"] = {
            "sampling": "control_points", "logical_batch_size": 8,
            "output_upsample": "trilinear", "presence_region": "historical_central_patch",
            "source_validity": "modern_backend_no_historical_availability_mask",
            "observation_only": True,
        }
        if distributed.is_main:
            print("fiber historical test: CP/full-grid/central presence region; observation only. "
                  "Modern source loading does not reproduce historical missing-data validity masks.", flush=True)
    effective_training["test_hang_diagnostics_enabled_config"] = (
        test_hang_diagnostics_config
    )
    effective_training["test_hang_diagnostics_enabled_cli"] = test_hang_diagnostics_cli
    effective_training["test_hang_diagnostics_enabled_effective"] = (
        test_hang_diagnostics_enabled
    )
    if resume_checkpoint is not None:
        effective_config.setdefault("training", {})["resume_cli"] = str(resume_checkpoint)
        effective_config.setdefault("training", {})["resume_effective"] = str(resume)
    effective_optimizer = {
        "param_group_lrs": [float(group.get("lr", math.nan)) for group in optimizer.param_groups],
        "param_group_weight_decays": [
            float(group.get("weight_decay", math.nan)) for group in optimizer.param_groups
        ],
    }
    effective_precision = {
        "mode": mixed_precision.mode,
        "enabled": mixed_precision.enabled,
        "device_type": mixed_precision.device_type,
        "dtype": None if mixed_precision.dtype is None else str(mixed_precision.dtype),
        "grad_scaler": _grad_scaler_enabled(grad_scaler),
    }
    writer = (summary_writer_factory or _make_summary_writer)(
        run_dir,
        enabled=distributed.is_main and bool(training.get("tensorboard_enabled", True)),
    )
    if writer is not None and handoff is None:
        writer.add_text("config/json", json.dumps(effective_config, indent=2, sort_keys=True), 0)
        if resume:
            writer.add_text("config/resume", f"resume={resume}\ncheckpoint_step={start_step}", 0)
        writer.add_text(
            "config/optimizer",
            json.dumps(effective_optimizer, indent=2, sort_keys=True),
            0,
        )
        writer.add_text(
            "config/mixed_precision",
            json.dumps(effective_precision, indent=2, sort_keys=True),
            0,
        )
        writer.add_text(
            "train_sample_3d/layout",
            "Rows: yx, zx, zy principal slices through the sampled CP, "
            "then GT-tangent and GT-perpendicular oblique slices. "
            "Single-output columns: image with projected GT line and predicted CP direction "
            "where applicable, target/context presence, raw prediction presence, "
            "prediction presence times abs dot with the slice normal, and prediction presence "
            "times abs dot with the GT tangent. Multi-output columns instead show first, "
            "second, normal-closest, other, max, min, and average prediction presence. "
            "In conditioned mode the first prediction is zero-query output and the second is "
            "the random-query output from the same loss forward. "
            "Multiple batch samples are concatenated side by side when configured.",
            0,
        )
        writer.add_text(
            "test_sample_3d/layout",
            "Rows: yx, zx, zy principal slices through the sampled test CP, "
            "then GT-tangent and GT-perpendicular oblique slices. "
            "Single-output columns: image with projected GT line and predicted CP direction "
            "where applicable, target/context presence, raw prediction presence, "
            "prediction presence times abs dot with the slice normal, and prediction presence "
            "times abs dot with the GT tangent. Multi-output columns instead show first, "
            "second, normal-closest, other, max, min, and average prediction presence. "
            "In conditioned mode the first prediction is zero-query output and the second is "
            "the random-query output from the same loss forward. "
            "Multiple batch samples are concatenated side by side when configured.",
            0,
        )

    max_steps_raw = int(training.get("max_steps", 1))
    if max_steps_raw < 0:
        raise ValueError("training.max_steps must be >= 0")
    max_steps: int | None = None if max_steps_raw == 0 else max_steps_raw
    if max_steps is not None and (max_steps < int(start_step) or max_steps == int(start_step) and handoff is None):
        raise ValueError(
            "training.max_steps must be greater than checkpoint step when resuming: "
            f"max_steps={max_steps} checkpoint_step={start_step}"
        )
    sample_index_limit = _training_sample_index_limit(training, loader.sample_count)
    optimizer_global_batch_size = int(
        training.get(
            "optimizer_global_batch_size",
            loader.config.batch_size * distributed.world_size,
        )
    )
    if handoff_progress is not None and (
        handoff_progress["effective_batch_size"] != optimizer_global_batch_size
        or handoff_progress["sample_frontier"] != start_step * optimizer_global_batch_size
    ):
        raise ValueError("handoff canonical sample frontier/effective batch mismatch")
    bounded_index = None
    if bounded_accumulation:
        sizes = physical_microbatch_sizes or (loader.config.batch_size,) * distributed.world_size
        bounded_index = EffectiveBatchSampleIndex(
            optimizer_global_batch_size, tuple(sizes), distributed.rank, int(start_step),
            2 if loader.config.embedding.paired_patches else 1,
        )
        optimizer_virtual_world_size = distributed.world_size
        optimizer_microbatch_takes = bounded_index.batch_sizes
    else:
        optimizer_virtual_world_size = _optimizer_virtual_world_size(
            local_batch_size=loader.config.batch_size,
            physical_world_size=distributed.world_size,
            optimizer_global_batch_size=optimizer_global_batch_size,
        )
        optimizer_microbatch_takes = _optimizer_microbatch_takes(
            local_batch_size=loader.config.batch_size,
            world_size=distributed.world_size,
            rank=distributed.rank,
            optimizer_global_batch_size=optimizer_global_batch_size,
        )
    sample_progress = _SampleProgress(
        start_update=start_step,
        start_samples=_resume_sample_count(
            progress_metadata, start_step,
            training.get("resume_samples_seen") if resume and handoff_progress is None else None,
        ),
        global_batch_size=optimizer_global_batch_size,
    )
    if writer is not None:
        writer = _SampleAxisWriter(writer, sample_progress)
        writer.add_text("config/progress_axis", "Cumulative global training samples; schedules use optimizer updates.", 0)
    optimizer_microbatch_count = len(optimizer_microbatch_takes)
    if loader.config.embedding.paired_patches and (optimizer_global_batch_size % 2 or
            any(t % 2 for t in optimizer_microbatch_takes)):
        raise ValueError("paired training requires pair-aligned microbatches")
    manual_gradient_sync = bounded_accumulation or optimizer_virtual_world_size != distributed.world_size
    if manual_gradient_sync and ddp_broadcast_buffers:
        raise ValueError(
            "optimizer_global_batch_size simulation requires "
            "training.ddp_broadcast_buffers=false because physical ranks may "
            "execute different numbers of logical batches"
        )
    if manual_gradient_sync and (
        int(normalization_module_counts.get("BatchNorm3d", 0)) > 0
        or int(normalization_module_counts.get("SyncBatchNorm", 0)) > 0
    ):
        raise ValueError(
            "optimizer_global_batch_size simulation requires stateless "
            "normalization such as GroupNorm"
        )
    train_model = compile_training_model(
        raw_model if manual_gradient_sync else distributed_model,
        str(training.get("compile_mode", "off")),
    )
    if (profile_replay_batch or profile_replay_cpu_batch) and optimizer_microbatch_count != 1:
        raise ValueError("profile replay modes do not support optimizer microbatching")
    scalar_interval = int(training.get("scalar_log_interval", 100))
    initial_scalar_log_steps = int(training.get("initial_scalar_log_steps", 100))
    gradient_norm_interval = int(training.get("gradient_norm_log_interval", 0))
    gradient_clip_raw = training.get("gradient_clip_max_norm")
    gradient_clip_max_norm = (
        None
        if gradient_clip_raw is None or float(gradient_clip_raw) == 0.0
        else float(gradient_clip_raw)
    )
    lr_range_test = _lr_range_test_config(training)
    learning_rate_warmup_steps = int(
        training.get("learning_rate_warmup_steps", 0)
    )
    force_scalar_reduce = training.get("profile_force_scalar_reduce_every_step", False)
    if scalar_interval <= 0:
        raise ValueError("training.scalar_log_interval must be > 0")
    if initial_scalar_log_steps < 0:
        raise ValueError("training.initial_scalar_log_steps must be >= 0")
    if gradient_norm_interval < 0:
        raise ValueError("training.gradient_norm_log_interval must be >= 0")
    if learning_rate_warmup_steps < 0:
        raise ValueError("training.learning_rate_warmup_steps must be >= 0")
    if gradient_clip_max_norm is not None and (
        not math.isfinite(gradient_clip_max_norm) or gradient_clip_max_norm <= 0.0
    ):
        raise ValueError("training.gradient_clip_max_norm must be finite and positive")
    if not isinstance(force_scalar_reduce, bool):
        raise ValueError(
            "training.profile_force_scalar_reduce_every_step must be a JSON boolean"
        )
    if force_scalar_reduce and not bool(training.get("profile_enabled", False)):
        raise ValueError(
            "training.profile_force_scalar_reduce_every_step requires profile_enabled"
        )
    checkpoint_interval = int(training.get("checkpoint_interval", 100))
    kept_snapshot_interval = int(training.get("kept_snapshot_interval", 10000))
    test_interval = int(training.get("test_interval", 0))
    _validate_snapshot_intervals(
        checkpoint_interval=checkpoint_interval,
        kept_snapshot_interval=kept_snapshot_interval,
        test_interval=test_interval,
    )
    sample_vis_interval = int(
        training.get("sample_vis_interval", training.get("train_sample_vis_interval", 1000))
    )
    sample_vis_count = int(
        training.get("sample_vis_count", training.get("train_sample_vis_count", 4))
    )
    if sample_vis_count <= 0:
        raise ValueError("training.sample_vis_count must be > 0")
    if lr_range_test is not None:
        expected_steps = lr_range_test.warmup_steps + lr_range_test.range_steps
        if int(start_step) != 0:
            raise ValueError("training.lr_range_test requires fresh initialization")
        if max_steps != expected_steps:
            raise ValueError(
                "training.max_steps must equal lr_range_test warmup_steps + "
                f"range_steps ({expected_steps})"
            )
        if learning_rate_warmup_steps:
            raise ValueError(
                "training.learning_rate_warmup_steps cannot be combined with "
                "training.lr_range_test"
            )
    test_sample_vis_interval = int(
        training.get(
            "test_sample_vis_interval",
            test_interval if test_interval > 0 else sample_vis_interval,
        )
    )
    test_sample_vis_count = int(training.get("test_sample_vis_count", sample_vis_count))
    if test_sample_vis_count <= 0:
        raise ValueError("training.test_sample_vis_count must be > 0")
    test_control_points, test_start_sample_index, test_sample_mode = _resolve_dense_test_selection(
        training,
        loader_sample_count=test_loader.sample_count if test_loader is not None else loader.sample_count,
        default_count=0,
    )
    direction_weight = float(training.get("direction_weight", 10.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    conditioned_loss_options = _conditioned_loss_options_from_training(training)
    loader_workers = _loader_worker_count(raw_config)
    loader_prefetch_factor = _loader_prefetch_factor(raw_config)
    loader_worker_device = _loader_worker_device(raw_config)
    loader_context = _loader_multiprocessing_context(raw_config)
    test_batch_indices = (
        _distributed_test_batch_indices(
            sample_count=test_control_points,
            batch_size=test_loader.config.batch_size,
            config=distributed,
        )
        if test_loader is not None and test_interval > 0
        else ()
    )
    # One source-tagged worker pool services training and evaluation. Transport
    # ordinals are independent of canonical dataset/RNG sample indices.
    sample_queue = None
    queued_train = None
    remaining_steps = None if max_steps is None else max(0, int(max_steps) - int(start_step))
    pipeline_batch_count = (1 if profile_replay_batch or profile_replay_cpu_batch else
                            (remaining_steps * optimizer_microbatch_count if remaining_steps is not None else None))
    sample_pipeline_workers = _training_sample_pipeline_workers(training)
    if batch_source is None:
        if pipeline_batch_count is None:
            raise ValueError("queued training requires a finite training.max_steps")
        local_workers = max(1, _rank_partition(
            sample_pipeline_workers or max(distributed.world_size, loader_workers),
            world_size=distributed.world_size, rank=distributed.rank))
        local_slots = max(loader.config.batch_size,
                          test_loader.config.batch_size if test_loader is not None else 1,
                          _rank_partition(int(training.get("sample_pipeline_slots", 512)),
                                          world_size=distributed.world_size, rank=distributed.rank))
        sources = {"train": SampleQueueSource(loader.config, sample_mode="random",
                    sample_index_limit=sample_index_limit,
                    defer_cuda_value_augmentation=cuda_value_augmentation)}
        if test_loader is not None:
            sources["test"] = SampleQueueSource(test_loader.config, sample_mode=test_sample_mode)
        if historical_loader is not None:
            sources["historical"] = SampleQueueSource(historical_loader.config, sample_mode="random")
        sample_queue = FiberTrace3DSampleQueue(
            sources=sources, workers=local_workers, slot_count=local_slots,
            initial_in_flight=min(local_slots, max(1, int(training.get("sample_pipeline_initial_in_flight", 16)))),
            load_affinity_workers=max(0, local_workers // 2), profile=profile_loader)
        index_map = bounded_index or VirtualBatchSampleIndex(
            batch_size=loader.config.batch_size, physical_world_size=distributed.world_size,
            rank=distributed.rank, virtual_world_size=optimizer_virtual_world_size, start_step=int(start_step))
        queued_train = QueuedBatchView(sample_queue, "train", TrainingBatchIndices(
            index_map, optimizer_microbatch_takes, pipeline_batch_count), limit=0)

    def admit_training_until_test(completed_step):
        if queued_train is not None:
            next_test = ((completed_step // test_interval + 1) * test_interval
                         if test_interval > 0 else max_steps)
            queued_train.set_limit(min(pipeline_batch_count,
                max(0, next_test - start_step) * optimizer_microbatch_count))

    def queued_evaluation(indices, at_step, *, source="test", evaluation_loader=None, count=None, offset=None):
        evaluation_loader = evaluation_loader or test_loader
        count = test_control_points if count is None else count
        offset = test_start_sample_index if offset is None else offset
        batches = queued_test_batches(indices, batch_size=evaluation_loader.config.batch_size,
            sample_count=count, start_sample_index=offset)
        if sample_queue is not None:
            return QueuedBatchView(sample_queue, source, batches)
        return batch_source.test_batches(step=at_step, global_batch_indices=indices,
            batch_size=evaluation_loader.config.batch_size, sample_count=count,
            start_sample_index=offset)
    best_metric = float((handoff_progress or {}).get("best_metric", math.inf))
    pending_test = (handoff_progress or {}).get("pending_test")
    cuda_batch_pipeline: FiberTrace3DCudaBatchPipeline | None = None
    step = int(start_step)

    def capture_boundary():
        if cuda_batch_pipeline is not None:
            cuda_batch_pipeline.pause()
        _synchronize_device(device)
        if writer is not None:
            writer.flush()
        return capture_handoff(
            model=raw_model, optimizer=optimizer, grad_scaler=grad_scaler,
            ema=ema,
            cuda_device=device if device.type == "cuda" else None,
            progress={"step": step, "effective_batch_size": optimizer_global_batch_size,
                      "sample_frontier": step * optimizer_global_batch_size,
                      "samples_seen": sample_progress.at(step),
                      "optimizer_global_batch_size": optimizer_global_batch_size,
                      "best_metric": best_metric, "run_datestr": str(run_datestr),
                      "pending_test": pending_test, "config_fingerprint": source_fingerprint},
        )

    ema_test_metrics = {}

    def save_test_snapshots(test_step, metric, metric_name, *, initial=False):
        nonlocal best_metric
        if metric is None:
            if refinement.config.update_enabled and refinement.config.enabled and not initial and test_step % checkpoint_interval == 0:
                _distributed_barrier(distributed)
                if distributed.is_main:
                    saved = active_store().save(snapshot_dir / f"fibers_step_{test_step:08d}.zip")
                    for name in [None, *ema.tracks]:
                        suffix = "" if name is None else f"_{name}"
                        filenames = [f"current{suffix}.pt"]
                        if kept_snapshot_interval > 0 and test_step % kept_snapshot_interval == 0:
                            filenames.append(f"step_{test_step:08d}{suffix}.pt")
                        for filename in filenames:
                            _save_snapshot(snapshot_dir / filename, model=raw_model, optimizer=optimizer,
                                step=test_step, config=raw_config, metric=None, grad_scaler=grad_scaler,
                                sample_progress=sample_progress, ema=ema, ema_name=name, refinement_state=saved)
                _distributed_barrier(distributed)
            return
        planned = []
        tracks = [(None, metric, metric_name, best_metric)]
        tracks.extend((name, ema_test_metrics[name], f"test_{name}/loss_total", track["best_metric"])
                      for name, track in ema.tracks.items())
        for name, value, value_name, previous_best in tracks:
            suffix = "" if name is None else f"_{name}"
            filenames = []
            if not initial:
                if test_step % checkpoint_interval == 0:
                    filenames.append(f"current{suffix}.pt")
                if kept_snapshot_interval > 0 and test_step % kept_snapshot_interval == 0:
                    filenames.append(f"step_{test_step:08d}{suffix}.pt")
            if initial or value < previous_best:
                if name is None:
                    best_metric = float(value)
                else:
                    ema.tracks[name]["best_metric"] = float(value)
                filenames.append(f"best{suffix}.pt")
            planned.extend((filename, name, value, value_name) for filename in filenames)
        # Update every track's best before serializing the resumable raw state.
        saved_refinement = None
        if planned and refinement.config.enabled and refinement.config.update_enabled:
            _distributed_barrier(distributed)
            if distributed.is_main:
                saved_refinement = active_store().save(snapshot_dir / f"fibers_step_{test_step:08d}.zip")
        for filename, name, value, value_name in planned:
            if distributed.is_main:
                _save_snapshot(snapshot_dir / filename, model=raw_model, optimizer=optimizer,
                    step=test_step, config=raw_config, metric=value, metric_name=value_name,
                    grad_scaler=grad_scaler, sample_progress=sample_progress, ema=ema, ema_name=name,
                    refinement_state=saved_refinement)
            _distributed_barrier(distributed)

    if distributed.is_main:
        print(
            "fiber_trace_3d train: "
            f"samples={loader.sample_count} batch_size={loader.config.batch_size} "
            f"global_batch_size={optimizer_global_batch_size} "
            f"virtual_world_size={optimizer_virtual_world_size} "
            f"rank0_microbatches_per_update={optimizer_microbatch_count} "
            f"max_sample_index={sample_index_limit} "
            f"patch_shape_zyx={loader.config.patch_shape_zyx} device={device} run_dir={run_dir} "
            f"ddp_enabled={distributed.enabled} ddp_world_size={distributed.world_size} "
            f"ddp_backend={distributed.backend or 'none'} "
            f"sync_batchnorm={sync_batchnorm} "
            f"ddp_broadcast_buffers={ddp_broadcast_buffers} "
            f"trace2cp_enabled={bool(trace2cp_loader is not None)} "
            f"loader_workers={loader_workers} loader_prefetch_factor={loader_prefetch_factor} "
            f"loader_worker_device={loader_worker_device} "
            f"loader_multiprocessing_context={loader_context or 'default'} "
            f"profile_replay_batch={profile_replay_batch} "
            f"test_loader_workers={loader_workers if test_loader is not None else 0} "
            f"test_global_batches={math.ceil(test_control_points / test_loader.config.batch_size) if test_loader is not None else 0} "
            f"test_batches_per_rank={len(test_batch_indices)} "
            f"optimizer_lr={effective_optimizer['param_group_lrs']} "
            f"optimizer_weight_decay={effective_optimizer['param_group_weight_decays']} "
            f"mixed_precision={effective_precision['mode']} "
            f"autocast_enabled={effective_precision['enabled']} "
            f"amp_grad_scaler={effective_precision['grad_scaler']} "
            f"kept_snapshot_interval={kept_snapshot_interval} "
            f"conditioned_decoder={_model_uses_conditioned_decoder(raw_model)} "
            f"conditioned_jitter_deg={conditioned_loss_options['perpendicular_jitter_degrees']} "
            f"conditioned_pos_w={conditioned_loss_options['positive_query_weight']} "
            f"conditioned_neg_w={conditioned_loss_options['negative_query_weight']}",
            flush=True,
        )
    if distributed.is_main and resume:
        print(
            "fiber_trace_3d resume: "
            f"checkpoint={resume} checkpoint_step={start_step} next_step={start_step + 1} "
            f"run_dir={run_dir}",
            flush=True,
        )

    def log_observation(prefix, values, step):
        print(f"{prefix} samples={sample_progress.at(step)} update={step} "
              f"loss_total={values['total']:.6f} angle_mean_deg={values['angle_mean_deg']:.2f} "
              f"{_embedding_log(values, raw_model) if 'embedding' in values else ''}", flush=True)
        if writer is not None:
            _log_test_scalars(writer, prefix, values, raw_model, step)

    def run_configured_tests(step: int, *, initial=False):
        nonlocal pending_test, ema_test_metrics
        assert diagnostics is not None
        test_started = time.monotonic()
        if physical_microbatch_sizes is not None and pending_test is None:
            pending_test = {"step": step, "initial": initial, "rows": [], "trace_rows": [],
                            "dense_logged": False, "elapsed": 0.0}
        diagnostics.arm_test_watchdog(step)
        diagnostics.marker("test-routine-begin", step=step)
        test_refinement_totals = (pending_test.setdefault("refinement_loss_totals", empty_refinement_statistics())
                                  if pending_test is not None else empty_refinement_statistics())
        def collect_evaluation(evaluation_loader, options, *, source, rows_key, visualization=None):
            nonlocal test_started
            count, offset = options["sample_count"], options["start_sample_index"]
            total = math.ceil(count / evaluation_loader.config.batch_size)
            completed = set()
            if pending_test is not None:
                if pending_test["step"] != step:
                    raise ValueError("pending test belongs to a different optimizer step")
                completed = {int(index) for index, _ in pending_test.setdefault(rows_key, [])}
            remaining = [index for index in range(total) if index not in completed]
            if remaining and batch_source is not None and source != "test":
                batch_source.select_test_source(step, source)
            wave_size = distributed.world_size if pending_test is not None else max(1, total)
            all_rows = []
            for start in range(0, len(remaining), wave_size):
                indices, padded = _distributed_evaluation_wave_indices(
                    remaining,
                    start=start,
                    wave_size=wave_size,
                    rank=distributed.rank,
                    world_size=distributed.world_size,
                    total_batch_count=total,
                    pad_empty_ranks=pending_test is not None,
                )
                iterator = queued_evaluation(indices, step, source=source,
                    evaluation_loader=evaluation_loader, count=count, offset=offset)
                try:
                    local_options = (
                        {**options, "prediction_observer": None}
                        if padded
                        else options
                    )
                    local = _evaluate_dense_loss_rows(raw_model, evaluation_loader, **local_options,
                        ema=ema, global_batch_indices=indices, batch_iterator=iterator,
                        microbatch_size=(min(loader.config.batch_size,
                            evaluation_microbatch_size or loader.config.batch_size)
                            if pending_test is not None else evaluation_microbatch_size))
                    if padded:
                        local = []
                finally:
                    iterator.close()
                refinement_stats = empty_refinement_statistics()
                if source == "test" and refinement.config.enabled and refinement.config.update_enabled:
                    # Raw and all EMA tracks have now consumed identical batch
                    # targets. Commit before a possible managed yield/restart.
                    refinement_stats = refinement.flush(distributed, phase="test", step=step,
                        batch=f"{remaining[start]}:{remaining[min(start+wave_size,len(remaining))-1]}")
                rows = _distributed_gather_dense_test_rows(local, distributed)
                if pending_test is None:
                    test_refinement_totals[:] = [a+b for a,b in zip(test_refinement_totals,refinement_stats)]
                    all_rows.extend(rows)
                    continue
                if source == "test" and refinement.config.enabled and refinement.config.update_enabled:
                    rows, refinement_stats = _distributed_broadcast_object(
                        (rows,refinement_stats) if distributed.is_main else None, distributed)
                    test_refinement_totals[:] = [a+b for a,b in zip(test_refinement_totals,refinement_stats)]
                else:
                    rows = _distributed_broadcast_object(rows if distributed.is_main else None, distributed)
                pending_test[rows_key].extend(rows)
                if visualization is not None:
                    sheets = _distributed_broadcast_object(
                        visualization.sheets if distributed.is_main else None, distributed)
                    pending_test["visualization_sheets"] = sheets
                    visualization.sheets = sheets
                pending_test["elapsed"] += _distributed_broadcast_object(
                    time.monotonic() - test_started if distributed.is_main else None, distributed)
                yield TrainingBoundary(step, False, capture_boundary, "test")
                test_started = time.monotonic()
            return pending_test[rows_key] if pending_test is not None else all_rows

        try:
            if batch_source is not None:
                batch_source.begin_test(step)
            _diagnostic_cuda_synchronize(
                diagnostics,
                phase="test-entry-cuda-sync",
                step=step,
                device=device,
            )
            metric: float | None = None
            metric_name: str | None = None
            test_vis_due = test_sample_vis_interval > 0 and step % test_sample_vis_interval == 0
            test_sheets = pending_test.setdefault("visualization_sheets", []) if pending_test is not None else []
            test_visualization = _ForwardVisualization(test_sample_vis_count, test_sheets)
            if test_loader is not None and test_interval > 0:
                options = dict(
                    device=device, start_sample_index=test_start_sample_index,
                    sample_count=test_control_points, sample_mode=test_sample_mode,
                    direction_weight=direction_weight, presence_weight=presence_weight,
                    conditioned_loss_options=conditioned_loss_options, mixed_precision=mixed_precision,
                    diagnostics=diagnostics, diagnostic_step=step,
                    output_upsample=training.get("test_output_upsample"),
                    prediction_observer=_combine_prediction_observers(
                        refinement if refinement.config.enabled and refinement.config.update_enabled else None,
                        test_visualization if distributed.is_main and writer is not None and test_vis_due else None),
                )
                gathered_rows = yield from collect_evaluation(
                    test_loader, options, source="test", rows_key="rows",
                    visualization=test_visualization if test_vis_due else None)
                if distributed.is_main:
                    test_losses = _mean_dense_test_rows(gathered_rows)
                    metric = float(test_losses["total"])
                    metric_name = "test/loss_total"
                    if pending_test is None or not pending_test["dense_logged"]:
                        print(
                            f"test samples={sample_progress.at(step)} update={step} loss_total={test_losses['total']:.6f} "
                            f"loss_direction={test_losses['direction']:.6f} "
                            f"loss_presence={test_losses['presence']:.6f} "
                            f"{_embedding_log(test_losses, raw_model)}"
                            f"angle_mean_deg={test_losses['angle_mean_deg']:.2f} "
                            f"branch0={test_losses['branch0_fraction']:.3f} "
                            f"branch1={test_losses['branch1_fraction']:.3f} "
                            f"selected_score={test_losses['selected_score_mean']:.3f}",
                            flush=True,
                        )
                    if writer is not None and (pending_test is None or not pending_test["dense_logged"]):
                        diagnostics.marker("test-tensorboard-scalars-begin", step=step)
                        _log_test_scalars(writer, "test", test_losses, raw_model, step)
                        log_refinement_scalars(writer,"test",test_refinement_totals,step)
                        diagnostics.marker("test-tensorboard-scalars-end", step=step)
                if (
                    distributed.is_main
                    and writer is not None
                    and test_sample_vis_interval > 0
                    and step % test_sample_vis_interval == 0
                    and (pending_test is None or not pending_test["dense_logged"])
                ):
                    test_visualization.write(writer, "test_sample_3d/principal_slices", step)
                ema_test_metrics = _distributed_broadcast_object(
                    {name: test_losses[f"{name}/total"] for name in ema.tracks}
                    if distributed.is_main else None, distributed)
                if distributed.is_main and (pending_test is None or not pending_test["dense_logged"]):
                    for name in ema.tracks:
                        values = {key.split("/", 1)[1]: value for key, value in test_losses.items()
                                  if key.startswith(name + "/")}
                        log_observation(f"test_{name}", values, step)
            if pending_test is not None:
                pending_test["dense_logged"] = True
            if historical_loader is not None and test_interval > 0:
                historical_options = dict(
                    device=device, start_sample_index=0, sample_count=historical_loader.sample_count,
                    sample_mode="random", direction_weight=direction_weight, presence_weight=presence_weight,
                    mixed_precision=mixed_precision, diagnostics=diagnostics, diagnostic_step=step,
                    output_upsample="trilinear", historical=True)
                historical_rows = yield from collect_evaluation(historical_loader,
                    historical_options, source="historical", rows_key="historical_rows")
                if distributed.is_main and (pending_test is None or not pending_test.get("historical_logged")):
                    historical_losses = _mean_dense_test_rows(historical_rows)
                    log_observation("test_historical", historical_losses, step)
                    for name in ema.tracks:
                        values = {key.split("/", 1)[1]: value for key, value in historical_losses.items()
                                  if key.startswith(name + "/")}
                        log_observation(f"test_historical_{name}", values, step)
                if pending_test is not None:
                    pending_test["historical_logged"] = True
            trace2cp_metric = None
            if physical_microbatch_sizes is not None and trace2cp_cfg.enabled and test_interval > 0:
                # Trace geometry stays CPU-side; a CP-to-next-CP segment is a
                # resumable unit. Only rank zero performs the existing metric.
                count = _distributed_broadcast_object(
                    max(1, trace2cp_cfg.control_points or trace2cp_loader.sample_count)
                    if distributed.is_main else None, distributed)
                offset = len(pending_test["trace_rows"])
                iterator = (_iter_trace2cp_metric_rows(raw_model, trace2cp_loader,
                    image_normalization=loader.config.image_normalization,
                    cfg=trace2cp_cfg, device=device, mixed_precision=mixed_precision,
                    start_offset=offset) if distributed.is_main else None)
                for _ in range(offset, count):
                    row = next(iterator) if distributed.is_main else None
                    row = _distributed_broadcast_object(row, distributed)
                    pending_test["trace_rows"].append(row)
                    pending_test["elapsed"] += _distributed_broadcast_object(
                        time.monotonic() - test_started if distributed.is_main else None, distributed)
                    yield TrainingBoundary(step, False, capture_boundary, "test")
                    test_started = time.monotonic()
                trace2cp_metric = _aggregate_trace2cp_metric_rows(pending_test["trace_rows"])
            if distributed.is_main and trace2cp_loader is not None and test_interval > 0:
                diagnostics.marker("test-trace2cp-begin", step=step)
                if trace2cp_metric is None:
                    trace2cp_metric = _evaluate_trace2cp_metric_fixed_set_3d(
                        raw_model, trace2cp_loader,
                        image_normalization=loader.config.image_normalization,
                        cfg=trace2cp_cfg, device=device, mixed_precision=mixed_precision,
                    )
                _diagnostic_cuda_synchronize(
                    diagnostics,
                    phase="test-trace2cp-cuda-sync",
                    step=step,
                    device=device,
                )
                diagnostics.marker("test-trace2cp-end", step=step)
                print(
                    f"test_trace2cp samples={sample_progress.at(step)} update={step} trace2cp_error={trace2cp_metric.error_mean:.6f} "
                    f"raw_y_error_mean_px={trace2cp_metric.raw_y_error_mean_px:.3f} "
                    f"segments={trace2cp_metric.segments} skipped={trace2cp_metric.skipped_segments}",
                    flush=True,
                )
                if writer is not None:
                    writer.add_scalar("test/trace2cp_error", trace2cp_metric.error_mean, step)
                    writer.add_scalar(
                        "test/trace2cp_raw_y_error_mean_px",
                        trace2cp_metric.raw_y_error_mean_px,
                        step,
                    )
                    writer.add_scalar("test/trace2cp_segments", trace2cp_metric.segments, step)
                    writer.add_scalar(
                        "test/trace2cp_skipped_segments",
                        trace2cp_metric.skipped_segments,
                        step,
                    )
            test_elapsed_seconds = time.monotonic() - test_started + (
                pending_test["elapsed"] if pending_test is not None else 0.0)
            if distributed.is_main:
                _report_test_timing(
                    step=step,
                    total_seconds=test_elapsed_seconds,
                    writer=writer,
                    samples_seen=sample_progress.at(step),
                )
            if writer is not None and metric is not None:
                diagnostics.marker("test-tensorboard-flush-begin", step=step)
                writer.flush()
                diagnostics.marker("test-tensorboard-flush-end", step=step)
            diagnostics.marker(
                "test-routine-end",
                step=step,
                elapsed_s=f"{test_elapsed_seconds:.3f}",
            )
            if batch_source is not None:
                batch_source.end_test(step)
            return metric, metric_name
        finally:
            diagnostics.cancel_test_watchdog(step=step)

    initial_metric, initial_metric_name = None, None
    try:
        if handoff is None or pending_test is not None:
            is_initial = pending_test["initial"] if pending_test is not None else True
            initial_metric, initial_metric_name = yield from run_configured_tests(start_step, initial=is_initial)
            initial_metric = _distributed_broadcast_object(initial_metric, distributed)
            initial_metric_name = _distributed_broadcast_object(initial_metric_name, distributed)
            save_test_snapshots(start_step, initial_metric, initial_metric_name, initial=is_initial)
            pending_test = None
    except BaseException:
        # This can yield before the training prefetcher exists. Closing a
        # cohort here must still release its writer and process group.
        if sample_queue is not None:
            sample_queue.close()
        if writer is not None:
            writer.close()
        diagnostics.close()
        _distributed_cleanup(distributed)
        raise
    _distributed_barrier(distributed)

    if batch_source is not None and not bounded_accumulation:
        raise ValueError("external canonical sample ownership requires bounded accumulation")
    train_iterator = train_dataloader = batch_source if batch_source is not None else queued_train
    admit_training_until_test(start_step)

    target_only_accumulation = bounded_accumulation
    if target_only_accumulation and optimizer_microbatch_count > 0:
        train_iterator = train_dataloader = MacroBatchTargetSource(
            train_iterator, config=loader.config, device=device,
            output_stride=_model_output_stride(raw_model), batch_sizes=optimizer_microbatch_takes,
            conditioned=_model_uses_conditioned_decoder(raw_model))

    cuda_batch_pipeline: FiberTrace3DCudaBatchPipeline | None = None
    if cuda_batch_depth > 0 and optimizer_microbatch_count > 0:
        if train_iterator is None:
            raise RuntimeError("CUDA batch preparation requires a shared sample source")
        cuda_batch_pipeline = FiberTrace3DCudaBatchPipeline(
            train_iterator,
            config=loader.config,
            device=device,
            output_stride=_model_output_stride(raw_model),
            depth=cuda_batch_depth,
            value_augmentation_suffix=cuda_value_augmentation,
        )
        train_iterator = cuda_batch_pipeline
        train_dataloader = cuda_batch_pipeline

    replayed_batch: FiberTrace3DBatch | None = None
    replayed_cpu_batch: FiberTrace3DBatch | None = None
    replayed_batch_sha256: str | None = None
    replayed_source_batch_sha256: str | None = None
    if profile_replay_batch:
        replayed_batch, _load_ms, _wait_ms, _to_device_ms, _target_ms = _next_training_batch(
            iterator=train_iterator,
            loader=loader,
            sample_index=_distributed_training_sample_index(
                int(start_step) + 1,
                batch_size=loader.config.batch_size,
                config=distributed,
            ),
            sample_index_limit=sample_index_limit,
            sample_mode="random",
            device=device,
            output_stride=_model_output_stride(raw_model),
        )
        _shutdown_dataloader_iterator(train_iterator)
        train_iterator = None
        train_dataloader = None
        replayed_batch_sha256 = _batch_tensor_sha256(replayed_batch)
        replayed_source_batch_sha256 = _source_batch_tensor_sha256(replayed_batch)
        _synchronize_device(device)
        _distributed_barrier(distributed)
        if distributed.is_main:
            print(
                "fiber_trace_3d profile replay: captured one GPU-resident batch per rank; "
                f"training loader stopped; rank0_sha256={replayed_batch_sha256} "
                f"source_sha256={replayed_source_batch_sha256}",
                flush=True,
            )
    elif profile_replay_cpu_batch:
        if train_iterator is None:
            raise RuntimeError("CPU replay requires a training DataLoader")
        replayed_cpu_batch = next(train_iterator)
        _shutdown_dataloader_iterator(train_iterator)
        train_iterator = None
        train_dataloader = None
        replayed_source_batch_sha256 = _source_batch_tensor_sha256(replayed_cpu_batch)
        _distributed_barrier(distributed)
        if distributed.is_main:
            print(
                "fiber_trace_3d profile CPU replay: captured one augmented CPU batch "
                "per rank; training loader stopped; "
                f"source_sha256={replayed_source_batch_sha256}",
                flush=True,
            )
    profile = TrainingProfile(training, run_dir, distributed.rank, device)
    throughput_start_step = (
        int(training.get("profile_throughput_start_step", 31))
        if profile.enabled
        else None
    )
    if throughput_start_step is not None and throughput_start_step <= int(start_step):
        raise ValueError("training.profile_throughput_start_step must exceed checkpoint step")
    throughput_started: float | None = None
    throughput_steps = 0

    try:
        step = int(start_step)
        yield TrainingBoundary(step, max_steps is not None and step >= max_steps, capture_boundary)
        while max_steps is None or step < max_steps:
            step += 1
            if throughput_start_step is not None and step == throughput_start_step:
                _distributed_barrier(distributed)
                _synchronize_device(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                throughput_started = time.perf_counter()
            profile.begin(step)
            train_vis_due = sample_vis_interval > 0 and (step == 1 or step % sample_vis_interval == 0)
            train_visualization = _ForwardVisualization(sample_vis_count) if writer is not None and train_vis_due else None
            console_metrics_due = _training_scalar_metrics_due(
                step,
                scalar_interval=scalar_interval,
                initial_steps=(0 if profile_replay_batch else initial_scalar_log_steps),
            )
            writer_metrics_due = bool(training.get("tensorboard_enabled", True)) and (
                step == 1 or step % scalar_interval == 0
            )
            scalar_metrics_due = (
                console_metrics_due or writer_metrics_due or force_scalar_reduce
            )
            learning_rate = float(optimizer.param_groups[0]["lr"])
            lr_range_index: int | None = None
            if lr_range_test is not None:
                learning_rate = _lr_range_learning_rate(lr_range_test, step)
                _set_optimizer_learning_rate(
                    optimizer,
                    learning_rate,
                    group_scales=optimizer_lr_group_scales,
                )
                lr_range_index = step - lr_range_test.warmup_steps - 1
            elif learning_rate_warmup_steps > 0 and step <= learning_rate_warmup_steps:
                learning_rate = _linear_warmup_learning_rate(
                    float(optimizer_hparams["lr"]),
                    step=step,
                    warmup_steps=learning_rate_warmup_steps,
                )
                _set_optimizer_learning_rate(
                    optimizer,
                    learning_rate,
                    group_scales=optimizer_lr_group_scales,
                )
            optimizer.zero_grad(set_to_none=True)
            fw_start: float | None = time.perf_counter() if optimizer_microbatch_count == 0 else None
            scaler_enabled = _grad_scaler_enabled(grad_scaler)
            load_ms = 0.0
            wait_ms = 0.0
            to_device_ms = 0.0
            target_ms = 0.0
            cuda_value_augmentation_ms = 0.0
            batch: FiberTrace3DBatch | None = None
            loss_statistics: _LossStatistics | None = None
            component_accumulator = None
            gaussian_presence = loader.config.presence_target_mode == "anisotropic_gaussian"
            def normalized_accumulator(counts):
                accumulator = NormalizedGradientAccumulator(raw_model, counts,
                    direction_weight=direction_weight, presence_weight=presence_weight,
                    positive_query_weight=conditioned_loss_options["positive_query_weight"],
                    negative_query_weight=conditioned_loss_options["negative_query_weight"],
                    distributed=distributed.enabled,
                    loss_scale=float(grad_scaler.get_scale()) if scaler_enabled else 1.0,
                    gaussian_presence=gaussian_presence)
                if scaler_enabled:
                    grad_scaler.scale(accumulator.statistics[0])
                return accumulator

            if target_only_accumulation and optimizer_microbatch_count == 0:
                component_accumulator = normalized_accumulator(torch.zeros(
                    len(loss_components(raw_model, gaussian_presence=gaussian_presence)),
                    device=device))
            for microbatch_index, take in enumerate(optimizer_microbatch_takes):
                local_batch_ordinal = (
                    (step - 1) * optimizer_microbatch_count + microbatch_index + 1
                )
                sample_index = bounded_index(
                    (step - start_step - 1) * bounded_index.local_samples
                    + sum(optimizer_microbatch_takes[:microbatch_index])
                ) if bounded_index is not None else VirtualBatchSampleIndex(
                    batch_size=loader.config.batch_size,
                    physical_world_size=distributed.world_size,
                    rank=distributed.rank,
                    virtual_world_size=optimizer_virtual_world_size,
                    start_step=0,
                )(
                    (local_batch_ordinal - 1) * loader.config.batch_size
                )
                if replayed_batch is not None:
                    next_batch, next_load_ms, next_wait_ms, next_to_device_ms, next_target_ms = (
                        _replayed_training_batch(replayed_batch)
                    )
                elif replayed_cpu_batch is not None:
                    next_batch, next_load_ms, next_wait_ms, next_to_device_ms, next_target_ms = (
                        _replayed_cpu_training_batch(
                            replayed_cpu_batch,
                            loader=loader,
                            device=device,
                            output_stride=_model_output_stride(raw_model),
                        )
                    )
                else:
                    next_batch, next_load_ms, next_wait_ms, next_to_device_ms, next_target_ms = (
                        _next_training_batch(
                            iterator=train_iterator,
                            loader=loader,
                            sample_index=sample_index,
                            sample_index_limit=sample_index_limit,
                            sample_mode="random",
                            device=device,
                            output_stride=_model_output_stride(raw_model),
                        )
                    )
                load_ms += next_load_ms
                wait_ms += next_wait_ms
                to_device_ms += next_to_device_ms
                target_ms += next_target_ms
                cuda_value_augmentation_ms += float(
                    (next_batch.profile_timings_ms or {}).get(
                        "cuda_pipeline_value_augmentation_ms", 0.0
                    )
                )
                if int(next_batch.volume.shape[0]) != int(take):
                    raise RuntimeError(
                        "training batch size differs from the canonical partition"
                    )
                if target_only_accumulation and component_accumulator is None:
                    if next_batch.normalization_counts is None:
                        raise RuntimeError("missing pre-forward effective-batch normalization counts")
                    with profile.stage("target_count_sync"):
                        component_accumulator = normalized_accumulator(next_batch.normalization_counts)
                if batch is None:
                    batch = next_batch
                sync_context = (
                    nullcontext()
                    if manual_gradient_sync or bounded_accumulation
                    else (
                        nullcontext()
                        if microbatch_index + 1 == optimizer_microbatch_count
                        or not distributed.enabled
                        else distributed_model.no_sync()
                    )
                )
                if fw_start is None:
                    # Preserve the historical metric boundary: exclude the
                    # first batch wait/transfer, but include the full optimizer
                    # update from its first forward onward.
                    fw_start = time.perf_counter()
                with sync_context:
                    with profile.stage("forward_loss"), _autocast_context(mixed_precision):
                        next_statistics = _forward_loss_statistics(
                            train_model,
                            next_batch,
                            conditioned_loss_options=conditioned_loss_options,
                            prediction_observer=_combine_prediction_observers(
                                refinement if refinement.config.enabled and refinement.config.update_enabled else None, train_visualization),
                        )
                    if component_accumulator is not None:
                        if scaler_enabled:
                            # Initialize GradScaler's device-local scale state;
                            # actual scaled VJPs are built by the accumulator.
                            grad_scaler.scale(next_statistics.direction_sum)
                        with profile.stage("backward"):
                            component_accumulator.add(next_statistics)
                        loss_statistics = component_accumulator.statistics
                    else:
                        loss_statistics = (
                            next_statistics
                            if loss_statistics is None
                            else loss_statistics + next_statistics
                        )
                    del next_statistics
            assert fw_start is not None
            assert component_accumulator is not None or batch is not None and loss_statistics is not None
            with profile.stage("loss_stat_sync"):
                if component_accumulator is not None:
                    loss_statistics = component_accumulator.reduce_and_assign()
                loss_tensors = _finalize_loss_statistics(
                    loss_statistics,
                    direction_weight=direction_weight,
                    presence_weight=presence_weight,
                    positive_query_weight=conditioned_loss_options["positive_query_weight"],
                    negative_query_weight=conditioned_loss_options["negative_query_weight"],
                    distributed=None if bounded_accumulation else distributed,
                )
            if bounded_accumulation:
                del component_accumulator
            elif scaler_enabled:
                assert grad_scaler is not None
                with profile.stage("backward"):
                    grad_scaler.scale(loss_tensors["total"]).backward()
            else:
                with profile.stage("backward"):
                    loss_tensors["total"].backward()
            if manual_gradient_sync and not bounded_accumulation:
                with profile.stage("manual_gradient_sync"):
                    _distributed_average_model_gradients(raw_model, distributed)

            gradient_norm_due = (
                gradient_clip_max_norm is not None
                or (
                    gradient_norm_interval > 0
                    and step % gradient_norm_interval == 0
                )
            )
            grad_norm_tensor: torch.Tensor | None = None
            grad_norm_l2: float | None = None
            gradient_clipped: bool | None = None
            if gradient_norm_due:
                if scaler_enabled:
                    assert grad_scaler is not None
                    grad_scaler.unscale_(optimizer)
                with profile.stage("gradient_norm_clip"):
                    grad_norm_tensor = _measure_and_clip_gradient_norm(
                        raw_model,
                        max_norm=gradient_clip_max_norm,
                    )
                materialize_grad_norm = (
                    scalar_metrics_due
                    or (
                        gradient_norm_interval > 0
                        and step % gradient_norm_interval == 0
                    )
                )
                if materialize_grad_norm:
                    grad_norm_l2 = float(grad_norm_tensor.detach().cpu())
                    if not math.isfinite(grad_norm_l2):
                        raise FloatingPointError(
                            f"non-finite global L2 gradient norm at step {step}: "
                            f"{grad_norm_l2}"
                        )
                    gradient_clipped = (
                        gradient_clip_max_norm is not None
                        and grad_norm_l2 > gradient_clip_max_norm
                    )
            if scaler_enabled:
                assert grad_scaler is not None
                scale_before = grad_scaler.get_scale() if ema.tracks else None
                with profile.stage("optimizer"):
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                optimizer_updated = not ema.tracks or grad_scaler.get_scale() >= scale_before
            else:
                with profile.stage("optimizer"):
                    optimizer.step()
                optimizer_updated = True
            if ema.tracks and optimizer_updated:
                with profile.stage("ema"):
                    ema.update(raw_model)
            losses: dict[str, float] | None = None
            if scalar_metrics_due:
                with profile.stage("scalar_reduce_sync"):
                    losses = _training_loss_metrics_if_due(
                        loss_tensors,
                        distributed,
                        due=True,
                        globally_reduced=True,
                    )
            step_ms = (time.perf_counter() - fw_start) * 1000.0

            if distributed.is_main and console_metrics_due:
                assert losses is not None
                gradient_log = (
                    ""
                    if grad_norm_l2 is None
                    else (
                        f"grad_norm_l2={grad_norm_l2:.6g} "
                        f"gradient_clipped={int(bool(gradient_clipped))} "
                    )
                )
                print(
                    f"samples={sample_progress.at(step)} update={step} loss_total={losses['total']:.6f} "
                    f"loss_direction={losses['direction']:.6f} "
                    f"loss_presence={losses['presence']:.6f} "
                    f"{_embedding_log(losses, raw_model)}"
                    f"angle_mean_deg={losses['angle_mean_deg']:.2f} "
                    f"branch0={losses['branch0_fraction']:.3f} "
                    f"branch1={losses['branch1_fraction']:.3f} "
                    f"selected_score={losses['selected_score_mean']:.3f} "
                    f"lr={learning_rate:.6g} "
                    f"{gradient_log}"
                    f"load_ms={load_ms:.1f} wait_ms={wait_ms:.1f} "
                    f"to_device_ms={to_device_ms:.1f} "
                    f"value_aug_cuda_ms={cuda_value_augmentation_ms:.1f} "
                    f"target_ms={target_ms:.1f} "
                    f"fw_bw_step_ms={step_ms:.1f}",
                    flush=True,
                )
            if writer is not None and writer_metrics_due:
                assert losses is not None
                writer.add_scalar("train/loss_total", losses["total"], step)
                writer.add_scalar("train/loss_direction", losses["direction"], step)
                writer.add_scalar("train/loss_presence", losses["presence"], step)
                _log_embedding_scalars(writer, "train", losses, raw_model, step)
                writer.add_scalar("train/angle_mean_deg", losses["angle_mean_deg"], step)
                writer.add_scalar("train/branch0_fraction", losses["branch0_fraction"], step)
                writer.add_scalar("train/branch1_fraction", losses["branch1_fraction"], step)
                writer.add_scalar("train/selected_score_mean", losses["selected_score_mean"], step)
                writer.add_scalar("train/learning_rate", learning_rate, step)
                if grad_norm_l2 is not None:
                    writer.add_scalar("train/grad_norm_l2", grad_norm_l2, step)
                    writer.add_scalar(
                        "train/gradient_clipped",
                        float(bool(gradient_clipped)),
                        step,
                    )
                writer.add_scalar("timing/load_ms", load_ms, step)
                writer.add_scalar("timing/load_wait_ms", wait_ms, step)
                writer.add_scalar("timing/batch_to_device_ms", to_device_ms, step)
                writer.add_scalar(
                    "timing/value_augmentation_cuda_ms",
                    cuda_value_augmentation_ms,
                    step,
                )
                writer.add_scalar("timing/target_ms", target_ms, step)
                writer.add_scalar("timing/fw_bw_step_ms", step_ms, step)
            if train_visualization is not None:
                train_visualization.write(writer, "train_sample_3d/principal_slices", step)
            if train_vis_due:
                _distributed_barrier(distributed)

            if refinement.config.enabled and refinement.config.update_enabled:
                refinement.flush(distributed, phase="train", step=step, writer=writer)
            test_metric = None
            test_metric_name = None
            test_due = test_interval > 0 and step % test_interval == 0
            if test_due:
                if cuda_batch_pipeline is not None:
                    cuda_batch_pipeline.pause()
                if queued_train is not None:
                    queued_train.release_current()
                test_metric, test_metric_name = yield from run_configured_tests(step)
                test_metric = _distributed_broadcast_object(test_metric, distributed)
                test_metric_name = _distributed_broadcast_object(test_metric_name, distributed)

            save_test_snapshots(step, test_metric, test_metric_name)
            pending_test = None
            if test_due:
                admit_training_until_test(step)
            if test_due and cuda_batch_pipeline is not None:
                cuda_batch_pipeline.resume()
            profile_timings = {
                "load_ms": load_ms,
                "wait_ms": wait_ms,
                "to_device_ms": to_device_ms,
                "value_augmentation_cuda_ms": cuda_value_augmentation_ms,
                "target_ms": target_ms,
                "fw_bw_step_ms": step_ms,
                "learning_rate": learning_rate,
                "loader_profile": (
                    batch.profile_timings_ms if profile_loader and batch is not None else None
                ),
            }
            if lr_range_index is not None:
                profile_timings["lr_range_index"] = lr_range_index
            if grad_norm_l2 is not None:
                profile_timings["grad_norm_l2"] = grad_norm_l2
                profile_timings["gradient_clipped"] = float(
                    bool(gradient_clipped)
                )
            if losses is not None:
                profile_timings.update(
                    {
                        f"loss_{key}": value
                        for key, value in losses.items()
                    }
                )
            profile.end(**profile_timings)
            if throughput_started is not None:
                throughput_steps += 1
            if throughput_started is not None and max_steps is not None and step == max_steps:
                _synchronize_device(device)
                local_elapsed = time.perf_counter() - throughput_started
                slowest_elapsed = _distributed_max_float(local_elapsed, distributed)
                throughput = {
                    "start_step": throughput_start_step,
                    "end_step": step,
                    "steps": throughput_steps,
                    "elapsed_seconds_slowest_rank": slowest_elapsed,
                    "world_size": distributed.world_size,
                    "batch_size_per_rank": loader.config.batch_size,
                    "global_batch_size": optimizer_global_batch_size,
                    "microbatches_per_update": optimizer_microbatch_count,
                    "global_samples_per_second": (
                        throughput_steps * optimizer_global_batch_size
                        / slowest_elapsed
                    ),
                    "samples_per_second_per_gpu": (
                        throughput_steps
                        * optimizer_global_batch_size
                        / distributed.world_size
                        / slowest_elapsed
                    ),
                    "replay_batch": profile_replay_batch,
                    "replay_cpu_batch": profile_replay_cpu_batch,
                    "sample_pipeline_workers": sample_pipeline_workers,
                    "cuda_batch_pipeline_depth": cuda_batch_depth,
                    "replay_batch_sha256": replayed_batch_sha256,
                    "replay_source_batch_sha256": replayed_source_batch_sha256,
                    "normalization_module_counts": normalization_module_counts,
                    "sync_batchnorm": sync_batchnorm,
                    "ddp_broadcast_buffers": ddp_broadcast_buffers,
                    "model_output_stride": _model_output_stride(raw_model),
                    "model_parameter_count": sum(
                        int(parameter.numel()) for parameter in raw_model.parameters()
                    ),
                    "output_shape_zyx": tuple(
                        int(value) for value in batch.presence_target.shape[-3:]
                    ) if batch is not None else None,
                    "peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda" else 0
                    ),
                    "peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda" else 0
                    ),
                    "maximum_live_pinned_bytes": (
                        int(cuda_batch_pipeline.max_pinned_bytes)
                        if cuda_batch_pipeline is not None else 0
                    ),
                }
                (run_dir / f"throughput_rank{distributed.rank}.json").write_text(
                    json.dumps(throughput, indent=2) + "\n"
                )
                if distributed.is_main:
                    print(
                        "fiber_trace_3d throughput: "
                        f"steps={throughput_steps} elapsed={slowest_elapsed:.3f}s "
                        f"global_samples_per_second={throughput['global_samples_per_second']:.3f} "
                        f"samples_per_second_per_gpu={throughput['samples_per_second_per_gpu']:.3f} "
                        f"replay_batch={profile_replay_batch}",
                        flush=True,
                    )
            yield TrainingBoundary(step, max_steps is not None and step >= max_steps, capture_boundary)
    finally:
        try:
            profile.close()
        finally:
            if test_dataloader is not None:
                test_dataloader = None
            if train_dataloader is not None:
                _shutdown_dataloader_iterator(train_iterator)
                train_iterator = None
                train_dataloader = None
            if sample_queue is not None:
                sample_queue.close()
            if writer is not None:
                writer.flush()
                writer.close()
            if diagnostics is not None:
                diagnostics.close()
            _distributed_cleanup(distributed)


def _image_to_u8(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not bool(mask.any()):
        return out
    values = arr[mask]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(values.min())
        hi = float(values.max())
    scaled = np.clip((arr - lo) / max(hi - lo, 1.0e-6), 0.0, 1.0)
    out[mask] = np.rint(scaled[mask] * 255.0).astype(np.uint8)
    return out


def _gray_to_rgb(values: np.ndarray, *, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(arr))
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    clipped = np.clip(arr, 0.0, 1.0)
    gray = np.rint(clipped * 255.0).astype(np.uint8)
    out[valid] = gray[valid, None]
    return out


def _mark_slice_cp(panel: np.ndarray, row: int, col: int) -> None:
    h, w = int(panel.shape[0]), int(panel.shape[1])
    r = int(np.clip(row, 0, max(h - 1, 0)))
    c = int(np.clip(col, 0, max(w - 1, 0)))
    color = np.asarray([255, 255, 255], dtype=np.uint8)
    for delta in range(4, 10):
        if 0 <= r - delta < h:
            panel[r - delta, c] = color
        if 0 <= r + delta < h:
            panel[r + delta, c] = color
        if 0 <= c - delta < w:
            panel[r, c - delta] = color
        if 0 <= c + delta < w:
            panel[r, c + delta] = color


def _draw_panel_point(panel: np.ndarray, row: int, col: int, color: tuple[int, int, int]) -> None:
    h, w = int(panel.shape[0]), int(panel.shape[1])
    r = int(round(float(row)))
    c = int(round(float(col)))
    if not (0 <= r < h and 0 <= c < w):
        return
    rgb = np.asarray(color, dtype=np.uint8)
    panel[r, c] = rgb
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            panel[rr, cc] = rgb


def _blend_panel_pixel(
    panel: np.ndarray,
    row: int,
    col: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    h, w = int(panel.shape[0]), int(panel.shape[1])
    if not (0 <= row < h and 0 <= col < w):
        return
    opacity = float(np.clip(alpha, 0.0, 1.0))
    if opacity <= 0.0:
        return
    src = np.asarray(color, dtype=np.float32)
    dst = panel[row, col].astype(np.float32, copy=False)
    panel[row, col] = np.rint(dst * (1.0 - opacity) + src * opacity).astype(np.uint8)


def _draw_panel_line_aa(
    panel: np.ndarray,
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    start = np.asarray(start_rc, dtype=np.float32)
    end = np.asarray(end_rc, dtype=np.float32)
    x0, y0 = float(start[1]), float(start[0])
    x1, y1 = float(end[1]), float(end[0])

    def ipart(value: float) -> int:
        return int(math.floor(value))

    def round_part(value: float) -> int:
        return int(math.floor(value + 0.5))

    def fpart(value: float) -> float:
        return value - math.floor(value)

    def rfpart(value: float) -> float:
        return 1.0 - fpart(value)

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) <= 1.0e-6:
        _blend_panel_pixel(panel, round_part(y0), round_part(x0), color, 1.0)
        return
    gradient = dy / dx

    def plot(x: int, y: int, brightness: float) -> None:
        if steep:
            _blend_panel_pixel(panel, x, y, color, brightness)
        else:
            _blend_panel_pixel(panel, y, x, color, brightness)

    xend = round_part(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)
    plot(xpxl1, ypxl1, rfpart(yend) * xgap)
    plot(xpxl1, ypxl1 + 1, fpart(yend) * xgap)
    intery = yend + gradient

    xend = round_part(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)
    plot(xpxl2, ypxl2, rfpart(yend) * xgap)
    plot(xpxl2, ypxl2 + 1, fpart(yend) * xgap)

    for x in range(xpxl1 + 1, xpxl2):
        y = ipart(intery)
        plot(x, y, rfpart(intery))
        plot(x, y + 1, fpart(intery))
        intery += gradient


def _draw_panel_line(
    panel: np.ndarray,
    start_rc: np.ndarray,
    end_rc: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    start = np.asarray(start_rc, dtype=np.float32)
    end = np.asarray(end_rc, dtype=np.float32)
    delta = end - start
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))))))
    for index in range(steps + 1):
        t = float(index) / float(steps)
        point = start * (1.0 - t) + end * t
        _draw_panel_point(panel, int(round(float(point[0]))), int(round(float(point[1]))), color)


def _draw_projected_gt_line(
    panel: np.ndarray,
    segments_start_zyx: np.ndarray,
    segments_end_zyx: np.ndarray,
    *,
    plane_axis: int,
    plane_coord: int,
    row_axis: int,
    col_axis: int,
    threshold_voxels: float = 2.0,
) -> None:
    if segments_start_zyx.size == 0:
        return
    threshold = float(threshold_voxels)
    color = (0, 255, 80)
    for start, end in zip(segments_start_zyx, segments_end_zyx, strict=False):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        delta = end - start
        max_delta = float(np.max(np.abs(delta)))
        steps = max(1, int(math.ceil(max_delta * 2.0)))
        prev_rc: np.ndarray | None = None
        for index in range(steps + 1):
            t = float(index) / float(steps)
            point = start * (1.0 - t) + end * t
            if abs(float(point[plane_axis]) - float(plane_coord)) > threshold:
                prev_rc = None
                continue
            rc = np.asarray([point[row_axis], point[col_axis]], dtype=np.float32)
            if prev_rc is not None:
                _draw_panel_line(panel, prev_rc, rc, color)
            else:
                _draw_panel_point(panel, int(round(float(rc[0]))), int(round(float(rc[1]))), color)
            prev_rc = rc


def _project_oblique_point_rc_dist(
    point_zyx: np.ndarray,
    *,
    center_zyx: np.ndarray,
    row_axis_zyx: np.ndarray,
    col_axis_zyx: np.ndarray,
    normal_zyx: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, float]:
    rel = np.asarray(point_zyx, dtype=np.float32) - np.asarray(center_zyx, dtype=np.float32)
    row_axis = _unit_np(row_axis_zyx, fallback=(0.0, 1.0, 0.0))
    col_axis = _unit_np(col_axis_zyx, fallback=(0.0, 0.0, 1.0))
    normal = _unit_np(normal_zyx, fallback=(1.0, 0.0, 0.0))
    rc = np.asarray(
        [
            float(np.dot(rel, row_axis)) + float(height - 1) * 0.5,
            float(np.dot(rel, col_axis)) + float(width - 1) * 0.5,
        ],
        dtype=np.float32,
    )
    distance = float(np.dot(rel, normal))
    return rc, distance


def _draw_projected_oblique_gt_line(
    panel: np.ndarray,
    segments_start_zyx: np.ndarray,
    segments_end_zyx: np.ndarray,
    *,
    center_zyx: np.ndarray,
    row_axis_zyx: np.ndarray,
    col_axis_zyx: np.ndarray,
    normal_zyx: np.ndarray,
    threshold_voxels: float = 2.0,
    color: tuple[int, int, int] = (0, 255, 80),
) -> None:
    if segments_start_zyx.size == 0:
        return
    height = int(panel.shape[0])
    width = int(panel.shape[1])
    threshold = float(threshold_voxels)
    for start, end in zip(segments_start_zyx, segments_end_zyx, strict=False):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        delta = end - start
        max_delta = float(np.max(np.abs(delta)))
        if not math.isfinite(max_delta):
            continue
        steps = max(1, int(math.ceil(max_delta * 2.0)))
        prev_rc: np.ndarray | None = None
        for index in range(steps + 1):
            t = float(index) / float(steps)
            point = start * (1.0 - t) + end * t
            rc, distance = _project_oblique_point_rc_dist(
                point,
                center_zyx=center_zyx,
                row_axis_zyx=row_axis_zyx,
                col_axis_zyx=col_axis_zyx,
                normal_zyx=normal_zyx,
                height=height,
                width=width,
            )
            if abs(distance) > threshold:
                prev_rc = None
                continue
            if prev_rc is not None:
                _draw_panel_line(panel, prev_rc, rc, color)
            else:
                _draw_panel_point(panel, int(round(float(rc[0]))), int(round(float(rc[1]))), color)
            prev_rc = rc


def _oblique_line_presence_for_display(
    *,
    height: int,
    width: int,
    segments_start_zyx: np.ndarray,
    segments_end_zyx: np.ndarray,
    center_zyx: np.ndarray,
    row_axis_zyx: np.ndarray,
    col_axis_zyx: np.ndarray,
    normal_zyx: np.ndarray,
    threshold_voxels: float = 2.0,
) -> np.ndarray:
    panel = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    _draw_projected_oblique_gt_line(
        panel,
        segments_start_zyx,
        segments_end_zyx,
        center_zyx=center_zyx,
        row_axis_zyx=row_axis_zyx,
        col_axis_zyx=col_axis_zyx,
        normal_zyx=normal_zyx,
        threshold_voxels=threshold_voxels,
        color=(255, 255, 255),
    )
    presence = (panel[..., 0] > 0).astype(np.float32)
    if not bool(np.any(presence > 0.0)):
        return presence
    pooled = F.max_pool2d(
        torch.as_tensor(presence).view(1, 1, int(height), int(width)),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    return pooled.numpy()


def _line_presence_for_display(
    patch_shape: tuple[int, int, int],
    segments_start_zyx: np.ndarray,
    segments_end_zyx: np.ndarray,
) -> np.ndarray:
    presence = np.zeros(tuple(int(v) for v in patch_shape), dtype=np.float32)
    if segments_start_zyx.size == 0:
        return presence
    shape = np.asarray(patch_shape, dtype=np.int64)
    for start, end in zip(segments_start_zyx, segments_end_zyx, strict=False):
        start = np.asarray(start, dtype=np.float32)
        end = np.asarray(end, dtype=np.float32)
        if not np.isfinite(start).all() or not np.isfinite(end).all():
            continue
        delta = end - start
        max_delta = float(np.max(np.abs(delta)))
        if not math.isfinite(max_delta):
            continue
        steps = max(1, int(math.ceil(max_delta)))
        for index in range(steps + 1):
            t = float(index) / float(steps)
            coord = np.rint(start * (1.0 - t) + end * t).astype(np.int64)
            if bool(np.all(coord >= 0) and np.all(coord < shape)):
                presence[int(coord[0]), int(coord[1]), int(coord[2])] = 1.0
    if not bool(np.any(presence > 0.0)):
        return presence
    pooled = F.max_pool3d(
        torch.as_tensor(presence).view(1, 1, *patch_shape),
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0]
    return pooled.numpy()


def _draw_projected_cp_direction(
    panel: np.ndarray,
    *,
    cp_row: int,
    cp_col: int,
    direction_zyx: np.ndarray,
    row_axis: int,
    col_axis: int,
) -> None:
    direction = np.asarray(direction_zyx, dtype=np.float32)
    full_norm = float(np.linalg.norm(direction))
    if not math.isfinite(full_norm) or full_norm <= 1.0e-6:
        return
    direction = direction / full_norm
    projected = np.asarray([direction[row_axis], direction[col_axis]], dtype=np.float32)
    projection_norm = float(np.linalg.norm(projected))
    if not math.isfinite(projection_norm) or projection_norm <= 1.0e-6:
        return
    projected = projected / projection_norm
    base_radius = max(8.0, min(float(panel.shape[0]), float(panel.shape[1])) * 0.08)
    radius = base_radius * projection_norm
    center = np.asarray([float(cp_row), float(cp_col)], dtype=np.float32)
    start = center - projected * radius
    end = center + projected * radius
    _draw_panel_line_aa(panel, start, end, (255, 80, 0))


def _zyx_to_xyz_np(vector_zyx: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_zyx, dtype=np.float32)
    return vector[[2, 1, 0]].astype(np.float32, copy=False)


def _unit_np(vector: np.ndarray, *, fallback: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 1.0e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (arr / np.float32(norm)).astype(np.float32, copy=False)


def _least_parallel_axis_zyx(vector_zyx: np.ndarray) -> np.ndarray:
    vector = _unit_np(vector_zyx, fallback=(0.0, 0.0, 1.0))
    axes = np.eye(3, dtype=np.float32)
    index = int(np.argmin(np.abs(axes @ vector)))
    return axes[index]


def _basis_for_normal_zyx(normal_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _unit_np(normal_zyx, fallback=(0.0, 0.0, 1.0))
    helper = _least_parallel_axis_zyx(normal)
    row_axis = _unit_np(np.cross(normal, helper), fallback=(0.0, 1.0, 0.0))
    col_axis = _unit_np(np.cross(row_axis, normal), fallback=(0.0, 0.0, 1.0))
    return row_axis, col_axis


def _tangent_slice_frame_zyx(tangent_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    col_axis = _unit_np(tangent_zyx, fallback=(0.0, 0.0, 1.0))
    helper = _least_parallel_axis_zyx(col_axis)
    row_axis = helper - np.float32(np.dot(helper, col_axis)) * col_axis
    row_axis = _unit_np(row_axis, fallback=(0.0, 1.0, 0.0))
    normal = _unit_np(np.cross(row_axis, col_axis), fallback=(1.0, 0.0, 0.0))
    return row_axis, col_axis, normal


def _oblique_coords_zyx(
    *,
    center_zyx: np.ndarray,
    row_axis_zyx: np.ndarray,
    col_axis_zyx: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    row_offsets = torch.arange(int(height), dtype=torch.float32, device=device) - (
        float(height - 1) * 0.5
    )
    col_offsets = torch.arange(int(width), dtype=torch.float32, device=device) - (
        float(width - 1) * 0.5
    )
    rr, cc = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    center = torch.as_tensor(center_zyx, dtype=torch.float32, device=device)
    row_axis = torch.as_tensor(row_axis_zyx, dtype=torch.float32, device=device)
    col_axis = torch.as_tensor(col_axis_zyx, dtype=torch.float32, device=device)
    return center.view(1, 1, 3) + rr[..., None] * row_axis.view(1, 1, 3) + cc[..., None] * col_axis.view(1, 1, 3)


def _sample_czyx_at_coords(volume_czyx: torch.Tensor, coords_zyx: torch.Tensor) -> torch.Tensor:
    if volume_czyx.ndim != 4:
        raise ValueError("volume_czyx must have shape C,D,H,W")
    if coords_zyx.ndim != 3 or int(coords_zyx.shape[-1]) != 3:
        raise ValueError("coords_zyx must have shape H,W,3")
    _c, d, h, w = (int(v) for v in volume_czyx.shape)
    coords = coords_zyx.to(dtype=torch.float32, device=volume_czyx.device)
    gx = coords[..., 2] * (2.0 / float(max(w - 1, 1))) - 1.0 if w > 1 else torch.zeros_like(coords[..., 2])
    gy = coords[..., 1] * (2.0 / float(max(h - 1, 1))) - 1.0 if h > 1 else torch.zeros_like(coords[..., 1])
    gz = coords[..., 0] * (2.0 / float(max(d - 1, 1))) - 1.0 if d > 1 else torch.zeros_like(coords[..., 0])
    grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(0).unsqueeze(1)
    return F.grid_sample(
        volume_czyx.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, :, 0].permute(1, 2, 0)


def _branch_presence_views(
    axes_xyz_khw3: np.ndarray,
    presence_khw: np.ndarray,
    *,
    normal_zyx: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    axes = np.asarray(axes_xyz_khw3, dtype=np.float32)
    presence = np.asarray(presence_khw, dtype=np.float32)
    if axes.ndim != 4 or axes.shape[-1] != 3:
        raise ValueError("axes_xyz_khw3 must have shape K,H,W,3")
    if presence.ndim != 3 or presence.shape[0] != axes.shape[0]:
        raise ValueError("presence_khw must have shape K,H,W matching axes")
    branch_count = int(presence.shape[0])
    branch0 = presence[0]
    if branch_count > 1:
        branch1 = presence[1]
    else:
        branch1 = np.zeros_like(branch0, dtype=np.float32)
    normal_xyz = _zyx_to_xyz_np(_unit_np(normal_zyx, fallback=(0.0, 0.0, 1.0)))
    dots = np.abs(np.sum(axes * normal_xyz.reshape(1, 1, 1, 3), axis=-1))
    close_index = np.argmax(dots, axis=0)
    if branch_count > 1:
        ordered = np.argsort(dots, axis=0)
        other_index = ordered[-2]
    else:
        other_index = close_index
    h, w = int(presence.shape[1]), int(presence.shape[2])
    rr = np.arange(h).reshape(h, 1)
    cc = np.arange(w).reshape(1, w)
    close = presence[close_index, rr, cc]
    if branch_count > 1:
        other = presence[other_index, rr, cc]
    else:
        other = np.zeros_like(close, dtype=np.float32)
    max_presence = np.max(presence, axis=0)
    min_presence = np.min(presence, axis=0)
    avg_presence = np.mean(presence, axis=0, dtype=np.float32)
    return (
        branch0.astype(np.float32, copy=False),
        branch1.astype(np.float32, copy=False),
        close.astype(np.float32, copy=False),
        other.astype(np.float32, copy=False),
        max_presence.astype(np.float32, copy=False),
        min_presence.astype(np.float32, copy=False),
        avg_presence.astype(np.float32, copy=False),
    )


def _branch_presence_views_from_sampled_output(
    sampled_hwc: torch.Tensor,
    *,
    normal_zyx: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    values = sampled_hwc.to(dtype=torch.float32)
    channels = int(values.shape[-1])
    if channels < 7 or channels % 7 != 0:
        raise ValueError("sampled model output channels must be a positive multiple of 7")
    branch_count = channels // 7
    dirs = []
    presences = []
    for branch in range(branch_count):
        start = branch * 7
        dirs.append(values[..., start : start + 6])
        presences.append(values[..., start + 6])
    dirs_t = torch.stack(dirs, dim=0)
    axes = decode_lasagna_direction_3x2_analytic(dirs_t).detach().cpu().numpy()
    presence = torch.stack(presences, dim=0).detach().cpu().numpy()
    return _branch_presence_views(axes, presence, normal_zyx=normal_zyx)


def _single_output_presence_views_from_sampled_output(
    sampled_hwc: torch.Tensor,
    *,
    normal_zyx: np.ndarray,
    tangent_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = sampled_hwc.to(dtype=torch.float32)
    channels = int(values.shape[-1])
    if channels != 7:
        raise ValueError("single-output presence views require exactly 7 channels")
    axes = decode_lasagna_direction_3x2_analytic(values[..., 0:6]).detach().cpu().numpy()
    presence = values[..., 6].detach().cpu().numpy().astype(np.float32, copy=False)
    normal_xyz = _zyx_to_xyz_np(_unit_np(normal_zyx, fallback=(0.0, 0.0, 1.0)))
    tangent_xyz = _zyx_to_xyz_np(_unit_np(tangent_zyx, fallback=(0.0, 0.0, 1.0)))
    normal_cos = np.abs(np.sum(axes * normal_xyz.reshape(1, 1, 3), axis=-1))
    tangent_cos = np.abs(np.sum(axes * tangent_xyz.reshape(1, 1, 3), axis=-1))
    return (
        presence,
        (presence * normal_cos).astype(np.float32, copy=False),
        (presence * tangent_cos).astype(np.float32, copy=False),
    )


def _sample_sheet_presence_views_from_sampled_output(
    sampled_hwc: torch.Tensor,
    *,
    normal_zyx: np.ndarray,
    tangent_zyx: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if int(sampled_hwc.shape[-1]) == 7:
        return _single_output_presence_views_from_sampled_output(
            sampled_hwc,
            normal_zyx=normal_zyx,
            tangent_zyx=tangent_zyx,
        )
    return _branch_presence_views_from_sampled_output(
        sampled_hwc,
        normal_zyx=normal_zyx,
    )


def _pad_panels_to_height(panels: list[np.ndarray]) -> list[np.ndarray]:
    height = max(int(panel.shape[0]) for panel in panels)
    padded_panels = []
    for panel in panels:
        if int(panel.shape[0]) < height:
            pad = np.zeros((height - int(panel.shape[0]), int(panel.shape[1]), 3), dtype=np.uint8)
            panel = np.concatenate([panel, pad], axis=0)
        padded_panels.append(panel)
    return padded_panels


def _make_sample_sheet_row(panels: list[np.ndarray], *, gap: int) -> np.ndarray:
    padded_panels = _pad_panels_to_height(panels)
    height = int(padded_panels[0].shape[0])
    sep = np.zeros((height, gap, 3), dtype=np.uint8)
    row = padded_panels[0]
    for panel in padded_panels[1:]:
        row = np.concatenate([row, sep, panel], axis=1)
    return row


def _make_oblique_sample_row(
    *,
    volume_czyx: torch.Tensor,
    valid_czyx: torch.Tensor,
    target_czyx: torch.Tensor | None,
    output_czyx: torch.Tensor,
    segments_start_zyx: np.ndarray,
    segments_end_zyx: np.ndarray,
    cp_zyx: np.ndarray,
    row_axis_zyx: np.ndarray,
    col_axis_zyx: np.ndarray,
    normal_zyx: np.ndarray,
    tangent_zyx: np.ndarray,
    height: int,
    width: int,
    gap: int,
) -> np.ndarray:
    coords = _oblique_coords_zyx(
        center_zyx=cp_zyx,
        row_axis_zyx=row_axis_zyx,
        col_axis_zyx=col_axis_zyx,
        height=height,
        width=width,
        device=output_czyx.device,
    )
    image = _sample_czyx_at_coords(volume_czyx, coords)[..., 0].detach().cpu().numpy()
    image_valid = (
        _sample_czyx_at_coords(valid_czyx.to(dtype=torch.float32), coords)[..., 0]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    )
    target_p = _oblique_line_presence_for_display(
        height=height,
        width=width,
        segments_start_zyx=segments_start_zyx,
        segments_end_zyx=segments_end_zyx,
        center_zyx=cp_zyx,
        row_axis_zyx=row_axis_zyx,
        col_axis_zyx=col_axis_zyx,
        normal_zyx=normal_zyx,
        threshold_voxels=2.0,
    )
    if not bool(np.any(target_p > 0.0)) and target_czyx is not None:
        target_p = _sample_czyx_at_coords(target_czyx, coords)[..., 0].detach().cpu().numpy()
    sampled_output = _sample_czyx_at_coords(output_czyx, coords)
    presence_views = _sample_sheet_presence_views_from_sampled_output(
        sampled_output,
        normal_zyx=normal_zyx,
        tangent_zyx=tangent_zyx,
    )
    image_rgb = np.repeat(_image_to_u8(image, image_valid)[..., None], 3, axis=2)
    target_rgb = _gray_to_rgb(target_p, mask=image_valid)
    presence_rgbs = [_gray_to_rgb(view, mask=image_valid) for view in presence_views]
    cp_row = int(round(float(height - 1) * 0.5))
    cp_col = int(round(float(width - 1) * 0.5))
    _draw_projected_oblique_gt_line(
        image_rgb,
        segments_start_zyx,
        segments_end_zyx,
        center_zyx=cp_zyx,
        row_axis_zyx=row_axis_zyx,
        col_axis_zyx=col_axis_zyx,
        normal_zyx=normal_zyx,
        threshold_voxels=2.0,
    )
    for panel in (image_rgb, target_rgb, *presence_rgbs):
        _mark_slice_cp(panel, cp_row, cp_col)
    return _make_sample_sheet_row(
        [
            image_rgb,
            target_rgb,
            *presence_rgbs,
        ],
        gap=gap,
    )


def _make_train_sample_3d_sheet(
    batch: FiberTrace3DBatch,
    output: torch.Tensor,
) -> np.ndarray:
    assert batch.presence_target is not None
    display_shape = tuple(int(value) for value in batch.volume.shape[-3:])
    if tuple(int(value) for value in output.shape[-3:]) != display_shape:
        output = F.interpolate(
            output.to(dtype=torch.float32),
            size=display_shape,
            mode="trilinear",
            align_corners=True,
        )
    if tuple(int(value) for value in batch.presence_target.shape[-3:]) != display_shape:
        batch = replace(
            batch,
            presence_target=F.interpolate(
                batch.presence_target.to(dtype=torch.float32),
                size=display_shape,
                mode="nearest",
            ),
        )
    volume = batch.volume[0, 0].detach().cpu().numpy()
    valid = (
        np.ones(volume.shape, dtype=bool)
        if batch.valid_mask is None
        else batch.valid_mask[0, 0].detach().cpu().numpy().astype(bool)
    )
    supervised_presence = F.max_pool3d(
        batch.presence_target[0:1],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].detach().cpu().numpy()

    cp_float = batch.cp_local_zyx[0].detach().cpu().numpy().astype(np.float32, copy=False)
    cp_float = np.clip(
        cp_float,
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        np.asarray(volume.shape, dtype=np.float32) - np.float32(1.0),
    )
    cp = np.rint(cp_float).astype(np.int64)
    z, y, x = (int(v) for v in cp)
    cp_encoded = direction_output(output)[0, :, z, y, x].view(1, 6)
    cp_pred_xyz = decode_lasagna_direction_3x2_analytic(cp_encoded)[0].detach().cpu().numpy()
    cp_pred_zyx = cp_pred_xyz[[2, 1, 0]].astype(np.float32, copy=False)
    segment_offset = int(batch.target_segment_offsets[0].detach().cpu())
    segment_count = int(batch.target_segment_counts[0].detach().cpu())
    segment_slice = slice(segment_offset, segment_offset + segment_count)
    segments_start_zyx = batch.target_segment_starts_zyx[segment_slice].detach().cpu().numpy()
    segments_end_zyx = batch.target_segment_ends_zyx[segment_slice].detach().cpu().numpy()
    tangent_zyx = batch.target_tangent_zyx[0].detach().cpu().numpy().astype(np.float32, copy=False)
    line_presence = _line_presence_for_display(
        tuple(int(v) for v in volume.shape),
        segments_start_zyx,
        segments_end_zyx,
    )
    target_presence = np.maximum(supervised_presence, line_presence)
    output_sample = output[0].detach()
    slice_specs = (
        (
            "yx",
            volume[z, :, :],
            valid[z, :, :],
            target_presence[z, :, :],
            output_sample[:, z, :, :].permute(1, 2, 0),
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            0,
            z,
            1,
            2,
            y,
            x,
        ),
        (
            "zx",
            volume[:, y, :],
            valid[:, y, :],
            target_presence[:, y, :],
            output_sample[:, :, y, :].permute(1, 2, 0),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            1,
            y,
            0,
            2,
            z,
            x,
        ),
        (
            "zy",
            volume[:, :, x],
            valid[:, :, x],
            target_presence[:, :, x],
            output_sample[:, :, :, x].permute(1, 2, 0),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            2,
            x,
            0,
            1,
            z,
            y,
        ),
    )

    rows: list[np.ndarray] = []
    gap = 4
    for (
        _name,
        image,
        image_valid,
        target_p,
        sampled_output,
        normal_zyx,
        plane_axis,
        plane_coord,
        row_axis,
        col_axis,
        cp_row,
        cp_col,
    ) in slice_specs:
        presence_views = _sample_sheet_presence_views_from_sampled_output(
            sampled_output,
            normal_zyx=normal_zyx,
            tangent_zyx=tangent_zyx,
        )
        image_rgb = np.repeat(_image_to_u8(image, image_valid)[..., None], 3, axis=2)
        target_rgb = _gray_to_rgb(target_p, mask=image_valid)
        presence_rgbs = [_gray_to_rgb(view, mask=image_valid) for view in presence_views]
        _draw_projected_gt_line(
            image_rgb,
            segments_start_zyx,
            segments_end_zyx,
            plane_axis=int(plane_axis),
            plane_coord=int(plane_coord),
            row_axis=int(row_axis),
            col_axis=int(col_axis),
            threshold_voxels=2.0,
        )
        _draw_projected_cp_direction(
            image_rgb,
            cp_row=int(cp_row),
            cp_col=int(cp_col),
            direction_zyx=cp_pred_zyx,
            row_axis=int(row_axis),
            col_axis=int(col_axis),
        )
        _mark_slice_cp(image_rgb, cp_row, cp_col)
        _mark_slice_cp(target_rgb, cp_row, cp_col)
        for panel in presence_rgbs:
            _mark_slice_cp(panel, cp_row, cp_col)
        rows.append(
            _make_sample_sheet_row(
                [
                    image_rgb,
                    target_rgb,
                    *presence_rgbs,
                ],
                gap=gap,
            )
        )

    tangent_row_axis, tangent_col_axis, tangent_normal = _tangent_slice_frame_zyx(tangent_zyx)
    cross_row_axis, cross_col_axis = _basis_for_normal_zyx(tangent_zyx)
    oblique_height = int(volume.shape[1])
    oblique_width = int(volume.shape[2])
    target_czyx = torch.as_tensor(
        target_presence,
        dtype=torch.float32,
        device=output_sample.device,
    ).view(1, *tuple(int(v) for v in target_presence.shape))
    rows.append(
        _make_oblique_sample_row(
            volume_czyx=batch.volume[0],
            valid_czyx=(
                torch.ones_like(batch.volume[0], dtype=torch.bool)
                if batch.valid_mask is None
                else batch.valid_mask[0]
            ),
            target_czyx=target_czyx,
            output_czyx=output_sample,
            segments_start_zyx=segments_start_zyx,
            segments_end_zyx=segments_end_zyx,
            cp_zyx=cp_float,
            row_axis_zyx=tangent_row_axis,
            col_axis_zyx=tangent_col_axis,
            normal_zyx=tangent_normal,
            tangent_zyx=tangent_zyx,
            height=oblique_height,
            width=oblique_width,
            gap=gap,
        )
    )
    rows.append(
        _make_oblique_sample_row(
            volume_czyx=batch.volume[0],
            valid_czyx=(
                torch.ones_like(batch.volume[0], dtype=torch.bool)
                if batch.valid_mask is None
                else batch.valid_mask[0]
            ),
            target_czyx=target_czyx,
            output_czyx=output_sample,
            segments_start_zyx=segments_start_zyx,
            segments_end_zyx=segments_end_zyx,
            cp_zyx=cp_float,
            row_axis_zyx=cross_row_axis,
            col_axis_zyx=cross_col_axis,
            normal_zyx=_unit_np(tangent_zyx, fallback=(0.0, 0.0, 1.0)),
            tangent_zyx=tangent_zyx,
            height=oblique_height,
            width=oblique_width,
            gap=gap,
        )
    )
    width = max(int(row.shape[1]) for row in rows)
    padded_rows = []
    for row in rows:
        if int(row.shape[1]) < width:
            pad = np.zeros((int(row.shape[0]), width - int(row.shape[1]), 3), dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded_rows.append(row)
    sep_row = np.zeros((gap, width, 3), dtype=np.uint8)
    sheet = padded_rows[0]
    for row in padded_rows[1:]:
        sheet = np.concatenate([sheet, sep_row, row], axis=0)
    return sheet


def _make_train_sample_3d_contact_sheet(
    batch: FiberTrace3DBatch,
    output: torch.Tensor,
    *,
    sample_count: int,
) -> np.ndarray:
    take = min(max(1, int(sample_count)), int(batch.volume.shape[0]))
    sheets: list[np.ndarray] = []
    for sample_index in range(take):
        sheets.append(
            _make_train_sample_3d_sheet(
                _slice_batch(batch, sample_index, sample_index + 1),
                output[sample_index : sample_index + 1],
            )
        )
    return _join_sample_sheets(sheets)


def _join_sample_sheets(sheets: list[np.ndarray]) -> np.ndarray:
    if len(sheets) == 1:
        return sheets[0]
    gap = 6
    height = max(int(sheet.shape[0]) for sheet in sheets)
    padded: list[np.ndarray] = []
    for sheet in sheets:
        if int(sheet.shape[0]) < height:
            pad = np.zeros((height - int(sheet.shape[0]), int(sheet.shape[1]), 3), dtype=np.uint8)
            sheet = np.concatenate([sheet, pad], axis=0)
        padded.append(sheet)
    sep = np.zeros((height, gap, 3), dtype=np.uint8)
    out = padded[0]
    for sheet in padded[1:]:
        out = np.concatenate([out, sep, sheet], axis=1)
    return out


class _ForwardVisualization:
    """Render a bounded set of actual loss-forward samples; retain CPU images only."""

    def __init__(self, sample_count, sheets=None):
        self.sample_count = max(0, int(sample_count))
        self.sheets = [] if sheets is None else sheets

    @torch.no_grad()
    def __call__(self, batch, output):
        remaining = self.sample_count - len(self.sheets)
        with torch.autocast(device_type=output.device.type, enabled=False):
            for index in range(min(max(0, remaining), int(batch.volume.shape[0]))):
                physical = _slice_batch(batch, index, index + 1)
                self.sheets.append(torch.from_numpy(_make_train_sample_3d_sheet(
                    physical, output[index:index + 1].detach().float())))

    def write(self, writer, tag, step):
        if self.sheets:
            writer.add_image(tag, _join_sample_sheets([np.asarray(sheet) for sheet in self.sheets]),
                             int(step), dataformats="HWC")


def _write_3d_sample_sheet(writer, tag, batch, output, step, *, sample_count=1):
    """Render supplied predictions; never load, augment, or call a model."""
    capture = _ForwardVisualization(sample_count)
    capture(batch, output)
    capture.write(writer, tag, step)


def _draw_trace2cp_3d_panel(
    image: np.ndarray,
    image_valid: np.ndarray,
    fields: Trace2Cp3DProjectedFields,
    source: Any,
    *,
    title: str,
    step_px: float,
    rf_margin_px: float,
):
    from PIL import Image, ImageDraw
    from vesuvius.neural_tracing.fiber_trace_2d.runner import (
        _trace_score_trace2cp_bidirectional,
    )

    base_u8 = _image_to_u8(image, image_valid & fields.valid_mask)
    rgb = np.repeat(base_u8[..., None], 3, axis=2)
    canvas = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    text_pad = 24
    padded = Image.new("RGBA", (canvas.width, canvas.height + text_pad), (0, 0, 0, 255))
    padded.alpha_composite(canvas, (0, text_pad))
    draw = ImageDraw.Draw(padded, "RGBA")
    draw.text((4, 4), title, fill=(255, 255, 255, 255))

    line = np.asarray(source.line_xy, dtype=np.float32)
    if line.ndim == 2 and line.shape[0] >= 2:
        pts = [(float(x), float(y) + text_pad) for x, y in line if np.isfinite(x) and np.isfinite(y)]
        if len(pts) >= 2:
            draw.line(pts, fill=(0, 255, 128, 120), width=1)

    step = max(8, int(round(min(fields.direction_xy.shape[:2]) / 32.0)))
    for y in range(step // 2, int(fields.direction_xy.shape[0]), step):
        for x in range(step // 2, int(fields.direction_xy.shape[1]), step):
            if not bool(fields.valid_mask[y, x]):
                continue
            dx, dy = fields.direction_xy[y, x]
            if not np.isfinite(dx) or not np.isfinite(dy):
                continue
            length = 5.0
            draw.line(
                [
                    (x - dx * length, y + text_pad - dy * length),
                    (x + dx * length, y + text_pad + dy * length),
                ],
                fill=(255, 220, 32, 180),
                width=1,
            )

    result = _trace_score_trace2cp_bidirectional(
        fields.direction_xy,
        np.asarray(source.start_control_point_xy, dtype=np.float32),
        np.asarray(source.target_control_point_xy, dtype=np.float32),
        valid_mask=fields.valid_mask,
        step_px=step_px,
        rf_margin_px=rf_margin_px,
    )
    for trace, color in (
        (result.forward.trace_xy, (64, 180, 255, 255)),
        (result.reverse.trace_xy, (255, 96, 220, 255)),
    ):
        trace_arr = np.asarray(trace, dtype=np.float32)
        if trace_arr.ndim == 2 and trace_arr.shape[0] >= 2:
            pts = [(float(x), float(y) + text_pad) for x, y in trace_arr if np.isfinite(x) and np.isfinite(y)]
            if len(pts) >= 2:
                draw.line(pts, fill=color, width=2)

    for xy, color in (
        (source.start_control_point_xy, (0, 255, 255, 255)),
        (source.target_control_point_xy, (255, 64, 220, 255)),
    ):
        x, y = (float(v) for v in xy)
        draw.ellipse((x - 4, y + text_pad - 4, x + 4, y + text_pad + 4), outline=color, width=2)

    if fields.presence_hw is None:
        return padded
    presence = np.asarray(fields.presence_hw, dtype=np.float32)
    presence_u8 = np.clip(presence, 0.0, 1.0)
    presence_rgb = np.zeros((*presence.shape, 3), dtype=np.uint8)
    presence_rgb[..., 0] = np.rint(presence_u8 * 255.0).astype(np.uint8)
    presence_rgb[..., 1] = np.rint(presence_u8 * 255.0).astype(np.uint8)
    presence_panel = Image.fromarray(presence_rgb, mode="RGB").convert("RGBA")
    presence_padded = Image.new("RGBA", (presence_panel.width, presence_panel.height + text_pad), (0, 0, 0, 255))
    presence_padded.alpha_composite(presence_panel, (0, text_pad))
    presence_draw = ImageDraw.Draw(presence_padded, "RGBA")
    presence_draw.text((4, 4), "projected 3D presence", fill=(255, 255, 255, 255))
    sheet = Image.new("RGBA", (padded.width + presence_padded.width, max(padded.height, presence_padded.height)), (0, 0, 0, 255))
    sheet.alpha_composite(padded, (0, 0))
    sheet.alpha_composite(presence_padded, (padded.width, 0))
    return sheet


def _trace2cp_loader_for_cli(
    raw_config: dict[str, Any],
    cfg: _Trace2Cp3DConfig,
    *,
    fiber_json: Path | None,
):
    if fiber_json is None:
        return _make_trace2cp_geometry_loader(raw_config, cfg)
    source_datasets = raw_config.get("test_datasets") or raw_config.get("datasets")
    if not isinstance(source_datasets, list) or len(source_datasets) != 1:
        raise ValueError("--fiber-json requires a config with exactly one dataset or test_datasets entry")
    dataset = dict(source_datasets[0])
    dataset.pop("fiber_glob", None)
    dataset["fiber_paths"] = [str(fiber_json)]
    cli_config = dict(raw_config)
    cli_config["test_datasets"] = [dataset]
    return _make_trace2cp_geometry_loader(cli_config, cfg)


def run_trace2cp_vis(
    config_path: str | Path,
    *,
    checkpoint: str | Path,
    export_dir: str | Path,
    sample_index: int,
    fiber_json: str | Path | None,
    step_px: float | None,
    rf_margin_px: float | None,
) -> None:
    raw_config = _load_raw_config(config_path)
    loader_config = load_config(config_path)
    trace_cfg = _trace2cp_3d_config(raw_config)
    if step_px is not None:
        trace_cfg = dataclass_replace(trace_cfg, step_px=float(step_px))
    if rf_margin_px is not None:
        trace_cfg = dataclass_replace(trace_cfg, rf_margin_px=float(rf_margin_px))
    geometry_loader = _trace2cp_loader_for_cli(
        raw_config,
        trace_cfg,
        fiber_json=None if fiber_json is None else Path(fiber_json),
    )
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    mixed_precision = _mixed_precision_config_from_training(training, device)
    model = build_fiber_trace_3d_model(raw_config).to(device)
    _load_snapshot(checkpoint, model=model, optimizer=None, map_location=device)
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = []
    errors: list[float] = []
    raw_errors: list[float] = []
    skipped = 0
    first_skip = ""
    if fiber_json is None:
        indices = [int(sample_index)]
        sample_mode = "random"
    else:
        indices = list(range(max(0, int(geometry_loader.sample_count) - 1)))
        sample_mode = "flat"
    for idx in indices:
        try:
            source = geometry_loader.build_trace2cp_segment_source(
                idx,
                target_offset=1,
                rf_margin_px=trace_cfg.rf_margin_px,
                device=torch.device("cpu"),
                sample_mode=sample_mode,
            )
            fields = _infer_trace2cp_fields_3d(
                model,
                source,
                image_normalization=loader_config.image_normalization,
                cfg=trace_cfg,
                device=device,
                mixed_precision=mixed_precision,
            )
            score = score_trace2cp_projected_fields(
                fields,
                start_xy=np.asarray(source.start_control_point_xy, dtype=np.float32),
                target_xy=np.asarray(source.target_control_point_xy, dtype=np.float32),
                step_px=trace_cfg.step_px,
                rf_margin_px=trace_cfg.rf_margin_px,
            )
            _sample, image, image_valid = geometry_loader.sample_trace2cp_segment_source(source)
            title = (
                f"sample={idx} trace2cp_error={score.trace2cp_error:.6f} "
                f"raw_y_px={score.raw_y_error_px:.2f}"
            )
            panels.append(
                _draw_trace2cp_3d_panel(
                    image,
                    image_valid,
                    fields,
                    source,
                    title=title,
                    step_px=trace_cfg.step_px,
                    rf_margin_px=trace_cfg.rf_margin_px,
                )
            )
            errors.append(float(score.trace2cp_error))
            raw_errors.append(float(score.raw_y_error_px))
        except ValueError as exc:
            skipped += 1
            if not first_skip:
                first_skip = " ".join(str(exc).split())
            continue
    if not errors:
        raise ValueError(
            f"no valid 3D Trace2CP segments for visualization: skipped={skipped} first_skip='{first_skip}'"
        )
    from PIL import Image

    width = max(panel.width for panel in panels)
    height = sum(panel.height for panel in panels)
    sheet = Image.new("RGBA", (width, height), (0, 0, 0, 255))
    y = 0
    for panel in panels:
        sheet.alpha_composite(panel, (0, y))
        y += panel.height
    output_path = out_dir / "trace2cp_3d_vis.jpg"
    sheet.convert("RGB").save(output_path, quality=95)
    if fiber_json is None:
        print(f"trace2cp_error={errors[0]:.8f}")
    else:
        print(f"trace2cp_error_mean={float(np.mean(errors)):.8f}")
    print(
        "trace2cp_3d "
        f"segments={len(errors)} skipped={skipped} "
        f"raw_y_error_mean_px={float(np.mean(raw_errors)):.3f} "
        f"export={output_path}",
        flush=True,
    )


def dataclass_replace(value: _Trace2Cp3DConfig, **kwargs: Any) -> _Trace2Cp3DConfig:
    data = value.__dict__.copy()
    data.update(kwargs)
    return _Trace2Cp3DConfig(**data)


def _identity_batch_collate(sample: Any) -> Any:
    if isinstance(sample, list):
        if len(sample) != 1:
            raise ValueError(
                "fiber_trace_3d DataLoader must yield one complete FiberTrace3DBatch per item"
            )
        return sample[0]
    return sample


class _FiberTrace3DBatchDataset(Dataset):
    _UNBOUNDED_BATCH_COUNT = 2**60

    def __init__(
        self,
        config_source: str | Path | Any,
        *,
        start_batch_index: int,
        batch_count: int | None,
        batch_index_stride: int = 1,
        sample_index_offset: int = 0,
        sample_index_limit: int | None = None,
        sample_mode: str,
        worker_device: str | torch.device = "cpu",
        profile: bool = False,
    ) -> None:
        self.config_source = config_source
        self.start_batch_index = int(start_batch_index)
        self.batch_index_stride = int(batch_index_stride)
        if self.batch_index_stride <= 0:
            raise ValueError("batch_index_stride must be > 0")
        self.sample_index_offset = int(sample_index_offset)
        if self.sample_index_offset < 0:
            raise ValueError("sample_index_offset must be >= 0")
        self.batch_count = None if batch_count is None else int(batch_count)
        self.sample_index_limit = 0 if sample_index_limit is None else int(sample_index_limit)
        if self.sample_index_limit < 0:
            raise ValueError("sample_index_limit must be >= 0")
        self.sample_mode = str(sample_mode)
        self.worker_device = str(worker_device)
        self.profile = bool(profile)
        self._loader: FiberTrace3DLoader | None = None
        self._pending_construct_ms = 0.0

    def __len__(self) -> int:
        if self.batch_count is None:
            return self._UNBOUNDED_BATCH_COUNT
        return max(0, int(self.batch_count))

    def _get_loader(self) -> FiberTrace3DLoader:
        if self._loader is None:
            start = time.perf_counter()
            if isinstance(self.config_source, (str, Path)):
                config = load_config(self.config_source)
            else:
                config = self.config_source
            self._loader = FiberTrace3DLoader(config)
            self._pending_construct_ms += (time.perf_counter() - start) * 1000.0
        return self._loader

    def __getitem__(self, index: int) -> FiberTrace3DBatch:
        if int(index) < 0 or int(index) >= len(self):
            raise IndexError(index)
        item_start_ns = time.time_ns()
        item_cpu_start = time.process_time()
        loader = self._get_loader()
        batch_index = self.start_batch_index + int(index) * self.batch_index_stride
        sample_index = self.sample_index_offset + batch_index * int(loader.config.batch_size)
        worker_device = torch.device(self.worker_device)
        batch = loader.load_batch(
            sample_index,
            sample_index_limit=self.sample_index_limit,
            sample_mode=self.sample_mode,
            device=worker_device,
            profile=self.profile,
        )
        if self.profile and self._pending_construct_ms > 0.0:
            timings = dict(batch.profile_timings_ms or {})
            timings["worker_loader_construct_ms"] = timings.get(
                "worker_loader_construct_ms",
                0.0,
            ) + float(self._pending_construct_ms)
            self._pending_construct_ms = 0.0
            batch = replace(batch, profile_timings_ms=timings)
        if self.profile:
            timings = dict(batch.profile_timings_ms or {})
            timings["worker_item_start_ns"] = float(item_start_ns)
            timings["worker_item_end_ns"] = float(time.time_ns())
            timings["worker_item_cpu_ms"] = (time.process_time() - item_cpu_start) * 1000.0
            timings["worker_item_index"] = float(index)
            worker_info = torch.utils.data.get_worker_info()
            timings["worker_id"] = float(worker_info.id if worker_info else -1)
            timings["worker_sample_index"] = float(sample_index)
            batch = replace(batch, profile_timings_ms=timings)
        if worker_device.type == "cuda":
            torch.cuda.synchronize(worker_device)
            batch = batch.to("cpu")
        return batch


def _loader_worker_count(raw_config: dict[str, Any]) -> int:
    training = dict(raw_config.get("training", {}))
    raw_workers = training.get("loader_workers", raw_config.get("loader_workers", 0))
    workers = int(raw_workers)
    if workers < 0:
        raise ValueError("training.loader_workers must be >= 0")
    return workers


def _rank_partition(total: int, *, world_size: int, rank: int) -> int:
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed partition")
    return total // world_size + int(rank < total % world_size)


def _training_sample_pipeline_workers(training: dict[str, Any]) -> int:
    workers = int(training.get("sample_pipeline_workers", 0))
    if workers < 0:
        raise ValueError("training.sample_pipeline_workers must be nonnegative")
    return workers


def _training_cuda_batch_pipeline_depth(training: dict[str, Any]) -> int:
    depth = int(training.get("sample_pipeline_cuda_prefetch_depth", 0))
    if depth < 0:
        raise ValueError(
            "training.sample_pipeline_cuda_prefetch_depth must be nonnegative"
        )
    return depth


def _training_cuda_value_augmentation(training: dict[str, Any]) -> str | None:
    mode = training.get("sample_pipeline_cuda_value_augmentation")
    if mode in {None, False}:
        return None
    if mode not in {"blur", "all"}:
        raise ValueError(
            "training.sample_pipeline_cuda_value_augmentation must be "
            "blur or all"
        )
    return str(mode)


def _training_scalar_metrics_due(
    step: int,
    *,
    scalar_interval: int,
    initial_steps: int,
) -> bool:
    """Return the rank-independent schedule for reducing/logging train metrics."""
    if int(step) <= 0:
        raise ValueError("training step must be positive")
    if int(scalar_interval) <= 0:
        raise ValueError("scalar_interval must be positive")
    if int(initial_steps) < 0:
        raise ValueError("initial_steps must be nonnegative")
    return int(step) <= int(initial_steps) or int(step) % int(scalar_interval) == 0


def _lr_range_test_config(training: dict[str, Any]) -> _LrRangeTestConfig | None:
    raw = training.get("lr_range_test")
    if raw is None or raw is False:
        return None
    if not isinstance(raw, dict):
        raise ValueError("training.lr_range_test must be an object")
    config = _LrRangeTestConfig(
        warmup_steps=int(raw.get("warmup_steps", 100)),
        range_steps=int(raw.get("range_steps", 1000)),
        start_lr=float(raw.get("start_lr", 1.0e-4)),
        end_lr=float(raw.get("end_lr", 1.0)),
    )
    if config.warmup_steps < 0:
        raise ValueError("training.lr_range_test.warmup_steps must be nonnegative")
    if config.range_steps < 2:
        raise ValueError("training.lr_range_test.range_steps must be >= 2")
    if not math.isfinite(config.start_lr) or config.start_lr <= 0.0:
        raise ValueError("training.lr_range_test.start_lr must be finite and positive")
    if not math.isfinite(config.end_lr) or config.end_lr <= config.start_lr:
        raise ValueError("training.lr_range_test.end_lr must exceed start_lr")
    return config


def _lr_range_learning_rate(config: _LrRangeTestConfig, step: int) -> float:
    """Return the LR for a one-indexed warmup-plus-exponential range step."""
    step_i = int(step)
    if step_i <= 0:
        raise ValueError("LR range step must be positive")
    if config.warmup_steps > 0 and step_i <= config.warmup_steps:
        return float(config.start_lr * float(step_i) / float(config.warmup_steps))
    range_index = step_i - config.warmup_steps - 1
    range_index = min(max(range_index, 0), config.range_steps - 1)
    fraction = float(range_index) / float(config.range_steps - 1)
    return float(
        config.start_lr
        * math.exp(math.log(config.end_lr / config.start_lr) * fraction)
    )


def _linear_warmup_learning_rate(
    peak_learning_rate: float,
    *,
    step: int,
    warmup_steps: int,
) -> float:
    """Return a one-indexed linear warmup capped at the configured peak LR."""
    peak = float(peak_learning_rate)
    step_i = int(step)
    warmup_i = int(warmup_steps)
    if not math.isfinite(peak) or peak <= 0.0:
        raise ValueError("peak learning rate must be finite and positive")
    if step_i <= 0:
        raise ValueError("warmup step must be positive")
    if warmup_i <= 0:
        raise ValueError("warmup_steps must be positive")
    return float(peak * min(step_i, warmup_i) / warmup_i)


def _optimizer_microbatch_takes(
    *,
    local_batch_size: int,
    world_size: int,
    rank: int,
    optimizer_global_batch_size: int,
) -> tuple[int, ...]:
    """Assign full virtual rank-local batches to one physical DDP rank."""
    local = int(local_batch_size)
    world = int(world_size)
    rank_i = int(rank)
    target = int(optimizer_global_batch_size)
    if local <= 0 or world <= 0 or target <= 0:
        raise ValueError("batch sizes and world_size must be positive")
    if not 0 <= rank_i < world:
        raise ValueError("rank must be in [0, world_size)")
    virtual_world = _optimizer_virtual_world_size(
        local_batch_size=local,
        physical_world_size=world,
        optimizer_global_batch_size=target,
    )
    local_batches = (virtual_world - rank_i + world - 1) // world
    return (local,) * local_batches


def _optimizer_virtual_world_size(
    *,
    local_batch_size: int,
    physical_world_size: int,
    optimizer_global_batch_size: int,
) -> int:
    """Return the number of full logical rank-local batches per update."""
    local = int(local_batch_size)
    physical = int(physical_world_size)
    target = int(optimizer_global_batch_size)
    if local <= 0 or physical <= 0 or target <= 0:
        raise ValueError("batch sizes and world_size must be positive")
    virtual, remainder = divmod(target, local)
    if remainder:
        raise ValueError(
            "optimizer_global_batch_size must be divisible by the local batch size"
        )
    if virtual < physical:
        raise ValueError(
            "optimizer_global_batch_size must cover at least one full local "
            "batch on every physical rank"
        )
    return virtual


def _distributed_average_model_gradients(
    model: torch.nn.Module,
    config: _DistributedConfig,
) -> None:
    """Apply DDP's SUM/world gradient reduction after local accumulation."""
    if not config.enabled:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            raise RuntimeError(
                "manual gradient synchronization requires every parameter to have a gradient"
            )
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(float(config.world_size))


def _measure_and_clip_gradient_norm(
    model: torch.nn.Module,
    *,
    max_norm: float | None,
) -> torch.Tensor:
    """Return the pre-clip global L2 norm and optionally clip in place."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=math.inf if max_norm is None else float(max_norm),
        norm_type=2.0,
        error_if_nonfinite=False,
        foreach=True,
    )


def _optimizer_learning_rate_scales(
    optimizer: torch.optim.Optimizer,
) -> tuple[float, ...]:
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    base = float(optimizer.param_groups[0]["lr"])
    if not math.isfinite(base) or base <= 0.0:
        raise ValueError("optimizer base learning rate must be finite and positive")
    return tuple(float(group["lr"]) / base for group in optimizer.param_groups)


def _set_optimizer_learning_rate(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    *,
    group_scales: Sequence[float] | None = None,
) -> None:
    scales = (
        tuple(1.0 for _ in optimizer.param_groups)
        if group_scales is None
        else tuple(float(value) for value in group_scales)
    )
    if len(scales) != len(optimizer.param_groups):
        raise ValueError("optimizer LR scale count does not match parameter groups")
    for group, scale in zip(optimizer.param_groups, scales, strict=True):
        group["lr"] = float(learning_rate) * scale


def _loader_prefetch_factor(raw_config: dict[str, Any]) -> int:
    training = dict(raw_config.get("training", {}))
    raw_factor = training.get("loader_prefetch_factor", raw_config.get("loader_prefetch_factor", 2))
    factor = int(raw_factor)
    if factor <= 0:
        raise ValueError("training.loader_prefetch_factor must be > 0")
    return factor


def _loader_worker_device(raw_config: dict[str, Any]) -> str:
    training = dict(raw_config.get("training", {}))
    return str(training.get("loader_worker_device", raw_config.get("loader_worker_device", "cpu")))


def _loader_multiprocessing_context(raw_config: dict[str, Any]) -> str | None:
    training = dict(raw_config.get("training", {}))
    explicit = training.get(
        "loader_multiprocessing_context",
        raw_config.get("loader_multiprocessing_context"),
    )
    if explicit is not None:
        value = str(explicit).strip().lower()
        if value in {"", "default", "none"}:
            return None
        if value not in mp.get_all_start_methods():
            raise ValueError(
                "training.loader_multiprocessing_context must be one of "
                f"{mp.get_all_start_methods()}, got {explicit!r}"
            )
        return value
    methods = set(mp.get_all_start_methods())
    if torch.device(_loader_worker_device(raw_config)).type == "cuda":
        return "spawn" if "spawn" in methods else None
    if "forkserver" in methods:
        return "forkserver"
    if "fork" in methods:
        return "fork"
    if "spawn" in methods:
        return "spawn"
    return None


def _make_batch_dataloader(
    config_source: str | Path | Any,
    *,
    raw_config: dict[str, Any],
    start_batch_index: int,
    batch_count: int | None,
    batch_index_stride: int = 1,
    sample_index_offset: int = 0,
    sample_index_limit: int | None = None,
    sample_mode: str,
    profile: bool = False,
) -> DataLoader | None:
    workers = _loader_worker_count(raw_config)
    if workers <= 0 or (batch_count is not None and int(batch_count) <= 0):
        return None
    dataset = _FiberTrace3DBatchDataset(
        config_source,
        start_batch_index=int(start_batch_index),
        batch_index_stride=int(batch_index_stride),
        sample_index_offset=int(sample_index_offset),
        batch_count=None if batch_count is None else int(batch_count),
        sample_index_limit=sample_index_limit,
        sample_mode=sample_mode,
        worker_device=_loader_worker_device(raw_config),
        profile=profile,
    )
    context = _loader_multiprocessing_context(raw_config)
    kwargs: dict[str, Any] = {}
    if context is not None:
        kwargs["multiprocessing_context"] = context
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        sampler=None,
        num_workers=workers,
        collate_fn=_identity_batch_collate,
        persistent_workers=True,
        prefetch_factor=_loader_prefetch_factor(raw_config),
        pin_memory=False,
        **kwargs,
    )


def _next_training_batch(
    *,
    iterator: Any | None,
    loader: FiberTrace3DLoader,
    sample_index: int,
    sample_index_limit: int | None = None,
    sample_mode: str,
    device: torch.device,
    profile_targets: bool = False,
    output_stride: int = 1,
) -> tuple[FiberTrace3DBatch, float, float, float, float]:
    wait_start = time.perf_counter()
    if iterator is None:
        batch = loader.load_batch(
            sample_index,
            sample_index_limit=sample_index_limit,
            sample_mode=sample_mode,
            device=device,
        )
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        target_start = time.perf_counter()
        batch = materialize_targets(
            batch,
            loader.config,
            profile=profile_targets,
            output_stride=output_stride,
        )
        if profile_targets and device.type == "cuda":
            torch.cuda.synchronize(device)
        target_ms = (time.perf_counter() - target_start) * 1000.0
        return batch, wait_ms + target_ms, wait_ms, 0.0, target_ms

    wait_start_ns = time.time_ns()
    batch = next(iterator)
    if batch.profile_timings_ms is not None:
        batch = replace(batch, profile_timings_ms={
            **batch.profile_timings_ms,
            "coordinator_wait_start_ns": float(wait_start_ns),
            "coordinator_received_ns": float(time.time_ns()),
        })
    wait_ms = (time.perf_counter() - wait_start) * 1000.0
    if isinstance(iterator, FiberTrace3DCudaBatchPipeline):
        timings = batch.profile_timings_ms or {}
        exposed_wait_ms = float(
            timings.get("cuda_pipeline_exposed_queue_wait_ms", 0.0)
        ) + float(timings.get("cuda_pipeline_exposed_event_wait_ms", 0.0))
        return (
            batch,
            exposed_wait_ms,
            exposed_wait_ms,
            float(timings.get("cuda_pipeline_transfer_ms", 0.0)),
            float(timings.get("cuda_pipeline_target_ms", 0.0)),
        )
    to_device_start = time.perf_counter()
    try:
        batch = batch.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        release_current = getattr(iterator, "release_current", None)
        if callable(release_current):
            release_current()
    to_device_ms = (time.perf_counter() - to_device_start) * 1000.0
    target_start = time.perf_counter()
    batch = materialize_targets(
        batch,
        loader.config,
        profile=profile_targets,
        output_stride=output_stride,
    )
    if profile_targets and device.type == "cuda":
        torch.cuda.synchronize(device)
    target_ms = (time.perf_counter() - target_start) * 1000.0
    return batch, wait_ms + to_device_ms + target_ms, wait_ms, to_device_ms, target_ms


_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _process_cpu_seconds(pid: int) -> float | None:
    stat_path = Path("/proc") / str(int(pid)) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        rest = text.rsplit(")", 1)[1].strip().split()
        utime_ticks = int(rest[11])
        stime_ticks = int(rest[12])
    except (IndexError, ValueError):
        return None
    return float(utime_ticks + stime_ticks) / float(_CLK_TCK)


def _dataloader_worker_pids(iterator: Any | None) -> tuple[int, ...]:
    workers = getattr(iterator, "_workers", None)
    if workers is None:
        return ()
    pids: list[int] = []
    for worker in workers:
        pid = getattr(worker, "pid", None)
        if pid is not None:
            pids.append(int(pid))
    return tuple(pids)


def _cpu_seconds_for_pids(pids: tuple[int, ...]) -> float | None:
    total = 0.0
    seen = False
    for pid in pids:
        seconds = _process_cpu_seconds(pid)
        if seconds is None:
            continue
        total += seconds
        seen = True
    return total if seen else None


def _worker_overlap_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    intervals: list[tuple[float, float]] = []
    cpu_ms_total = 0.0
    for row in rows:
        start = float(row.get("worker_item_start_ns", 0.0))
        end = float(row.get("worker_item_end_ns", 0.0))
        if end > start:
            intervals.append((start / 1.0e6, end / 1.0e6))
            cpu_ms_total += float(row.get("worker_item_cpu_ms", 0.0))
    if not intervals:
        return {}
    events: list[tuple[float, int]] = []
    for start_ms, end_ms in intervals:
        events.append((start_ms, 1))
        events.append((end_ms, -1))
    events.sort(key=lambda item: (item[0], -item[1]))
    first = min(start for start, _end in intervals)
    last = max(end for _start, end in intervals)
    active = 0
    prev = events[0][0]
    active_area_ms = 0.0
    max_active = 0
    for timestamp, delta in events:
        if timestamp > prev:
            active_area_ms += active * (timestamp - prev)
            prev = timestamp
        active += delta
        max_active = max(max_active, active)
    span_ms = max(last - first, 1.0e-6)
    construct_rows = sum(1 for row in rows if float(row.get("worker_loader_construct_ms", 0.0)) > 0.0)
    return {
        "items": float(len(intervals)),
        "span_ms": span_ms,
        "avg_active": active_area_ms / span_ms,
        "max_active": float(max_active),
        "worker_cpu_x": cpu_ms_total / span_ms,
        "construct_items": float(construct_rows),
    }


def run_benchmark(config_path: str | Path, *, load_only: bool, batches: int) -> None:
    raw_config = _load_raw_config(config_path)
    loader = FiberTrace3DLoader(load_config(config_path))
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    mixed_precision = _mixed_precision_config_from_training(training, device)
    model = build_fiber_trace_3d_model(raw_config).to(device)
    model.eval()
    direction_weight = float(training.get("direction_weight", 10.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    conditioned_loss_options = _conditioned_loss_options_from_training(training)
    loader_workers = _loader_worker_count(raw_config)
    loader_prefetch_factor = _loader_prefetch_factor(raw_config)
    loader_worker_device = _loader_worker_device(raw_config)
    loader_context = _loader_multiprocessing_context(raw_config)
    sample_index_limit = _training_sample_index_limit(training, loader.sample_count)
    dataloader = _make_batch_dataloader(
        config_path,
        raw_config=raw_config,
        start_batch_index=0,
        batch_count=int(batches),
        sample_index_limit=sample_index_limit,
        sample_mode="random",
        profile=True,
    )
    iterator = iter(dataloader) if dataloader is not None else None
    cpu_pids = (os.getpid(),) + _dataloader_worker_pids(iterator)
    print(
        "fiber_trace_3d benchmark: "
        f"loader_workers={loader_workers} loader_prefetch_factor={loader_prefetch_factor} "
        f"loader_worker_device={loader_worker_device} "
        f"loader_multiprocessing_context={loader_context or 'default'} "
        f"device={device} load_only={bool(load_only)} "
        f"mixed_precision={mixed_precision.mode} autocast_enabled={mixed_precision.enabled}",
        flush=True,
    )
    print(
        "batch patches total_ms load_ms wait_ms to_device_ms target_ms fw_ms "
        "worker_ms worker_cpu cpu/w construct_ms desc_ms params_ms geom_ms coord_ms valid_ms "
        "sample_ms tensor_ms value_ms spec_ms line_ms map_ms clip_ms "
        "gpu_ms line_idx cp_idx scatter dir_enc gpu_mask segs linePts dirPts posK "
        "stack_ms cpu_ms cpu_x"
    )
    profile_rows: list[dict[str, float]] = []
    for batch_index in range(1, int(batches) + 1):
        start = time.perf_counter()
        cpu_start = _cpu_seconds_for_pids(cpu_pids)
        batch, load_ms, wait_ms, to_device_ms, target_ms = _next_training_batch(
            iterator=iterator,
            loader=loader,
            sample_index=(batch_index - 1) * loader.config.batch_size,
            sample_index_limit=sample_index_limit,
            sample_mode="random",
            device=device,
            profile_targets=True,
            output_stride=_model_output_stride(model),
        )
        fw_ms = 0.0
        if not load_only:
            fw_start = time.perf_counter()
            with torch.no_grad():
                with _autocast_context(mixed_precision):
                    _forward_loss(
                        model,
                        batch,
                        direction_weight=direction_weight,
                        presence_weight=presence_weight,
                        conditioned_loss_options=conditioned_loss_options,
                        backward=False,
                    )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            fw_ms = (time.perf_counter() - fw_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        cpu_end = _cpu_seconds_for_pids(cpu_pids)
        cpu_ms = math.nan
        cpu_factor = math.nan
        if cpu_start is not None and cpu_end is not None and cpu_end >= cpu_start:
            cpu_ms = (cpu_end - cpu_start) * 1000.0
            cpu_factor = cpu_ms / max(total_ms, 1.0e-6)
        timings = batch.profile_timings_ms or {}

        def timing_ms(key: str) -> float:
            return float(timings.get(key, 0.0))

        profile_rows.append({key: float(value) for key, value in timings.items()})
        print(
            f"{batch_index:5d} {loader.config.batch_size:7d} "
            f"{total_ms:8.2f} {load_ms:8.2f} {wait_ms:8.2f} {to_device_ms:12.2f} "
            f"{target_ms:9.2f} {fw_ms:8.2f} {timing_ms('batch_total_ms'):9.2f} "
            f"{timing_ms('batch_cpu_ms'):10.2f} "
            f"{timing_ms('batch_cpu_ms') / max(timing_ms('batch_total_ms'), 1.0e-6):5.2f} "
            f"{timing_ms('worker_loader_construct_ms'):12.2f} "
            f"{timing_ms('descriptor_ms'):8.2f} {timing_ms('augment_params_ms'):9.2f} "
            f"{timing_ms('geometry_ms'):7.2f} {timing_ms('coord_to_numpy_ms'):8.2f} "
            f"{timing_ms('coord_valid_ms'):8.2f} {timing_ms('volume_sample_ms'):9.2f} "
            f"{timing_ms('volume_tensor_ms'):9.2f} "
            f"{timing_ms('value_augmentation_ms'):8.2f} "
            f"{timing_ms('target_spec_total_ms'):7.2f} "
            f"{timing_ms('target_line_window_ms'):7.2f} "
            f"{timing_ms('target_points_to_output_ms'):6.2f} "
            f"{timing_ms('target_clip_ms'):7.2f} "
            f"{timing_ms('target_gpu_total_ms'):7.2f} "
            f"{timing_ms('target_line_index_ms'):8.2f} "
            f"{timing_ms('target_cp_index_ms'):6.2f} "
            f"{timing_ms('target_presence_scatter_ms'):7.2f} "
            f"{timing_ms('target_direction_encode_ms'):7.2f} "
            f"{timing_ms('target_gpu_mask_ms'):8.2f} "
            f"{timing_ms('target_line_segments'):5.0f} "
            f"{timing_ms('target_line_points'):7.0f} "
            f"{timing_ms('target_direction_points'):6.0f} "
            f"{timing_ms('target_gpu_positive_voxels') / 1.0e3:5.1f} "
            f"{timing_ms('batch_stack_ms'):8.2f} "
            f"{cpu_ms:8.2f} {cpu_factor:6.2f}",
            flush=True,
        )
    overlap = _worker_overlap_summary(profile_rows)
    if overlap:
        print(
            "fiber_trace_3d worker overlap: "
            f"items={int(overlap['items'])} span_ms={overlap['span_ms']:.1f} "
            f"avg_active={overlap['avg_active']:.2f} max_active={int(overlap['max_active'])} "
            f"worker_cpu_x={overlap['worker_cpu_x']:.2f} "
            f"construct_items={int(overlap['construct_items'])}",
            flush=True,
        )
    iterator = None
    dataloader = None


def run_prefetch(
    config_path: str | Path,
    *,
    prefetch_steps: int | None,
    workers: int | None,
) -> None:
    raw_config = _load_raw_config(config_path)
    training = dict(raw_config.get("training", {}))
    loader = FiberTrace3DLoader(load_config(config_path))
    sample_index_limit = _training_sample_index_limit(training, loader.sample_count)
    sample_count = _resolve_prefetch_sample_count(
        training=training,
        loader_sample_count=loader.sample_count,
        batch_size=loader.config.batch_size,
        prefetch_steps=prefetch_steps,
    )
    summary = loader.prefetch(
        0,
        sample_count,
        workers=workers,
        sample_index_limit=sample_index_limit,
        sample_mode="random",
    )
    summaries: dict[str, Any] = {"train": summary}
    if raw_config.get("test_datasets") and (prefetch_steps == 0 or prefetch_steps is None):
        test_raw = _make_test_loader_raw_config(raw_config, training)
        tmp_path = Path("/tmp") / f"fiber_trace_3d_prefetch_test_{int(time.time() * 1000)}.json"
        tmp_path.write_text(json.dumps(_json_safe(test_raw)), encoding="utf-8")
        try:
            test_loader = FiberTrace3DLoader(load_config(tmp_path))
            summaries["test"] = test_loader.prefetch(
                0,
                test_loader.sample_count,
                workers=workers,
                sample_index_limit=0,
                sample_mode="flat",
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    print("fiber_trace_3d prefetch summary: " + json.dumps(summaries, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-steps", type=int, default=None)
    parser.add_argument("--prefetch-workers", type=int, default=None)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=10)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--trace2cp-vis", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fiber-json", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--trace2cp-step-px", type=float, default=None)
    parser.add_argument("--trace2cp-rf-margin-px", type=float, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    launch_mode = parser.add_mutually_exclusive_group()
    launch_mode.add_argument("--manager-socket", type=Path, default=None,
                            help="override manager socket (default: available /run/villa-gpu-manager/manager.sock, otherwise ~/.local/run/villa-gpu-manager/manager.sock); training is managed by default")
    launch_mode.add_argument("--unmanaged", action="store_true",
                            help="explicit testing/debug only: bypass cooperative GPU management")
    parser.add_argument(
        "--test-hang-diagnostics",
        action="store_true",
        help="write per-rank hang diagnostics and arm the rank-0 test watchdog",
    )
    args = parser.parse_args()
    auxiliary_mode = args.prefetch or args.trace2cp_vis or args.benchmark
    if (args.manager_socket is not None or args.unmanaged) and auxiliary_mode:
        raise SystemExit("--manager-socket and --unmanaged apply only to training")
    if args.test_hang_diagnostics and auxiliary_mode:
        raise SystemExit("--test-hang-diagnostics is supported only for normal training")
    if args.prefetch:
        _require_single_process_cli_mode("--prefetch")
        run_prefetch(
            args.config,
            prefetch_steps=args.prefetch_steps,
            workers=args.prefetch_workers,
        )
    elif args.trace2cp_vis:
        _require_single_process_cli_mode("--trace2cp-vis")
        if args.checkpoint is None:
            raise SystemExit("--trace2cp-vis requires --checkpoint")
        if args.export_dir is None:
            raise SystemExit("--trace2cp-vis requires --export-dir")
        run_trace2cp_vis(
            args.config,
            checkpoint=args.checkpoint,
            export_dir=args.export_dir,
            sample_index=int(args.sample_index),
            fiber_json=args.fiber_json,
            step_px=args.trace2cp_step_px,
            rf_margin_px=args.trace2cp_rf_margin_px,
        )
    elif args.benchmark:
        _require_single_process_cli_mode("--benchmark")
        run_benchmark(
            args.config,
            load_only=bool(args.load_only),
            batches=int(args.benchmark_batches),
        )
    else:
        run_training(args.config, resume_checkpoint=args.resume,
            test_hang_diagnostics=bool(args.test_hang_diagnostics),
            manager_socket=str(args.manager_socket) if args.manager_socket is not None else None,
            unmanaged=bool(args.unmanaged))


if __name__ == "__main__":
    main()
