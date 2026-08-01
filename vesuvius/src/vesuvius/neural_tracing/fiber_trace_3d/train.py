from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    DEFAULT_VOLUME_CACHE_MEMORY_MIB,
    FiberTrace3DBatch,
    FiberTrace3DLoader,
    _normalize_image,
    _read_raw_block,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
    encode_lasagna_direction_3x2,
)
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
class _DistributedConfig:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    backend: str
    device: torch.device


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
            f"{mode_name} is single-process only. Run normal training under torchrun, "
            "or run this subcommand without torchrun/WORLD_SIZE>1."
        )


def _distributed_init(config: _DistributedConfig) -> None:
    if not config.enabled:
        return
    if config.device.type == "cuda":
        torch.cuda.set_device(config.local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=config.backend)


def _distributed_barrier(config: _DistributedConfig) -> None:
    if config.enabled and dist.is_initialized():
        dist.barrier()


def _distributed_cleanup(config: _DistributedConfig) -> None:
    if config.enabled and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_broadcast_object(value: Any, config: _DistributedConfig, *, src: int = 0) -> Any:
    if not config.enabled:
        return value
    objects = [value]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


def _distributed_should_use_sync_batchnorm(config: _DistributedConfig) -> bool:
    return bool(config.enabled and config.device.type == "cuda")


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (DistributedDataParallel, torch.nn.DataParallel)):
        return model.module
    return model


def _wrap_distributed_model(
    model: torch.nn.Module,
    config: _DistributedConfig,
) -> torch.nn.Module:
    if not config.enabled:
        return model
    if config.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
        )
    return DistributedDataParallel(model)


def _mixed_precision_config_from_training(
    training: dict[str, Any],
    device: torch.device,
) -> _MixedPrecisionConfig:
    raw = training.get("mixed_precision", "off")
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
    test_raw["datasets"] = raw_config["test_datasets"]
    test_raw.pop("test_datasets", None)
    if not bool(training.get("test_augment_enabled", False)):
        test_raw["augment_enabled"] = False
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


def _mean_over_nonempty_groups(
    group_sum: torch.Tensor,
    group_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = group_count > 0.0
    group_mean = torch.where(
        valid,
        group_sum / group_count.clamp_min(1.0),
        torch.zeros_like(group_sum),
    )
    valid_count = valid.to(dtype=group_sum.dtype).sum()
    mean = group_mean.sum() / valid_count.clamp_min(1.0)
    return mean, valid_count


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
    selected_chunk_branch = _enforce_two_branch_min_fraction(
        selected_chunk_branch,
        chunk_score,
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
) -> dict[str, torch.Tensor]:
    require_materialized_targets(batch)
    assert batch.direction_indices_bzyx is not None
    assert batch.direction_target_sparse is not None
    assert batch.direction_weight_sparse is not None
    assert batch.presence_target is not None
    assert batch.presence_mask is not None
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
        direction_loss = direction_error.mean()
        selected_axis = pred_axis[row_index, selected_branch]
        agreement = torch.abs(torch.sum(selected_axis * target_axis, dim=-1)).clamp(0.0, 1.0)
        angle_mean_deg = torch.rad2deg(torch.acos(agreement)).mean()
        selected_presence = pred_presence_sparse[row_index, selected_branch]
        positive_presence_mask = batch.presence_mask[
            indices[:, 0],
            0,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        positive_presence_bce = _safe_probability_bce(
            selected_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.ones_like(selected_presence),
            reduction="none",
        )
        branch_patch_count = int(pred_presences.shape[0]) * branch_count
        positive_group_index = indices[:, 0] * branch_count + selected_branch
        positive_mask_f = positive_presence_mask.to(dtype=positive_presence_bce.dtype)
        positive_group_sum = torch.zeros(
            (branch_patch_count,),
            dtype=positive_presence_bce.dtype,
            device=positive_presence_bce.device,
        )
        positive_group_count = torch.zeros_like(positive_group_sum)
        positive_group_sum.scatter_add_(
            0,
            positive_group_index,
            positive_presence_bce * positive_mask_f,
        )
        positive_group_count.scatter_add_(0, positive_group_index, positive_mask_f)
        positive_presence_loss, positive_presence_group_count = _mean_over_nonempty_groups(
            positive_group_sum,
            positive_group_count,
        )
        selected_score_mean = score[row_index, selected_branch].mean()
        branch0_fraction = (selected_branch == 0).to(dtype=torch.float32).mean()
        branch1_fraction = (
            (selected_branch == 1).to(dtype=torch.float32).mean()
            if branch_count > 1
            else selected_branch.to(dtype=torch.float32).sum() * 0.0
        )
    else:
        direction_loss = pred_dirs.sum() * 0.0
        angle_mean_deg = pred_dirs.sum() * 0.0
        positive_presence_loss = pred_presences.sum() * 0.0
        positive_presence_group_count = torch.zeros(
            (),
            dtype=pred_presences.dtype,
            device=pred_presences.device,
        )
        selected_score_mean = pred_presences.sum() * 0.0
        branch0_fraction = pred_presences.sum() * 0.0
        branch1_fraction = pred_presences.sum() * 0.0

    presence_target = batch.presence_target.unsqueeze(1).expand_as(pred_presences)
    presence_mask = batch.presence_mask.unsqueeze(1).expand_as(pred_presences)
    negative_mask = (presence_target <= 0.5) & presence_mask
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
    negative_presence_loss, negative_presence_group_count = _mean_over_nonempty_groups(
        negative_group_sum,
        negative_group_count,
    )
    has_positive = (positive_presence_group_count > 0.0).to(dtype=pred_presences.dtype)
    has_negative = (negative_presence_group_count > 0.0).to(dtype=pred_presences.dtype)
    presence_loss = (
        positive_presence_loss * has_positive
        + negative_presence_loss * has_negative
    )
    total = float(direction_weight) * direction_loss + float(presence_weight) * presence_loss
    return {
        "total": total,
        "direction": direction_loss,
        "presence": presence_loss,
        "angle_mean_deg": angle_mean_deg,
        "branch0_fraction": branch0_fraction,
        "branch1_fraction": branch1_fraction,
        "selected_score_mean": selected_score_mean,
    }


def compute_conditioned_losses(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    perpendicular_jitter_degrees: float = 45.0,
    positive_query_weight: float = 1.0,
    negative_query_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    require_materialized_targets(batch)
    assert batch.direction_indices_bzyx is not None
    assert batch.direction_target_sparse is not None
    assert batch.direction_weight_sparse is not None
    assert batch.presence_target is not None
    assert batch.presence_mask is not None
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
        direction_loss = direction_error.mean(dim=(0, 2)).mean()
        pred_axis = decode_lasagna_direction_3x2_analytic(pred_dirs.reshape(-1, 6)).reshape(
            int(indices.shape[0]),
            positive_query_count,
            3,
        )
        agreement = torch.abs(torch.sum(pred_axis * target_axis[:, None, :], dim=-1)).clamp(
            0.0,
            1.0,
        )
        angle_mean_deg = torch.rad2deg(torch.acos(agreement)).mean()
        positive_presence_mask = batch.presence_mask.to(device=zero_output.device)[
            indices[:, 0],
            0,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        positive_presence_bce = _safe_probability_bce(
            pred_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
            torch.ones_like(pred_presence),
            reduction="none",
        )
        positive_mask_f = positive_presence_mask[:, None].to(dtype=positive_presence_bce.dtype)
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
            (positive_presence_bce * positive_mask_f).reshape(-1),
        )
        positive_group_count.scatter_add_(
            0,
            group_index.reshape(-1),
            positive_mask_f.expand_as(positive_presence_bce).reshape(-1),
        )
        positive_presence_loss, positive_presence_group_count = _mean_over_nonempty_groups(
            positive_group_sum,
            positive_group_count,
        )
        selected_score_mean = (agreement * pred_presence.clamp(0.0, 1.0)).mean()
        branch0_fraction = torch.full(
            (),
            0.5,
            dtype=pred_presence.dtype,
            device=pred_presence.device,
        )
        branch1_fraction = torch.full_like(branch0_fraction, 0.5)
    else:
        direction_loss = zero_output.sum() * 0.0
        angle_mean_deg = zero_output.sum() * 0.0
        positive_presence_loss = zero_output.sum() * 0.0
        positive_presence_group_count = torch.zeros(
            (),
            dtype=zero_output.dtype,
            device=zero_output.device,
        )
        selected_score_mean = zero_output.sum() * 0.0
        branch0_fraction = zero_output.sum() * 0.0
        branch1_fraction = zero_output.sum() * 0.0

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
    negative_mask_f = batch.presence_mask.to(
        device=negative_presence_bce.device,
        dtype=negative_presence_bce.dtype,
    )[:, None].expand_as(negative_presence_bce)
    negative_group_sum = (negative_presence_bce * negative_mask_f).sum(
        dim=(2, 3, 4, 5)
    ).reshape(-1)
    negative_group_count = negative_mask_f.sum(dim=(2, 3, 4, 5)).reshape(-1)
    negative_presence_loss, negative_presence_group_count = _mean_over_nonempty_groups(
        negative_group_sum,
        negative_group_count,
    )
    has_positive = (positive_presence_group_count > 0.0).to(dtype=zero_output.dtype)
    has_negative = (negative_presence_group_count > 0.0).to(dtype=zero_output.dtype)
    positive_component = (
        float(positive_query_weight) * positive_presence_loss * has_positive
    )
    negative_component = (
        float(negative_query_weight) * negative_presence_loss * has_negative
    )
    presence_loss = positive_component + negative_component
    total = float(direction_weight) * direction_loss + float(presence_weight) * presence_loss
    return {
        "total": total,
        "direction": direction_loss,
        "presence": presence_loss,
        "angle_mean_deg": angle_mean_deg,
        "branch0_fraction": branch0_fraction,
        "branch1_fraction": branch1_fraction,
        "selected_score_mean": selected_score_mean,
    }


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
) -> None:
    snapshot_model = _unwrap_model(model)
    payload = {
        "model": snapshot_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "config": _json_safe(config),
        "metric": metric,
        "metric_name": metric_name,
    }
    if _grad_scaler_enabled(grad_scaler):
        payload["grad_scaler"] = grad_scaler.state_dict()
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
) -> int:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
        if optimizer_hparams is not None:
            _apply_optimizer_hparams(optimizer, optimizer_hparams)
    if (
        _grad_scaler_enabled(grad_scaler)
        and isinstance(payload, dict)
        and "grad_scaler" in payload
    ):
        grad_scaler.load_state_dict(payload["grad_scaler"])
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
    if int(cfg.control_points) == 0:
        sample_count = int(geometry_loader.sample_count)
        start_sample_index = 0
        sample_mode = "flat"
    else:
        sample_count = int(cfg.control_points)
        start_sample_index = int(cfg.start_sample_index)
        sample_mode = str(cfg.sample_mode)
    errors: list[float] = []
    raw_errors: list[float] = []
    skipped = 0
    first_skip = ""
    for offset in range(max(1, sample_count)):
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
            skipped += 1
            if not first_skip:
                first_skip = " ".join(str(exc).split())
            continue
        errors.append(float(score.trace2cp_error))
        raw_errors.append(float(score.raw_y_error_px))
    if not errors:
        raise ValueError(
            "3D test Trace2CP metric found no valid CP-to-next-CP segments: "
            f"start_sample_index={start_sample_index} sample_count={sample_count} "
            f"skipped={skipped} first_skip='{first_skip}'"
        )
    return _Trace2Cp3DMetricEvalResult(
        error_mean=float(np.mean(np.asarray(errors, dtype=np.float64))),
        raw_y_error_mean_px=float(np.mean(np.asarray(raw_errors, dtype=np.float64))),
        segments=len(errors),
        skipped_segments=int(skipped),
        first_skip_reason=first_skip,
    )


@torch.no_grad()
def evaluate_dense_loss(
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
) -> dict[str, float]:
    model.eval()
    total_rows: list[dict[str, float]] = []
    consumed = 0
    while consumed < sample_count:
        batch = loader.load_batch(
            start_sample_index + consumed,
            sample_mode=sample_mode,
            sample_index_limit=sample_index_limit,
            device=device,
        )
        batch = materialize_targets(batch, loader.config)
        take = min(int(batch.volume.shape[0]), sample_count - consumed)
        if take < int(batch.volume.shape[0]):
            batch = _slice_batch(batch, 0, take)
        with _autocast_context(mixed_precision):
            rows = _forward_loss(
                model,
                batch,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
                conditioned_loss_options=conditioned_loss_options,
                backward=False,
            )
        total_rows.append(rows)
        consumed += take
    model.train()
    if not total_rows:
        return {
            "total": math.inf,
            "direction": math.inf,
            "presence": math.inf,
            "angle_mean_deg": math.inf,
            "branch0_fraction": math.inf,
            "branch1_fraction": math.inf,
            "selected_score_mean": math.inf,
        }
    return {
        key: float(sum(row[key] for row in total_rows) / len(total_rows))
        for key in total_rows[0]
    }


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
        volume=batch.volume[start:stop],
        valid_mask=batch.valid_mask[start:stop],
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


def _forward_loss_tensors(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    conditioned_loss_options: dict[str, float] | None = None,
    training_loss: bool,
) -> dict[str, torch.Tensor]:
    if _model_uses_conditioned_decoder(model):
        return compute_conditioned_losses(
            model,
            batch,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
            **dict(conditioned_loss_options or {}),
        )
    output = model(batch.volume)
    return compute_losses(
        output,
        batch,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
        branch_selection_mode=(
            "train_offset_grid_min_fraction" if training_loss else "eval_voxel"
        ),
    )


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


def run_training(config_path: str | Path, *, resume_checkpoint: str | Path | None = None) -> None:
    raw_config = _load_raw_config(config_path)
    loader_config = load_config(config_path)
    training = dict(raw_config.get("training", {}))
    base_device = _device_from_training(training)
    distributed = _distributed_config_from_env(base_device)
    _distributed_init(distributed)
    device = distributed.device
    writer = None
    train_dataloader = None
    train_iterator = None
    loader = FiberTrace3DLoader(loader_config)
    test_loader = None
    if distributed.is_main and raw_config.get("test_datasets"):
        test_raw = _make_test_loader_raw_config(raw_config, training)
        tmp_path = Path("/tmp") / f"fiber_trace_3d_test_{int(time.time() * 1000)}.json"
        tmp_path.write_text(json.dumps(_json_safe(test_raw)), encoding="utf-8")
        try:
            test_loader = FiberTrace3DLoader(load_config(tmp_path))
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
    trace2cp_cfg = _trace2cp_3d_config(raw_config)
    trace2cp_loader = (
        _make_trace2cp_geometry_loader(raw_config, trace2cp_cfg)
        if distributed.is_main and trace2cp_cfg.enabled
        else None
    )

    raw_model = build_fiber_trace_3d_model(raw_config)
    if _distributed_should_use_sync_batchnorm(distributed):
        raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
    raw_model = raw_model.to(device)
    mixed_precision = _mixed_precision_config_from_training(training, device)
    grad_scaler = _make_grad_scaler(mixed_precision)
    optimizer_hparams = _optimizer_hparams_from_training(training)
    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        **optimizer_hparams,
    )
    resume = (
        str(resume_checkpoint)
        if resume_checkpoint is not None
        else training.get("resume") or raw_config.get("resume")
    )
    start_step = 0
    if resume:
        start_step = _load_snapshot(
            resume,
            model=raw_model,
            optimizer=optimizer,
            optimizer_hparams=optimizer_hparams,
            grad_scaler=grad_scaler,
            map_location=device,
        )

    train_model = _wrap_distributed_model(raw_model, distributed)
    run_datestr = str(
        training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_datestr = _distributed_broadcast_object(
        run_datestr if distributed.is_main else None,
        distributed,
    )
    run_dir, snapshot_dir = _resolve_run_layout(
        raw_config,
        date_str_override=str(run_datestr),
    )
    _distributed_barrier(distributed)
    effective_config = _json_safe(raw_config)
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
    writer = _make_summary_writer(
        run_dir,
        enabled=distributed.is_main and bool(training.get("tensorboard_enabled", True)),
    )
    if writer is not None:
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
            "the recurrent output conditioned on the first decoded direction. "
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
            "the recurrent output conditioned on the first decoded direction. "
            "Multiple batch samples are concatenated side by side when configured.",
            0,
        )

    max_steps_raw = int(training.get("max_steps", 1))
    if max_steps_raw < 0:
        raise ValueError("training.max_steps must be >= 0")
    max_steps: int | None = None if max_steps_raw == 0 else max_steps_raw
    if max_steps is not None and max_steps <= int(start_step):
        raise ValueError(
            "training.max_steps must be greater than checkpoint step when resuming: "
            f"max_steps={max_steps} checkpoint_step={start_step}"
        )
    sample_index_limit = _training_sample_index_limit(training, loader.sample_count)
    scalar_interval = int(training.get("scalar_log_interval", 100))
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
    best_metric = math.inf

    if distributed.is_main:
        print(
            "fiber_trace_3d train: "
            f"samples={loader.sample_count} batch_size={loader.config.batch_size} "
            f"global_batch_size={loader.config.batch_size * distributed.world_size} "
            f"max_sample_index={sample_index_limit} "
            f"patch_shape_zyx={loader.config.patch_shape_zyx} device={device} run_dir={run_dir} "
            f"ddp_enabled={distributed.enabled} ddp_world_size={distributed.world_size} "
            f"ddp_backend={distributed.backend or 'none'} "
            f"sync_batchnorm={_distributed_should_use_sync_batchnorm(distributed)} "
            f"trace2cp_enabled={bool(trace2cp_loader is not None)} "
            f"loader_workers={loader_workers} loader_prefetch_factor={loader_prefetch_factor} "
            f"loader_worker_device={loader_worker_device} "
            f"loader_multiprocessing_context={loader_context or 'default'} "
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

    def run_configured_tests(step: int) -> tuple[float | None, str | None]:
        metric: float | None = None
        metric_name: str | None = None
        if test_loader is not None and test_interval > 0:
            test_losses = evaluate_dense_loss(
                raw_model,
                test_loader,
                device=device,
                start_sample_index=test_start_sample_index,
                sample_count=test_control_points,
                sample_mode=test_sample_mode,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
                conditioned_loss_options=conditioned_loss_options,
                mixed_precision=mixed_precision,
            )
            metric = float(test_losses["total"])
            metric_name = "test/loss_total"
            print(
                f"test step={step} loss_total={test_losses['total']:.6f} "
                f"loss_direction={test_losses['direction']:.6f} "
                f"loss_presence={test_losses['presence']:.6f} "
                f"angle_mean_deg={test_losses['angle_mean_deg']:.2f} "
                f"branch0={test_losses['branch0_fraction']:.3f} "
                f"branch1={test_losses['branch1_fraction']:.3f} "
                f"selected_score={test_losses['selected_score_mean']:.3f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("test/loss_total", test_losses["total"], step)
                writer.add_scalar("test/loss_direction", test_losses["direction"], step)
                writer.add_scalar("test/loss_presence", test_losses["presence"], step)
                writer.add_scalar("test/angle_mean_deg", test_losses["angle_mean_deg"], step)
                writer.add_scalar("test/branch0_fraction", test_losses["branch0_fraction"], step)
                writer.add_scalar("test/branch1_fraction", test_losses["branch1_fraction"], step)
                writer.add_scalar("test/selected_score_mean", test_losses["selected_score_mean"], step)
                if test_sample_vis_interval > 0 and step % test_sample_vis_interval == 0:
                    vis_batch = test_loader.load_batch(
                        test_start_sample_index,
                        sample_mode=test_sample_mode,
                        device=device,
                    )
                    vis_batch = materialize_targets(vis_batch, test_loader.config)
                    _write_3d_sample_sheet(
                        writer,
                        "test_sample_3d/principal_slices",
                        raw_model,
                        vis_batch,
                        step,
                        sample_count=test_sample_vis_count,
                        mixed_precision=mixed_precision,
                    )
        if trace2cp_loader is not None and test_interval > 0:
            trace2cp_metric = _evaluate_trace2cp_metric_fixed_set_3d(
                raw_model,
                trace2cp_loader,
                image_normalization=loader.config.image_normalization,
                cfg=trace2cp_cfg,
                device=device,
                mixed_precision=mixed_precision,
            )
            print(
                f"test_trace2cp step={step} trace2cp_error={trace2cp_metric.error_mean:.6f} "
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
        if writer is not None and metric is not None:
            writer.flush()
        return metric, metric_name

    initial_metric, initial_metric_name = (
        run_configured_tests(start_step) if distributed.is_main else (None, None)
    )
    initial_metric = _distributed_broadcast_object(initial_metric, distributed)
    initial_metric_name = _distributed_broadcast_object(initial_metric_name, distributed)
    if initial_metric is not None and initial_metric_name is not None:
        best_metric = float(initial_metric)
        if distributed.is_main:
            _save_snapshot(
                snapshot_dir / "best.pt",
                model=raw_model,
                optimizer=optimizer,
                step=start_step,
                config=raw_config,
                metric=best_metric,
                metric_name=initial_metric_name,
                grad_scaler=grad_scaler,
            )
    _distributed_barrier(distributed)

    remaining_steps = None if max_steps is None else max(0, int(max_steps) - int(start_step))
    train_dataloader = _make_batch_dataloader(
        config_path,
        raw_config=raw_config,
        start_batch_index=start_step * distributed.world_size + distributed.rank,
        batch_index_stride=distributed.world_size,
        batch_count=remaining_steps,
        sample_index_limit=sample_index_limit,
        sample_mode="random",
    )
    train_iterator = iter(train_dataloader) if train_dataloader is not None else None
    try:
        step = int(start_step)
        while max_steps is None or step < max_steps:
            step += 1
            sample_index = _distributed_training_sample_index(
                step,
                batch_size=loader.config.batch_size,
                config=distributed,
            )
            batch, load_ms, wait_ms, to_device_ms, target_ms = _next_training_batch(
                iterator=train_iterator,
                loader=loader,
                sample_index=sample_index,
                sample_index_limit=sample_index_limit,
                sample_mode="random",
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            fw_start = time.perf_counter()
            with _autocast_context(mixed_precision):
                loss_tensors = _forward_loss_tensors(
                    train_model,
                    batch,
                    direction_weight=direction_weight,
                    presence_weight=presence_weight,
                    conditioned_loss_options=conditioned_loss_options,
                    training_loss=True,
                )
            if _grad_scaler_enabled(grad_scaler):
                assert grad_scaler is not None
                grad_scaler.scale(loss_tensors["total"]).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss_tensors["total"].backward()
                optimizer.step()
            reduced_loss_tensors = _distributed_mean_loss_tensors(loss_tensors, distributed)
            losses = _losses_to_float_dict(reduced_loss_tensors)
            step_ms = (time.perf_counter() - fw_start) * 1000.0

            if distributed.is_main and (step <= 100 or step % scalar_interval == 0):
                print(
                    f"step={step} loss_total={losses['total']:.6f} "
                    f"loss_direction={losses['direction']:.6f} "
                    f"loss_presence={losses['presence']:.6f} "
                    f"angle_mean_deg={losses['angle_mean_deg']:.2f} "
                    f"branch0={losses['branch0_fraction']:.3f} "
                    f"branch1={losses['branch1_fraction']:.3f} "
                    f"selected_score={losses['selected_score_mean']:.3f} "
                    f"load_ms={load_ms:.1f} wait_ms={wait_ms:.1f} "
                    f"to_device_ms={to_device_ms:.1f} "
                    f"target_ms={target_ms:.1f} "
                    f"fw_bw_step_ms={step_ms:.1f}",
                    flush=True,
                )
            if writer is not None and (step == 1 or step % scalar_interval == 0):
                writer.add_scalar("train/loss_total", losses["total"], step)
                writer.add_scalar("train/loss_direction", losses["direction"], step)
                writer.add_scalar("train/loss_presence", losses["presence"], step)
                writer.add_scalar("train/angle_mean_deg", losses["angle_mean_deg"], step)
                writer.add_scalar("train/branch0_fraction", losses["branch0_fraction"], step)
                writer.add_scalar("train/branch1_fraction", losses["branch1_fraction"], step)
                writer.add_scalar("train/selected_score_mean", losses["selected_score_mean"], step)
                writer.add_scalar("timing/load_ms", load_ms, step)
                writer.add_scalar("timing/load_wait_ms", wait_ms, step)
                writer.add_scalar("timing/batch_to_device_ms", to_device_ms, step)
                writer.add_scalar("timing/target_ms", target_ms, step)
                writer.add_scalar("timing/fw_bw_step_ms", step_ms, step)
            train_vis_due = sample_vis_interval > 0 and (
                step == 1 or step % sample_vis_interval == 0
            )
            if writer is not None and train_vis_due:
                _write_3d_sample_sheet(
                    writer,
                    "train_sample_3d/principal_slices",
                    raw_model,
                    batch,
                    step,
                    sample_count=sample_vis_count,
                    mixed_precision=mixed_precision,
                )
            if train_vis_due:
                _distributed_barrier(distributed)

            test_metric = None
            test_metric_name = None
            if test_interval > 0 and step % test_interval == 0:
                test_metric, test_metric_name = (
                    run_configured_tests(step) if distributed.is_main else (None, None)
                )
                test_metric = _distributed_broadcast_object(test_metric, distributed)
                test_metric_name = _distributed_broadcast_object(test_metric_name, distributed)

            if test_metric is not None and step % checkpoint_interval == 0:
                if distributed.is_main:
                    _save_snapshot(
                        snapshot_dir / "current.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        step=step,
                        config=raw_config,
                        metric=test_metric,
                        metric_name=test_metric_name,
                        grad_scaler=grad_scaler,
                    )
                _distributed_barrier(distributed)
            if (
                test_metric is not None
                and kept_snapshot_interval > 0
                and step % kept_snapshot_interval == 0
            ):
                if distributed.is_main:
                    _save_snapshot(
                        snapshot_dir / f"step_{step:08d}.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        step=step,
                        config=raw_config,
                        metric=test_metric,
                        metric_name=test_metric_name,
                        grad_scaler=grad_scaler,
                    )
                _distributed_barrier(distributed)
            if test_metric is not None and test_metric < best_metric:
                best_metric = float(test_metric)
                if distributed.is_main:
                    _save_snapshot(
                        snapshot_dir / "best.pt",
                        model=raw_model,
                        optimizer=optimizer,
                        step=step,
                        config=raw_config,
                        metric=best_metric,
                        metric_name=test_metric_name,
                        grad_scaler=grad_scaler,
                    )
                _distributed_barrier(distributed)
    finally:
        if train_dataloader is not None:
            train_iterator = None
            train_dataloader = None
        if writer is not None:
            writer.flush()
            writer.close()
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
    volume = batch.volume[0, 0].detach().cpu().numpy()
    valid = batch.valid_mask[0, 0].detach().cpu().numpy().astype(bool)
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
            valid_czyx=batch.valid_mask[0],
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
            valid_czyx=batch.valid_mask[0],
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


def _write_3d_sample_sheet(
    writer: Any,
    tag: str,
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    step: int,
    *,
    sample_count: int = 1,
    mixed_precision: _MixedPrecisionConfig | None = None,
) -> None:
    was_training = bool(model.training)
    model.eval()
    take = min(max(1, int(sample_count)), int(batch.volume.shape[0]))
    with torch.no_grad():
        with _autocast_context(mixed_precision):
            if _model_uses_conditioned_decoder(model) and hasattr(
                model,
                "forward_recurrent_grouped",
            ):
                vis_output = model.forward_recurrent_grouped(batch.volume[:take], steps=2)
            else:
                vis_output = model(batch.volume[:take])
        vis_output = vis_output.float()
    if was_training:
        model.train()
    writer.add_image(
        tag,
        _make_train_sample_3d_contact_sheet(batch, vis_output, sample_count=take),
        int(step),
        dataformats="HWC",
    )


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
        sample_index = batch_index * int(loader.config.batch_size)
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
        batch = materialize_targets(batch, loader.config, profile=profile_targets)
        if profile_targets and device.type == "cuda":
            torch.cuda.synchronize(device)
        target_ms = (time.perf_counter() - target_start) * 1000.0
        return batch, wait_ms + target_ms, wait_ms, 0.0, target_ms

    batch = next(iterator)
    wait_ms = (time.perf_counter() - wait_start) * 1000.0
    to_device_start = time.perf_counter()
    batch = batch.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    to_device_ms = (time.perf_counter() - to_device_start) * 1000.0
    target_start = time.perf_counter()
    batch = materialize_targets(batch, loader.config, profile=profile_targets)
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
    args = parser.parse_args()
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
        run_training(args.config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
