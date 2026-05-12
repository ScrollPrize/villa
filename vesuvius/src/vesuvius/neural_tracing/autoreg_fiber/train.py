from __future__ import annotations

import functools
import importlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from vesuvius.models.training.optimizers import create_optimizer
from vesuvius.neural_tracing.autoreg_fiber.config import load_autoreg_fiber_config, validate_autoreg_fiber_config
from vesuvius.neural_tracing.autoreg_fiber.dataset import (
    AutoregFiberDataset,
    autoreg_fiber_collate,
    split_indices_by_fiber_id,
)
from vesuvius.neural_tracing.autoreg_fiber.infer import infer_autoreg_fiber
from vesuvius.neural_tracing.autoreg_fiber.losses import compute_autoreg_fiber_losses
from vesuvius.neural_tracing.autoreg_fiber.model import AutoregFiberModel


_SENSITIVE_WANDB_CONFIG_KEY_PARTS = (
    "api_key",
    "apikey",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass
class _DistributedRuntime:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    initialized_process_group: bool

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def _initialize_distributed_runtime(device: str | torch.device | None = None) -> _DistributedRuntime:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested_device = torch.device(device) if device is not None else None
    if env_world_size <= 1:
        runtime_device = requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _DistributedRuntime(False, 0, 0, 1, runtime_device, None, False)

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if requested_device is not None and requested_device.type != "cuda":
        runtime_device = requested_device
        backend = "gloo"
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        runtime_device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        runtime_device = torch.device("cpu")
        backend = "gloo"

    initialized = False
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required when WORLD_SIZE > 1")
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        initialized = True
    return _DistributedRuntime(
        True,
        int(dist.get_rank()),
        int(local_rank),
        int(dist.get_world_size()),
        runtime_device,
        str(backend),
        initialized,
    )


def _unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def _ddp_find_unused_parameters_enabled(cfg: dict) -> bool:
    explicit = cfg.get("ddp_find_unused_parameters")
    if explicit is not None:
        return bool(explicit)
    staged_loss_starts = (
        int(cfg.get("offset_loss_start_step", 0)),
        int(cfg.get("position_refine_start_step", 0)),
        int(cfg.get("xyz_soft_loss_start_step", 0)),
        int(cfg.get("segment_vector_loss_start_step", 0)),
    )
    return str(cfg.get("coarse_prediction_mode", "joint_pointer")) == "axis_factorized" or any(
        step > 0 for step in staged_loss_starts
    )


def _wrap_model_for_ddp(model: AutoregFiberModel, runtime: _DistributedRuntime, cfg: dict):
    if not runtime.is_distributed:
        return model
    return DDP(
        model,
        device_ids=[runtime.local_rank] if runtime.device.type == "cuda" else None,
        output_device=runtime.local_rank if runtime.device.type == "cuda" else None,
        find_unused_parameters=_ddp_find_unused_parameters_enabled(cfg),
        broadcast_buffers=False,
    )


def _seed_worker(worker_id: int, *, base_seed: int, rank: int) -> None:
    worker_seed = int(base_seed) + (int(rank) * 1000) + int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))
    torch.manual_seed(worker_seed)


def _metric_dict_mean_across_ranks(
    metrics: dict[str, float],
    *,
    device: torch.device,
    runtime: _DistributedRuntime,
) -> dict[str, float]:
    if not runtime.is_distributed or not metrics:
        return dict(metrics)
    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values / float(runtime.world_size)
    return {key: float(value.item()) for key, value in zip(keys, values, strict=True)}


def _distributed_all_finite(value: float, *, device: torch.device, runtime: _DistributedRuntime) -> bool:
    finite = torch.tensor(1 if np.isfinite(float(value)) else 0, device=device, dtype=torch.int32)
    if runtime.is_distributed:
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(int(finite.item()) == 1)


def _maybe_barrier(runtime: _DistributedRuntime) -> None:
    if runtime.is_distributed:
        dist.barrier()


def _move_batch_to_device(batch: dict, device: torch.device) -> dict:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        elif key == "prompt_tokens":
            moved[key] = {
                inner_key: inner_value.to(device) if torch.is_tensor(inner_value) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def _next_batch(iterator, dataloader, *, on_reset=None):
    try:
        batch = next(iterator)
    except StopIteration:
        if on_reset is not None:
            on_reset()
        iterator = iter(dataloader)
        batch = next(iterator)
    return batch, iterator


def _split_random_indices(num_items: int, *, seed: int, val_fraction: float) -> tuple[list[int], list[int]]:
    if num_items <= 0:
        raise ValueError("autoreg_fiber training requires a non-empty dataset")
    if num_items < 2 or float(val_fraction) <= 0.0:
        return list(range(num_items)), []
    num_val = int(round(num_items * float(val_fraction)))
    num_val = max(1, min(num_val, num_items - 1))
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(num_items).tolist()
    return sorted(indices[num_val:]), sorted(indices[:num_val])


def _split_dataset(dataset: Dataset, *, cfg: dict, seed: int, val_fraction: float) -> tuple[Dataset, Dataset | None, dict[str, float]]:
    total = len(dataset)
    if isinstance(dataset, AutoregFiberDataset):
        train_indices, val_indices = split_indices_by_fiber_id(
            dataset.sample_plans,
            val_fraction=float(val_fraction),
            seed=int(seed),
        )
        train_ids = {dataset.sample_plans[idx].fiber_id for idx in train_indices}
        val_ids = {dataset.sample_plans[idx].fiber_id for idx in val_indices}
        if train_ids & val_ids:
            raise RuntimeError("autoreg_fiber train/val split leaked a fiber id")
        diagnostics = {
            "num_train_fibers": float(len(train_ids)),
            "num_val_fibers": float(len(val_ids)),
            "val_fraction_actual": float(len(val_indices)) / float(max(total, 1)),
        }
    else:
        train_indices, val_indices = _split_random_indices(total, seed=seed, val_fraction=val_fraction)
        diagnostics = {"val_fraction_actual": float(len(val_indices)) / float(max(total, 1))}
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    return train_dataset, val_dataset, diagnostics


def _make_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
    sampler=None,
    worker_init_fn=None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle and sampler is None),
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=autoreg_fiber_collate,
        persistent_workers=bool(num_workers > 0),
        generator=generator,
        worker_init_fn=worker_init_fn,
    )


def _maybe_import_wandb(cfg: dict):
    if not cfg.get("wandb_project"):
        return None
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "wandb_project is configured for autoreg_fiber training, but the 'wandb' package is not installed."
        ) from exc


def _is_sensitive_wandb_config_key(key: str) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_WANDB_CONFIG_KEY_PARTS)


def _sanitize_wandb_config_value(value, *, key: str | None = None):
    if key is not None and _is_sensitive_wandb_config_key(key):
        return "[redacted]"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(child_key): _sanitize_wandb_config_value(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_wandb_config_value(item) for item in value]
    return value


def _sanitize_wandb_config(config: dict) -> dict:
    return {
        str(key): _sanitize_wandb_config_value(value, key=str(key))
        for key, value in dict(config or {}).items()
    }


def _load_checkpoint_payload(path: str | Path | None):
    if path is None:
        return None
    return torch.load(Path(path), map_location="cpu", weights_only=False)


def _validate_checkpoint_compatibility(cfg: dict, ckpt_payload: dict | None) -> None:
    if ckpt_payload is None:
        return
    ckpt_config = ckpt_payload.get("config", {})
    if not isinstance(ckpt_config, dict):
        return
    keys = ("input_shape", "patch_size", "offset_num_bins", "decoder_dim", "decoder_depth", "decoder_num_heads")
    mismatches = []
    for key in keys:
        if str(ckpt_config.get(key)) != str(cfg.get(key)):
            mismatches.append(f"{key}: checkpoint={ckpt_config.get(key)!r} current={cfg.get(key)!r}")
    if mismatches:
        raise ValueError("load_ckpt uses an incompatible autoreg_fiber config: " + "; ".join(mismatches))


def _make_checkpoint_payload(
    *,
    model: AutoregFiberModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> dict:
    raw_model = _unwrap_model(model)
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": int(step),
        "wandb_run_id": config.get("wandb_run_id"),
        "distributed_world_size": int(dist.get_world_size()) if dist.is_available() and dist.is_initialized() else 1,
        "distributed_backend": dist.get_backend() if dist.is_available() and dist.is_initialized() else None,
        "mixed_precision": str(config.get("mixed_precision", "no")),
    }
    if scheduler is not None:
        payload["lr_scheduler"] = scheduler.state_dict()
    return payload


def _save_checkpoint(
    *,
    out_dir: Path,
    filename: str,
    model: AutoregFiberModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: dict,
    step: int,
) -> Path:
    path = out_dir / filename
    torch.save(
        _make_checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            step=step,
        ),
        path,
    )
    return path


def load_autoreg_fiber_model_from_checkpoint(
    path: str | Path,
    *,
    config_overrides: dict | None = None,
    map_location: str | torch.device = "cpu",
) -> AutoregFiberModel:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    cfg = dict(payload.get("config") or {})
    if config_overrides:
        cfg.update(config_overrides)
    cfg = validate_autoreg_fiber_config(cfg)
    model = AutoregFiberModel(cfg)
    model.load_state_dict(payload["model"])
    return model


def _loss_dict_to_metrics(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    metrics = {}
    for key, value in loss_dict.items():
        metrics[key] = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
    return metrics


def _mean_metric_dict(metric_dicts: list[dict[str, float]], *, prefix: str) -> dict[str, float]:
    if not metric_dicts:
        return {}
    sums: dict[str, float] = {}
    for metrics in metric_dicts:
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value)
    return {f"{prefix}{key}": value / float(len(metric_dicts)) for key, value in sums.items()}


def _scheduled_sampling_prob(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("scheduled_sampling_enabled", False)):
        return 0.0
    start_step = int(cfg.get("scheduled_sampling_start_step", 0))
    if global_step < start_step:
        return 0.0
    max_prob = float(cfg.get("scheduled_sampling_max_prob", 0.0))
    ramp_steps = int(cfg.get("scheduled_sampling_ramp_steps", 0))
    if ramp_steps <= 0:
        return max_prob
    progress = min(1.0, max(0.0, float(global_step - start_step) / float(ramp_steps)))
    return max_prob * progress


def _offset_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if int(global_step) < int(cfg.get("offset_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("offset_loss_weight", 1.0))


def _position_refine_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("position_refine_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("position_refine_start_step", 5000)):
        return 0.0
    return float(cfg.get("position_refine_weight", 0.0))


def _xyz_soft_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("xyz_soft_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("xyz_soft_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("xyz_soft_loss_weight", 0.0))


def _segment_vector_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("segment_vector_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("segment_vector_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("segment_vector_loss_weight", 0.0))


def _straightness_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("straightness_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("straightness_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("straightness_loss_weight", 0.0))


def _tube_radius_loss_weight_active(cfg: dict, *, global_step: int) -> float:
    if not bool(cfg.get("tube_radius_loss_enabled", True)):
        return 0.0
    if int(global_step) < int(cfg.get("tube_radius_loss_start_step", 0)):
        return 0.0
    return float(cfg.get("tube_radius_loss_weight", 0.0))


def _scheduled_sampling_feedback_state(cfg: dict, *, global_step: int) -> tuple[bool, bool]:
    offset_feedback_enabled = _offset_loss_weight_active(cfg, global_step=global_step) > 0.0
    refine_feedback_enabled = _position_refine_weight_active(cfg, global_step=global_step) > 0.0
    return bool(offset_feedback_enabled), bool(offset_feedback_enabled and refine_feedback_enabled)


def _as_numpy_array(value, *, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _as_numpy_mask(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(value, dtype=bool)


def _draw_line_2d(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        if 0 <= r0 < canvas.shape[0] and 0 <= c0 < canvas.shape[1]:
            canvas[r0, c0] = 1.0
        if r0 == r1 and c0 == c1:
            break
        err2 = 2 * err
        if err2 > -dr:
            err -= dr
            c0 += sc
        if err2 < dc:
            err += dc
            r0 += sr


def _draw_line_2d_thick(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int, *, thickness: int) -> None:
    radius = max(0, int(thickness) - 1)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            _draw_line_2d(canvas, r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _normalize_slice_to_rgb(slice_2d: np.ndarray) -> np.ndarray:
    slice_arr = np.asarray(slice_2d, dtype=np.float32)
    finite = np.isfinite(slice_arr)
    if not bool(finite.any()):
        gray = np.zeros_like(slice_arr, dtype=np.uint8)
    else:
        low = float(slice_arr[finite].min())
        high = float(slice_arr[finite].max())
        if high <= low + 1e-6:
            gray = np.zeros_like(slice_arr, dtype=np.uint8)
        else:
            normalized = np.clip((slice_arr - low) / (high - low), 0.0, 1.0)
            gray = (255.0 * normalized).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


_SKELETON_COLORS = {
    "prompt": (90, 180, 255),
    "gt": (110, 235, 110),
    "pred": (255, 190, 90),
}


def _valid_polyline_points(points_local: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_local, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"fiber points must have shape [N,3], got {points.shape!r}")
    finite = np.isfinite(points).all(axis=-1)
    if valid_mask is None:
        valid = finite
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape[0] != points.shape[0]:
            mask = mask[: points.shape[0]]
            if mask.shape[0] < points.shape[0]:
                mask = np.pad(mask, (0, points.shape[0] - mask.shape[0]), constant_values=False)
        valid = finite & mask
    masked = points.copy()
    masked[~valid] = np.nan
    return masked, valid


def _prepend_prompt_anchor(
    points_local: np.ndarray,
    valid_mask: np.ndarray,
    *,
    prompt_points_local: np.ndarray,
    prompt_valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prompt_points, prompt_valid = _valid_polyline_points(prompt_points_local, prompt_valid_mask)
    valid_indices = np.flatnonzero(prompt_valid)
    if valid_indices.size == 0:
        return points_local, valid_mask
    anchor = prompt_points[int(valid_indices[-1]) : int(valid_indices[-1]) + 1]
    return np.concatenate([anchor, points_local], axis=0), np.concatenate([np.asarray([True]), valid_mask.astype(bool)], axis=0)


def _iter_polyline_segments(points_local: np.ndarray):
    points = np.asarray(points_local, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    for idx in range(max(0, points.shape[0] - 1)):
        if valid[idx] and valid[idx + 1]:
            yield points[idx], points[idx + 1]


def _scale_coordinate(value: float, *, source_size: int, target_size: int) -> int:
    if int(source_size) <= 1 or int(target_size) <= 1:
        return 0
    scaled = float(value) * (float(target_size) - 1.0) / (float(source_size) - 1.0)
    return int(np.clip(np.rint(scaled), 0, int(target_size) - 1))


def _project_point(
    point: np.ndarray,
    axes: tuple[int, int],
    panel_shape: tuple[int, int],
    *,
    source_shape: tuple[int, int] | None = None,
) -> tuple[int, int]:
    if source_shape is None:
        row = int(np.clip(np.rint(float(point[axes[0]])), 0, panel_shape[0] - 1))
        col = int(np.clip(np.rint(float(point[axes[1]])), 0, panel_shape[1] - 1))
    else:
        row = _scale_coordinate(float(point[axes[0]]), source_size=int(source_shape[0]), target_size=int(panel_shape[0]))
        col = _scale_coordinate(float(point[axes[1]]), source_size=int(source_shape[1]), target_size=int(panel_shape[1]))
    return row, col


def _rasterize_polyline_projection(
    points_local: np.ndarray,
    *,
    axes: tuple[int, int],
    panel_shape: tuple[int, int],
    line_thickness: int,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    panel = np.zeros(panel_shape, dtype=np.float32)
    points = np.asarray(points_local, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    for point in points[valid]:
        row, col = _project_point(point, axes, panel_shape, source_shape=source_shape)
        _draw_line_2d_thick(panel, row, col, row, col, thickness=int(line_thickness))
    for p0, p1 in _iter_polyline_segments(points):
        r0, c0 = _project_point(p0, axes, panel_shape, source_shape=source_shape)
        r1, c1 = _project_point(p1, axes, panel_shape, source_shape=source_shape)
        _draw_line_2d_thick(panel, r0, c0, r1, c1, thickness=int(line_thickness))
    return panel


def _edge_segment_on_z_slice(
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    z_slice: int,
    depth_tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    z_min = float(z_slice) - float(depth_tolerance)
    z_max = float(z_slice) + float(depth_tolerance)
    z0 = float(p0[0])
    z1 = float(p1[0])
    dz = z1 - z0

    if abs(dz) < 1e-6:
        if z_min <= z0 <= z_max:
            return p0[1:].copy(), p1[1:].copy()
        return None

    t0 = (z_min - z0) / dz
    t1 = (z_max - z0) / dz
    t_lo = max(0.0, min(t0, t1))
    t_hi = min(1.0, max(t0, t1))
    if t_hi < 0.0 or t_lo > 1.0 or t_lo > t_hi:
        return None

    pa = p0 + t_lo * (p1 - p0)
    pb = p0 + t_hi * (p1 - p0)
    return pa[1:].copy(), pb[1:].copy()


def _blend_line_mask(canvas: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int], alpha: float = 0.8) -> np.ndarray:
    blended = np.asarray(canvas, dtype=np.float32).copy()
    mask_bool = mask > 0.0
    if not bool(mask_bool.any()):
        return canvas
    color_arr = np.asarray(color, dtype=np.float32)
    blended[mask_bool] = (1.0 - float(alpha)) * blended[mask_bool] + float(alpha) * color_arr[None, :]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def _rasterize_polyline_on_xy_slice(
    points_local: np.ndarray,
    *,
    z_slice: int,
    panel_shape: tuple[int, int],
    line_thickness: int,
    depth_tolerance: float,
    source_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    mask = np.zeros(panel_shape, dtype=np.float32)
    points = np.asarray(points_local, dtype=np.float32)
    source = source_shape or panel_shape
    valid = np.isfinite(points).all(axis=-1)
    for point in points[valid]:
        if abs(float(point[0]) - float(z_slice)) <= float(depth_tolerance):
            row = _scale_coordinate(float(point[1]), source_size=int(source[0]), target_size=int(panel_shape[0]))
            col = _scale_coordinate(float(point[2]), source_size=int(source[1]), target_size=int(panel_shape[1]))
            _draw_line_2d_thick(mask, row, col, row, col, thickness=int(line_thickness))
    for p0, p1 in _iter_polyline_segments(points):
        clipped = _edge_segment_on_z_slice(p0, p1, z_slice=z_slice, depth_tolerance=depth_tolerance)
        if clipped is None:
            continue
        a_yx, b_yx = clipped
        r0 = _scale_coordinate(float(a_yx[0]), source_size=int(source[0]), target_size=int(panel_shape[0]))
        c0 = _scale_coordinate(float(a_yx[1]), source_size=int(source[1]), target_size=int(panel_shape[1]))
        r1 = _scale_coordinate(float(b_yx[0]), source_size=int(source[0]), target_size=int(panel_shape[0]))
        c1 = _scale_coordinate(float(b_yx[1]), source_size=int(source[1]), target_size=int(panel_shape[1]))
        _draw_line_2d_thick(mask, r0, c0, r1, c1, thickness=int(line_thickness))
    return mask


def _resize_rgb(image: np.ndarray, *, shape: tuple[int, int], resample=None) -> np.ndarray:
    from PIL import Image

    if resample is None:
        resample = getattr(Image.Resampling, "BILINEAR", Image.BILINEAR)
    rgb = np.asarray(image, dtype=np.uint8)
    pil_image = Image.fromarray(rgb)
    return np.asarray(pil_image.resize((int(shape[1]), int(shape[0])), resample=resample), dtype=np.uint8)


def _projection_background_rgb(volume: np.ndarray | None, *, axes: tuple[int, int], panel_shape: tuple[int, int]) -> np.ndarray:
    if volume is None:
        return np.zeros((*panel_shape, 3), dtype=np.uint8)
    volume_np = np.asarray(volume, dtype=np.float32)
    if volume_np.ndim == 4:
        volume_np = volume_np[0]
    if volume_np.ndim != 3:
        return np.zeros((*panel_shape, 3), dtype=np.uint8)
    projection_axis = ({0, 1, 2} - set(axes)).pop()
    projected = np.max(volume_np, axis=projection_axis)
    return _resize_rgb(_normalize_slice_to_rgb(projected), shape=panel_shape)


def _preview_panel_shape(source_shape: tuple[int, int], *, min_height: int = 256, max_height: int = 384) -> tuple[int, int]:
    source_h = max(1, int(source_shape[0]))
    source_w = max(1, int(source_shape[1]))
    height = int(np.clip(source_h * 2, int(min_height), int(max_height)))
    width = max(1, int(round(float(source_w) * float(height) / float(source_h))))
    return height, width


def _xy_preview_shape(source_shape: tuple[int, int]) -> tuple[int, int]:
    source_h = max(1, int(source_shape[0]))
    source_w = max(1, int(source_shape[1]))
    height = int(np.clip(source_h * 4, 512, 768))
    width = max(1, int(round(float(source_w) * float(height) / float(source_h))))
    return height, width


def _finite_points(points_local: np.ndarray) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1)
    return points[valid].astype(np.float32, copy=False)


def _xy_zoom_window(
    prompt_points_local: np.ndarray,
    target_points_local: np.ndarray,
    pred_points_local: np.ndarray,
    *,
    volume_shape: tuple[int, int, int],
    depth_tolerance: float,
) -> tuple[int, int, int, int, int, int]:
    supervised = np.concatenate(
        [_finite_points(prompt_points_local), _finite_points(target_points_local)],
        axis=0,
    )
    if supervised.shape[0] == 0:
        supervised = _finite_points(pred_points_local)
    if supervised.shape[0] == 0:
        depth, height, width = volume_shape
        return 0, max(1, int(depth)) - 1, 0, max(1, int(height)), 0, max(1, int(width))

    depth, height, width = (int(v) for v in volume_shape)
    z_pad = max(2, int(np.ceil(float(depth_tolerance))))
    z0 = max(0, int(np.floor(float(supervised[:, 0].min()))) - z_pad)
    z1 = min(depth - 1, int(np.ceil(float(supervised[:, 0].max()))) + z_pad)

    y_min = float(supervised[:, 1].min())
    y_max = float(supervised[:, 1].max())
    x_min = float(supervised[:, 2].min())
    x_max = float(supervised[:, 2].max())
    xy_span = max(y_max - y_min, x_max - x_min, 1.0)
    margin = int(np.clip(round(8.0 + 2.0 * xy_span), 12, 40))
    y0 = max(0, int(np.floor(y_min)) - margin)
    y1 = min(height, int(np.ceil(y_max)) + margin + 1)
    x0 = max(0, int(np.floor(x_min)) - margin)
    x1 = min(width, int(np.ceil(x_max)) + margin + 1)
    if y1 <= y0:
        y1 = min(height, y0 + 1)
    if x1 <= x0:
        x1 = min(width, x0 + 1)
    return z0, z1, y0, y1, x0, x1


def _points_for_xy_window(
    points_local: np.ndarray,
    *,
    z0: int,
    z1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    points = np.asarray(points_local, dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    finite = np.isfinite(points).all(axis=-1)
    inside = (
        finite
        & (points[:, 0] >= float(z0))
        & (points[:, 0] <= float(z1))
        & (points[:, 1] >= float(y0))
        & (points[:, 1] < float(y1))
        & (points[:, 2] >= float(x0))
        & (points[:, 2] < float(x1))
    )
    points[~inside] = np.nan
    points[:, 1] -= float(y0)
    points[:, 2] -= float(x0)
    return points.astype(np.float32, copy=False)


def _text_bbox(draw, text: str, font) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])
    except AttributeError:
        return tuple(int(v) for v in draw.textsize(text, font=font))


def _legend_header(canvas: np.ndarray, *, title: str | None = None) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    header_height = 38
    header = np.zeros((header_height, canvas.shape[1], 3), dtype=np.uint8)
    header[..., :] = np.asarray((24, 24, 24), dtype=np.uint8)
    image = Image.fromarray(header)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x = 8
    if title:
        draw.text((x, 6), title, fill=(245, 245, 245), font=font)
        title_width, _ = _text_bbox(draw, title, font)
        x += title_width + 18
    for label, color in (
        ("prompt", _SKELETON_COLORS["prompt"]),
        ("GT", _SKELETON_COLORS["gt"]),
        ("pred", _SKELETON_COLORS["pred"]),
    ):
        if x + 58 > canvas.shape[1]:
            break
        draw.rectangle((x, 12, x + 12, 24), fill=color)
        draw.text((x + 17, 10), label, fill=(235, 235, 235), font=font)
        label_width, _ = _text_bbox(draw, label, font)
        x += 28 + label_width
    return np.concatenate([np.asarray(image, dtype=np.uint8), canvas], axis=0)


def _panel_label_strip(panel: np.ndarray, *, label: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    strip_height = 24
    strip = np.full((strip_height, panel.shape[1], 3), 18, dtype=np.uint8)
    image = Image.fromarray(strip)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text_width, text_height = _text_bbox(draw, label, font)
    draw.text(
        ((panel.shape[1] - text_width) // 2, max(1, (strip_height - text_height) // 2)),
        label,
        fill=(235, 235, 235),
        font=font,
    )
    return np.concatenate([np.asarray(image, dtype=np.uint8), panel], axis=0)


def _pad_image_height(image: np.ndarray, *, height: int) -> np.ndarray:
    if int(image.shape[0]) >= int(height):
        return image
    pad_rows = int(height) - int(image.shape[0])
    return np.pad(image, ((0, pad_rows), (0, 0), (0, 0)), mode="constant")


def _make_projection_canvas(
    *,
    prompt_points_local: np.ndarray,
    target_points_local: np.ndarray,
    pred_points_local: np.ndarray,
    crop_shape: tuple[int, int, int],
    line_thickness: int = 1,
    volume: np.ndarray | None = None,
) -> np.ndarray:
    panel_specs = [
        ("ZY", (0, 1), (int(crop_shape[0]), int(crop_shape[1]))),
        ("ZX", (0, 2), (int(crop_shape[0]), int(crop_shape[2]))),
        ("YX", (1, 2), (int(crop_shape[1]), int(crop_shape[2]))),
    ]
    panels = []
    for name, axes, source_shape in panel_specs:
        panel_shape = _preview_panel_shape(source_shape)
        panel = _projection_background_rgb(volume, axes=axes, panel_shape=panel_shape)
        prompt_mask = _rasterize_polyline_projection(
            prompt_points_local,
            axes=axes,
            panel_shape=panel_shape,
            line_thickness=int(line_thickness),
            source_shape=source_shape,
        )
        target_mask = _rasterize_polyline_projection(
            target_points_local,
            axes=axes,
            panel_shape=panel_shape,
            line_thickness=int(line_thickness),
            source_shape=source_shape,
        )
        pred_mask = _rasterize_polyline_projection(
            pred_points_local,
            axes=axes,
            panel_shape=panel_shape,
            line_thickness=int(line_thickness),
            source_shape=source_shape,
        )
        panel = _blend_line_mask(panel, target_mask, color=_SKELETON_COLORS["gt"], alpha=0.95)
        panel = _blend_line_mask(panel, pred_mask, color=_SKELETON_COLORS["pred"], alpha=0.95)
        panel = _blend_line_mask(panel, prompt_mask, color=_SKELETON_COLORS["prompt"], alpha=0.95)
        panels.append(_panel_label_strip(panel, label=name))

    target_height = max(int(panel.shape[0]) for panel in panels)
    panels = [_pad_image_height(panel, height=target_height) for panel in panels]
    separator = np.full((target_height, 6, 3), 24, dtype=np.uint8)
    body = np.concatenate([panels[0], separator, panels[1], separator.copy(), panels[2]], axis=1)
    return _legend_header(body, title="Projection")


def _make_xy_slice_overlay_canvas(
    *,
    volume: np.ndarray,
    prompt_points_local: np.ndarray,
    target_points_local: np.ndarray,
    pred_points_local: np.ndarray,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    volume_np = np.asarray(volume, dtype=np.float32)
    if volume_np.ndim == 4:
        volume_np = volume_np[0]
    if volume_np.ndim != 3:
        raise ValueError(f"volume must have shape [D,H,W] or [1,D,H,W], got {volume_np.shape!r}")
    z0, z1, y0, y1, x0, x1 = _xy_zoom_window(
        prompt_points_local,
        target_points_local,
        pred_points_local,
        volume_shape=tuple(int(v) for v in volume_np.shape),
        depth_tolerance=float(depth_tolerance),
    )
    slab = volume_np[z0:z1 + 1, y0:y1, x0:x1]
    if slab.size == 0:
        slab = volume_np[max(0, min(volume_np.shape[0] - 1, z0)):max(0, min(volume_np.shape[0], z0 + 1)), y0:y1, x0:x1]
    mip = np.max(slab, axis=0) if slab.size > 0 else np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.float32)
    source_shape = tuple(int(v) for v in mip.shape)
    panel_shape = _xy_preview_shape(source_shape)
    slice_rgb = _resize_rgb(_normalize_slice_to_rgb(mip), shape=panel_shape)
    prompt_xy_points = _points_for_xy_window(prompt_points_local, z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)
    target_xy_points = _points_for_xy_window(target_points_local, z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)
    pred_xy_points = _points_for_xy_window(pred_points_local, z0=z0, z1=z1, y0=y0, y1=y1, x0=x0, x1=x1)
    prompt_mask = _rasterize_polyline_on_xy_slice(
        prompt_xy_points,
        z_slice=z0,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(max(1, z1 - z0)),
        source_shape=source_shape,
    )
    target_mask = _rasterize_polyline_on_xy_slice(
        target_xy_points,
        z_slice=z0,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(max(1, z1 - z0)),
        source_shape=source_shape,
    )
    pred_mask = _rasterize_polyline_on_xy_slice(
        pred_xy_points,
        z_slice=z0,
        panel_shape=panel_shape,
        line_thickness=int(line_thickness),
        depth_tolerance=float(max(1, z1 - z0)),
        source_shape=source_shape,
    )
    overlay = slice_rgb
    overlay = _blend_line_mask(overlay, target_mask, color=_SKELETON_COLORS["gt"], alpha=0.95)
    overlay = _blend_line_mask(overlay, pred_mask, color=_SKELETON_COLORS["pred"], alpha=0.95)
    overlay = _blend_line_mask(overlay, prompt_mask, color=_SKELETON_COLORS["prompt"], alpha=0.95)
    overlay = _panel_label_strip(overlay, label="XY")
    return _legend_header(overlay, title=f"z={z0}-{z1}")


def _extract_teacher_forced_polylines(batch: dict, outputs: dict, *, sample_idx: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prompt_points, prompt_valid = _valid_polyline_points(
        _as_numpy_array(batch["prompt_tokens"]["xyz"][sample_idx]),
        _as_numpy_mask(batch["prompt_tokens"]["valid_mask"][sample_idx]),
    )
    target_points, target_valid = _valid_polyline_points(
        _as_numpy_array(batch["target_xyz"][sample_idx]),
        _as_numpy_mask(batch["target_valid_mask"][sample_idx]),
    )
    pred_tensor = outputs.get("pred_xyz_refined", outputs["pred_xyz"])
    pred_points, pred_valid = _valid_polyline_points(
        _as_numpy_array(pred_tensor[sample_idx]),
        _as_numpy_mask(batch["target_valid_mask"][sample_idx]),
    )
    target_points, target_valid = _prepend_prompt_anchor(
        target_points,
        target_valid,
        prompt_points_local=prompt_points,
        prompt_valid_mask=prompt_valid,
    )
    pred_points, pred_valid = _prepend_prompt_anchor(
        pred_points,
        pred_valid,
        prompt_points_local=prompt_points,
        prompt_valid_mask=prompt_valid,
    )
    target_points[~target_valid] = np.nan
    pred_points[~pred_valid] = np.nan
    return prompt_points, target_points, pred_points


def _make_teacher_forced_prediction_canvas(batch: dict, outputs: dict, *, sample_idx: int = 0, line_thickness: int = 1) -> np.ndarray:
    prompt_points, target_points, pred_points = _extract_teacher_forced_polylines(batch, outputs, sample_idx=sample_idx)
    crop_shape = tuple(int(v) for v in batch["volume"][sample_idx].shape[-3:])
    return _make_projection_canvas(
        prompt_points_local=prompt_points,
        target_points_local=target_points,
        pred_points_local=pred_points,
        crop_shape=crop_shape,
        line_thickness=int(line_thickness),
        volume=_as_numpy_array(batch["volume"][sample_idx]),
    )


def _make_teacher_forced_xy_slice_canvas(
    batch: dict,
    outputs: dict,
    *,
    sample_idx: int = 0,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    prompt_points, target_points, pred_points = _extract_teacher_forced_polylines(batch, outputs, sample_idx=sample_idx)
    volume = _as_numpy_array(batch["volume"][sample_idx])
    return _make_xy_slice_overlay_canvas(
        volume=volume,
        prompt_points_local=prompt_points,
        target_points_local=target_points,
        pred_points_local=pred_points,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )


def _extract_inference_polylines(raw_sample: dict, inference_result: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror of :func:`_extract_teacher_forced_polylines` for an autoregressive
    rollout. ``raw_sample`` is a single uncollated dataset item, and
    ``inference_result`` is the return value of :func:`infer_autoreg_fiber`
    on that sample. Pred points come from ``predicted_fiber_local_zyx``; if
    the rollout stopped early (``stop_probability`` triggered) the trailing
    positions are marked invalid so the rasteriser skips them."""

    prompt_points, prompt_valid = _valid_polyline_points(
        _as_numpy_array(raw_sample["prompt_tokens"]["xyz"]),
        _as_numpy_mask(raw_sample["prompt_tokens"]["valid_mask"]),
    )
    target_points, target_valid = _valid_polyline_points(
        _as_numpy_array(raw_sample["target_xyz"]),
        _as_numpy_mask(raw_sample["target_valid_mask"]),
    )
    pred_local = np.asarray(inference_result["predicted_fiber_local_zyx"], dtype=np.float32)
    target_len = int(target_points.shape[0])
    pred_points = np.full((target_len, 3), np.nan, dtype=np.float32)
    pred_valid = np.zeros((target_len,), dtype=bool)
    n = min(int(pred_local.shape[0]), target_len)
    if n > 0:
        pred_points[:n] = pred_local[:n]
        pred_valid[:n] = True
    target_points, target_valid = _prepend_prompt_anchor(
        target_points,
        target_valid,
        prompt_points_local=prompt_points,
        prompt_valid_mask=prompt_valid,
    )
    pred_points, pred_valid = _prepend_prompt_anchor(
        pred_points,
        pred_valid,
        prompt_points_local=prompt_points,
        prompt_valid_mask=prompt_valid,
    )
    target_points[~target_valid] = np.nan
    pred_points[~pred_valid] = np.nan
    return prompt_points, target_points, pred_points


def _make_inference_prediction_canvas(
    raw_sample: dict,
    inference_result: dict,
    *,
    line_thickness: int = 1,
) -> np.ndarray:
    """Projection canvas (ZY/ZX/YX triptych) of an autoregressive rollout
    overlaid on the sample's volume. Same renderer as the teacher-forced
    canvas — only the source of the prediction polyline differs."""

    prompt_points, target_points, pred_points = _extract_inference_polylines(raw_sample, inference_result)
    crop_shape = tuple(int(v) for v in _as_numpy_array(raw_sample["volume"]).shape[-3:])
    return _make_projection_canvas(
        prompt_points_local=prompt_points,
        target_points_local=target_points,
        pred_points_local=pred_points,
        crop_shape=crop_shape,
        line_thickness=int(line_thickness),
        volume=_as_numpy_array(raw_sample["volume"]),
    )


def _make_inference_xy_slice_canvas(
    raw_sample: dict,
    inference_result: dict,
    *,
    line_thickness: int,
    depth_tolerance: float,
) -> np.ndarray:
    prompt_points, target_points, pred_points = _extract_inference_polylines(raw_sample, inference_result)
    volume = _as_numpy_array(raw_sample["volume"])
    return _make_xy_slice_overlay_canvas(
        volume=volume,
        prompt_points_local=prompt_points,
        target_points_local=target_points,
        pred_points_local=pred_points,
        line_thickness=int(line_thickness),
        depth_tolerance=float(depth_tolerance),
    )


def _autoreg_fiber_base_dataset(dataset: Dataset) -> AutoregFiberDataset | None:
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current if isinstance(current, AutoregFiberDataset) else None


def _autoreg_fiber_sample_plans(dataset: Dataset) -> list:
    if isinstance(dataset, AutoregFiberDataset):
        return list(dataset.sample_plans)
    if isinstance(dataset, Subset):
        parent_plans = _autoreg_fiber_sample_plans(dataset.dataset)
        if not parent_plans:
            return []
        return [parent_plans[int(index)] for index in dataset.indices]
    return []


def _wandb_dataset_summary(
    *,
    dataset: Dataset,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    max_table_rows: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    base = _autoreg_fiber_base_dataset(dataset)
    if base is None:
        scalars = {
            "dataset_length": float(len(dataset)),
            "train_windows": float(len(train_dataset)),
            "val_windows": 0.0 if val_dataset is None else float(len(val_dataset)),
        }
        return scalars, []

    full_plans = _autoreg_fiber_sample_plans(base)
    train_plans = _autoreg_fiber_sample_plans(train_dataset)
    val_plans = [] if val_dataset is None else _autoreg_fiber_sample_plans(val_dataset)

    metadata_by_fiber_id: dict[str, dict[str, Any]] = {}
    point_counts: list[int] = []
    marker_counts: Counter[str] = Counter()
    target_volume_counts: Counter[str] = Counter()
    annotation_ids = set()
    transform_checksums = set()
    for fiber_idx, (points, metadata) in enumerate(base.fibers):
        fiber_id = f"{metadata.get('annotation_id', fiber_idx)}:{metadata.get('tree_id', fiber_idx)}"
        metadata_by_fiber_id[fiber_id] = dict(metadata)
        point_count = int(metadata.get("point_count", points.shape[0]))
        point_counts.append(point_count)
        marker_counts[str(metadata.get("marker", "unknown"))] += 1
        target_volume_counts[str(metadata.get("target_volume", "unknown"))] += 1
        if metadata.get("annotation_id") is not None:
            annotation_ids.add(str(metadata["annotation_id"]))
        if metadata.get("transform_checksum") is not None:
            transform_checksums.add(str(metadata["transform_checksum"]))

    train_ids = {plan.fiber_id for plan in train_plans}
    val_ids = {plan.fiber_id for plan in val_plans}
    scalars = {
        "num_fiber_cache_files": float(len(base.fiber_cache_paths)),
        "num_fibers": float(len(base.fibers)),
        "num_annotations": float(len(annotation_ids)),
        "num_transform_checksums": float(len(transform_checksums)),
        "sample_windows": float(len(full_plans)),
        "train_windows": float(len(train_plans)),
        "val_windows": float(len(val_plans)),
        "train_unique_fibers": float(len(train_ids)),
        "val_unique_fibers": float(len(val_ids)),
    }
    if point_counts:
        scalars["point_count_min"] = float(min(point_counts))
        scalars["point_count_mean"] = float(sum(point_counts) / len(point_counts))
        scalars["point_count_max"] = float(max(point_counts))
    for marker, count in sorted(marker_counts.items()):
        scalars[f"marker_{marker}_fibers"] = float(count)
    for target_volume, count in sorted(target_volume_counts.items()):
        scalars[f"target_{target_volume}_fibers"] = float(count)

    group_rows: dict[tuple[str, str], dict[str, Any]] = {}
    group_point_counts: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for fiber_id, metadata in metadata_by_fiber_id.items():
        marker = str(metadata.get("marker", "unknown"))
        target_volume = str(metadata.get("target_volume", "unknown"))
        key = (marker, target_volume)
        row = group_rows.setdefault(
            key,
            {
                "marker": marker,
                "target_volume": target_volume,
                "fibers": 0,
                "sample_windows": 0,
                "train_windows": 0,
                "val_windows": 0,
                "train_fibers": 0,
                "val_fibers": 0,
                "point_count_min": 0,
                "point_count_mean": 0.0,
                "point_count_max": 0,
            },
        )
        row["fibers"] += 1
        if fiber_id in train_ids:
            row["train_fibers"] += 1
        if fiber_id in val_ids:
            row["val_fibers"] += 1
        group_point_counts[key].append(int(metadata.get("point_count", 0)))

    for plan in full_plans:
        metadata = metadata_by_fiber_id.get(plan.fiber_id, {})
        key = (str(metadata.get("marker", "unknown")), str(metadata.get("target_volume", "unknown")))
        group_rows[key]["sample_windows"] += 1
    for plan in train_plans:
        metadata = metadata_by_fiber_id.get(plan.fiber_id, {})
        key = (str(metadata.get("marker", "unknown")), str(metadata.get("target_volume", "unknown")))
        group_rows[key]["train_windows"] += 1
    for plan in val_plans:
        metadata = metadata_by_fiber_id.get(plan.fiber_id, {})
        key = (str(metadata.get("marker", "unknown")), str(metadata.get("target_volume", "unknown")))
        group_rows[key]["val_windows"] += 1

    for key, counts in group_point_counts.items():
        if not counts:
            continue
        row = group_rows[key]
        row["point_count_min"] = int(min(counts))
        row["point_count_mean"] = float(sum(counts) / len(counts))
        row["point_count_max"] = int(max(counts))

    rows = sorted(group_rows.values(), key=lambda row: (str(row["marker"]), str(row["target_volume"])))
    return scalars, rows[: int(max_table_rows)]


def _log_wandb_dataset_summary(
    *,
    wandb,
    wandb_run,
    dataset: Dataset,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    cfg: dict,
    step: int,
) -> None:
    if wandb is None or not bool(cfg.get("wandb_log_dataset_summary", True)):
        return
    scalars, rows = _wandb_dataset_summary(
        dataset=dataset,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        max_table_rows=int(cfg.get("wandb_dataset_table_max_rows", 256)),
    )
    payload = {f"data/{key}": value for key, value in sorted(scalars.items())}
    if rows and hasattr(wandb, "Table"):
        columns = [
            "marker",
            "target_volume",
            "fibers",
            "sample_windows",
            "train_windows",
            "val_windows",
            "train_fibers",
            "val_fibers",
            "point_count_min",
            "point_count_mean",
            "point_count_max",
        ]
        payload["data/fiber_groups"] = wandb.Table(
            columns=columns,
            data=[[row.get(column) for column in columns] for row in rows],
        )
    if not payload:
        return

    active_run = getattr(wandb, "run", None) or wandb_run
    run_summary = getattr(active_run, "summary", None)
    if run_summary is not None:
        for key, value in payload.items():
            if key != "data/fiber_groups":
                run_summary[key] = value
    wandb.log(payload, step=int(step))


@torch.no_grad()
def _make_validation_teacher_forced_batch(
    *,
    model: AutoregFiberModel,
    dataset: Dataset,
    cfg: dict,
    device: torch.device,
) -> tuple[dict, dict]:
    raw_sample = dataset[0]
    raw_batch = autoreg_fiber_collate([raw_sample])
    batch = _move_batch_to_device(raw_batch, device)
    model.eval()
    amp_enabled = str(cfg.get("mixed_precision", "no")).lower() != "no" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        outputs = model(batch, scheduled_sampling_prob=0.0)
    model.train()
    return batch, outputs


def _wandb_metrics_payload(metrics: dict[str, float]) -> dict[str, float]:
    data_metric_keys = {
        "num_train_fibers",
        "num_val_fibers",
        "val_fraction_actual",
    }
    payload = {}
    for key, value in metrics.items():
        if key.startswith("rollout_val_"):
            payload[f"rollout_val/{key[12:]}"] = value
        elif key.startswith("val_"):
            payload[f"val/{key[4:]}"] = value
        elif key in data_metric_keys:
            payload[f"data/{key}"] = value
        else:
            payload[f"train/{key}"] = value
    return payload


def _slice_target_batch(batch: dict, *, max_length: int) -> dict:
    """Return a shallow-copy view of ``batch`` whose ``target_*`` tensors are
    sliced to at most ``max_length`` positions along the sequence dim.

    Used by the rollout-in-the-loop branch so the trainer can apply
    ``compute_autoreg_fiber_losses`` against the rollout's ``K``-step
    output without padding/copying the full target_length tensor.
    """

    out = dict(batch)
    K = int(max_length)
    target_keys = (
        "target_coarse_ids",
        "target_offset_bins",
        "target_xyz",
        "target_bin_center_xyz",
        "target_valid_mask",
        "target_stop",
        "target_supervision_mask",
        "target_mask",
        "target_positions",
    )
    for key in target_keys:
        if key in batch and isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 2:
            out[key] = batch[key][:, :K]
    if "target_lengths" in batch and torch.is_tensor(batch["target_lengths"]):
        capped = torch.clamp(batch["target_lengths"], max=K)
        out["target_lengths"] = capped
    return out


def _compute_losses_for_step(outputs: dict, batch: dict, cfg: dict, *, global_step: int) -> dict[str, torch.Tensor]:
    return compute_autoreg_fiber_losses(
        outputs,
        batch,
        offset_num_bins=tuple(int(v) for v in cfg["offset_num_bins"]),
        offset_loss_weight_active=_offset_loss_weight_active(cfg, global_step=global_step),
        position_refine_weight_active=_position_refine_weight_active(cfg, global_step=global_step),
        position_refine_loss_type=str(cfg.get("position_refine_loss", "huber")),
        xyz_soft_loss_weight_active=_xyz_soft_loss_weight_active(cfg, global_step=global_step),
        xyz_soft_loss_type=str(cfg.get("xyz_soft_loss", "huber")),
        segment_vector_loss_weight_active=_segment_vector_loss_weight_active(cfg, global_step=global_step),
        segment_vector_loss_type=str(cfg.get("segment_vector_loss", "huber")),
        straightness_loss_weight_active=_straightness_loss_weight_active(cfg, global_step=global_step),
        straightness_loss_type=str(cfg.get("straightness_loss", "huber")),
        tube_radius_loss_weight_active=_tube_radius_loss_weight_active(cfg, global_step=global_step),
        tube_radius_loss_type=str(cfg.get("tube_radius_loss", "huber")),
        distance_aware_coarse_targets_enabled=bool(cfg.get("distance_aware_coarse_targets_enabled", True)),
        distance_aware_coarse_target_radius=int(cfg.get("distance_aware_coarse_target_radius", 1)),
        distance_aware_coarse_target_sigma=float(cfg.get("distance_aware_coarse_target_sigma", 1.0)),
        distance_aware_coarse_target_loss=str(cfg.get("distance_aware_coarse_target_loss", "soft_ce")),
    )


@torch.no_grad()
def _evaluate_validation(
    *,
    model: AutoregFiberModel,
    dataloader: DataLoader,
    iterator,
    cfg: dict,
    device: torch.device,
    global_step: int,
) -> tuple[dict[str, float], Any]:
    model.eval()
    amp_enabled = str(cfg.get("mixed_precision", "no")).lower() != "no" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_enabled else torch.float32
    metric_dicts: list[dict[str, float]] = []
    for _ in range(int(cfg["val_batches_per_log"])):
        raw_batch, iterator = _next_batch(iterator, dataloader)
        batch = _move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch, scheduled_sampling_prob=0.0)
            loss_dict = _compute_losses_for_step(outputs, batch, cfg, global_step=global_step)
        metric_dicts.append(_loss_dict_to_metrics(loss_dict))
    model.train()
    return _mean_metric_dict(metric_dicts, prefix="val_"), iterator


@torch.no_grad()
def _evaluate_rollout_validation(*, model: AutoregFiberModel, dataset: Dataset, cfg: dict) -> dict[str, float]:
    num_examples = min(int(cfg.get("rollout_val_examples_per_log", 1)), len(dataset))
    if num_examples <= 0:
        return {}
    metrics = []
    model.eval()
    for idx in range(num_examples):
        sample = dataset[idx]
        result = infer_autoreg_fiber(
            model,
            sample,
            greedy=True,
            max_steps=None if cfg.get("rollout_val_max_steps") is None else int(cfg["rollout_val_max_steps"]),
        )
        pred = torch.from_numpy(np.asarray(result["predicted_fiber_local_zyx"], dtype=np.float32))
        target = sample["target_xyz"][: pred.shape[0]]
        if pred.numel() == 0 or target.numel() == 0:
            xyz_l1 = 0.0
        else:
            xyz_l1 = float((pred - target).abs().mean().item())
        metrics.append(
            {
                "xyz_l1": xyz_l1,
                "stop_count_error": float(abs(int(pred.shape[0]) - int(sample["target_lengths"].item()))),
            }
        )
    model.train()
    return _mean_metric_dict(metrics, prefix="rollout_val_")


def run_autoreg_fiber_training(
    config: dict,
    *,
    dataset=None,
    model: AutoregFiberModel | None = None,
    device: str | torch.device | None = None,
    max_steps: int | None = None,
) -> dict:
    cfg = validate_autoreg_fiber_config(config)
    runtime = _initialize_distributed_runtime(device)
    _seed_everything(int(cfg["seed"]) + int(runtime.rank))

    out_dir = Path(cfg["out_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset is None:
        dataset = AutoregFiberDataset(cfg)
    train_dataset, val_dataset, split_diagnostics = _split_dataset(
        dataset,
        cfg=cfg,
        seed=int(cfg["seed"]),
        val_fraction=float(cfg.get("val_fraction", 0.0)),
    )
    train_worker_init = functools.partial(_seed_worker, base_seed=int(cfg["seed"]), rank=int(runtime.rank))
    val_worker_init = functools.partial(_seed_worker, base_seed=int(cfg["seed"]) + 1, rank=0)
    train_sampler = None
    if runtime.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(runtime.world_size),
            rank=int(runtime.rank),
            shuffle=True,
            drop_last=False,
            seed=int(cfg["seed"]),
        )
    train_dataloader = _make_dataloader(
        train_dataset,
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg["num_workers"]),
        shuffle=not runtime.is_distributed,
        seed=int(cfg["seed"]),
        sampler=train_sampler,
        worker_init_fn=train_worker_init,
    )
    val_dataloader = None
    if val_dataset is not None and runtime.is_main_process:
        val_dataloader = _make_dataloader(
            val_dataset,
            batch_size=int(cfg["batch_size"]),
            num_workers=int(cfg["val_num_workers"]),
            shuffle=False,
            seed=int(cfg["seed"]) + 1,
            worker_init_fn=val_worker_init,
        )

    if model is None:
        model = AutoregFiberModel(cfg)
    raw_model = model.to(runtime.device)
    raw_model.train()

    optimizer = create_optimizer(dict(cfg["optimizer"]), raw_model)
    scheduler = None
    scheduler_name = str(cfg.get("scheduler", "constant")).lower()
    total_steps = int(max_steps or cfg["num_steps"])
    if scheduler_name != "constant":
        from vesuvius.models.training.lr_schedulers import get_scheduler

        scheduler = get_scheduler(
            scheduler_type=scheduler_name,
            optimizer=optimizer,
            initial_lr=float(cfg["optimizer"]["learning_rate"]),
            max_steps=total_steps,
            **dict(cfg.get("scheduler_kwargs") or {}),
        )

    preloaded_ckpt = _load_checkpoint_payload(cfg.get("load_ckpt"))
    _validate_checkpoint_compatibility(cfg, preloaded_ckpt)
    start_step = 0
    if preloaded_ckpt is not None:
        raw_model.load_state_dict(preloaded_ckpt["model"])
        if not bool(cfg.get("load_weights_only", False)):
            start_step = int(preloaded_ckpt.get("step", 0))
            if "optimizer" in preloaded_ckpt:
                optimizer.load_state_dict(preloaded_ckpt["optimizer"])
            if scheduler is not None and "lr_scheduler" in preloaded_ckpt:
                scheduler.load_state_dict(preloaded_ckpt["lr_scheduler"])

    train_model = _wrap_model_for_ddp(raw_model, runtime, cfg)
    wandb = _maybe_import_wandb(cfg) if runtime.is_main_process else None
    wandb_run = None
    saved_checkpoints: list[str] = []
    final_checkpoint_path = None
    history: list[dict[str, float]] = []
    progress_bar = None
    startup_started = time.perf_counter()
    startup_ms = 0.0

    try:
        if wandb is not None and runtime.is_main_process:
            wandb_kwargs = {"project": cfg["wandb_project"], "config": _sanitize_wandb_config(cfg)}
            if cfg.get("wandb_entity") is not None:
                wandb_kwargs["entity"] = cfg["wandb_entity"]
            if cfg.get("wandb_run_name") is not None:
                wandb_kwargs["name"] = cfg["wandb_run_name"]
            if bool(cfg.get("wandb_resume", False)):
                wandb_kwargs["resume"] = cfg.get("wandb_resume_mode", "allow")
                if cfg.get("wandb_run_id") is not None:
                    wandb_kwargs["id"] = cfg["wandb_run_id"]
            wandb_run = wandb.init(**wandb_kwargs)
            active_run = getattr(wandb, "run", None) or wandb_run
            active_run_id = getattr(active_run, "id", None)
            if active_run_id is not None:
                cfg["wandb_run_id"] = str(active_run_id)
            _log_wandb_dataset_summary(
                wandb=wandb,
                wandb_run=wandb_run,
                dataset=dataset,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                cfg=cfg,
                step=start_step,
            )

        if bool(cfg.get("ckpt_at_step_zero", False)) and start_step == 0 and runtime.is_main_process:
            ckpt_path = _save_checkpoint(
                out_dir=out_dir,
                filename="ckpt_000000.pth",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=0,
            )
            saved_checkpoints.append(str(ckpt_path))
        if runtime.is_distributed:
            _maybe_barrier(runtime)

        train_sampler_epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(train_sampler_epoch)

        def _on_train_iterator_reset():
            nonlocal train_sampler_epoch
            if train_sampler is not None:
                train_sampler_epoch += 1
                train_sampler.set_epoch(train_sampler_epoch)

        train_iterator = iter(train_dataloader)
        val_iterator = iter(val_dataloader) if val_dataloader is not None else None
        global_step = int(start_step)
        progress_bar = tqdm(total=max(0, total_steps - global_step), desc="autoreg_fiber", leave=False) if runtime.is_main_process else None
        startup_ms = 1000.0 * (time.perf_counter() - startup_started)
        amp_enabled = str(cfg.get("mixed_precision", "no")).lower() != "no" and runtime.device.type == "cuda"
        amp_dtype = torch.bfloat16 if amp_enabled else torch.float32

        while global_step < total_steps:
            raw_batch, train_iterator = _next_batch(
                train_iterator,
                train_dataloader,
                on_reset=_on_train_iterator_reset if train_sampler is not None else None,
            )
            batch = _move_batch_to_device(raw_batch, runtime.device)
            optimizer.zero_grad(set_to_none=True)
            scheduled_sampling_prob = _scheduled_sampling_prob(cfg, global_step=global_step)
            offset_feedback_enabled, refine_feedback_enabled = _scheduled_sampling_feedback_state(cfg, global_step=global_step)
            with torch.autocast(device_type=runtime.device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = train_model(
                    batch,
                    scheduled_sampling_prob=scheduled_sampling_prob,
                    scheduled_sampling_pattern=str(cfg.get("scheduled_sampling_mode", "linear_token_greedy")),
                    scheduled_sampling_offset_feedback_enabled=offset_feedback_enabled,
                    scheduled_sampling_refine_feedback_enabled=refine_feedback_enabled,
                )
                loss_dict = _compute_losses_for_step(outputs, batch, cfg, global_step=global_step)
            # Two-step backward: free the teacher-forced autograd graph
            # *before* running the rollout's K-deep forward. Holding both
            # graphs simultaneously OOMs on 80 GB H100s at 4/GPU; backwarding
            # the teacher loss first releases its activations, and the rollout
            # gradients then accumulate into the same param .grad tensors
            # without zero_grad in between.
            teacher_loss = loss_dict["loss"]
            if not torch.isfinite(teacher_loss):
                raise RuntimeError(
                    f"Encountered non-finite teacher training loss at step {global_step}: {teacher_loss.item()}"
                )
            teacher_loss.backward()
            rollout_fired = False
            rollout_loss_value: float | None = None
            if (
                bool(cfg.get("rollout_in_loop_enabled", False))
                and global_step >= int(cfg.get("rollout_in_loop_start_step", 0))
                and float(cfg.get("rollout_in_loop_prob", 0.0)) > 0.0
            ):
                rollout_prob = float(cfg["rollout_in_loop_prob"])
                # Drive the per-step coin flip from the same RNG on every
                # rank so DDP stays in sync (without an extra all_reduce).
                rng_value = float(
                    torch.rand(
                        (),
                        device=runtime.device,
                        generator=None,
                    ).item()
                )
                if runtime.world_size > 1:
                    rng_tensor = torch.tensor([rng_value], device=runtime.device)
                    torch.distributed.broadcast(rng_tensor, src=0)
                    rng_value = float(rng_tensor.item())
                if rng_value < rollout_prob:
                    rollout_fired = True
                    rollout_steps = int(cfg["rollout_in_loop_steps"])
                    rollout_batch = _slice_target_batch(batch, max_length=rollout_steps)
                    with torch.autocast(
                        device_type=runtime.device.type, dtype=amp_dtype, enabled=amp_enabled
                    ):
                        rollout_outputs = train_model(rollout_batch, rollout_steps=rollout_steps)
                        rollout_loss_dict = _compute_losses_for_step(
                            rollout_outputs, rollout_batch, cfg, global_step=global_step
                        )
                    weight = float(cfg.get("rollout_in_loop_loss_weight", 1.0))
                    weighted_rollout_loss = weight * rollout_loss_dict["loss"]
                    if not torch.isfinite(weighted_rollout_loss):
                        raise RuntimeError(
                            f"Encountered non-finite rollout training loss at step {global_step}: {weighted_rollout_loss.item()}"
                        )
                    weighted_rollout_loss.backward()
                    rollout_loss_value = float(rollout_loss_dict["loss"].detach().item())
                    # Surface every per-loss metric under a ``rollout_loss_*``
                    # namespace so the dashboard can plot them next to the
                    # teacher-forced counterparts.
                    for k, v in rollout_loss_dict.items():
                        if isinstance(v, torch.Tensor):
                            v = float(v.detach().item())
                        loss_dict[f"rollout_loss_{k}"] = torch.as_tensor(v)
            grad_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=float(cfg["grad_clip"]))
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            skipped_step = 0.0
            if _distributed_all_finite(grad_norm_value, device=runtime.device, runtime=runtime):
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            else:
                skipped_step = 1.0
            global_step += 1

            metrics = _loss_dict_to_metrics(loss_dict)
            metrics["current_lr"] = float(optimizer.param_groups[0]["lr"])
            metrics["grad_norm"] = grad_norm_value
            metrics["scheduled_sampling_prob"] = float(scheduled_sampling_prob)
            metrics["rollout_in_loop_fired"] = 1.0 if rollout_fired else 0.0
            if rollout_loss_value is not None:
                metrics["rollout_loss"] = rollout_loss_value
            metrics["step"] = float(global_step)
            metrics.update(split_diagnostics)
            if skipped_step > 0.0:
                metrics["skipped_step_nonfinite_grad"] = skipped_step
            metrics = _metric_dict_mean_across_ranks(metrics, device=runtime.device, runtime=runtime)

            should_run_validation_step = val_dataset is not None and global_step % int(cfg["log_frequency"]) == 0
            should_run_validation = runtime.is_main_process and val_dataloader is not None and should_run_validation_step
            if should_run_validation:
                val_metrics, val_iterator = _evaluate_validation(
                    model=raw_model,
                    dataloader=val_dataloader,
                    iterator=val_iterator,
                    cfg=cfg,
                    device=runtime.device,
                    global_step=global_step,
                )
                metrics.update(val_metrics)
                metrics.update(_evaluate_rollout_validation(model=raw_model, dataset=val_dataset, cfg=cfg))

            wandb_payload = _wandb_metrics_payload(metrics)
            should_log_projection_images_step = (
                bool(cfg.get("wandb_log_images", True)) and
                global_step % int(cfg["wandb_image_frequency"]) == 0
            )
            should_log_xy_images_step = (
                bool(cfg.get("wandb_log_images", True)) and
                bool(cfg.get("wandb_log_xy_slice_images", True)) and
                global_step % int(cfg["wandb_xy_slice_image_frequency"]) == 0
            )
            should_log_projection_images = runtime.is_main_process and wandb is not None and should_log_projection_images_step
            should_log_xy_images = runtime.is_main_process and wandb is not None and should_log_xy_images_step
            if should_log_projection_images or should_log_xy_images:
                # Train side stays teacher-forced (the current batch's outputs).
                # Val side switches to autoregressive rollout so engineers see
                # the real model behaviour, not just teacher-forced predictions
                # that hide exposure-bias drift. Backports the pattern from
                # autoreg_mesh/train.py:1540-1582.
                raw_val_sample = None
                val_infer = None
                if val_dataset is not None and len(val_dataset) > 0:
                    raw_val_sample = val_dataset[0]
                    raw_model.eval()
                    try:
                        val_infer = infer_autoreg_fiber(raw_model, raw_val_sample, greedy=True)
                    finally:
                        raw_model.train()
                if should_log_projection_images:
                    train_projection_image = wandb.Image(
                        _make_teacher_forced_prediction_canvas(
                            batch,
                            outputs,
                            sample_idx=0,
                            line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                        ),
                        caption=f"step={global_step} train teacher-forced fiber skeleton",
                    )
                    wandb_payload["train/example_projection"] = train_projection_image
                    if raw_val_sample is not None and val_infer is not None:
                        val_projection_image = wandb.Image(
                            _make_inference_prediction_canvas(
                                raw_val_sample,
                                val_infer,
                                line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                            ),
                            caption=f"step={global_step} val autoregressive fiber skeleton",
                        )
                        wandb_payload["val/example_projection"] = val_projection_image
                if should_log_xy_images:
                    wandb_payload["train/example_xy"] = wandb.Image(
                        _make_teacher_forced_xy_slice_canvas(
                            batch,
                            outputs,
                            sample_idx=0,
                            line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                            depth_tolerance=float(cfg.get("wandb_xy_slice_depth_tolerance", 0.75)),
                        ),
                        caption=f"step={global_step} train xy slice fiber skeleton",
                    )
                    if raw_val_sample is not None and val_infer is not None:
                        wandb_payload["val/example_xy"] = wandb.Image(
                            _make_inference_xy_slice_canvas(
                                raw_val_sample,
                                val_infer,
                                line_thickness=int(cfg.get("wandb_xy_slice_line_thickness", 1)),
                                depth_tolerance=float(cfg.get("wandb_xy_slice_depth_tolerance", 0.75)),
                            ),
                            caption=f"step={global_step} val autoregressive xy slice fiber skeleton",
                        )

            if runtime.is_main_process:
                history.append(dict(metrics))
            if wandb is not None and runtime.is_main_process:
                wandb.log(wandb_payload, step=global_step)
            if progress_bar is not None:
                progress_bar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
                progress_bar.update(1)

            should_write_checkpoint_step = global_step % int(cfg["ckpt_frequency"]) == 0
            should_write_checkpoint = runtime.is_main_process and should_write_checkpoint_step
            if should_write_checkpoint:
                ckpt_path = _save_checkpoint(
                    out_dir=out_dir,
                    filename=f"ckpt_{global_step:06}.pth",
                    model=raw_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=cfg,
                    step=global_step,
                )
                saved_checkpoints.append(str(ckpt_path))
            if runtime.is_distributed and (should_run_validation_step or should_write_checkpoint_step):
                _maybe_barrier(runtime)

        if progress_bar is not None:
            progress_bar.close()

        if bool(cfg.get("save_final_checkpoint", True)) and runtime.is_main_process:
            final_ckpt = _save_checkpoint(
                out_dir=out_dir,
                filename="final.pth",
                model=raw_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=cfg,
                step=global_step,
            )
            final_checkpoint_path = str(final_ckpt)
            saved_checkpoints.append(final_checkpoint_path)

        return {
            "model": raw_model,
            "optimizer": optimizer,
            "history": history if runtime.is_main_process else [],
            "final_metrics": history[-1] if history and runtime.is_main_process else {},
            "start_step": start_step,
            "wandb_run_id": cfg.get("wandb_run_id"),
            "checkpoint_paths": saved_checkpoints if runtime.is_main_process else [],
            "final_checkpoint_path": final_checkpoint_path if runtime.is_main_process else None,
            "out_dir": str(out_dir),
            "is_main_process": runtime.is_main_process,
            "rank": int(runtime.rank),
            "world_size": int(runtime.world_size),
            "device": str(runtime.device),
            "startup_ms": float(startup_ms),
            "split_diagnostics": dict(split_diagnostics),
        }
    finally:
        if progress_bar is not None:
            progress_bar.close()
        if wandb is not None and runtime.is_main_process:
            wandb.finish()
        if runtime.is_distributed and dist.is_available() and dist.is_initialized() and runtime.initialized_process_group:
            dist.destroy_process_group()


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    cfg = load_autoreg_fiber_config(Path(config_path))
    result = run_autoreg_fiber_training(cfg)
    if bool(result.get("is_main_process", True)):
        print(json.dumps(result["final_metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    train()


__all__ = [
    "load_autoreg_fiber_model_from_checkpoint",
    "run_autoreg_fiber_training",
    "train",
]
