from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
)
from vesuvius.neural_tracing.fiber_trace_3d.model import (
    build_fiber_trace_3d_model,
    direction_output,
    presence_output,
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


def _resolve_run_layout(config: dict[str, Any]) -> tuple[Path, Path]:
    training = dict(config.get("training", {}))
    run_path = Path(str(training.get("run_path", config.get("run_path", "runs/fiber_trace_3d"))))
    run_name = _sanitize_run_name(training.get("run_name", config.get("run_name", "fiber_trace_3d")))
    date_str = str(training.get("run_datestr") or datetime.now().strftime("%Y%m%d_%H%M%S"))
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


def _resolve_dense_test_selection(
    training: dict[str, Any],
    *,
    loader_sample_count: int,
    default_count: int,
) -> tuple[int, int, str]:
    raw_count = int(training.get("test_control_points", int(default_count)))
    if raw_count <= 0:
        return int(loader_sample_count), 0, "flat"
    return raw_count, int(training.get("test_start_sample_index", 0)), "random"


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=value.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (value * mask_f).sum() / denom


def compute_losses(
    output: torch.Tensor,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
) -> dict[str, torch.Tensor]:
    require_materialized_targets(batch)
    assert batch.direction_indices_bzyx is not None
    assert batch.direction_target_sparse is not None
    assert batch.direction_weight_sparse is not None
    assert batch.presence_target is not None
    assert batch.presence_mask is not None
    pred_dir = direction_output(output)
    pred_presence = presence_output(output)
    indices = batch.direction_indices_bzyx.to(dtype=torch.long)
    if int(indices.shape[0]) > 0:
        pred_sparse = pred_dir[
            indices[:, 0],
            :,
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
        ]
        direction_error = (pred_sparse - batch.direction_target_sparse) ** 2
        direction_error = direction_error * batch.direction_weight_sparse
        direction_loss = direction_error.mean()
        pred_axis = decode_lasagna_direction_3x2_analytic(pred_sparse)
        target_axis = decode_lasagna_direction_3x2_analytic(batch.direction_target_sparse)
        agreement = torch.abs(torch.sum(pred_axis * target_axis, dim=-1)).clamp(0.0, 1.0)
        angle_mean_deg = torch.rad2deg(torch.acos(agreement)).mean()
    else:
        direction_loss = pred_dir.sum() * 0.0
        angle_mean_deg = pred_dir.sum() * 0.0

    presence_mask = batch.presence_mask.expand_as(pred_presence)
    presence_bce = F.binary_cross_entropy(
        pred_presence.clamp(1.0e-6, 1.0 - 1.0e-6),
        batch.presence_target,
        reduction="none",
    )
    pos = (batch.presence_target > 0.5) & presence_mask
    neg = (batch.presence_target <= 0.5) & presence_mask
    if bool(pos.any()) and bool(neg.any()):
        presence_loss = 0.5 * _masked_mean(presence_bce, pos) + 0.5 * _masked_mean(
            presence_bce, neg
        )
    else:
        presence_loss = _masked_mean(presence_bce, presence_mask)
    total = float(direction_weight) * direction_loss + float(presence_weight) * presence_loss
    return {
        "total": total,
        "direction": direction_loss,
        "presence": presence_loss,
        "angle_mean_deg": angle_mean_deg,
    }


def _save_snapshot(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, Any],
    metric: float | None,
    metric_name: str | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "config": _json_safe(config),
            "metric": metric,
            "metric_name": metric_name,
        },
        path,
    )


def _load_snapshot(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str = "cpu",
) -> int:
    payload = torch.load(path, map_location=map_location)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
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
            output = model(block_t.view(1, 1, *block.shape))[0]
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
        rows = _forward_loss(
            model,
            batch,
            direction_weight=direction_weight,
            presence_weight=presence_weight,
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
        sample_indices=batch.sample_indices[start:stop],
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


def _forward_loss(
    model: torch.nn.Module,
    batch: FiberTrace3DBatch,
    *,
    direction_weight: float,
    presence_weight: float,
    backward: bool,
) -> dict[str, float]:
    output = model(batch.volume)
    losses = compute_losses(
        output,
        batch,
        direction_weight=direction_weight,
        presence_weight=presence_weight,
    )
    if backward:
        losses["total"].backward()
    return {key: float(value.detach().cpu()) for key, value in losses.items()}


def run_training(config_path: str | Path, *, resume_checkpoint: str | Path | None = None) -> None:
    raw_config = _load_raw_config(config_path)
    loader_config = load_config(config_path)
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    loader = FiberTrace3DLoader(loader_config)
    test_loader = None
    if raw_config.get("test_datasets"):
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
        if trace2cp_cfg.enabled
        else None
    )

    model = build_fiber_trace_3d_model(raw_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 1.0e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    resume = (
        str(resume_checkpoint)
        if resume_checkpoint is not None
        else training.get("resume") or raw_config.get("resume")
    )
    start_step = 0
    if resume:
        start_step = _load_snapshot(resume, model=model, optimizer=optimizer, map_location=device)

    run_dir, snapshot_dir = _resolve_run_layout(raw_config)
    effective_config = _json_safe(raw_config)
    if resume_checkpoint is not None:
        effective_config.setdefault("training", {})["resume_cli"] = str(resume_checkpoint)
        effective_config.setdefault("training", {})["resume_effective"] = str(resume)
    writer = _make_summary_writer(
        run_dir,
        enabled=bool(training.get("tensorboard_enabled", True)),
    )
    if writer is not None:
        writer.add_text("config/json", json.dumps(effective_config, indent=2, sort_keys=True), 0)
        if resume:
            writer.add_text("config/resume", f"resume={resume}\ncheckpoint_step={start_step}", 0)
        writer.add_text(
            "train_sample_3d/layout",
            "Rows: yx, zx, zy principal slices through the sampled CP. "
            "Columns: image with projected GT line and predicted CP direction, target/context presence, predicted presence. "
            "Multiple batch samples are stacked vertically when configured.",
            0,
        )
        writer.add_text(
            "test_sample_3d/layout",
            "Rows: yx, zx, zy principal slices through the sampled test CP. "
            "Columns: image with projected GT line and predicted CP direction, target/context presence, predicted presence. "
            "Multiple batch samples are stacked vertically when configured.",
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
    test_interval = int(training.get("test_interval", 0))
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
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    loader_workers = _loader_worker_count(raw_config)
    loader_prefetch_factor = _loader_prefetch_factor(raw_config)
    loader_worker_device = _loader_worker_device(raw_config)
    loader_context = _loader_multiprocessing_context(raw_config)
    best_metric = math.inf

    print(
        "fiber_trace_3d train: "
        f"samples={loader.sample_count} batch_size={loader.config.batch_size} "
        f"max_sample_index={sample_index_limit} "
        f"patch_shape_zyx={loader.config.patch_shape_zyx} device={device} run_dir={run_dir} "
        f"trace2cp_enabled={bool(trace2cp_loader is not None)} "
        f"loader_workers={loader_workers} loader_prefetch_factor={loader_prefetch_factor} "
        f"loader_worker_device={loader_worker_device} "
        f"loader_multiprocessing_context={loader_context or 'default'}",
        flush=True,
    )
    if resume:
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
                model,
                test_loader,
                device=device,
                start_sample_index=test_start_sample_index,
                sample_count=test_control_points,
                sample_mode=test_sample_mode,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
            )
            metric = float(test_losses["total"])
            metric_name = "test/loss_total"
            print(
                f"test step={step} loss_total={test_losses['total']:.6f} "
                f"loss_direction={test_losses['direction']:.6f} "
                f"loss_presence={test_losses['presence']:.6f} "
                f"angle_mean_deg={test_losses['angle_mean_deg']:.2f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("test/loss_total", test_losses["total"], step)
                writer.add_scalar("test/loss_direction", test_losses["direction"], step)
                writer.add_scalar("test/loss_presence", test_losses["presence"], step)
                writer.add_scalar("test/angle_mean_deg", test_losses["angle_mean_deg"], step)
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
                        model,
                        vis_batch,
                        step,
                        sample_count=test_sample_vis_count,
                    )
        if trace2cp_loader is not None and test_interval > 0:
            trace2cp_metric = _evaluate_trace2cp_metric_fixed_set_3d(
                model,
                trace2cp_loader,
                image_normalization=loader.config.image_normalization,
                cfg=trace2cp_cfg,
                device=device,
            )
            metric = float(trace2cp_metric.error_mean)
            metric_name = "test/trace2cp_error"
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

    initial_metric, initial_metric_name = run_configured_tests(start_step)
    if initial_metric is not None and initial_metric_name is not None:
        best_metric = float(initial_metric)
        _save_snapshot(
            snapshot_dir / "best.pt",
            model=model,
            optimizer=optimizer,
            step=start_step,
            config=raw_config,
            metric=best_metric,
            metric_name=initial_metric_name,
        )

    remaining_steps = None if max_steps is None else max(0, int(max_steps) - int(start_step))
    train_dataloader = _make_batch_dataloader(
        config_path,
        raw_config=raw_config,
        start_batch_index=start_step,
        batch_count=remaining_steps,
        sample_index_limit=sample_index_limit,
        sample_mode="random",
    )
    train_iterator = iter(train_dataloader) if train_dataloader is not None else None
    try:
        step = int(start_step)
        while max_steps is None or step < max_steps:
            step += 1
            sample_index = (step - 1) * loader.config.batch_size
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
            losses = _forward_loss(
                model,
                batch,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
                backward=True,
            )
            optimizer.step()
            step_ms = (time.perf_counter() - fw_start) * 1000.0

            if step <= 100 or step % scalar_interval == 0:
                print(
                    f"step={step} loss_total={losses['total']:.6f} "
                    f"loss_direction={losses['direction']:.6f} "
                    f"loss_presence={losses['presence']:.6f} "
                    f"angle_mean_deg={losses['angle_mean_deg']:.2f} "
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
                writer.add_scalar("timing/load_ms", load_ms, step)
                writer.add_scalar("timing/load_wait_ms", wait_ms, step)
                writer.add_scalar("timing/batch_to_device_ms", to_device_ms, step)
                writer.add_scalar("timing/target_ms", target_ms, step)
                writer.add_scalar("timing/fw_bw_step_ms", step_ms, step)
            if writer is not None and sample_vis_interval > 0 and (
                step == 1 or step % sample_vis_interval == 0
            ):
                _write_3d_sample_sheet(
                    writer,
                    "train_sample_3d/principal_slices",
                    model,
                    batch,
                    step,
                    sample_count=sample_vis_count,
                )

            metric = losses["total"]
            metric_name = "train/loss_total"
            if test_interval > 0 and step % test_interval == 0:
                test_metric, test_metric_name = run_configured_tests(step)
                if test_metric is not None and test_metric_name is not None:
                    metric = test_metric
                    metric_name = test_metric_name

            if step % checkpoint_interval == 0 or (max_steps is not None and step == max_steps):
                _save_snapshot(
                    snapshot_dir / "current.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=raw_config,
                    metric=metric,
                    metric_name=metric_name,
                )
            if metric < best_metric:
                best_metric = float(metric)
                _save_snapshot(
                    snapshot_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    config=raw_config,
                    metric=best_metric,
                    metric_name=metric_name,
                )
    finally:
        if train_dataloader is not None:
            train_iterator = None
            train_dataloader = None
    if writer is not None:
        writer.flush()
        writer.close()


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
    pred_presence = presence_output(output)[0, 0].detach().cpu().numpy()

    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long).detach().cpu().numpy()
    cp = np.clip(cp, [0, 0, 0], np.asarray(volume.shape, dtype=np.int64) - 1)
    z, y, x = (int(v) for v in cp)
    cp_encoded = direction_output(output)[0, :, z, y, x].view(1, 6)
    cp_pred_xyz = decode_lasagna_direction_3x2_analytic(cp_encoded)[0].detach().cpu().numpy()
    cp_pred_zyx = cp_pred_xyz[[2, 1, 0]].astype(np.float32, copy=False)
    segment_offset = int(batch.target_segment_offsets[0].detach().cpu())
    segment_count = int(batch.target_segment_counts[0].detach().cpu())
    segment_slice = slice(segment_offset, segment_offset + segment_count)
    segments_start_zyx = batch.target_segment_starts_zyx[segment_slice].detach().cpu().numpy()
    segments_end_zyx = batch.target_segment_ends_zyx[segment_slice].detach().cpu().numpy()
    line_presence = _line_presence_for_display(
        tuple(int(v) for v in volume.shape),
        segments_start_zyx,
        segments_end_zyx,
    )
    target_presence = np.maximum(supervised_presence, line_presence)
    slice_specs = (
        ("yx", volume[z, :, :], valid[z, :, :], target_presence[z, :, :], pred_presence[z, :, :], 0, z, 1, 2, y, x),
        ("zx", volume[:, y, :], valid[:, y, :], target_presence[:, y, :], pred_presence[:, y, :], 1, y, 0, 2, z, x),
        ("zy", volume[:, :, x], valid[:, :, x], target_presence[:, :, x], pred_presence[:, :, x], 2, x, 0, 1, z, y),
    )

    rows: list[np.ndarray] = []
    gap = 4
    for (
        _name,
        image,
        image_valid,
        target_p,
        pred_p,
        plane_axis,
        plane_coord,
        row_axis,
        col_axis,
        cp_row,
        cp_col,
    ) in slice_specs:
        image_rgb = np.repeat(_image_to_u8(image, image_valid)[..., None], 3, axis=2)
        target_rgb = _gray_to_rgb(target_p, mask=image_valid)
        pred_rgb = _gray_to_rgb(pred_p, mask=image_valid)
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
        _mark_slice_cp(pred_rgb, cp_row, cp_col)
        panels = [image_rgb, target_rgb, pred_rgb]
        height = max(int(panel.shape[0]) for panel in panels)
        padded_panels = []
        for panel in panels:
            if int(panel.shape[0]) < height:
                pad = np.zeros((height - int(panel.shape[0]), int(panel.shape[1]), 3), dtype=np.uint8)
                panel = np.concatenate([panel, pad], axis=0)
            padded_panels.append(panel)
        sep = np.zeros((height, gap, 3), dtype=np.uint8)
        row = padded_panels[0]
        for panel in padded_panels[1:]:
            row = np.concatenate([row, sep, panel], axis=1)
        rows.append(row)
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
) -> None:
    was_training = bool(model.training)
    model.eval()
    take = min(max(1, int(sample_count)), int(batch.volume.shape[0]))
    with torch.no_grad():
        vis_output = model(batch.volume[:take])
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
        sample_index_limit: int | None = None,
        sample_mode: str,
        worker_device: str | torch.device = "cpu",
        profile: bool = False,
    ) -> None:
        self.config_source = config_source
        self.start_batch_index = int(start_batch_index)
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
        batch_index = self.start_batch_index + int(index)
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
    model = build_fiber_trace_3d_model(raw_config).to(device)
    model.eval()
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
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
        f"device={device} load_only={bool(load_only)}",
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
                _forward_loss(
                    model,
                    batch,
                    direction_weight=direction_weight,
                    presence_weight=presence_weight,
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
        run_prefetch(
            args.config,
            prefetch_steps=args.prefetch_steps,
            workers=args.prefetch_workers,
        )
    elif args.trace2cp_vis:
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
        run_benchmark(
            args.config,
            load_only=bool(args.load_only),
            batches=int(args.benchmark_batches),
        )
    else:
        run_training(args.config, resume_checkpoint=args.resume)


if __name__ == "__main__":
    main()
