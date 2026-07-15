from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.loader import (
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
    pred_dir = direction_output(output)
    pred_presence = presence_output(output)
    direction_mask = batch.direction_mask.expand_as(pred_dir)
    direction_error = (pred_dir - batch.direction_target) ** 2
    direction_error = direction_error * batch.direction_weight
    direction_loss = _masked_mean(direction_error, direction_mask)

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
            volume_cache_memory_mib=raw_config.get("volume_cache_memory_mib"),
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
    direction_weight: float,
    presence_weight: float,
) -> dict[str, float]:
    model.eval()
    total_rows: list[dict[str, float]] = []
    consumed = 0
    while consumed < sample_count:
        batch = loader.load_batch(
            start_sample_index + consumed,
            sample_mode="random",
            device=device,
        )
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
        return {"total": math.inf, "direction": math.inf, "presence": math.inf}
    return {
        key: float(sum(row[key] for row in total_rows) / len(total_rows))
        for key in total_rows[0]
    }


def _slice_batch(batch: FiberTrace3DBatch, start: int, stop: int) -> FiberTrace3DBatch:
    return FiberTrace3DBatch(
        volume=batch.volume[start:stop],
        valid_mask=batch.valid_mask[start:stop],
        direction_target=batch.direction_target[start:stop],
        direction_weight=batch.direction_weight[start:stop],
        direction_mask=batch.direction_mask[start:stop],
        presence_target=batch.presence_target[start:stop],
        presence_mask=batch.presence_mask[start:stop],
        cp_local_zyx=batch.cp_local_zyx[start:stop],
        crop_origin_zyx=batch.crop_origin_zyx[start:stop],
        sample_indices=batch.sample_indices[start:stop],
        record_indices=batch.record_indices[start:stop],
        control_point_indices=batch.control_point_indices[start:stop],
        fiber_paths=batch.fiber_paths[start:stop],
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


def run_training(config_path: str | Path) -> None:
    raw_config = _load_raw_config(config_path)
    loader_config = load_config(config_path)
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    loader = FiberTrace3DLoader(loader_config)
    test_loader = None
    if raw_config.get("test_datasets"):
        test_raw = dict(raw_config)
        test_raw["datasets"] = raw_config["test_datasets"]
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
    resume = training.get("resume") or raw_config.get("resume")
    start_step = 0
    if resume:
        start_step = _load_snapshot(resume, model=model, optimizer=optimizer, map_location=device)

    run_dir, snapshot_dir = _resolve_run_layout(raw_config)
    writer = _make_summary_writer(
        run_dir,
        enabled=bool(training.get("tensorboard_enabled", True)),
    )
    if writer is not None:
        writer.add_text("config/json", json.dumps(_json_safe(raw_config), indent=2, sort_keys=True), 0)
        writer.add_text(
            "train_sample_3d/layout",
            "Rows: yx, zx, zy principal slices through the sampled CP. "
            "Columns: image, target presence, predicted presence, direction angle error.",
            0,
        )

    max_steps = int(training.get("max_steps", 1))
    if max_steps <= 0:
        max_steps = max(1, math.ceil(loader.sample_count / max(loader.config.batch_size, 1)))
    scalar_interval = int(training.get("scalar_log_interval", 100))
    checkpoint_interval = int(training.get("checkpoint_interval", 100))
    test_interval = int(training.get("test_interval", 0))
    sample_vis_interval = int(
        training.get("sample_vis_interval", training.get("train_sample_vis_interval", 1000))
    )
    test_control_points = int(training.get("test_control_points", loader.config.batch_size))
    if test_control_points <= 0:
        test_control_points = test_loader.sample_count if test_loader is not None else loader.sample_count
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    best_metric = math.inf

    print(
        "fiber_trace_3d train: "
        f"samples={loader.sample_count} batch_size={loader.config.batch_size} "
        f"patch_shape_zyx={loader.config.patch_shape_zyx} device={device} run_dir={run_dir} "
        f"trace2cp_enabled={bool(trace2cp_loader is not None)}",
        flush=True,
    )

    for step in range(start_step + 1, max_steps + 1):
        load_start = time.perf_counter()
        sample_index = (step - 1) * loader.config.batch_size
        batch = loader.load_batch(sample_index, sample_mode="random", device=device)
        load_ms = (time.perf_counter() - load_start) * 1000.0
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
                f"load_ms={load_ms:.1f} fw_bw_step_ms={step_ms:.1f}",
                flush=True,
            )
        if writer is not None and (step == 1 or step % scalar_interval == 0):
            writer.add_scalar("train/loss_total", losses["total"], step)
            writer.add_scalar("train/loss_direction", losses["direction"], step)
            writer.add_scalar("train/loss_presence", losses["presence"], step)
            writer.add_scalar("timing/load_ms", load_ms, step)
            writer.add_scalar("timing/fw_bw_step_ms", step_ms, step)
        if writer is not None and sample_vis_interval > 0 and (
            step == 1 or step % sample_vis_interval == 0
        ):
            was_training = bool(model.training)
            model.eval()
            with torch.no_grad():
                vis_output = model(batch.volume[:1])
            if was_training:
                model.train()
            writer.add_image(
                "train_sample_3d/principal_slices",
                _make_train_sample_3d_sheet(batch, vis_output),
                step,
                dataformats="HWC",
            )

        metric = losses["total"]
        metric_name = "train/loss_total"
        if test_loader is not None and test_interval > 0 and step % test_interval == 0:
            test_losses = evaluate_dense_loss(
                model,
                test_loader,
                device=device,
                start_sample_index=int(training.get("test_start_sample_index", 0)),
                sample_count=test_control_points,
                direction_weight=direction_weight,
                presence_weight=presence_weight,
            )
            metric = test_losses["total"]
            metric_name = "test/loss_total"
            print(
                f"test step={step} loss_total={test_losses['total']:.6f} "
                f"loss_direction={test_losses['direction']:.6f} "
                f"loss_presence={test_losses['presence']:.6f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("test/loss_total", test_losses["total"], step)
                writer.add_scalar("test/loss_direction", test_losses["direction"], step)
                writer.add_scalar("test/loss_presence", test_losses["presence"], step)
        if trace2cp_loader is not None and test_interval > 0 and step % test_interval == 0:
            trace2cp_metric = _evaluate_trace2cp_metric_fixed_set_3d(
                model,
                trace2cp_loader,
                image_normalization=loader.config.image_normalization,
                cfg=trace2cp_cfg,
                device=device,
            )
            metric = trace2cp_metric.error_mean
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

        if step % checkpoint_interval == 0 or step == max_steps:
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


def _angle_error_rgb(angle_degrees: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_degrees, dtype=np.float32)
    valid = np.asarray(mask, dtype=bool) & np.isfinite(angle)
    scaled = np.clip(angle / 90.0, 0.0, 1.0)
    out = np.zeros((*angle.shape, 3), dtype=np.uint8)
    out[..., 0] = np.rint(scaled * 255.0).astype(np.uint8)
    out[..., 2] = np.rint((1.0 - scaled) * 255.0).astype(np.uint8)
    out[~valid] = 0
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


def _make_train_sample_3d_sheet(
    batch: FiberTrace3DBatch,
    output: torch.Tensor,
) -> np.ndarray:
    volume = batch.volume[0, 0].detach().cpu().numpy()
    valid = batch.valid_mask[0, 0].detach().cpu().numpy().astype(bool)
    target_presence = batch.presence_target[0, 0].detach().cpu().numpy()
    pred_presence = presence_output(output)[0, 0].detach().cpu().numpy()
    direction_mask = batch.direction_mask[0, 0].detach().cpu().numpy().astype(bool)

    target_encoded = batch.direction_target[0].permute(1, 2, 3, 0)
    pred_encoded = direction_output(output)[0].permute(1, 2, 3, 0)
    target_dir = decode_lasagna_direction_3x2_analytic(target_encoded)
    pred_dir = decode_lasagna_direction_3x2_analytic(pred_encoded)
    agreement = torch.abs(torch.sum(target_dir * pred_dir, dim=-1)).clamp(0.0, 1.0)
    angle = torch.rad2deg(torch.acos(agreement)).detach().cpu().numpy()

    cp = torch.round(batch.cp_local_zyx[0]).to(dtype=torch.long).detach().cpu().numpy()
    cp = np.clip(cp, [0, 0, 0], np.asarray(volume.shape, dtype=np.int64) - 1)
    z, y, x = (int(v) for v in cp)
    slice_specs = (
        ("yx", volume[z, :, :], valid[z, :, :], target_presence[z, :, :], pred_presence[z, :, :], angle[z, :, :], direction_mask[z, :, :], y, x),
        ("zx", volume[:, y, :], valid[:, y, :], target_presence[:, y, :], pred_presence[:, y, :], angle[:, y, :], direction_mask[:, y, :], z, x),
        ("zy", volume[:, :, x], valid[:, :, x], target_presence[:, :, x], pred_presence[:, :, x], angle[:, :, x], direction_mask[:, :, x], z, y),
    )

    rows: list[np.ndarray] = []
    gap = 4
    for _name, image, image_valid, target_p, pred_p, angle_p, angle_mask, cp_row, cp_col in slice_specs:
        image_rgb = np.repeat(_image_to_u8(image, image_valid)[..., None], 3, axis=2)
        target_rgb = _gray_to_rgb(target_p, mask=image_valid)
        pred_rgb = _gray_to_rgb(pred_p, mask=image_valid)
        angle_rgb = _angle_error_rgb(angle_p, mask=angle_mask)
        panels = [image_rgb, target_rgb, pred_rgb, angle_rgb]
        for panel in panels:
            _mark_slice_cp(panel, cp_row, cp_col)
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


def run_benchmark(config_path: str | Path, *, load_only: bool, batches: int) -> None:
    raw_config = _load_raw_config(config_path)
    loader = FiberTrace3DLoader(load_config(config_path))
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    model = build_fiber_trace_3d_model(raw_config).to(device)
    model.eval()
    direction_weight = float(training.get("direction_weight", 1.0))
    presence_weight = float(training.get("presence_weight", 1.0))
    print("batch patches total_ms load_ms fw_ms")
    for batch_index in range(1, int(batches) + 1):
        start = time.perf_counter()
        batch = loader.load_batch(
            (batch_index - 1) * loader.config.batch_size,
            sample_mode="random",
            device=device,
        )
        load_ms = (time.perf_counter() - start) * 1000.0
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
        print(
            f"{batch_index:5d} {loader.config.batch_size:7d} "
            f"{total_ms:8.2f} {load_ms:8.2f} {fw_ms:8.2f}",
            flush=True,
        )


def run_prefetch(config_path: str | Path, *, prefetch_steps: int, workers: int | None) -> None:
    loader = FiberTrace3DLoader(load_config(config_path))
    if int(prefetch_steps) == 0:
        sample_count = loader.sample_count
    else:
        sample_count = int(prefetch_steps) * int(loader.config.batch_size)
    summary = loader.prefetch(0, sample_count, workers=workers)
    print("fiber_trace_3d prefetch summary: " + json.dumps(summary, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--prefetch-steps", type=int, default=1)
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
    args = parser.parse_args()
    if args.prefetch:
        run_prefetch(
            args.config,
            prefetch_steps=int(args.prefetch_steps),
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
        run_training(args.config)


if __name__ == "__main__":
    main()
