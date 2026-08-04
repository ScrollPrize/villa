from __future__ import annotations

import argparse
import json
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vesuvius.neural_tracing.fiber_trace_2d.augmentation import (
    apply_value_augmentation_batch,
    overlay_line_coords_rgb,
    resolve_torch_device,
    value_only_params,
)
from vesuvius.neural_tracing.fiber_trace_2d.direction import (
    DirectionSupervision,
    build_direction_supervision,
    decode_lasagna_direction_xy,
    direction_angle_error_degrees,
    direction_mse_loss,
    line_cp_and_tangent_xy,
)
from vesuvius.neural_tracing.fiber_trace_2d.embedding import (
    ContrastiveEmbeddingMetrics,
    PresenceMetrics,
    contrastive_embedding_loss,
    contrastive_negative_reachable_mask,
    embedding_similarity_to_cp,
    presence_loss,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import FiberStrip2DBatch, FiberStrip2DLoader, SamplerFactory, load_config
from vesuvius.neural_tracing.fiber_trace_2d.model import (
    FiberStripDirectionModelConfig,
    FiberStripDirectionNet,
    direction_output,
    presence_output,
)
from vesuvius.neural_tracing.fiber_trace_2d.runner import (
    _predict_direction_field,
    _trace2cp_metric_bidirectional,
)


@dataclass(frozen=True)
class FiberStripTrainingConfig:
    run_path: str = "runs/fiber_trace_2d"
    run_name: str = "fiber_strip_direction"
    max_steps: int = 1000
    max_sample_index: int = 0
    learning_rate: float = 1.0e-3
    scalar_log_interval: int = 100
    tensorboard_image_interval: int = 1000
    checkpoint_interval: int = 1000
    test_interval: int = 1000
    test_control_points: int = 4
    test_start_sample_index: int = 0
    test_trace2cp_step_px: float = 4.0
    test_trace2cp_rf_margin_px: float | None = None
    train_control_points_per_step: int = 4
    device: str = "auto"
    tensorboard_enabled: bool = True
    model_hidden_channels: int = 64
    model_depth: int = 10
    presence_enabled: bool = False
    presence_weight: float = 1.0
    top_view_enabled: bool = False
    top_view_direction_weight: float = 1.0
    top_view_dt_weight: float = 1.0
    top_view_dt_radius_px: float = 30.0
    contrastive_enabled: bool = False
    contrastive_embedding_channels: int = 0
    contrastive_control_points_per_fiber: int = 8
    contrastive_weight: float = 1.0
    contrastive_negative_margin: float = 0.0
    pipeline_enabled: bool = True
    pipeline_depth: int = 16
    pipeline_workers: int = 8
    pipeline_isolated_loaders: bool = False


def _training_config_from_raw(raw: dict[str, Any]) -> FiberStripTrainingConfig:
    train = raw.get("training", {})
    if train is None:
        train = {}
    if not isinstance(train, dict):
        raise ValueError("'training' must be a JSON object when provided")

    def get(name: str, default: Any) -> Any:
        return train.get(name, raw.get(f"train_{name}", default))

    config = FiberStripTrainingConfig(
        run_path=str(get("run_path", "runs/fiber_trace_2d")),
        run_name=str(get("run_name", "fiber_strip_direction")),
        max_steps=int(get("max_steps", 1000)),
        max_sample_index=int(get("max_sample_index", 0)),
        learning_rate=float(get("learning_rate", 1.0e-3)),
        scalar_log_interval=max(1, int(get("scalar_log_interval", 100))),
        tensorboard_image_interval=max(1, int(get("tensorboard_image_interval", 1000))),
        checkpoint_interval=max(1, int(get("checkpoint_interval", 1000))),
        test_interval=max(1, int(get("test_interval", get("checkpoint_interval", 1000)))),
        test_control_points=int(get("test_control_points", get("control_points_per_step", 4))),
        test_start_sample_index=max(0, int(get("test_start_sample_index", 0))),
        test_trace2cp_step_px=float(get("test_trace2cp_step_px", 4.0)),
        test_trace2cp_rf_margin_px=(
            None
            if get("test_trace2cp_rf_margin_px", None) is None
            else float(get("test_trace2cp_rf_margin_px", None))
        ),
        train_control_points_per_step=max(1, int(get("control_points_per_step", get("control_points", 4)))),
        device=str(get("device", "auto")),
        tensorboard_enabled=bool(get("tensorboard_enabled", True)),
        model_hidden_channels=max(1, int(get("model_hidden_channels", 64))),
        model_depth=max(1, int(get("model_depth", 10))),
        presence_enabled=bool(get("presence_enabled", False)),
        presence_weight=float(get("presence_weight", 1.0)),
        top_view_enabled=bool(get("top_view_enabled", False)),
        top_view_direction_weight=float(get("top_view_direction_weight", 1.0)),
        top_view_dt_weight=float(get("top_view_dt_weight", 1.0)),
        top_view_dt_radius_px=float(get("top_view_dt_radius_px", 30.0)),
        contrastive_enabled=bool(get("contrastive_enabled", False)),
        contrastive_embedding_channels=max(0, int(get("contrastive_embedding_channels", 0))),
        contrastive_control_points_per_fiber=max(1, int(get("contrastive_control_points_per_fiber", 8))),
        contrastive_weight=float(get("contrastive_weight", 1.0)),
        contrastive_negative_margin=float(get("contrastive_negative_margin", 0.0)),
        pipeline_enabled=bool(get("pipeline_enabled", True)),
        pipeline_depth=max(1, int(get("pipeline_depth", 16))),
        pipeline_workers=max(0, int(get("pipeline_workers", 8))),
        pipeline_isolated_loaders=bool(get("pipeline_isolated_loaders", False)),
    )
    if config.max_steps < 0:
        raise ValueError("training.max_steps must be >= 0")
    if config.max_sample_index < 0:
        raise ValueError("training.max_sample_index must be >= 0")
    if config.test_control_points < 0:
        raise ValueError("training.test_control_points must be >= 0")
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive and finite")
    if not math.isfinite(config.test_trace2cp_step_px) or config.test_trace2cp_step_px <= 0.0:
        raise ValueError("training.test_trace2cp_step_px must be positive and finite")
    if config.contrastive_enabled and config.contrastive_embedding_channels <= 0:
        raise ValueError("training.contrastive_embedding_channels must be > 0 when contrastive is enabled")
    if not math.isfinite(config.presence_weight) or config.presence_weight < 0.0:
        raise ValueError("training.presence_weight must be non-negative and finite")
    if not math.isfinite(config.top_view_direction_weight) or config.top_view_direction_weight < 0.0:
        raise ValueError("training.top_view_direction_weight must be non-negative and finite")
    if not math.isfinite(config.top_view_dt_weight) or config.top_view_dt_weight < 0.0:
        raise ValueError("training.top_view_dt_weight must be non-negative and finite")
    if not math.isfinite(config.top_view_dt_radius_px) or config.top_view_dt_radius_px <= 0.0:
        raise ValueError("training.top_view_dt_radius_px must be positive and finite")
    if config.contrastive_enabled and (
        config.train_control_points_per_step % config.contrastive_control_points_per_fiber != 0
    ):
        raise ValueError(
            "training.control_points_per_step must be divisible by "
            "training.contrastive_control_points_per_fiber when contrastive is enabled"
        )
    if not math.isfinite(config.contrastive_weight) or config.contrastive_weight < 0.0:
        raise ValueError("training.contrastive_weight must be non-negative and finite")
    if not math.isfinite(config.contrastive_negative_margin):
        raise ValueError("training.contrastive_negative_margin must be finite")
    if config.test_trace2cp_rf_margin_px is not None and (
        not math.isfinite(config.test_trace2cp_rf_margin_px)
        or config.test_trace2cp_rf_margin_px < 0.0
    ):
        raise ValueError("training.test_trace2cp_rf_margin_px must be non-negative and finite")
    return config


def _test_loader_config_from_raw(
    raw: dict[str, Any], loader_config: Any
) -> Any | None:
    test_datasets = raw.get("test_datasets")
    if test_datasets is None:
        return None
    if not isinstance(test_datasets, list) or not test_datasets:
        raise ValueError("'test_datasets' must be a non-empty list when provided")
    return replace(loader_config, datasets=tuple(dict(entry) for entry in test_datasets))


def _validate_training_batch_config(training: FiberStripTrainingConfig, loader_config: Any) -> None:
    loader_batch_size = int(loader_config.batch_size)
    control_points = int(training.train_control_points_per_step)
    if loader_batch_size != control_points:
        raise ValueError(
            "training.control_points_per_step must match top-level batch_size "
            "because training loads one control-point batch per step: "
            f"control_points_per_step={control_points} batch_size={loader_batch_size}"
        )


def _contrastive_negative_candidate_mask(loader_config: Any) -> np.ndarray:
    shift_x = float(loader_config.augment.shift_x) if bool(loader_config.augment.enabled) else 0.0
    shift_y = float(loader_config.augment.shift_y) if bool(loader_config.augment.enabled) else 0.0
    return contrastive_negative_reachable_mask(
        tuple(int(v) for v in loader_config.patch_shape_hw),
        shift_x=shift_x,
        shift_y=shift_y,
        neighborhood_radius=1,
    )


def _presence_channel_count(training: FiberStripTrainingConfig | None) -> int:
    return 1 if training is not None and bool(training.presence_enabled) else 0


def _embedding_channel_count(training: FiberStripTrainingConfig | None) -> int:
    if training is None or not bool(training.contrastive_enabled):
        return 0
    return max(0, int(training.contrastive_embedding_channels))


def _top_view_scalar_channel_count(training: FiberStripTrainingConfig | None) -> int:
    return 1 if training is not None and bool(training.top_view_enabled) else 0


def _load_raw_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a JSON object")
    return raw


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


def _make_run_dir(config: FiberStripTrainingConfig) -> Path:
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(config.run_path).expanduser()
    base_name = f"{config.run_name}_{date}"
    run_dir = run_root / base_name
    suffix = 0
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            suffix += 1
            run_dir = run_root / f"{base_name}_{suffix:02d}"
    (run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def _make_summary_writer(run_dir: Path, *, enabled: bool, purge_step: int | None = None):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError(
            "TensorBoard logging requires the tensorboard package. "
            "Install tensorboard or set training.tensorboard_enabled=false."
        ) from exc
    kwargs: dict[str, Any] = {}
    if purge_step is not None:
        kwargs["purge_step"] = int(purge_step)
    return SummaryWriter(log_dir=str(run_dir), **kwargs)


def _flatten_batch(batch: FiberStrip2DBatch) -> tuple[np.ndarray, np.ndarray]:
    images = np.asarray(batch.images, dtype=np.float32)
    valid = np.asarray(batch.valid_mask, dtype=bool)
    if images.ndim != 5 or images.shape[2] != 1:
        raise ValueError("batch.images must have shape B,Z,1,H,W")
    b, z, _, h, w = images.shape
    return images.reshape(b * z, 1, h, w), valid.reshape(b * z, h, w)


def _prepare_images(images_np: np.ndarray, valid_np: np.ndarray, *, device: torch.device) -> torch.Tensor:
    images = torch.as_tensor(images_np, dtype=torch.float32, device=device)
    valid = torch.as_tensor(valid_np, dtype=torch.bool, device=device).unsqueeze(1)
    return _normalize_image_tensor(images, valid)


def _normalize_image_tensor(images: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)
    counts = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1)
    masked = torch.where(valid, images, torch.zeros_like(images))
    mean = masked.sum(dim=(2, 3), keepdim=True) / counts
    var = torch.where(valid, (images - mean) ** 2, torch.zeros_like(images)).sum(dim=(2, 3), keepdim=True) / counts
    std = torch.sqrt(var.clamp_min(1.0e-6))
    return torch.where(valid, (images - mean) / std, torch.zeros_like(images))


def _batch_images_to_tensors(
    batch: FiberStrip2DBatch,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    images_np, valid_np = _flatten_batch(batch)
    flat_count = int(images_np.shape[0])
    height, width = int(images_np.shape[2]), int(images_np.shape[3])
    flat_images = images_np.reshape(flat_count, height, width)
    flat_valids = valid_np.reshape(flat_count, height, width)

    params = tuple(batch.augmentation_params)
    if batch.augmentation_params and len(params) != flat_count:
        raise ValueError(
            "batch does not carry image augmentation parameters: "
            f"params={len(params)} expected={flat_count}"
        )
    if params and not all(param is None for param in params):
        value_params = [value_only_params(param) for param in params if param is not None]
        if len(value_params) != flat_count:
            raise ValueError("partially missing image augmentation parameters in batch")
        image_t, valid_t = apply_value_augmentation_batch(
            flat_images,
            flat_valids,
            value_params,
            device=device,
        )
    else:
        image_t = torch.as_tensor(flat_images, dtype=torch.float32, device=device)
        valid_t = torch.as_tensor(flat_valids, dtype=torch.bool, device=device)
    return image_t.unsqueeze(1), valid_t, valid_np


@dataclass(frozen=True)
class TopDistanceTransformSupervision:
    patch_indices: torch.Tensor
    y: torch.Tensor
    x: torch.Tensor
    target: torch.Tensor
    cp_xy: torch.Tensor
    normal_xy: torch.Tensor


def build_top_distance_transform_supervision(
    samples: tuple[Any, ...] | list[Any],
    valid_mask: np.ndarray | torch.Tensor,
    *,
    radius_px: float,
    device: torch.device,
) -> TopDistanceTransformSupervision:
    valid_np = np.asarray(valid_mask, dtype=bool)
    if valid_np.ndim != 3:
        raise ValueError("top DT valid_mask must have shape N,H,W")
    if len(samples) != int(valid_np.shape[0]):
        raise ValueError("samples length must match top DT valid-mask patch count")
    radius = float(radius_px)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("top DT radius must be positive and finite")
    height, width = int(valid_np.shape[1]), int(valid_np.shape[2])
    max_steps = int(math.ceil(math.hypot(float(height), float(width)))) + 2
    patch_rows: list[int] = []
    y_rows: list[int] = []
    x_rows: list[int] = []
    targets: list[float] = []
    cp_rows: list[np.ndarray] = []
    normal_rows: list[np.ndarray] = []
    for patch_index, sample in enumerate(samples):
        cp_and_tangent = line_cp_and_tangent_xy(
            sample.line_xy,
            getattr(sample, "control_point_xy", None),
        )
        if cp_and_tangent is None:
            continue
        cp_xy, tangent_xy = cp_and_tangent
        normal_xy = np.asarray([-float(tangent_xy[1]), float(tangent_xy[0])], dtype=np.float32)
        norm = float(np.linalg.norm(normal_xy))
        if not np.isfinite(norm) or norm <= 1.0e-6:
            continue
        normal_xy = (normal_xy / np.float32(norm)).astype(np.float32, copy=False)
        seen: set[tuple[int, int]] = set()
        for step in range(-max_steps, max_steps + 1):
            xy = np.asarray(cp_xy, dtype=np.float32) + normal_xy * np.float32(step)
            rounded = np.rint(xy).astype(np.int64)
            x = int(rounded[0])
            y = int(rounded[1])
            if not (0 <= y < height and 0 <= x < width):
                continue
            key = (y, x)
            if key in seen:
                continue
            seen.add(key)
            if not bool(valid_np[patch_index, y, x]):
                continue
            delta = np.asarray([float(x), float(y)], dtype=np.float32) - np.asarray(cp_xy, dtype=np.float32)
            distance = abs(float(np.dot(delta, normal_xy)))
            target = max(0.0, 1.0 - distance / radius)
            patch_rows.append(int(patch_index))
            y_rows.append(y)
            x_rows.append(x)
            targets.append(float(target))
            cp_rows.append(np.asarray(cp_xy, dtype=np.float32))
            normal_rows.append(normal_xy.astype(np.float32))
    if not patch_rows:
        empty_i = torch.zeros((0,), dtype=torch.long, device=device)
        return TopDistanceTransformSupervision(
            patch_indices=empty_i,
            y=empty_i,
            x=empty_i,
            target=torch.zeros((0,), dtype=torch.float32, device=device),
            cp_xy=torch.zeros((0, 2), dtype=torch.float32, device=device),
            normal_xy=torch.zeros((0, 2), dtype=torch.float32, device=device),
        )
    return TopDistanceTransformSupervision(
        patch_indices=torch.as_tensor(patch_rows, dtype=torch.long, device=device),
        y=torch.as_tensor(y_rows, dtype=torch.long, device=device),
        x=torch.as_tensor(x_rows, dtype=torch.long, device=device),
        target=torch.as_tensor(targets, dtype=torch.float32, device=device),
        cp_xy=torch.as_tensor(np.stack(cp_rows, axis=0), dtype=torch.float32, device=device),
        normal_xy=torch.as_tensor(np.stack(normal_rows, axis=0), dtype=torch.float32, device=device),
    )


def top_distance_transform_mse_loss(
    prediction: torch.Tensor,
    supervision: TopDistanceTransformSupervision,
) -> torch.Tensor:
    if prediction.ndim != 4 or int(prediction.shape[1]) != 1:
        raise ValueError("top DT prediction must have shape N,1,H,W")
    if int(supervision.patch_indices.numel()) == 0:
        raise ValueError("top DT loss requires at least one valid supervised pixel")
    gathered = prediction[supervision.patch_indices, 0, supervision.y, supervision.x]
    return torch.nn.functional.mse_loss(gathered, supervision.target)


@dataclass(frozen=True)
class _DirectionMetrics:
    angle_mean_deg: float
    loss_direction: float
    loss_presence: float = 0.0
    presence_positive_loss: float = 0.0
    presence_negative_loss: float = 0.0
    presence_positive_samples: int = 0
    presence_negative_samples: int = 0
    loss_contrastive: float = 0.0
    contrastive_positive_loss: float = 0.0
    contrastive_negative_loss: float = 0.0
    contrastive_pixel_negative_loss: float = 0.0
    contrastive_positive_samples: int = 0
    contrastive_negative_samples: int = 0
    contrastive_pixel_negative_samples: int = 0
    contrastive_similarity_mean_loss: float = 0.0
    contrastive_similarity_mean_value: float = 0.0
    contrastive_similarity_mean_target: float = 0.0
    contrastive_similarity_mean_samples: int = 0
    top_loss_direction: float = 0.0
    top_loss_dt: float = 0.0
    top_angle_mean_deg: float = 0.0
    top_direction_samples: int = 0
    top_dt_samples: int = 0


@dataclass(frozen=True)
class _BenchmarkSummary:
    batches: int
    patches: int
    elapsed_ms: float
    patches_per_second: float
    stage_ms_per_patch: dict[str, float]


@dataclass(frozen=True)
class _LoadedTrainingBatch:
    step: int
    raw_start_sample_index: int
    batch: FiberStrip2DBatch
    top_batch: FiberStrip2DBatch | None
    profile: dict[str, float]
    load_ms: float


class _PipelineLoaderProvider:
    def __init__(self, loader: FiberStrip2DLoader, *, isolated: bool) -> None:
        self.base_loader = loader
        self.isolated = bool(isolated)
        self.local = threading.local()
        self.lock = threading.Lock()
        self.clones: list[FiberStrip2DLoader] = []

    def get(self) -> FiberStrip2DLoader:
        if not self.isolated or not hasattr(self.base_loader, "clone"):
            return self.base_loader
        loader = getattr(self.local, "loader", None)
        if isinstance(loader, FiberStrip2DLoader):
            return loader
        loader = self.base_loader.clone()
        self.local.loader = loader
        with self.lock:
            self.clones.append(loader)
        return loader

    def close(self) -> None:
        with self.lock:
            clones = list(self.clones)
            self.clones.clear()
        for loader in clones:
            loader.close()


@dataclass(frozen=True)
class _PreparedTrainingBatch:
    loaded: _LoadedTrainingBatch
    images: torch.Tensor
    supervision: DirectionSupervision
    valid_np: np.ndarray
    prep_ms: float
    top_images: torch.Tensor | None = None
    top_supervision: DirectionSupervision | None = None
    top_dt_supervision: TopDistanceTransformSupervision | None = None
    top_valid_np: np.ndarray | None = None
    prep_gpu_ms: float = 0.0
    prep_wait_ms: float = 0.0
    prep_submit_ms: float = 0.0
    stream: torch.cuda.Stream | None = None
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    @property
    def batch(self) -> FiberStrip2DBatch:
        return self.loaded.batch

    @property
    def top_batch(self) -> FiberStrip2DBatch | None:
        return self.loaded.top_batch

    @property
    def load_ms(self) -> float:
        return self.loaded.load_ms

    @property
    def raw_start_sample_index(self) -> int:
        return self.loaded.raw_start_sample_index


def _prepare_loaded_training_batch(
    loaded: _LoadedTrainingBatch,
    *,
    device: torch.device,
    stream: torch.cuda.Stream | None = None,
    top_dt_radius_px: float = 30.0,
) -> _PreparedTrainingBatch:
    t0 = time.perf_counter()
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    def prepare_top() -> tuple[
        torch.Tensor | None,
        DirectionSupervision | None,
        TopDistanceTransformSupervision | None,
        np.ndarray | None,
    ]:
        if loaded.top_batch is None:
            return None, None, None, None
        top_images_raw, top_valid_t, top_valid_np = _batch_images_to_tensors(loaded.top_batch, device=device)
        top_images = _normalize_image_tensor(top_images_raw, top_valid_t)
        top_supervision = build_direction_supervision(loaded.top_batch.samples, top_valid_np, device=device)
        top_dt_supervision = build_top_distance_transform_supervision(
            loaded.top_batch.samples,
            top_valid_np,
            radius_px=top_dt_radius_px,
            device=device,
        )
        return top_images, top_supervision, top_dt_supervision, top_valid_np

    context = torch.cuda.stream(stream) if stream is not None else None
    if context is None:
        images_raw, valid_t, valid_np = _batch_images_to_tensors(loaded.batch, device=device)
        images = _normalize_image_tensor(images_raw, valid_t)
        supervision = build_direction_supervision(loaded.batch.samples, valid_np, device=device)
        top_images, top_supervision, top_dt_supervision, top_valid_np = prepare_top()
    else:
        with context:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            images_raw, valid_t, valid_np = _batch_images_to_tensors(loaded.batch, device=device)
            images = _normalize_image_tensor(images_raw, valid_t)
            supervision = build_direction_supervision(loaded.batch.samples, valid_np, device=device)
            top_images, top_supervision, top_dt_supervision, top_valid_np = prepare_top()
            end_event.record(stream)
    prep_ms = (time.perf_counter() - t0) * 1000.0
    if stream is None:
        _sync_device(device)
    return _PreparedTrainingBatch(
        loaded=loaded,
        images=images,
        supervision=supervision,
        valid_np=valid_np,
        top_images=top_images,
        top_supervision=top_supervision,
        top_dt_supervision=top_dt_supervision,
        top_valid_np=top_valid_np,
        prep_ms=prep_ms,
        stream=stream,
        start_event=start_event,
        end_event=end_event,
    )


def _wait_for_prepared_batch(prepared: _PreparedTrainingBatch, *, device: torch.device) -> _PreparedTrainingBatch:
    if prepared.end_event is None:
        return prepared
    wait_start = time.perf_counter()
    prepared.end_event.synchronize()
    wait_ms = (time.perf_counter() - wait_start) * 1000.0
    prep_gpu_ms = (
        prepared.start_event.elapsed_time(prepared.end_event)
        if prepared.start_event is not None
        else 0.0
    )
    if prepared.stream is not None and device.type == "cuda":
        torch.cuda.current_stream(device).wait_event(prepared.end_event)
    return replace(prepared, prep_wait_ms=wait_ms, prep_gpu_ms=float(prep_gpu_ms))


def _load_and_prepare_training_step(
    loader_provider: _PipelineLoaderProvider,
    training: FiberStripTrainingConfig,
    *,
    step: int,
    sample_mode: str,
    profile_enabled: bool,
    apply_image_augmentation: bool,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> _PreparedTrainingBatch:
    loaded = _load_training_batch(
        loader_provider.get(),
        training,
        step=step,
        sample_mode=sample_mode,
        profile_enabled=profile_enabled,
        apply_image_augmentation=apply_image_augmentation,
    )
    prepared = _prepare_loaded_training_batch(
        loaded,
        device=device,
        stream=stream,
        top_dt_radius_px=training.top_view_dt_radius_px,
    )
    return prepared


def _load_training_batch_from_provider(
    loader_provider: _PipelineLoaderProvider,
    training: FiberStripTrainingConfig,
    *,
    step: int,
    sample_mode: str,
    profile_enabled: bool,
    apply_image_augmentation: bool,
) -> _LoadedTrainingBatch:
    return _load_training_batch(
        loader_provider.get(),
        training,
        step=step,
        sample_mode=sample_mode,
        profile_enabled=profile_enabled,
        apply_image_augmentation=apply_image_augmentation,
    )


class _CudaPreparedBatchPipeline:
    def __init__(
        self,
        loader: FiberStrip2DLoader,
        training: FiberStripTrainingConfig,
        *,
        device: torch.device,
        sample_mode: str,
        start_step: int,
        max_step: int | None,
        profile_enabled: bool,
        apply_image_augmentation: bool,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("_CudaPreparedBatchPipeline requires a CUDA device")
        self.loader = loader
        self.loader_provider = _PipelineLoaderProvider(
            loader,
            isolated=bool(training.pipeline_isolated_loaders),
        )
        self.training = training
        self.device = device
        self.sample_mode = sample_mode
        self.max_step = None if max_step is None else int(max_step)
        self.profile_enabled = bool(profile_enabled)
        self.apply_image_augmentation = bool(apply_image_augmentation)
        self.depth = max(1, int(training.pipeline_depth))
        configured_workers = int(training.pipeline_workers)
        self.worker_count = max(1, configured_workers if configured_workers > 0 else self.depth)
        self.streams = [torch.cuda.Stream(device=device) for _ in range(self.worker_count)]
        self.executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="fiber-strip-train-load-prep",
        )
        self.pending: dict[int, Future[_PreparedTrainingBatch]] = {}
        self.next_submit_step = int(start_step)
        self.next_consume_step = int(start_step)
        self.closed = False
        self._submit_until_full()

    def _submit_until_full(self) -> float:
        submit_start = time.perf_counter()
        while not self.closed and len(self.pending) < self.depth:
            if self.max_step is not None and self.next_submit_step > self.max_step:
                break
            step = self.next_submit_step
            self.next_submit_step += 1
            stream = self.streams[(step - 1) % len(self.streams)]
            self.pending[step] = self.executor.submit(
                _load_and_prepare_training_step,
                self.loader_provider,
                self.training,
                step=step,
                sample_mode=self.sample_mode,
                profile_enabled=self.profile_enabled,
                apply_image_augmentation=self.apply_image_augmentation,
                device=self.device,
                stream=stream,
            )
        return (time.perf_counter() - submit_start) * 1000.0

    def next(self) -> _PreparedTrainingBatch:
        if self.closed:
            raise RuntimeError("training preparation pipeline is closed")
        if not self.pending:
            self._submit_until_full()
        step = self.next_consume_step
        future = self.pending.get(step)
        if future is None:
            self._submit_until_full()
            future = self.pending.get(step)
        if future is None:
            raise StopIteration
        prepared = future.result()
        del self.pending[step]
        self.next_consume_step += 1
        prepared = _wait_for_prepared_batch(prepared, device=self.device)
        submit_ms = self._submit_until_full()
        return replace(prepared, prep_submit_ms=submit_ms)

    def close(self) -> None:
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.loader_provider.close()
        self.pending.clear()


def _training_raw_start_sample_index(step: int, training: FiberStripTrainingConfig) -> int:
    if bool(training.contrastive_enabled):
        groups_per_step = max(
            1,
            int(training.train_control_points_per_step)
            // int(training.contrastive_control_points_per_fiber),
        )
        return (int(step) - 1) * groups_per_step
    return (int(step) - 1) * int(training.train_control_points_per_step)


def _load_training_batch(
    loader: FiberStrip2DLoader,
    training: FiberStripTrainingConfig,
    *,
    step: int,
    sample_mode: str,
    profile_enabled: bool,
    apply_image_augmentation: bool,
) -> _LoadedTrainingBatch:
    raw_start_sample_index = _training_raw_start_sample_index(step, training)
    loader_profile: dict[str, float] = {}
    t0 = time.perf_counter()
    if bool(training.contrastive_enabled):
        batch = loader.load_fiber_group_batch(
            raw_start_sample_index,
            batch_size=training.train_control_points_per_step,
            control_points_per_group=training.contrastive_control_points_per_fiber,
            sample_index_limit=training.max_sample_index,
            profile=loader_profile if profile_enabled else None,
            apply_image_augmentation=apply_image_augmentation,
            include_line_xy=True,
            include_coords=False,
        )
        top_batch = (
            loader.load_top_batch_for_batch(
                batch,
                profile=loader_profile if profile_enabled else None,
                include_line_xy=True,
                include_coords=False,
            )
            if bool(training.top_view_enabled)
            else None
        )
    else:
        batch = loader.load_batch(
            raw_start_sample_index,
            batch_size=training.train_control_points_per_step,
            sample_mode=sample_mode,
            sample_index_limit=training.max_sample_index,
            profile=loader_profile if profile_enabled else None,
            apply_image_augmentation=apply_image_augmentation,
            include_line_xy=True,
            include_coords=False,
        )
        top_batch = (
            loader.load_top_batch_for_batch(
                batch,
                profile=loader_profile if profile_enabled else None,
                include_line_xy=True,
                include_coords=False,
            )
            if bool(training.top_view_enabled)
            else None
        )
    load_ms = (time.perf_counter() - t0) * 1000.0
    return _LoadedTrainingBatch(
        step=int(step),
        raw_start_sample_index=raw_start_sample_index,
        batch=batch,
        top_batch=top_batch,
        profile=loader_profile,
        load_ms=load_ms,
    )


class _TrainingBatchPipeline:
    def __init__(
        self,
        loader: FiberStrip2DLoader,
        training: FiberStripTrainingConfig,
        *,
        sample_mode: str,
        start_step: int,
        max_step: int | None,
        profile_enabled: bool,
        apply_image_augmentation: bool,
    ) -> None:
        self.loader = loader
        self.loader_provider = _PipelineLoaderProvider(
            loader,
            isolated=bool(training.pipeline_isolated_loaders),
        )
        self.training = training
        self.sample_mode = sample_mode
        self.next_submit_step = int(start_step)
        self.next_consume_step = int(start_step)
        self.max_step = None if max_step is None else int(max_step)
        self.profile_enabled = bool(profile_enabled)
        self.apply_image_augmentation = bool(apply_image_augmentation)
        self.depth = max(1, int(training.pipeline_depth))
        configured_workers = int(training.pipeline_workers)
        self.worker_count = max(1, configured_workers if configured_workers > 0 else self.depth)
        self.executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="fiber-strip-train-load",
        )
        self.pending: dict[int, Future[_LoadedTrainingBatch]] = {}
        self.closed = False
        self._submit_until_full()

    def _can_submit(self) -> bool:
        return self.max_step is None or self.next_submit_step <= self.max_step

    def _submit_until_full(self) -> None:
        while not self.closed and len(self.pending) < self.depth and self._can_submit():
            step = self.next_submit_step
            self.next_submit_step += 1
            self.pending[step] = self.executor.submit(
                _load_training_batch_from_provider,
                self.loader_provider,
                self.training,
                step=step,
                sample_mode=self.sample_mode,
                profile_enabled=self.profile_enabled,
                apply_image_augmentation=self.apply_image_augmentation,
            )

    def next(self) -> tuple[_LoadedTrainingBatch, float]:
        if self.closed:
            raise RuntimeError("training batch pipeline is closed")
        step = self.next_consume_step
        future = self.pending.get(step)
        if future is None:
            self._submit_until_full()
            future = self.pending.get(step)
        if future is None:
            raise StopIteration
        wait_start = time.perf_counter()
        result = future.result()
        wait_ms = (time.perf_counter() - wait_start) * 1000.0
        del self.pending[step]
        self.next_consume_step += 1
        self._submit_until_full()
        return result, wait_ms

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self.pending.values():
            future.cancel()
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.loader_provider.close()
        self.pending.clear()

    def __enter__(self) -> "_TrainingBatchPipeline":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def _compute_batch_loss(
    model: FiberStripDirectionNet,
    batch: FiberStrip2DBatch,
    *,
    device: torch.device,
    training: FiberStripTrainingConfig | None = None,
    contrastive_negative_mask: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, DirectionSupervision, _DirectionMetrics]:
    images_np, valid_np = _flatten_batch(batch)
    images = _prepare_images(images_np, valid_np, device=device)
    supervision = build_direction_supervision(batch.samples, valid_np, device=device)
    outputs = model(images)
    direction = direction_output(outputs)
    direction_loss = direction_mse_loss(direction, supervision)
    loss = direction_loss
    presence_metrics = PresenceMetrics(0.0, 0.0, 0.0, 0, 0)
    presence_channels = _presence_channel_count(training)
    if training is not None and bool(training.presence_enabled) and float(training.presence_weight) > 0.0:
        presence, presence_metrics = presence_loss(
            outputs,
            supervision,
            batch.samples,
            valid_np,
            weight=training.presence_weight,
            negative_candidate_mask=contrastive_negative_mask,
            presence_channels=presence_channels,
        )
        loss = loss + presence
    contrastive_metrics = ContrastiveEmbeddingMetrics(0.0, 0.0, 0.0, 0, 0)
    if training is not None and bool(training.contrastive_enabled) and float(training.contrastive_weight) > 0.0:
        contrastive, contrastive_metrics = contrastive_embedding_loss(
            outputs,
            supervision,
            batch.samples,
            valid_np,
            weight=training.contrastive_weight,
            negative_margin=training.contrastive_negative_margin,
            negative_candidate_mask=contrastive_negative_mask,
            presence_channels=presence_channels,
        )
        loss = loss + contrastive
    angle_error = direction_angle_error_degrees(direction, supervision)
    metrics = _DirectionMetrics(
        angle_mean_deg=float(angle_error.detach().mean().cpu().item()),
        loss_direction=float(direction_loss.detach().cpu().item()),
        loss_presence=presence_metrics.loss,
        presence_positive_loss=presence_metrics.positive_loss,
        presence_negative_loss=presence_metrics.negative_loss,
        presence_positive_samples=presence_metrics.positive_samples,
        presence_negative_samples=presence_metrics.negative_samples,
        loss_contrastive=contrastive_metrics.loss,
        contrastive_positive_loss=contrastive_metrics.positive_loss,
        contrastive_negative_loss=contrastive_metrics.negative_loss,
        contrastive_pixel_negative_loss=contrastive_metrics.pixel_negative_loss,
        contrastive_positive_samples=contrastive_metrics.positive_samples,
        contrastive_negative_samples=contrastive_metrics.negative_samples,
        contrastive_pixel_negative_samples=contrastive_metrics.pixel_negative_samples,
        contrastive_similarity_mean_loss=contrastive_metrics.similarity_mean_loss,
        contrastive_similarity_mean_value=contrastive_metrics.similarity_mean_value,
        contrastive_similarity_mean_target=contrastive_metrics.similarity_mean_target,
        contrastive_similarity_mean_samples=contrastive_metrics.similarity_mean_samples,
    )
    return loss, outputs, supervision, metrics


def _compute_prepared_batch_loss(
    model: FiberStripDirectionNet,
    prepared: _PreparedTrainingBatch,
    training: FiberStripTrainingConfig | None = None,
    contrastive_negative_mask: np.ndarray | torch.Tensor | None = None,
    top_model: FiberStripDirectionNet | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, DirectionSupervision, _DirectionMetrics]:
    outputs = model(prepared.images)
    direction = direction_output(outputs)
    direction_loss = direction_mse_loss(direction, prepared.supervision)
    loss = direction_loss
    presence_metrics = PresenceMetrics(0.0, 0.0, 0.0, 0, 0)
    presence_channels = _presence_channel_count(training)
    if training is not None and bool(training.presence_enabled) and float(training.presence_weight) > 0.0:
        presence, presence_metrics = presence_loss(
            outputs,
            prepared.supervision,
            prepared.batch.samples,
            prepared.valid_np,
            weight=training.presence_weight,
            negative_candidate_mask=contrastive_negative_mask,
            presence_channels=presence_channels,
        )
        loss = loss + presence
    contrastive_metrics = ContrastiveEmbeddingMetrics(0.0, 0.0, 0.0, 0, 0)
    if training is not None and bool(training.contrastive_enabled) and float(training.contrastive_weight) > 0.0:
        contrastive, contrastive_metrics = contrastive_embedding_loss(
            outputs,
            prepared.supervision,
            prepared.batch.samples,
            prepared.valid_np,
            weight=training.contrastive_weight,
            negative_margin=training.contrastive_negative_margin,
            negative_candidate_mask=contrastive_negative_mask,
            presence_channels=presence_channels,
        )
        loss = loss + contrastive
    angle_error = direction_angle_error_degrees(direction, prepared.supervision)
    top_outputs: torch.Tensor | None = None
    top_loss_direction_value = 0.0
    top_loss_dt_value = 0.0
    top_angle_mean_deg = 0.0
    top_direction_samples = 0
    top_dt_samples = 0
    if training is not None and bool(training.top_view_enabled):
        if top_model is None:
            raise ValueError("top_view_enabled training requires a top_model")
        if prepared.top_images is None or prepared.top_supervision is None or prepared.top_dt_supervision is None:
            raise ValueError("top_view_enabled training requires prepared top-view tensors")
        top_outputs = top_model(prepared.top_images)
        top_direction = direction_output(top_outputs)
        top_direction_loss = direction_mse_loss(top_direction, prepared.top_supervision)
        top_dt_prediction = presence_output(top_outputs, presence_channels=1)
        top_dt_loss = top_distance_transform_mse_loss(top_dt_prediction, prepared.top_dt_supervision)
        if float(training.top_view_direction_weight) > 0.0:
            loss = loss + top_direction_loss * float(training.top_view_direction_weight)
        if float(training.top_view_dt_weight) > 0.0:
            loss = loss + top_dt_loss * float(training.top_view_dt_weight)
        top_angle = direction_angle_error_degrees(top_direction, prepared.top_supervision)
        top_loss_direction_value = float(top_direction_loss.detach().cpu().item())
        top_loss_dt_value = float(top_dt_loss.detach().cpu().item())
        top_angle_mean_deg = float(top_angle.detach().mean().cpu().item())
        top_direction_samples = int(prepared.top_supervision.target.shape[0])
        top_dt_samples = int(prepared.top_dt_supervision.target.shape[0])
    metrics = _DirectionMetrics(
        angle_mean_deg=float(angle_error.detach().mean().cpu().item()),
        loss_direction=float(direction_loss.detach().cpu().item()),
        loss_presence=presence_metrics.loss,
        presence_positive_loss=presence_metrics.positive_loss,
        presence_negative_loss=presence_metrics.negative_loss,
        presence_positive_samples=presence_metrics.positive_samples,
        presence_negative_samples=presence_metrics.negative_samples,
        loss_contrastive=contrastive_metrics.loss,
        contrastive_positive_loss=contrastive_metrics.positive_loss,
        contrastive_negative_loss=contrastive_metrics.negative_loss,
        contrastive_pixel_negative_loss=contrastive_metrics.pixel_negative_loss,
        contrastive_positive_samples=contrastive_metrics.positive_samples,
        contrastive_negative_samples=contrastive_metrics.negative_samples,
        contrastive_pixel_negative_samples=contrastive_metrics.pixel_negative_samples,
        contrastive_similarity_mean_loss=contrastive_metrics.similarity_mean_loss,
        contrastive_similarity_mean_value=contrastive_metrics.similarity_mean_value,
        contrastive_similarity_mean_target=contrastive_metrics.similarity_mean_target,
        contrastive_similarity_mean_samples=contrastive_metrics.similarity_mean_samples,
        top_loss_direction=top_loss_direction_value,
        top_loss_dt=top_loss_dt_value,
        top_angle_mean_deg=top_angle_mean_deg,
        top_direction_samples=top_direction_samples,
        top_dt_samples=top_dt_samples,
    )
    return loss, outputs, top_outputs, prepared.supervision, metrics


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stage_ms(profile: dict[str, float], key: str) -> float:
    return float(profile.get(key, 0.0))


def _benchmark_stage_totals(loader_profile: dict[str, float], train_profile: dict[str, float]) -> dict[str, float]:
    loader_wall = _stage_ms(loader_profile, "load_batch_wall")
    loader_worker = _stage_ms(loader_profile, "load_batch_worker")
    descriptor = _stage_ms(loader_profile, "random_order") + _stage_ms(loader_profile, "descriptor")
    compact_geometry = _stage_ms(loader_profile, "compact_geometry")
    source_geom = (
        _stage_ms(loader_profile, "line_window")
        + _stage_ms(loader_profile, "lasagna_normals")
        + _stage_ms(loader_profile, "strip_coords")
    )
    line = _stage_ms(loader_profile, "line_coords")
    coord_gen = (
        descriptor
        + compact_geometry
        + source_geom
        + line
    )
    return {
        "loader_wall": loader_wall,
        "loader_worker": loader_worker,
        "loader_thread_factor": loader_worker / loader_wall if loader_wall > 0.0 else 0.0,
        "pipeline_wait": _stage_ms(train_profile, "pipeline_wait"),
        "prep": _stage_ms(train_profile, "prep"),
        "prep_gpu": _stage_ms(train_profile, "prep_gpu"),
        "prep_wait": _stage_ms(train_profile, "prep_wait"),
        "prep_submit": _stage_ms(train_profile, "prep_submit"),
        "coord_gen": coord_gen,
        "descriptor": descriptor,
        "compact_geometry": compact_geometry,
        "source_geom": source_geom,
        "line": line,
        "coord_aug": _stage_ms(loader_profile, "coord_augmentation"),
        "loading": _stage_ms(loader_profile, "volume_sample"),
        "image_aug": _stage_ms(loader_profile, "value_augmentation") + _stage_ms(train_profile, "value_augmentation"),
        "fw": _stage_ms(train_profile, "fw"),
        "bw_step": _stage_ms(train_profile, "bw_step"),
        "outside_fw_bw": (
            loader_wall
            + _stage_ms(train_profile, "pipeline_wait")
            + _stage_ms(train_profile, "prep")
            + _stage_ms(train_profile, "prep_wait")
            + _stage_ms(train_profile, "prep_submit")
            + _stage_ms(loader_profile, "value_augmentation")
            + _stage_ms(train_profile, "value_augmentation")
        ),
    }


def _print_profile_header() -> None:
    print(
        "fiber_trace_2d profile columns: "
        "batch=batch-index patches=CNN image patches "
        "total/cpu/wall/work/wait/prep/prep_gpu/prep_wait/submit/outside/coord/desc/geom/source/line/coord_aug/load/img_aug/fw/bw_step=ms per patch "
        "ctf=process cpu-time/total-wall-time tf=loader worker-time/load-wall-time",
        flush=True,
    )
    print(
        f"{'batch':>5} {'patches':>7} {'total':>9} {'cpu':>9} {'ctf':>6} "
        f"{'wall':>9} {'work':>9} {'tf':>6} {'wait':>9} "
        f"{'prep':>9} {'prep_gpu':>9} {'prep_wait':>9} {'submit':>9} {'outside':>9} "
        f"{'coord':>9} {'desc':>9} "
        f"{'geom':>9} {'source':>9} {'line':>9} {'coord_aug':>9} "
        f"{'load':>9} {'img_aug':>9} {'fw':>9} {'bw_step':>9}",
        flush=True,
    )


def _print_profile_row(batch_index: int, patch_count: int, elapsed_ms: float, stages: dict[str, float]) -> None:
    denom = max(1, int(patch_count))
    print(
        f"{int(batch_index):5d} {int(patch_count):7d} "
        f"{elapsed_ms / denom:9.2f} "
        f"{stages.get('process_cpu', 0.0) / denom:9.2f} "
        f"{(stages.get('process_cpu', 0.0) / elapsed_ms if elapsed_ms > 0.0 else 0.0):6.2f} "
        f"{stages.get('loader_wall', 0.0) / denom:9.2f} "
        f"{stages.get('loader_worker', 0.0) / denom:9.2f} "
        f"{stages.get('loader_thread_factor', 0.0):6.2f} "
        f"{stages.get('pipeline_wait', 0.0) / denom:9.2f} "
        f"{stages.get('prep', 0.0) / denom:9.2f} "
        f"{stages.get('prep_gpu', 0.0) / denom:9.2f} "
        f"{stages.get('prep_wait', 0.0) / denom:9.2f} "
        f"{stages.get('prep_submit', 0.0) / denom:9.2f} "
        f"{stages.get('outside_fw_bw', 0.0) / denom:9.2f} "
        f"{stages.get('coord_gen', 0.0) / denom:9.2f} "
        f"{stages.get('descriptor', 0.0) / denom:9.2f} "
        f"{stages.get('compact_geometry', 0.0) / denom:9.2f} "
        f"{stages.get('source_geom', 0.0) / denom:9.2f} "
        f"{stages.get('line', 0.0) / denom:9.2f} "
        f"{stages.get('coord_aug', 0.0) / denom:9.2f} "
        f"{stages.get('loading', 0.0) / denom:9.2f} "
        f"{stages.get('image_aug', 0.0) / denom:9.2f} "
        f"{stages.get('fw', 0.0) / denom:9.2f} "
        f"{stages.get('bw_step', 0.0) / denom:9.2f}",
        flush=True,
    )


def _print_benchmark_summary(summary: _BenchmarkSummary, *, profile: bool) -> None:
    print(
        "fiber_trace_2d benchmark complete "
        f"batches={summary.batches} patches={summary.patches} "
        f"elapsed_ms={summary.elapsed_ms:.1f} patches_per_second={summary.patches_per_second:.2f}",
        flush=True,
    )
    if profile:
        print("fiber_trace_2d profile summary ms_per_patch:", flush=True)
        for key in (
            "process_cpu",
            "loader_wall",
            "loader_worker",
            "pipeline_wait",
            "prep",
            "prep_gpu",
            "prep_wait",
            "prep_submit",
            "outside_fw_bw",
            "coord_gen",
            "descriptor",
            "compact_geometry",
            "source_geom",
            "line",
            "coord_aug",
            "loading",
            "image_aug",
            "fw",
            "bw_step",
        ):
            print(f"  {key}={summary.stage_ms_per_patch.get(key, 0.0):.3f}", flush=True)
        wall = summary.stage_ms_per_patch.get("loader_wall", 0.0)
        worker = summary.stage_ms_per_patch.get("loader_worker", 0.0)
        factor = worker / wall if wall > 0.0 else 0.0
        print(f"  loader_thread_factor={factor:.3f}", flush=True)
        process_cpu_ms = summary.stage_ms_per_patch.get("process_cpu", 0.0) * max(1, int(summary.patches))
        process_factor = process_cpu_ms / summary.elapsed_ms if summary.elapsed_ms > 0.0 else 0.0
        print(f"  process_cpu_factor={process_factor:.3f}", flush=True)


def run_benchmark(
    config_path: str | Path,
    *,
    sampler_factory: SamplerFactory | None = None,
    batches: int = 100,
    profile: bool = False,
    load_only: bool = False,
) -> _BenchmarkSummary:
    if int(batches) <= 0:
        raise ValueError("benchmark batches must be > 0")
    raw_config = _load_raw_config(config_path)
    training = _training_config_from_raw(raw_config)
    loader_config = load_config(config_path)
    _validate_training_batch_config(training, loader_config)
    contrastive_negative_mask = (
        _contrastive_negative_candidate_mask(loader_config)
        if (
            (bool(training.contrastive_enabled) and float(training.contrastive_weight) > 0.0)
            or (bool(training.presence_enabled) and float(training.presence_weight) > 0.0)
        )
        else None
    )
    loader = FiberStrip2DLoader(loader_config, sampler_factory=sampler_factory)
    device = resolve_torch_device(training.device)
    model: FiberStripDirectionNet | None = None
    top_model: FiberStripDirectionNet | None = None
    optimizer: torch.optim.Optimizer | None = None
    if not load_only:
        model = FiberStripDirectionNet(
            FiberStripDirectionModelConfig(
                in_channels=1,
                hidden_channels=training.model_hidden_channels,
                depth=training.model_depth,
                presence_channels=_presence_channel_count(training),
                embedding_channels=_embedding_channel_count(training),
            )
        ).to(device)
        parameters: list[torch.nn.Parameter] = list(model.parameters())
        if bool(training.top_view_enabled):
            top_model = FiberStripDirectionNet(
                FiberStripDirectionModelConfig(
                    in_channels=1,
                    hidden_channels=training.model_hidden_channels,
                    depth=training.model_depth,
                    presence_channels=_top_view_scalar_channel_count(training),
                    embedding_channels=0,
                )
            ).to(device)
            parameters.extend(top_model.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=training.learning_rate)
        model.train()
        if top_model is not None:
            top_model.train()

    sample_mode = "random"
    stage_totals = {
        key: 0.0
        for key in (
            "loader_wall",
            "loader_worker",
            "loader_thread_factor",
            "process_cpu",
            "pipeline_wait",
            "prep",
            "prep_gpu",
            "prep_wait",
            "prep_submit",
            "outside_fw_bw",
            "coord_gen",
            "descriptor",
            "compact_geometry",
            "source_geom",
            "line",
            "coord_aug",
            "loading",
            "image_aug",
            "fw",
            "bw_step",
        )
    }
    total_patches = 0
    total_batches = int(batches)
    if profile:
        _print_profile_header()

    _sync_device(device)
    wall_start = time.perf_counter()
    use_prep_pipeline = bool(training.pipeline_enabled and not load_only and device.type == "cuda")
    use_load_pipeline = bool(training.pipeline_enabled and load_only)
    pipeline: _TrainingBatchPipeline | None = None
    prep_pipeline: _CudaPreparedBatchPipeline | None = None
    if use_prep_pipeline:
        prep_pipeline = _CudaPreparedBatchPipeline(
            loader,
            training,
            device=device,
            sample_mode=sample_mode,
            start_step=1,
            max_step=total_batches,
            profile_enabled=profile,
            apply_image_augmentation=False,
        )
    elif use_load_pipeline:
        pipeline = _TrainingBatchPipeline(
            loader,
            training,
            sample_mode=sample_mode,
            start_step=1,
            max_step=total_batches,
            profile_enabled=profile,
            apply_image_augmentation=False,
        )
    try:
        for batch_index in range(1, total_batches + 1):
            batch_start = time.perf_counter()
            batch_cpu_start = time.process_time()
            train_profile: dict[str, float] = {}
            prepared: _PreparedTrainingBatch | None = None
            if prep_pipeline is not None:
                prepared = prep_pipeline.next()
                loaded = prepared.loaded
                train_profile["pipeline_wait"] = loaded.profile.get("pipeline_wait", 0.0)
                train_profile["prep"] = prepared.prep_ms
                train_profile["prep_gpu"] = prepared.prep_gpu_ms
                train_profile["prep_wait"] = prepared.prep_wait_ms
                train_profile["prep_submit"] = prepared.prep_submit_ms
            elif pipeline is None:
                loaded = _load_training_batch(
                    loader,
                    training,
                    step=batch_index,
                    sample_mode=sample_mode,
                    profile_enabled=profile,
                    apply_image_augmentation=False if not load_only else False,
                )
            else:
                loaded, wait_ms = pipeline.next()
                train_profile["pipeline_wait"] = wait_ms
            loader_profile = loaded.profile
            batch = loaded.batch
            images_np, valid_np = _flatten_batch(batch)
            patch_count = int(images_np.shape[0])
            if not load_only:
                assert model is not None
                assert optimizer is not None
                if prepared is None:
                    prep_start = time.perf_counter()
                    prepared = _prepare_loaded_training_batch(
                        loaded,
                        device=device,
                        top_dt_radius_px=training.top_view_dt_radius_px,
                    )
                    train_profile["prep"] = (time.perf_counter() - prep_start) * 1000.0
                    train_profile["prep_gpu"] = prepared.prep_gpu_ms
                    train_profile["prep_wait"] = prepared.prep_wait_ms

                optimizer.zero_grad(set_to_none=True)
                _sync_device(device)
                fw_start = time.perf_counter()
                outputs = model(prepared.images)
                loss = direction_mse_loss(direction_output(outputs), prepared.supervision)
                presence_channels = _presence_channel_count(training)
                if bool(training.presence_enabled) and float(training.presence_weight) > 0.0:
                    presence, _ = presence_loss(
                        outputs,
                        prepared.supervision,
                        prepared.batch.samples,
                        prepared.valid_np,
                        weight=training.presence_weight,
                        negative_candidate_mask=contrastive_negative_mask,
                        presence_channels=presence_channels,
                    )
                    loss = loss + presence
                if bool(training.contrastive_enabled) and float(training.contrastive_weight) > 0.0:
                    contrastive, _ = contrastive_embedding_loss(
                        outputs,
                        prepared.supervision,
                        prepared.batch.samples,
                        prepared.valid_np,
                        weight=training.contrastive_weight,
                        negative_margin=training.contrastive_negative_margin,
                        negative_candidate_mask=contrastive_negative_mask,
                        presence_channels=presence_channels,
                    )
                    loss = loss + contrastive
                if bool(training.top_view_enabled):
                    if top_model is None:
                        raise ValueError("top_view_enabled benchmark requires a top model")
                    if prepared.top_images is None or prepared.top_supervision is None or prepared.top_dt_supervision is None:
                        raise ValueError("top_view_enabled benchmark requires prepared top-view tensors")
                    top_outputs = top_model(prepared.top_images)
                    top_direction_loss = direction_mse_loss(
                        direction_output(top_outputs),
                        prepared.top_supervision,
                    )
                    top_dt_loss = top_distance_transform_mse_loss(
                        presence_output(top_outputs, presence_channels=1),
                        prepared.top_dt_supervision,
                    )
                    loss = (
                        loss
                        + top_direction_loss * float(training.top_view_direction_weight)
                        + top_dt_loss * float(training.top_view_dt_weight)
                    )
                _sync_device(device)
                train_profile["fw"] = (time.perf_counter() - fw_start) * 1000.0

                bw_start = time.perf_counter()
                loss.backward()
                optimizer.step()
                _sync_device(device)
                train_profile["bw_step"] = (time.perf_counter() - bw_start) * 1000.0

            batch_elapsed_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_process_cpu_ms = (time.process_time() - batch_cpu_start) * 1000.0
            stages = _benchmark_stage_totals(loader_profile, train_profile)
            stages["process_cpu"] = batch_process_cpu_ms
            for key, value in stages.items():
                stage_totals[key] += float(value)
            total_patches += patch_count
            if profile:
                _print_profile_row(batch_index, patch_count, batch_elapsed_ms, stages)
    finally:
        if prep_pipeline is not None:
            prep_pipeline.close()
        elif pipeline is not None:
            pipeline.close()

    _sync_device(device)
    elapsed_ms = (time.perf_counter() - wall_start) * 1000.0
    patches_per_second = float(total_patches) / max(elapsed_ms / 1000.0, 1.0e-9)
    stage_ms_per_patch = {
        key: float(value) / max(1, int(total_patches))
        for key, value in stage_totals.items()
    }
    summary = _BenchmarkSummary(
        batches=total_batches,
        patches=total_patches,
        elapsed_ms=elapsed_ms,
        patches_per_second=patches_per_second,
        stage_ms_per_patch=stage_ms_per_patch,
    )
    _print_benchmark_summary(summary, profile=profile)
    return summary


@dataclass(frozen=True)
class _EvalResult:
    loss: float
    angle_mean_deg: float
    supervision_samples: int
    batch: FiberStrip2DBatch
    outputs: torch.Tensor
    top_batch: FiberStrip2DBatch | None = None
    top_outputs: torch.Tensor | None = None
    top_loss_direction: float = 0.0
    top_loss_dt: float = 0.0
    top_angle_mean_deg: float = 0.0
    top_direction_samples: int = 0
    top_dt_samples: int = 0


@dataclass(frozen=True)
class _Trace2CpMetricEvalResult:
    error_mean: float
    raw_y_error_mean_px: float
    segments: int
    skipped_segments: int
    first_skip_reason: str


@dataclass(frozen=True)
class _ResolvedTestSelection:
    start_sample_index: int
    sample_count: int
    sample_mode: str


def _resolve_test_selection(
    training: FiberStripTrainingConfig,
    loader: FiberStrip2DLoader,
) -> _ResolvedTestSelection:
    if int(training.test_control_points) == 0:
        return _ResolvedTestSelection(
            start_sample_index=0,
            sample_count=int(loader.sample_count),
            sample_mode="flat",
        )
    return _ResolvedTestSelection(
        start_sample_index=int(training.test_start_sample_index),
        sample_count=int(training.test_control_points),
        sample_mode="random",
    )


def _evaluate_fixed_batch(
    model: FiberStripDirectionNet,
    loader: FiberStrip2DLoader,
    *,
    device: torch.device,
    start_sample_index: int,
    batch_size: int,
    sample_mode: str,
    training: FiberStripTrainingConfig | None = None,
    top_model: FiberStripDirectionNet | None = None,
) -> _EvalResult:
    was_training = model.training
    top_was_training = False if top_model is None else top_model.training
    model.eval()
    if top_model is not None:
        top_model.eval()
    with torch.no_grad():
        batch = loader.load_batch(
            start_sample_index,
            batch_size=batch_size,
            sample_mode=sample_mode,
        )
        loss, outputs, supervision, metrics = _compute_batch_loss(model, batch, device=device)
        loss_value = float(loss.detach().cpu().item())
        outputs_cpu = outputs.detach().cpu()
        top_batch = None
        top_outputs_cpu = None
        top_loss_direction = 0.0
        top_loss_dt = 0.0
        top_angle_mean_deg = 0.0
        top_direction_samples = 0
        top_dt_samples = 0
        if training is not None and bool(training.top_view_enabled):
            if top_model is None:
                raise ValueError("top_view_enabled test evaluation requires a top model")
            top_batch = loader.load_top_batch_for_batch(
                batch,
                include_line_xy=True,
                include_coords=False,
            )
            top_images_np, top_valid_np = _flatten_batch(top_batch)
            top_images = _prepare_images(top_images_np, top_valid_np, device=device)
            top_supervision = build_direction_supervision(top_batch.samples, top_valid_np, device=device)
            top_dt_supervision = build_top_distance_transform_supervision(
                top_batch.samples,
                top_valid_np,
                radius_px=training.top_view_dt_radius_px,
                device=device,
            )
            top_outputs = top_model(top_images)
            top_direction_loss_t = direction_mse_loss(direction_output(top_outputs), top_supervision)
            top_dt_loss_t = top_distance_transform_mse_loss(
                presence_output(top_outputs, presence_channels=1),
                top_dt_supervision,
            )
            top_angle = direction_angle_error_degrees(direction_output(top_outputs), top_supervision)
            top_outputs_cpu = top_outputs.detach().cpu()
            top_loss_direction = float(top_direction_loss_t.detach().cpu().item())
            top_loss_dt = float(top_dt_loss_t.detach().cpu().item())
            top_angle_mean_deg = float(top_angle.detach().mean().cpu().item())
            top_direction_samples = int(top_supervision.target.shape[0])
            top_dt_samples = int(top_dt_supervision.target.shape[0])
    if was_training:
        model.train()
    if top_model is not None and top_was_training:
        top_model.train()
    return _EvalResult(
        loss=loss_value,
        angle_mean_deg=metrics.angle_mean_deg,
        supervision_samples=int(supervision.target.shape[0]),
        batch=batch,
        outputs=outputs_cpu,
        top_batch=top_batch,
        top_outputs=top_outputs_cpu,
        top_loss_direction=top_loss_direction,
        top_loss_dt=top_loss_dt,
        top_angle_mean_deg=top_angle_mean_deg,
        top_direction_samples=top_direction_samples,
        top_dt_samples=top_dt_samples,
    )


def _evaluate_trace2cp_metric_fixed_set(
    model: FiberStripDirectionNet,
    loader: FiberStrip2DLoader,
    *,
    device: torch.device,
    start_sample_index: int,
    sample_count: int,
    step_px: float,
    rf_margin_px: float,
    sample_mode: str,
) -> _Trace2CpMetricEvalResult:
    was_training = model.training
    model.eval()
    errors: list[float] = []
    raw_errors: list[float] = []
    skipped = 0
    first_skip = ""
    with torch.no_grad():
        for offset in range(max(1, int(sample_count))):
            sample_index = int(start_sample_index) + int(offset)
            try:
                sample, image, valid_mask = loader.build_trace2cp_segment_patch(
                    sample_index,
                    target_offset=1,
                    rf_margin_px=rf_margin_px,
                    device=device,
                    sample_mode=sample_mode,
                )
                direction_xy = _predict_direction_field(model, image, valid_mask, device=device)
                metric = _trace2cp_metric_bidirectional(
                    direction_xy,
                    np.asarray(sample.start_control_point_xy, dtype=np.float32),
                    np.asarray(sample.target_control_point_xy, dtype=np.float32),
                    valid_mask=valid_mask,
                    step_px=step_px,
                    rf_margin_px=rf_margin_px,
                )
            except ValueError as exc:
                skipped += 1
                if not first_skip:
                    first_skip = " ".join(str(exc).split())
                continue
            errors.append(float(metric.error))
            raw_errors.append(float(metric.raw_y_error_px))
    if was_training:
        model.train()
    if not errors:
        raise ValueError(
            "test Trace2CP metric found no valid CP-to-next-CP segments: "
            f"start_sample_index={int(start_sample_index)} sample_count={int(sample_count)} "
            f"skipped={int(skipped)} first_skip='{first_skip}'"
        )
    return _Trace2CpMetricEvalResult(
        error_mean=float(np.mean(np.asarray(errors, dtype=np.float64))),
        raw_y_error_mean_px=float(np.mean(np.asarray(raw_errors, dtype=np.float64))),
        segments=len(errors),
        skipped_segments=int(skipped),
        first_skip_reason=first_skip,
    )


def _cache_scalars(cache_stats: Any | None) -> dict[str, float]:
    if cache_stats is None:
        return {}
    return {
        "cache/hits": float(getattr(cache_stats, "cache_hits", 0)),
        "cache/downloads": float(getattr(cache_stats, "downloads", 0)),
        "cache/misses": float(getattr(cache_stats, "missing", 0)) + float(getattr(cache_stats, "negative_hits", 0)),
        "cache/hit_mib": float(getattr(cache_stats, "cache_hit_bytes", 0)) / (1024.0 * 1024.0),
        "cache/download_mib": float(getattr(cache_stats, "download_bytes", 0)) / (1024.0 * 1024.0),
        "cache/hit_ms": float(getattr(cache_stats, "cache_hit_ms", 0.0)),
        "cache/download_ms": float(getattr(cache_stats, "download_ms", 0.0)),
    }


def _merge_prefetch_summaries(*summaries: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for summary in summaries:
        for key, value in summary.items():
            if isinstance(value, bool):
                merged[key] = value
            elif isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged.setdefault(key, value)
    return merged


def _should_print_training_step(
    step: int,
    *,
    scalar_log_interval: int,
    start_sample_index: int,
    sample_count: int,
    startup_sample_print_count: int = 100,
) -> bool:
    if int(step) <= 1:
        return True
    sample_end = int(start_sample_index) + max(1, int(sample_count))
    if int(start_sample_index) < int(startup_sample_print_count) and sample_end > 0:
        return True
    return int(step) % max(1, int(scalar_log_interval)) == 0


def _to_u8_image(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if bool(valid.any()):
        values = arr[valid]
        lo = float(values.min())
        hi = float(values.max())
        scale = 255.0 / max(hi - lo, 1.0e-6)
        out[valid] = np.clip((arr[valid] - lo) * scale, 0.0, 255.0).astype(np.uint8)
    return out


def _select_visualization_patch_indices(batch: FiberStrip2DBatch, *, max_patches: int) -> list[int]:
    images = np.asarray(batch.images)
    if images.ndim != 5 or images.shape[2] != 1:
        raise ValueError("batch.images must have shape B,Z,1,H,W")
    control_points = int(images.shape[0])
    offsets = int(images.shape[1])
    if control_points <= 0 or offsets <= 0 or max_patches <= 0:
        return []
    strip_offsets = np.asarray(batch.strip_z_offsets, dtype=np.float32).reshape(-1)
    center_offset_index = (
        int(np.argmin(np.abs(strip_offsets[:offsets])))
        if int(strip_offsets.shape[0]) >= offsets
        else offsets // 2
    )
    selected: list[int] = []
    used: set[int] = set()

    for cp_index in range(control_points):
        patch_index = cp_index * offsets + center_offset_index
        selected.append(patch_index)
        used.add(patch_index)
        if len(selected) >= max_patches:
            return selected

    for patch_index in range(control_points * offsets):
        if patch_index in used:
            continue
        selected.append(patch_index)
        if len(selected) >= max_patches:
            break
    return selected


def _draw_predicted_cp_direction(
    rgb: np.ndarray,
    *,
    cp_xy: np.ndarray,
    prediction_xy: np.ndarray | None,
) -> np.ndarray:
    from PIL import Image, ImageDraw

    pil = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")
    x0, y0 = float(cp_xy[0]), float(cp_xy[1])
    if prediction_xy is not None and np.isfinite(prediction_xy).all():
        direction = np.asarray(prediction_xy, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 1.0e-6:
            direction = direction / norm
            length = 10.0
            draw.line(
                (
                    x0,
                    y0,
                    x0 + direction[0] * length,
                    y0 + direction[1] * length,
                ),
                fill=(0, 255, 0, 240),
                width=1,
            )
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"), dtype=np.uint8)


def _make_training_visualization(
    batch: FiberStrip2DBatch,
    outputs: torch.Tensor,
    *,
    max_patches: int = 8,
) -> np.ndarray:
    flat_images, flat_valid = _flatten_batch(batch)
    outputs_cpu = outputs.detach().cpu()
    cells: list[np.ndarray] = []
    for patch_index in _select_visualization_patch_indices(batch, max_patches=max_patches):
        if (
            patch_index >= int(flat_images.shape[0])
            or patch_index >= int(outputs_cpu.shape[0])
            or patch_index >= len(batch.samples)
        ):
            continue
        sample = batch.samples[patch_index]
        image_u8 = _to_u8_image(flat_images[patch_index, 0], flat_valid[patch_index])
        rgb = overlay_line_coords_rgb(image_u8, sample.line_xy, opacity=0.5, thickness=1)
        cp_tangent = line_cp_and_tangent_xy(sample.line_xy, getattr(sample, "control_point_xy", None))
        prediction_xy = None
        if cp_tangent is not None:
            cp_xy, _ = cp_tangent
            center = np.rint(cp_xy).astype(np.int64)
            y = int(np.clip(center[1], 0, flat_valid.shape[1] - 1))
            x = int(np.clip(center[0], 0, flat_valid.shape[2] - 1))
            encoded = outputs_cpu[patch_index, :2, y, x]
            prediction_xy = decode_lasagna_direction_xy(encoded).cpu().numpy()
            rgb = _draw_predicted_cp_direction(
                rgb,
                cp_xy=cp_xy,
                prediction_xy=prediction_xy,
            )
        cells.append(rgb)
    if not cells:
        return np.zeros((3, 1, 1), dtype=np.uint8)
    h, w = cells[0].shape[:2]
    cols = min(4, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = cell
    return np.transpose(sheet, (2, 0, 1))


def _similarity_to_u8(similarity: np.ndarray) -> np.ndarray:
    arr = np.asarray(similarity, dtype=np.float32)
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def _make_embedding_similarity_visualization(
    batch: FiberStrip2DBatch,
    outputs: torch.Tensor,
    *,
    max_patches: int = 8,
    presence_channels: int = 0,
) -> np.ndarray:
    if int(outputs.shape[1]) <= 2 + int(presence_channels):
        return np.zeros((3, 1, 1), dtype=np.uint8)
    _, flat_valid = _flatten_batch(batch)
    cells: list[np.ndarray] = []
    for patch_index in _select_visualization_patch_indices(batch, max_patches=max_patches):
        if patch_index >= int(outputs.shape[0]) or patch_index >= len(batch.samples):
            continue
        sample = batch.samples[patch_index]
        cp_tangent = line_cp_and_tangent_xy(sample.line_xy, getattr(sample, "control_point_xy", None))
        if cp_tangent is None:
            continue
        cp_xy, _ = cp_tangent
        similarity = embedding_similarity_to_cp(
            outputs,
            patch_index=patch_index,
            cp_xy=cp_xy,
            valid_mask=flat_valid[patch_index],
            presence_channels=presence_channels,
        )
        image_u8 = _similarity_to_u8(similarity)
        rgb = np.repeat(image_u8[:, :, None], 3, axis=2)
        rgb = _draw_predicted_cp_direction(rgb, cp_xy=cp_xy, prediction_xy=None)
        cells.append(rgb)
    if not cells:
        return np.zeros((3, 1, 1), dtype=np.uint8)
    h, w = cells[0].shape[:2]
    cols = min(4, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = cell
    return np.transpose(sheet, (2, 0, 1))


def _make_presence_visualization(
    batch: FiberStrip2DBatch,
    outputs: torch.Tensor,
    *,
    max_patches: int = 8,
    presence_channels: int = 1,
) -> np.ndarray:
    if int(presence_channels) <= 0 or int(outputs.shape[1]) < 3:
        return np.zeros((3, 1, 1), dtype=np.uint8)
    _, flat_valid = _flatten_batch(batch)
    presence = outputs.detach().cpu()[:, 2, :, :].numpy().astype(np.float32)
    cells: list[np.ndarray] = []
    for patch_index in _select_visualization_patch_indices(batch, max_patches=max_patches):
        if patch_index >= int(presence.shape[0]) or patch_index >= len(batch.samples):
            continue
        values = np.where(flat_valid[patch_index], presence[patch_index], 0.0)
        image_u8 = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
        rgb = np.repeat(image_u8[:, :, None], 3, axis=2)
        sample = batch.samples[patch_index]
        cp_tangent = line_cp_and_tangent_xy(sample.line_xy, getattr(sample, "control_point_xy", None))
        if cp_tangent is not None:
            cp_xy, _ = cp_tangent
            rgb = _draw_predicted_cp_direction(rgb, cp_xy=cp_xy, prediction_xy=None)
        cells.append(rgb)
    if not cells:
        return np.zeros((3, 1, 1), dtype=np.uint8)
    h, w = cells[0].shape[:2]
    cols = min(4, len(cells))
    rows = int(math.ceil(len(cells) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, cell in enumerate(cells):
        row = i // cols
        col = i % cols
        sheet[row * h : (row + 1) * h, col * w : (col + 1) * w] = cell
    return np.transpose(sheet, (2, 0, 1))


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: FiberStripDirectionNet,
    top_model: FiberStripDirectionNet | None = None,
    optimizer: torch.optim.Optimizer,
    loss: float,
    raw_config: dict[str, Any],
    metric_name: str = "loss",
) -> None:
    payload = {
        "step": int(step),
        "loss": float(loss),
        "metric_name": str(metric_name),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": _json_safe(raw_config),
    }
    if top_model is not None:
        payload["top_model_state_dict"] = top_model.state_dict()
    torch.save(
        payload,
        path,
    )


def _torch_load_checkpoint(path: Path, *, map_location: torch.device | str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _resolve_resume_checkpoint(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"--resume checkpoint does not exist: {resolved}")
    return resolved


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _load_training_resume_checkpoint(
    path: Path,
    *,
    model: FiberStripDirectionNet,
    top_model: FiberStripDirectionNet | None = None,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    checkpoint = _torch_load_checkpoint(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path} is not a fiber_trace_2d training checkpoint")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is missing model_state_dict")
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError(f"{path} is missing optimizer_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"])
    if top_model is not None:
        if "top_model_state_dict" not in checkpoint:
            raise ValueError(f"{path} is missing top_model_state_dict required by top_view_enabled training")
        top_model.load_state_dict(checkpoint["top_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    _move_optimizer_state_to_device(optimizer, device)
    try:
        step = int(checkpoint.get("step", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} has an invalid checkpoint step") from exc
    if step < 0:
        raise ValueError(f"{path} has an invalid negative checkpoint step: {step}")
    return checkpoint, step


def run_training(
    config_path: str | Path,
    *,
    sampler_factory: SamplerFactory | None = None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    raw_config = _load_raw_config(config_path)
    training = _training_config_from_raw(raw_config)
    resume_path = _resolve_resume_checkpoint(resume_checkpoint)
    loader_config = load_config(config_path)
    _validate_training_batch_config(training, loader_config)
    contrastive_negative_mask = (
        _contrastive_negative_candidate_mask(loader_config)
        if (
            (bool(training.contrastive_enabled) and float(training.contrastive_weight) > 0.0)
            or (bool(training.presence_enabled) and float(training.presence_weight) > 0.0)
        )
        else None
    )
    loader = FiberStrip2DLoader(loader_config, sampler_factory=sampler_factory)
    test_loader_config = _test_loader_config_from_raw(raw_config, loader_config)
    test_loader = (
        None
        if test_loader_config is None
        else FiberStrip2DLoader(test_loader_config, sampler_factory=sampler_factory)
    )
    device = resolve_torch_device(training.device)
    model = FiberStripDirectionNet(
        FiberStripDirectionModelConfig(
            in_channels=1,
            hidden_channels=training.model_hidden_channels,
            depth=training.model_depth,
            presence_channels=_presence_channel_count(training),
            embedding_channels=_embedding_channel_count(training),
        )
    ).to(device)
    top_model = (
        FiberStripDirectionNet(
            FiberStripDirectionModelConfig(
                in_channels=1,
                hidden_channels=training.model_hidden_channels,
                depth=training.model_depth,
                presence_channels=_top_view_scalar_channel_count(training),
                embedding_channels=0,
            )
        ).to(device)
        if bool(training.top_view_enabled)
        else None
    )
    optimizer_parameters: list[torch.nn.Parameter] = list(model.parameters())
    if top_model is not None:
        optimizer_parameters.extend(top_model.parameters())
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=training.learning_rate)

    resume_step = 0
    run_dir = _make_run_dir(training)
    resume_metric_name = ""
    resume_metric_value = float("nan")
    if resume_path is not None:
        resume_checkpoint_data, resume_step = _load_training_resume_checkpoint(
            resume_path,
            model=model,
            top_model=top_model,
            optimizer=optimizer,
            device=device,
        )
        resume_metric_name = str(resume_checkpoint_data.get("metric_name", "loss"))
        try:
            resume_metric_value = float(resume_checkpoint_data.get("loss", float("nan")))
        except (TypeError, ValueError):
            resume_metric_value = float("nan")
    start_step = int(resume_step) + 1
    snapshots = run_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    writer = _make_summary_writer(run_dir, enabled=training.tensorboard_enabled)
    if writer is not None:
        writer.add_text("config/json", json.dumps(_json_safe(raw_config), indent=2, sort_keys=True), 0)
        writer.flush()
    print(f"fiber_trace_2d train run_dir={run_dir}", flush=True)
    print(
        "fiber_trace_2d train pipeline "
        f"enabled={bool(training.pipeline_enabled and device.type == 'cuda')} "
        f"depth={int(training.pipeline_depth)} workers={int(training.pipeline_workers)} "
        f"isolated_loaders={bool(training.pipeline_isolated_loaders)} "
        f"loader_workers={int(loader_config.loader_workers)} "
        f"volume_cache_memory_mib={loader_config.volume_cache_memory_mib}",
        flush=True,
    )
    if resume_path is not None:
        print(
            "fiber_trace_2d train resume "
            f"checkpoint={resume_path} checkpoint_step={resume_step} "
            f"next_step={resume_step + 1} metric_name={resume_metric_name} "
            f"metric_value={resume_metric_value:.8f} run_dir={run_dir}",
            flush=True,
        )

    sample_mode = "random"
    finite_steps = training.max_steps > 0
    best_metric = float("inf")
    last_loss = float("nan")
    last_angle_mean_deg = float("nan")
    last_test_loss = float("nan")
    last_test_angle_mean_deg = float("nan")
    last_test_trace2cp_error = float("nan")
    use_pipeline = bool(training.pipeline_enabled and device.type == "cuda")
    pipeline: _TrainingBatchPipeline | None = None
    prep_pipeline: _CudaPreparedBatchPipeline | None = None
    try:
        finite_max_step = int(training.max_steps) if finite_steps else None
        if use_pipeline:
            prep_pipeline = _CudaPreparedBatchPipeline(
                loader,
                training,
                device=device,
                sample_mode=sample_mode,
                start_step=start_step,
                max_step=finite_max_step,
                profile_enabled=False,
                apply_image_augmentation=False,
            )
        step = start_step
        while True:
            if finite_steps and step > training.max_steps:
                break
            step_wall_start = time.perf_counter()
            prep_ms = 0.0
            prep_gpu_ms = 0.0
            prep_wait_ms = 0.0
            prep_submit_ms = 0.0
            train_ms = 0.0
            if prep_pipeline is not None:
                prepared = prep_pipeline.next()
                loaded = prepared.loaded
                wait_ms = float(loaded.profile.get("pipeline_wait", 0.0))
                prep_ms = float(prepared.prep_ms)
                prep_gpu_ms = float(prepared.prep_gpu_ms)
                prep_wait_ms = float(prepared.prep_wait_ms)
                prep_submit_ms = float(prepared.prep_submit_ms)
            elif pipeline is None:
                loaded = _load_training_batch(
                    loader,
                    training,
                    step=step,
                    sample_mode=sample_mode,
                    profile_enabled=False,
                    apply_image_augmentation=False,
                )
                wait_ms = 0.0
                prepared = _prepare_loaded_training_batch(
                    loaded,
                    device=device,
                    top_dt_radius_px=training.top_view_dt_radius_px,
                )
                prep_ms = float(prepared.prep_ms)
            else:
                loaded, wait_ms = pipeline.next()
                prepared = _prepare_loaded_training_batch(
                    loaded,
                    device=device,
                    top_dt_radius_px=training.top_view_dt_radius_px,
                )
                prep_ms = float(prepared.prep_ms)
            raw_start_sample_index = loaded.raw_start_sample_index
            batch = loaded.batch
            load_ms = float(loaded.load_ms)

            model.train()
            if top_model is not None:
                top_model.train()
            train_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss, outputs, top_outputs, supervision, metrics = _compute_prepared_batch_loss(
                model,
                prepared,
                training,
                contrastive_negative_mask=contrastive_negative_mask,
                top_model=top_model,
            )
            loss.backward()
            optimizer.step()
            _sync_device(device)
            train_ms = (time.perf_counter() - train_start) * 1000.0
            loss_value = float(loss.detach().cpu().item())
            last_loss = loss_value
            last_angle_mean_deg = metrics.angle_mean_deg
            should_test = (
                test_loader is not None
                and (step == 1 or step % training.test_interval == 0 or (finite_steps and step == training.max_steps))
            )
            test_result = None
            test_trace2cp_metric = None
            if should_test:
                test_selection = _resolve_test_selection(training, test_loader)
                test_result = _evaluate_fixed_batch(
                    model,
                    test_loader,
                    device=device,
                    start_sample_index=test_selection.start_sample_index,
                    batch_size=test_selection.sample_count,
                    sample_mode=test_selection.sample_mode,
                    training=training,
                    top_model=top_model,
                )
                last_test_loss = test_result.loss
                last_test_angle_mean_deg = test_result.angle_mean_deg
                trace2cp_margin = (
                    float(training.model_depth)
                    if training.test_trace2cp_rf_margin_px is None
                    else float(training.test_trace2cp_rf_margin_px)
                )
                test_trace2cp_metric = _evaluate_trace2cp_metric_fixed_set(
                    model,
                    test_loader,
                    device=device,
                    start_sample_index=test_selection.start_sample_index,
                    sample_count=test_selection.sample_count,
                    step_px=training.test_trace2cp_step_px,
                    rf_margin_px=trace2cp_margin,
                    sample_mode=test_selection.sample_mode,
                )
                last_test_trace2cp_error = test_trace2cp_metric.error_mean

            is_first_run_step = step == start_step
            wrote_tensorboard = False
            if writer is not None and (
                is_first_run_step or step == 1 or step % training.scalar_log_interval == 0
            ):
                writer.add_scalar("train/loss_total", loss_value, step)
                writer.add_scalar("train/loss_direction", metrics.loss_direction, step)
                if bool(training.presence_enabled):
                    writer.add_scalar("train/loss_presence", metrics.loss_presence, step)
                    writer.add_scalar("train/presence_positive_loss", metrics.presence_positive_loss, step)
                    writer.add_scalar("train/presence_negative_loss", metrics.presence_negative_loss, step)
                    writer.add_scalar("train/presence_positive_samples", metrics.presence_positive_samples, step)
                    writer.add_scalar("train/presence_negative_samples", metrics.presence_negative_samples, step)
                if bool(training.contrastive_enabled):
                    writer.add_scalar("train/loss_contrastive", metrics.loss_contrastive, step)
                    writer.add_scalar(
                        "train/contrastive_positive_loss",
                        metrics.contrastive_positive_loss,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_negative_loss",
                        metrics.contrastive_negative_loss,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_pixel_negative_loss",
                        metrics.contrastive_pixel_negative_loss,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_similarity_mean_loss",
                        metrics.contrastive_similarity_mean_loss,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_similarity_mean_value",
                        metrics.contrastive_similarity_mean_value,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_similarity_mean_target",
                        metrics.contrastive_similarity_mean_target,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_positive_samples",
                        metrics.contrastive_positive_samples,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_negative_samples",
                        metrics.contrastive_negative_samples,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_pixel_negative_samples",
                        metrics.contrastive_pixel_negative_samples,
                        step,
                    )
                    writer.add_scalar(
                        "train/contrastive_similarity_mean_samples",
                        metrics.contrastive_similarity_mean_samples,
                        step,
                    )
                writer.add_scalar("train/angle_error_mean_deg", metrics.angle_mean_deg, step)
                writer.add_scalar("train/supervision_samples", int(supervision.target.shape[0]), step)
                if bool(training.top_view_enabled):
                    writer.add_scalar("train/top_loss_direction", metrics.top_loss_direction, step)
                    writer.add_scalar("train/top_loss_dt", metrics.top_loss_dt, step)
                    writer.add_scalar("train/top_angle_error_mean_deg", metrics.top_angle_mean_deg, step)
                    writer.add_scalar("train/top_direction_samples", metrics.top_direction_samples, step)
                    writer.add_scalar("train/top_dt_samples", metrics.top_dt_samples, step)
                writer.add_scalar("timing/load_ms", load_ms, step)
                writer.add_scalar("timing/pipeline_wait_ms", wait_ms, step)
                writer.add_scalar("timing/prep_enqueue_ms", prep_ms, step)
                writer.add_scalar("timing/prep_gpu_ms", prep_gpu_ms, step)
                writer.add_scalar("timing/prep_wait_ms", prep_wait_ms, step)
                writer.add_scalar("timing/prep_submit_ms", prep_submit_ms, step)
                writer.add_scalar("timing/train_ms", train_ms, step)
                for key, value in _cache_scalars(batch.cache_stats).items():
                    writer.add_scalar(key, value, step)
                wrote_tensorboard = True
            if writer is not None and test_result is not None:
                writer.add_scalar("test/loss_direction", test_result.loss, step)
                writer.add_scalar("test/angle_error_mean_deg", test_result.angle_mean_deg, step)
                writer.add_scalar("test/supervision_samples", test_result.supervision_samples, step)
                if bool(training.top_view_enabled):
                    writer.add_scalar("test/top_loss_direction", test_result.top_loss_direction, step)
                    writer.add_scalar("test/top_loss_dt", test_result.top_loss_dt, step)
                    writer.add_scalar("test/top_angle_error_mean_deg", test_result.top_angle_mean_deg, step)
                    writer.add_scalar("test/top_direction_samples", test_result.top_direction_samples, step)
                    writer.add_scalar("test/top_dt_samples", test_result.top_dt_samples, step)
                if test_trace2cp_metric is not None:
                    writer.add_scalar("test/trace2cp_error", test_trace2cp_metric.error_mean, step)
                    writer.add_scalar(
                        "test/trace2cp_raw_y_error_mean_px",
                        test_trace2cp_metric.raw_y_error_mean_px,
                        step,
                    )
                    writer.add_scalar("test/trace2cp_segments", test_trace2cp_metric.segments, step)
                    writer.add_scalar("test/trace2cp_skipped_segments", test_trace2cp_metric.skipped_segments, step)
                for key, value in _cache_scalars(test_result.batch.cache_stats).items():
                    writer.add_scalar(f"test_{key}", value, step)
                wrote_tensorboard = True
            if writer is not None and (
                is_first_run_step or step == 1 or step % training.tensorboard_image_interval == 0
            ):
                writer.add_image("train/batch_direction_overlay", _make_training_visualization(batch, outputs), step)
                if bool(training.presence_enabled):
                    writer.add_image(
                        "train/batch_presence",
                        _make_presence_visualization(
                            batch,
                            outputs,
                            presence_channels=_presence_channel_count(training),
                        ),
                        step,
                    )
                if bool(training.contrastive_enabled):
                    writer.add_image(
                        "train/batch_embedding_similarity",
                        _make_embedding_similarity_visualization(
                            batch,
                            outputs,
                            presence_channels=_presence_channel_count(training),
                        ),
                        step,
                    )
                if bool(training.top_view_enabled) and prepared.top_batch is not None and top_outputs is not None:
                    writer.add_image(
                        "train/top_batch_direction_overlay",
                        _make_training_visualization(prepared.top_batch, top_outputs),
                        step,
                    )
                    writer.add_image(
                        "train/top_batch_distance_transform",
                        _make_presence_visualization(
                            prepared.top_batch,
                            top_outputs,
                            presence_channels=1,
                        ),
                        step,
                    )
                wrote_tensorboard = True
            if writer is not None and test_result is not None:
                writer.add_image(
                    "test/batch_direction_overlay",
                    _make_training_visualization(test_result.batch, test_result.outputs),
                    step,
                )
                if bool(training.presence_enabled):
                    writer.add_image(
                        "test/batch_presence",
                        _make_presence_visualization(
                            test_result.batch,
                            test_result.outputs,
                            presence_channels=_presence_channel_count(training),
                        ),
                        step,
                    )
                if bool(training.contrastive_enabled):
                    writer.add_image(
                        "test/batch_embedding_similarity",
                        _make_embedding_similarity_visualization(
                            test_result.batch,
                            test_result.outputs,
                            presence_channels=_presence_channel_count(training),
                        ),
                        step,
                    )
                if (
                    bool(training.top_view_enabled)
                    and test_result.top_batch is not None
                    and test_result.top_outputs is not None
                ):
                    writer.add_image(
                        "test/top_batch_direction_overlay",
                        _make_training_visualization(test_result.top_batch, test_result.top_outputs),
                        step,
                    )
                    writer.add_image(
                        "test/top_batch_distance_transform",
                        _make_presence_visualization(
                            test_result.top_batch,
                            test_result.top_outputs,
                            presence_channels=1,
                        ),
                        step,
                    )
                wrote_tensorboard = True
            if writer is not None and wrote_tensorboard:
                writer.flush()

            should_save_current = (
                bool(should_test)
                if test_loader is not None
                else (step == 1 or step % training.checkpoint_interval == 0 or (finite_steps and step == training.max_steps))
            )
            if should_save_current:
                checkpoint_loss = (
                    test_trace2cp_metric.error_mean
                    if test_trace2cp_metric is not None
                    else (test_result.loss if test_result is not None else loss_value)
                )
                checkpoint_metric = (
                    "test/trace2cp_error"
                    if test_trace2cp_metric is not None
                    else ("test/loss_direction" if test_result is not None else "train/loss_total")
                )
                _save_checkpoint(
                    snapshots / "current.pt",
                    step=step,
                    model=model,
                    top_model=top_model,
                    optimizer=optimizer,
                    loss=checkpoint_loss,
                    raw_config=raw_config,
                    metric_name=checkpoint_metric,
                )
            if test_loader is None or test_result is not None:
                metric_value = (
                    test_trace2cp_metric.error_mean
                    if test_trace2cp_metric is not None
                    else (test_result.loss if test_result is not None else loss_value)
                )
                metric_name = (
                    "test/trace2cp_error"
                    if test_trace2cp_metric is not None
                    else ("test/loss_direction" if test_result is not None else "train/loss_total")
                )
                if metric_value < best_metric:
                    best_metric = metric_value
                    _save_checkpoint(
                        snapshots / "best.pt",
                        step=step,
                        model=model,
                        top_model=top_model,
                        optimizer=optimizer,
                        loss=metric_value,
                        raw_config=raw_config,
                        metric_name=metric_name,
                    )
            if _should_print_training_step(
                step,
                scalar_log_interval=training.scalar_log_interval,
                start_sample_index=raw_start_sample_index,
                sample_count=training.train_control_points_per_step,
            ):
                step_ms = (time.perf_counter() - step_wall_start) * 1000.0
                if writer is not None and (step == 1 or step % training.scalar_log_interval == 0):
                    writer.add_scalar("timing/step_ms", step_ms, step)
                test_part = (
                    ""
                    if test_result is None
                    else (
                        f" test_loss_direction={test_result.loss:.6f} "
                        f"test_angle_mean_deg={test_result.angle_mean_deg:.2f}"
                        + (
                            ""
                            if not bool(training.top_view_enabled)
                            else (
                                f" test_top_loss_direction={test_result.top_loss_direction:.6f} "
                                f"test_top_loss_dt={test_result.top_loss_dt:.6f} "
                                f"test_top_angle_mean_deg={test_result.top_angle_mean_deg:.2f}"
                            )
                        )
                        + (
                            ""
                            if test_trace2cp_metric is None
                            else (
                                f" test_trace2cp_error={test_trace2cp_metric.error_mean:.6f} "
                                f"test_trace2cp_raw_y_px={test_trace2cp_metric.raw_y_error_mean_px:.3f} "
                                f"test_trace2cp_segments={test_trace2cp_metric.segments} "
                                f"test_trace2cp_skipped={test_trace2cp_metric.skipped_segments}"
                            )
                        )
                    )
                )
                top_part = (
                    ""
                    if not bool(training.top_view_enabled)
                    else (
                        f" top_loss_direction={metrics.top_loss_direction:.6f} "
                        f"top_loss_dt={metrics.top_loss_dt:.6f} "
                        f"top_angle_mean_deg={metrics.top_angle_mean_deg:.2f} "
                        f"top_dt_samples={metrics.top_dt_samples}"
                    )
                )
                print(
                    f"step={step} loss_total={loss_value:.6f} "
                    f"loss_direction={metrics.loss_direction:.6f} "
                    f"loss_presence={metrics.loss_presence:.6f} "
                    f"loss_contrastive={metrics.loss_contrastive:.6f} "
                    f"angle_mean_deg={metrics.angle_mean_deg:.2f} "
                    f"supervision_samples={int(supervision.target.shape[0])}{top_part}{test_part} "
                    f"load_ms={load_ms:.1f} wait_ms={wait_ms:.1f} "
                    f"prep_ms={prep_ms:.1f} prep_gpu_ms={prep_gpu_ms:.1f} "
                    f"prep_wait_ms={prep_wait_ms:.1f} prep_submit_ms={prep_submit_ms:.1f} "
                    f"train_ms={train_ms:.1f} step_ms={step_ms:.1f}",
                    flush=True,
                )
            step += 1
    finally:
        if prep_pipeline is not None:
            prep_pipeline.close()
        elif pipeline is not None:
            pipeline.close()
        if writer is not None:
            writer.flush()
            writer.close()
    test_complete = (
        ""
        if math.isnan(last_test_loss)
        else (
            f" test_loss_direction={last_test_loss:.6f} "
            f"test_angle_mean_deg={last_test_angle_mean_deg:.2f} "
            f"test_trace2cp_error={last_test_trace2cp_error:.6f}"
        )
    )
    print(
        f"fiber_trace_2d train complete step={training.max_steps} "
        f"loss_total={last_loss:.6f} angle_mean_deg={last_angle_mean_deg:.2f}{test_complete}",
        flush=True,
    )
    return run_dir


def prefetch_training(
    config_path: str | Path,
    *,
    prefetch_steps: int | None = None,
    prefetch_start_step: int = 1,
    sampler_factory: SamplerFactory | None = None,
) -> dict[str, Any]:
    raw_config = _load_raw_config(config_path)
    training = _training_config_from_raw(raw_config)
    if int(prefetch_start_step) <= 0:
        raise ValueError("--prefetch-start-step must be >= 1")
    if prefetch_steps is not None and int(prefetch_steps) < 0:
        raise ValueError("--prefetch-steps must be >= 0")

    loader_config = load_config(config_path)
    loader = FiberStrip2DLoader(loader_config, sampler_factory=sampler_factory)
    sample_mode = "random"
    requested_prefetch_steps = None if prefetch_steps is None else int(prefetch_steps)
    prefetch_full_dataset = requested_prefetch_steps == 0 or (
        requested_prefetch_steps is None and training.max_steps == 0
    )
    if prefetch_full_dataset:
        effective_steps: int | str = "dataset"
        sample_count = int(training.max_sample_index) if training.max_sample_index > 0 else int(loader.sample_count)
    else:
        effective_steps = training.max_steps if requested_prefetch_steps is None else requested_prefetch_steps
        sample_count = int(effective_steps) * int(training.train_control_points_per_step)
    start_sample_index = 0 if prefetch_full_dataset else (
        (int(prefetch_start_step) - 1) * int(training.train_control_points_per_step)
    )
    effective_steps_text = str(effective_steps)
    print(
        "fiber_trace_2d prefetch "
        f"start_step={int(prefetch_start_step)} steps={effective_steps_text} "
        f"control_points_per_step={int(training.train_control_points_per_step)} "
        f"sample_mode={sample_mode} start_sample_index={start_sample_index} "
        f"max_sample_index={int(training.max_sample_index)} samples={sample_count}",
        flush=True,
    )
    summary = loader.prefetch(
        start_sample_index,
        sample_count,
        sample_mode=sample_mode,
        sample_index_limit=training.max_sample_index,
        include_top_view=bool(training.top_view_enabled),
    )
    if prefetch_full_dataset:
        test_loader_config = _test_loader_config_from_raw(raw_config, loader_config)
        if test_loader_config is not None:
            test_loader = FiberStrip2DLoader(test_loader_config, sampler_factory=sampler_factory)
            print(
                "fiber_trace_2d prefetch test_datasets "
                f"sample_mode={sample_mode} start_sample_index=0 samples={int(test_loader.sample_count)}",
                flush=True,
            )
            test_summary = test_loader.prefetch(
                0,
                int(test_loader.sample_count),
                sample_mode=sample_mode,
                include_top_view=bool(training.top_view_enabled),
            )
            summary = _merge_prefetch_summaries(summary, test_summary)
    print(
        "fiber_trace_2d prefetch complete "
        f"generated={int(summary.get('generated', 0))} missing={int(summary.get('missing', 0))} "
        f"downloaded={int(summary.get('downloaded', 0))} errors={int(summary.get('errors', 0))}",
        flush=True,
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train V0 2D fiber-strip direction model")
    parser.add_argument("config", help="Path to fiber_trace_2d JSON config")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from an existing checkpoint while creating a fresh timestamped run directory",
    )
    parser.add_argument("--prefetch", action="store_true", help="Prefetch training chunks and exit without training")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 100 training batches without testing, TensorBoard, or snapshots, then report patch samples/s",
    )
    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=100,
        help="Number of batches for --benchmark/--profile/--load-only; default 100",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Print benchmark/training-stage timing rows and per-patch summary for the 100-batch benchmark run",
    )
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Run the 100-batch benchmark loader path only, skipping image augmentation and model training work",
    )
    parser.add_argument(
        "--prefetch-steps",
        type=int,
        default=None,
        help=(
            "Training steps to prefetch; positive values override training.max_steps, "
            "0 prefetches the full configured CP dataset once, omitted uses training.max_steps"
        ),
    )
    parser.add_argument(
        "--prefetch-start-step",
        type=int,
        default=1,
        help="1-based training step whose deterministic sample range starts prefetching",
    )
    args = parser.parse_args(argv)
    if args.prefetch:
        try:
            prefetch_training(
                args.config,
                prefetch_steps=args.prefetch_steps,
                prefetch_start_step=args.prefetch_start_step,
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    if args.benchmark or args.profile or args.load_only:
        try:
            run_benchmark(
                args.config,
                batches=int(args.benchmark_batches),
                profile=bool(args.profile),
                load_only=bool(args.load_only),
            )
        except ValueError as exc:
            parser.error(str(exc))
        return
    try:
        run_training(args.config, resume_checkpoint=args.resume)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
