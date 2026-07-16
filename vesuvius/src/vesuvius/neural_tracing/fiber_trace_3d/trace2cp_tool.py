from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from vesuvius.neural_tracing.fiber_trace_3d.direction import (
    decode_lasagna_direction_3x2_analytic,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DConfig,
    FiberTrace3DLoader,
    _normalize_image,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.model import (
    build_fiber_trace_3d_model,
)
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _device_from_training,
    _load_raw_config,
    _load_snapshot,
    _make_trace2cp_geometry_loader,
    _trace2cp_3d_config,
    dataclass_replace,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import _Trace2CpSegmentSource
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index


_EPS = 1.0e-12
_REMOTE_PREFIXES = ("http://", "https://", "s3://")


@dataclass(frozen=True)
class NativeTrace2CpConfig:
    step_voxels: float = 4.0
    cone_angle_degrees: float = 25.0
    cone_grid_size: int = 25
    max_step_factor: float = 3.0
    max_steps: int | None = None
    trace_step_limit: int | None = None
    inference_patch_shape_zyx: tuple[int, int, int] = (64, 64, 64)
    core_margin_voxels: int = 8


@dataclass(frozen=True)
class NativeTraceStep:
    point_zyx: np.ndarray
    direction_loss: float
    presence_loss: float
    total_loss: float
    rejected_candidates: int


@dataclass(frozen=True)
class NativeTraceResult:
    trace_zyx: np.ndarray
    reached_target_plane: bool
    reason: str
    steps: tuple[NativeTraceStep, ...]


@dataclass(frozen=True)
class NativeTraceFusionResult:
    fused_zyx: np.ndarray
    closest_progress: float
    raw_gap_voxels: float
    considered_gap_voxels: float
    center_penalty: float
    closest_midpoint_zyx: np.ndarray
    closest_forward_zyx: np.ndarray
    closest_reverse_zyx: np.ndarray
    reached_overlap: bool
    reason: str


@dataclass(frozen=True)
class NativeTracePairResult:
    forward: NativeTraceResult
    reverse: NativeTraceResult
    fusion: NativeTraceFusionResult
    fused_zyx: np.ndarray
    plane_error: float
    closest_target_error: float
    span_voxels: float


@dataclass(frozen=True)
class _InferredBlock:
    origin_zyx: np.ndarray
    shape_zyx: tuple[int, int, int]
    core_lo_zyx: np.ndarray
    core_hi_zyx: np.ndarray
    output_czyx: torch.Tensor
    valid_mask_zyx: torch.Tensor


@dataclass(frozen=True)
class _NativeTrace2CpSelection:
    record: Any
    record_index: int
    sample_index: int
    sample_mode: str
    start_cp_index: int
    target_cp_index: int
    explicit_segment: bool


def _resolve_config_relative_path(path: str | Path, raw_config: dict[str, Any]) -> str:
    path_s = str(path)
    if path_s.startswith(_REMOTE_PREFIXES):
        return path_s
    path_obj = Path(path_s).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    config_dir = raw_config.get("_config_dir")
    if config_dir is not None:
        return str((Path(str(config_dir)) / path_obj).resolve())
    return str((Path.cwd() / path_obj).resolve())


def _as_zyx3(value: Any, *, key: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        result = (int(value), int(value), int(value))
    elif isinstance(value, (list, tuple)) and len(value) == 3:
        result = tuple(int(v) for v in value)
    else:
        raise ValueError(f"{key} must be an int or length-3 sequence")
    if any(v <= 0 for v in result):
        raise ValueError(f"{key} values must be positive")
    return result


def _unit(vector: np.ndarray, *, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if math.isfinite(norm) and norm > _EPS:
        return (arr / norm).astype(np.float32)
    if fallback is None:
        fallback = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return _unit(fallback)


def _require_unit(vector: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= _EPS:
        raise ValueError(f"{label} must be finite and non-zero")
    return (arr / norm).astype(np.float32)


def _align_axis(axis: np.ndarray, reference: np.ndarray) -> np.ndarray:
    aligned = _unit(axis)
    ref = _unit(reference)
    if float(np.dot(aligned, ref)) < 0.0:
        aligned = -aligned
    return aligned.astype(np.float32, copy=False)


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unit = _unit(axis)
    candidates = np.eye(3, dtype=np.float32)
    ref = candidates[int(np.argmin(np.abs(candidates @ unit)))]
    b0 = _unit(np.cross(unit, ref))
    b1 = _unit(np.cross(unit, b0))
    return b0, b1


def generate_cone_candidates(
    axis_zyx: np.ndarray,
    *,
    max_angle_degrees: float,
    grid_size: int = 25,
) -> np.ndarray:
    """Generate deterministic unit candidate directions inside a 3D cone."""

    axis = _unit(axis_zyx)
    max_angle = math.radians(max(0.0, float(max_angle_degrees)))
    grid_count = int(grid_size)
    if grid_count <= 0:
        raise ValueError("grid_size must be positive")
    if max_angle <= 0.0 or grid_count == 1:
        return axis.reshape(1, 3).astype(np.float32)
    b0, b1 = _orthonormal_basis(axis)
    lin = np.linspace(-1.0, 1.0, grid_count, dtype=np.float32)
    uu, vv = np.meshgrid(lin, lin, indexing="xy")
    a = uu.reshape(-1).astype(np.float32)
    b = vv.reshape(-1).astype(np.float32)
    disk_x = np.zeros_like(a)
    disk_y = np.zeros_like(b)
    nonzero = (a != 0.0) | (b != 0.0)
    a_nz = a[nonzero]
    b_nz = b[nonzero]
    use_a = np.abs(a_nz) > np.abs(b_nz)
    r = np.empty_like(a_nz)
    phi = np.empty_like(a_nz)
    r[use_a] = a_nz[use_a]
    phi[use_a] = (np.float32(math.pi / 4.0) * b_nz[use_a]) / a_nz[use_a]
    r[~use_a] = b_nz[~use_a]
    phi[~use_a] = np.float32(math.pi / 2.0) - (
        np.float32(math.pi / 4.0) * a_nz[~use_a]
    ) / b_nz[~use_a]
    disk_x[nonzero] = r * np.cos(phi)
    disk_y[nonzero] = r * np.sin(phi)
    tangent_scale = np.float32(math.tan(max_angle))
    directions = (
        axis[None, :]
        + tangent_scale * disk_x[:, None] * b0[None, :]
        + tangent_scale * disk_y[:, None] * b1[None, :]
    )
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, np.float32(_EPS))
    center_index = int(np.argmin(disk_x * disk_x + disk_y * disk_y))
    order = np.concatenate(
        [
            np.asarray([center_index], dtype=np.int64),
            np.asarray(
                [idx for idx in range(int(directions.shape[0])) if idx != center_index],
                dtype=np.int64,
            ),
        ]
    )
    return directions[order].astype(np.float32, copy=False)


def _grid_sample_channels_at_points(
    values_czyx: torch.Tensor,
    points_zyx: np.ndarray,
    *,
    origin_zyx: np.ndarray,
) -> torch.Tensor:
    if values_czyx.ndim != 4:
        raise ValueError("values_czyx must have shape C,Z,Y,X")
    _channels, depth, height, width = (int(v) for v in values_czyx.shape)
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_zyx must have shape [N,3]")
    if int(points.shape[0]) == 0:
        return torch.zeros((0, int(values_czyx.shape[0])), dtype=values_czyx.dtype, device=values_czyx.device)
    local = torch.as_tensor(
        points - np.asarray(origin_zyx, dtype=np.float32),
        dtype=torch.float32,
        device=values_czyx.device,
    )
    if depth > 1:
        gz = local[:, 0] * (2.0 / float(depth - 1)) - 1.0
    else:
        gz = torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=values_czyx.device)
    if height > 1:
        gy = local[:, 1] * (2.0 / float(height - 1)) - 1.0
    else:
        gy = torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=values_czyx.device)
    if width > 1:
        gx = local[:, 2] * (2.0 / float(width - 1)) - 1.0
    else:
        gx = torch.zeros((int(points.shape[0]),), dtype=torch.float32, device=values_czyx.device)
    grid = torch.stack([gx, gy, gz], dim=1).view(1, int(points.shape[0]), 1, 1, 3)
    sampled = F.grid_sample(
        values_czyx.view(1, *values_czyx.shape),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[0, :, :, 0, 0].transpose(0, 1).contiguous()


class NativeTraceFieldCache:
    """Lazy overlapped 3D model-output cache with trusted-core point routing."""

    def __init__(
        self,
        *,
        record: Any,
        model: torch.nn.Module,
        config: FiberTrace3DConfig,
        image_normalization: str,
        patch_shape_zyx: tuple[int, int, int],
        core_margin_voxels: int,
        device: torch.device,
    ) -> None:
        self.record = record
        self.model = model
        self.config = config
        self.image_normalization = str(image_normalization)
        self.patch_shape_zyx = tuple(int(v) for v in patch_shape_zyx)
        self.core_margin = int(core_margin_voxels)
        if self.core_margin < 0:
            raise ValueError("core_margin_voxels must be >= 0")
        if any(v <= 2 * self.core_margin for v in self.patch_shape_zyx):
            raise ValueError(
                "inference_patch_shape_zyx must be larger than 2 * core_margin_voxels"
            )
        self.core_shape_zyx = tuple(v - 2 * self.core_margin for v in self.patch_shape_zyx)
        self.device = torch.device(device)
        self._blocks: dict[tuple[int, int, int], _InferredBlock] = {}

    def _block_origin_for_point(self, point_zyx: np.ndarray) -> np.ndarray:
        point = np.asarray(point_zyx, dtype=np.float64)
        stride = np.asarray(self.core_shape_zyx, dtype=np.float64)
        margin = float(self.core_margin)
        origin = np.floor((point - margin) / stride).astype(np.int64) * np.asarray(
            self.core_shape_zyx, dtype=np.int64
        )
        return origin.astype(np.int64)

    def _block_origins_for_points(self, points_zyx: np.ndarray) -> np.ndarray:
        points = np.asarray(points_zyx, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        stride = np.asarray(self.core_shape_zyx, dtype=np.float64)
        origin = np.floor((points - float(self.core_margin)) / stride).astype(np.int64)
        origin *= np.asarray(self.core_shape_zyx, dtype=np.int64)[None, :]
        return origin.astype(np.int64, copy=False)

    def _block_coords_base_and_valid(
        self, origin_zyx: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = np.asarray(self.patch_shape_zyx, dtype=np.int64)
        zz, yy, xx = np.meshgrid(
            np.arange(int(shape[0]), dtype=np.float32) + np.float32(origin_zyx[0]),
            np.arange(int(shape[1]), dtype=np.float32) + np.float32(origin_zyx[1]),
            np.arange(int(shape[2]), dtype=np.float32) + np.float32(origin_zyx[2]),
            indexing="ij",
        )
        coords_selected = np.stack([zz, yy, xx], axis=-1).astype(np.float32, copy=False)
        spacing = np.float32(getattr(self.record, "volume_spacing_base", 1.0))
        coords_base = np.ascontiguousarray(coords_selected * spacing)
        base_shape = np.asarray(getattr(self.record, "base_shape_zyx"), dtype=np.float32)
        if base_shape.shape != (3,):
            raise ValueError("native 3D Trace2CP record.base_shape_zyx must have length 3")
        valid = (
            np.isfinite(coords_base).all(axis=-1)
            & (coords_base[..., 0] >= 0.0)
            & (coords_base[..., 0] <= float(base_shape[0] - 1.0))
            & (coords_base[..., 1] >= 0.0)
            & (coords_base[..., 1] <= float(base_shape[1] - 1.0))
            & (coords_base[..., 2] >= 0.0)
            & (coords_base[..., 2] <= float(base_shape[2] - 1.0))
        )
        return coords_base, np.ascontiguousarray(valid)

    def _sample_block_volume(self, origin_zyx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        sampler = getattr(self.record, "sampler", None)
        if sampler is None:
            raise ValueError("native 3D Trace2CP record must provide a coordinate sampler")
        if hasattr(sampler, "blocking") and not bool(getattr(sampler, "blocking")):
            raise ValueError("native 3D Trace2CP requires blocking coordinate sampling")
        coords_base, valid = self._block_coords_base_and_valid(origin_zyx)
        result = sampler.sample_coord_batch(coords_base, valid)
        stats = dict(getattr(result, "stats", {}) or {})
        error_chunks = int(stats.get("error_chunks", 0) or 0)
        if error_chunks > 0:
            raise ValueError(
                "native 3D Trace2CP block sampling encountered chunk errors; "
                f"stats={stats}"
            )
        if "requested_level_only" in stats and not bool(stats.get("requested_level_only")):
            raise ValueError(
                "native 3D Trace2CP block sampling did not use requested-level-only mode; "
                f"stats={stats}"
            )
        fallback_levels = int(stats.get("fallback_levels", 0) or 0)
        if fallback_levels != 0:
            raise ValueError(
                "native 3D Trace2CP block sampling used scale fallback; "
                f"stats={stats}"
            )
        sampled_np = np.asarray(result.image, dtype=np.float32)
        sampled_valid_np = np.asarray(result.valid_mask, dtype=bool) & valid
        expected_shape = self.patch_shape_zyx
        if sampled_np.shape != expected_shape:
            raise ValueError(
                "native 3D Trace2CP sampler returned incompatible image shape: "
                f"shape={sampled_np.shape} expected={expected_shape}"
            )
        if sampled_valid_np.shape != expected_shape:
            raise ValueError(
                "native 3D Trace2CP sampler returned incompatible valid-mask shape: "
                f"shape={sampled_valid_np.shape} expected={expected_shape}"
            )
        sampled = torch.as_tensor(sampled_np, dtype=torch.float32, device=self.device)
        sampled_valid = torch.as_tensor(sampled_valid_np, dtype=torch.bool, device=self.device)
        sampled = torch.where(sampled_valid, sampled, torch.zeros_like(sampled))
        return sampled, sampled_valid

    @torch.no_grad()
    def _infer_block(self, origin_zyx: np.ndarray) -> _InferredBlock:
        origin = np.asarray(origin_zyx, dtype=np.int64)
        key = tuple(int(v) for v in origin)
        block = self._blocks.get(key)
        if block is not None:
            return block
        raw_t, valid = self._sample_block_volume(origin)
        image = _normalize_image(raw_t, valid, self.image_normalization)
        was_training = bool(self.model.training)
        self.model.eval()
        output = (
            self.model(image.view(1, 1, *self.patch_shape_zyx))[0]
            .detach()
            .to(device=torch.device("cpu"), dtype=torch.float32)
            .contiguous()
        )
        valid_cpu = valid.detach().to(device=torch.device("cpu")).contiguous()
        if was_training:
            self.model.train()
        core_lo = origin + int(self.core_margin)
        core_hi = origin + np.asarray(self.patch_shape_zyx, dtype=np.int64) - int(self.core_margin)
        block = _InferredBlock(
            origin_zyx=origin.astype(np.int64),
            shape_zyx=self.patch_shape_zyx,
            core_lo_zyx=core_lo.astype(np.float32),
            core_hi_zyx=core_hi.astype(np.float32),
            output_czyx=output,
            valid_mask_zyx=valid_cpu,
        )
        self._blocks[key] = block
        return block

    def block_for_point(self, point_zyx: np.ndarray) -> _InferredBlock:
        point = np.asarray(point_zyx, dtype=np.float32)
        origin = self._block_origin_for_point(point)
        block = self._infer_block(origin)
        inside = np.all(point >= block.core_lo_zyx) and np.all(point < block.core_hi_zyx)
        if not bool(inside):
            raise ValueError(
                "native 3D Trace2CP point is outside trusted inference core: "
                f"point_zyx={point.tolist()} core_lo={block.core_lo_zyx.tolist()} "
                f"core_hi={block.core_hi_zyx.tolist()}"
            )
        return block

    @torch.no_grad()
    def sample_points_torch(
        self,
        points_zyx: np.ndarray,
        *,
        progress_label: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = np.asarray(points_zyx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        count = int(points.shape[0])
        directions = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        presence = torch.zeros((count,), dtype=torch.float32, device=self.device)
        valid = torch.zeros((count,), dtype=torch.bool, device=self.device)
        if count == 0:
            return directions, presence, valid

        origins = self._block_origins_for_points(points)
        unique_origins, inverse = np.unique(origins, axis=0, return_inverse=True)
        shape = np.asarray(self.patch_shape_zyx, dtype=np.float32)
        progress_start = time.perf_counter()
        last_progress_time = 0.0
        new_blocks = 0
        cached_blocks = 0
        valid_done = 0

        def emit_progress(block_index: int, *, force: bool = False) -> None:
            nonlocal last_progress_time
            if progress_label is None:
                return
            now = time.perf_counter()
            if not force and now - last_progress_time < 0.25:
                return
            last_progress_time = now
            _emit_native_progress(
                f"strip presence {progress_label}",
                block_index,
                int(unique_origins.shape[0]),
                progress_start,
                detail=(
                    f"points={count} valid={valid_done} "
                    f"new={new_blocks} cached={cached_blocks} "
                    f"cache_blocks={len(self._blocks)}"
                ),
            )

        emit_progress(0, force=True)
        for unique_index, origin in enumerate(unique_origins):
            indices = np.flatnonzero(inverse == int(unique_index))
            if indices.size == 0:
                emit_progress(int(unique_index) + 1)
                continue
            key = tuple(int(v) for v in np.asarray(origin, dtype=np.int64))
            was_cached = key in self._blocks
            block = self._infer_block(origin)
            if was_cached:
                cached_blocks += 1
            else:
                new_blocks += 1
            group_points = points[indices]
            local = group_points - block.origin_zyx.astype(np.float32)
            inside_core = np.all(group_points >= block.core_lo_zyx, axis=1) & np.all(
                group_points < block.core_hi_zyx,
                axis=1,
            )
            inside_block = np.all(local >= 0.0, axis=1) & np.all(
                local <= (shape - np.asarray([1.0, 1.0, 1.0], dtype=np.float32)),
                axis=1,
            )
            usable = inside_core & inside_block
            if not bool(np.any(usable)):
                emit_progress(int(unique_index) + 1)
                continue
            usable_indices = indices[usable]
            usable_points = points[usable_indices]
            block_output = block.output_czyx.to(device=self.device, non_blocking=True)
            sampled = _grid_sample_channels_at_points(
                block_output,
                usable_points,
                origin_zyx=block.origin_zyx,
            )
            if int(sampled.shape[1]) < 6:
                raise ValueError("native 3D Trace2CP model output has fewer than six channels")
            block_valid = block.valid_mask_zyx.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            valid_values = _grid_sample_channels_at_points(
                block_valid.view(1, *block.shape_zyx),
                usable_points,
                origin_zyx=block.origin_zyx,
            )[:, 0]
            axis_xyz = decode_lasagna_direction_3x2_analytic(sampled[:, :6])
            axis_zyx = axis_xyz[:, [2, 1, 0]].to(dtype=torch.float32)
            axis_zyx = F.normalize(axis_zyx, p=2.0, dim=1, eps=float(_EPS))
            group_presence = (
                sampled[:, 6].to(dtype=torch.float32).clamp(0.0, 1.0)
                if int(sampled.shape[1]) >= 7
                else torch.ones((int(sampled.shape[0]),), dtype=torch.float32, device=self.device)
            )
            group_valid = (valid_values > 0.5) & torch.isfinite(axis_zyx).all(dim=1)
            valid_done += int(torch.count_nonzero(group_valid).detach().cpu())
            index_t = torch.as_tensor(usable_indices, dtype=torch.long, device=self.device)
            directions[index_t] = axis_zyx
            presence[index_t] = group_presence
            valid[index_t] = group_valid
            emit_progress(int(unique_index) + 1)
        emit_progress(int(unique_origins.shape[0]), force=True)
        return directions, presence, valid

    def sample_point(self, point_zyx: np.ndarray) -> tuple[np.ndarray, float, bool]:
        directions, presence, valid = self.sample_points_torch(
            np.asarray(point_zyx, dtype=np.float32).reshape(1, 3)
        )
        if not bool(valid[0].detach().cpu()):
            return np.zeros((3,), dtype=np.float32), 0.0, False
        axis_zyx = directions[0].detach().cpu().numpy().astype(np.float32)
        return _unit(axis_zyx), float(presence[0].detach().cpu()), True


def _align_axes_torch(axes: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    axes_n = F.normalize(axes.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    refs_n = F.normalize(references.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    sign = torch.where(
        torch.sum(axes_n * refs_n, dim=-1, keepdim=True) >= 0.0,
        torch.ones((), dtype=torch.float32, device=axes_n.device),
        -torch.ones((), dtype=torch.float32, device=axes_n.device),
    )
    return axes_n * sign


def _format_eta(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0.0:
        return "?"
    seconds_i = int(round(float(seconds)))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _emit_native_progress(
    label: str,
    current: int,
    total: int,
    start_time: float,
    *,
    detail: str = "",
) -> None:
    total_i = max(1, int(total))
    current_i = max(0, min(int(current), total_i))
    progress = float(current_i) / float(total_i)
    elapsed = max(0.0, time.perf_counter() - float(start_time))
    eta = None if progress <= 1.0e-6 else elapsed * (1.0 - progress) / progress
    width = 24
    filled = int(math.floor(width * progress))
    bar = "#" * filled + "-" * (width - filled)
    suffix = "" if not detail else f" {detail}"
    print(
        f"native {label} [{bar}] {current_i}/{total_i} "
        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}{suffix}",
        flush=True,
    )


def _score_candidate_batch(
    cache: NativeTraceFieldCache,
    *,
    current_direction: np.ndarray,
    candidate_directions: np.ndarray,
    next_points: np.ndarray,
) -> tuple[int | None, float, float, float, int]:
    candidates = torch.as_tensor(
        np.asarray(candidate_directions, dtype=np.float32),
        dtype=torch.float32,
        device=cache.device,
    )
    candidates = F.normalize(candidates, p=2.0, dim=1, eps=float(_EPS))
    current = torch.as_tensor(
        np.asarray(current_direction, dtype=np.float32).reshape(1, 3),
        dtype=torch.float32,
        device=cache.device,
    )
    current = F.normalize(current, p=2.0, dim=1, eps=float(_EPS))
    candidates = _align_axes_torch(candidates, current.expand_as(candidates))
    next_directions, presences, valid = cache.sample_points_torch(next_points)
    current_dot = torch.sum(current.expand_as(candidates) * candidates, dim=1).clamp(-1.0, 1.0)
    next_aligned = _align_axes_torch(next_directions, candidates)
    next_dot = torch.sum(next_aligned * candidates, dim=1).clamp(-1.0, 1.0)
    presence = presences.clamp(0.0, 1.0)
    score = current_dot * next_dot * presence
    score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    direction_loss = 1.0 - 0.5 * (current_dot + next_dot)
    presence_loss = 1.0 - presence
    total_loss = 1.0 - score
    valid_count = int(torch.count_nonzero(valid).detach().cpu())
    rejected = int(valid.numel()) - valid_count
    if valid_count == 0:
        return None, math.inf, math.inf, math.inf, rejected
    best_index = int(torch.argmax(score).detach().cpu())
    return (
        best_index,
        float(total_loss[best_index].detach().cpu()),
        float(direction_loss[best_index].detach().cpu()),
        float(presence_loss[best_index].detach().cpu()),
        rejected,
    )


def _plane_distance(point_zyx: np.ndarray, plane_point_zyx: np.ndarray, normal_zyx: np.ndarray) -> float:
    return float(np.dot(np.asarray(point_zyx, dtype=np.float64) - plane_point_zyx, normal_zyx))


def _interpolate_plane_crossing(
    start_zyx: np.ndarray,
    end_zyx: np.ndarray,
    *,
    plane_point_zyx: np.ndarray,
    plane_normal_zyx: np.ndarray,
) -> np.ndarray | None:
    d0 = _plane_distance(start_zyx, plane_point_zyx, plane_normal_zyx)
    d1 = _plane_distance(end_zyx, plane_point_zyx, plane_normal_zyx)
    if d0 == 0.0:
        return np.asarray(start_zyx, dtype=np.float32)
    if d0 * d1 > 0.0:
        return None
    denom = d0 - d1
    if abs(denom) <= _EPS:
        return np.asarray(end_zyx, dtype=np.float32)
    t = float(np.clip(d0 / denom, 0.0, 1.0))
    return (
        np.asarray(start_zyx, dtype=np.float64) * (1.0 - t)
        + np.asarray(end_zyx, dtype=np.float64) * t
    ).astype(np.float32)


def _fiber_line_tangent_zyx_toward_target(
    record: Any,
    *,
    start_control_point_index: int,
    target_control_point_index: int,
) -> np.ndarray:
    fiber = record.fiber
    line_points_xyz = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    if line_points_xyz.ndim != 2 or line_points_xyz.shape[1] != 3:
        raise ValueError("fiber line_points_xyz must have shape [N, 3]")
    start_line_index = control_point_line_index(fiber, int(start_control_point_index))
    target_line_index = control_point_line_index(fiber, int(target_control_point_index))
    if int(start_line_index) == int(target_line_index):
        raise ValueError("native 3D Trace2CP start and target line indices must differ")
    step = 1 if int(target_line_index) > int(start_line_index) else -1
    next_line_index = int(start_line_index) + step
    if next_line_index < 0 or next_line_index >= int(line_points_xyz.shape[0]):
        raise ValueError(
            "native 3D Trace2CP cannot derive CP-local tangent: "
            f"start_line_index={int(start_line_index)} target_line_index={int(target_line_index)}"
        )
    tangent_xyz = line_points_xyz[next_line_index] - line_points_xyz[int(start_line_index)]
    spacing = float(getattr(record, "volume_spacing_base", 1.0))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume_spacing_base for native 3D tangent: {spacing!r}")
    tangent_zyx = tangent_xyz[[2, 1, 0]] / spacing
    return _require_unit(
        tangent_zyx,
        label=(
            "native 3D Trace2CP CP-local fiber tangent "
            f"{int(start_control_point_index)}->{int(target_control_point_index)}"
        ),
    )


def trace_native_3d_one_way(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    progress_label: str | None = None,
) -> NativeTraceResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    plane_normal = _unit(target - start)
    span = float(np.linalg.norm(target - start))
    if span <= _EPS:
        raise ValueError("native 3D Trace2CP start and target CPs must differ")
    if not math.isfinite(float(cfg.step_voxels)) or float(cfg.step_voxels) <= 0.0:
        raise ValueError("step_voxels must be positive")
    if not math.isfinite(float(cfg.max_step_factor)) or float(cfg.max_step_factor) <= 0.0:
        raise ValueError("max_step_factor must be positive")
    if cfg.max_steps is not None and int(cfg.max_steps) <= 0:
        raise ValueError("max_steps must be positive when set")
    if cfg.trace_step_limit is not None and int(cfg.trace_step_limit) <= 0:
        raise ValueError("trace_step_limit must be positive when set")
    dynamic_limit = max(
        1,
        int(math.ceil(float(cfg.max_step_factor) * span / float(cfg.step_voxels))),
    )
    limit_candidates: list[tuple[int, str]] = [(dynamic_limit, "max_step_factor")]
    if cfg.max_steps is not None:
        limit_candidates.append((int(cfg.max_steps), "max_steps"))
    if cfg.trace_step_limit is not None:
        limit_candidates.append((int(cfg.trace_step_limit), "trace_step_limit"))
    step_limit, limit_reason = min(limit_candidates, key=lambda item: item[0])
    progress_max = max(1, step_limit)
    last_progress_time = 0.0
    trace_start_time = time.perf_counter()

    def emit_progress(point_zyx: np.ndarray, step: int, *, reason: str | None = None) -> None:
        nonlocal last_progress_time
        if progress_label is None:
            return
        now = time.perf_counter()
        if reason is None and step > 0 and now - last_progress_time < 0.25:
            return
        last_progress_time = now
        progress = float(
            np.dot(np.asarray(point_zyx, dtype=np.float32) - start, plane_normal)
            / max(span, _EPS)
        )
        progress = float(np.clip(progress, 0.0, 1.0))
        elapsed = max(0.0, now - trace_start_time)
        eta = None if progress <= 1.0e-6 else elapsed * (1.0 - progress) / progress
        bar_width = 24
        filled = int(math.floor(bar_width * progress))
        bar = "#" * filled + "-" * (bar_width - filled)
        suffix = "" if reason is None else f" reason={reason}"
        end = "\r" if reason is None else "\n"
        print(
            f"native trace {progress_label} [{bar}] "
            f"{progress * 100.0:5.1f}% step={int(step)}/{progress_max} "
            f"eta={_format_eta(eta)} blocks={len(cache._blocks)}{suffix}",
            end=end,
            flush=True,
        )

    emit_progress(start, 0)
    initial_sampled_direction, _presence, valid = cache.sample_point(start)
    if not valid:
        raise ValueError(f"native 3D Trace2CP start point is invalid: {start.tolist()}")
    initial_direction = _require_unit(
        initial_direction_zyx,
        label="native 3D Trace2CP initial_direction_zyx",
    )
    previous_direction = initial_direction.astype(np.float32, copy=False)
    candidates_unit = generate_cone_candidates(
        previous_direction,
        max_angle_degrees=cfg.cone_angle_degrees,
        grid_size=cfg.cone_grid_size,
    )
    trace: list[np.ndarray] = [start.astype(np.float32)]
    steps: list[NativeTraceStep] = []
    current = start.astype(np.float32)
    for _step_index in range(step_limit):
        if _step_index == 0:
            sampled_direction = initial_sampled_direction
            valid = True
        else:
            sampled_direction, _presence, valid = cache.sample_point(current)
        if not valid:
            emit_progress(current, _step_index, reason="invalid_current_point")
            return NativeTraceResult(
                trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                reached_target_plane=False,
                reason="invalid_current_point",
                steps=tuple(steps),
            )
        if _step_index == 0:
            current_direction = initial_direction.astype(np.float32, copy=False)
        else:
            current_direction = _align_axis(sampled_direction, previous_direction)
        candidates = generate_cone_candidates(
            current_direction,
            max_angle_degrees=cfg.cone_angle_degrees,
            grid_size=cfg.cone_grid_size,
        )
        if candidates.shape == candidates_unit.shape:
            candidates_unit = candidates
        next_points = current[None, :] + candidates_unit * np.float32(cfg.step_voxels)
        best_index, _total, direction_loss, presence_loss, rejected = _score_candidate_batch(
            cache,
            current_direction=current_direction,
            candidate_directions=candidates_unit,
            next_points=next_points,
        )
        if best_index is None:
            emit_progress(current, _step_index, reason="all_candidates_invalid")
            return NativeTraceResult(
                trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                reached_target_plane=False,
                reason="all_candidates_invalid",
                steps=tuple(steps),
            )
        chosen_direction = _align_axis(candidates_unit[int(best_index)], current_direction)
        next_point = (current + chosen_direction * np.float32(cfg.step_voxels)).astype(np.float32)
        crossing = _interpolate_plane_crossing(
            current,
            next_point,
            plane_point_zyx=target,
            plane_normal_zyx=plane_normal,
        )
        if crossing is not None:
            trace.append(crossing.astype(np.float32))
            steps.append(
                NativeTraceStep(
                    point_zyx=crossing.astype(np.float32),
                    direction_loss=float(direction_loss),
                    presence_loss=float(presence_loss),
                    total_loss=float(_total),
                    rejected_candidates=int(rejected),
                )
            )
            emit_progress(crossing, _step_index + 1, reason="target_plane")
            return NativeTraceResult(
                trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                reached_target_plane=True,
                reason="target_plane",
                steps=tuple(steps),
            )
        trace.append(next_point.astype(np.float32))
        steps.append(
            NativeTraceStep(
                point_zyx=next_point.astype(np.float32),
                direction_loss=float(direction_loss),
                presence_loss=float(presence_loss),
                total_loss=float(_total),
                rejected_candidates=int(rejected),
            )
        )
        previous_direction = chosen_direction.astype(np.float32)
        current = next_point
        emit_progress(current, _step_index + 1)
    emit_progress(current, progress_max, reason=limit_reason)
    return NativeTraceResult(
        trace_zyx=np.stack(trace, axis=0).astype(np.float32),
        reached_target_plane=False,
        reason=limit_reason,
        steps=tuple(steps),
    )


def _trace_progress(
    points_zyx: np.ndarray,
    *,
    start_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    span_voxels: float,
) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]
    return (
        (points - np.asarray(start_zyx, dtype=np.float32)[None, :])
        @ np.asarray(axis_zyx, dtype=np.float32)
    ) / np.float32(max(float(span_voxels), _EPS))


def _trace_point_at_progress(
    trace_zyx: np.ndarray,
    progress_value: float,
    *,
    start_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    span_voxels: float,
) -> np.ndarray | None:
    trace = np.asarray(trace_zyx, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] == 0:
        return None
    if trace.shape[0] == 1:
        progress = float(
            _trace_progress(
                trace[:1],
                start_zyx=start_zyx,
                axis_zyx=axis_zyx,
                span_voxels=span_voxels,
            )[0]
        )
        if abs(progress - float(progress_value)) <= 1.0e-5:
            return trace[0].astype(np.float32)
        return None
    progress = _trace_progress(
        trace,
        start_zyx=start_zyx,
        axis_zyx=axis_zyx,
        span_voxels=span_voxels,
    ).astype(np.float64, copy=False)
    target = float(progress_value)
    for idx in range(trace.shape[0] - 1):
        p0 = float(progress[idx])
        p1 = float(progress[idx + 1])
        if not (math.isfinite(p0) and math.isfinite(p1)):
            continue
        lo = min(p0, p1) - 1.0e-6
        hi = max(p0, p1) + 1.0e-6
        if target < lo or target > hi:
            continue
        denom = p1 - p0
        if abs(denom) <= 1.0e-12:
            alpha = 0.0
        else:
            alpha = float(np.clip((target - p0) / denom, 0.0, 1.0))
        return (
            trace[idx].astype(np.float64) * (1.0 - alpha)
            + trace[idx + 1].astype(np.float64) * alpha
        ).astype(np.float32)
    nearest_index = int(np.argmin(np.abs(progress - target)))
    if abs(float(progress[nearest_index]) - target) <= 1.0e-5:
        return trace[nearest_index].astype(np.float32)
    return None


def _resample_trace_between_progress(
    trace_zyx: np.ndarray,
    *,
    progress_start: float,
    progress_end: float,
    anchor_zyx: np.ndarray,
    start_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    span_voxels: float,
    step_voxels: float,
) -> np.ndarray:
    distance = abs(float(progress_end) - float(progress_start)) * max(float(span_voxels), 0.0)
    count = max(2, int(math.ceil(distance / max(float(step_voxels), _EPS))) + 1)
    sample_t = np.linspace(float(progress_start), float(progress_end), count, dtype=np.float32)
    points = [
        point
        for point in (
            _trace_point_at_progress(
                trace_zyx,
                float(t),
                start_zyx=start_zyx,
                axis_zyx=axis_zyx,
                span_voxels=span_voxels,
            )
            for t in sample_t
        )
        if point is not None
    ]
    anchor = np.asarray(anchor_zyx, dtype=np.float32)
    if not points:
        return anchor[None, :].astype(np.float32)
    out = np.stack(points, axis=0).astype(np.float32)
    if float(np.linalg.norm(out[0].astype(np.float64) - anchor.astype(np.float64))) > 1.0e-4:
        out = np.concatenate([anchor[None, :], out], axis=0)
    else:
        out[0] = anchor
    return out.astype(np.float32)


def _warp_partial_trace_to_meet(
    partial_zyx: np.ndarray,
    *,
    anchor_progress: float,
    meet_progress: float,
    anchor_zyx: np.ndarray,
    source_meet_zyx: np.ndarray,
    target_meet_zyx: np.ndarray,
    start_zyx: np.ndarray,
    axis_zyx: np.ndarray,
    span_voxels: float,
) -> np.ndarray:
    partial = np.asarray(partial_zyx, dtype=np.float32)
    if partial.ndim != 2 or partial.shape[1] != 3 or partial.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    warped = partial.astype(np.float64, copy=True)
    progress = _trace_progress(
        warped.astype(np.float32),
        start_zyx=start_zyx,
        axis_zyx=axis_zyx,
        span_voxels=span_voxels,
    ).astype(np.float64, copy=False)
    denom = abs(float(meet_progress) - float(anchor_progress))
    if denom <= 1.0e-8:
        blend = np.linspace(0.0, 1.0, warped.shape[0], dtype=np.float64)
    else:
        blend = np.clip(
            np.abs(progress - float(anchor_progress)) / denom,
            0.0,
            1.0,
        )
    delta = (
        np.asarray(target_meet_zyx, dtype=np.float64)
        - np.asarray(source_meet_zyx, dtype=np.float64)
    )
    warped += blend[:, None] * delta[None, :]
    warped[0] = np.asarray(anchor_zyx, dtype=np.float64)
    warped[-1] = np.asarray(target_meet_zyx, dtype=np.float64)
    return warped.astype(np.float32)


def _resample_polyline_by_arclength_zyx(
    points_zyx: np.ndarray,
    *,
    step_voxels: float,
) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_zyx must have shape N,3")
    if points.shape[0] <= 2:
        return points.astype(np.float32, copy=True)
    deltas = np.diff(points.astype(np.float64), axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    keep = np.concatenate([[True], lengths > 1.0e-8])
    points = points[keep]
    if points.shape[0] <= 2:
        return points.astype(np.float32, copy=True)
    lengths = np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 1.0e-8:
        return points[[0, -1]].astype(np.float32)
    count = max(2, int(math.ceil(total / max(float(step_voxels), _EPS))) + 1)
    sample_s = np.linspace(0.0, total, count, dtype=np.float64)
    return np.stack(
        [np.interp(sample_s, cumulative, points[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)


def fuse_forward_reverse_traces(
    forward_zyx: np.ndarray,
    reverse_zyx: np.ndarray,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    step_voxels: float,
) -> NativeTraceFusionResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    axis = _unit(target - start)
    span = float(np.linalg.norm(target - start))
    empty = np.zeros((0, 3), dtype=np.float32)
    empty_point = np.full((3,), np.nan, dtype=np.float32)
    if span <= _EPS:
        return NativeTraceFusionResult(
            fused_zyx=empty,
            closest_progress=float("nan"),
            raw_gap_voxels=float("nan"),
            considered_gap_voxels=float("nan"),
            center_penalty=float("nan"),
            closest_midpoint_zyx=empty_point,
            closest_forward_zyx=empty_point,
            closest_reverse_zyx=empty_point,
            reached_overlap=False,
            reason="degenerate_cp_span",
        )
    forward = np.asarray(forward_zyx, dtype=np.float32)
    reverse = np.asarray(reverse_zyx, dtype=np.float32)
    if (
        forward.ndim != 2
        or reverse.ndim != 2
        or forward.shape[1:] != (3,)
        or reverse.shape[1:] != (3,)
        or forward.shape[0] < 1
        or reverse.shape[0] < 1
    ):
        return NativeTraceFusionResult(
            fused_zyx=empty,
            closest_progress=float("nan"),
            raw_gap_voxels=float("nan"),
            considered_gap_voxels=float("nan"),
            center_penalty=float("nan"),
            closest_midpoint_zyx=empty_point,
            closest_forward_zyx=empty_point,
            closest_reverse_zyx=empty_point,
            reached_overlap=False,
            reason="invalid_trace_shape",
        )
    forward_progress = _trace_progress(
        forward,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
    )
    reverse_progress = _trace_progress(
        reverse,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
    )
    finite_forward = forward_progress[np.isfinite(forward_progress)]
    finite_reverse = reverse_progress[np.isfinite(reverse_progress)]
    if finite_forward.size == 0 or finite_reverse.size == 0:
        return NativeTraceFusionResult(
            fused_zyx=empty,
            closest_progress=float("nan"),
            raw_gap_voxels=float("nan"),
            considered_gap_voxels=float("nan"),
            center_penalty=float("nan"),
            closest_midpoint_zyx=empty_point,
            closest_forward_zyx=empty_point,
            closest_reverse_zyx=empty_point,
            reached_overlap=False,
            reason="nonfinite_trace_progress",
        )
    overlap_lo = max(0.0, float(np.min(finite_forward)), float(np.min(finite_reverse)))
    overlap_hi = min(1.0, float(np.max(finite_forward)), float(np.max(finite_reverse)))
    if overlap_hi < overlap_lo - 1.0e-6:
        return NativeTraceFusionResult(
            fused_zyx=empty,
            closest_progress=float("nan"),
            raw_gap_voxels=float("nan"),
            considered_gap_voxels=float("nan"),
            center_penalty=float("nan"),
            closest_midpoint_zyx=empty_point,
            closest_forward_zyx=empty_point,
            closest_reverse_zyx=empty_point,
            reached_overlap=False,
            reason="no_trace_overlap",
        )
    overlap_lo = float(np.clip(overlap_lo, 0.0, 1.0))
    overlap_hi = float(np.clip(overlap_hi, 0.0, 1.0))
    sample_count = max(
        2,
        int(math.ceil(max(0.0, overlap_hi - overlap_lo) * span / max(float(step_voxels), _EPS)))
        + 1,
    )
    candidate_progress = np.linspace(overlap_lo, overlap_hi, sample_count, dtype=np.float32)
    best: tuple[float, float, float, np.ndarray, np.ndarray] | None = None
    for progress_value in candidate_progress:
        forward_point = _trace_point_at_progress(
            forward,
            float(progress_value),
            start_zyx=start,
            axis_zyx=axis,
            span_voxels=span,
        )
        reverse_point = _trace_point_at_progress(
            reverse,
            float(progress_value),
            start_zyx=start,
            axis_zyx=axis,
            span_voxels=span,
        )
        if forward_point is None or reverse_point is None:
            continue
        gap = float(np.linalg.norm(forward_point.astype(np.float64) - reverse_point.astype(np.float64)))
        center_penalty = 1.0 + float(np.clip(abs(float(progress_value) - 0.5) / 0.5, 0.0, 1.0))
        considered = gap * center_penalty
        center_tie = abs(float(progress_value) - 0.5)
        key = (considered, center_tie)
        if best is None or key < (best[0], best[1]):
            best = (considered, center_tie, float(progress_value), forward_point, reverse_point)
    if best is None:
        return NativeTraceFusionResult(
            fused_zyx=empty,
            closest_progress=float("nan"),
            raw_gap_voxels=float("nan"),
            considered_gap_voxels=float("nan"),
            center_penalty=float("nan"),
            closest_midpoint_zyx=empty_point,
            closest_forward_zyx=empty_point,
            closest_reverse_zyx=empty_point,
            reached_overlap=False,
            reason="no_interpolated_overlap",
        )
    considered_gap, _center_tie, closest_progress, closest_forward, closest_reverse = best
    raw_gap = float(
        np.linalg.norm(closest_forward.astype(np.float64) - closest_reverse.astype(np.float64))
    )
    center_penalty = float(considered_gap / max(raw_gap, _EPS)) if raw_gap > _EPS else (
        1.0 + float(np.clip(abs(float(closest_progress) - 0.5) / 0.5, 0.0, 1.0))
    )
    midpoint = ((closest_forward.astype(np.float64) + closest_reverse.astype(np.float64)) * 0.5).astype(np.float32)
    forward_partial = _resample_trace_between_progress(
        forward,
        progress_start=0.0,
        progress_end=closest_progress,
        anchor_zyx=start,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
        step_voxels=step_voxels,
    )
    reverse_partial = _resample_trace_between_progress(
        reverse,
        progress_start=1.0,
        progress_end=closest_progress,
        anchor_zyx=target,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
        step_voxels=step_voxels,
    )
    forward_warped = _warp_partial_trace_to_meet(
        forward_partial,
        anchor_progress=0.0,
        meet_progress=closest_progress,
        anchor_zyx=start,
        source_meet_zyx=closest_forward,
        target_meet_zyx=midpoint,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
    )
    reverse_warped = _warp_partial_trace_to_meet(
        reverse_partial,
        anchor_progress=1.0,
        meet_progress=closest_progress,
        anchor_zyx=target,
        source_meet_zyx=closest_reverse,
        target_meet_zyx=midpoint,
        start_zyx=start,
        axis_zyx=axis,
        span_voxels=span,
    )
    reverse_meet_to_target = reverse_warped[::-1].copy()
    if forward_warped.shape[0] == 0:
        fused_dense = reverse_meet_to_target
    elif reverse_meet_to_target.shape[0] == 0:
        fused_dense = forward_warped
    else:
        fused_dense = np.concatenate([forward_warped, reverse_meet_to_target[1:]], axis=0)
    if fused_dense.shape[0] >= 1:
        fused_dense[0] = start
        fused_dense[-1] = target
    fused = _resample_polyline_by_arclength_zyx(
        fused_dense,
        step_voxels=step_voxels,
    )
    if fused.shape[0] >= 1:
        fused[0] = start
        fused[-1] = target
    return NativeTraceFusionResult(
        fused_zyx=fused.astype(np.float32),
        closest_progress=float(closest_progress),
        raw_gap_voxels=float(raw_gap),
        considered_gap_voxels=float(considered_gap),
        center_penalty=float(center_penalty),
        closest_midpoint_zyx=midpoint.astype(np.float32),
        closest_forward_zyx=closest_forward.astype(np.float32),
        closest_reverse_zyx=closest_reverse.astype(np.float32),
        reached_overlap=True,
        reason="closest_overlap",
    )


def trace_native_3d_pair(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    forward_initial_direction_zyx: np.ndarray,
    reverse_initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    progress: bool = False,
) -> NativeTracePairResult:
    forward = trace_native_3d_one_way(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        initial_direction_zyx=forward_initial_direction_zyx,
        cfg=cfg,
        progress_label="fw" if progress else None,
    )
    reverse = trace_native_3d_one_way(
        cache,
        start_zyx=target_zyx,
        target_zyx=start_zyx,
        initial_direction_zyx=reverse_initial_direction_zyx,
        cfg=cfg,
        progress_label="bw" if progress else None,
    )
    fusion = fuse_forward_reverse_traces(
        forward.trace_zyx,
        reverse.trace_zyx,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        step_voxels=float(cfg.step_voxels),
    )
    span = float(np.linalg.norm(np.asarray(target_zyx, dtype=np.float32) - np.asarray(start_zyx, dtype=np.float32)))
    normal = _unit(np.asarray(target_zyx, dtype=np.float32) - np.asarray(start_zyx, dtype=np.float32))
    forward_plane = abs(_plane_distance(forward.trace_zyx[-1], target_zyx, normal))
    reverse_plane = abs(_plane_distance(reverse.trace_zyx[-1], start_zyx, -normal))
    plane_error = (forward_plane + reverse_plane) * 0.5 / max(span, _EPS)
    closest_forward = float(np.min(np.linalg.norm(forward.trace_zyx - target_zyx[None, :], axis=1)))
    closest_reverse = float(np.min(np.linalg.norm(reverse.trace_zyx - start_zyx[None, :], axis=1)))
    closest_error = (closest_forward + closest_reverse) * 0.5 / max(span, _EPS)
    return NativeTracePairResult(
        forward=forward,
        reverse=reverse,
        fusion=fusion,
        fused_zyx=fusion.fused_zyx,
        plane_error=float(plane_error),
        closest_target_error=float(closest_error),
        span_voxels=float(span),
    )


def _image_to_u8(
    image: np.ndarray,
    valid: np.ndarray,
    *,
    normalization: str,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if not bool(mask.any()):
        return out
    norm_t = _normalize_image(
        torch.as_tensor(arr, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        normalization,
    )
    norm = norm_t.detach().cpu().numpy().astype(np.float32, copy=False)
    mode = str(normalization).lower()
    if mode == "zscore":
        scaled = (np.clip(norm, -3.0, 3.0) + 3.0) * (255.0 / 6.0)
    elif mode == "minmax":
        scaled = np.clip(norm, 0.0, 1.0) * 255.0
    elif mode in {"none", "raw", "identity"}:
        scaled = np.clip(norm, 0.0, 255.0)
    else:
        raise ValueError(f"unsupported image_normalization {normalization!r}")
    out[mask] = np.rint(scaled[mask]).astype(np.uint8)
    return out


def _presence_to_u8(presence: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(presence, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    if bool(mask.any()):
        out[mask] = np.rint(np.clip(arr[mask], 0.0, 1.0) * 255.0).astype(np.uint8)
    return out


def _as_numpy_array(value: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(dtype, copy=False)
    return np.asarray(value, dtype=dtype)


def _sample_presence_on_strip(
    cache: NativeTraceFieldCache,
    coords_xyz_base: np.ndarray,
    grid_valid: np.ndarray,
    *,
    spacing_base: float,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords_xyz_base, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords_xyz_base must have shape H,W,3")
    valid = np.asarray(grid_valid, dtype=bool)
    if valid.shape != coords.shape[:2]:
        raise ValueError(
            "grid_valid shape must match coords: "
            f"valid={valid.shape} coords={coords.shape[:2]}"
        )
    spacing = float(spacing_base)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid spacing_base {spacing_base!r}")

    flat_coords = coords.reshape(-1, 3)
    flat_valid = valid.reshape(-1) & np.isfinite(flat_coords).all(axis=1)
    presence = np.zeros((flat_coords.shape[0],), dtype=np.float32)
    out_valid = np.zeros((flat_coords.shape[0],), dtype=bool)
    if bool(np.any(flat_valid)):
        points_zyx_selected = (
            flat_coords[flat_valid][:, [2, 1, 0]].astype(np.float32, copy=False)
            / np.float32(spacing)
        )
        _directions, sampled_presence, sampled_valid = cache.sample_points_torch(
            points_zyx_selected,
            progress_label=progress_label,
        )
        presence_values = sampled_presence.detach().cpu().numpy().astype(np.float32, copy=False)
        valid_values = sampled_valid.detach().cpu().numpy().astype(bool, copy=False)
        flat_indices = np.flatnonzero(flat_valid)
        presence[flat_indices] = presence_values
        out_valid[flat_indices] = valid_values
    return (
        presence.reshape(coords.shape[:2]).astype(np.float32, copy=False),
        out_valid.reshape(coords.shape[:2]),
    )


def _closest_source_line_projection(
    trace_xyz_base: np.ndarray,
    source: _Trace2CpSegmentSource,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = np.asarray(trace_xyz_base, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("trace_xyz_base must have shape N,3")
    line_xyz = np.asarray(source.line_window.line_points_xyz, dtype=np.float32)
    line_xy = np.asarray(source.line_xy, dtype=np.float32)
    if line_xyz.ndim != 2 or line_xyz.shape[1] != 3 or int(line_xyz.shape[0]) < 2:
        raise ValueError("source line must have at least two XYZ points")
    if line_xy.ndim != 2 or line_xy.shape[1] != 2 or line_xy.shape[0] != line_xyz.shape[0]:
        raise ValueError("source line_xy must match source line points")

    seg_start = line_xyz[:-1]
    seg_vec = line_xyz[1:] - line_xyz[:-1]
    seg_xy_start = line_xy[:-1]
    seg_xy_vec = line_xy[1:] - line_xy[:-1]
    seg_len2 = np.sum(seg_vec * seg_vec, axis=1)
    seg_valid = np.isfinite(seg_len2) & (seg_len2 > np.float32(_EPS))
    seg_valid &= np.isfinite(seg_start).all(axis=1) & np.isfinite(seg_vec).all(axis=1)
    seg_valid &= np.isfinite(seg_xy_start).all(axis=1) & np.isfinite(seg_xy_vec).all(axis=1)
    if not bool(np.any(seg_valid)):
        raise ValueError("source line has no finite non-degenerate segments")

    seg_start = seg_start[seg_valid]
    seg_vec = seg_vec[seg_valid]
    seg_xy_start = seg_xy_start[seg_valid]
    seg_xy_vec = seg_xy_vec[seg_valid]
    seg_len2 = seg_len2[seg_valid]

    projected_xyz = np.full_like(trace, np.nan, dtype=np.float32)
    projected_xy = np.full((trace.shape[0], 2), np.nan, dtype=np.float32)
    projected_valid = np.isfinite(trace).all(axis=1)
    chunk = 512
    for start in range(0, int(trace.shape[0]), chunk):
        stop = min(int(trace.shape[0]), start + chunk)
        points = trace[start:stop]
        finite = projected_valid[start:stop]
        if not bool(np.any(finite)):
            continue
        diff = points[:, None, :] - seg_start[None, :, :]
        t = np.sum(diff * seg_vec[None, :, :], axis=2) / seg_len2[None, :]
        t = np.clip(t, 0.0, 1.0)
        closest = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]
        dist2 = np.sum((points[:, None, :] - closest) ** 2, axis=2)
        dist2[~finite, :] = np.inf
        best = np.argmin(dist2, axis=1)
        best_t = t[np.arange(stop - start), best].astype(np.float32, copy=False)
        projected_xyz[start:stop] = (
            seg_start[best] + best_t[:, None] * seg_vec[best]
        ).astype(np.float32, copy=False)
        projected_xy[start:stop] = (
            seg_xy_start[best] + best_t[:, None] * seg_xy_vec[best]
        ).astype(np.float32, copy=False)
        projected_valid[start:stop] &= np.isfinite(dist2[np.arange(stop - start), best])
    return projected_xyz, projected_xy, projected_valid


def _sample_source_axes_at_xy(
    source: _Trace2CpSegmentSource,
    axis_name: str,
    points_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    axes_value = getattr(source.grid, axis_name)
    if axes_value is None:
        raise ValueError(f"source grid missing {axis_name}")
    axes = _as_numpy_array(axes_value, dtype=np.float32)
    valid = _as_numpy_array(source.grid.valid_mask, dtype=bool)
    points = np.asarray(points_xy, dtype=np.float32)
    if axes.ndim != 3 or axes.shape[2] != 3:
        raise ValueError(f"{axis_name} must have shape H,W,3")
    if valid.shape != axes.shape[:2]:
        raise ValueError("source grid valid mask shape does not match axis grid")
    height, width = axes.shape[:2]
    finite_points = np.isfinite(points).all(axis=1)
    x = np.zeros((points.shape[0],), dtype=np.int64)
    y = np.zeros((points.shape[0],), dtype=np.int64)
    if bool(np.any(finite_points)):
        x[finite_points] = np.rint(points[finite_points, 0]).astype(np.int64)
        y[finite_points] = np.rint(points[finite_points, 1]).astype(np.int64)
    in_bounds = finite_points & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled = np.zeros((points.shape[0], 3), dtype=np.float32)
    sampled_valid = np.zeros((points.shape[0],), dtype=bool)
    if bool(np.any(in_bounds)):
        indices = np.flatnonzero(in_bounds)
        sampled[indices] = axes[y[indices], x[indices]]
        sampled_valid[indices] = valid[y[indices], x[indices]]
    norms = np.linalg.norm(sampled, axis=1)
    finite = np.isfinite(sampled).all(axis=1) & np.isfinite(norms) & (norms > np.float32(_EPS))
    ok = sampled_valid & finite
    sampled[ok] = sampled[ok] / norms[ok, None].astype(np.float32)
    sampled[~ok] = 0.0
    return sampled, ok


def _project_trace_to_initial_strip(
    source: _Trace2CpSegmentSource,
    trace_xyz_base: np.ndarray,
    *,
    axis_name: str,
) -> np.ndarray:
    trace = np.asarray(trace_xyz_base, dtype=np.float32)
    if int(trace.shape[0]) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    projected_xyz, projected_xy, projected_valid = _closest_source_line_projection(trace, source)
    axes, axes_valid = _sample_source_axes_at_xy(source, axis_name, projected_xy)
    spacing = float(source.record.volume_spacing_base)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume spacing for native trace projection: {spacing}")
    offsets = np.sum((trace - projected_xyz) * axes, axis=1) / np.float32(spacing)
    xy = projected_xy.copy()
    xy[:, 1] += offsets.astype(np.float32, copy=False)
    valid = projected_valid & axes_valid & np.isfinite(xy).all(axis=1)
    return xy[valid].astype(np.float32, copy=False)


def _volume_trace_to_source_trace_xyz(
    source: _Trace2CpSegmentSource,
    trace_xyz_base: np.ndarray,
) -> np.ndarray:
    trace = np.asarray(trace_xyz_base, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3:
        raise ValueError("trace_xyz_base must have shape N,3")
    if int(trace.shape[0]) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    projected_xyz, projected_xy, projected_valid = _closest_source_line_projection(trace, source)
    row_axes, row_axes_valid = _sample_source_axes_at_xy(
        source,
        "offset_axis_xyz",
        projected_xy,
    )
    side_axes, side_axes_valid = _sample_source_axes_at_xy(
        source,
        "side_axis_xyz",
        projected_xy,
    )
    spacing = float(source.record.volume_spacing_base)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume spacing for native trace source conversion: {spacing}")
    delta = trace - projected_xyz
    y_offsets = np.sum(delta * row_axes, axis=1) / np.float32(spacing)
    z_offsets = np.sum(delta * side_axes, axis=1) / np.float32(spacing)
    source_trace = np.stack(
        [
            projected_xy[:, 0],
            projected_xy[:, 1] + y_offsets.astype(np.float32, copy=False),
            z_offsets.astype(np.float32, copy=False),
        ],
        axis=1,
    )
    valid = (
        projected_valid
        & row_axes_valid
        & side_axes_valid
        & np.isfinite(source_trace).all(axis=1)
    )
    source_trace = source_trace[valid].astype(np.float32, copy=False)
    if int(source_trace.shape[0]) < 2:
        raise ValueError(
            "native fused trace cannot be converted into source-strip coordinates: "
            f"valid_points={int(source_trace.shape[0])} total_points={int(trace.shape[0])}"
        )
    return source_trace


def _trace_overlays_for_view(
    source: _Trace2CpSegmentSource,
    result: NativeTracePairResult,
    *,
    axis_name: str,
) -> tuple[tuple[np.ndarray, tuple[int, int, int, int]], ...]:
    spacing = float(source.record.volume_spacing_base)
    traces = (
        (
            _trace_zyx_to_base_xyz(result.forward.trace_zyx, spacing),
            (64, 170, 255, 220),
        ),
        (
            _trace_zyx_to_base_xyz(result.reverse.trace_zyx, spacing),
            (255, 80, 220, 220),
        ),
        (
            _trace_zyx_to_base_xyz(result.fused_zyx, spacing),
            (255, 220, 0, 235),
        ),
    )
    overlays: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for trace_xyz, color in traces:
        xy = _project_trace_to_initial_strip(source, trace_xyz, axis_name=axis_name)
        if int(xy.shape[0]) >= 2:
            overlays.append((xy, color))
    return tuple(overlays)


def _adaptive_trace2cp_cross_strip_height(
    max_height: int,
    overlay_groups: tuple[tuple[tuple[np.ndarray, tuple[int, int, int, int]], ...], ...],
    *,
    expansion: float = 1.5,
    padding_px: float = 2.0,
) -> int:
    configured = int(max_height)
    if configured <= 0:
        raise ValueError(f"invalid maximum trace2cp strip height {max_height!r}")
    if configured == 1:
        return 1
    center_y = (float(configured) - 1.0) * 0.5
    required_half = 0.0
    for overlays in overlay_groups:
        for overlay_xy, _color in overlays:
            overlay = np.asarray(overlay_xy, dtype=np.float32)
            if overlay.ndim != 2 or overlay.shape[1] < 2:
                continue
            finite = np.isfinite(overlay[:, 1])
            if not bool(np.any(finite)):
                continue
            required_half = max(
                required_half,
                float(np.max(np.abs(overlay[finite, 1].astype(np.float64) - center_y))),
            )
    half = int(math.ceil(required_half * float(expansion) + float(padding_px)))
    half = max(1, half)
    max_half = max(1, (configured - 1) // 2)
    half = min(half, max_half)
    return int(2 * half + 1)


def _draw_trace_panel(
    image_u8: np.ndarray,
    valid: np.ndarray,
    line_xy: np.ndarray,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    *,
    title: str,
    overlays: tuple[tuple[np.ndarray, tuple[int, int, int, int]], ...] = (),
    line_width: int = 2,
    overlay_width: int = 2,
):
    from PIL import Image, ImageDraw

    img = np.asarray(image_u8, dtype=np.uint8)
    if img.ndim == 2:
        base = np.repeat(img[..., None], 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        base = img
    else:
        raise ValueError("image_u8 must have shape H,W or H,W,3")
    mask = np.asarray(valid, dtype=bool)
    if mask.shape == base.shape[:2]:
        base = base.copy()
        base[~mask] = 0
    canvas = Image.fromarray(base, mode="RGB").convert("RGBA")
    text_pad = 24
    padded = Image.new("RGBA", (canvas.width, canvas.height + text_pad), (0, 0, 0, 255))
    padded.alpha_composite(canvas, (0, text_pad))
    draw = ImageDraw.Draw(padded, "RGBA")
    draw.text((4, 4), title, fill=(255, 255, 255, 255))
    line = np.asarray(line_xy, dtype=np.float32)
    pts = [(float(x), float(y) + text_pad) for x, y in line if np.isfinite(x) and np.isfinite(y)]
    if len(pts) >= 2:
        draw.line(pts, fill=(0, 255, 128, 170), width=max(1, int(line_width)))
    for overlay_xy, color in overlays:
        overlay = np.asarray(overlay_xy, dtype=np.float32)
        overlay_pts = [
            (float(x), float(y) + text_pad)
            for x, y in overlay
            if np.isfinite(x) and np.isfinite(y)
        ]
        if len(overlay_pts) >= 2:
            draw.line(overlay_pts, fill=color, width=max(1, int(overlay_width)))
    for xy, color in (
        (start_xy, (0, 255, 255, 255)),
        (target_xy, (255, 64, 220, 255)),
    ):
        x, y = (float(v) for v in xy)
        draw.ellipse((x - 4, y + text_pad - 4, x + 4, y + text_pad + 4), outline=color, width=2)
    return padded


def _trace_zyx_to_base_xyz(trace_zyx: np.ndarray, spacing_base: float) -> np.ndarray:
    trace = np.asarray(trace_zyx, dtype=np.float32) * np.float32(spacing_base)
    return trace[:, [2, 1, 0]].astype(np.float32, copy=False)


def _make_native_trace_visualization(
    geometry_loader: Any,
    source: _Trace2CpSegmentSource,
    result: NativeTracePairResult,
    *,
    cache: NativeTraceFieldCache,
    image_normalization: str,
    partial_output_path: Path | None = None,
):
    from PIL import Image, ImageDraw

    progress_start = time.perf_counter()
    has_fused_trace = bool(np.asarray(result.fused_zyx).ndim == 2 and result.fused_zyx.shape[0] >= 2)
    progress_total = 16 if has_fused_trace else 8
    progress_step = 0
    panel_rows: list[list[Any | None]] = []

    def run_stage(label: str, fn: Any):
        nonlocal progress_step
        print(
            f"native strip render start stage={progress_step + 1}/{progress_total} {label}",
            flush=True,
        )
        write_partial(f"stage_start={label}")
        stage_start = time.perf_counter()
        result_value = fn()
        progress_step += 1
        _emit_native_progress(
            "strip render",
            progress_step,
            progress_total,
            progress_start,
            detail=f"stage={label} stage_ms={(time.perf_counter() - stage_start) * 1000.0:.1f}",
        )
        write_partial(f"stage_done={label}")
        return result_value

    def compose_panel_rows(rows: list[list[Any | None]], *, status_text: str | None = None):
        left_panels = [row[0] for row in rows if row[0] is not None]
        right_panels = [row[1] for row in rows if row[1] is not None]
        if not left_panels and not right_panels:
            sheet = Image.new("RGBA", (720, 96), (0, 0, 0, 255))
            draw = ImageDraw.Draw(sheet, "RGBA")
            draw.text((8, 8), "native 3D Trace2CP render", fill=(255, 255, 255, 255))
            draw.text(
                (8, 34),
                "waiting for first panel",
                fill=(180, 180, 180, 255),
            )
            if status_text:
                draw.text((8, 60), status_text, fill=(120, 220, 255, 255))
            return sheet
        left_width = max((panel.width for panel in left_panels), default=0)
        right_width = max((panel.width for panel in right_panels), default=0)
        row_heights = [
            max(
                row[0].height if row[0] is not None else 0,
                row[1].height if row[1] is not None else 0,
            )
            for row in rows
        ]
        sheet = Image.new(
            "RGBA",
            (max(1, left_width + right_width), max(1, int(sum(row_heights)))),
            (0, 0, 0, 255),
        )
        y = 0
        for row_height, row in zip(row_heights, rows):
            left, right = row
            if left is not None:
                sheet.alpha_composite(left, (0, y))
            if right is not None:
                sheet.alpha_composite(right, (left_width, y))
            y += int(row_height)
        return sheet

    def write_partial(label: str) -> None:
        if partial_output_path is None:
            return
        path = Path(partial_output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        compose_panel_rows(
            panel_rows,
            status_text=f"{progress_step}/{progress_total} {label}",
        ).convert("RGB").save(path, quality=90)
        print(f"native strip render partial={path} {label}", flush=True)

    def add_panel(row_index: int, column_index: int, panel: Any, label: str) -> None:
        while len(panel_rows) <= int(row_index):
            panel_rows.append([None, None])
        panel_rows[int(row_index)][int(column_index)] = panel
        write_partial(f"panel={label}")

    spacing = float(source.record.volume_spacing_base)

    original_side_overlays, original_top_overlays = run_stage(
        "original-trace-overlays",
        lambda: (
            _trace_overlays_for_view(source, result, axis_name="offset_axis_xyz"),
            _trace_overlays_for_view(source, result, axis_name="side_axis_xyz"),
        ),
    )
    fused_source = None
    if has_fused_trace:
        fused_trace_xyz = _trace_zyx_to_base_xyz(result.fused_zyx, spacing)
        fused_source_trace = run_stage(
            "fused-source-trace",
            lambda: _volume_trace_to_source_trace_xyz(source, fused_trace_xyz),
        )
        fused_source = run_stage(
            "fused-source",
            lambda: geometry_loader.build_trace2cp_refined_segment_source(
                source,
                fused_source_trace,
                device=torch.device("cpu"),
            ),
        )
    else:
        print(
            "native strip render skipped fused panels "
            f"fusion_reason={result.fusion.reason}",
            flush=True,
        )

    _sample, side_image, side_valid = run_stage(
        "original-side-volume",
        lambda: geometry_loader.sample_trace2cp_segment_source(source),
    )
    add_panel(
        0,
        0,
        _draw_trace_panel(
            _image_to_u8(side_image, side_valid, normalization=image_normalization),
            side_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"initial side input ({image_normalization})",
            overlays=original_side_overlays,
        ),
        "initial_side_input",
    )
    top_image, top_valid = run_stage(
        "original-top-volume",
        lambda: geometry_loader.sample_trace2cp_top_strip_source(source),
    )
    add_panel(
        1,
        0,
        _draw_trace_panel(
            _image_to_u8(top_image, top_valid, normalization=image_normalization),
            top_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"initial top input ({image_normalization})",
            overlays=original_top_overlays,
        ),
        "initial_top_input",
    )
    side_coords_xyz, side_grid_valid = run_stage(
        "original-side-coords",
        lambda: geometry_loader.trace2cp_segment_coords_xyz(source),
    )
    top_coords_xyz, top_grid_valid = run_stage(
        "original-top-coords",
        lambda: geometry_loader.trace2cp_top_strip_coords_xyz(source),
    )
    side_presence, side_presence_valid = run_stage(
        "original-side-presence",
        lambda: _sample_presence_on_strip(
            cache,
            side_coords_xyz,
            np.asarray(side_grid_valid, dtype=bool) & np.asarray(side_valid, dtype=bool),
            spacing_base=spacing,
            progress_label="side",
        ),
    )
    add_panel(
        0,
        1,
        _draw_trace_panel(
            _presence_to_u8(side_presence, side_presence_valid),
            side_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title="initial side 3D presence",
            overlays=original_side_overlays,
        ),
        "initial_side_presence",
    )
    top_presence, top_presence_valid = run_stage(
        "original-top-presence",
        lambda: _sample_presence_on_strip(
            cache,
            top_coords_xyz,
            np.asarray(top_grid_valid, dtype=bool) & np.asarray(top_valid, dtype=bool),
            spacing_base=spacing,
            progress_label="top",
        ),
    )
    add_panel(
        1,
        1,
        _draw_trace_panel(
            _presence_to_u8(top_presence, top_presence_valid),
            top_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title="initial top 3D presence",
            overlays=original_top_overlays,
        ),
        "initial_top_presence",
    )

    if fused_source is not None:
        _fused_sample, fused_side_image, fused_side_valid = run_stage(
            "fused-side-volume",
            lambda: geometry_loader.sample_trace2cp_segment_source(fused_source),
        )
        add_panel(
            2,
            0,
            _draw_trace_panel(
                _image_to_u8(fused_side_image, fused_side_valid, normalization=image_normalization),
                fused_side_valid,
                fused_source.line_xy,
                fused_source.start_control_point_xy,
                fused_source.target_control_point_xy,
                title=f"fused side input ({image_normalization})",
                line_width=1,
            ),
            "fused_side_input",
        )
        fused_top_image, fused_top_valid = run_stage(
            "fused-top-volume",
            lambda: geometry_loader.sample_trace2cp_top_strip_source(fused_source),
        )
        add_panel(
            3,
            0,
            _draw_trace_panel(
                _image_to_u8(fused_top_image, fused_top_valid, normalization=image_normalization),
                fused_top_valid,
                fused_source.line_xy,
                fused_source.start_control_point_xy,
                fused_source.target_control_point_xy,
                title=f"fused top input ({image_normalization})",
                line_width=1,
            ),
            "fused_top_input",
        )
        fused_side_coords_xyz, fused_side_grid_valid = run_stage(
            "fused-side-coords",
            lambda: geometry_loader.trace2cp_segment_coords_xyz(fused_source),
        )
        fused_top_coords_xyz, fused_top_grid_valid = run_stage(
            "fused-top-coords",
            lambda: geometry_loader.trace2cp_top_strip_coords_xyz(fused_source),
        )
        fused_side_presence, fused_side_presence_valid = run_stage(
            "fused-side-presence",
            lambda: _sample_presence_on_strip(
                cache,
                fused_side_coords_xyz,
                np.asarray(fused_side_grid_valid, dtype=bool)
                & np.asarray(fused_side_valid, dtype=bool),
                spacing_base=spacing,
                progress_label="fused-side",
            ),
        )
        add_panel(
            2,
            1,
            _draw_trace_panel(
                _presence_to_u8(fused_side_presence, fused_side_presence_valid),
                fused_side_presence_valid,
                fused_source.line_xy,
                fused_source.start_control_point_xy,
                fused_source.target_control_point_xy,
                title="fused side 3D presence",
                line_width=1,
            ),
            "fused_side_presence",
        )
        fused_top_presence, fused_top_presence_valid = run_stage(
            "fused-top-presence",
            lambda: _sample_presence_on_strip(
                cache,
                fused_top_coords_xyz,
                np.asarray(fused_top_grid_valid, dtype=bool)
                & np.asarray(fused_top_valid, dtype=bool),
                spacing_base=spacing,
                progress_label="fused-top",
            ),
        )
        add_panel(
            3,
            1,
            _draw_trace_panel(
                _presence_to_u8(fused_top_presence, fused_top_presence_valid),
                fused_top_presence_valid,
                fused_source.line_xy,
                fused_source.start_control_point_xy,
                fused_source.target_control_point_xy,
                title="fused top 3D presence",
                line_width=1,
            ),
            "fused_top_presence",
        )

    def compose_sheet():
        return compose_panel_rows(panel_rows)

    return run_stage("compose", compose_sheet)


def _tool_raw_config(raw_config: dict[str, Any], *, fiber_json: Path | None) -> dict[str, Any]:
    source_datasets = raw_config.get("test_datasets") or raw_config.get("datasets")
    if not isinstance(source_datasets, list) or not source_datasets:
        raise ValueError("native 3D Trace2CP requires datasets or test_datasets")
    datasets = [dict(entry) for entry in source_datasets]
    if fiber_json is not None:
        if len(datasets) != 1:
            raise ValueError("--fiber-json requires exactly one dataset or test_datasets entry")
        dataset = dict(datasets[0])
        dataset.pop("fiber_glob", None)
        dataset["fiber_paths"] = [str(fiber_json)]
        datasets = [dataset]
    patched = dict(raw_config)
    patched["datasets"] = datasets
    patched["test_datasets"] = datasets
    return patched


def _load_tool_config(
    config_path: str | Path,
    raw_config: dict[str, Any],
    *,
    fiber_json: Path | None,
) -> FiberTrace3DConfig:
    patched = _tool_raw_config(raw_config, fiber_json=fiber_json)
    base = load_config(config_path)
    return replace(base, datasets=tuple(dict(entry) for entry in patched["datasets"]))


def _native_trace2cp_geometry_config(raw_config: dict[str, Any]):
    patched = dict(raw_config)
    training = dict(raw_config.get("training", {}))
    training["test_trace2cp_enabled"] = False
    patched["training"] = training
    return _trace2cp_3d_config(patched)


def _resolve_native_trace2cp_selection(
    loader: Any,
    *,
    sample_index: int,
    fiber_json: Path | None,
    start_cp_index: int | None,
    target_cp_index: int | None,
    target_offset: int,
    sample_mode: str | None,
) -> _NativeTrace2CpSelection:
    explicit_segment = start_cp_index is not None or target_cp_index is not None
    if explicit_segment:
        if start_cp_index is None or target_cp_index is None:
            raise ValueError("--start-cp-index and --target-cp-index must be provided together")
        if fiber_json is None:
            raise ValueError("--start-cp-index/--target-cp-index require --fiber-json")
        if sample_mode is not None and str(sample_mode) != "flat":
            raise ValueError("explicit CP segment selection requires --sample-mode flat or omitted")
        records = getattr(loader, "records", ())
        if len(records) != 1:
            raise ValueError("explicit CP segment selection requires exactly one loaded fiber")
        record = records[0]
        record_index = 0
        mode = "flat"
        resolved_start_cp_index = int(start_cp_index)
        resolved_target_cp_index = int(target_cp_index)
        selected_sample_index = int(resolved_start_cp_index)
    else:
        mode = ("flat" if fiber_json is not None else "random") if sample_mode is None else str(sample_mode)
        record, record_index, resolved_start_cp_index = loader.descriptor_for_sample_index(
            int(sample_index),
            sample_mode=mode,
        )
        selected_sample_index = int(sample_index)
        resolved_target_cp_index = int(resolved_start_cp_index) + int(target_offset)

    cp_count = int(record.fiber.control_points_zyx.shape[0])
    if resolved_start_cp_index < 0 or resolved_start_cp_index >= cp_count:
        raise ValueError(
            f"start CP index {resolved_start_cp_index} out of range for {cp_count} control points"
        )
    if resolved_target_cp_index < 0 or resolved_target_cp_index >= cp_count:
        raise ValueError(
            f"target CP index {resolved_target_cp_index} out of range for {cp_count} control points"
        )
    if int(resolved_start_cp_index) == int(resolved_target_cp_index):
        raise ValueError("native Trace2CP start and target CP indices must differ")

    return _NativeTrace2CpSelection(
        record=record,
        record_index=int(record_index),
        sample_index=int(selected_sample_index),
        sample_mode=mode,
        start_cp_index=int(resolved_start_cp_index),
        target_cp_index=int(resolved_target_cp_index),
        explicit_segment=bool(explicit_segment),
    )


def _line_arc_lengths(points_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("line points must have shape [N, 3]")
    if points.shape[0] <= 1:
        return np.zeros((points.shape[0],), dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def _format_triplet(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    return f"({arr[0]:.3f}, {arr[1]:.3f}, {arr[2]:.3f})"


def _print_native_trace_segment_debug(selection: _NativeTrace2CpSelection) -> None:
    record = selection.record
    fiber = record.fiber
    line_points_xyz = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    start_line_index = control_point_line_index(fiber, int(selection.start_cp_index))
    target_line_index = control_point_line_index(fiber, int(selection.target_cp_index))
    lo = min(int(start_line_index), int(target_line_index))
    hi = max(int(start_line_index), int(target_line_index))
    segment_xyz = line_points_xyz[lo : hi + 1]
    if segment_xyz.size == 0:
        raise ValueError(f"empty native Trace2CP line segment range {lo}:{hi}")
    cumulative = _line_arc_lengths(line_points_xyz)
    length_base = abs(float(cumulative[int(target_line_index)] - cumulative[int(start_line_index)]))
    spacing = float(getattr(record, "volume_spacing_base", 1.0))
    bbox_min_xyz = np.min(segment_xyz, axis=0)
    bbox_max_xyz = np.max(segment_xyz, axis=0)
    bbox_min_zyx = bbox_min_xyz[[2, 1, 0]]
    bbox_max_zyx = bbox_max_xyz[[2, 1, 0]]
    bbox_size_zyx = bbox_max_zyx - bbox_min_zyx
    start_zyx = np.asarray(fiber.control_points_zyx[int(selection.start_cp_index)], dtype=np.float64)
    target_zyx = np.asarray(fiber.control_points_zyx[int(selection.target_cp_index)], dtype=np.float64)
    print(
        "native_trace2cp_3d segment "
        f"sample_mode={selection.sample_mode} sample_index={int(selection.sample_index)} "
        f"record_index={int(selection.record_index)} explicit_segment={selection.explicit_segment} "
        f"start_cp={int(selection.start_cp_index)} target_cp={int(selection.target_cp_index)} "
        f"start_line={int(start_line_index)} target_line={int(target_line_index)} "
        f"line_range={lo}:{hi + 1} points={int(segment_xyz.shape[0])} "
        f"length_base={length_base:.3f} length_scaled={length_base / spacing:.3f} "
        f"bbox_min_zyx_base={_format_triplet(bbox_min_zyx)} "
        f"bbox_max_zyx_base={_format_triplet(bbox_max_zyx)} "
        f"bbox_size_zyx_base={_format_triplet(bbox_size_zyx)} "
        f"bbox_min_zyx_scaled={_format_triplet(bbox_min_zyx / spacing)} "
        f"bbox_max_zyx_scaled={_format_triplet(bbox_max_zyx / spacing)} "
        f"start_cp_zyx_base={_format_triplet(start_zyx)} "
        f"target_cp_zyx_base={_format_triplet(target_zyx)}",
        flush=True,
    )


def run_native_trace2cp(
    config_path: str | Path,
    *,
    checkpoint: str | Path,
    export_dir: str | Path,
    sample_index: int,
    fiber_json: str | Path | None = None,
    start_cp_index: int | None = None,
    target_cp_index: int | None = None,
    target_offset: int = 1,
    sample_mode: str | None = None,
    native_cfg: NativeTrace2CpConfig | None = None,
) -> NativeTracePairResult:
    raw_config = _load_raw_config(config_path)
    cfg = NativeTrace2CpConfig() if native_cfg is None else native_cfg
    fiber_path = None if fiber_json is None else Path(fiber_json)
    tool_raw_config = _tool_raw_config(raw_config, fiber_json=fiber_path)
    loader_config = _load_tool_config(config_path, raw_config, fiber_json=fiber_path)
    loader = FiberTrace3DLoader(loader_config)
    trace2cp_cfg = _native_trace2cp_geometry_config(raw_config)
    trace2cp_cfg = dataclass_replace(
        trace2cp_cfg,
        rf_margin_px=max(float(trace2cp_cfg.rf_margin_px), float(cfg.core_margin_voxels)),
    )
    geometry_loader = _make_trace2cp_geometry_loader(
        tool_raw_config,
        trace2cp_cfg,
    )
    selection = _resolve_native_trace2cp_selection(
        loader,
        sample_index=int(sample_index),
        fiber_json=fiber_path,
        start_cp_index=start_cp_index,
        target_cp_index=target_cp_index,
        target_offset=int(target_offset),
        sample_mode=sample_mode,
    )
    _print_native_trace_segment_debug(selection)
    record = selection.record
    start_zyx = (
        np.asarray(record.fiber.control_points_zyx[int(selection.start_cp_index)], dtype=np.float32)
        / np.float32(record.volume_spacing_base)
    )
    target_zyx = (
        np.asarray(record.fiber.control_points_zyx[int(selection.target_cp_index)], dtype=np.float32)
        / np.float32(record.volume_spacing_base)
    )
    forward_initial_direction = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=int(selection.start_cp_index),
        target_control_point_index=int(selection.target_cp_index),
    )
    reverse_initial_direction = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=int(selection.target_cp_index),
        target_control_point_index=int(selection.start_cp_index),
    )
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    model = build_fiber_trace_3d_model(raw_config).to(device)
    _load_snapshot(checkpoint, model=model, optimizer=None, map_location=device)
    model.eval()
    cache = NativeTraceFieldCache(
        record=record,
        model=model,
        config=loader_config,
        image_normalization=loader_config.image_normalization,
        patch_shape_zyx=cfg.inference_patch_shape_zyx,
        core_margin_voxels=cfg.core_margin_voxels,
        device=device,
    )
    print(
        "native_trace2cp_3d input "
        f"base_volume_scale={getattr(record, 'volume_scale', 'unknown')} "
        f"volume_spacing_base={float(getattr(record, 'volume_spacing_base', 1.0)):.6g} "
        f"image_normalization={loader_config.image_normalization} "
        f"sampler={type(getattr(record, 'sampler', None)).__name__} "
        f"blocking={getattr(getattr(record, 'sampler', None), 'blocking', 'n/a')}",
        flush=True,
    )
    result = trace_native_3d_pair(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        forward_initial_direction_zyx=forward_initial_direction,
        reverse_initial_direction_zyx=reverse_initial_direction,
        cfg=cfg,
        progress=True,
    )
    def build_source(cross_strip_height_px: int | None = None) -> _Trace2CpSegmentSource:
        return geometry_loader.build_trace2cp_segment_source(
            int(selection.sample_index),
            target_control_point_index=int(selection.target_cp_index)
            if selection.explicit_segment
            else None,
            target_offset=int(target_offset),
            rf_margin_px=trace2cp_cfg.rf_margin_px,
            cross_strip_height_px=cross_strip_height_px,
            device=torch.device("cpu"),
            sample_mode=selection.sample_mode,
        )

    max_source = build_source()
    max_side_overlays = _trace_overlays_for_view(
        max_source,
        result,
        axis_name="offset_axis_xyz",
    )
    max_top_overlays = _trace_overlays_for_view(
        max_source,
        result,
        axis_name="side_axis_xyz",
    )
    adaptive_height = _adaptive_trace2cp_cross_strip_height(
        int(max_source.source_shape_hw[0]),
        (max_side_overlays, max_top_overlays),
    )
    source = (
        max_source
        if int(adaptive_height) == int(max_source.source_shape_hw[0])
        else build_source(cross_strip_height_px=int(adaptive_height))
    )
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "trace2cp_native_3d_vis.jpg"
    sheet = _make_native_trace_visualization(
        geometry_loader,
        source,
        result,
        cache=cache,
        image_normalization=loader_config.image_normalization,
        partial_output_path=image_path,
    )
    sheet.convert("RGB").save(image_path, quality=95)
    summary = {
        "sample_index": int(selection.sample_index),
        "fiber_path": "" if record.fiber.path is None else str(record.fiber.path),
        "start_control_point_index": int(selection.start_cp_index),
        "target_control_point_index": int(selection.target_cp_index),
        "native_trace2cp_plane_error": float(result.plane_error),
        "native_trace2cp_closest_target_error": float(result.closest_target_error),
        "span_voxels": float(result.span_voxels),
        "forward_reached": bool(result.forward.reached_target_plane),
        "forward_reason": result.forward.reason,
        "reverse_reached": bool(result.reverse.reached_target_plane),
        "reverse_reason": result.reverse.reason,
        "forward_steps": int(len(result.forward.steps)),
        "reverse_steps": int(len(result.reverse.steps)),
        "step_voxels": float(cfg.step_voxels),
        "max_step_factor": float(cfg.max_step_factor),
        "max_steps": None if cfg.max_steps is None else int(cfg.max_steps),
        "trace_step_limit": None if cfg.trace_step_limit is None else int(cfg.trace_step_limit),
        "visualization_cross_strip_height_px": int(source.source_shape_hw[0]),
        "visualization_max_cross_strip_height_px": int(max_source.source_shape_hw[0]),
        "fusion_reason": result.fusion.reason,
        "fusion_reached_overlap": bool(result.fusion.reached_overlap),
        "fusion_closest_progress": float(result.fusion.closest_progress),
        "fusion_raw_gap_voxels": float(result.fusion.raw_gap_voxels),
        "fusion_considered_gap_voxels": float(result.fusion.considered_gap_voxels),
        "fusion_center_penalty": float(result.fusion.center_penalty),
        "fusion_closest_forward_zyx": [
            float(v) for v in np.asarray(result.fusion.closest_forward_zyx, dtype=np.float32)
        ],
        "fusion_closest_reverse_zyx": [
            float(v) for v in np.asarray(result.fusion.closest_reverse_zyx, dtype=np.float32)
        ],
        "fusion_closest_midpoint_zyx": [
            float(v) for v in np.asarray(result.fusion.closest_midpoint_zyx, dtype=np.float32)
        ],
        "inferred_blocks": int(len(cache._blocks)),
        "export": str(image_path),
    }
    summary_path = out_dir / "trace2cp_native_3d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"native_trace2cp_plane_error={result.plane_error:.8f}", flush=True)
    print(f"native_trace2cp_closest_target_error={result.closest_target_error:.8f}", flush=True)
    print(
        "native_trace2cp_fusion "
        f"reason={result.fusion.reason} "
        f"overlap={result.fusion.reached_overlap} "
        f"progress={result.fusion.closest_progress:.6f} "
        f"raw_gap={result.fusion.raw_gap_voxels:.6f} "
        f"considered_gap={result.fusion.considered_gap_voxels:.6f} "
        f"center_penalty={result.fusion.center_penalty:.6f}",
        flush=True,
    )
    print(
        "native_trace2cp_3d "
        f"sample_index={selection.sample_index} start_cp={selection.start_cp_index} "
        f"target_cp={selection.target_cp_index} "
        f"forward_reached={result.forward.reached_target_plane} reverse_reached={result.reverse.reached_target_plane} "
        f"blocks={len(cache._blocks)} export={image_path}",
        flush=True,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native 3D Trace2CP cone tracer")
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fiber-json", type=Path, default=None)
    parser.add_argument("--start-cp-index", type=int, default=None)
    parser.add_argument("--target-cp-index", type=int, default=None)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--sample-mode", choices=("random", "flat"), default=None)
    parser.add_argument("--step-voxels", type=float, default=4.0)
    parser.add_argument("--cone-angle-degrees", type=float, default=25.0)
    parser.add_argument("--cone-grid-size", type=int, default=25)
    parser.add_argument("--max-step-factor", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--trace-step-limit", type=int, default=None)
    parser.add_argument("--inference-patch-shape-zyx", nargs=3, type=int, default=None)
    parser.add_argument("--core-margin-voxels", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patch_shape = (
        (64, 64, 64)
        if args.inference_patch_shape_zyx is None
        else _as_zyx3(args.inference_patch_shape_zyx, key="--inference-patch-shape-zyx")
    )
    native_cfg = NativeTrace2CpConfig(
        step_voxels=float(args.step_voxels),
        cone_angle_degrees=float(args.cone_angle_degrees),
        cone_grid_size=int(args.cone_grid_size),
        max_step_factor=float(args.max_step_factor),
        max_steps=None if args.max_steps is None else int(args.max_steps),
        trace_step_limit=None if args.trace_step_limit is None else int(args.trace_step_limit),
        inference_patch_shape_zyx=patch_shape,
        core_margin_voxels=int(args.core_margin_voxels),
    )
    run_native_trace2cp(
        args.config,
        checkpoint=args.checkpoint,
        export_dir=args.export_dir,
        sample_index=int(args.sample_index),
        fiber_json=args.fiber_json,
        start_cp_index=args.start_cp_index,
        target_cp_index=args.target_cp_index,
        target_offset=int(args.target_offset),
        sample_mode=args.sample_mode,
        native_cfg=native_cfg,
    )


if __name__ == "__main__":
    main()
