from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

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
    cone_angle_step_degrees: float = 5.0
    beam_width: int = 8
    beam_prune_distance_voxels: float = 1.0
    beam_lookahead_steps: int = 1
    candidate_substeps: int = 1
    smoothness_weight: float = 2.0
    smoothness_tangent_weight: float | None = 10.0
    smoothness_normal_weight: float | None = 0.1
    smoothness_free_angle_degrees: float = 0.0
    cumulative_smoothness_steps: int = 4
    cumulative_smoothness_tangent_weight: float = 2.0
    all_pairs_direction_product: bool = True
    max_step_factor: float = 3.0
    max_steps: int | None = None
    trace_step_limit: int | None = None
    inference_patch_shape_zyx: tuple[int, int, int] = (64, 64, 64)
    core_margin_voxels: int = 20
    whole_fiber_error_threshold_voxels: float = 100.0


@dataclass(frozen=True)
class NativeTraceStep:
    point_zyx: np.ndarray
    direction_loss: float
    presence_loss: float
    total_loss: float
    rejected_candidates: int
    smoothness_loss: float = 0.0


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
class NativeWholeFiberSegmentResult:
    start_cp_index: int
    target_cp_index: int
    trace_zyx: np.ndarray
    start_zyx: np.ndarray
    target_zyx: np.ndarray
    reached_target_plane: bool
    success: bool
    restart: bool
    reason: str
    in_plane_error_voxels: float
    reference_arc_distance_voxels: float
    step_count: int


@dataclass(frozen=True)
class NativeWholeFiberResult:
    segments: tuple[NativeWholeFiberSegmentResult, ...]
    restart_count: int
    restart_rate: float
    segment_count: int
    stitched_trace_zyx: np.ndarray
    inferred_blocks: int


@dataclass(frozen=True)
class _NativeWholeFiberVisualSpan:
    start_cp_index: int
    end_cp_index: int
    segments: tuple[NativeWholeFiberSegmentResult, ...]
    restart_after: bool


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


@dataclass(frozen=True)
class _NativeBeamNode:
    point_zyx: np.ndarray
    previous_direction_zyx: np.ndarray
    history_direction_zyx: np.ndarray
    parent: "_NativeBeamNode | None"
    step: NativeTraceStep | None
    cumulative_loss: float
    depth: int


@dataclass(frozen=True)
class _NativeBeamTensorGeneration:
    points_zyx: torch.Tensor
    previous_directions_zyx: torch.Tensor
    history_directions_zyx: torch.Tensor
    cumulative_loss: torch.Tensor
    depth: torch.Tensor
    parent_indices: torch.Tensor | None
    step_direction_loss: torch.Tensor | None
    step_presence_loss: torch.Tensor | None
    step_total_loss: torch.Tensor | None
    step_smoothness_loss: torch.Tensor | None
    step_rejected_candidates: torch.Tensor | None


NativeTraceNormalSampler = Callable[
    [torch.Tensor | np.ndarray],
    tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray],
]


_CONE_OFFSET_TABLE_CACHE: dict[
    tuple[float, float, int, str, torch.dtype], torch.Tensor
] = {}


def _native_trace2cp_whole_fiber_mode(
    *,
    fiber_json: Path | None,
    sample_index: int | None,
    start_cp_index: int | None,
    target_cp_index: int | None,
) -> bool:
    if (start_cp_index is None) != (target_cp_index is None):
        raise ValueError("--start-cp-index and --target-cp-index must be provided together")
    return (
        fiber_json is not None
        and sample_index is None
        and start_cp_index is None
        and target_cp_index is None
    )


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


def _orthonormal_basis_torch(axes_n3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    axes = F.normalize(axes_n3.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    if axes.ndim != 2 or int(axes.shape[1]) != 3:
        raise ValueError("axes_n3 must have shape [N,3]")
    basis = torch.eye(3, dtype=torch.float32, device=axes.device)
    ref_index = torch.argmin(torch.abs(axes @ basis.T), dim=1)
    refs = basis[ref_index]
    b0 = F.normalize(torch.cross(axes, refs, dim=1), p=2.0, dim=1, eps=float(_EPS))
    b1 = F.normalize(torch.cross(axes, b0, dim=1), p=2.0, dim=1, eps=float(_EPS))
    return b0, b1


def _angle_step_cone_offsets_np(
    *,
    max_angle_degrees: float,
    angle_step_degrees: float,
) -> np.ndarray:
    max_angle = max(0.0, float(max_angle_degrees))
    step = float(angle_step_degrees)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("angle_step_degrees must be positive")
    if max_angle <= 0.0:
        return np.zeros((1, 2), dtype=np.float32)
    max_steps = int(math.floor(max_angle / step + 1.0e-6))
    values = np.arange(-max_steps, max_steps + 1, dtype=np.float32) * np.float32(step)
    uu, vv = np.meshgrid(values, values, indexing="xy")
    u_deg = uu.reshape(-1).astype(np.float32)
    v_deg = vv.reshape(-1).astype(np.float32)
    radius2 = u_deg * u_deg + v_deg * v_deg
    keep = radius2 <= np.float32(max_angle * max_angle + 1.0e-5)
    u_deg = u_deg[keep]
    v_deg = v_deg[keep]
    radius2 = radius2[keep]
    if not np.any((u_deg == 0.0) & (v_deg == 0.0)):
        u_deg = np.concatenate([np.asarray([0.0], dtype=np.float32), u_deg])
        v_deg = np.concatenate([np.asarray([0.0], dtype=np.float32), v_deg])
        radius2 = np.concatenate([np.asarray([0.0], dtype=np.float32), radius2])
    order = np.lexsort((v_deg, u_deg, radius2))
    u = np.tan(np.deg2rad(u_deg.astype(np.float64))).astype(np.float32)
    v = np.tan(np.deg2rad(v_deg.astype(np.float64))).astype(np.float32)
    return np.stack([u[order], v[order]], axis=1).astype(np.float32, copy=False)


def _legacy_grid_cone_offsets_np(
    *,
    max_angle_degrees: float,
    grid_size: int,
) -> np.ndarray:
    max_angle = math.radians(max(0.0, float(max_angle_degrees)))
    grid_count = int(grid_size)
    if grid_count <= 0:
        raise ValueError("grid_size must be positive")
    if max_angle <= 0.0 or grid_count == 1:
        return np.zeros((1, 2), dtype=np.float32)
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
    offsets = np.stack([tangent_scale * disk_x, tangent_scale * disk_y], axis=1)
    center_index = int(np.argmin(disk_x * disk_x + disk_y * disk_y))
    order = np.concatenate(
        [
            np.asarray([center_index], dtype=np.int64),
            np.asarray(
                [idx for idx in range(int(offsets.shape[0])) if idx != center_index],
                dtype=np.int64,
            ),
        ]
    )
    return offsets[order].astype(np.float32, copy=False)


def _cone_offset_table_torch(
    cfg: NativeTrace2CpConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (
        float(cfg.cone_angle_degrees),
        float(cfg.cone_angle_step_degrees),
        int(cfg.cone_grid_size),
        str(device),
        dtype,
    )
    cached = _CONE_OFFSET_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    if float(cfg.cone_angle_step_degrees) > 0.0:
        offsets_np = _angle_step_cone_offsets_np(
            max_angle_degrees=float(cfg.cone_angle_degrees),
            angle_step_degrees=float(cfg.cone_angle_step_degrees),
        )
    else:
        offsets_np = _legacy_grid_cone_offsets_np(
            max_angle_degrees=float(cfg.cone_angle_degrees),
            grid_size=int(cfg.cone_grid_size),
        )
    table = torch.as_tensor(offsets_np, dtype=dtype, device=device)
    _CONE_OFFSET_TABLE_CACHE[key] = table
    return table


def _trace_candidate_directions_torch(
    axes_n3: torch.Tensor,
    cfg: NativeTrace2CpConfig,
) -> torch.Tensor:
    axes = F.normalize(axes_n3.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    if axes.ndim != 2 or int(axes.shape[1]) != 3:
        raise ValueError("axes_n3 must have shape [N,3]")
    offsets = _cone_offset_table_torch(
        cfg,
        device=axes.device,
        dtype=axes.dtype,
    )
    b0, b1 = _orthonormal_basis_torch(axes)
    directions = (
        axes[:, None, :]
        + offsets[None, :, 0, None] * b0[:, None, :]
        + offsets[None, :, 1, None] * b1[:, None, :]
    )
    return F.normalize(directions, p=2.0, dim=2, eps=float(_EPS))


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


def generate_cone_candidates_by_angle_step(
    axis_zyx: np.ndarray,
    *,
    max_angle_degrees: float,
    angle_step_degrees: float = 5.0,
) -> np.ndarray:
    """Generate deterministic unit candidate directions by angular tangent steps."""

    axis = _unit(axis_zyx)
    max_angle = max(0.0, float(max_angle_degrees))
    step = float(angle_step_degrees)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("angle_step_degrees must be positive")
    if max_angle <= 0.0:
        return axis.reshape(1, 3).astype(np.float32)
    b0, b1 = _orthonormal_basis(axis)
    max_steps = int(math.floor(max_angle / step + 1.0e-6))
    values = np.arange(-max_steps, max_steps + 1, dtype=np.float32) * np.float32(step)
    uu, vv = np.meshgrid(values, values, indexing="xy")
    u_deg = uu.reshape(-1).astype(np.float32)
    v_deg = vv.reshape(-1).astype(np.float32)
    radius2 = u_deg * u_deg + v_deg * v_deg
    keep = radius2 <= np.float32(max_angle * max_angle + 1.0e-5)
    u_deg = u_deg[keep]
    v_deg = v_deg[keep]
    radius2 = radius2[keep]
    if not np.any((u_deg == 0.0) & (v_deg == 0.0)):
        u_deg = np.concatenate([np.asarray([0.0], dtype=np.float32), u_deg])
        v_deg = np.concatenate([np.asarray([0.0], dtype=np.float32), v_deg])
        radius2 = np.concatenate([np.asarray([0.0], dtype=np.float32), radius2])
    u = np.tan(np.deg2rad(u_deg.astype(np.float64))).astype(np.float32)
    v = np.tan(np.deg2rad(v_deg.astype(np.float64))).astype(np.float32)
    directions = axis[None, :] + u[:, None] * b0[None, :] + v[:, None] * b1[None, :]
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, np.float32(_EPS))
    order = np.lexsort((v_deg, u_deg, radius2))
    return directions[order].astype(np.float32, copy=False)


def _trace_candidate_directions(
    axis_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
) -> np.ndarray:
    if float(cfg.cone_angle_step_degrees) > 0.0:
        return generate_cone_candidates_by_angle_step(
            axis_zyx,
            max_angle_degrees=float(cfg.cone_angle_degrees),
            angle_step_degrees=float(cfg.cone_angle_step_degrees),
        )
    return generate_cone_candidates(
        axis_zyx,
        max_angle_degrees=float(cfg.cone_angle_degrees),
        grid_size=int(cfg.cone_grid_size),
    )


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


def _direction_branch_count_from_channels(channels: int) -> int:
    channels_i = int(channels)
    if channels_i < 6:
        raise ValueError("native 3D Trace2CP model output has fewer than six channels")
    if channels_i == 6:
        return 1
    if channels_i < 7 or channels_i % 7 != 0:
        raise ValueError(
            "native 3D Trace2CP model output channels must be 6 or a positive "
            f"multiple of 7; got {channels_i}"
        )
    return channels_i // 7


def _decode_grouped_direction_presence(
    sampled_nchannels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampled_nchannels.ndim != 2:
        raise ValueError("sampled_nchannels must have shape N,C")
    channels = int(sampled_nchannels.shape[1])
    branch_count = _direction_branch_count_from_channels(channels)
    directions: list[torch.Tensor] = []
    presences: list[torch.Tensor] = []
    for branch in range(branch_count):
        start = 0 if channels == 6 else branch * 7
        axis_xyz = decode_lasagna_direction_3x2_analytic(
            sampled_nchannels[:, start : start + 6]
        )
        axis_zyx = axis_xyz[:, [2, 1, 0]].to(dtype=torch.float32)
        axis_zyx = F.normalize(axis_zyx, p=2.0, dim=1, eps=float(_EPS))
        directions.append(axis_zyx)
        if channels == 6:
            presences.append(
                torch.ones(
                    (int(sampled_nchannels.shape[0]),),
                    dtype=torch.float32,
                    device=sampled_nchannels.device,
                )
            )
        else:
            presences.append(
                sampled_nchannels[:, start + 6].to(dtype=torch.float32).clamp(0.0, 1.0)
            )
    return torch.stack(directions, dim=1), torch.stack(presences, dim=1)


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
    def sample_point_choices_torch(
        self,
        points_zyx: np.ndarray,
        *,
        progress_label: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = np.asarray(points_zyx, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        count = int(points.shape[0])
        if count == 0:
            return (
                torch.zeros((0, 1, 3), dtype=torch.float32, device=self.device),
                torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                torch.zeros((0, 1), dtype=torch.bool, device=self.device),
            )

        origins = self._block_origins_for_points(points)
        unique_origins, inverse = np.unique(origins, axis=0, return_inverse=True)
        shape = np.asarray(self.patch_shape_zyx, dtype=np.float32)
        progress_start = time.perf_counter()
        last_progress_time = 0.0
        new_blocks = 0
        cached_blocks = 0
        valid_done = 0
        branch_count: int | None = None
        directions: torch.Tensor | None = None
        presence: torch.Tensor | None = None
        valid: torch.Tensor | None = None

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
            sampled_branch_count = _direction_branch_count_from_channels(int(sampled.shape[1]))
            if branch_count is None:
                branch_count = sampled_branch_count
                directions = torch.zeros(
                    (count, branch_count, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
                presence = torch.zeros(
                    (count, branch_count),
                    dtype=torch.float32,
                    device=self.device,
                )
                valid = torch.zeros(
                    (count, branch_count),
                    dtype=torch.bool,
                    device=self.device,
                )
            elif sampled_branch_count != branch_count:
                raise ValueError(
                    "native 3D Trace2CP sampled blocks disagree on branch count: "
                    f"{sampled_branch_count} != {branch_count}"
                )
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
            axis_choices_zyx, group_presence = _decode_grouped_direction_presence(sampled)
            group_valid = (
                (valid_values[:, None] > 0.5)
                & torch.isfinite(axis_choices_zyx).all(dim=2)
                & torch.isfinite(group_presence)
            )
            valid_done += int(torch.count_nonzero(group_valid.any(dim=1)).detach().cpu())
            index_t = torch.as_tensor(usable_indices, dtype=torch.long, device=self.device)
            assert directions is not None
            assert presence is not None
            assert valid is not None
            directions[index_t] = axis_choices_zyx
            presence[index_t] = group_presence
            valid[index_t] = group_valid
            emit_progress(int(unique_index) + 1)
        emit_progress(int(unique_origins.shape[0]), force=True)
        if directions is None or presence is None or valid is None:
            directions = torch.zeros((count, 1, 3), dtype=torch.float32, device=self.device)
            presence = torch.zeros((count, 1), dtype=torch.float32, device=self.device)
            valid = torch.zeros((count, 1), dtype=torch.bool, device=self.device)
        return directions, presence, valid

    @torch.no_grad()
    def sample_points_torch(
        self,
        points_zyx: np.ndarray,
        *,
        progress_label: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        directions, presence, valid = self.sample_point_choices_torch(
            points_zyx,
            progress_label=progress_label,
        )
        if int(directions.shape[1]) == 1:
            return directions[:, 0], presence[:, 0], valid[:, 0]
        masked_presence = torch.where(
            valid,
            presence,
            torch.full_like(presence, -torch.inf),
        )
        best_branch = torch.argmax(masked_presence, dim=1)
        rows = torch.arange(int(directions.shape[0]), dtype=torch.long, device=self.device)
        any_valid = valid.any(dim=1)
        selected_direction = directions[rows, best_branch]
        selected_presence = presence[rows, best_branch]
        selected_direction = torch.where(
            any_valid[:, None],
            selected_direction,
            torch.zeros_like(selected_direction),
        )
        selected_presence = torch.where(
            any_valid,
            selected_presence,
            torch.zeros_like(selected_presence),
        )
        return selected_direction, selected_presence, any_valid

    def sample_point(
        self,
        point_zyx: np.ndarray,
        *,
        reference_direction_zyx: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        directions, presence, valid = self.sample_point_choices_torch(
            np.asarray(point_zyx, dtype=np.float32).reshape(1, 3)
        )
        if not bool(valid[0].any().detach().cpu()):
            return np.zeros((3,), dtype=np.float32), 0.0, False
        if reference_direction_zyx is None:
            score = torch.where(
                valid[0],
                presence[0],
                torch.full_like(presence[0], -torch.inf),
            )
            aligned = directions[0]
        else:
            reference = torch.as_tensor(
                np.asarray(reference_direction_zyx, dtype=np.float32).reshape(1, 3),
                dtype=torch.float32,
                device=self.device,
            )
            reference = F.normalize(reference, p=2.0, dim=1, eps=float(_EPS))
            aligned = _align_axes_torch(directions[0], reference.expand_as(directions[0]))
            dot = torch.sum(aligned * reference.expand_as(aligned), dim=1).clamp(0.0, 1.0)
            score = torch.where(
                valid[0],
                dot * presence[0].clamp(0.0, 1.0),
                torch.full_like(presence[0], -torch.inf),
            )
        branch_index = int(torch.argmax(score).detach().cpu())
        axis_zyx = aligned[branch_index].detach().cpu().numpy().astype(np.float32)
        return _unit(axis_zyx), float(presence[0, branch_index].detach().cpu()), True


def _align_axes_torch(axes: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    axes_n = F.normalize(axes.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    refs_n = F.normalize(references.to(dtype=torch.float32), p=2.0, dim=-1, eps=float(_EPS))
    sign = torch.where(
        torch.sum(axes_n * refs_n, dim=-1, keepdim=True) >= 0.0,
        torch.ones((), dtype=torch.float32, device=axes_n.device),
        -torch.ones((), dtype=torch.float32, device=axes_n.device),
    )
    return axes_n * sign


def _points_to_numpy(points_zyx: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(points_zyx, torch.Tensor):
        return (
            points_zyx.detach()
            .to(device=torch.device("cpu"), dtype=torch.float32)
            .numpy()
            .astype(np.float32, copy=False)
        )
    return np.asarray(points_zyx, dtype=np.float32)


@dataclass(frozen=True)
class _NativeLasagnaNormalSampler:
    geometry_loader: Any
    trace_record: Any
    normal_record: Any

    def __call__(
        self,
        points_zyx_selected: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points_selected = _points_to_numpy(points_zyx_selected)
        if points_selected.ndim != 2 or points_selected.shape[1] != 3:
            raise ValueError("points_zyx_selected must have shape [N,3]")
        device = (
            points_zyx_selected.device
            if isinstance(points_zyx_selected, torch.Tensor)
            else torch.device("cpu")
        )
        count = int(points_selected.shape[0])
        if count == 0:
            return (
                torch.zeros((0, 3), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.bool, device=device),
            )
        spacing = float(getattr(self.trace_record, "volume_spacing_base", 1.0))
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"invalid volume_spacing_base for candidate normal sampling: {spacing!r}")
        points_base = points_selected.astype(np.float64, copy=False) * float(spacing)
        normals_xyz, valid, _invalid = self.geometry_loader._lasagna_normals_at_zyx_batch(
            self.normal_record,
            points_base,
            line_indices=np.arange(count, dtype=np.int64),
        )
        normals_zyx = np.asarray(normals_xyz, dtype=np.float32)[:, [2, 1, 0]]
        norms = np.linalg.norm(normals_zyx, axis=1)
        ok = np.asarray(valid, dtype=bool) & np.isfinite(normals_zyx).all(axis=1)
        ok &= np.isfinite(norms) & (norms > np.float32(_EPS))
        normals_zyx[ok] /= norms[ok, None].astype(np.float32, copy=False)
        normals_zyx[~ok] = 0.0
        return (
            torch.as_tensor(normals_zyx, dtype=torch.float32, device=device),
            torch.as_tensor(ok, dtype=torch.bool, device=device),
        )


def _fiber_path_key(record: Any) -> str:
    path = getattr(getattr(record, "fiber", None), "path", None)
    return "" if path is None else str(path)


def _records_refer_to_same_fiber(left: Any, right: Any) -> bool:
    left_fiber = getattr(left, "fiber", None)
    right_fiber = getattr(right, "fiber", None)
    if left_fiber is None or right_fiber is None:
        return False
    if _fiber_path_key(left) != _fiber_path_key(right):
        return False
    if str(getattr(left, "volume_path", "")) != str(getattr(right, "volume_path", "")):
        return False
    if int(getattr(left, "volume_scale", -1)) != int(getattr(right, "volume_scale", -2)):
        return False
    if abs(float(getattr(left, "volume_spacing_base", 0.0)) - float(getattr(right, "volume_spacing_base", 1.0))) > 1.0e-6:
        return False
    left_line = np.asarray(left_fiber.line_points_xyz, dtype=np.float32)
    right_line = np.asarray(right_fiber.line_points_xyz, dtype=np.float32)
    left_cp = np.asarray(left_fiber.control_points_xyz, dtype=np.float32)
    right_cp = np.asarray(right_fiber.control_points_xyz, dtype=np.float32)
    return (
        left_line.shape == right_line.shape
        and left_cp.shape == right_cp.shape
        and np.allclose(left_line, right_line, rtol=0.0, atol=1.0e-4)
        and np.allclose(left_cp, right_cp, rtol=0.0, atol=1.0e-4)
    )


def _native_trace_geometry_normal_record(geometry_loader: Any, trace_record: Any) -> Any:
    if hasattr(trace_record, "grad_mag"):
        return trace_record
    records = tuple(getattr(geometry_loader, "records", ()))
    if not records:
        raise ValueError("native 3D Trace2CP normal sampling requires geometry-loader records")
    matches = [record for record in records if _records_refer_to_same_fiber(trace_record, record)]
    if len(matches) == 1:
        return matches[0]
    if len(records) == 1:
        return records[0]
    raise ValueError(
        "native 3D Trace2CP could not map the 3D trace record to a Lasagna geometry record: "
        f"fiber_path='{_fiber_path_key(trace_record)}' "
        f"volume_path='{getattr(trace_record, 'volume_path', '')}' "
        f"volume_scale={getattr(trace_record, 'volume_scale', 'unknown')} "
        f"matches={len(matches)} records={len(records)}"
    )


def _cache_device(cache: Any, fallback: torch.device | None = None) -> torch.device:
    if hasattr(cache, "device"):
        return torch.device(getattr(cache, "device"))
    if fallback is not None:
        return torch.device(fallback)
    return torch.device("cpu")


def _sample_point_choices_for_points_torch(
    cache: Any,
    points_zyx: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points_np = _points_to_numpy(points_zyx)
    fallback_device = points_zyx.device if isinstance(points_zyx, torch.Tensor) else None
    device = _cache_device(cache, fallback=fallback_device)
    if hasattr(cache, "sample_point_choices_torch"):
        directions, presence, valid = cache.sample_point_choices_torch(points_np)
    else:
        directions_one, presence_one, valid_one = cache.sample_points_torch(points_np)
        directions = directions_one[:, None, :]
        presence = presence_one[:, None]
        valid = valid_one[:, None]
    return (
        directions.to(device=device, dtype=torch.float32),
        presence.to(device=device, dtype=torch.float32),
        valid.to(device=device, dtype=torch.bool),
    )


def _sample_trace_points_aligned_torch(
    cache: Any,
    points_zyx: torch.Tensor,
    *,
    reference_directions_zyx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = _cache_device(cache, fallback=points_zyx.device)
    points = points_zyx.to(device=device, dtype=torch.float32)
    references = F.normalize(
        reference_directions_zyx.to(device=device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    if not hasattr(cache, "sample_point_choices_torch") and hasattr(cache, "sample_point"):
        point_np = _points_to_numpy(points)
        ref_np = _points_to_numpy(references)
        selected: list[np.ndarray] = []
        presences: list[float] = []
        valids: list[bool] = []
        for point, reference in zip(point_np, ref_np, strict=True):
            sampled_direction, sampled_presence, valid = cache.sample_point(point)
            if bool(valid):
                selected.append(_align_axis(sampled_direction, reference))
                presences.append(float(sampled_presence))
                valids.append(True)
            else:
                selected.append(np.zeros((3,), dtype=np.float32))
                presences.append(0.0)
                valids.append(False)
        return (
            torch.as_tensor(np.stack(selected, axis=0), dtype=torch.float32, device=device),
            torch.as_tensor(presences, dtype=torch.float32, device=device),
            torch.as_tensor(valids, dtype=torch.bool, device=device),
        )
    directions, presence, valid = _sample_point_choices_for_points_torch(cache, points)
    if int(directions.shape[0]) != int(points.shape[0]):
        raise ValueError("sampled direction count does not match point count")
    if int(directions.shape[0]) == 0:
        return (
            torch.zeros((0, 3), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.bool, device=device),
        )
    aligned = _align_axes_torch(directions, references[:, None, :].expand_as(directions))
    dot = torch.sum(aligned * references[:, None, :].expand_as(aligned), dim=2).clamp(0.0, 1.0)
    score = torch.where(
        valid,
        dot * presence.clamp(0.0, 1.0),
        torch.full_like(presence, -torch.inf),
    )
    best_branch = torch.argmax(score, dim=1)
    rows = torch.arange(int(points.shape[0]), dtype=torch.long, device=device)
    any_valid = valid.any(dim=1)
    selected_direction = aligned[rows, best_branch]
    selected_presence = presence[rows, best_branch]
    selected_direction = torch.where(
        any_valid[:, None],
        selected_direction,
        torch.zeros_like(selected_direction),
    )
    selected_presence = torch.where(
        any_valid,
        selected_presence,
        torch.zeros_like(selected_presence),
    )
    return selected_direction, selected_presence, any_valid


def _sample_trace_point_aligned(
    cache: Any,
    point_zyx: np.ndarray,
    *,
    reference_direction_zyx: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    if hasattr(cache, "sample_point_choices_torch"):
        directions, presence, valid = cache.sample_point_choices_torch(
            np.asarray(point_zyx, dtype=np.float32).reshape(1, 3)
        )
        if not bool(valid[0].any().detach().cpu()):
            return np.zeros((3,), dtype=np.float32), 0.0, False
        reference = torch.as_tensor(
            np.asarray(reference_direction_zyx, dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=directions.device,
        )
        reference = F.normalize(reference, p=2.0, dim=1, eps=float(_EPS))
        choices = directions[0]
        aligned = _align_axes_torch(choices, reference.expand_as(choices))
        dot = torch.sum(aligned * reference.expand_as(aligned), dim=1).clamp(0.0, 1.0)
        score = torch.where(
            valid[0],
            dot * presence[0].clamp(0.0, 1.0),
            torch.full_like(presence[0], -torch.inf),
        )
        branch_index = int(torch.argmax(score).detach().cpu())
        axis_zyx = aligned[branch_index].detach().cpu().numpy().astype(np.float32)
        return _unit(axis_zyx), float(presence[0, branch_index].detach().cpu()), True
    sampled_direction, sampled_presence, valid = cache.sample_point(point_zyx)
    if not bool(valid):
        return np.zeros((3,), dtype=np.float32), 0.0, False
    return _align_axis(sampled_direction, reference_direction_zyx), float(sampled_presence), True


def _optional_nonnegative_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative when set")
    return out


def _sample_candidate_normals_torch(
    normal_sampler: NativeTraceNormalSampler | None,
    points_zyx: torch.Tensor | np.ndarray,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if normal_sampler is None:
        return None
    point_shape = tuple(int(v) for v in points_zyx.shape[:-1])
    if len(point_shape) == 0:
        raise ValueError("candidate normal points must have shape [...,3]")
    if int(points_zyx.shape[-1]) != 3:
        raise ValueError("candidate normal points must have shape [...,3]")
    flat_points = points_zyx.reshape(-1, 3)
    normals, valid = normal_sampler(flat_points)
    normals_t = torch.as_tensor(normals, dtype=torch.float32, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    expected_count = int(flat_points.shape[0])
    if normals_t.shape != (expected_count, 3):
        raise ValueError("candidate normal sampler must return normals with shape [N,3]")
    if valid_t.shape != (expected_count,):
        raise ValueError("candidate normal sampler must return valid mask with shape [N]")
    return normals_t.reshape(*point_shape, 3), valid_t.reshape(*point_shape)


def _native_smoothness_loss_torch(
    previous: torch.Tensor,
    candidates: torch.Tensor,
    *,
    candidate_normals: torch.Tensor | None,
    candidate_normals_valid: torch.Tensor | None,
    smoothness_weight: float,
    smoothness_tangent_weight: float | None,
    smoothness_normal_weight: float | None,
    smoothness_free_angle_degrees: float,
) -> torch.Tensor:
    if not math.isfinite(float(smoothness_weight)) or float(smoothness_weight) < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    tangent_weight = _optional_nonnegative_float(
        smoothness_tangent_weight,
        name="smoothness_tangent_weight",
    )
    normal_weight = _optional_nonnegative_float(
        smoothness_normal_weight,
        name="smoothness_normal_weight",
    )
    if (
        not math.isfinite(float(smoothness_free_angle_degrees))
        or float(smoothness_free_angle_degrees) < 0.0
    ):
        raise ValueError("smoothness_free_angle_degrees must be finite and non-negative")

    smooth_dot = torch.sum(previous[:, None, :] * candidates, dim=2).clamp(-1.0, 1.0)
    free_angle = math.radians(float(smoothness_free_angle_degrees))
    isotropic = (
        torch.clamp(torch.acos(smooth_dot) - float(free_angle), min=0.0).square()
        * float(smoothness_weight)
    )
    split_requested = tangent_weight is not None or normal_weight is not None
    if not split_requested:
        return isotropic
    if candidate_normals is None:
        return isotropic
    if candidate_normals.ndim == 4:
        normals = candidate_normals[:, :, -1, :]
        if candidate_normals_valid is None:
            normals_valid = None
        elif candidate_normals_valid.ndim == 3:
            normals_valid = candidate_normals_valid[:, :, -1]
        else:
            normals_valid = candidate_normals_valid
    else:
        normals = candidate_normals
        normals_valid = candidate_normals_valid
    normals = normals.to(device=candidates.device, dtype=torch.float32)
    if normals_valid is not None:
        normals_valid = normals_valid.to(device=candidates.device, dtype=torch.bool)
    if normals.shape != candidates.shape:
        raise ValueError("candidate_normals must have shape [N,M,3] or [N,M,S,3]")
    if normals_valid is not None and normals_valid.shape != candidates.shape[:2]:
        raise ValueError("candidate_normals_valid must have shape [N,M] or [N,M,S]")

    tangent_w = float(smoothness_weight) if tangent_weight is None else float(tangent_weight)
    normal_w = float(smoothness_weight) if normal_weight is None else float(normal_weight)
    normal_norm = torch.linalg.norm(normals.to(dtype=torch.float32), dim=2)
    finite_normal = torch.isfinite(normals).all(dim=2) & torch.isfinite(normal_norm)
    finite_normal = finite_normal & (normal_norm > float(_EPS))
    if normals_valid is not None:
        finite_normal = finite_normal & normals_valid.to(device=normals.device, dtype=torch.bool)
    unit_normal = F.normalize(normals, p=2.0, dim=2, eps=float(_EPS))
    previous_expand = previous[:, None, :].expand_as(candidates)
    previous_dot_n = torch.sum(previous_expand * unit_normal, dim=2).clamp(-1.0, 1.0)
    candidate_dot_n = torch.sum(candidates * unit_normal, dim=2).clamp(-1.0, 1.0)
    previous_tangent = previous_expand - previous_dot_n[:, :, None] * unit_normal
    candidate_tangent = candidates - candidate_dot_n[:, :, None] * unit_normal
    previous_tangent_norm = torch.linalg.norm(previous_tangent, dim=2)
    candidate_tangent_norm = torch.linalg.norm(candidate_tangent, dim=2)
    tangent_ok = (previous_tangent_norm > float(_EPS)) & (candidate_tangent_norm > float(_EPS))
    previous_tangent = F.normalize(previous_tangent, p=2.0, dim=2, eps=float(_EPS))
    candidate_tangent = F.normalize(candidate_tangent, p=2.0, dim=2, eps=float(_EPS))
    tangent_dot = torch.sum(previous_tangent * candidate_tangent, dim=2).clamp(-1.0, 1.0)
    tangent_angle = torch.acos(tangent_dot)
    isotropic_angle = torch.acos(smooth_dot)
    tangent_angle = torch.where(tangent_ok, tangent_angle, isotropic_angle)
    normal_angle = torch.abs(torch.asin(candidate_dot_n) - torch.asin(previous_dot_n))
    split = (
        torch.clamp(tangent_angle - float(free_angle), min=0.0).square() * tangent_w
        + torch.clamp(normal_angle - float(free_angle), min=0.0).square() * normal_w
    )
    return torch.where(finite_normal, split, isotropic)


def _native_first_step_normal_gate_torch(
    current: torch.Tensor,
    candidates: torch.Tensor,
    *,
    candidate_normals: torch.Tensor,
    candidate_normals_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_normals.ndim == 4:
        normals = candidate_normals[:, :, -1, :]
        if candidate_normals_valid is None:
            normals_valid = None
        elif candidate_normals_valid.ndim == 3:
            normals_valid = candidate_normals_valid[:, :, -1]
        else:
            normals_valid = candidate_normals_valid
    else:
        normals = candidate_normals
        normals_valid = candidate_normals_valid
    normals = normals.to(device=candidates.device, dtype=torch.float32)
    if normals_valid is not None:
        normals_valid = normals_valid.to(device=candidates.device, dtype=torch.bool)
    if candidates.ndim != 3 or int(candidates.shape[2]) != 3:
        raise ValueError("candidates must have shape [N,M,3]")
    if current.shape != (int(candidates.shape[0]), 3):
        raise ValueError("current must have shape [N,3]")
    if normals.shape != candidates.shape:
        raise ValueError("candidate_normals must have shape [N,M,3] or [N,M,S,3]")
    if normals_valid is not None and normals_valid.shape != candidates.shape[:2]:
        raise ValueError("candidate_normals_valid must have shape [N,M] or [N,M,S]")

    normal_norm = torch.linalg.norm(normals, dim=2)
    gate_valid = torch.isfinite(normals).all(dim=2) & torch.isfinite(normal_norm)
    gate_valid = gate_valid & (normal_norm > float(_EPS))
    if normals_valid is not None:
        gate_valid = gate_valid & normals_valid

    unit_normal = F.normalize(normals, p=2.0, dim=2, eps=float(_EPS))
    current_expand = current[:, None, :].expand_as(candidates)
    current_normal = torch.sum(current_expand * unit_normal, dim=2).clamp(-1.0, 1.0)
    candidate_normal = torch.sum(candidates * unit_normal, dim=2).clamp(-1.0, 1.0)
    normal_angle = torch.abs(torch.asin(candidate_normal) - torch.asin(current_normal))
    gate = torch.cos(normal_angle).clamp(0.0, 1.0)
    gate_valid = gate_valid & torch.isfinite(gate)
    return gate, gate_valid


def _native_cumulative_tangent_smoothness_loss_torch(
    history: torch.Tensor,
    candidates: torch.Tensor,
    *,
    candidate_normals: torch.Tensor | None,
    candidate_normals_valid: torch.Tensor | None,
    cumulative_smoothness_tangent_weight: float,
    smoothness_free_angle_degrees: float,
) -> torch.Tensor:
    weight = float(cumulative_smoothness_tangent_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("cumulative_smoothness_tangent_weight must be finite and non-negative")
    if (
        not math.isfinite(float(smoothness_free_angle_degrees))
        or float(smoothness_free_angle_degrees) < 0.0
    ):
        raise ValueError("smoothness_free_angle_degrees must be finite and non-negative")
    if candidates.ndim != 3 or int(candidates.shape[2]) != 3:
        raise ValueError("candidates must have shape [N,M,3]")
    state_count = int(candidates.shape[0])
    if history.shape != (state_count, 3):
        raise ValueError("history must have shape [N,3]")
    if weight <= 0.0 or candidate_normals is None:
        return torch.zeros(candidates.shape[:2], dtype=torch.float32, device=candidates.device)

    if candidate_normals.ndim == 4:
        normals = candidate_normals[:, :, -1, :]
        if candidate_normals_valid is None:
            normals_valid = None
        elif candidate_normals_valid.ndim == 3:
            normals_valid = candidate_normals_valid[:, :, -1]
        else:
            normals_valid = candidate_normals_valid
    else:
        normals = candidate_normals
        normals_valid = candidate_normals_valid
    normals = normals.to(device=candidates.device, dtype=torch.float32)
    if normals_valid is not None:
        normals_valid = normals_valid.to(device=candidates.device, dtype=torch.bool)
    if normals.shape != candidates.shape:
        raise ValueError("candidate_normals must have shape [N,M,3] or [N,M,S,3]")
    if normals_valid is not None and normals_valid.shape != candidates.shape[:2]:
        raise ValueError("candidate_normals_valid must have shape [N,M] or [N,M,S]")

    normal_norm = torch.linalg.norm(normals, dim=2)
    finite_normal = torch.isfinite(normals).all(dim=2) & torch.isfinite(normal_norm)
    finite_normal = finite_normal & (normal_norm > float(_EPS))
    if normals_valid is not None:
        finite_normal = finite_normal & normals_valid
    unit_normal = F.normalize(normals, p=2.0, dim=2, eps=float(_EPS))
    history_expand = F.normalize(
        history.to(device=candidates.device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )[:, None, :].expand_as(candidates)
    history_dot_n = torch.sum(history_expand * unit_normal, dim=2).clamp(-1.0, 1.0)
    candidate_dot_n = torch.sum(candidates * unit_normal, dim=2).clamp(-1.0, 1.0)
    history_tangent = history_expand - history_dot_n[:, :, None] * unit_normal
    candidate_tangent = candidates - candidate_dot_n[:, :, None] * unit_normal
    history_norm = torch.linalg.norm(history_tangent, dim=2)
    candidate_norm = torch.linalg.norm(candidate_tangent, dim=2)
    tangent_ok = (history_norm > float(_EPS)) & (candidate_norm > float(_EPS))
    history_tangent = F.normalize(history_tangent, p=2.0, dim=2, eps=float(_EPS))
    candidate_tangent = F.normalize(candidate_tangent, p=2.0, dim=2, eps=float(_EPS))
    tangent_dot = torch.sum(history_tangent * candidate_tangent, dim=2).clamp(-1.0, 1.0)
    tangent_angle = torch.acos(tangent_dot)
    free_angle = math.radians(float(smoothness_free_angle_degrees))
    loss = torch.clamp(tangent_angle - float(free_angle), min=0.0).square() * weight
    valid = finite_normal & tangent_ok & torch.isfinite(loss)
    return torch.where(valid, loss, torch.zeros_like(loss))


def _update_native_history_direction_torch(
    history: torch.Tensor,
    chosen_directions: torch.Tensor,
    depth: torch.Tensor,
    *,
    cumulative_smoothness_steps: int,
) -> torch.Tensor:
    steps = int(cumulative_smoothness_steps)
    if steps <= 1:
        return F.normalize(chosen_directions, p=2.0, dim=1, eps=float(_EPS))
    history_t = F.normalize(
        history.to(device=chosen_directions.device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    chosen_t = F.normalize(
        chosen_directions.to(device=chosen_directions.device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    depth_t = depth.to(device=chosen_directions.device, dtype=torch.long)
    root = depth_t <= 0
    count = torch.clamp(depth_t.to(dtype=torch.float32), min=1.0, max=float(steps - 1))
    blended = history_t * count[:, None] + chosen_t
    updated = F.normalize(blended, p=2.0, dim=1, eps=float(_EPS))
    return torch.where(root[:, None], chosen_t, updated)


def _update_native_history_direction_np(
    history: np.ndarray,
    chosen_direction: np.ndarray,
    *,
    depth: int,
    cumulative_smoothness_steps: int,
) -> np.ndarray:
    steps = int(cumulative_smoothness_steps)
    chosen = _unit(np.asarray(chosen_direction, dtype=np.float32))
    if int(depth) <= 0 or steps <= 1:
        return chosen.astype(np.float32)
    count = float(min(max(int(depth), 1), steps - 1))
    blended = _unit(np.asarray(history, dtype=np.float32)) * count + chosen
    return _unit(blended).astype(np.float32)


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


def _score_candidate_loss_tensors(
    cache: NativeTraceFieldCache,
    *,
    current_direction: np.ndarray,
    previous_step_direction: np.ndarray,
    candidate_directions: np.ndarray,
    next_points: np.ndarray,
    current_point: np.ndarray | None = None,
    step_voxels: float | None = None,
    candidate_substeps: int = 1,
    smoothness_weight: float = 2.0,
    smoothness_tangent_weight: float | None = None,
    smoothness_normal_weight: float | None = None,
    smoothness_free_angle_degrees: float = 0.0,
    cumulative_smoothness_tangent_weight: float = 2.0,
    all_pairs_direction_product: bool = True,
    candidate_normals: torch.Tensor | np.ndarray | None = None,
    candidate_normals_valid: torch.Tensor | np.ndarray | None = None,
    history_direction: torch.Tensor | np.ndarray | None = None,
    first_step_mask: torch.Tensor | np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = _cache_device(cache)
    current_points_t = None
    if current_point is not None:
        current_points_t = torch.as_tensor(
            np.asarray(current_point, dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=device,
        )
    (
        total_loss,
        direction_loss,
        presence_loss,
        smoothness_loss,
        candidate_valid,
        rejected_per_state,
    ) = _score_candidate_loss_tensors_batched(
        cache,
        current_directions=torch.as_tensor(
            np.asarray(current_direction, dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=device,
        ),
        previous_step_directions=torch.as_tensor(
            np.asarray(previous_step_direction, dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=device,
        ),
        candidate_directions=torch.as_tensor(
            np.asarray(candidate_directions, dtype=np.float32).reshape(
                1, int(np.asarray(candidate_directions).shape[0]), 3
            ),
            dtype=torch.float32,
            device=device,
        ),
        next_points=torch.as_tensor(
            np.asarray(next_points, dtype=np.float32).reshape(
                1, int(np.asarray(next_points).shape[0]), 3
            ),
            dtype=torch.float32,
            device=device,
        ),
        current_points=current_points_t,
        step_voxels=step_voxels,
        candidate_substeps=candidate_substeps,
        smoothness_weight=smoothness_weight,
        smoothness_tangent_weight=smoothness_tangent_weight,
        smoothness_normal_weight=smoothness_normal_weight,
        smoothness_free_angle_degrees=smoothness_free_angle_degrees,
        cumulative_smoothness_tangent_weight=cumulative_smoothness_tangent_weight,
        all_pairs_direction_product=bool(all_pairs_direction_product),
        candidate_normals=None
        if candidate_normals is None
        else torch.as_tensor(candidate_normals, dtype=torch.float32, device=device),
        candidate_normals_valid=None
        if candidate_normals_valid is None
        else torch.as_tensor(candidate_normals_valid, dtype=torch.bool, device=device),
        history_directions=None
        if history_direction is None
        else torch.as_tensor(
            np.asarray(history_direction, dtype=np.float32).reshape(1, 3),
            dtype=torch.float32,
            device=device,
        ),
        first_step_mask=None
        if first_step_mask is None
        else torch.as_tensor(first_step_mask, dtype=torch.bool, device=device),
    )
    return (
        total_loss[0],
        direction_loss[0],
        presence_loss[0],
        smoothness_loss[0],
        candidate_valid[0],
        int(rejected_per_state[0].detach().cpu()),
    )


def _score_candidate_loss_tensors_batched(
    cache: NativeTraceFieldCache,
    *,
    current_directions: torch.Tensor,
    previous_step_directions: torch.Tensor,
    candidate_directions: torch.Tensor,
    next_points: torch.Tensor,
    current_points: torch.Tensor | None = None,
    step_voxels: float | None = None,
    candidate_substeps: int = 1,
    smoothness_weight: float = 2.0,
    smoothness_tangent_weight: float | None = None,
    smoothness_normal_weight: float | None = None,
    smoothness_free_angle_degrees: float = 0.0,
    cumulative_smoothness_tangent_weight: float = 2.0,
    all_pairs_direction_product: bool = True,
    candidate_normals: torch.Tensor | None = None,
    candidate_normals_valid: torch.Tensor | None = None,
    history_directions: torch.Tensor | None = None,
    first_step_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    substeps = int(candidate_substeps)
    if substeps < 1:
        raise ValueError("candidate_substeps must be at least 1")
    device = _cache_device(cache, fallback=current_directions.device)
    current = F.normalize(
        current_directions.to(device=device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    previous = F.normalize(
        previous_step_directions.to(device=device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    candidates = F.normalize(
        candidate_directions.to(device=device, dtype=torch.float32),
        p=2.0,
        dim=2,
        eps=float(_EPS),
    )
    points = next_points.to(device=device, dtype=torch.float32)
    if candidates.ndim != 3 or int(candidates.shape[2]) != 3:
        raise ValueError("candidate_directions must have shape [N,M,3]")
    if points.shape != candidates.shape:
        raise ValueError("next_points must have the same [N,M,3] shape as candidates")
    state_count = int(candidates.shape[0])
    candidate_count = int(candidates.shape[1])
    if current.shape != (state_count, 3):
        raise ValueError("current_directions must have shape [N,3]")
    if previous.shape != (state_count, 3):
        raise ValueError("previous_step_directions must have shape [N,3]")
    history = None
    if history_directions is not None:
        history = F.normalize(
            history_directions.to(device=device, dtype=torch.float32),
            p=2.0,
            dim=1,
            eps=float(_EPS),
        )
        if history.shape != (state_count, 3):
            raise ValueError("history_directions must have shape [N,3]")
    if first_step_mask is None:
        first_step_t = torch.zeros((state_count,), dtype=torch.bool, device=device)
    else:
        first_step_t = first_step_mask.to(device=device, dtype=torch.bool)
        if first_step_t.shape != (state_count,):
            raise ValueError("first_step_mask must have shape [N]")
    candidates = _align_axes_torch(
        candidates,
        current[:, None, :].expand_as(candidates),
    )
    current_dot = torch.sum(current[:, None, :] * candidates, dim=2).clamp(-1.0, 1.0)
    previous_dot = torch.sum(previous[:, None, :] * candidates, dim=2).clamp(
        0.0,
        1.0,
    )
    normals_t = (
        None
        if candidate_normals is None
        else candidate_normals.to(device=device, dtype=torch.float32)
    )
    normals_valid_t = (
        None
        if candidate_normals_valid is None
        else candidate_normals_valid.to(device=device, dtype=torch.bool)
    )
    if normals_t is not None and bool(torch.any(first_step_t).detach().cpu()):
        normal_gate, normal_gate_valid = _native_first_step_normal_gate_torch(
            current,
            candidates,
            candidate_normals=normals_t,
            candidate_normals_valid=normals_valid_t,
        )
        use_normal_gate = first_step_t[:, None] & normal_gate_valid
        current_dot = torch.where(use_normal_gate, normal_gate, current_dot)
    if bool(torch.any(first_step_t).detach().cpu()):
        previous_dot = torch.where(
            first_step_t[:, None],
            torch.ones_like(previous_dot),
            previous_dot,
        )
    smoothness_loss = _native_smoothness_loss_torch(
        previous,
        candidates,
        candidate_normals=normals_t,
        candidate_normals_valid=normals_valid_t,
        smoothness_weight=float(smoothness_weight),
        smoothness_tangent_weight=smoothness_tangent_weight,
        smoothness_normal_weight=smoothness_normal_weight,
        smoothness_free_angle_degrees=float(smoothness_free_angle_degrees),
    )
    if bool(torch.any(first_step_t).detach().cpu()):
        smoothness_loss = torch.where(
            first_step_t[:, None],
            torch.zeros_like(smoothness_loss),
            smoothness_loss,
        )
    if history is not None:
        cumulative_loss = _native_cumulative_tangent_smoothness_loss_torch(
            history,
            candidates,
            candidate_normals=normals_t,
            candidate_normals_valid=normals_valid_t,
            cumulative_smoothness_tangent_weight=float(
                cumulative_smoothness_tangent_weight
            ),
            smoothness_free_angle_degrees=float(smoothness_free_angle_degrees),
        )
        if bool(torch.any(first_step_t).detach().cpu()):
            cumulative_loss = torch.where(
                first_step_t[:, None],
                torch.zeros_like(cumulative_loss),
                cumulative_loss,
            )
        smoothness_loss = smoothness_loss + cumulative_loss

    if substeps == 1:
        flat_points = points.reshape(state_count * candidate_count, 3)
        next_direction_choices, presence_choices, valid_choices = (
            _sample_point_choices_for_points_torch(
                cache,
                flat_points,
            )
        )
        branch_count = int(next_direction_choices.shape[1])
        next_direction_choices = next_direction_choices.reshape(
            state_count,
            candidate_count,
            branch_count,
            3,
        )
        presence_choices = presence_choices.reshape(state_count, candidate_count, branch_count)
        valid_choices = valid_choices.reshape(state_count, candidate_count, branch_count)
        candidate_choices = candidates[:, :, None, :].expand_as(next_direction_choices)
        next_aligned = _align_axes_torch(next_direction_choices, candidate_choices)
        next_dot = torch.sum(next_aligned * candidate_choices, dim=3).clamp(-1.0, 1.0)
        presence = presence_choices.clamp(0.0, 1.0)
        if bool(all_pairs_direction_product):
            previous_choices = previous[:, None, None, :].expand_as(next_aligned)
            current_choices = current[:, None, None, :].expand_as(next_aligned)
            previous_current_dot = torch.sum(
                previous_choices * current_choices,
                dim=3,
            ).clamp(0.0, 1.0)
            previous_next_dot = torch.sum(previous_choices * next_aligned, dim=3).clamp(
                0.0,
                1.0,
            )
            current_next_dot = torch.sum(current_choices * next_aligned, dim=3).clamp(
                0.0,
                1.0,
            )
            if bool(torch.any(first_step_t).detach().cpu()):
                neutral = first_step_t[:, None, None]
                previous_current_dot = torch.where(
                    neutral,
                    torch.ones_like(previous_current_dot),
                    previous_current_dot,
                )
                previous_next_dot = torch.where(
                    neutral,
                    torch.ones_like(previous_next_dot),
                    previous_next_dot,
                )
                current_next_dot = torch.where(
                    neutral,
                    torch.ones_like(current_next_dot),
                    current_next_dot,
                )
            score = (
                previous_dot[:, :, None]
                * current_dot[:, :, None]
                * next_dot
                * previous_current_dot
                * previous_next_dot
                * current_next_dot
                * presence
            )
        else:
            score = current_dot[:, :, None] * next_dot * presence
        direction_loss = 1.0 - 0.5 * (current_dot[:, :, None] + next_dot)
        presence_loss = 1.0 - presence
        total_loss = 1.0 - score
        total_loss = total_loss + smoothness_loss[:, :, None]
        total_loss = torch.where(
            valid_choices,
            total_loss,
            torch.full_like(total_loss, torch.inf),
        )
        candidate_valid = valid_choices.any(dim=2)
        rejected_per_state = int(candidate_count) - torch.count_nonzero(candidate_valid, dim=1)
        return (
            total_loss,
            direction_loss,
            presence_loss,
            smoothness_loss,
            candidate_valid,
            rejected_per_state,
        )

    if current_points is None:
        raise ValueError("current_points is required when candidate_substeps > 1")
    if step_voxels is None:
        raise ValueError("step_voxels is required when candidate_substeps > 1")
    step = float(step_voxels)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_voxels must be finite and positive")
    current_points_t = current_points.to(device=device, dtype=torch.float32)
    if current_points_t.shape != (state_count, 3):
        raise ValueError("current_points must have shape [N,3]")
    sub_t = (
        torch.arange(1, substeps + 1, dtype=torch.float32, device=device)
        / float(substeps)
    )
    substep_points = (
        current_points_t[:, None, None, :]
        + candidates[:, :, None, :] * float(step) * sub_t[None, None, :, None]
    )
    flat_points = substep_points.reshape(state_count * candidate_count * substeps, 3)
    next_direction_choices, presence_choices, valid_choices = _sample_point_choices_for_points_torch(
        cache,
        flat_points,
    )
    branch_count = int(next_direction_choices.shape[1])
    next_direction_choices = next_direction_choices.reshape(
        state_count,
        candidate_count,
        substeps,
        branch_count,
        3,
    )
    presence_choices = presence_choices.reshape(
        state_count,
        candidate_count,
        substeps,
        branch_count,
    )
    valid_choices = valid_choices.reshape(
        state_count,
        candidate_count,
        substeps,
        branch_count,
    )
    candidate_choices = candidates[:, :, None, None, :].expand_as(next_direction_choices)
    next_aligned = _align_axes_torch(next_direction_choices, candidate_choices)
    next_dot = torch.sum(next_aligned * candidate_choices, dim=4).clamp(-1.0, 1.0)
    presence = presence_choices.clamp(0.0, 1.0)
    if bool(all_pairs_direction_product):
        previous_choices = previous[:, None, None, None, :].expand_as(next_aligned)
        current_choices = current[:, None, None, None, :].expand_as(next_aligned)
        previous_current_dot = torch.sum(previous_choices * current_choices, dim=4).clamp(
            0.0,
            1.0,
        )
        previous_next_dot = torch.sum(previous_choices * next_aligned, dim=4).clamp(
            0.0,
            1.0,
        )
        current_next_dot = torch.sum(current_choices * next_aligned, dim=4).clamp(
            0.0,
            1.0,
        )
        if bool(torch.any(first_step_t).detach().cpu()):
            neutral = first_step_t[:, None, None, None]
            previous_current_dot = torch.where(
                neutral,
                torch.ones_like(previous_current_dot),
                previous_current_dot,
            )
            previous_next_dot = torch.where(
                neutral,
                torch.ones_like(previous_next_dot),
                previous_next_dot,
            )
            current_next_dot = torch.where(
                neutral,
                torch.ones_like(current_next_dot),
                current_next_dot,
            )
        substep_raw_score = (
            previous_dot[:, :, None, None]
            * current_dot[:, :, None, None]
            * next_dot
            * previous_current_dot
            * previous_next_dot
            * current_next_dot
            * presence
        )
    else:
        substep_raw_score = next_dot * presence
    substep_score = torch.where(
        valid_choices,
        substep_raw_score,
        torch.full_like(presence, -torch.inf),
    )
    best_substep_score, best_substep_branch = torch.max(substep_score, dim=3)
    substep_valid = valid_choices.any(dim=3)
    safe_best_score = torch.where(
        substep_valid,
        best_substep_score,
        torch.zeros_like(best_substep_score),
    )
    segment_score = torch.mean(safe_best_score, dim=2)
    gather_index = best_substep_branch[:, :, :, None]
    best_substep_dot = torch.gather(next_dot, dim=3, index=gather_index).squeeze(3)
    best_substep_presence = torch.gather(presence, dim=3, index=gather_index).squeeze(3)
    direction_loss_2d = torch.mean(
        1.0 - 0.5 * (current_dot[:, :, None] + best_substep_dot),
        dim=2,
    )
    presence_loss_2d = torch.mean(1.0 - best_substep_presence, dim=2)
    if bool(all_pairs_direction_product):
        total_loss_2d = 1.0 - segment_score
    else:
        total_loss_2d = 1.0 - current_dot * segment_score
    total_loss_2d = total_loss_2d + smoothness_loss
    candidate_valid = substep_valid.all(dim=2)
    total_loss_2d = torch.where(
        candidate_valid,
        total_loss_2d,
        torch.full_like(total_loss_2d, torch.inf),
    )
    rejected_per_state = int(candidate_count) - torch.count_nonzero(candidate_valid, dim=1)
    return (
        total_loss_2d[:, :, None],
        direction_loss_2d[:, :, None],
        presence_loss_2d[:, :, None],
        smoothness_loss,
        candidate_valid,
        rejected_per_state,
    )


def _score_candidate_batch(
    cache: NativeTraceFieldCache,
    *,
    current_direction: np.ndarray,
    previous_step_direction: np.ndarray,
    candidate_directions: np.ndarray,
    next_points: np.ndarray,
    current_point: np.ndarray | None = None,
    step_voxels: float | None = None,
    candidate_substeps: int = 1,
    smoothness_weight: float = 2.0,
    smoothness_tangent_weight: float | None = None,
    smoothness_normal_weight: float | None = None,
    smoothness_free_angle_degrees: float = 0.0,
    cumulative_smoothness_tangent_weight: float = 2.0,
    all_pairs_direction_product: bool = True,
    candidate_normals: torch.Tensor | np.ndarray | None = None,
    candidate_normals_valid: torch.Tensor | np.ndarray | None = None,
    history_direction: torch.Tensor | np.ndarray | None = None,
    first_step: bool = False,
) -> tuple[int | None, float, float, float, float, int]:
    if candidate_normals is None:
        candidate_normals_batched = None
    elif isinstance(candidate_normals, torch.Tensor):
        candidate_normals_batched = candidate_normals.reshape(1, -1, 3)
    else:
        candidate_normals_batched = np.asarray(candidate_normals, dtype=np.float32).reshape(1, -1, 3)
    if candidate_normals_valid is None:
        candidate_normals_valid_batched = None
    elif isinstance(candidate_normals_valid, torch.Tensor):
        candidate_normals_valid_batched = candidate_normals_valid.reshape(1, -1)
    else:
        candidate_normals_valid_batched = np.asarray(candidate_normals_valid, dtype=bool).reshape(1, -1)
    (
        total_loss,
        direction_loss,
        presence_loss,
        smoothness_loss,
        candidate_valid,
        rejected,
    ) = _score_candidate_loss_tensors(
        cache,
        current_direction=current_direction,
        previous_step_direction=previous_step_direction,
        candidate_directions=candidate_directions,
        next_points=next_points,
        current_point=current_point,
        step_voxels=step_voxels,
        candidate_substeps=candidate_substeps,
        smoothness_weight=smoothness_weight,
        smoothness_tangent_weight=smoothness_tangent_weight,
        smoothness_normal_weight=smoothness_normal_weight,
        smoothness_free_angle_degrees=smoothness_free_angle_degrees,
        cumulative_smoothness_tangent_weight=cumulative_smoothness_tangent_weight,
        all_pairs_direction_product=bool(all_pairs_direction_product),
        candidate_normals=candidate_normals_batched,
        candidate_normals_valid=candidate_normals_valid_batched,
        history_direction=history_direction,
        first_step_mask=np.asarray([bool(first_step)], dtype=bool),
    )
    valid_count = int(torch.count_nonzero(candidate_valid).detach().cpu())
    if valid_count == 0:
        return None, math.inf, math.inf, math.inf, math.inf, rejected
    best_flat_index = int(torch.argmin(total_loss.reshape(-1)).detach().cpu())
    branch_count = int(total_loss.shape[1])
    best_index = int(best_flat_index // branch_count)
    best_branch = int(best_flat_index % branch_count)
    return (
        best_index,
        float(total_loss[best_index, best_branch].detach().cpu()),
        float(direction_loss[best_index, best_branch].detach().cpu()),
        float(presence_loss[best_index, best_branch].detach().cpu()),
        float(smoothness_loss[best_index].detach().cpu()),
        rejected,
    )


def _plane_distance(point_zyx: np.ndarray, plane_point_zyx: np.ndarray, normal_zyx: np.ndarray) -> float:
    return float(np.dot(np.asarray(point_zyx, dtype=np.float64) - plane_point_zyx, normal_zyx))


def _target_plane_in_plane_error_voxels(
    point_zyx: np.ndarray,
    *,
    target_zyx: np.ndarray,
    plane_normal_zyx: np.ndarray,
) -> float:
    normal = _unit(np.asarray(plane_normal_zyx, dtype=np.float32))
    delta = np.asarray(point_zyx, dtype=np.float64) - np.asarray(target_zyx, dtype=np.float64)
    in_plane = delta - float(np.dot(delta, normal.astype(np.float64))) * normal.astype(np.float64)
    return float(np.linalg.norm(in_plane))


def _native_trace_step_limit(
    *,
    span_voxels: float,
    cfg: NativeTrace2CpConfig,
) -> tuple[int, str]:
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
        int(math.ceil(float(cfg.max_step_factor) * float(span_voxels) / float(cfg.step_voxels))),
    )
    limit_candidates: list[tuple[int, str]] = [(dynamic_limit, "max_step_factor")]
    if cfg.max_steps is not None:
        limit_candidates.append((int(cfg.max_steps), "max_steps"))
    if cfg.trace_step_limit is not None:
        limit_candidates.append((int(cfg.trace_step_limit), "trace_step_limit"))
    return min(limit_candidates, key=lambda item: item[0])


def _native_trace_cfg_with_effective_smoothness(
    cfg: NativeTrace2CpConfig,
    *,
    normal_sampler: NativeTraceNormalSampler | None,
) -> NativeTrace2CpConfig:
    if int(cfg.cumulative_smoothness_steps) < 1:
        raise ValueError("cumulative_smoothness_steps must be at least 1")
    _optional_nonnegative_float(
        cfg.cumulative_smoothness_tangent_weight,
        name="cumulative_smoothness_tangent_weight",
    )
    tangent = _optional_nonnegative_float(
        cfg.smoothness_tangent_weight,
        name="smoothness_tangent_weight",
    )
    normal = _optional_nonnegative_float(
        cfg.smoothness_normal_weight,
        name="smoothness_normal_weight",
    )
    if normal_sampler is None:
        return replace(cfg, smoothness_tangent_weight=None, smoothness_normal_weight=None)
    fallback = float(cfg.smoothness_weight)
    if not math.isfinite(fallback) or fallback < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    return replace(
        cfg,
        smoothness_tangent_weight=fallback if tangent is None else tangent,
        smoothness_normal_weight=fallback if normal is None else normal,
    )


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


def _trace_native_3d_one_way_greedy(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_plane_normal_zyx: np.ndarray | None = None,
    budget_span_voxels: float | None = None,
    progress_label: str | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTraceResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    plane_normal = (
        _unit(target - start)
        if target_plane_normal_zyx is None
        else _require_unit(target_plane_normal_zyx, label="target_plane_normal_zyx")
    )
    span = float(np.linalg.norm(target - start))
    if span <= _EPS:
        raise ValueError("native 3D Trace2CP start and target CPs must differ")
    budget_span = span if budget_span_voxels is None else float(budget_span_voxels)
    if not math.isfinite(budget_span) or budget_span <= _EPS:
        raise ValueError("budget_span_voxels must be positive when set")
    step_limit, limit_reason = _native_trace_step_limit(
        span_voxels=budget_span,
        cfg=cfg,
    )
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
            f"eta={_format_eta(eta)} blocks={len(cache._blocks)} "
            f"substeps={int(cfg.candidate_substeps)}{suffix}",
            end=end,
            flush=True,
        )

    emit_progress(start, 0)
    initial_sampled_direction, _presence, valid = _sample_trace_point_aligned(
        cache,
        start,
        reference_direction_zyx=initial_direction_zyx,
    )
    if not valid:
        raise ValueError(f"native 3D Trace2CP start point is invalid: {start.tolist()}")
    initial_direction = _require_unit(
        initial_direction_zyx,
        label="native 3D Trace2CP initial_direction_zyx",
    )
    previous_direction = initial_direction.astype(np.float32, copy=False)
    history_direction = initial_direction.astype(np.float32, copy=False)
    trace: list[np.ndarray] = [start.astype(np.float32)]
    steps: list[NativeTraceStep] = []
    current = start.astype(np.float32)
    for _step_index in range(step_limit):
        if _step_index == 0:
            sampled_direction = initial_sampled_direction
            valid = True
        else:
            sampled_direction, _presence, valid = _sample_trace_point_aligned(
                cache,
                current,
                reference_direction_zyx=previous_direction,
            )
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
        candidates_unit = _trace_candidate_directions(current_direction, cfg)
        next_points = current[None, :] + candidates_unit * np.float32(cfg.step_voxels)
        sampled_normals = _sample_candidate_normals_torch(
            normal_sampler,
            next_points,
            device=_cache_device(cache),
        )
        candidate_normals = None if sampled_normals is None else sampled_normals[0]
        candidate_normals_valid = None if sampled_normals is None else sampled_normals[1]
        best_index, _total, direction_loss, presence_loss, smoothness_loss, rejected = (
            _score_candidate_batch(
                cache,
                current_direction=current_direction,
                previous_step_direction=previous_direction,
                candidate_directions=candidates_unit,
                next_points=next_points,
                current_point=current,
                step_voxels=float(cfg.step_voxels),
                candidate_substeps=int(cfg.candidate_substeps),
                smoothness_weight=float(cfg.smoothness_weight),
                smoothness_tangent_weight=cfg.smoothness_tangent_weight,
                smoothness_normal_weight=cfg.smoothness_normal_weight,
                smoothness_free_angle_degrees=float(cfg.smoothness_free_angle_degrees),
                cumulative_smoothness_tangent_weight=float(
                    cfg.cumulative_smoothness_tangent_weight
                ),
                all_pairs_direction_product=bool(cfg.all_pairs_direction_product),
                candidate_normals=candidate_normals,
                candidate_normals_valid=candidate_normals_valid,
                history_direction=history_direction,
                first_step=(_step_index == 0),
            )
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
                    smoothness_loss=float(smoothness_loss),
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
                smoothness_loss=float(smoothness_loss),
            )
        )
        history_direction = _update_native_history_direction_np(
            history_direction,
            chosen_direction,
            depth=_step_index,
            cumulative_smoothness_steps=int(cfg.cumulative_smoothness_steps),
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


def _beam_node_result(
    node: _NativeBeamNode,
    *,
    reached_target_plane: bool,
    reason: str,
) -> NativeTraceResult:
    nodes: list[_NativeBeamNode] = []
    current: _NativeBeamNode | None = node
    while current is not None:
        nodes.append(current)
        current = current.parent
    nodes.reverse()
    trace = np.stack([np.asarray(item.point_zyx, dtype=np.float32) for item in nodes], axis=0)
    steps = tuple(item.step for item in nodes[1:] if item.step is not None)
    return NativeTraceResult(
        trace_zyx=trace.astype(np.float32, copy=False),
        reached_target_plane=bool(reached_target_plane),
        reason=reason,
        steps=steps,
    )


def _prune_native_beam_nodes(
    nodes: list[_NativeBeamNode],
    *,
    beam_width: int,
    prune_distance_voxels: float,
) -> list[_NativeBeamNode]:
    if not nodes:
        return []
    width = max(1, int(beam_width))
    distance = max(0.0, float(prune_distance_voxels))
    ordered = sorted(nodes, key=lambda item: (float(item.cumulative_loss), int(item.depth)))
    kept: list[_NativeBeamNode] = []
    for node in ordered:
        if distance > 0.0 and any(
            float(
                np.linalg.norm(
                    np.asarray(node.point_zyx, dtype=np.float64)
                    - np.asarray(existing.point_zyx, dtype=np.float64)
                )
            )
            < distance
            for existing in kept
        ):
            continue
        kept.append(node)
        if len(kept) >= width:
            break
    if kept:
        return kept
    return ordered[:width]


def _prune_native_beam_tensor_indices(
    generation: _NativeBeamTensorGeneration,
    *,
    beam_width: int,
    prune_distance_voxels: float,
) -> torch.Tensor:
    count = int(generation.points_zyx.shape[0])
    if count == 0:
        return torch.zeros((0,), dtype=torch.long, device=generation.points_zyx.device)
    width = max(1, int(beam_width))
    distance = max(0.0, float(prune_distance_voxels))
    score = generation.cumulative_loss.to(dtype=torch.float64)
    score = score + generation.depth.to(dtype=torch.float64) * 1.0e-12
    if distance <= 0.0:
        keep_count = min(width, count)
        return torch.topk(score, k=keep_count, largest=False, sorted=True).indices.to(dtype=torch.long)

    available = torch.isfinite(score)
    kept: list[torch.Tensor] = []
    distance2 = float(distance) * float(distance)
    for _ in range(width):
        masked_score = torch.where(
            available,
            score,
            torch.full_like(score, torch.inf),
        )
        best_index = torch.argmin(masked_score)
        if not bool(torch.isfinite(masked_score[best_index]).detach().cpu()):
            break
        kept.append(best_index.to(dtype=torch.long))
        delta = generation.points_zyx - generation.points_zyx[best_index].view(1, 3)
        far_enough = torch.sum(delta * delta, dim=1) >= distance2
        available = available & far_enough
    if kept:
        return torch.stack(kept, dim=0).to(dtype=torch.long)
    return torch.argmin(score).view(1).to(dtype=torch.long)


def _native_beam_tensor_node(
    *,
    generations: list[_NativeBeamTensorGeneration],
    root_nodes: list[_NativeBeamNode],
    generation_index: int,
    state_index: int,
) -> _NativeBeamNode:
    if generation_index < 0 or generation_index >= len(generations):
        raise ValueError("generation_index is out of range")
    chain: list[tuple[int, int]] = []
    idx = int(state_index)
    for gen_idx in range(int(generation_index), 0, -1):
        gen = generations[gen_idx]
        if gen.parent_indices is None:
            raise ValueError("non-root tensor generation is missing parent indices")
        chain.append((gen_idx, idx))
        idx = int(gen.parent_indices[idx].detach().cpu())
    if idx < 0 or idx >= len(root_nodes):
        raise ValueError("root tensor state index is out of range")
    node = root_nodes[idx]
    for gen_idx, item_idx in reversed(chain):
        gen = generations[gen_idx]
        if (
            gen.step_direction_loss is None
            or gen.step_presence_loss is None
            or gen.step_total_loss is None
            or gen.step_smoothness_loss is None
            or gen.step_rejected_candidates is None
        ):
            raise ValueError("tensor generation is missing step diagnostics")
        point = gen.points_zyx[item_idx].detach().cpu().numpy().astype(np.float32)
        previous_direction = (
            gen.previous_directions_zyx[item_idx].detach().cpu().numpy().astype(np.float32)
        )
        history_direction = (
            gen.history_directions_zyx[item_idx].detach().cpu().numpy().astype(np.float32)
        )
        step = NativeTraceStep(
            point_zyx=point,
            direction_loss=float(gen.step_direction_loss[item_idx].detach().cpu()),
            presence_loss=float(gen.step_presence_loss[item_idx].detach().cpu()),
            total_loss=float(gen.step_total_loss[item_idx].detach().cpu()),
            rejected_candidates=int(gen.step_rejected_candidates[item_idx].detach().cpu()),
            smoothness_loss=float(gen.step_smoothness_loss[item_idx].detach().cpu()),
        )
        node = _NativeBeamNode(
            point_zyx=point,
            previous_direction_zyx=previous_direction,
            history_direction_zyx=history_direction,
            parent=node,
            step=step,
            cumulative_loss=float(gen.cumulative_loss[item_idx].detach().cpu()),
            depth=int(gen.depth[item_idx].detach().cpu()),
        )
    return node


def _trace_native_3d_one_way_beam(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_plane_normal_zyx: np.ndarray | None = None,
    budget_span_voxels: float | None = None,
    progress_label: str | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTraceResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    plane_normal = (
        _unit(target - start)
        if target_plane_normal_zyx is None
        else _require_unit(target_plane_normal_zyx, label="target_plane_normal_zyx")
    )
    span = float(np.linalg.norm(target - start))
    if span <= _EPS:
        raise ValueError("native 3D Trace2CP start and target CPs must differ")
    budget_span = span if budget_span_voxels is None else float(budget_span_voxels)
    if not math.isfinite(budget_span) or budget_span <= _EPS:
        raise ValueError("budget_span_voxels must be positive when set")
    beam_width = int(cfg.beam_width)
    if beam_width <= 1:
        raise ValueError("_trace_native_3d_one_way_beam requires beam_width > 1")
    if not math.isfinite(float(cfg.beam_prune_distance_voxels)) or float(cfg.beam_prune_distance_voxels) < 0.0:
        raise ValueError("beam_prune_distance_voxels must be finite and non-negative")
    lookahead_steps = int(cfg.beam_lookahead_steps)
    if lookahead_steps <= 0:
        raise ValueError("beam_lookahead_steps must be positive")
    step_limit, limit_reason = _native_trace_step_limit(
        span_voxels=budget_span,
        cfg=cfg,
    )
    progress_max = max(1, step_limit)
    last_progress_time = 0.0
    trace_start_time = time.perf_counter()

    def node_progress(point_zyx: np.ndarray) -> float:
        return float(
            np.dot(np.asarray(point_zyx, dtype=np.float32) - start, plane_normal)
            / max(span, _EPS)
        )

    def emit_progress(
        point_zyx: np.ndarray,
        step: int,
        *,
        reason: str | None = None,
        active_beams: int = 0,
    ) -> None:
        nonlocal last_progress_time
        if progress_label is None:
            return
        now = time.perf_counter()
        if reason is None and step > 0 and now - last_progress_time < 0.25:
            return
        last_progress_time = now
        progress = float(np.clip(node_progress(point_zyx), 0.0, 1.0))
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
            f"eta={_format_eta(eta)} blocks={len(cache._blocks)} "
            f"beams={int(active_beams)}/{beam_width} lookahead={lookahead_steps} "
            f"substeps={int(cfg.candidate_substeps)}{suffix}",
            end=end,
            flush=True,
        )

    emit_progress(start, 0, active_beams=1)
    initial_sampled_direction, _presence, valid = _sample_trace_point_aligned(
        cache,
        start,
        reference_direction_zyx=initial_direction_zyx,
    )
    if not valid:
        raise ValueError(f"native 3D Trace2CP start point is invalid: {start.tolist()}")
    _ = initial_sampled_direction
    initial_direction = _require_unit(
        initial_direction_zyx,
        label="native 3D Trace2CP initial_direction_zyx",
    )
    start_node = _NativeBeamNode(
        point_zyx=start.astype(np.float32),
        previous_direction_zyx=initial_direction.astype(np.float32, copy=False),
        history_direction_zyx=initial_direction.astype(np.float32, copy=False),
        parent=None,
        step=None,
        cumulative_loss=0.0,
        depth=0,
    )
    live: list[_NativeBeamNode] = [start_node]
    best_live = start_node
    committed_step = 0
    device = _cache_device(cache)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    plane_normal_t = torch.as_tensor(plane_normal, dtype=torch.float32, device=device)
    initial_direction_t = torch.as_tensor(
        initial_direction.reshape(1, 3),
        dtype=torch.float32,
        device=device,
    )
    while committed_step < step_limit:
        root_points = torch.as_tensor(
            np.stack([np.asarray(node.point_zyx, dtype=np.float32) for node in live], axis=0),
            dtype=torch.float32,
            device=device,
        )
        root_previous = torch.as_tensor(
            np.stack(
                [np.asarray(node.previous_direction_zyx, dtype=np.float32) for node in live],
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )
        root_history = torch.as_tensor(
            np.stack(
                [np.asarray(node.history_direction_zyx, dtype=np.float32) for node in live],
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )
        root_cumulative = torch.as_tensor(
            [float(node.cumulative_loss) for node in live],
            dtype=torch.float32,
            device=device,
        )
        root_depth = torch.as_tensor(
            [int(node.depth) for node in live],
            dtype=torch.long,
            device=device,
        )
        generations: list[_NativeBeamTensorGeneration] = [
            _NativeBeamTensorGeneration(
                points_zyx=root_points,
                previous_directions_zyx=F.normalize(
                    root_previous,
                    p=2.0,
                    dim=1,
                    eps=float(_EPS),
                ),
                history_directions_zyx=F.normalize(
                    root_history,
                    p=2.0,
                    dim=1,
                    eps=float(_EPS),
                ),
                cumulative_loss=root_cumulative,
                depth=root_depth,
                parent_indices=None,
                step_direction_loss=None,
                step_presence_loss=None,
                step_total_loss=None,
                step_smoothness_loss=None,
                step_rejected_candidates=None,
            )
        ]
        frontier_generation_index = 0
        reached_generation_index: int | None = None
        reached_state_index: int | None = None
        expanded_steps = 0
        max_expand = min(lookahead_steps, step_limit - committed_step)
        for lookahead_index in range(max_expand):
            frontier_gen = generations[frontier_generation_index]
            current_points = frontier_gen.points_zyx
            previous_directions = F.normalize(
                frontier_gen.previous_directions_zyx,
                p=2.0,
                dim=1,
                eps=float(_EPS),
            )
            history_directions = F.normalize(
                frontier_gen.history_directions_zyx,
                p=2.0,
                dim=1,
                eps=float(_EPS),
            )
            current_directions, _current_presence, state_valid = _sample_trace_points_aligned_torch(
                cache,
                current_points,
                reference_directions_zyx=previous_directions,
            )
            root_mask = frontier_gen.depth == 0
            if bool(torch.any(root_mask).detach().cpu()):
                current_directions = current_directions.clone()
                state_valid = state_valid.clone()
                current_directions[root_mask] = initial_direction_t.expand(
                    int(current_directions.shape[0]),
                    3,
                )[root_mask]
                state_valid[root_mask] = True
            valid_state_indices = torch.nonzero(state_valid, as_tuple=False).flatten()
            if int(valid_state_indices.numel()) == 0:
                break
            current_points_v = current_points[valid_state_indices]
            previous_directions_v = previous_directions[valid_state_indices]
            history_directions_v = history_directions[valid_state_indices]
            current_directions_v = F.normalize(
                current_directions[valid_state_indices],
                p=2.0,
                dim=1,
                eps=float(_EPS),
            )
            candidate_dirs = _trace_candidate_directions_torch(current_directions_v, cfg)
            next_points = current_points_v[:, None, :] + candidate_dirs * float(cfg.step_voxels)
            sampled_normals = _sample_candidate_normals_torch(
                normal_sampler,
                next_points,
                device=device,
            )
            candidate_normals_t = None if sampled_normals is None else sampled_normals[0]
            candidate_normals_valid_t = None if sampled_normals is None else sampled_normals[1]
            first_step_mask_t = frontier_gen.depth[valid_state_indices] == 0
            (
                total_loss_t,
                direction_loss_t,
                presence_loss_t,
                smoothness_loss_t,
                candidate_valid_t,
                rejected_per_state_t,
            ) = _score_candidate_loss_tensors_batched(
                cache,
                current_directions=current_directions_v,
                previous_step_directions=previous_directions_v,
                candidate_directions=candidate_dirs,
                next_points=next_points,
                current_points=current_points_v,
                step_voxels=float(cfg.step_voxels),
                candidate_substeps=int(cfg.candidate_substeps),
                smoothness_weight=float(cfg.smoothness_weight),
                smoothness_tangent_weight=cfg.smoothness_tangent_weight,
                smoothness_normal_weight=cfg.smoothness_normal_weight,
                smoothness_free_angle_degrees=float(cfg.smoothness_free_angle_degrees),
                cumulative_smoothness_tangent_weight=float(
                    cfg.cumulative_smoothness_tangent_weight
                ),
                all_pairs_direction_product=bool(cfg.all_pairs_direction_product),
                candidate_normals=candidate_normals_t,
                candidate_normals_valid=candidate_normals_valid_t,
                history_directions=history_directions_v,
                first_step_mask=first_step_mask_t,
            )
            candidate_best_loss_t, candidate_best_branch_t = torch.min(total_loss_t, dim=2)
            candidate_valid_t = candidate_valid_t & torch.isfinite(candidate_best_loss_t)
            child_local_state, child_candidate = torch.nonzero(
                candidate_valid_t,
                as_tuple=True,
            )
            if int(child_local_state.numel()) == 0:
                break
            child_parent_indices = valid_state_indices[child_local_state]
            child_branch = candidate_best_branch_t[child_local_state, child_candidate]
            child_total_loss = candidate_best_loss_t[child_local_state, child_candidate]
            child_direction_loss = direction_loss_t[
                child_local_state,
                child_candidate,
                child_branch,
            ]
            child_presence_loss = presence_loss_t[
                child_local_state,
                child_candidate,
                child_branch,
            ]
            child_smoothness_loss = smoothness_loss_t[child_local_state, child_candidate]
            child_rejected = rejected_per_state_t[child_local_state].to(dtype=torch.long)
            chosen_directions = _align_axes_torch(
                candidate_dirs[child_local_state, child_candidate],
                current_directions_v[child_local_state],
            )
            child_next_points = (
                current_points_v[child_local_state]
                + chosen_directions * float(cfg.step_voxels)
            )
            child_history = _update_native_history_direction_torch(
                history_directions_v[child_local_state],
                chosen_directions,
                frontier_gen.depth[child_parent_indices],
                cumulative_smoothness_steps=int(cfg.cumulative_smoothness_steps),
            )
            d0 = torch.sum(
                (current_points_v[child_local_state] - target_t.view(1, 3))
                * plane_normal_t.view(1, 3),
                dim=1,
            )
            d1 = torch.sum(
                (child_next_points - target_t.view(1, 3)) * plane_normal_t.view(1, 3),
                dim=1,
            )
            reached_mask = (d0 == 0.0) | (d0 * d1 <= 0.0)
            denom = d0 - d1
            safe_denom = torch.where(
                torch.abs(denom) > float(_EPS),
                denom,
                torch.ones_like(denom),
            )
            crossing_t = torch.where(
                torch.abs(denom) > float(_EPS),
                torch.clamp(d0 / safe_denom, 0.0, 1.0),
                torch.ones_like(d0),
            )
            crossing_t = torch.where(d0 == 0.0, torch.zeros_like(crossing_t), crossing_t)
            crossing_points = (
                current_points_v[child_local_state] * (1.0 - crossing_t[:, None])
                + child_next_points * crossing_t[:, None]
            )
            child_points = torch.where(
                reached_mask[:, None],
                crossing_points,
                child_next_points,
            )
            child_generation = _NativeBeamTensorGeneration(
                points_zyx=child_points,
                previous_directions_zyx=chosen_directions,
                history_directions_zyx=child_history,
                cumulative_loss=frontier_gen.cumulative_loss[child_parent_indices]
                + child_total_loss,
                depth=frontier_gen.depth[child_parent_indices] + 1,
                parent_indices=child_parent_indices.to(dtype=torch.long),
                step_direction_loss=child_direction_loss,
                step_presence_loss=child_presence_loss,
                step_total_loss=child_total_loss,
                step_smoothness_loss=child_smoothness_loss,
                step_rejected_candidates=child_rejected,
            )
            generations.append(child_generation)
            frontier_generation_index = len(generations) - 1
            expanded_steps = lookahead_index + 1
            if bool(torch.any(reached_mask).detach().cpu()):
                reached_indices = torch.nonzero(reached_mask, as_tuple=False).flatten()
                reached_loss = child_generation.cumulative_loss[reached_indices]
                best_reached_local = int(torch.argmin(reached_loss).detach().cpu())
                reached_generation_index = frontier_generation_index
                reached_state_index = int(reached_indices[best_reached_local].detach().cpu())
                break
        if reached_generation_index is not None and reached_state_index is not None:
            best = _native_beam_tensor_node(
                generations=generations,
                root_nodes=live,
                generation_index=reached_generation_index,
                state_index=reached_state_index,
            )
            emit_progress(
                best.point_zyx,
                min(step_limit, committed_step + expanded_steps),
                reason="target_plane",
                active_beams=len(live),
            )
            return _beam_node_result(best, reached_target_plane=True, reason="target_plane")
        if expanded_steps == 0 or frontier_generation_index == 0:
            emit_progress(
                best_live.point_zyx,
                committed_step,
                reason="all_candidates_invalid",
                active_beams=0,
            )
            return _beam_node_result(
                best_live,
                reached_target_plane=False,
                reason="all_candidates_invalid",
            )
        frontier = generations[frontier_generation_index]
        kept_indices = _prune_native_beam_tensor_indices(
            frontier,
            beam_width=beam_width,
            prune_distance_voxels=float(cfg.beam_prune_distance_voxels),
        )
        live = [
            _native_beam_tensor_node(
                generations=generations,
                root_nodes=live,
                generation_index=frontier_generation_index,
                state_index=int(index.detach().cpu()),
            )
            for index in kept_indices
        ]
        if not live:
            emit_progress(
                best_live.point_zyx,
                committed_step,
                reason="all_candidates_invalid",
                active_beams=0,
            )
            return _beam_node_result(
                best_live,
                reached_target_plane=False,
                reason="all_candidates_invalid",
            )
        committed_step += max(1, expanded_steps)
        best_live = min(live, key=lambda item: (float(item.cumulative_loss), int(item.depth)))
        progress_node = max(live, key=lambda item: node_progress(item.point_zyx))
        emit_progress(progress_node.point_zyx, committed_step, active_beams=len(live))
    emit_progress(best_live.point_zyx, progress_max, reason=limit_reason, active_beams=len(live))
    return _beam_node_result(
        best_live,
        reached_target_plane=False,
        reason=limit_reason,
    )


def trace_native_3d_one_way(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_plane_normal_zyx: np.ndarray | None = None,
    budget_span_voxels: float | None = None,
    progress_label: str | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTraceResult:
    if int(cfg.candidate_substeps) < 1:
        raise ValueError("candidate_substeps must be at least 1")
    cfg = _native_trace_cfg_with_effective_smoothness(cfg, normal_sampler=normal_sampler)
    if int(cfg.beam_width) <= 1:
        return _trace_native_3d_one_way_greedy(
            cache,
            start_zyx=start_zyx,
            target_zyx=target_zyx,
            initial_direction_zyx=initial_direction_zyx,
            cfg=cfg,
            target_plane_normal_zyx=target_plane_normal_zyx,
            budget_span_voxels=budget_span_voxels,
            progress_label=progress_label,
            normal_sampler=normal_sampler,
        )
    return _trace_native_3d_one_way_beam(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        initial_direction_zyx=initial_direction_zyx,
        cfg=cfg,
        target_plane_normal_zyx=target_plane_normal_zyx,
        budget_span_voxels=budget_span_voxels,
        progress_label=progress_label,
        normal_sampler=normal_sampler,
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


def _polyline_cumulative_arclengths_zyx(points_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_zyx must have shape N,3")
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if points.shape[0] == 1:
        return np.zeros((1,), dtype=np.float64)
    deltas = np.diff(points.astype(np.float64), axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(lengths)]).astype(np.float64)


def _deduplicate_polyline_zyx(points_zyx: np.ndarray) -> np.ndarray:
    points = np.asarray(points_zyx, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_zyx must have shape N,3")
    if points.shape[0] <= 1:
        return points.astype(np.float32, copy=True)
    lengths = np.linalg.norm(np.diff(points.astype(np.float64), axis=0), axis=1)
    keep = np.concatenate([[True], lengths > 1.0e-8])
    return points[keep].astype(np.float32, copy=True)


def _resample_polyline_with_lengths_zyx(
    points_zyx: np.ndarray,
    *,
    step_voxels: float,
) -> tuple[np.ndarray, np.ndarray]:
    points = _deduplicate_polyline_zyx(points_zyx)
    if points.shape[0] == 0:
        return points.astype(np.float32), np.zeros((0,), dtype=np.float64)
    cumulative = _polyline_cumulative_arclengths_zyx(points)
    total = float(cumulative[-1])
    if points.shape[0] <= 2 or not math.isfinite(total) or total <= 1.0e-8:
        return points.astype(np.float32, copy=True), cumulative.astype(np.float64, copy=True)
    stride = max(float(step_voxels), _EPS)
    sample_s = np.arange(0.0, total, stride, dtype=np.float64)
    if sample_s.size == 0 or abs(float(sample_s[-1]) - total) > 1.0e-8:
        sample_s = np.concatenate([sample_s, np.asarray([total], dtype=np.float64)])
    else:
        sample_s[-1] = total
    sampled = np.stack(
        [np.interp(sample_s, cumulative, points[:, axis]) for axis in range(3)],
        axis=1,
    ).astype(np.float32)
    sampled[0] = points[0]
    sampled[-1] = points[-1]
    return sampled, sample_s.astype(np.float64, copy=False)


def _warp_partial_trace_to_midpoint_by_arclength(
    partial_zyx: np.ndarray,
    *,
    anchor_zyx: np.ndarray,
    source_meet_zyx: np.ndarray,
    target_midpoint_zyx: np.ndarray,
) -> np.ndarray:
    partial = np.asarray(partial_zyx, dtype=np.float32)
    if partial.ndim != 2 or partial.shape[1] != 3 or partial.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if partial.shape[0] == 1:
        partial = np.stack(
            [
                np.asarray(anchor_zyx, dtype=np.float32),
                np.asarray(source_meet_zyx, dtype=np.float32),
            ],
            axis=0,
        )
    warped = partial.astype(np.float64, copy=True)
    arclengths = _polyline_cumulative_arclengths_zyx(warped.astype(np.float32))
    total = float(arclengths[-1]) if arclengths.size else 0.0
    if total <= 1.0e-8:
        blend = np.linspace(0.0, 1.0, warped.shape[0], dtype=np.float64)
    else:
        blend = np.clip(arclengths / total, 0.0, 1.0)
    delta = (
        np.asarray(target_midpoint_zyx, dtype=np.float64)
        - np.asarray(source_meet_zyx, dtype=np.float64)
    )
    warped += blend[:, None] * delta[None, :]
    warped[0] = np.asarray(anchor_zyx, dtype=np.float64)
    warped[-1] = np.asarray(target_midpoint_zyx, dtype=np.float64)
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
    points = _deduplicate_polyline_zyx(points)
    if points.shape[0] <= 2:
        return points.astype(np.float32, copy=True)
    cumulative = _polyline_cumulative_arclengths_zyx(points)
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
    forward_dense, forward_arclengths = _resample_polyline_with_lengths_zyx(
        forward,
        step_voxels=step_voxels,
    )
    reverse_dense, reverse_arclengths = _resample_polyline_with_lengths_zyx(
        reverse,
        step_voxels=step_voxels,
    )
    finite_forward = np.isfinite(forward_dense).all(axis=1) & np.isfinite(forward_arclengths)
    finite_reverse = np.isfinite(reverse_dense).all(axis=1) & np.isfinite(reverse_arclengths)
    if not bool(np.any(finite_forward)) or not bool(np.any(finite_reverse)):
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
            reason="nonfinite_trace_points",
        )

    forward_valid_indices = np.nonzero(finite_forward)[0]
    reverse_valid_indices = np.nonzero(finite_reverse)[0]
    forward_points = forward_dense[forward_valid_indices].astype(np.float64, copy=False)
    reverse_points = reverse_dense[reverse_valid_indices].astype(np.float64, copy=False)
    forward_lengths = forward_arclengths[forward_valid_indices].astype(np.float64, copy=False)
    reverse_lengths = reverse_arclengths[reverse_valid_indices].astype(np.float64, copy=False)

    gap_factor = 2.0
    best: tuple[float, float, float, float, int, int] | None = None
    # Keep the pairwise gap matrix bounded while still vectorizing each chunk.
    max_gap_values = 2_000_000
    chunk_size = max(1, int(max_gap_values // max(int(reverse_points.shape[0]), 1)))
    for chunk_start in range(0, int(forward_points.shape[0]), chunk_size):
        chunk_end = min(chunk_start + chunk_size, int(forward_points.shape[0]))
        forward_chunk = forward_points[chunk_start:chunk_end]
        forward_length_chunk = forward_lengths[chunk_start:chunk_end]
        gaps = np.linalg.norm(
            forward_chunk[:, None, :] - reverse_points[None, :, :],
            axis=2,
        )
        combined_lengths = forward_length_chunk[:, None] + reverse_lengths[None, :]
        scores = gap_factor * gaps + combined_lengths
        finite_scores = np.isfinite(scores)
        if not bool(np.any(finite_scores)):
            continue
        min_score = float(np.min(scores[finite_scores]))
        flat_candidates = np.flatnonzero(
            np.isclose(scores.reshape(-1), min_score, rtol=0.0, atol=1.0e-9)
        )
        if flat_candidates.size == 0:
            flat_candidates = np.asarray([int(np.argmin(scores.reshape(-1)))], dtype=np.int64)
        flat_scores = scores.reshape(-1)[flat_candidates]
        flat_gaps = gaps.reshape(-1)[flat_candidates]
        flat_lengths = combined_lengths.reshape(-1)[flat_candidates]
        min_lengths = np.minimum(forward_length_chunk[:, None], reverse_lengths[None, :])
        flat_min_lengths = min_lengths.reshape(-1)[flat_candidates]
        local_choice = int(
            np.lexsort((-flat_lengths, -flat_min_lengths, flat_gaps, flat_scores))[0]
        )
        flat_index = int(flat_candidates[local_choice])
        local_forward_index, local_reverse_index = np.unravel_index(flat_index, scores.shape)
        score = float(scores[local_forward_index, local_reverse_index])
        gap = float(gaps[local_forward_index, local_reverse_index])
        combined_length = float(combined_lengths[local_forward_index, local_reverse_index])
        min_length = float(min_lengths[local_forward_index, local_reverse_index])
        forward_index = int(forward_valid_indices[chunk_start + int(local_forward_index)])
        reverse_index = int(reverse_valid_indices[int(local_reverse_index)])
        key = (score, gap, -min_length, -combined_length)
        if best is None or key < (best[0], best[1], -best[2], -best[3]):
            best = (score, gap, min_length, combined_length, forward_index, reverse_index)
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
            reason="no_pairwise_trace_meeting",
        )
    considered_gap, raw_gap, _min_length, _combined_length, forward_index, reverse_index = best
    closest_forward = forward_dense[int(forward_index)].astype(np.float32)
    closest_reverse = reverse_dense[int(reverse_index)].astype(np.float32)
    midpoint = ((closest_forward.astype(np.float64) + closest_reverse.astype(np.float64)) * 0.5).astype(np.float32)
    closest_progress = float(
        _trace_progress(
            midpoint[None, :],
            start_zyx=start,
            axis_zyx=axis,
            span_voxels=span,
        )[0]
    )
    forward_partial = forward_dense[: int(forward_index) + 1].astype(np.float32, copy=True)
    reverse_partial = reverse_dense[: int(reverse_index) + 1].astype(np.float32, copy=True)
    if forward_partial.shape[0] == 0:
        forward_partial = start[None, :].astype(np.float32)
    if reverse_partial.shape[0] == 0:
        reverse_partial = target[None, :].astype(np.float32)
    forward_partial[0] = start
    forward_partial[-1] = closest_forward
    reverse_partial[0] = target
    reverse_partial[-1] = closest_reverse
    forward_warped = _warp_partial_trace_to_midpoint_by_arclength(
        forward_partial,
        anchor_zyx=start,
        source_meet_zyx=closest_forward,
        target_midpoint_zyx=midpoint,
    )
    reverse_warped = _warp_partial_trace_to_midpoint_by_arclength(
        reverse_partial,
        anchor_zyx=target,
        source_meet_zyx=closest_reverse,
        target_midpoint_zyx=midpoint,
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
        center_penalty=1.0,
        closest_midpoint_zyx=midpoint.astype(np.float32),
        closest_forward_zyx=closest_forward.astype(np.float32),
        closest_reverse_zyx=closest_reverse.astype(np.float32),
        reached_overlap=True,
        reason="pairwise_arc_length_meeting",
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
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTracePairResult:
    forward = trace_native_3d_one_way(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        initial_direction_zyx=forward_initial_direction_zyx,
        cfg=cfg,
        progress_label="fw" if progress else None,
        normal_sampler=normal_sampler,
    )
    reverse = trace_native_3d_one_way(
        cache,
        start_zyx=target_zyx,
        target_zyx=start_zyx,
        initial_direction_zyx=reverse_initial_direction_zyx,
        cfg=cfg,
        progress_label="bw" if progress else None,
        normal_sampler=normal_sampler,
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


def _terminal_trace_direction(trace_zyx: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    trace = np.asarray(trace_zyx, dtype=np.float32)
    if trace.ndim == 2 and trace.shape[1] == 3 and trace.shape[0] >= 2:
        for index in range(int(trace.shape[0]) - 1, 0, -1):
            delta = trace[index] - trace[index - 1]
            if float(np.linalg.norm(delta.astype(np.float64))) > _EPS:
                return _unit(delta, fallback=fallback)
    return _unit(fallback)


def _record_control_points_selected_zyx(record: Any) -> np.ndarray:
    spacing = float(getattr(record, "volume_spacing_base", 1.0))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume_spacing_base for native whole-fiber trace: {spacing!r}")
    cps = np.asarray(record.fiber.control_points_zyx, dtype=np.float32)
    if cps.ndim != 2 or cps.shape[1] != 3:
        raise ValueError("fiber control_points_zyx must have shape [N,3]")
    return (cps / np.float32(spacing)).astype(np.float32, copy=False)


def _reference_line_arc_lengths_selected(record: Any) -> np.ndarray:
    spacing = float(getattr(record, "volume_spacing_base", 1.0))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume_spacing_base for native whole-fiber arcs: {spacing!r}")
    return (_line_arc_lengths(np.asarray(record.fiber.line_points_xyz, dtype=np.float32)) / spacing).astype(
        np.float64,
        copy=False,
    )


def _control_point_reference_arc_voxels(record: Any) -> np.ndarray:
    cumulative = _reference_line_arc_lengths_selected(record)
    arcs = []
    for cp_index in range(int(record.fiber.control_points_zyx.shape[0])):
        line_index = control_point_line_index(record.fiber, int(cp_index))
        arcs.append(float(cumulative[int(line_index)]))
    return np.asarray(arcs, dtype=np.float64)


def trace_native_3d_whole_fiber(
    cache: NativeTraceFieldCache,
    *,
    record: Any,
    cfg: NativeTrace2CpConfig,
    error_threshold_voxels: float,
    progress: bool = False,
    segment_callback: Callable[[NativeWholeFiberSegmentResult, NativeWholeFiberResult | None], None] | None = None,
    trace_segment_fn: Callable[..., NativeTraceResult] | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeWholeFiberResult:
    cp_points = _record_control_points_selected_zyx(record)
    cp_count = int(cp_points.shape[0])
    if cp_count < 2:
        raise ValueError("native whole-fiber Trace2CP requires at least two control points")
    threshold = float(error_threshold_voxels)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("whole-fiber error threshold must be finite and >= 0")
    arc_by_cp = _control_point_reference_arc_voxels(record)
    segment_count = cp_count - 1
    tracer = trace_native_3d_one_way if trace_segment_fn is None else trace_segment_fn
    run_start = time.perf_counter()
    segments: list[NativeWholeFiberSegmentResult] = []
    stitched_parts: list[np.ndarray] = []
    restart_count = 0
    last_success_cp_index = 0
    current_point = cp_points[0].astype(np.float32)
    current_direction = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=0,
        target_control_point_index=1,
    )

    def emit_progress(segment_index: int, segment: NativeWholeFiberSegmentResult | None = None) -> None:
        if not progress:
            return
        done = int(segment_index)
        elapsed = max(0.0, time.perf_counter() - run_start)
        frac = float(done) / float(max(1, segment_count))
        eta = None if frac <= 1.0e-6 else elapsed * (1.0 - frac) / frac
        status = "pending" if segment is None else ("ok" if segment.success else f"restart:{segment.reason}")
        _emit_native_progress(
            "whole fiber",
            done,
            segment_count,
            run_start,
            detail=(
                f"segment={min(done + 1, segment_count)}/{segment_count} "
                f"status={status} restarts={restart_count} "
                f"rate={restart_count / max(1, done):.6f} "
                f"eta={_format_eta(eta)} blocks={len(cache._blocks)}"
            ),
        )

    emit_progress(0)
    for segment_index in range(segment_count):
        start_cp = int(segment_index)
        target_cp = int(segment_index + 1)
        previous_segment_success = bool(segments and segments[-1].success)
        target_point = cp_points[target_cp].astype(np.float32)
        reference_start = cp_points[start_cp].astype(np.float32)
        segment_axis = _unit(target_point - reference_start, fallback=current_direction)
        segment_span = float(np.linalg.norm(target_point - reference_start))
        if segment_span <= _EPS:
            raise ValueError(f"degenerate native whole-fiber CP span {start_cp}->{target_cp}")
        result = tracer(
            cache,
            start_zyx=current_point.astype(np.float32),
            target_zyx=target_point,
            initial_direction_zyx=current_direction,
            cfg=cfg,
            target_plane_normal_zyx=segment_axis,
            budget_span_voxels=segment_span,
            progress_label=None,
            normal_sampler=normal_sampler,
        )
        crossing = result.trace_zyx[-1].astype(np.float32)
        in_plane_error = (
            _target_plane_in_plane_error_voxels(
                crossing,
                target_zyx=target_point,
                plane_normal_zyx=segment_axis,
            )
            if bool(result.reached_target_plane)
            else float("inf")
        )
        success = bool(result.reached_target_plane) and in_plane_error <= threshold
        if success:
            reason = result.reason
            restart = False
            reference_arc = float(arc_by_cp[target_cp])
            current_point = crossing
            current_direction = _terminal_trace_direction(result.trace_zyx, fallback=current_direction)
            last_success_cp_index = target_cp
        else:
            reason = result.reason if not bool(result.reached_target_plane) else "in_plane_error"
            restart = True
            restart_count += 1
            reference_arc = float(arc_by_cp[last_success_cp_index])
            current_point = target_point
            if target_cp < cp_count - 1:
                current_direction = _fiber_line_tangent_zyx_toward_target(
                    record,
                    start_control_point_index=target_cp,
                    target_control_point_index=target_cp + 1,
                )
        trace = np.asarray(result.trace_zyx, dtype=np.float32)
        if stitched_parts and trace.shape[0] > 0 and previous_segment_success and success:
            stitched_parts.append(trace[1:].copy())
        else:
            stitched_parts.append(trace.copy())
        segment = NativeWholeFiberSegmentResult(
            start_cp_index=start_cp,
            target_cp_index=target_cp,
            trace_zyx=trace,
            start_zyx=np.asarray(result.trace_zyx[0], dtype=np.float32),
            target_zyx=target_point.astype(np.float32),
            reached_target_plane=bool(result.reached_target_plane),
            success=bool(success),
            restart=bool(restart),
            reason=str(reason),
            in_plane_error_voxels=float(in_plane_error),
            reference_arc_distance_voxels=float(reference_arc),
            step_count=int(len(result.steps)),
        )
        segments.append(segment)
        partial = NativeWholeFiberResult(
            segments=tuple(segments),
            restart_count=int(restart_count),
            restart_rate=float(restart_count / max(1, len(segments))),
            segment_count=int(segment_count),
            stitched_trace_zyx=(
                np.concatenate([part for part in stitched_parts if part.size], axis=0).astype(np.float32)
                if any(part.size for part in stitched_parts)
                else np.zeros((0, 3), dtype=np.float32)
            ),
            inferred_blocks=int(len(cache._blocks)),
        )
        if segment_callback is not None:
            segment_callback(segment, partial)
        emit_progress(segment_index + 1, segment)
    stitched = (
        np.concatenate([part for part in stitched_parts if part.size], axis=0).astype(np.float32)
        if any(part.size for part in stitched_parts)
        else np.zeros((0, 3), dtype=np.float32)
    )
    return NativeWholeFiberResult(
        segments=tuple(segments),
        restart_count=int(restart_count),
        restart_rate=float(restart_count / max(1, segment_count)),
        segment_count=int(segment_count),
        stitched_trace_zyx=stitched,
        inferred_blocks=int(len(cache._blocks)),
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


def _trim_failed_overlay_before_target(
    overlay_xy: np.ndarray,
    *,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    margin_px: float = 8.0,
) -> np.ndarray:
    overlay = np.asarray(overlay_xy, dtype=np.float32)
    if overlay.ndim != 2 or overlay.shape[1] != 2 or overlay.shape[0] <= 1:
        return overlay.astype(np.float32, copy=True)
    start_x = float(np.asarray(start_xy, dtype=np.float32)[0])
    target_x = float(np.asarray(target_xy, dtype=np.float32)[0])
    sign = 1.0 if target_x >= start_x else -1.0
    cutoff = target_x - sign * max(0.0, float(margin_px))
    if sign >= 0.0:
        keep = overlay[:, 0] <= cutoff
    else:
        keep = overlay[:, 0] >= cutoff
    keep &= np.isfinite(overlay).all(axis=1)
    if int(np.count_nonzero(keep)) >= 2:
        return overlay[keep].astype(np.float32, copy=False)
    finite = overlay[np.isfinite(overlay).all(axis=1)]
    return finite[: min(2, int(finite.shape[0]))].astype(np.float32, copy=False)


def _trace2cp_source_control_point_xy(
    source: _Trace2CpSegmentSource,
    control_point_index: int,
) -> np.ndarray:
    cp_index = int(control_point_index)
    if cp_index == int(source.start_control_point_index):
        return np.asarray(source.start_control_point_xy, dtype=np.float32)
    if cp_index == int(source.target_control_point_index):
        return np.asarray(source.target_control_point_xy, dtype=np.float32)
    line_index = control_point_line_index(source.record.fiber, cp_index)
    matches = np.flatnonzero(np.asarray(source.line_point_indices, dtype=np.int64) == int(line_index))
    if matches.size == 0:
        raise ValueError(
            "Trace2CP source does not contain requested control point line index: "
            f"control_point_index={cp_index} line_index={int(line_index)} "
            f"source_start_cp={int(source.start_control_point_index)} "
            f"source_target_cp={int(source.target_control_point_index)}"
        )
    return np.asarray(source.line_xy[int(matches[0])], dtype=np.float32)


def _whole_fiber_segment_overlays_for_view(
    source: _Trace2CpSegmentSource,
    segment: NativeWholeFiberSegmentResult,
    *,
    axis_name: str,
) -> tuple[tuple[np.ndarray, tuple[int, int, int, int]], ...]:
    spacing = float(source.record.volume_spacing_base)
    trace_xyz = _trace_zyx_to_base_xyz(segment.trace_zyx, spacing)
    xy = _project_trace_to_initial_strip(source, trace_xyz, axis_name=axis_name)
    if int(xy.shape[0]) < 2:
        return ()
    if not segment.success:
        start_xy = _trace2cp_source_control_point_xy(source, int(segment.start_cp_index))
        target_xy = _trace2cp_source_control_point_xy(source, int(segment.target_cp_index))
        xy = _trim_failed_overlay_before_target(
            xy,
            start_xy=start_xy,
            target_xy=target_xy,
        )
    if int(xy.shape[0]) < 2:
        return ()
    color = (255, 220, 0, 235) if segment.success else (255, 80, 64, 235)
    return ((xy.astype(np.float32, copy=False), color),)


def _whole_fiber_segment_group_overlays_for_view(
    source: _Trace2CpSegmentSource,
    segments: tuple[NativeWholeFiberSegmentResult, ...],
    *,
    axis_name: str,
) -> tuple[tuple[np.ndarray, tuple[int, int, int, int]], ...]:
    overlays: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []
    for segment in segments:
        overlays.extend(
            _whole_fiber_segment_overlays_for_view(
                source,
                segment,
                axis_name=axis_name,
            )
        )
    return tuple(overlays)


def _native_whole_fiber_visual_spans(
    segments: tuple[NativeWholeFiberSegmentResult, ...] | list[NativeWholeFiberSegmentResult],
) -> tuple[_NativeWholeFiberVisualSpan, ...]:
    spans: list[_NativeWholeFiberVisualSpan] = []
    active: list[NativeWholeFiberSegmentResult] = []
    active_start: int | None = None
    for segment in segments:
        if active_start is None:
            active_start = int(segment.start_cp_index)
        active.append(segment)
        if not bool(segment.success):
            spans.append(
                _NativeWholeFiberVisualSpan(
                    start_cp_index=int(active_start),
                    end_cp_index=int(segment.target_cp_index),
                    segments=tuple(active),
                    restart_after=True,
                )
            )
            active = []
            active_start = int(segment.target_cp_index)
    if active:
        assert active_start is not None
        spans.append(
            _NativeWholeFiberVisualSpan(
                start_cp_index=int(active_start),
                end_cp_index=int(active[-1].target_cp_index),
                segments=tuple(active),
                restart_after=False,
            )
        )
    return tuple(
        span
        for span in spans
        if int(span.start_cp_index) != int(span.end_cp_index) and span.segments
    )


def _compose_whole_fiber_panel_blocks(
    panel_blocks: list[tuple[Any, Any, Any, Any]],
    *,
    status_text: str | None = None,
):
    from PIL import Image, ImageDraw

    if not panel_blocks:
        sheet = Image.new("RGBA", (760, 96), (0, 0, 0, 255))
        draw = ImageDraw.Draw(sheet, "RGBA")
        draw.text((8, 8), "native 3D whole-fiber Trace2CP", fill=(255, 255, 255, 255))
        draw.text((8, 34), "waiting for first segment", fill=(180, 180, 180, 255))
        if status_text:
            draw.text((8, 60), status_text, fill=(120, 220, 255, 255))
        return sheet
    row_count = 4
    row_heights = [
        max(int(block[row].height) for block in panel_blocks)
        for row in range(row_count)
    ]
    separator_width = 12
    block_widths = [max(int(panel.width) for panel in block) for block in panel_blocks]
    width = int(sum(block_widths) + separator_width * max(0, len(panel_blocks) - 1))
    height = int(sum(row_heights))
    sheet = Image.new("RGBA", (max(1, width), max(1, height)), (0, 0, 0, 255))
    x = 0
    for block_index, (block, block_width) in enumerate(zip(panel_blocks, block_widths)):
        y = 0
        for row, panel in enumerate(block):
            sheet.alpha_composite(panel, (x, y))
            y += row_heights[row]
        x += int(block_width)
        if block_index + 1 < len(panel_blocks):
            x += separator_width
    return sheet


def _build_native_whole_fiber_span_source(
    geometry_loader: Any,
    *,
    start_cp_index: int,
    end_cp_index: int,
    trace2cp_rf_margin_px: float,
    strip_cross_width_px: int = 64,
):
    if int(start_cp_index) == int(end_cp_index):
        raise ValueError(
            "native whole-fiber visual span must contain at least one CP segment: "
            f"start_cp={int(start_cp_index)} end_cp={int(end_cp_index)}"
        )
    return geometry_loader.build_trace2cp_segment_source(
        int(start_cp_index),
        target_control_point_index=int(end_cp_index),
        rf_margin_px=float(trace2cp_rf_margin_px),
        cross_strip_height_px=int(strip_cross_width_px),
        device=torch.device("cpu"),
        sample_mode="flat",
    )


def _render_native_whole_fiber_span_panels(
    geometry_loader: Any,
    *,
    span: _NativeWholeFiberVisualSpan,
    trace2cp_rf_margin_px: float,
    cache: NativeTraceFieldCache,
    image_normalization: str,
    strip_cross_width_px: int = 64,
) -> tuple[Any, Any, Any, Any]:
    source = _build_native_whole_fiber_span_source(
        geometry_loader,
        start_cp_index=int(span.start_cp_index),
        end_cp_index=int(span.end_cp_index),
        trace2cp_rf_margin_px=float(trace2cp_rf_margin_px),
        strip_cross_width_px=int(strip_cross_width_px),
    )
    side_overlays = _whole_fiber_segment_group_overlays_for_view(
        source,
        span.segments,
        axis_name="offset_axis_xyz",
    )
    top_overlays = _whole_fiber_segment_group_overlays_for_view(
        source,
        span.segments,
        axis_name="side_axis_xyz",
    )
    _sample, side_image, side_valid = geometry_loader.sample_trace2cp_segment_source(source)
    top_image, top_valid = geometry_loader.sample_trace2cp_top_strip_source(source)
    side_coords_xyz, side_grid_valid = geometry_loader.trace2cp_segment_coords_xyz(source)
    top_coords_xyz, top_grid_valid = geometry_loader.trace2cp_top_strip_coords_xyz(source)
    spacing = float(source.record.volume_spacing_base)
    side_presence, side_presence_valid = _sample_presence_on_strip(
        cache,
        side_coords_xyz,
        np.asarray(side_grid_valid, dtype=bool) & np.asarray(side_valid, dtype=bool),
        spacing_base=spacing,
        progress_label=f"span{span.start_cp_index}-{span.end_cp_index}-side",
    )
    top_presence, top_presence_valid = _sample_presence_on_strip(
        cache,
        top_coords_xyz,
        np.asarray(top_grid_valid, dtype=bool) & np.asarray(top_valid, dtype=bool),
        spacing_base=spacing,
        progress_label=f"span{span.start_cp_index}-{span.end_cp_index}-top",
    )
    failed = any(not bool(segment.success) for segment in span.segments)
    title_suffix = (
        f"cp {span.start_cp_index}->{span.end_cp_index} "
        f"segments={len(span.segments)} {'restart' if span.restart_after or failed else 'ok'}"
    )
    return (
        _draw_trace_panel(
            _image_to_u8(side_image, side_valid, normalization=image_normalization),
            side_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"side input {title_suffix}",
            overlays=side_overlays,
        ),
        _draw_trace_panel(
            _presence_to_u8(side_presence, side_presence_valid),
            side_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"side 3D presence {title_suffix}",
            overlays=side_overlays,
        ),
        _draw_trace_panel(
            _image_to_u8(top_image, top_valid, normalization=image_normalization),
            top_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"top input {title_suffix}",
            overlays=top_overlays,
        ),
        _draw_trace_panel(
            _presence_to_u8(top_presence, top_presence_valid),
            top_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"top 3D presence {title_suffix}",
            overlays=top_overlays,
        ),
    )


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


def _native_trace_smoothness_summary(cfg: NativeTrace2CpConfig) -> dict[str, Any]:
    return {
        "smoothness_weight": float(cfg.smoothness_weight),
        "smoothness_tangent_weight": None
        if cfg.smoothness_tangent_weight is None
        else float(cfg.smoothness_tangent_weight),
        "smoothness_normal_weight": None
        if cfg.smoothness_normal_weight is None
        else float(cfg.smoothness_normal_weight),
        "smoothness_normal_aware": bool(
            cfg.smoothness_tangent_weight is not None
            or cfg.smoothness_normal_weight is not None
        ),
        "smoothness_free_angle_degrees": float(cfg.smoothness_free_angle_degrees),
        "cumulative_smoothness_steps": int(cfg.cumulative_smoothness_steps),
        "cumulative_smoothness_tangent_weight": float(
            cfg.cumulative_smoothness_tangent_weight
        ),
        "all_pairs_direction_product": bool(cfg.all_pairs_direction_product),
        "first_step_cp_tangent_relaxed": True,
    }


def run_native_trace2cp(
    config_path: str | Path,
    *,
    checkpoint: str | Path,
    export_dir: str | Path,
    sample_index: int | None,
    fiber_json: str | Path | None = None,
    start_cp_index: int | None = None,
    target_cp_index: int | None = None,
    target_offset: int = 1,
    sample_mode: str | None = None,
    native_cfg: NativeTrace2CpConfig | None = None,
) -> NativeTracePairResult | NativeWholeFiberResult:
    raw_config = _load_raw_config(config_path)
    cfg = NativeTrace2CpConfig() if native_cfg is None else native_cfg
    fiber_path = None if fiber_json is None else Path(fiber_json)
    whole_mode = _native_trace2cp_whole_fiber_mode(
        fiber_json=fiber_path,
        sample_index=sample_index,
        start_cp_index=start_cp_index,
        target_cp_index=target_cp_index,
    )
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
    selection: _NativeTrace2CpSelection | None = None
    start_zyx: np.ndarray | None = None
    target_zyx: np.ndarray | None = None
    forward_initial_direction: np.ndarray | None = None
    reverse_initial_direction: np.ndarray | None = None
    if whole_mode:
        if sample_mode is not None and str(sample_mode) != "flat":
            raise ValueError("whole-fiber --fiber-json mode requires --sample-mode flat or omitted")
        records = getattr(loader, "records", ())
        if len(records) != 1:
            raise ValueError("whole-fiber --fiber-json mode requires exactly one loaded fiber")
        record = records[0]
    else:
        selection = _resolve_native_trace2cp_selection(
            loader,
            sample_index=13 if sample_index is None else int(sample_index),
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
    normal_sampler = _NativeLasagnaNormalSampler(
        geometry_loader=geometry_loader,
        trace_record=record,
        normal_record=_native_trace_geometry_normal_record(geometry_loader, record),
    )
    cfg = _native_trace_cfg_with_effective_smoothness(cfg, normal_sampler=normal_sampler)
    print(
        "native_trace2cp_3d input "
        f"base_volume_scale={getattr(record, 'volume_scale', 'unknown')} "
        f"volume_spacing_base={float(getattr(record, 'volume_spacing_base', 1.0)):.6g} "
        f"image_normalization={loader_config.image_normalization} "
        f"sampler={type(getattr(record, 'sampler', None)).__name__} "
        f"blocking={getattr(getattr(record, 'sampler', None), 'blocking', 'n/a')}",
        flush=True,
    )
    if whole_mode:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / "trace2cp_native_3d_vis.jpg"
        closed_panel_blocks: list[tuple[Any, Any, Any, Any]] = []
        active_segments: list[NativeWholeFiberSegmentResult] = []
        active_start_cp_index = 0
        _compose_whole_fiber_panel_blocks(
            closed_panel_blocks,
            status_text="initializing whole-fiber trace",
        ).convert("RGB").save(image_path, quality=90)
        print(f"native whole-fiber partial={image_path} initializing", flush=True)

        def on_segment(
            segment: NativeWholeFiberSegmentResult,
            partial: NativeWholeFiberResult | None,
        ) -> None:
            nonlocal active_start_cp_index, active_segments
            print(
                "native whole-fiber render segment "
                f"{segment.start_cp_index}->{segment.target_cp_index} "
                f"success={segment.success} reason={segment.reason} "
                f"error={segment.in_plane_error_voxels:.3f}",
                flush=True,
            )
            if not active_segments:
                active_start_cp_index = int(segment.start_cp_index)
            active_segments.append(segment)
            active_span = _NativeWholeFiberVisualSpan(
                start_cp_index=int(active_start_cp_index),
                end_cp_index=int(segment.target_cp_index),
                segments=tuple(active_segments),
                restart_after=not bool(segment.success),
            )
            panels = _render_native_whole_fiber_span_panels(
                geometry_loader,
                span=active_span,
                trace2cp_rf_margin_px=float(trace2cp_cfg.rf_margin_px),
                cache=cache,
                image_normalization=loader_config.image_normalization,
                strip_cross_width_px=64,
            )
            current_blocks = [*closed_panel_blocks, panels]
            restarts = 0 if partial is None else int(partial.restart_count)
            rate = 0.0 if partial is None else float(partial.restart_rate)
            _compose_whole_fiber_panel_blocks(
                current_blocks,
                status_text=(
                    f"segments={len(partial.segments) if partial is not None else 0} "
                    f"spans={len(current_blocks)} restarts={restarts} "
                    f"rate={rate:.6f}"
                ),
            ).convert("RGB").save(image_path, quality=90)
            print(
                "native whole-fiber partial="
                f"{image_path} segment={segment.start_cp_index}->{segment.target_cp_index}",
                flush=True,
            )
            if not bool(segment.success):
                closed_panel_blocks.append(panels)
                active_segments = []
                active_start_cp_index = int(segment.target_cp_index)

        cp_count = int(record.fiber.control_points_zyx.shape[0])
        print(
            "native_trace2cp_3d whole_fiber "
            f"fiber_path={'' if record.fiber.path is None else record.fiber.path} "
            f"control_points={cp_count} segments={max(0, cp_count - 1)} "
            f"threshold_voxels={float(cfg.whole_fiber_error_threshold_voxels):.3f}",
            flush=True,
        )
        whole = trace_native_3d_whole_fiber(
            cache,
            record=record,
            cfg=cfg,
            error_threshold_voxels=float(cfg.whole_fiber_error_threshold_voxels),
            progress=True,
            segment_callback=on_segment,
            normal_sampler=normal_sampler,
        )
        summary = {
            "mode": "whole_fiber",
            "fiber_path": "" if record.fiber.path is None else str(record.fiber.path),
            "control_point_count": int(cp_count),
            "segment_count": int(whole.segment_count),
            "restart_count": int(whole.restart_count),
            "native_trace2cp_fiber_restart_rate": float(whole.restart_rate),
            "whole_fiber_error_threshold_voxels": float(cfg.whole_fiber_error_threshold_voxels),
            "step_voxels": float(cfg.step_voxels),
            "beam_width": int(cfg.beam_width),
            "beam_prune_distance_voxels": float(cfg.beam_prune_distance_voxels),
            "beam_lookahead_steps": int(cfg.beam_lookahead_steps),
            "candidate_substeps": int(cfg.candidate_substeps),
            "cone_angle_degrees": float(cfg.cone_angle_degrees),
            "cone_angle_step_degrees": float(cfg.cone_angle_step_degrees),
            "cone_grid_size": int(cfg.cone_grid_size),
            "max_step_factor": float(cfg.max_step_factor),
            **_native_trace_smoothness_summary(cfg),
            "max_steps": None if cfg.max_steps is None else int(cfg.max_steps),
            "trace_step_limit": None if cfg.trace_step_limit is None else int(cfg.trace_step_limit),
            "inferred_blocks": int(len(cache._blocks)),
            "export": str(image_path),
            "segments": [
                {
                    "start_control_point_index": int(segment.start_cp_index),
                    "target_control_point_index": int(segment.target_cp_index),
                    "success": bool(segment.success),
                    "restart": bool(segment.restart),
                    "reason": segment.reason,
                    "reached_target_plane": bool(segment.reached_target_plane),
                    "in_plane_error_voxels": float(segment.in_plane_error_voxels),
                    "reference_arc_distance_voxels": float(segment.reference_arc_distance_voxels),
                    "step_count": int(segment.step_count),
                    "restart_point_zyx": [
                        float(v) for v in np.asarray(segment.target_zyx, dtype=np.float32)
                    ]
                    if segment.restart
                    else None,
                }
                for segment in whole.segments
            ],
        }
        summary_path = out_dir / "trace2cp_native_3d_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "native_trace2cp_fiber_restart_rate="
            f"{whole.restart_rate:.8f} restarts={whole.restart_count} "
            f"segments={whole.segment_count}",
            flush=True,
        )
        print(
            "native_trace2cp_3d whole_fiber "
            f"blocks={len(cache._blocks)} export={image_path}",
            flush=True,
        )
        return whole

    if (
        selection is None
        or start_zyx is None
        or target_zyx is None
        or forward_initial_direction is None
        or reverse_initial_direction is None
    ):
        raise RuntimeError("native 3D Trace2CP single-pair selection was not initialized")
    result = trace_native_3d_pair(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        forward_initial_direction_zyx=forward_initial_direction,
        reverse_initial_direction_zyx=reverse_initial_direction,
        cfg=cfg,
        progress=True,
        normal_sampler=normal_sampler,
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
        "beam_width": int(cfg.beam_width),
        "beam_prune_distance_voxels": float(cfg.beam_prune_distance_voxels),
        "beam_lookahead_steps": int(cfg.beam_lookahead_steps),
        "candidate_substeps": int(cfg.candidate_substeps),
        "cone_angle_degrees": float(cfg.cone_angle_degrees),
        "cone_angle_step_degrees": float(cfg.cone_angle_step_degrees),
        "cone_grid_size": int(cfg.cone_grid_size),
        "max_step_factor": float(cfg.max_step_factor),
        **_native_trace_smoothness_summary(cfg),
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
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--fiber-json", type=Path, default=None)
    parser.add_argument("--start-cp-index", type=int, default=None)
    parser.add_argument("--target-cp-index", type=int, default=None)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument("--sample-mode", choices=("random", "flat"), default=None)
    parser.add_argument("--step-voxels", type=float, default=4.0)
    parser.add_argument("--cone-angle-degrees", type=float, default=25.0)
    parser.add_argument("--cone-grid-size", type=int, default=25)
    parser.add_argument("--cone-angle-step-degrees", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-prune-distance-voxels", type=float, default=1.0)
    parser.add_argument("--beam-lookahead-steps", type=int, default=1)
    parser.add_argument("--candidate-substeps", type=int, default=1)
    parser.add_argument("--smoothness-weight", type=float, default=2.0)
    parser.add_argument("--smoothness-tangent-weight", type=float, default=10.0)
    parser.add_argument("--smoothness-normal-weight", type=float, default=0.1)
    parser.add_argument("--smoothness-free-angle-degrees", type=float, default=0.0)
    parser.add_argument("--cumulative-smoothness-steps", type=int, default=4)
    parser.add_argument("--cumulative-smoothness-tangent-weight", type=float, default=2.0)
    parser.add_argument(
        "--no-all-pairs-direction-product",
        action="store_true",
        help="Use the legacy current/candidate two-dot direction product.",
    )
    parser.add_argument("--max-step-factor", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--trace-step-limit", type=int, default=None)
    parser.add_argument("--inference-patch-shape-zyx", nargs=3, type=int, default=None)
    parser.add_argument("--core-margin-voxels", type=int, default=20)
    parser.add_argument("--whole-fiber-error-threshold-voxels", type=float, default=100.0)
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
        cone_angle_step_degrees=float(args.cone_angle_step_degrees),
        beam_width=int(args.beam_width),
        beam_prune_distance_voxels=float(args.beam_prune_distance_voxels),
        beam_lookahead_steps=int(args.beam_lookahead_steps),
        candidate_substeps=int(args.candidate_substeps),
        smoothness_weight=float(args.smoothness_weight),
        smoothness_tangent_weight=None
        if args.smoothness_tangent_weight is None
        else float(args.smoothness_tangent_weight),
        smoothness_normal_weight=None
        if args.smoothness_normal_weight is None
        else float(args.smoothness_normal_weight),
        smoothness_free_angle_degrees=float(args.smoothness_free_angle_degrees),
        cumulative_smoothness_steps=int(args.cumulative_smoothness_steps),
        cumulative_smoothness_tangent_weight=float(
            args.cumulative_smoothness_tangent_weight
        ),
        all_pairs_direction_product=not bool(args.no_all_pairs_direction_product),
        max_step_factor=float(args.max_step_factor),
        max_steps=None if args.max_steps is None else int(args.max_steps),
        trace_step_limit=None if args.trace_step_limit is None else int(args.trace_step_limit),
        inference_patch_shape_zyx=patch_shape,
        core_margin_voxels=int(args.core_margin_voxels),
        whole_fiber_error_threshold_voxels=float(args.whole_fiber_error_threshold_voxels),
    )
    run_native_trace2cp(
        args.config,
        checkpoint=args.checkpoint,
        export_dir=args.export_dir,
        sample_index=None if args.sample_index is None else int(args.sample_index),
        fiber_json=args.fiber_json,
        start_cp_index=args.start_cp_index,
        target_cp_index=args.target_cp_index,
        target_offset=int(args.target_offset),
        sample_mode=args.sample_mode,
        native_cfg=native_cfg,
    )


if __name__ == "__main__":
    main()
