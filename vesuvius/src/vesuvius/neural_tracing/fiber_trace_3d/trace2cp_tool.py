from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from lasagna.tiled_predict3d import _pyrdown3d
except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
    from tiled_predict3d import _pyrdown3d

from vesuvius.neural_tracing.fiber_trace_3d.inference_adapter import (
    FiberTrace3DPredictAdapter,
)
from vesuvius.neural_tracing.fiber_trace_3d.loader import (
    FiberTrace3DConfig,
    FiberTrace3DLoader,
    _gaussian_kernel1d,
    _normalize_image,
    load_config,
)
from vesuvius.neural_tracing.fiber_trace_3d.prediction import (
    decode_grouped_direction_presence,
    direction_branch_count_from_channels,
)
from vesuvius.neural_tracing.fiber_trace_3d.train import (
    _device_from_training,
    _load_raw_config,
    _make_trace2cp_geometry_loader,
    _trace2cp_3d_config,
    dataclass_replace,
)
from vesuvius.neural_tracing.fiber_trace_2d.loader import _Trace2CpSegmentSource
from vesuvius.neural_tracing.fiber_trace_2d.strip_geometry import control_point_line_index


_EPS = 1.0e-12
_REMOTE_PREFIXES = ("http://", "https://", "s3://")
_GIB = 1024**3
_NATIVE_WHOLE_FIBER_VIS_SPLIT_TARGET_PX = 32_000
_NATIVE_WHOLE_FIBER_VIS_JPEG_SAFE_PX = 64_000
_SPARSE_NORMAL_CHANNELS = frozenset({"grad_mag", "nx", "ny"})
_NATIVE_NORMAL_SAMPLER_MODES = ("sparse-corner-principal", "baseline")
_NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS = ("eigh", "analytic")
_NATIVE_NORMAL_PRINCIPAL_AXIS_CLI_CHOICES = (
    "config",
    *_NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS,
)
_NORMAL_CORNER_BITS_XYZ = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 1.0, 1.0),
)


@dataclass
class _NativeTraceProfileStat:
    count: int = 0
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0


class _NullNativeTraceProfileSpan:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        return False


class _NativeTraceProfileSpan:
    def __init__(
        self,
        profiler: "_NativeTraceProfiler",
        name: str,
        *,
        sync_device: torch.device | None = None,
    ) -> None:
        self.profiler = profiler
        self.name = str(name)
        self.sync_device = None if sync_device is None else torch.device(sync_device)
        self.wall_start = 0.0
        self.cpu_start = 0.0

    def __enter__(self):
        self.profiler.sync(self.sync_device)
        self.wall_start = time.perf_counter()
        self.cpu_start = time.process_time()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> bool:
        self.profiler.sync(self.sync_device)
        self.profiler.add(
            self.name,
            wall_seconds=time.perf_counter() - self.wall_start,
            cpu_seconds=time.process_time() - self.cpu_start,
        )
        return False


class _NativeTraceProfiler:
    def __init__(self) -> None:
        self._stats: OrderedDict[str, _NativeTraceProfileStat] = OrderedDict()
        self._total_wall_start = time.perf_counter()
        self._total_cpu_start = time.process_time()
        self.total_wall_seconds: float | None = None
        self.total_cpu_seconds: float | None = None

    @staticmethod
    def sync(device: torch.device | None) -> None:
        if device is None:
            return
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def measure(
        self,
        name: str,
        *,
        sync_device: torch.device | None = None,
    ) -> _NativeTraceProfileSpan:
        return _NativeTraceProfileSpan(self, name, sync_device=sync_device)

    def add(self, name: str, *, wall_seconds: float, cpu_seconds: float) -> None:
        key = str(name)
        stat = self._stats.get(key)
        if stat is None:
            stat = _NativeTraceProfileStat()
            self._stats[key] = stat
        stat.count += 1
        stat.wall_seconds += float(wall_seconds)
        stat.cpu_seconds += float(cpu_seconds)

    def finish_total(self) -> None:
        if self.total_wall_seconds is None:
            self.total_wall_seconds = time.perf_counter() - self._total_wall_start
            self.total_cpu_seconds = time.process_time() - self._total_cpu_start

    def restart_total(self) -> None:
        self._total_wall_start = time.perf_counter()
        self._total_cpu_start = time.process_time()
        self.total_wall_seconds = None
        self.total_cpu_seconds = None

    def total_wall(self) -> float:
        if self.total_wall_seconds is None:
            return time.perf_counter() - self._total_wall_start
        return float(self.total_wall_seconds)

    def total_cpu(self) -> float:
        if self.total_cpu_seconds is None:
            return time.process_time() - self._total_cpu_start
        return float(self.total_cpu_seconds)

    def summary(self) -> dict[str, Any]:
        self.finish_total()
        total_wall = max(float(self.total_wall_seconds or 0.0), _EPS)
        return {
            "total_wall_seconds": float(self.total_wall_seconds or 0.0),
            "total_cpu_seconds": float(self.total_cpu_seconds or 0.0),
            "stages": {
                name: {
                    "count": int(stat.count),
                    "wall_seconds": float(stat.wall_seconds),
                    "cpu_seconds": float(stat.cpu_seconds),
                    "wall_percent": float(stat.wall_seconds * 100.0 / total_wall),
                }
                for name, stat in self._stats.items()
            },
        }

    def print_table(self) -> None:
        self.finish_total()
        total_wall = max(float(self.total_wall_seconds or 0.0), _EPS)
        print(
            "native_trace2cp_profile columns: "
            "stage=profiled stage n=events wall_s=wall seconds cpu_s=process CPU seconds "
            "wall_pct=percent of measured Trace2CP wall runtime",
            flush=True,
        )
        print(f"{'stage':32s} {'n':>6s} {'wall_s':>10s} {'cpu_s':>10s} {'wall_pct':>9s}", flush=True)
        print(
            f"{'total':32s} {1:6d} "
            f"{float(self.total_wall_seconds or 0.0):10.3f} "
            f"{float(self.total_cpu_seconds or 0.0):10.3f} "
            f"{100.0:9.2f}",
            flush=True,
        )
        for name, stat in sorted(
            self._stats.items(),
            key=lambda item: float(item[1].wall_seconds),
            reverse=True,
        ):
            print(
                f"{name[:32]:32s} {int(stat.count):6d} "
                f"{float(stat.wall_seconds):10.3f} "
                f"{float(stat.cpu_seconds):10.3f} "
                f"{float(stat.wall_seconds) * 100.0 / total_wall:9.2f}",
                flush=True,
            )


class _ArgumentDefaultsHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=56, width=140)

    def _format_action_invocation(self, action: argparse.Action) -> str:
        if not action.option_strings:
            return super()._format_action_invocation(action)
        return ", ".join(action.option_strings)

    def _get_help_string(self, action: argparse.Action) -> str:
        help_text = "" if action.help is None else str(action.help)
        if (
            action.default is argparse.SUPPRESS
            or bool(action.required)
            or not action.option_strings
        ):
            return help_text
        if "%(default)" in help_text:
            return help_text
        prefix = "[%(default)s]"
        return prefix if not help_text else f"{prefix} {help_text}"


def _fill_missing_argparse_default_help(parser: argparse.ArgumentParser) -> None:
    for action in parser._actions:
        if (
            action.help is None
            and action.option_strings
            and action.default is not argparse.SUPPRESS
            and not bool(action.required)
        ):
            action.help = "[%(default)s]"


@dataclass(frozen=True)
class NativeTrace2CpConfig:
    step_voxels: float = 4.0
    cone_angle_degrees: float = 25.0
    cone_grid_size: int = 25
    cone_angle_step_degrees: float = 5.0
    beam_width: int = 8
    beam_prune_distance_voxels: float = 1.0
    beam_lookahead_steps: int = 2
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
    inference_patch_shape_zyx: tuple[int, int, int] = (128, 128, 128)
    core_margin_voxels: int = 48
    inference_scaledown_power: int = 0
    inference_blur_sigma_voxels: float = 0.0
    inference_block_batch_size: int = 2
    whole_fiber_error_threshold_voxels: float = 10.0
    max_cached_inference_gib: float = 8.0


@dataclass(frozen=True)
class NativeTraceStep:
    point_zyx: np.ndarray
    direction_loss: float
    presence_loss: float
    total_loss: float
    rejected_candidates: int
    smoothness_loss: float = 0.0


@dataclass(frozen=True)
class NativeTargetPlane:
    name: str
    point_zyx: np.ndarray
    normal_zyx: np.ndarray


@dataclass(frozen=True)
class NativeTargetPlaneCrossing:
    name: str
    point_zyx: np.ndarray
    error_voxels: float


@dataclass(frozen=True)
class NativeTraceResult:
    trace_zyx: np.ndarray
    reached_target_plane: bool
    reason: str
    steps: tuple[NativeTraceStep, ...]
    target_plane_crossings: tuple[NativeTargetPlaneCrossing, ...] = ()
    selected_target_plane_name: str | None = None
    selected_target_plane_crossing_zyx: np.ndarray | None = None
    selected_target_plane_error_voxels: float = math.inf


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
    selected_target_plane_name: str | None
    selected_target_plane_crossing_zyx: np.ndarray | None
    reference_arc_distance_voxels: float
    step_count: int


@dataclass(frozen=True)
class NativeWholeFiberResult:
    segments: tuple[NativeWholeFiberSegmentResult, ...]
    restart_count: int
    restarts_per_kvx: float
    segment_count: int
    reference_length_voxels: float
    reference_length_meters: float | None
    restarts_per_meter: float | None
    stitched_trace_zyx: np.ndarray
    inferred_blocks: int


@dataclass(frozen=True)
class NativeMultiFiberResult:
    results: tuple[NativeWholeFiberResult, ...]
    restart_count: int
    restarts_per_kvx: float
    segment_count: int
    reference_length_voxels: float
    reference_length_meters: float | None
    restarts_per_meter: float | None
    run_count: int


@dataclass(frozen=True)
class _NativeWholeFiberVisualSpan:
    start_cp_index: int
    end_cp_index: int
    segments: tuple[NativeWholeFiberSegmentResult, ...]
    restart_after: bool


@dataclass(frozen=True)
class _InferredBlock:
    origin_zyx: np.ndarray
    sample_origin_zyx: np.ndarray
    sample_spacing_zyx: np.ndarray
    shape_zyx: tuple[int, int, int]
    core_lo_zyx: np.ndarray
    core_hi_zyx: np.ndarray
    sample_origin_zyx_t: torch.Tensor
    sample_spacing_zyx_t: torch.Tensor
    core_lo_zyx_t: torch.Tensor
    core_hi_zyx_t: torch.Tensor
    shape_max_zyx_t: torch.Tensor
    output_czyx: torch.Tensor
    valid_mask_zyx: torch.Tensor
    cache_nbytes: int


class NativeTracePredictionField(Protocol):
    total_inferred_blocks: int
    evicted_inferred_blocks: int
    resident_inferred_block_bytes: int

    def sample_point_choices_torch(
        self,
        points_zyx: torch.Tensor | np.ndarray,
        *,
        progress_label: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ...


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
    sampled_current_direction_zyx: np.ndarray | None = None
    sampled_current_valid: bool = False
    target_plane_crossed: np.ndarray | None = None
    target_plane_crossings_zyx: np.ndarray | None = None


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
    sampled_current_directions_zyx: torch.Tensor | None = None
    sampled_current_valid: torch.Tensor | None = None
    target_plane_crossed: torch.Tensor | None = None
    target_plane_crossings_zyx: torch.Tensor | None = None


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


def _cache_bytes_from_gib(value: float) -> int:
    gib = float(value)
    if not math.isfinite(gib) or gib < 0.0:
        raise ValueError("max cached inference GiB must be finite and >= 0")
    return int(round(gib * float(_GIB)))


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


def _scaledown_factor_from_power(power: int) -> int:
    exponent = int(power)
    if exponent < 0:
        raise ValueError("inference_scaledown_power must be >= 0")
    return int(1 << exponent)


def _validate_inference_scaledown(
    *,
    patch_shape_zyx: tuple[int, int, int],
    core_margin_voxels: int,
    inference_scaledown_power: int,
) -> tuple[int, tuple[int, int, int], int]:
    factor = _scaledown_factor_from_power(int(inference_scaledown_power))
    patch_shape = tuple(int(v) for v in patch_shape_zyx)
    if any(v <= 0 for v in patch_shape):
        raise ValueError("inference_patch_shape_zyx axes must be > 0")
    margin = int(core_margin_voxels)
    if margin < 0:
        raise ValueError("core_margin_voxels must be >= 0")
    if any(v <= 2 * margin for v in patch_shape):
        raise ValueError(
            "inference_patch_shape_zyx must be larger than 2 * core_margin_voxels"
        )
    if any(v % factor != 0 for v in patch_shape):
        raise ValueError(
            "inference_patch_shape_zyx axes must be evenly divisible by "
            f"the inference scaledown factor {factor}"
        )
    if margin % factor != 0:
        raise ValueError(
            "core_margin_voxels must be evenly divisible by "
            f"the inference scaledown factor {factor}"
        )
    scaled_shape = tuple(int(v // factor) for v in patch_shape)
    scaled_margin = int(margin // factor)
    if any(v <= 2 * scaled_margin for v in scaled_shape):
        raise ValueError(
            "scaled inference field must be larger than 2 * scaled core margin"
        )
    return factor, scaled_shape, scaled_margin


def _pyramid_downsample_3d(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    factor_i = int(factor)
    if factor_i <= 1:
        return tensor
    if tensor.ndim != 5:
        raise ValueError("pyramid downsample expects tensor shape B,C,D,H,W")
    batch, channels, depth, height, width = (int(v) for v in tensor.shape)
    flat = tensor.reshape(batch * channels, depth, height, width)
    down = _pyrdown3d(flat, factor=factor_i)
    return down.reshape(batch, channels, *down.shape[-3:])


def _all_valid_downsample_3d(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    factor_i = int(factor)
    if factor_i <= 1:
        return tensor
    if tensor.ndim != 5:
        raise ValueError("validity downsample expects tensor shape B,C,D,H,W")
    return F.avg_pool3d(tensor, kernel_size=factor_i, stride=factor_i)


def _gaussian_blur_channels_3d(
    tensor: torch.Tensor,
    sigma_voxels: float,
) -> torch.Tensor:
    sigma = float(sigma_voxels)
    if sigma <= 0.0:
        return tensor
    if tensor.ndim != 5:
        raise ValueError("3D Gaussian blur expects tensor shape B,C,D,H,W")
    channels = int(tensor.shape[1])
    if channels <= 0:
        return tensor
    kernel = _gaussian_kernel1d(sigma, device=tensor.device).to(dtype=tensor.dtype)
    radius = int((int(kernel.numel()) - 1) // 2)
    out = tensor
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = int(kernel.numel())
        weight = kernel.view(*shape).repeat(channels, 1, 1, 1, 1)
        padding = [0, 0, 0, 0, 0, 0]
        padding[(2 - axis) * 2] = radius
        padding[(2 - axis) * 2 + 1] = radius
        out = F.conv3d(
            F.pad(out, padding, mode="replicate"),
            weight,
            groups=channels,
        )
    return out


def _grid_sample_channels_at_points(
    values_czyx: torch.Tensor,
    points_zyx: torch.Tensor | np.ndarray,
    *,
    origin_zyx: np.ndarray,
    spacing_zyx: np.ndarray | float | int = 1.0,
) -> torch.Tensor:
    if values_czyx.ndim != 4:
        raise ValueError("values_czyx must have shape C,Z,Y,X")
    _channels, depth, height, width = (int(v) for v in values_czyx.shape)
    points = torch.as_tensor(points_zyx, dtype=torch.float32, device=values_czyx.device)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_zyx must have shape [N,3]")
    if int(points.shape[0]) == 0:
        return torch.zeros((0, int(values_czyx.shape[0])), dtype=values_czyx.dtype, device=values_czyx.device)
    origin = torch.as_tensor(origin_zyx, dtype=torch.float32, device=values_czyx.device)
    local = points - origin.view(1, 3)
    spacing_np = np.asarray(spacing_zyx, dtype=np.float32)
    if spacing_np.ndim == 0:
        spacing_np = np.full((3,), float(spacing_np), dtype=np.float32)
    spacing_np = spacing_np.reshape(-1)
    if int(spacing_np.size) != 3:
        raise ValueError("spacing_zyx must be a scalar or length-3 vector")
    if not bool(np.all(spacing_np > 0.0)):
        raise ValueError("spacing_zyx values must be > 0")
    spacing = torch.as_tensor(spacing_np, dtype=torch.float32, device=values_czyx.device)
    local = local / spacing.view(1, 3)
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
        prediction_adapter: Any,
        model: torch.nn.Module,
        patch_shape_zyx: tuple[int, int, int],
        core_margin_voxels: int,
        inference_scaledown_power: int = 0,
        inference_blur_sigma_voxels: float = 0.0,
        device: torch.device,
        max_cached_bytes: int | None = 8 * 1024**3,
        inference_block_batch_size: int = 1,
        profiler: _NativeTraceProfiler | None = None,
    ) -> None:
        self.record = record
        self.prediction_adapter = prediction_adapter
        self.model = model
        self.patch_shape_zyx = tuple(int(v) for v in patch_shape_zyx)
        self.core_margin = int(core_margin_voxels)
        (
            self.inference_scaledown_factor,
            self.inference_field_shape_zyx,
            self.scaled_core_margin,
        ) = _validate_inference_scaledown(
            patch_shape_zyx=self.patch_shape_zyx,
            core_margin_voxels=self.core_margin,
            inference_scaledown_power=int(inference_scaledown_power),
        )
        self.inference_scaledown_power = int(inference_scaledown_power)
        self.inference_blur_sigma_voxels = float(inference_blur_sigma_voxels)
        if self.inference_blur_sigma_voxels < 0.0:
            raise ValueError("inference_blur_sigma_voxels must be >= 0")
        self.core_shape_zyx = tuple(v - 2 * self.core_margin for v in self.patch_shape_zyx)
        self.device = torch.device(device)
        if max_cached_bytes is not None and int(max_cached_bytes) < 0:
            raise ValueError("max_cached_bytes must be >= 0 or None")
        self.max_cached_bytes = None if max_cached_bytes is None else int(max_cached_bytes)
        self.inference_block_batch_size = max(1, int(inference_block_batch_size))
        self.profiler = profiler
        self._blocks: OrderedDict[tuple[int, int, int], _InferredBlock] = OrderedDict()
        self.total_inferred_blocks = 0
        self.evicted_inferred_blocks = 0
        self.resident_inferred_block_bytes = 0

    def _measure(
        self,
        name: str,
        *,
        sync: bool = False,
    ) -> _NativeTraceProfileSpan | _NullNativeTraceProfileSpan:
        if self.profiler is None:
            return _NullNativeTraceProfileSpan()
        return self.profiler.measure(
            name,
            sync_device=self.device if bool(sync) else None,
        )

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

    def _block_origins_for_points_torch(self, points_zyx: torch.Tensor) -> torch.Tensor:
        points = points_zyx.to(device=self.device, dtype=torch.float32)
        if points.ndim != 2 or int(points.shape[1]) != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        stride = torch.as_tensor(
            self.core_shape_zyx,
            dtype=torch.float32,
            device=self.device,
        )
        origin = torch.floor((points - float(self.core_margin)) / stride).to(dtype=torch.int64)
        origin = origin * torch.as_tensor(
            self.core_shape_zyx,
            dtype=torch.int64,
            device=self.device,
        ).view(1, 3)
        return origin.contiguous()

    def _sample_block_volume(self, origin_zyx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        sampler = getattr(self.record, "sampler", None)
        if sampler is None:
            raise ValueError("native 3D Trace2CP record must provide a coordinate sampler")
        if hasattr(sampler, "blocking") and not bool(getattr(sampler, "blocking")):
            raise ValueError("native 3D Trace2CP requires blocking coordinate sampling")
        if not hasattr(sampler, "sample_block_zyx"):
            raise ValueError(
                "native 3D Trace2CP requires a sampler with sample_block_zyx support"
            )
        origin = np.asarray(origin_zyx, dtype=np.int64)
        with self._measure("src_read_block"):
            result = sampler.sample_block_zyx(origin, self.patch_shape_zyx)
        with self._measure("src_validate"):
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
            sampled_valid_np = np.asarray(result.valid_mask, dtype=bool)
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
        with self._measure("src_tensor"):
            sampled = torch.as_tensor(sampled_np, dtype=torch.float32, device=self.device)
            sampled_valid = torch.as_tensor(sampled_valid_np, dtype=torch.bool, device=self.device)
            sampled = torch.where(sampled_valid, sampled, torch.zeros_like(sampled))
        return sampled, sampled_valid

    def _cached_output_slices(self) -> tuple[tuple[slice, slice, slice], np.ndarray]:
        lo = int(self.scaled_core_margin)
        hi = (
            np.asarray(self.inference_field_shape_zyx, dtype=np.int64)
            - int(self.scaled_core_margin)
        )
        # Keep a one-voxel upper halo so trilinear samples inside the trusted
        # core can read the same interpolation corners as the full output.
        crop_hi = np.minimum(
            np.asarray(self.inference_field_shape_zyx, dtype=np.int64),
            hi + np.asarray([1, 1, 1], dtype=np.int64),
        )
        slices = tuple(slice(lo, int(axis_hi)) for axis_hi in crop_hi)
        return slices, np.asarray([lo, lo, lo], dtype=np.int64)

    def _store_inferred_block(
        self,
        *,
        origin: np.ndarray,
        output: torch.Tensor,
        valid: torch.Tensor,
    ) -> _InferredBlock:
        origin_i64 = np.asarray(origin, dtype=np.int64)
        factor = int(self.inference_scaledown_factor)
        core_lo = origin_i64 + int(self.core_margin)
        core_hi = (
            origin_i64
            + np.asarray(self.patch_shape_zyx, dtype=np.int64)
            - int(self.core_margin)
        )
        crop_slices, crop_lo = self._cached_output_slices()
        expected_shape = tuple(int(v) for v in self.inference_field_shape_zyx)
        if tuple(int(v) for v in output.shape[-3:]) != expected_shape:
            raise ValueError(
                "native 3D Trace2CP model output has incompatible scaled shape: "
                f"shape={tuple(int(v) for v in output.shape[-3:])} expected={expected_shape}"
            )
        if tuple(int(v) for v in valid.shape[-3:]) != expected_shape:
            raise ValueError(
                "native 3D Trace2CP valid mask has incompatible scaled shape: "
                f"shape={tuple(int(v) for v in valid.shape[-3:])} expected={expected_shape}"
            )
        output_crop = output[
            :,
            crop_slices[0],
            crop_slices[1],
            crop_slices[2],
        ].contiguous()
        valid_crop = valid[crop_slices[0], crop_slices[1], crop_slices[2]].contiguous()
        sample_origin = (origin_i64 + crop_lo * factor).astype(np.float32)
        sample_spacing = np.asarray([factor, factor, factor], dtype=np.float32)
        core_lo_f = core_lo.astype(np.float32)
        core_hi_f = core_hi.astype(np.float32)
        shape_max = (np.asarray(valid_crop.shape, dtype=np.float32) - 1.0).astype(np.float32)
        cache_nbytes = int(output_crop.numel() * output_crop.element_size())
        cache_nbytes += int(valid_crop.numel() * valid_crop.element_size())
        block = _InferredBlock(
            origin_zyx=origin_i64.astype(np.int64),
            sample_origin_zyx=sample_origin,
            sample_spacing_zyx=sample_spacing,
            shape_zyx=tuple(int(v) for v in valid_crop.shape),
            core_lo_zyx=core_lo_f,
            core_hi_zyx=core_hi_f,
            sample_origin_zyx_t=torch.as_tensor(
                sample_origin,
                dtype=torch.float32,
                device=self.device,
            ),
            sample_spacing_zyx_t=torch.as_tensor(
                sample_spacing,
                dtype=torch.float32,
                device=self.device,
            ),
            core_lo_zyx_t=torch.as_tensor(
                core_lo_f,
                dtype=torch.float32,
                device=self.device,
            ),
            core_hi_zyx_t=torch.as_tensor(
                core_hi_f,
                dtype=torch.float32,
                device=self.device,
            ),
            shape_max_zyx_t=torch.as_tensor(
                shape_max,
                dtype=torch.float32,
                device=self.device,
            ),
            output_czyx=output_crop,
            valid_mask_zyx=valid_crop,
            cache_nbytes=cache_nbytes,
        )
        key = tuple(int(v) for v in origin_i64)
        with self._measure("inference_cache_store"):
            if self.max_cached_bytes is None or self.max_cached_bytes > 0:
                self._blocks[key] = block
                self._blocks.move_to_end(key)
                self.resident_inferred_block_bytes += int(block.cache_nbytes)
                if self.max_cached_bytes is not None:
                    while (
                        self._blocks
                        and self.resident_inferred_block_bytes > self.max_cached_bytes
                    ):
                        _evicted_key, evicted = self._blocks.popitem(last=False)
                        self.resident_inferred_block_bytes -= int(evicted.cache_nbytes)
                        self.evicted_inferred_blocks += 1
        return block

    @torch.no_grad()
    def _infer_blocks(
        self,
        origins_zyx: np.ndarray,
    ) -> dict[tuple[int, int, int], _InferredBlock]:
        origins = np.asarray(origins_zyx, dtype=np.int64).reshape(-1, 3)
        blocks_by_key: dict[tuple[int, int, int], _InferredBlock] = {}
        missing: list[np.ndarray] = []
        for origin in origins:
            key = tuple(int(v) for v in origin)
            block = self._blocks.get(key)
            if block is not None:
                with self._measure("inference_cache_hit"):
                    self._blocks.move_to_end(key)
                blocks_by_key[key] = block
            else:
                missing.append(origin.astype(np.int64, copy=True))
        if not missing:
            return blocks_by_key

        batch_size = max(1, int(self.inference_block_batch_size))
        for start in range(0, len(missing), batch_size):
            batch_origins = missing[start : start + batch_size]
            raw_blocks: list[torch.Tensor] = []
            valid_blocks: list[torch.Tensor] = []
            for origin in batch_origins:
                raw_t, valid_t = self._sample_block_volume(origin)
                raw_blocks.append(raw_t)
                valid_blocks.append(valid_t)
            with self._measure("inference_preprocess", sync=True):
                raw_batch = torch.stack(raw_blocks, dim=0).view(
                    len(batch_origins),
                    1,
                    *self.patch_shape_zyx,
                )
                valid_batch = torch.stack(valid_blocks, dim=0).view(
                    len(batch_origins),
                    1,
                    *self.patch_shape_zyx,
                )
                model_input = self.prediction_adapter.preprocess_tile(raw_batch, valid_batch)
            with self._measure("inference_forward", sync=True):
                model_output = self.prediction_adapter.run_tile_inference(
                    self.model,
                    model_input,
                    device=self.device,
                )
            with self._measure("inference_decode_cache", sync=True):
                products = self.prediction_adapter.product_tensors_from_output(model_output)
                product_tensors: list[torch.Tensor] = []
                for product in self.prediction_adapter.output_products:
                    tensor = products.get(product.name)
                    if tensor is None:
                        raise ValueError(
                            f"fiber 3D prediction adapter did not return product {product.name!r}"
                        )
                    if (
                        tensor.ndim != 5
                        or int(tensor.shape[0]) != len(batch_origins)
                        or int(tensor.shape[1]) != 7
                    ):
                        raise ValueError(
                            "fiber 3D prediction adapter product tensors must have shape "
                            f"B,7,D,H,W; got {tuple(int(v) for v in tensor.shape)}"
                        )
                    product_tensors.append(tensor)
                if int(self.inference_scaledown_factor) > 1:
                    factor = int(self.inference_scaledown_factor)
                    product_tensors = [
                        _pyramid_downsample_3d(tensor, factor)
                        for tensor in product_tensors
                    ]
                    valid_for_cache = (
                        _all_valid_downsample_3d(
                            valid_batch.to(dtype=torch.float32),
                            factor,
                        )
                        >= 1.0
                    )
                else:
                    valid_for_cache = valid_batch
                if self.inference_blur_sigma_voxels > 0.0:
                    scaled_sigma = (
                        float(self.inference_blur_sigma_voxels)
                        / float(self.inference_scaledown_factor)
                    )
                    product_tensors = [
                        _gaussian_blur_channels_3d(tensor, scaled_sigma)
                        for tensor in product_tensors
                    ]
                valid_cache = valid_for_cache.detach().to(
                    device=self.device,
                    dtype=torch.bool,
                ).contiguous()
                for batch_index, origin in enumerate(batch_origins):
                    output = torch.cat(
                        [tensor[batch_index] for tensor in product_tensors],
                        dim=0,
                    ).detach().to(
                        device=self.device,
                        dtype=torch.float32,
                    ).contiguous()
                    block = self._store_inferred_block(
                        origin=origin,
                        output=output,
                        valid=valid_cache[batch_index, 0],
                    )
                    blocks_by_key[tuple(int(v) for v in origin)] = block
                self.total_inferred_blocks += len(batch_origins)
        return blocks_by_key

    @torch.no_grad()
    def _infer_block(self, origin_zyx: np.ndarray) -> _InferredBlock:
        origin = np.asarray(origin_zyx, dtype=np.int64)
        key = tuple(int(v) for v in origin)
        return self._infer_blocks(origin.reshape(1, 3))[key]

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
        points_zyx: torch.Tensor | np.ndarray,
        *,
        progress_label: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points_t = torch.as_tensor(points_zyx, dtype=torch.float32, device=self.device)
        if points_t.ndim != 2 or points_t.shape[1] != 3:
            raise ValueError("points_zyx must have shape [N,3]")
        count = int(points_t.shape[0])
        if count == 0:
            return (
                torch.zeros((0, 1, 3), dtype=torch.float32, device=self.device),
                torch.zeros((0, 1), dtype=torch.float32, device=self.device),
                torch.zeros((0, 1), dtype=torch.bool, device=self.device),
            )

        with self._measure("field_lookup_origin"):
            origins_t = self._block_origins_for_points_torch(points_t)
            unique_origins_t, inverse_t = torch.unique(
                origins_t,
                dim=0,
                return_inverse=True,
            )
            unique_origins = (
                unique_origins_t.detach()
                .to(device=torch.device("cpu"), dtype=torch.int64)
                .numpy()
            )
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
                    f"cache_blocks={len(self._blocks)} "
                    f"cache_gib={self.resident_inferred_block_bytes / float(_GIB):.3f}"
                ),
            )

        emit_progress(0, force=True)
        unique_keys = [
            tuple(int(v) for v in np.asarray(origin, dtype=np.int64))
            for origin in unique_origins
        ]
        cached_before = {key for key in unique_keys if key in self._blocks}
        blocks_by_key = self._infer_blocks(unique_origins)
        cached_blocks = len(cached_before)
        new_blocks = len(unique_keys) - cached_blocks
        channel_count: int | None = None
        block_shape: tuple[int, int, int] | None = None
        groups: list[tuple[_InferredBlock, torch.Tensor, torch.Tensor]] = []
        with self._measure("field_lookup_route"):
            for unique_index, _origin in enumerate(unique_origins):
                indices = torch.nonzero(
                    inverse_t == int(unique_index),
                    as_tuple=False,
                ).flatten()
                if int(indices.numel()) == 0:
                    emit_progress(int(unique_index) + 1)
                    continue
                key = unique_keys[int(unique_index)]
                block = blocks_by_key[key]
                sampled_branch_count = direction_branch_count_from_channels(
                    int(block.output_czyx.shape[0])
                )
                if branch_count is None:
                    branch_count = sampled_branch_count
                    channel_count = int(block.output_czyx.shape[0])
                    block_shape = tuple(int(v) for v in block.shape_zyx)
                elif sampled_branch_count != branch_count:
                    raise ValueError(
                        "native 3D Trace2CP sampled blocks disagree on branch count: "
                        f"{sampled_branch_count} != {branch_count}"
                    )
                elif int(block.output_czyx.shape[0]) != int(channel_count):
                    raise ValueError(
                        "native 3D Trace2CP sampled blocks disagree on channel count: "
                        f"{int(block.output_czyx.shape[0])} != {int(channel_count)}"
                    )
                elif tuple(int(v) for v in block.shape_zyx) != block_shape:
                    raise ValueError(
                        "native 3D Trace2CP sampled blocks disagree on cached shape: "
                        f"{tuple(int(v) for v in block.shape_zyx)} != {block_shape}"
                    )
                group_points = points_t.index_select(0, indices.to(device=self.device))
                groups.append((block, indices.to(device=self.device), group_points))
                emit_progress(int(unique_index) + 1)
        if branch_count is None or channel_count is None or block_shape is None:
            directions = torch.zeros((count, 1, 3), dtype=torch.float32, device=self.device)
            presence = torch.zeros((count, 1), dtype=torch.float32, device=self.device)
            valid = torch.zeros((count, 1), dtype=torch.bool, device=self.device)
            emit_progress(int(unique_origins.shape[0]), force=True)
            return directions, presence, valid

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
        if groups:
            group_count = len(groups)
            max_group_size = max(int(indices.numel()) for _block, indices, _points in groups)
            depth, height, width = (int(v) for v in block_shape)
            with self._measure("field_sample_lookup"):
                block_values = torch.empty(
                    (group_count, channel_count + 1, depth, height, width),
                    dtype=torch.float32,
                    device=self.device,
                )
                grid = torch.zeros(
                    (group_count, max_group_size, 1, 1, 3),
                    dtype=torch.float32,
                    device=self.device,
                )
                group_mask = torch.zeros(
                    (group_count, max_group_size),
                    dtype=torch.bool,
                    device=self.device,
                )
                point_valid_mask = torch.zeros(
                    (group_count, max_group_size),
                    dtype=torch.bool,
                    device=self.device,
                )
                scatter_indices = torch.zeros(
                    (group_count, max_group_size),
                    dtype=torch.long,
                    device=self.device,
                )
                for group_index, (block, usable_indices, usable_points) in enumerate(groups):
                    n_points = int(usable_points.shape[0])
                    block_values[group_index, :channel_count] = block.output_czyx.to(
                        dtype=torch.float32
                    )
                    block_values[group_index, channel_count] = block.valid_mask_zyx.to(
                        dtype=torch.float32
                    )
                    origin_t = block.sample_origin_zyx_t
                    spacing_t = block.sample_spacing_zyx_t
                    local_t = (usable_points - origin_t.view(1, 3)) / spacing_t.view(1, 3)
                    core_lo_t = block.core_lo_zyx_t
                    core_hi_t = block.core_hi_zyx_t
                    shape_t = block.shape_max_zyx_t
                    inside_core_t = torch.all(
                        (usable_points >= core_lo_t.view(1, 3))
                        & (usable_points < core_hi_t.view(1, 3)),
                        dim=1,
                    )
                    inside_block_t = torch.all(
                        (local_t >= 0.0) & (local_t <= shape_t.view(1, 3)),
                        dim=1,
                    )
                    if depth > 1:
                        gz = local_t[:, 0] * (2.0 / float(depth - 1)) - 1.0
                    else:
                        gz = torch.zeros((n_points,), dtype=torch.float32, device=self.device)
                    if height > 1:
                        gy = local_t[:, 1] * (2.0 / float(height - 1)) - 1.0
                    else:
                        gy = torch.zeros((n_points,), dtype=torch.float32, device=self.device)
                    if width > 1:
                        gx = local_t[:, 2] * (2.0 / float(width - 1)) - 1.0
                    else:
                        gx = torch.zeros((n_points,), dtype=torch.float32, device=self.device)
                    grid[group_index, :n_points, 0, 0] = torch.stack([gx, gy, gz], dim=1)
                    group_mask[group_index, :n_points] = True
                    point_valid_mask[group_index, :n_points] = inside_core_t & inside_block_t
                    scatter_indices[group_index, :n_points] = torch.as_tensor(
                        usable_indices,
                        dtype=torch.long,
                        device=self.device,
                    )
                sampled = F.grid_sample(
                    block_values,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=True,
                )
                sampled = sampled[:, :, :, 0, 0].permute(0, 2, 1).contiguous()
                flat_mask = group_mask.reshape(-1)
                point_valid_flat = point_valid_mask.reshape(-1)
                sampled_flat = sampled.reshape(group_count * max_group_size, channel_count + 1)
                scatter_flat = scatter_indices.reshape(-1)
                point_valid_flat = point_valid_flat[flat_mask]
                sampled_flat = sampled_flat[flat_mask]
                scatter_flat = scatter_flat[flat_mask]
                axis_choices_zyx, sampled_presence, decoded_valid = (
                    decode_grouped_direction_presence(sampled_flat[:, :channel_count])
                )
                sampled_valid = sampled_flat[:, channel_count] > 0.5
                sampled_valid = point_valid_flat[:, None] & sampled_valid[:, None] & decoded_valid
                directions[scatter_flat] = axis_choices_zyx
                presence[scatter_flat] = sampled_presence
                valid[scatter_flat] = sampled_valid
                if progress_label is not None:
                    valid_done = int(
                        torch.count_nonzero(valid.any(dim=1)).detach().cpu()
                    )
        emit_progress(int(unique_origins.shape[0]), force=True)
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


def _principal_tensor_axes_torch_analytic(
    tensors_t: torch.Tensor,
    fallback_axis: torch.Tensor,
) -> torch.Tensor:
    a00 = tensors_t[:, 0, 0]
    a01 = tensors_t[:, 0, 1]
    a02 = tensors_t[:, 0, 2]
    a11 = tensors_t[:, 1, 1]
    a12 = tensors_t[:, 1, 2]
    a22 = tensors_t[:, 2, 2]
    q = (a00 + a11 + a22) / 3.0
    b00 = a00 - q
    b11 = a11 - q
    b22 = a22 - q
    p1 = a01 * a01 + a02 * a02 + a12 * a12
    p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=0.0))
    inv_p = torch.where(p > float(_EPS), 1.0 / p, torch.zeros_like(p))
    c00 = b00 * inv_p
    c01 = a01 * inv_p
    c02 = a02 * inv_p
    c11 = b11 * inv_p
    c12 = a12 * inv_p
    c22 = b22 * inv_p
    det_c = (
        c00 * (c11 * c22 - c12 * c12)
        - c01 * (c01 * c22 - c12 * c02)
        + c02 * (c01 * c12 - c11 * c02)
    )
    r = torch.clamp(det_c * 0.5, -1.0, 1.0)
    phi = torch.acos(r) / 3.0
    lambda_max = q + 2.0 * p * torch.cos(phi)
    diag = torch.stack([a00, a11, a22], dim=1)
    lambda_max = torch.where(
        p > float(_EPS),
        lambda_max,
        torch.max(diag, dim=1).values,
    )

    matrix = tensors_t.clone()
    rows = torch.arange(int(tensors_t.shape[0]), dtype=torch.long, device=tensors_t.device)
    matrix[rows, 0, 0] -= lambda_max
    matrix[rows, 1, 1] -= lambda_max
    matrix[rows, 2, 2] -= lambda_max
    r0 = matrix[:, 0, :]
    r1 = matrix[:, 1, :]
    r2 = matrix[:, 2, :]
    candidates = torch.stack(
        [
            torch.cross(r0, r1, dim=1),
            torch.cross(r0, r2, dim=1),
            torch.cross(r1, r2, dim=1),
            fallback_axis,
        ],
        dim=1,
    )
    candidate_norm2 = torch.sum(candidates * candidates, dim=2)
    best = torch.argmax(candidate_norm2, dim=1)
    return candidates[rows, best]


def _principal_tensor_axes_torch(
    tensors: torch.Tensor,
    hints: torch.Tensor,
    *,
    method: str = "eigh",
) -> torch.Tensor:
    tensors_t = tensors.to(dtype=torch.float32)
    hints_t = hints.to(device=tensors_t.device, dtype=torch.float32)
    if tensors_t.ndim != 3 or tuple(int(v) for v in tensors_t.shape[1:]) != (3, 3):
        raise ValueError("tensors must have shape [N,3,3]")
    if tuple(int(v) for v in hints_t.shape) != (int(tensors_t.shape[0]), 3):
        raise ValueError("hints must have shape [N,3]")
    method_s = str(method)
    if method_s not in _NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS:
        raise ValueError(f"unsupported principal-axis method: {method_s!r}")
    count = int(tensors_t.shape[0])
    axes = hints_t.clone()
    hint_norm = torch.linalg.norm(axes, dim=1)
    valid_hint = torch.isfinite(hint_norm) & (hint_norm > float(_EPS))
    diag = torch.stack(
        [tensors_t[:, 0, 0], tensors_t[:, 1, 1], tensors_t[:, 2, 2]],
        dim=1,
    )
    fallback_axis = torch.zeros((count, 3), dtype=torch.float32, device=tensors_t.device)
    fallback_axis[
        torch.arange(count, device=tensors_t.device),
        torch.argmax(diag, dim=1),
    ] = 1.0
    if method_s == "eigh":
        eigenvalues, eigenvectors = torch.linalg.eigh(tensors_t)
        best = torch.argmax(eigenvalues, dim=1)
        rows = torch.arange(count, dtype=torch.long, device=tensors_t.device)
        axes = eigenvectors[rows, :, best]
    else:
        axes = _principal_tensor_axes_torch_analytic(tensors_t, fallback_axis)
    axes = torch.where(torch.isfinite(axes).all(dim=1)[:, None], axes, fallback_axis)
    norms = torch.linalg.norm(axes, dim=1)
    valid_axes = torch.isfinite(norms) & (norms > float(_EPS))
    axes = torch.where(
        valid_axes[:, None],
        axes / norms.clamp_min(float(_EPS))[:, None],
        torch.zeros_like(axes),
    )
    hint_unit = torch.where(
        valid_hint[:, None],
        hints_t / hint_norm.clamp_min(float(_EPS))[:, None],
        torch.zeros_like(hints_t),
    )
    flip = valid_hint & (torch.sum(axes * hint_unit, dim=1) < 0.0)
    axes = torch.where(flip[:, None], -axes, axes)
    no_hint_flip = (~valid_hint) & (axes[:, 2] < 0.0)
    axes = torch.where(no_hint_flip[:, None], -axes, axes)
    return axes


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
    profiler: _NativeTraceProfiler | None = None

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
        span = (
            self.profiler.measure("lasagna_normal_sample")
            if self.profiler is not None
            else _NullNativeTraceProfileSpan()
        )
        with span:
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


def _import_lasagna_fit_data() -> Any:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "lasagna" / "fit_data.py"
        if candidate.exists():
            lasagna_dir = str(candidate.parent)
            if lasagna_dir not in sys.path:
                sys.path.insert(0, lasagna_dir)
            break
    try:
        from lasagna import fit_data
    except ImportError:  # pragma: no cover - supports PYTHONPATH=lasagna style runs.
        import fit_data  # type: ignore
    return fit_data


def _native_trace_manifest_path_for_record(
    record: Any,
    *,
    raw_config: dict[str, Any],
) -> str | None:
    dataset_config = getattr(record, "dataset_config", None)
    if not isinstance(dataset_config, dict):
        return None
    manifest_path = dataset_config.get("lasagna_manifest_path")
    if not manifest_path:
        return None
    return _resolve_config_relative_path(
        manifest_path,
        {"_config_dir": raw_config.get("_config_dir")},
    )


@dataclass
class _SparseCornerLasagnaNormalSampler:
    trace_record: Any
    data: Any
    device: torch.device
    principal_axis_method: str = "eigh"
    profiler: _NativeTraceProfiler | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if str(self.principal_axis_method) not in _NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS:
            raise ValueError(
                "unsupported sparse normal principal-axis method: "
                f"{self.principal_axis_method!r}"
            )

    def _measure(
        self,
        name: str,
        *,
        sync: bool = False,
    ) -> _NativeTraceProfileSpan | _NullNativeTraceProfileSpan:
        if self.profiler is None:
            return _NullNativeTraceProfileSpan()
        return self.profiler.measure(
            name,
            sync_device=self.device if bool(sync) else None,
        )

    def _selected_zyx_to_fullres_xyz(self, points_zyx_selected: torch.Tensor) -> torch.Tensor:
        spacing = float(getattr(self.trace_record, "volume_spacing_base", 1.0))
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise ValueError(f"invalid volume_spacing_base for debug normal sampling: {spacing!r}")
        points_base_zyx = points_zyx_selected.to(dtype=torch.float32) * float(spacing)
        return torch.stack(
            [points_base_zyx[:, 2], points_base_zyx[:, 1], points_base_zyx[:, 0]],
            dim=1,
        )

    def _channel_shape_xyz_tensor(self, channel: str) -> torch.Tensor:
        shape_zyx: tuple[int, int, int] | None = None
        for cache in getattr(self.data, "sparse_caches", {}).values():
            if channel in set(getattr(cache, "channels", ())):
                shape_zyx = tuple(int(v) for v in getattr(cache, "vol_shape_zyx"))
                break
        if shape_zyx is None:
            tensor = getattr(self.data, channel, None)
            shape_zyx = tuple(int(v) for v in self.data._size_of(tensor))
        z, y, x = shape_zyx
        return torch.tensor([x, y, z], dtype=torch.float32, device=self.device)

    def _prefetch_sparse_points(self, xyz_fullres: torch.Tensor, *, channels: set[str]) -> None:
        sparse_caches = getattr(self.data, "sparse_caches", None) or {}
        for cache in sparse_caches.values():
            cache_channels = set(getattr(cache, "channels", ()))
            if not (cache_channels & channels):
                continue
            cache.prefetch(
                xyz_fullres,
                self.data.origin_fullres,
                self.data._spacing_for(cache.channels[0]),
            )
        for cache in sparse_caches.values():
            if set(getattr(cache, "channels", ())) & channels:
                cache.sync()

    def _corner_principal_normals(
        self,
        points_fullres_xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = int(points_fullres_xyz.shape[0])
        if count == 0:
            return (
                torch.zeros((0, 3), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.bool, device=self.device),
            )
        nx_spacing = tuple(float(v) for v in self.data._spacing_for("nx"))
        ny_spacing = tuple(float(v) for v in self.data._spacing_for("ny"))
        if any(abs(a - b) > 1.0e-6 for a, b in zip(nx_spacing, ny_spacing)):
            raise ValueError(
                f"Lasagna sparse nx and ny spacing mismatch: nx={nx_spacing} ny={ny_spacing}"
            )
        origin = torch.tensor(self.data.origin_fullres, dtype=torch.float32, device=self.device)
        normal_spacing = torch.tensor(nx_spacing, dtype=torch.float32, device=self.device)
        normal_shape_xyz = self._channel_shape_xyz_tensor("nx")
        local_xyz = (points_fullres_xyz - origin.view(1, 3)) / normal_spacing.view(1, 3)
        in_bounds = (
            torch.isfinite(local_xyz).all(dim=1)
            & torch.all(local_xyz >= 0.0, dim=1)
            & torch.all(local_xyz <= (normal_shape_xyz - 1.0), dim=1)
        )
        base_xyz = torch.floor(local_xyz).to(dtype=torch.long)
        frac_xyz = local_xyz - base_xyz.to(dtype=torch.float32)
        max_index_xyz = (normal_shape_xyz.to(dtype=torch.long) - 1).clamp_min(0)
        corner_bits = torch.tensor(
            _NORMAL_CORNER_BITS_XYZ,
            dtype=torch.float32,
            device=self.device,
        )
        corner_index_xyz = (
            base_xyz[:, None, :] + corner_bits.to(dtype=torch.long)[None, :, :]
        ).clamp(min=0)
        corner_index_xyz = torch.minimum(corner_index_xyz, max_index_xyz.view(1, 1, 3))
        corner_points_fullres_xyz = (
            origin.view(1, 1, 3)
            + corner_index_xyz.to(dtype=torch.float32) * normal_spacing.view(1, 1, 3)
        ).reshape(1, 1, count * 8, 3).contiguous()
        point_query = points_fullres_xyz.reshape(1, 1, count, 3).contiguous()
        with self._measure("sparse_normal_prefetch", sync=True):
            self._prefetch_sparse_points(
                torch.cat([point_query, corner_points_fullres_xyz], dim=2),
                channels=set(_SPARSE_NORMAL_CHANNELS),
            )
        with self._measure("sparse_normal_sample", sync=True):
            grad_sampled = self.data.grid_sample_fullres(
                point_query,
                channels={"grad_mag"},
            )
            normal_sampled = self.data.grid_sample_fullres(
                corner_points_fullres_xyz,
                channels={"nx", "ny"},
            )
            if (
                grad_sampled.grad_mag is None
                or normal_sampled.nx is None
                or normal_sampled.ny is None
            ):
                raise ValueError("sparse-corner normal sampler did not return grad_mag/nx/ny")
            grad = grad_sampled.grad_mag.reshape(count).to(device=self.device, dtype=torch.float32)
            nx = normal_sampled.nx.reshape(count, 8).to(device=self.device, dtype=torch.float32)
            ny = normal_sampled.ny.reshape(count, 8).to(device=self.device, dtype=torch.float32)
        nz = torch.sqrt(torch.clamp(1.0 - nx * nx - ny * ny, min=0.0))
        decoded = torch.stack([nx, ny, nz], dim=2)
        norm = torch.linalg.norm(decoded, dim=2)
        corner_valid = torch.isfinite(norm) & (norm > float(_EPS))
        decoded = decoded / norm.clamp_min(float(_EPS))[:, :, None]
        decoded = torch.where(corner_valid[:, :, None], decoded, torch.zeros_like(decoded))
        corner_weights = torch.where(
            corner_bits[None, :, :] > 0.5,
            frac_xyz[:, None, :],
            1.0 - frac_xyz[:, None, :],
        )
        weights = torch.prod(corner_weights, dim=2)
        weights = torch.where(corner_valid, weights, torch.zeros_like(weights))
        hint = torch.sum(decoded * weights[:, :, None], dim=1)
        tensor = torch.einsum("nc,nci,ncj->nij", weights, decoded, decoded)
        total_weight = torch.sum(weights, dim=1)
        valid_t = (
            in_bounds
            & torch.isfinite(grad)
            & (grad > 0.0)
            & torch.isfinite(total_weight)
            & (total_weight > float(_EPS))
        )
        axes = _principal_tensor_axes_torch(
            tensor,
            hint,
            method=str(self.principal_axis_method),
        )
        axis_norm = torch.linalg.norm(axes, dim=1)
        axis_valid = torch.isfinite(axis_norm) & (axis_norm > float(_EPS))
        valid_t = valid_t & axis_valid
        axes = torch.where(valid_t[:, None], axes, torch.zeros_like(axes))
        normals_t = axes[:, [2, 1, 0]].contiguous()
        return normals_t, valid_t

    def __call__(
        self,
        points_zyx_selected: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        points = torch.as_tensor(points_zyx_selected, dtype=torch.float32, device=self.device)
        if points.ndim != 2 or int(points.shape[1]) != 3:
            raise ValueError("points_zyx_selected must have shape [N,3]")
        if int(points.shape[0]) == 0:
            return (
                torch.zeros((0, 3), dtype=torch.float32, device=self.device),
                torch.zeros((0,), dtype=torch.bool, device=self.device),
            )
        points_fullres_xyz = self._selected_zyx_to_fullres_xyz(points)
        return self._corner_principal_normals(points_fullres_xyz)


def _make_sparse_corner_normal_sampler(
    *,
    trace_record: Any,
    raw_config: dict[str, Any],
    device: torch.device,
    principal_axis_method: str = "eigh",
    profiler: _NativeTraceProfiler | None,
) -> NativeTraceNormalSampler:
    manifest_path = _native_trace_manifest_path_for_record(trace_record, raw_config=raw_config)
    if manifest_path is None:
        raise ValueError(
            "native 3D Trace2CP sparse normal sampling requires dataset lasagna_manifest_path"
        )
    fit_data = _import_lasagna_fit_data()
    volume = fit_data.LasagnaVolume.load(str(manifest_path))
    channels = set(volume.all_channels())
    missing = _SPARSE_NORMAL_CHANNELS - channels
    if missing:
        raise ValueError(
            "native 3D Trace2CP sparse normal sampling requires Lasagna "
            f"channels {sorted(_SPARSE_NORMAL_CHANNELS)}; missing {sorted(missing)} "
            f"in manifest {manifest_path}"
        )
    skip_channels = {channel for channel in channels if channel not in _SPARSE_NORMAL_CHANNELS}
    data = fit_data.load_3d_streaming(
        path=str(manifest_path),
        device=torch.device(device),
        skip_channels=skip_channels,
        sparse_prefetch_backend="tensorstore",
    )
    if not getattr(data, "sparse_caches", None):
        raise ValueError(
            "native 3D Trace2CP sparse normal sampling requires Lasagna sparse caches"
        )
    return _SparseCornerLasagnaNormalSampler(
        trace_record=trace_record,
        data=data,
        device=device,
        principal_axis_method=str(principal_axis_method),
        profiler=profiler,
    )


@dataclass
class _FailFastNormalComparisonSampler:
    primary: NativeTraceNormalSampler
    alternates: tuple[tuple[str, NativeTraceNormalSampler], ...]
    angle_threshold_degrees: float = 1.0
    call_count: int = 0

    @staticmethod
    def _valid_to_numpy(valid: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(valid, torch.Tensor):
            return valid.detach().to(device=torch.device("cpu"), dtype=torch.bool).numpy()
        return np.asarray(valid, dtype=bool)

    def _raise_valid_mismatch(
        self,
        *,
        label: str,
        point_index: int,
        points: np.ndarray,
        primary_normals: np.ndarray,
        primary_valid: np.ndarray,
        alt_normals: np.ndarray,
        alt_valid: np.ndarray,
    ) -> None:
        raise ValueError(
            "native 3D Trace2CP normal comparison failed: "
            f"sampler={label} call={int(self.call_count)} point_index={int(point_index)} "
            f"reason=valid_mismatch point_zyx_selected={points[int(point_index)].tolist()} "
            f"baseline_valid={bool(primary_valid[int(point_index)])} "
            f"accelerated_valid={bool(alt_valid[int(point_index)])} "
            f"baseline_normal_zyx={primary_normals[int(point_index)].tolist()} "
            f"accelerated_normal_zyx={alt_normals[int(point_index)].tolist()}"
        )

    def _raise_angle_mismatch(
        self,
        *,
        label: str,
        point_index: int,
        angle_degrees: float,
        points: np.ndarray,
        primary_normals: np.ndarray,
        alt_normals: np.ndarray,
    ) -> None:
        raise ValueError(
            "native 3D Trace2CP normal comparison failed: "
            f"sampler={label} call={int(self.call_count)} point_index={int(point_index)} "
            f"reason=angle_mismatch angle_degrees={float(angle_degrees):.6f} "
            f"threshold_degrees={float(self.angle_threshold_degrees):.6f} "
            f"point_zyx_selected={points[int(point_index)].tolist()} "
            f"baseline_normal_zyx={primary_normals[int(point_index)].tolist()} "
            f"accelerated_normal_zyx={alt_normals[int(point_index)].tolist()}"
        )

    def _compare(
        self,
        *,
        label: str,
        points: np.ndarray,
        primary_normals: torch.Tensor | np.ndarray,
        primary_valid: torch.Tensor | np.ndarray,
        alt_normals: torch.Tensor | np.ndarray,
        alt_valid: torch.Tensor | np.ndarray,
    ) -> None:
        primary_n = _points_to_numpy(primary_normals)
        alt_n = _points_to_numpy(alt_normals)
        primary_v = self._valid_to_numpy(primary_valid)
        alt_v = self._valid_to_numpy(alt_valid)
        if primary_n.shape != alt_n.shape:
            raise ValueError(
                "native 3D Trace2CP normal comparison failed: "
                f"sampler={label} call={int(self.call_count)} reason=shape_mismatch "
                f"baseline_shape={primary_n.shape} accelerated_shape={alt_n.shape}"
            )
        if primary_v.shape != alt_v.shape:
            raise ValueError(
                "native 3D Trace2CP normal comparison failed: "
                f"sampler={label} call={int(self.call_count)} reason=valid_shape_mismatch "
                f"baseline_shape={primary_v.shape} accelerated_shape={alt_v.shape}"
            )
        mismatch = np.flatnonzero(primary_v ^ alt_v)
        if mismatch.size:
            self._raise_valid_mismatch(
                label=label,
                point_index=int(mismatch[0]),
                points=points,
                primary_normals=primary_n,
                primary_valid=primary_v,
                alt_normals=alt_n,
                alt_valid=alt_v,
            )
        both = primary_v & alt_v
        if not bool(np.any(both)):
            return
        primary_b = primary_n[both].astype(np.float64, copy=False)
        alt_b = alt_n[both].astype(np.float64, copy=False)
        primary_norm = np.linalg.norm(primary_b, axis=1)
        alt_norm = np.linalg.norm(alt_b, axis=1)
        finite = (
            np.isfinite(primary_b).all(axis=1)
            & np.isfinite(alt_b).all(axis=1)
            & np.isfinite(primary_norm)
            & np.isfinite(alt_norm)
            & (primary_norm > 1.0e-12)
            & (alt_norm > 1.0e-12)
        )
        if not bool(np.all(finite)):
            bad_local = int(np.flatnonzero(~finite)[0])
            bad_index = int(np.flatnonzero(both)[bad_local])
            self._raise_valid_mismatch(
                label=label,
                point_index=bad_index,
                points=points,
                primary_normals=primary_n,
                primary_valid=primary_v,
                alt_normals=alt_n,
                alt_valid=alt_v,
            )
        primary_b = primary_b / primary_norm[:, None]
        alt_b = alt_b / alt_norm[:, None]
        dot = np.abs(np.sum(primary_b * alt_b, axis=1))
        angles = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        bad = np.flatnonzero(angles > float(self.angle_threshold_degrees))
        if bad.size:
            bad_local = int(bad[0])
            bad_index = int(np.flatnonzero(both)[bad_local])
            self._raise_angle_mismatch(
                label=label,
                point_index=bad_index,
                angle_degrees=float(angles[bad_local]),
                points=points,
                primary_normals=primary_n,
                alt_normals=alt_n,
            )

    def __call__(
        self,
        points_zyx_selected: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        self.call_count += 1
        points = _points_to_numpy(points_zyx_selected)
        primary_normals, primary_valid = self.primary(points_zyx_selected)
        selected_normals: torch.Tensor | np.ndarray = primary_normals
        selected_valid: torch.Tensor | np.ndarray = primary_valid
        for label, sampler in self.alternates:
            alt_normals, alt_valid = sampler(points_zyx_selected)
            self._compare(
                label=label,
                points=points,
                primary_normals=primary_normals,
                primary_valid=primary_valid,
                alt_normals=alt_normals,
                alt_valid=alt_valid,
            )
            if selected_normals is primary_normals:
                selected_normals = alt_normals
                selected_valid = alt_valid
        return selected_normals, selected_valid


def _make_debug_normal_comparison_sampler(
    *,
    primary: NativeTraceNormalSampler,
    trace_record: Any,
    raw_config: dict[str, Any],
    device: torch.device,
    mode: str,
    principal_axis_method: str,
    angle_threshold_degrees: float,
    profiler: _NativeTraceProfiler | None,
) -> NativeTraceNormalSampler:
    if mode != "sparse-corner-principal":
        raise ValueError(f"unsupported debug normal comparison mode: {mode!r}")
    if (
        not math.isfinite(float(angle_threshold_degrees))
        or float(angle_threshold_degrees) < 0.0
    ):
        raise ValueError("debug normal angle threshold must be finite and non-negative")
    alternates = (
        (
            "sparse-corner-principal",
            _make_sparse_corner_normal_sampler(
                trace_record=trace_record,
                device=device,
                raw_config=raw_config,
                principal_axis_method=str(principal_axis_method),
                profiler=profiler,
            ),
        ),
    )
    return _FailFastNormalComparisonSampler(
        primary=primary,
        alternates=alternates,
        angle_threshold_degrees=float(angle_threshold_degrees),
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


def _profile_span_for_cache(
    cache: Any,
    name: str,
    *,
    sync: bool = False,
) -> _NativeTraceProfileSpan | _NullNativeTraceProfileSpan:
    profiler = getattr(cache, "profiler", None)
    if profiler is None:
        return _NullNativeTraceProfileSpan()
    return profiler.measure(
        name,
        sync_device=_cache_device(cache) if bool(sync) else None,
    )


def _sample_point_choices_for_points_torch(
    cache: Any,
    points_zyx: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fallback_device = points_zyx.device if isinstance(points_zyx, torch.Tensor) else None
    device = _cache_device(cache, fallback=fallback_device)
    if hasattr(cache, "sample_point_choices_torch"):
        points = torch.as_tensor(points_zyx, dtype=torch.float32, device=device)
        directions, presence, valid = cache.sample_point_choices_torch(points)
    else:
        points_np = _points_to_numpy(points_zyx)
        directions_one, presence_one, valid_one = cache.sample_points_torch(points_np)
        directions = directions_one[:, None, :]
        presence = presence_one[:, None]
        valid = valid_one[:, None]
    return (
        directions.to(device=device, dtype=torch.float32),
        presence.to(device=device, dtype=torch.float32),
        valid.to(device=device, dtype=torch.bool),
    )


def _select_aligned_point_choices_torch(
    directions: torch.Tensor,
    presence: torch.Tensor,
    valid: torch.Tensor,
    reference_directions_zyx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if directions.ndim != 3 or int(directions.shape[2]) != 3:
        raise ValueError("directions must have shape [N,B,3]")
    if presence.shape != directions.shape[:2]:
        raise ValueError("presence must have shape [N,B]")
    if valid.shape != directions.shape[:2]:
        raise ValueError("valid must have shape [N,B]")
    device = directions.device
    references = F.normalize(
        reference_directions_zyx.to(device=device, dtype=torch.float32),
        p=2.0,
        dim=1,
        eps=float(_EPS),
    )
    if references.shape != (int(directions.shape[0]), 3):
        raise ValueError("reference_directions_zyx must have shape [N,3]")
    aligned = _align_axes_torch(directions, references[:, None, :].expand_as(directions))
    dot = torch.sum(aligned * references[:, None, :].expand_as(aligned), dim=2).clamp(0.0, 1.0)
    score = torch.where(
        valid.to(device=device, dtype=torch.bool),
        dot * presence.to(device=device, dtype=torch.float32).clamp(0.0, 1.0),
        torch.full_like(presence.to(device=device, dtype=torch.float32), -torch.inf),
    )
    best_branch = torch.argmax(score, dim=1)
    rows = torch.arange(int(directions.shape[0]), dtype=torch.long, device=device)
    any_valid = valid.to(device=device, dtype=torch.bool).any(dim=1)
    selected_direction = aligned[rows, best_branch]
    selected_presence = presence.to(device=device, dtype=torch.float32)[rows, best_branch]
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
    return _select_aligned_point_choices_torch(directions, presence, valid, references)


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


def _sample_trace_start_direction_aligned(
    cache: Any,
    point_zyx: np.ndarray,
    *,
    cp_reference_direction_zyx: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    reference_np = _require_unit(
        cp_reference_direction_zyx,
        label="native 3D Trace2CP CP reference direction",
    )
    if hasattr(cache, "sample_point_choices_torch"):
        directions, presence, valid = cache.sample_point_choices_torch(
            np.asarray(point_zyx, dtype=np.float32).reshape(1, 3)
        )
        if not bool(valid[0].any().detach().cpu()):
            return np.zeros((3,), dtype=np.float32), 0.0, False
        reference = torch.as_tensor(
            reference_np.reshape(1, 3),
            dtype=torch.float32,
            device=directions.device,
        )
        reference = F.normalize(reference, p=2.0, dim=1, eps=float(_EPS))
        aligned = _align_axes_torch(directions[0], reference.expand_as(directions[0]))
        dot = torch.sum(aligned * reference.expand_as(aligned), dim=1).clamp(0.0, 1.0)
        score = torch.where(
            valid[0],
            dot,
            torch.full_like(dot, -torch.inf),
        )
        branch_index = int(torch.argmax(score).detach().cpu())
        axis_zyx = aligned[branch_index].detach().cpu().numpy().astype(np.float32)
        return _unit(axis_zyx), float(presence[0, branch_index].detach().cpu()), True
    sampled_direction, sampled_presence, valid = cache.sample_point(point_zyx)
    if not bool(valid):
        return np.zeros((3,), dtype=np.float32), 0.0, False
    return _align_axis(sampled_direction, reference_np), float(sampled_presence), True


def _optional_nonnegative_float(value: float | None, *, name: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative when set")
    return out


def _native_trace_requires_normal_sampler(cfg: NativeTrace2CpConfig) -> bool:
    tangent = _optional_nonnegative_float(
        cfg.smoothness_tangent_weight,
        name="smoothness_tangent_weight",
    )
    normal = _optional_nonnegative_float(
        cfg.smoothness_normal_weight,
        name="smoothness_normal_weight",
    )
    cumulative = _optional_nonnegative_float(
        cfg.cumulative_smoothness_tangent_weight,
        name="cumulative_smoothness_tangent_weight",
    )
    return bool(
        (tangent is not None and tangent > 0.0)
        or (normal is not None and normal > 0.0)
        or (cumulative is not None and cumulative > 0.0)
    )


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
        if (
            (tangent_weight is not None and tangent_weight > 0.0)
            or (normal_weight is not None and normal_weight > 0.0)
        ):
            raise ValueError(
                "candidate_normals are required for native 3D Trace2CP "
                "tangent/normal smoothness"
            )
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
    if weight <= 0.0:
        return torch.zeros(candidates.shape[:2], dtype=torch.float32, device=candidates.device)
    if candidate_normals is None:
        raise ValueError(
            "candidate_normals are required for native 3D Trace2CP "
            "cumulative tangent smoothness"
        )

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
    persist_line: bool = False,
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
    message = (
        f"native {label} [{bar}] {current_i}/{total_i} "
        f"elapsed={_format_eta(elapsed)} eta={_format_eta(eta)}{suffix}"
    )
    complete = current_i >= total_i
    print(f"\r{message}", end="\n" if complete or persist_line else "", flush=True)


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
    return_next_selected: bool = False,
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
):
    substeps = int(candidate_substeps)
    if substeps < 1:
        raise ValueError("candidate_substeps must be at least 1")
    device = _cache_device(cache, fallback=current_directions.device)
    with _profile_span_for_cache(cache, "score_prepare"):
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
    with _profile_span_for_cache(cache, "score_align_dots"):
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
    with _profile_span_for_cache(cache, "score_smoothness"):
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
            smoothness_loss = smoothness_loss + cumulative_loss

    if substeps == 1:
        flat_points = points.reshape(state_count * candidate_count, 3)
        with _profile_span_for_cache(cache, "score_sample_points"):
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
        presence_choices = presence_choices.reshape(
            state_count,
            candidate_count,
            branch_count,
        )
        valid_choices = valid_choices.reshape(state_count, candidate_count, branch_count)
        with _profile_span_for_cache(cache, "score_branch_math"):
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
        result = (
            total_loss,
            direction_loss,
            presence_loss,
            smoothness_loss,
            candidate_valid,
            rejected_per_state,
        )
        if not bool(return_next_selected):
            return result
        selected_direction, selected_presence, selected_valid = (
            _select_aligned_point_choices_torch(
                next_direction_choices.reshape(state_count * candidate_count, branch_count, 3),
                presence_choices.reshape(state_count * candidate_count, branch_count),
                valid_choices.reshape(state_count * candidate_count, branch_count),
                candidates.reshape(state_count * candidate_count, 3),
            )
        )
        return (
            *result,
            selected_direction.reshape(state_count, candidate_count, 3),
            selected_presence.reshape(state_count, candidate_count),
            selected_valid.reshape(state_count, candidate_count),
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
    result = (
        total_loss_2d[:, :, None],
        direction_loss_2d[:, :, None],
        presence_loss_2d[:, :, None],
        smoothness_loss,
        candidate_valid,
        rejected_per_state,
    )
    if not bool(return_next_selected):
        return result
    endpoint_directions = next_direction_choices[:, :, -1].reshape(
        state_count * candidate_count,
        branch_count,
        3,
    )
    endpoint_presence = presence_choices[:, :, -1].reshape(
        state_count * candidate_count,
        branch_count,
    )
    endpoint_valid = valid_choices[:, :, -1].reshape(
        state_count * candidate_count,
        branch_count,
    )
    selected_direction, selected_presence, selected_valid = _select_aligned_point_choices_torch(
        endpoint_directions,
        endpoint_presence,
        endpoint_valid,
        candidates.reshape(state_count * candidate_count, 3),
    )
    return (
        *result,
        selected_direction.reshape(state_count, candidate_count, 3),
        selected_presence.reshape(state_count, candidate_count),
        selected_valid.reshape(state_count, candidate_count),
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


def _normalize_target_planes(
    *,
    target_zyx: np.ndarray,
    target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
    target_plane_normal_zyx: np.ndarray | None = None,
) -> tuple[NativeTargetPlane, ...]:
    if target_planes_zyx is not None and target_plane_normal_zyx is not None:
        raise ValueError("provide target_planes_zyx or target_plane_normal_zyx, not both")
    target = np.asarray(target_zyx, dtype=np.float32)
    planes: list[NativeTargetPlane] = []
    if target_planes_zyx is not None:
        for index, plane in enumerate(target_planes_zyx):
            point = np.asarray(plane.point_zyx, dtype=np.float32)
            normal = _require_unit(
                plane.normal_zyx,
                label=f"target plane {index} ({plane.name}) normal",
            )
            planes.append(
                NativeTargetPlane(
                    name=str(plane.name),
                    point_zyx=point.astype(np.float32, copy=False),
                    normal_zyx=normal.astype(np.float32, copy=False),
                )
            )
    elif target_plane_normal_zyx is not None:
        planes.append(
            NativeTargetPlane(
                name="explicit",
                point_zyx=target.astype(np.float32, copy=False),
                normal_zyx=_require_unit(
                    target_plane_normal_zyx,
                    label="target_plane_normal_zyx",
                ).astype(np.float32, copy=False),
            )
        )
    else:
        raise ValueError(
            "native 3D Trace2CP requires explicit target planes; "
            "CP-to-CP chord target planes are not allowed"
        )
    if not planes:
        raise ValueError("native 3D Trace2CP requires at least one target plane")
    names = [plane.name for plane in planes]
    if len(set(names)) != len(names):
        raise ValueError(f"native 3D Trace2CP target plane names must be unique: {names}")
    return tuple(planes)


def _missing_target_plane_reason(
    base_reason: str,
    planes: Sequence[NativeTargetPlane],
    crossed: Sequence[bool],
) -> str:
    missing = [
        str(plane.name)
        for plane, is_crossed in zip(planes, crossed, strict=True)
        if not bool(is_crossed)
    ]
    if not missing:
        return str(base_reason)
    return f"{base_reason}:missing_target_planes={','.join(missing)}"


def _target_plane_crossing_summary(
    planes: Sequence[NativeTargetPlane],
    crossed: Sequence[bool],
    crossings_zyx: np.ndarray,
) -> tuple[tuple[NativeTargetPlaneCrossing, ...], NativeTargetPlaneCrossing | None]:
    points = np.asarray(crossings_zyx, dtype=np.float32)
    out: list[NativeTargetPlaneCrossing] = []
    for index, (plane, is_crossed) in enumerate(zip(planes, crossed, strict=True)):
        if not bool(is_crossed):
            continue
        point = points[index]
        error = _target_plane_in_plane_error_voxels(
            point,
            target_zyx=plane.point_zyx,
            plane_normal_zyx=plane.normal_zyx,
        )
        out.append(
            NativeTargetPlaneCrossing(
                name=str(plane.name),
                point_zyx=point.astype(np.float32, copy=False),
                error_voxels=float(error),
            )
        )
    selected = None if not out else min(out, key=lambda item: item.error_voxels)
    return tuple(out), selected


def _update_target_plane_crossings(
    *,
    start_zyx: np.ndarray,
    end_zyx: np.ndarray,
    planes: Sequence[NativeTargetPlane],
    crossed: np.ndarray,
    crossings_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    next_crossed = np.asarray(crossed, dtype=bool).copy()
    next_crossings = np.asarray(crossings_zyx, dtype=np.float32).copy()
    for index, plane in enumerate(planes):
        if bool(next_crossed[index]):
            continue
        crossing = _interpolate_plane_crossing(
            start_zyx,
            end_zyx,
            plane_point_zyx=plane.point_zyx,
            plane_normal_zyx=plane.normal_zyx,
        )
        if crossing is None:
            continue
        next_crossed[index] = True
        next_crossings[index] = crossing.astype(np.float32, copy=False)
    return next_crossed, next_crossings


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
    fallback = float(cfg.smoothness_weight)
    if not math.isfinite(fallback) or fallback < 0.0:
        raise ValueError("smoothness_weight must be finite and non-negative")
    if normal_sampler is None:
        if _native_trace_requires_normal_sampler(cfg):
            raise ValueError(
                "native 3D Trace2CP requires Lasagna normals for "
                "tangent/normal smoothness; provide a normal_sampler"
            )
        return replace(cfg, smoothness_tangent_weight=None, smoothness_normal_weight=None)
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


def _fiber_target_line_neighbor_planes_zyx(
    record: Any,
    *,
    target_control_point_index: int,
    target_zyx: np.ndarray | None = None,
) -> tuple[NativeTargetPlane, ...]:
    fiber = record.fiber
    line_points_xyz = np.asarray(fiber.line_points_xyz, dtype=np.float64)
    if line_points_xyz.ndim != 2 or line_points_xyz.shape[1] != 3:
        raise ValueError("fiber line_points_xyz must have shape [N, 3]")
    target_line_index = int(control_point_line_index(fiber, int(target_control_point_index)))
    if target_line_index < 0 or target_line_index >= int(line_points_xyz.shape[0]):
        raise ValueError(
            "native 3D Trace2CP target CP line index is out of range: "
            f"target_cp={int(target_control_point_index)} "
            f"line_index={target_line_index} line_points={int(line_points_xyz.shape[0])}"
        )
    spacing = float(getattr(record, "volume_spacing_base", 1.0))
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume_spacing_base for native 3D target planes: {spacing!r}")
    target = (
        np.asarray(record.fiber.control_points_zyx[int(target_control_point_index)], dtype=np.float32)
        / np.float32(spacing)
        if target_zyx is None
        else np.asarray(target_zyx, dtype=np.float32)
    )
    center_xyz = line_points_xyz[target_line_index]
    planes: list[NativeTargetPlane] = []
    for name, neighbor_index in (
        ("line_next", target_line_index + 1),
        ("line_prev", target_line_index - 1),
    ):
        if neighbor_index < 0 or neighbor_index >= int(line_points_xyz.shape[0]):
            continue
        normal_zyx = (line_points_xyz[neighbor_index] - center_xyz)[[2, 1, 0]] / spacing
        planes.append(
            NativeTargetPlane(
                name=name,
                point_zyx=target.astype(np.float32, copy=False),
                normal_zyx=_require_unit(
                    normal_zyx,
                    label=(
                        "native 3D Trace2CP target line-neighbor plane "
                        f"{name} cp={int(target_control_point_index)}"
                    ),
                ).astype(np.float32, copy=False),
            )
        )
    if not planes:
        raise ValueError(
            "native 3D Trace2CP target CP has no line-neighbor target planes: "
            f"target_cp={int(target_control_point_index)} line_index={target_line_index}"
        )
    return tuple(planes)


def _fiber_target_planes_zyx(
    cache: NativeTraceFieldCache,
    record: Any,
    *,
    target_control_point_index: int,
    target_zyx: np.ndarray,
    inference_reference_direction_zyx: np.ndarray,
    include_inference_plane: bool = True,
) -> tuple[NativeTargetPlane, ...]:
    target = np.asarray(target_zyx, dtype=np.float32)
    planes = list(
        _fiber_target_line_neighbor_planes_zyx(
            record,
            target_control_point_index=int(target_control_point_index),
            target_zyx=target,
        )
    )
    if bool(include_inference_plane):
        direction, _presence, valid = _sample_trace_start_direction_aligned(
            cache,
            target,
            cp_reference_direction_zyx=inference_reference_direction_zyx,
        )
        if not bool(valid):
            raise ValueError(
                "native 3D Trace2CP target inference plane sample is invalid: "
                f"target_cp={int(target_control_point_index)} "
                f"target_zyx={target.tolist()}"
            )
        planes.append(
            NativeTargetPlane(
                name="inferred_direction",
                point_zyx=target.astype(np.float32, copy=False),
                normal_zyx=_require_unit(
                    direction,
                    label=(
                        "native 3D Trace2CP target inferred-direction plane "
                        f"cp={int(target_control_point_index)}"
                    ),
                ).astype(np.float32, copy=False),
            )
        )
    return _normalize_target_planes(target_zyx=target, target_planes_zyx=planes)


def _trace_native_3d_one_way_greedy(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
    target_plane_normal_zyx: np.ndarray | None = None,
    budget_span_voxels: float | None = None,
    progress_label: str | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTraceResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    target_planes = _normalize_target_planes(
        target_zyx=target,
        target_planes_zyx=target_planes_zyx,
        target_plane_normal_zyx=target_plane_normal_zyx,
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
        progress = 1.0 - float(
            np.linalg.norm(np.asarray(point_zyx, dtype=np.float32) - target)
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
    with _profile_span_for_cache(cache, "trace_start_sample"):
        initial_direction, _presence, valid = _sample_trace_start_direction_aligned(
            cache,
            start,
            cp_reference_direction_zyx=initial_direction_zyx,
        )
    if not valid:
        raise ValueError(f"native 3D Trace2CP start point is invalid: {start.tolist()}")
    previous_direction = initial_direction.astype(np.float32, copy=False)
    history_direction = initial_direction.astype(np.float32, copy=False)
    trace: list[np.ndarray] = [start.astype(np.float32)]
    steps: list[NativeTraceStep] = []
    current = start.astype(np.float32)
    target_plane_crossed = np.zeros((len(target_planes),), dtype=bool)
    target_plane_crossings = np.zeros((len(target_planes), 3), dtype=np.float32)
    target_plane_crossed, target_plane_crossings = _update_target_plane_crossings(
        start_zyx=start,
        end_zyx=start,
        planes=target_planes,
        crossed=target_plane_crossed,
        crossings_zyx=target_plane_crossings,
    )
    for _step_index in range(step_limit):
        if _step_index == 0:
            current_direction = initial_direction.astype(np.float32, copy=False)
        else:
            with _profile_span_for_cache(cache, "trace_current_sample"):
                sampled_direction, _presence, valid = _sample_trace_point_aligned(
                    cache,
                    current,
                    reference_direction_zyx=previous_direction,
                )
            if not valid:
                emit_progress(current, _step_index, reason="invalid_current_point")
                crossings, selected = _target_plane_crossing_summary(
                    target_planes,
                    target_plane_crossed,
                    target_plane_crossings,
                )
                return NativeTraceResult(
                    trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                    reached_target_plane=False,
                    reason=_missing_target_plane_reason(
                        "invalid_current_point",
                        target_planes,
                        target_plane_crossed,
                    ),
                    steps=tuple(steps),
                    target_plane_crossings=crossings,
                    selected_target_plane_name=None if selected is None else selected.name,
                    selected_target_plane_crossing_zyx=None
                    if selected is None
                    else selected.point_zyx,
                    selected_target_plane_error_voxels=math.inf
                    if selected is None
                    else float(selected.error_voxels),
                )
            current_direction = _align_axis(sampled_direction, previous_direction)
        with _profile_span_for_cache(cache, "trace_candidate_grid"):
            candidates_unit = _trace_candidate_directions(current_direction, cfg)
            next_points = current[None, :] + candidates_unit * np.float32(cfg.step_voxels)
        with _profile_span_for_cache(cache, "trace_candidate_normals"):
            sampled_normals = _sample_candidate_normals_torch(
                normal_sampler,
                next_points,
                device=_cache_device(cache),
            )
        candidate_normals = None if sampled_normals is None else sampled_normals[0]
        candidate_normals_valid = None if sampled_normals is None else sampled_normals[1]
        with _profile_span_for_cache(cache, "trace_candidate_score"):
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
                )
            )
        if best_index is None:
            emit_progress(current, _step_index, reason="all_candidates_invalid")
            crossings, selected = _target_plane_crossing_summary(
                target_planes,
                target_plane_crossed,
                target_plane_crossings,
            )
            return NativeTraceResult(
                trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                reached_target_plane=False,
                reason=_missing_target_plane_reason(
                    "all_candidates_invalid",
                    target_planes,
                    target_plane_crossed,
                ),
                steps=tuple(steps),
                target_plane_crossings=crossings,
                selected_target_plane_name=None if selected is None else selected.name,
                selected_target_plane_crossing_zyx=None
                if selected is None
                else selected.point_zyx,
                selected_target_plane_error_voxels=math.inf
                if selected is None
                else float(selected.error_voxels),
            )
        chosen_direction = _align_axis(candidates_unit[int(best_index)], current_direction)
        next_point = (current + chosen_direction * np.float32(cfg.step_voxels)).astype(np.float32)
        next_crossed, next_crossings = _update_target_plane_crossings(
            start_zyx=current,
            end_zyx=next_point,
            planes=target_planes,
            crossed=target_plane_crossed,
            crossings_zyx=target_plane_crossings,
        )
        if bool(np.all(next_crossed)):
            crossings, selected = _target_plane_crossing_summary(
                target_planes,
                next_crossed,
                next_crossings,
            )
            if selected is None:
                raise RuntimeError("all target planes crossed but no selected crossing was found")
            trace.append(selected.point_zyx.astype(np.float32))
            steps.append(
                NativeTraceStep(
                    point_zyx=selected.point_zyx.astype(np.float32),
                    direction_loss=float(direction_loss),
                    presence_loss=float(presence_loss),
                    total_loss=float(_total),
                    rejected_candidates=int(rejected),
                    smoothness_loss=float(smoothness_loss),
                )
            )
            emit_progress(selected.point_zyx, _step_index + 1, reason="target_plane")
            return NativeTraceResult(
                trace_zyx=np.stack(trace, axis=0).astype(np.float32),
                reached_target_plane=True,
                reason="target_plane",
                steps=tuple(steps),
                target_plane_crossings=crossings,
                selected_target_plane_name=selected.name,
                selected_target_plane_crossing_zyx=selected.point_zyx,
                selected_target_plane_error_voxels=float(selected.error_voxels),
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
        target_plane_crossed = next_crossed
        target_plane_crossings = next_crossings
        emit_progress(current, _step_index + 1)
    emit_progress(current, progress_max, reason=limit_reason)
    crossings, selected = _target_plane_crossing_summary(
        target_planes,
        target_plane_crossed,
        target_plane_crossings,
    )
    return NativeTraceResult(
        trace_zyx=np.stack(trace, axis=0).astype(np.float32),
        reached_target_plane=False,
        reason=_missing_target_plane_reason(limit_reason, target_planes, target_plane_crossed),
        steps=tuple(steps),
        target_plane_crossings=crossings,
        selected_target_plane_name=None if selected is None else selected.name,
        selected_target_plane_crossing_zyx=None if selected is None else selected.point_zyx,
        selected_target_plane_error_voxels=math.inf
        if selected is None
        else float(selected.error_voxels),
    )


def _beam_node_result(
    node: _NativeBeamNode,
    *,
    reached_target_plane: bool,
    reason: str,
    target_planes: Sequence[NativeTargetPlane] | None = None,
    selected_crossing: NativeTargetPlaneCrossing | None = None,
) -> NativeTraceResult:
    nodes: list[_NativeBeamNode] = []
    current: _NativeBeamNode | None = node
    while current is not None:
        nodes.append(current)
        current = current.parent
    nodes.reverse()
    trace = np.stack([np.asarray(item.point_zyx, dtype=np.float32) for item in nodes], axis=0)
    if selected_crossing is not None and trace.shape[0] > 0:
        trace[-1] = selected_crossing.point_zyx.astype(np.float32, copy=False)
    steps = tuple(item.step for item in nodes[1:] if item.step is not None)
    crossings: tuple[NativeTargetPlaneCrossing, ...] = ()
    selected = selected_crossing
    if target_planes is not None and node.target_plane_crossed is not None and node.target_plane_crossings_zyx is not None:
        crossings, fallback_selected = _target_plane_crossing_summary(
            target_planes,
            np.asarray(node.target_plane_crossed, dtype=bool),
            np.asarray(node.target_plane_crossings_zyx, dtype=np.float32),
        )
        if selected is None:
            selected = fallback_selected
    return NativeTraceResult(
        trace_zyx=trace.astype(np.float32, copy=False),
        reached_target_plane=bool(reached_target_plane),
        reason=reason,
        steps=steps,
        target_plane_crossings=crossings,
        selected_target_plane_name=None if selected is None else selected.name,
        selected_target_plane_crossing_zyx=None if selected is None else selected.point_zyx,
        selected_target_plane_error_voxels=math.inf
        if selected is None
        else float(selected.error_voxels),
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
    for _ in range(min(width, count)):
        masked_score = torch.where(
            available,
            score,
            torch.full_like(score, torch.inf),
        )
        best_index = torch.argmin(masked_score)
        kept.append(best_index.to(dtype=torch.long))
        delta = generation.points_zyx - generation.points_zyx[best_index].view(1, 3)
        far_enough = torch.sum(delta * delta, dim=1) >= distance2
        available = available & far_enough
    if kept:
        kept_t = torch.stack(kept, dim=0).to(dtype=torch.long)
        kept_t = kept_t[torch.isfinite(score[kept_t])]
        if int(kept_t.numel()) > 0:
            return kept_t
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
        if (
            gen.sampled_current_directions_zyx is not None
            and gen.sampled_current_valid is not None
        ):
            sampled_current_direction = (
                gen.sampled_current_directions_zyx[item_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            sampled_current_valid = bool(gen.sampled_current_valid[item_idx].detach().cpu())
        else:
            sampled_current_direction = None
            sampled_current_valid = False
        if (
            gen.target_plane_crossed is not None
            and gen.target_plane_crossings_zyx is not None
        ):
            target_plane_crossed = (
                gen.target_plane_crossed[item_idx].detach().cpu().numpy().astype(bool)
            )
            target_plane_crossings = (
                gen.target_plane_crossings_zyx[item_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        else:
            target_plane_crossed = None
            target_plane_crossings = None
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
            sampled_current_direction_zyx=sampled_current_direction,
            sampled_current_valid=sampled_current_valid,
            target_plane_crossed=target_plane_crossed,
            target_plane_crossings_zyx=target_plane_crossings,
        )
    return node


def _trace_native_3d_one_way_beam(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
    target_plane_normal_zyx: np.ndarray | None = None,
    budget_span_voxels: float | None = None,
    progress_label: str | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
) -> NativeTraceResult:
    start = np.asarray(start_zyx, dtype=np.float32)
    target = np.asarray(target_zyx, dtype=np.float32)
    target_planes = _normalize_target_planes(
        target_zyx=target,
        target_planes_zyx=target_planes_zyx,
        target_plane_normal_zyx=target_plane_normal_zyx,
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
        return 1.0 - float(
            np.linalg.norm(np.asarray(point_zyx, dtype=np.float32) - target)
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
    with _profile_span_for_cache(cache, "trace_start_sample"):
        initial_direction, _presence, valid = _sample_trace_start_direction_aligned(
            cache,
            start,
            cp_reference_direction_zyx=initial_direction_zyx,
        )
    if not valid:
        raise ValueError(f"native 3D Trace2CP start point is invalid: {start.tolist()}")
    initial_crossed = np.zeros((len(target_planes),), dtype=bool)
    initial_crossings = np.zeros((len(target_planes), 3), dtype=np.float32)
    initial_crossed, initial_crossings = _update_target_plane_crossings(
        start_zyx=start,
        end_zyx=start,
        planes=target_planes,
        crossed=initial_crossed,
        crossings_zyx=initial_crossings,
    )
    start_node = _NativeBeamNode(
        point_zyx=start.astype(np.float32),
        previous_direction_zyx=initial_direction.astype(np.float32, copy=False),
        history_direction_zyx=initial_direction.astype(np.float32, copy=False),
        parent=None,
        step=None,
        cumulative_loss=0.0,
        depth=0,
        sampled_current_direction_zyx=initial_direction.astype(np.float32, copy=False),
        sampled_current_valid=True,
        target_plane_crossed=initial_crossed,
        target_plane_crossings_zyx=initial_crossings,
    )
    live: list[_NativeBeamNode] = [start_node]
    best_live = start_node
    committed_step = 0
    device = _cache_device(cache)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    target_plane_points_t = torch.as_tensor(
        np.stack([plane.point_zyx for plane in target_planes], axis=0),
        dtype=torch.float32,
        device=device,
    )
    target_plane_normals_t = torch.as_tensor(
        np.stack([plane.normal_zyx for plane in target_planes], axis=0),
        dtype=torch.float32,
        device=device,
    )
    initial_direction_t = torch.as_tensor(
        initial_direction.reshape(1, 3),
        dtype=torch.float32,
        device=device,
    )
    while committed_step < step_limit:
        with _profile_span_for_cache(cache, "trace_beam_state"):
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
            root_sampled_current = torch.as_tensor(
                np.stack(
                    [
                        (
                            np.asarray(node.sampled_current_direction_zyx, dtype=np.float32)
                            if node.sampled_current_valid
                            and node.sampled_current_direction_zyx is not None
                            else np.zeros((3,), dtype=np.float32)
                        )
                        for node in live
                    ],
                    axis=0,
                ),
                dtype=torch.float32,
                device=device,
            )
            root_sampled_valid = torch.as_tensor(
                [bool(node.sampled_current_valid) for node in live],
                dtype=torch.bool,
                device=device,
            )
            root_target_plane_crossed = torch.as_tensor(
                np.stack(
                    [
                        (
                            np.asarray(node.target_plane_crossed, dtype=bool)
                            if node.target_plane_crossed is not None
                            else np.zeros((len(target_planes),), dtype=bool)
                        )
                        for node in live
                    ],
                    axis=0,
                ),
                dtype=torch.bool,
                device=device,
            )
            root_target_plane_crossings = torch.as_tensor(
                np.stack(
                    [
                        (
                            np.asarray(node.target_plane_crossings_zyx, dtype=np.float32)
                            if node.target_plane_crossings_zyx is not None
                            else np.zeros((len(target_planes), 3), dtype=np.float32)
                        )
                        for node in live
                    ],
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
                    sampled_current_directions_zyx=F.normalize(
                        root_sampled_current,
                        p=2.0,
                        dim=1,
                        eps=float(_EPS),
                    ),
                    sampled_current_valid=root_sampled_valid,
                    target_plane_crossed=root_target_plane_crossed,
                    target_plane_crossings_zyx=root_target_plane_crossings,
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
            with _profile_span_for_cache(cache, "trace_current_sample"):
                if (
                    frontier_gen.sampled_current_directions_zyx is not None
                    and frontier_gen.sampled_current_valid is not None
                ):
                    current_directions = F.normalize(
                        frontier_gen.sampled_current_directions_zyx.to(
                            device=device,
                            dtype=torch.float32,
                        ),
                        p=2.0,
                        dim=1,
                        eps=float(_EPS),
                    )
                    state_valid = frontier_gen.sampled_current_valid.to(
                        device=device,
                        dtype=torch.bool,
                    )
                else:
                    current_directions, _current_presence, state_valid = (
                        _sample_trace_points_aligned_torch(
                            cache,
                            current_points,
                            reference_directions_zyx=previous_directions,
                        )
                    )
                    root_mask = frontier_gen.depth == 0
                    current_directions = torch.where(
                        root_mask[:, None],
                        initial_direction_t.expand(int(current_directions.shape[0]), 3),
                        current_directions,
                    )
                    state_valid = torch.where(
                        root_mask,
                        torch.ones_like(state_valid),
                        state_valid,
                    )
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
            with _profile_span_for_cache(cache, "trace_candidate_grid"):
                candidate_dirs = _trace_candidate_directions_torch(current_directions_v, cfg)
                next_points = (
                    current_points_v[:, None, :] + candidate_dirs * float(cfg.step_voxels)
                )
            with _profile_span_for_cache(cache, "trace_candidate_normals"):
                sampled_normals = _sample_candidate_normals_torch(
                    normal_sampler,
                    next_points,
                    device=device,
                )
            candidate_normals_t = None if sampled_normals is None else sampled_normals[0]
            candidate_normals_valid_t = None if sampled_normals is None else sampled_normals[1]
            with _profile_span_for_cache(cache, "trace_candidate_score"):
                (
                    total_loss_t,
                    direction_loss_t,
                    presence_loss_t,
                    smoothness_loss_t,
                    candidate_valid_t,
                    rejected_per_state_t,
                    next_current_direction_t,
                    _next_current_presence_t,
                    next_current_valid_t,
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
                    return_next_selected=True,
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
            if hasattr(cache, "sample_point_choices_torch"):
                child_sampled_current = F.normalize(
                    next_current_direction_t[child_local_state, child_candidate],
                    p=2.0,
                    dim=1,
                    eps=float(_EPS),
                )
                child_sampled_valid = next_current_valid_t[child_local_state, child_candidate]
            else:
                child_sampled_current = None
                child_sampled_valid = None
            if (
                frontier_gen.target_plane_crossed is None
                or frontier_gen.target_plane_crossings_zyx is None
            ):
                raise RuntimeError("beam generation is missing target-plane crossing state")
            parent_crossed = frontier_gen.target_plane_crossed[child_parent_indices]
            parent_crossings = frontier_gen.target_plane_crossings_zyx[child_parent_indices]
            current_child_points = current_points_v[child_local_state]
            d0 = torch.sum(
                (current_child_points[:, None, :] - target_plane_points_t[None, :, :])
                * target_plane_normals_t[None, :, :],
                dim=2,
            )
            d1 = torch.sum(
                (child_next_points[:, None, :] - target_plane_points_t[None, :, :])
                * target_plane_normals_t[None, :, :],
                dim=2,
            )
            crossed_now = (~parent_crossed) & ((d0 == 0.0) | (d0 * d1 <= 0.0))
            child_crossed = parent_crossed | crossed_now
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
                current_child_points[:, None, :] * (1.0 - crossing_t[:, :, None])
                + child_next_points[:, None, :] * crossing_t[:, :, None]
            )
            child_crossings = torch.where(
                crossed_now[:, :, None],
                crossing_points,
                parent_crossings,
            )
            reached_mask = torch.all(child_crossed, dim=1)
            deltas = child_crossings - target_plane_points_t[None, :, :]
            normal_projection = torch.sum(
                deltas * target_plane_normals_t[None, :, :],
                dim=2,
            )
            in_plane = (
                deltas
                - normal_projection[:, :, None] * target_plane_normals_t[None, :, :]
            )
            crossing_errors = torch.linalg.norm(in_plane, dim=2)
            crossing_errors = torch.where(
                child_crossed,
                crossing_errors,
                torch.full_like(crossing_errors, torch.inf),
            )
            selected_plane_index = torch.argmin(crossing_errors, dim=1)
            selected_points = child_crossings[
                torch.arange(int(child_crossings.shape[0]), device=device),
                selected_plane_index,
            ]
            child_points = torch.where(
                reached_mask[:, None],
                selected_points,
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
                sampled_current_directions_zyx=child_sampled_current,
                sampled_current_valid=child_sampled_valid,
                target_plane_crossed=child_crossed,
                target_plane_crossings_zyx=child_crossings,
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
            crossings, selected = _target_plane_crossing_summary(
                target_planes,
                np.asarray(best.target_plane_crossed, dtype=bool),
                np.asarray(best.target_plane_crossings_zyx, dtype=np.float32),
            )
            if selected is None:
                raise RuntimeError("reached target planes but no selected crossing was found")
            emit_progress(
                selected.point_zyx,
                min(step_limit, committed_step + expanded_steps),
                reason="target_plane",
                active_beams=len(live),
            )
            return _beam_node_result(
                best,
                reached_target_plane=True,
                reason="target_plane",
                target_planes=target_planes,
                selected_crossing=selected,
            )
        if expanded_steps == 0 or frontier_generation_index == 0:
            emit_progress(
                best_live.point_zyx,
                committed_step,
                reason="all_candidates_invalid",
                active_beams=0,
            )
            best_crossed = (
                np.asarray(best_live.target_plane_crossed, dtype=bool)
                if best_live.target_plane_crossed is not None
                else np.zeros((len(target_planes),), dtype=bool)
            )
            return _beam_node_result(
                best_live,
                reached_target_plane=False,
                reason=_missing_target_plane_reason(
                    "all_candidates_invalid",
                    target_planes,
                    best_crossed,
                ),
                target_planes=target_planes,
            )
        frontier = generations[frontier_generation_index]
        with _profile_span_for_cache(cache, "trace_beam_prune"):
            kept_indices = _prune_native_beam_tensor_indices(
                frontier,
                beam_width=beam_width,
                prune_distance_voxels=float(cfg.beam_prune_distance_voxels),
            )
        with _profile_span_for_cache(cache, "trace_beam_rebuild"):
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
            best_crossed = (
                np.asarray(best_live.target_plane_crossed, dtype=bool)
                if best_live.target_plane_crossed is not None
                else np.zeros((len(target_planes),), dtype=bool)
            )
            return _beam_node_result(
                best_live,
                reached_target_plane=False,
                reason=_missing_target_plane_reason(
                    "all_candidates_invalid",
                    target_planes,
                    best_crossed,
                ),
                target_planes=target_planes,
            )
        committed_step += max(1, expanded_steps)
        best_live = min(live, key=lambda item: (float(item.cumulative_loss), int(item.depth)))
        progress_node = max(live, key=lambda item: node_progress(item.point_zyx))
        emit_progress(progress_node.point_zyx, committed_step, active_beams=len(live))
    emit_progress(best_live.point_zyx, progress_max, reason=limit_reason, active_beams=len(live))
    best_crossed = (
        np.asarray(best_live.target_plane_crossed, dtype=bool)
        if best_live.target_plane_crossed is not None
        else np.zeros((len(target_planes),), dtype=bool)
    )
    return _beam_node_result(
        best_live,
        reached_target_plane=False,
        reason=_missing_target_plane_reason(limit_reason, target_planes, best_crossed),
        target_planes=target_planes,
    )


def trace_native_3d_one_way(
    cache: NativeTraceFieldCache,
    *,
    start_zyx: np.ndarray,
    target_zyx: np.ndarray,
    initial_direction_zyx: np.ndarray,
    cfg: NativeTrace2CpConfig,
    target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
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
            target_planes_zyx=target_planes_zyx,
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
        target_planes_zyx=target_planes_zyx,
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
    forward_target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
    reverse_target_planes_zyx: Sequence[NativeTargetPlane] | None = None,
    progress: bool = False,
    normal_sampler: NativeTraceNormalSampler | None = None,
    profiler: _NativeTraceProfiler | None = None,
) -> NativeTracePairResult:
    measure = profiler.measure if profiler is not None else None
    forward_planes = (
        tuple(forward_target_planes_zyx)
        if forward_target_planes_zyx is not None
        else (
            NativeTargetPlane(
                name="initial_direction",
                point_zyx=np.asarray(target_zyx, dtype=np.float32),
                normal_zyx=_require_unit(
                    forward_initial_direction_zyx,
                    label="forward_initial_direction_zyx target plane",
                ),
            ),
        )
    )
    reverse_planes = (
        tuple(reverse_target_planes_zyx)
        if reverse_target_planes_zyx is not None
        else (
            NativeTargetPlane(
                name="initial_direction",
                point_zyx=np.asarray(start_zyx, dtype=np.float32),
                normal_zyx=_require_unit(
                    reverse_initial_direction_zyx,
                    label="reverse_initial_direction_zyx target plane",
                ),
            ),
        )
    )
    with (
        measure("trace_forward") if measure is not None else _NullNativeTraceProfileSpan()
    ):
        forward = trace_native_3d_one_way(
            cache,
            start_zyx=start_zyx,
            target_zyx=target_zyx,
            initial_direction_zyx=forward_initial_direction_zyx,
            cfg=cfg,
            target_planes_zyx=forward_planes,
            progress_label="fw" if progress else None,
            normal_sampler=normal_sampler,
        )
    with (
        measure("trace_reverse") if measure is not None else _NullNativeTraceProfileSpan()
    ):
        reverse = trace_native_3d_one_way(
            cache,
            start_zyx=target_zyx,
            target_zyx=start_zyx,
            initial_direction_zyx=reverse_initial_direction_zyx,
            cfg=cfg,
            target_planes_zyx=reverse_planes,
            progress_label="bw" if progress else None,
            normal_sampler=normal_sampler,
        )
    with (
        measure("trace_fusion") if measure is not None else _NullNativeTraceProfileSpan()
    ):
        fusion = fuse_forward_reverse_traces(
            forward.trace_zyx,
            reverse.trace_zyx,
            start_zyx=start_zyx,
            target_zyx=target_zyx,
            step_voxels=float(cfg.step_voxels),
        )
    span = float(np.linalg.norm(np.asarray(target_zyx, dtype=np.float32) - np.asarray(start_zyx, dtype=np.float32)))
    forward_plane = float(forward.selected_target_plane_error_voxels)
    reverse_plane = float(reverse.selected_target_plane_error_voxels)
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


def _vc3d_voxel_size_m(record: Any) -> float | None:
    sampler = getattr(record, "sampler", None)
    volume = getattr(sampler, "volume", None)
    metadata = getattr(volume, "metadata", None)
    if metadata is None:
        return None
    try:
        voxelsize_um = metadata["voxelsize"]
    except (KeyError, TypeError, AttributeError):
        return None
    try:
        voxelsize_um = float(voxelsize_um)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(voxelsize_um) or voxelsize_um <= 0.0:
        return None
    return voxelsize_um * 1.0e-6


def _reference_line_length_meters_between_cps(
    record: Any,
    *,
    start_cp_index: int,
    end_cp_index: int,
) -> float | None:
    voxel_size_m = _vc3d_voxel_size_m(record)
    if voxel_size_m is None:
        return None
    line_xyz = np.asarray(record.fiber.line_points_xyz, dtype=np.float64)
    if line_xyz.ndim != 2 or line_xyz.shape[1] != 3:
        return None
    start_line = control_point_line_index(record.fiber, int(start_cp_index))
    end_line = control_point_line_index(record.fiber, int(end_cp_index))
    lo = min(int(start_line), int(end_line))
    hi = max(int(start_line), int(end_line))
    if hi <= lo:
        return 0.0
    diffs = np.diff(line_xyz[lo : hi + 1], axis=0)
    length = float(np.linalg.norm(diffs, axis=1).sum()) * float(voxel_size_m)
    return length if math.isfinite(length) and length >= 0.0 else None


def _restarts_per_kvx(restart_count: int, reference_length_voxels: float) -> float:
    length = float(reference_length_voxels)
    if not math.isfinite(length) or length <= _EPS:
        return math.inf if int(restart_count) > 0 else 0.0
    return float(int(restart_count)) * 1000.0 / length


def _restarts_per_meter(restart_count: int, reference_length_meters: float | None) -> float | None:
    if reference_length_meters is None:
        return None
    length = float(reference_length_meters)
    if not math.isfinite(length) or length <= _EPS:
        return math.inf if int(restart_count) > 0 else 0.0
    return float(int(restart_count)) / length


def _mean_trace_run_length_meters(
    restart_count: int,
    reference_length_meters: float | None,
) -> float | None:
    return _mean_trace_run_length_meters_for_runs(
        max(1, int(restart_count) + 1),
        reference_length_meters,
    )


def _mean_trace_run_length_meters_for_runs(
    run_count: int,
    reference_length_meters: float | None,
) -> float | None:
    if reference_length_meters is None:
        return None
    length = float(reference_length_meters)
    if not math.isfinite(length) or length < 0.0:
        return None
    return length / float(max(1, int(run_count)))


def _format_trace2cp_kvx_rate(restarts_per_kvx: float) -> str:
    return f"{float(restarts_per_kvx):.1f}"


def _format_trace2cp_meter_rate(
    restarts_per_meter: float | None,
    *,
    restart_count: int,
    reference_length_meters: float | None,
) -> str:
    return _format_trace2cp_meter_rate_for_runs(
        restarts_per_meter,
        run_count=max(1, int(restart_count) + 1),
        reference_length_meters=reference_length_meters,
    )


def _format_trace2cp_meter_rate_for_runs(
    restarts_per_meter: float | None,
    *,
    run_count: int,
    reference_length_meters: float | None,
) -> str:
    if restarts_per_meter is None or reference_length_meters is None:
        return ""
    mean_run_meters = _mean_trace_run_length_meters_for_runs(
        int(run_count),
        float(reference_length_meters),
    )
    if mean_run_meters is None:
        return f"err/m={float(restarts_per_meter):.1f}"
    return f"err/m={float(restarts_per_meter):.1f} ({mean_run_meters * 1000.0:.1f}mm)"


def trace_native_3d_whole_fiber(
    cache: NativeTraceFieldCache,
    *,
    record: Any,
    cfg: NativeTrace2CpConfig,
    error_threshold_voxels: float,
    start_cp_index: int = 0,
    progress: bool = False,
    segment_callback: Callable[[NativeWholeFiberSegmentResult, NativeWholeFiberResult | None], None] | None = None,
    trace_segment_fn: Callable[..., NativeTraceResult] | None = None,
    normal_sampler: NativeTraceNormalSampler | None = None,
    profiler: _NativeTraceProfiler | None = None,
) -> NativeWholeFiberResult:
    cp_points = _record_control_points_selected_zyx(record)
    cp_count = int(cp_points.shape[0])
    if cp_count < 2:
        raise ValueError("native whole-fiber Trace2CP requires at least two control points")
    threshold = float(error_threshold_voxels)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("whole-fiber error threshold must be finite and >= 0")
    first_cp = int(start_cp_index)
    if first_cp < 0 or first_cp >= cp_count - 1:
        raise ValueError(
            "native whole-fiber start CP index must leave at least one target segment: "
            f"start_cp_index={first_cp} control_points={cp_count}"
        )
    final_cp = cp_count - 1
    arc_by_cp = _control_point_reference_arc_voxels(record)
    reference_origin_arc = float(arc_by_cp[first_cp])
    total_reference_length_voxels = float(
        abs(float(arc_by_cp[final_cp]) - reference_origin_arc)
    )
    total_reference_length_meters = _reference_line_length_meters_between_cps(
        record,
        start_cp_index=first_cp,
        end_cp_index=final_cp,
    )
    segment_count = final_cp - first_cp
    tracer = trace_native_3d_one_way if trace_segment_fn is None else trace_segment_fn
    run_start = time.perf_counter()
    segments: list[NativeWholeFiberSegmentResult] = []
    stitched_parts: list[np.ndarray] = []
    restart_count = 0
    last_persisted_restart_count = 0
    last_success_cp_index = first_cp
    current_point = cp_points[first_cp].astype(np.float32)
    current_direction = _fiber_line_tangent_zyx_toward_target(
        record,
        start_control_point_index=first_cp,
        target_control_point_index=first_cp + 1,
    )

    def emit_progress(completed_segments: int, segment: NativeWholeFiberSegmentResult | None = None) -> None:
        nonlocal last_persisted_restart_count
        if not progress:
            return
        done = int(completed_segments)
        absolute_done_cp = min(first_cp + done, final_cp)
        elapsed = max(0.0, time.perf_counter() - run_start)
        frac = float(done) / float(max(1, segment_count))
        eta = None if frac <= 1.0e-6 else elapsed * (1.0 - frac) / frac
        status = "pending" if segment is None else ("ok" if segment.success else f"restart:{segment.reason}")
        reference_length = (
            float(abs(float(arc_by_cp[absolute_done_cp]) - reference_origin_arc))
            if done > 0
            else 0.0
        )
        restarts_per_kvx = _restarts_per_kvx(restart_count, reference_length)
        reference_length_meters = (
            _reference_line_length_meters_between_cps(
                record,
                start_cp_index=first_cp,
                end_cp_index=absolute_done_cp,
            )
            if done > 0
            else None
        )
        restarts_per_meter = _restarts_per_meter(restart_count, reference_length_meters)
        persist_restart_line = restart_count > last_persisted_restart_count
        physical_detail = ""
        if restarts_per_meter is not None and reference_length_meters is not None:
            physical_detail = (
                _format_trace2cp_meter_rate(
                    restarts_per_meter,
                    restart_count=restart_count,
                    reference_length_meters=reference_length_meters,
                )
                + " "
            )
        _emit_native_progress(
            "whole fiber",
            done,
            segment_count,
            run_start,
            detail=(
                f"segment={min(done + 1, segment_count)}/{segment_count} "
                f"cp={first_cp}->{absolute_done_cp}/{final_cp} "
                f"status={status} restarts={restart_count} "
                f"err/kvx={_format_trace2cp_kvx_rate(restarts_per_kvx)} "
                f"{physical_detail}"
                f"eta={_format_eta(eta)} blocks={len(cache._blocks)}"
            ),
            persist_line=persist_restart_line,
        )
        if persist_restart_line:
            last_persisted_restart_count = int(restart_count)

    def inferred_block_count() -> int:
        return int(getattr(cache, "total_inferred_blocks", len(cache._blocks)))

    emit_progress(0)
    for segment_offset in range(segment_count):
        start_cp = int(first_cp + segment_offset)
        target_cp = int(start_cp + 1)
        previous_segment_success = bool(segments and segments[-1].success)
        target_point = cp_points[target_cp].astype(np.float32)
        reference_start = cp_points[start_cp].astype(np.float32)
        segment_span = float(np.linalg.norm(target_point - reference_start))
        if segment_span <= _EPS:
            raise ValueError(f"degenerate native whole-fiber CP span {start_cp}->{target_cp}")
        target_reference_direction = _fiber_line_tangent_zyx_toward_target(
            record,
            start_control_point_index=target_cp,
            target_control_point_index=start_cp,
        )
        target_planes = _fiber_target_planes_zyx(
            cache,
            record,
            target_control_point_index=target_cp,
            target_zyx=target_point,
            inference_reference_direction_zyx=target_reference_direction,
            include_inference_plane=trace_segment_fn is None,
        )
        with (
            profiler.measure("trace_segment")
            if profiler is not None
            else _NullNativeTraceProfileSpan()
        ):
            result = tracer(
                cache,
                start_zyx=current_point.astype(np.float32),
                target_zyx=target_point,
                initial_direction_zyx=current_direction,
                cfg=cfg,
                target_planes_zyx=target_planes,
                budget_span_voxels=segment_span,
                progress_label=None,
                normal_sampler=normal_sampler,
            )
        crossing = (
            np.asarray(result.selected_target_plane_crossing_zyx, dtype=np.float32)
            if bool(result.reached_target_plane)
            and result.selected_target_plane_crossing_zyx is not None
            else result.trace_zyx[-1].astype(np.float32)
        )
        if bool(result.reached_target_plane):
            if math.isfinite(float(result.selected_target_plane_error_voxels)):
                in_plane_error = float(result.selected_target_plane_error_voxels)
                selected_plane_name = result.selected_target_plane_name
                selected_crossing = result.selected_target_plane_crossing_zyx
            else:
                crossing_candidates = []
                for plane in target_planes:
                    error = _target_plane_in_plane_error_voxels(
                        crossing,
                        target_zyx=plane.point_zyx,
                        plane_normal_zyx=plane.normal_zyx,
                    )
                    crossing_candidates.append(
                        NativeTargetPlaneCrossing(
                            name=plane.name,
                            point_zyx=crossing.astype(np.float32, copy=False),
                            error_voxels=float(error),
                        )
                    )
                selected = min(crossing_candidates, key=lambda item: item.error_voxels)
                in_plane_error = float(selected.error_voxels)
                selected_plane_name = selected.name
                selected_crossing = selected.point_zyx
        else:
            in_plane_error = float("inf")
            selected_plane_name = result.selected_target_plane_name
            selected_crossing = result.selected_target_plane_crossing_zyx
        success = bool(result.reached_target_plane) and in_plane_error <= threshold
        if success:
            reason = result.reason
            restart = False
            reference_arc = float(abs(float(arc_by_cp[target_cp]) - reference_origin_arc))
            current_point = crossing
            current_direction = _terminal_trace_direction(result.trace_zyx, fallback=current_direction)
            last_success_cp_index = target_cp
        else:
            reason = result.reason if not bool(result.reached_target_plane) else "in_plane_error"
            restart = True
            restart_count += 1
            reference_arc = float(
                abs(float(arc_by_cp[last_success_cp_index]) - reference_origin_arc)
            )
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
            selected_target_plane_name=selected_plane_name,
            selected_target_plane_crossing_zyx=None
            if selected_crossing is None
            else np.asarray(selected_crossing, dtype=np.float32),
            reference_arc_distance_voxels=float(reference_arc),
            step_count=int(len(result.steps)),
        )
        segments.append(segment)
        partial_reference_length_voxels = float(
            abs(float(arc_by_cp[target_cp]) - reference_origin_arc)
        )
        partial_reference_length_meters = _reference_line_length_meters_between_cps(
            record,
            start_cp_index=first_cp,
            end_cp_index=target_cp,
        )
        partial = NativeWholeFiberResult(
            segments=tuple(segments),
            restart_count=int(restart_count),
            segment_count=int(segment_count),
            restarts_per_kvx=_restarts_per_kvx(restart_count, partial_reference_length_voxels),
            reference_length_voxels=partial_reference_length_voxels,
            reference_length_meters=partial_reference_length_meters,
            restarts_per_meter=_restarts_per_meter(restart_count, partial_reference_length_meters),
            stitched_trace_zyx=(
                np.concatenate([part for part in stitched_parts if part.size], axis=0).astype(np.float32)
                if any(part.size for part in stitched_parts)
                else np.zeros((0, 3), dtype=np.float32)
            ),
            inferred_blocks=inferred_block_count(),
        )
        if segment_callback is not None:
            with (
                profiler.measure("render_segment_callback")
                if profiler is not None
                else _NullNativeTraceProfileSpan()
            ):
                segment_callback(segment, partial)
        emit_progress(segment_offset + 1, segment)
    stitched = (
        np.concatenate([part for part in stitched_parts if part.size], axis=0).astype(np.float32)
        if any(part.size for part in stitched_parts)
        else np.zeros((0, 3), dtype=np.float32)
    )
    return NativeWholeFiberResult(
        segments=tuple(segments),
        restart_count=int(restart_count),
        segment_count=int(segment_count),
        restarts_per_kvx=_restarts_per_kvx(restart_count, total_reference_length_voxels),
        reference_length_voxels=float(total_reference_length_voxels),
        reference_length_meters=total_reference_length_meters,
        restarts_per_meter=_restarts_per_meter(restart_count, total_reference_length_meters),
        stitched_trace_zyx=stitched,
        inferred_blocks=inferred_block_count(),
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


def _project_points_to_source_strip_preserve_slots(
    source: _Trace2CpSegmentSource,
    points_xyz_base: np.ndarray,
    *,
    axis_name: str,
) -> np.ndarray:
    points = np.asarray(points_xyz_base, dtype=np.float32)
    if int(points.shape[0]) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    projected_xyz, projected_xy, projected_valid = _closest_source_line_projection(points, source)
    axes, axes_valid = _sample_source_axes_at_xy(source, axis_name, projected_xy)
    spacing = float(source.record.volume_spacing_base)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid volume spacing for native point projection: {spacing}")
    offsets = np.sum((points - projected_xyz) * axes, axis=1) / np.float32(spacing)
    xy = projected_xy.copy()
    xy[:, 1] += offsets.astype(np.float32, copy=False)
    valid = projected_valid & axes_valid & np.isfinite(xy).all(axis=1)
    xy[~valid] = np.nan
    return xy.astype(np.float32, copy=False)


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
    control_points_xy: np.ndarray | None = None,
    control_point_labels: Sequence[str] | None = None,
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
    if control_points_xy is not None:
        cp_points = np.asarray(control_points_xy, dtype=np.float32)
        if cp_points.ndim == 2 and cp_points.shape[1] == 2:
            labels = list(control_point_labels or ())
            bottom_labels: list[tuple[float, str]] = []
            for cp_index, (x_f, y_f) in enumerate(cp_points):
                if not (np.isfinite(x_f) and np.isfinite(y_f)):
                    continue
                x, y = float(x_f), float(y_f) + text_pad
                draw.ellipse(
                    (x - 2.5, y - 2.5, x + 2.5, y + 2.5),
                    fill=(255, 255, 0, 210),
                    outline=(0, 0, 0, 180),
                    width=1,
                )
                if cp_index < len(labels):
                    label = str(labels[cp_index])
                    if label:
                        bottom_labels.append((x, label))
            for x, label in bottom_labels:
                try:
                    bbox = draw.textbbox((0, 0), label)
                    text_w = float(bbox[2] - bbox[0])
                    text_h = float(bbox[3] - bbox[1])
                except AttributeError:
                    text_w = float(6 * len(label))
                    text_h = 10.0
                text_x = min(
                    max(2.0, float(x) - 0.5 * text_w),
                    max(2.0, float(padded.width) - text_w - 2.0),
                )
                text_y = float(text_pad + canvas.height) - text_h - 3.0
                draw.rectangle(
                    (
                        text_x - 1.0,
                        text_y - 1.0,
                        text_x + text_w + 1.0,
                        text_y + text_h + 1.0,
                    ),
                    fill=(0, 0, 0, 150),
                )
                draw.text((text_x, text_y), label, fill=(255, 255, 255, 240))
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
        fused_trace_xyz = run_stage(
            "fused-source-trace",
            lambda: _trace_zyx_to_base_xyz(result.fused_zyx, spacing),
        )
        fused_source = run_stage(
            "fused-source",
            lambda: geometry_loader.build_trace2cp_volume_trace_segment_source(
                source,
                fused_trace_xyz,
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


def _whole_fiber_span_control_points_xyz_base(
    source: _Trace2CpSegmentSource,
    span: _NativeWholeFiberVisualSpan,
) -> np.ndarray:
    start = int(span.start_cp_index)
    end = int(span.end_cp_index)
    step = 1 if end >= start else -1
    indices = np.arange(start, end + step, step, dtype=np.int64)
    cps_zyx = np.asarray(source.record.fiber.control_points_zyx, dtype=np.float32)
    valid = (indices >= 0) & (indices < int(cps_zyx.shape[0]))
    indices = indices[valid]
    if int(indices.size) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return cps_zyx[indices][:, [2, 1, 0]].astype(np.float32, copy=False)


def _whole_fiber_span_control_points_for_view(
    source: _Trace2CpSegmentSource,
    span: _NativeWholeFiberVisualSpan,
    *,
    axis_name: str,
) -> np.ndarray:
    cps_xyz = _whole_fiber_span_control_points_xyz_base(source, span)
    if int(cps_xyz.shape[0]) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return _project_points_to_source_strip_preserve_slots(source, cps_xyz, axis_name=axis_name)


def _whole_fiber_span_control_point_labels(
    span: _NativeWholeFiberVisualSpan,
) -> tuple[str, ...]:
    start = int(span.start_cp_index)
    end = int(span.end_cp_index)
    step = 1 if end >= start else -1
    cp_indices = np.arange(start, end + step, step, dtype=np.int64)
    labels_by_cp: dict[int, str] = {start: f"cp={start} d=0.0"}
    for segment in span.segments:
        cp_index = int(segment.target_cp_index)
        if not bool(segment.reached_target_plane):
            labels_by_cp[cp_index] = f"cp={cp_index} miss"
            continue
        distance = float(segment.in_plane_error_voxels)
        if not math.isfinite(distance):
            labels_by_cp[cp_index] = f"cp={cp_index} d=inf"
        else:
            labels_by_cp[cp_index] = f"cp={cp_index} d={distance:.1f}"
    return tuple(labels_by_cp.get(int(cp_index), "") for cp_index in cp_indices)


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


def _native_whole_fiber_segment_trace_length_voxels(
    segment: NativeWholeFiberSegmentResult,
) -> float:
    trace = np.asarray(segment.trace_zyx, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[0] < 2:
        return 0.0
    length = float(np.linalg.norm(np.diff(trace, axis=0), axis=1).sum())
    return length if math.isfinite(length) and length >= 0.0 else 0.0


def _native_whole_fiber_visual_span_estimated_width_px(
    span: _NativeWholeFiberVisualSpan,
    *,
    trace2cp_rf_margin_px: float,
) -> float:
    trace_length = sum(
        _native_whole_fiber_segment_trace_length_voxels(segment)
        for segment in span.segments
    )
    return float(trace_length) + 2.0 * max(0.0, float(trace2cp_rf_margin_px))


def _split_native_whole_fiber_visual_span(
    span: _NativeWholeFiberVisualSpan,
    *,
    max_width_px: int,
    trace2cp_rf_margin_px: float,
) -> tuple[_NativeWholeFiberVisualSpan, ...]:
    limit = max(1.0, float(max_width_px))
    chunks: list[_NativeWholeFiberVisualSpan] = []
    active: list[NativeWholeFiberSegmentResult] = []
    active_width = 2.0 * max(0.0, float(trace2cp_rf_margin_px))
    active_start = int(span.start_cp_index)
    for segment in span.segments:
        segment_width = _native_whole_fiber_segment_trace_length_voxels(segment)
        if active and active_width + segment_width > limit:
            chunks.append(
                _NativeWholeFiberVisualSpan(
                    start_cp_index=active_start,
                    end_cp_index=int(active[-1].target_cp_index),
                    segments=tuple(active),
                    restart_after=False,
                )
            )
            active = []
            active_start = int(segment.start_cp_index)
            active_width = 2.0 * max(0.0, float(trace2cp_rf_margin_px))
        active.append(segment)
        active_width += segment_width
    if active:
        chunks.append(
            _NativeWholeFiberVisualSpan(
                start_cp_index=active_start,
                end_cp_index=int(active[-1].target_cp_index),
                segments=tuple(active),
                restart_after=bool(span.restart_after),
            )
        )
    return tuple(chunks) if chunks else (span,)


def _whole_fiber_panel_block_width(block: tuple[Any, ...]) -> int:
    return max((int(panel.width) for panel in block), default=0)


def _split_wide_whole_fiber_panel_block(
    block: tuple[Any, ...],
    *,
    max_width_px: int,
) -> tuple[tuple[Any, ...], ...]:
    from PIL import Image

    width = _whole_fiber_panel_block_width(block)
    limit = max(1, int(max_width_px))
    if width <= limit:
        return (block,)
    parts: list[tuple[Any, ...]] = []
    for left in range(0, width, limit):
        right = min(width, left + limit)
        chunk_width = int(right - left)
        cropped: list[Any] = []
        for panel in block:
            image = panel.convert("RGBA") if not isinstance(panel, Image.Image) else panel
            canvas = Image.new("RGBA", (chunk_width, int(image.height)), (0, 0, 0, 255))
            if left < int(image.width):
                crop = image.crop((left, 0, min(right, int(image.width)), int(image.height)))
                canvas.alpha_composite(crop, (0, 0))
            cropped.append(canvas)
        parts.append(tuple(cropped))
    return tuple(parts)


def _split_whole_fiber_panel_blocks_for_pages(
    panel_blocks: Sequence[tuple[Any, ...]],
    *,
    split_target_px: int = _NATIVE_WHOLE_FIBER_VIS_SPLIT_TARGET_PX,
    single_block_max_px: int = _NATIVE_WHOLE_FIBER_VIS_JPEG_SAFE_PX,
) -> tuple[tuple[tuple[Any, ...], ...], ...]:
    pages: list[list[tuple[Any, ...]]] = []
    current: list[tuple[Any, ...]] = []
    current_width = 0
    separator_width = 12
    target = max(1, int(split_target_px))
    for block in panel_blocks:
        for chunk in _split_wide_whole_fiber_panel_block(
            tuple(block),
            max_width_px=max(1, int(single_block_max_px)),
        ):
            chunk_width = _whole_fiber_panel_block_width(chunk)
            projected_width = (
                current_width + separator_width + chunk_width
                if current
                else chunk_width
            )
            if current and projected_width > target:
                pages.append(current)
                current = []
                current_width = 0
            current.append(chunk)
            current_width = (
                current_width + separator_width + chunk_width
                if current_width > 0
                else chunk_width
            )
    if current:
        pages.append(current)
    return tuple(tuple(page) for page in pages)


def _compose_whole_fiber_panel_blocks(
    panel_blocks: list[tuple[Any, ...]],
    *,
    prefix_sheet: Any | None = None,
    status_text: str | None = None,
):
    from PIL import Image, ImageDraw

    if not panel_blocks:
        if prefix_sheet is not None:
            return prefix_sheet.copy()
        sheet = Image.new("RGBA", (760, 96), (0, 0, 0, 255))
        draw = ImageDraw.Draw(sheet, "RGBA")
        draw.text((8, 8), "native 3D whole-fiber Trace2CP", fill=(255, 255, 255, 255))
        draw.text((8, 34), "waiting for first segment", fill=(180, 180, 180, 255))
        if status_text:
            draw.text((8, 60), status_text, fill=(120, 220, 255, 255))
        return sheet
    row_count = max(int(len(block)) for block in panel_blocks)
    row_heights = [
        max(int(block[row].height) for block in panel_blocks if row < int(len(block)))
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
        for row in range(row_count):
            if row < int(len(block)):
                sheet.alpha_composite(block[row], (x, y))
            y += row_heights[row]
        x += int(block_width)
        if block_index + 1 < len(panel_blocks):
            x += separator_width
    if prefix_sheet is not None:
        prefix = prefix_sheet.convert("RGBA")
        combined = Image.new(
            "RGBA",
            (
                int(prefix.width) + separator_width + int(sheet.width),
                max(int(prefix.height), int(sheet.height)),
            ),
            (0, 0, 0, 255),
        )
        combined.alpha_composite(prefix, (0, 0))
        combined.alpha_composite(sheet, (int(prefix.width) + separator_width, 0))
        return combined
    return sheet


def _whole_fiber_split_page_path(image_path: Path, page_index: int) -> Path:
    return image_path.with_name(
        f"{image_path.stem}_{int(page_index):03d}{image_path.suffix}"
    )


def _save_whole_fiber_panel_pages(
    panel_blocks: Sequence[tuple[Any, ...]],
    *,
    image_path: Path,
    status_text: str | None = None,
    quality: int = 90,
    split_target_px: int = _NATIVE_WHOLE_FIBER_VIS_SPLIT_TARGET_PX,
    single_block_max_px: int = _NATIVE_WHOLE_FIBER_VIS_JPEG_SAFE_PX,
) -> list[Path]:
    pages = _split_whole_fiber_panel_blocks_for_pages(
        panel_blocks,
        split_target_px=int(split_target_px),
        single_block_max_px=int(single_block_max_px),
    )
    if not pages:
        sheet = _compose_whole_fiber_panel_blocks([], status_text=status_text)
        sheet.convert("RGB").save(image_path, quality=int(quality))
        return [image_path]
    written: list[Path] = []
    if len(pages) == 1:
        sheet = _compose_whole_fiber_panel_blocks(list(pages[0]), status_text=status_text)
        sheet.convert("RGB").save(image_path, quality=int(quality))
        return [image_path]
    for page_index, page_blocks in enumerate(pages):
        path = image_path if page_index == 0 else _whole_fiber_split_page_path(image_path, page_index)
        sheet = _compose_whole_fiber_panel_blocks(list(page_blocks), status_text=status_text)
        sheet.convert("RGB").save(path, quality=int(quality))
        written.append(path)
    return written


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


def _trim_failed_volume_trace_before_target(
    source: _Trace2CpSegmentSource,
    trace_xyz: np.ndarray,
    *,
    start_xy: np.ndarray,
    target_xy: np.ndarray,
    margin_px: float = 8.0,
) -> np.ndarray:
    trace = np.asarray(trace_xyz, dtype=np.float32)
    if trace.ndim != 2 or trace.shape[1] != 3 or trace.shape[0] <= 1:
        return trace.astype(np.float32, copy=True)
    try:
        _projected_xyz, projected_xy, projected_valid = _closest_source_line_projection(
            trace,
            source,
        )
    except (AttributeError, ValueError):
        finite = trace[np.isfinite(trace).all(axis=1)]
        return finite.astype(np.float32, copy=False)
    start_x = float(np.asarray(start_xy, dtype=np.float32)[0])
    target_x = float(np.asarray(target_xy, dtype=np.float32)[0])
    sign = 1.0 if target_x >= start_x else -1.0
    cutoff = target_x - sign * max(0.0, float(margin_px))
    if sign >= 0.0:
        keep = projected_xy[:, 0] <= cutoff
    else:
        keep = projected_xy[:, 0] >= cutoff
    keep &= projected_valid & np.isfinite(projected_xy[:, 0]) & np.isfinite(trace).all(axis=1)
    if int(np.count_nonzero(keep)) >= 2:
        return trace[keep].astype(np.float32, copy=False)
    finite = trace[np.isfinite(trace).all(axis=1)]
    return finite[: min(2, int(finite.shape[0]))].astype(np.float32, copy=False)


def _native_whole_fiber_span_volume_trace(
    source: _Trace2CpSegmentSource,
    span: _NativeWholeFiberVisualSpan,
) -> np.ndarray:
    spacing = float(source.record.volume_spacing_base)
    parts: list[np.ndarray] = []
    previous_success = False
    for segment in span.segments:
        trace_xyz = _trace_zyx_to_base_xyz(segment.trace_zyx, spacing)
        if not bool(segment.success):
            trace_xyz = _trim_failed_volume_trace_before_target(
                source,
                trace_xyz,
                start_xy=_trace2cp_source_control_point_xy(source, int(segment.start_cp_index)),
                target_xy=_trace2cp_source_control_point_xy(source, int(segment.target_cp_index)),
            )
        if int(trace_xyz.shape[0]) < 2:
            continue
        if parts and previous_success and bool(segment.success):
            parts.append(trace_xyz[1:].copy())
        else:
            parts.append(trace_xyz.copy())
        previous_success = bool(segment.success)
    if not any(part.size for part in parts):
        raise ValueError(
            "native whole-fiber regenerated strip has no usable traced points: "
            f"start_cp={span.start_cp_index} end_cp={span.end_cp_index}"
        )
    trace = np.concatenate([part for part in parts if part.size], axis=0).astype(np.float32)
    if int(trace.shape[0]) < 2:
        raise ValueError(
            "native whole-fiber regenerated strip needs at least two traced points: "
            f"points={int(trace.shape[0])} start_cp={span.start_cp_index} "
            f"end_cp={span.end_cp_index}"
        )
    return trace


def _render_native_whole_fiber_span_panels(
    geometry_loader: Any,
    *,
    span: _NativeWholeFiberVisualSpan,
    trace2cp_rf_margin_px: float,
    cache: NativeTraceFieldCache,
    image_normalization: str,
    strip_cross_width_px: int = 64,
) -> tuple[Any, ...]:
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
    side_control_points = _whole_fiber_span_control_points_for_view(
        source,
        span,
        axis_name="offset_axis_xyz",
    )
    top_control_points = _whole_fiber_span_control_points_for_view(
        source,
        span,
        axis_name="side_axis_xyz",
    )
    control_point_labels = _whole_fiber_span_control_point_labels(span)
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
    regenerated_source_trace = _native_whole_fiber_span_volume_trace(source, span)
    regenerated_source = geometry_loader.build_trace2cp_volume_trace_segment_source(
        source,
        regenerated_source_trace,
        device=torch.device("cpu"),
    )
    regenerated_side_control_points = _whole_fiber_span_control_points_for_view(
        regenerated_source,
        span,
        axis_name="offset_axis_xyz",
    )
    regenerated_top_control_points = _whole_fiber_span_control_points_for_view(
        regenerated_source,
        span,
        axis_name="side_axis_xyz",
    )
    _regenerated_sample, regenerated_side_image, regenerated_side_valid = (
        geometry_loader.sample_trace2cp_segment_source(regenerated_source)
    )
    regenerated_top_image, regenerated_top_valid = geometry_loader.sample_trace2cp_top_strip_source(
        regenerated_source
    )
    regenerated_side_coords_xyz, regenerated_side_grid_valid = (
        geometry_loader.trace2cp_segment_coords_xyz(regenerated_source)
    )
    regenerated_top_coords_xyz, regenerated_top_grid_valid = (
        geometry_loader.trace2cp_top_strip_coords_xyz(regenerated_source)
    )
    regenerated_side_presence, regenerated_side_presence_valid = _sample_presence_on_strip(
        cache,
        regenerated_side_coords_xyz,
        np.asarray(regenerated_side_grid_valid, dtype=bool)
        & np.asarray(regenerated_side_valid, dtype=bool),
        spacing_base=spacing,
        progress_label=f"span{span.start_cp_index}-{span.end_cp_index}-regenerated-side",
    )
    regenerated_top_presence, regenerated_top_presence_valid = _sample_presence_on_strip(
        cache,
        regenerated_top_coords_xyz,
        np.asarray(regenerated_top_grid_valid, dtype=bool)
        & np.asarray(regenerated_top_valid, dtype=bool),
        spacing_base=spacing,
        progress_label=f"span{span.start_cp_index}-{span.end_cp_index}-regenerated-top",
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
            control_points_xy=side_control_points,
            control_point_labels=control_point_labels,
            overlays=side_overlays,
        ),
        _draw_trace_panel(
            _presence_to_u8(side_presence, side_presence_valid),
            side_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"side 3D presence {title_suffix}",
            control_points_xy=side_control_points,
            control_point_labels=control_point_labels,
            overlays=side_overlays,
        ),
        _draw_trace_panel(
            _image_to_u8(top_image, top_valid, normalization=image_normalization),
            top_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"top input {title_suffix}",
            control_points_xy=top_control_points,
            control_point_labels=control_point_labels,
            overlays=top_overlays,
        ),
        _draw_trace_panel(
            _presence_to_u8(top_presence, top_presence_valid),
            top_presence_valid,
            source.line_xy,
            source.start_control_point_xy,
            source.target_control_point_xy,
            title=f"top 3D presence {title_suffix}",
            control_points_xy=top_control_points,
            control_point_labels=control_point_labels,
            overlays=top_overlays,
        ),
        _draw_trace_panel(
            _image_to_u8(
                regenerated_side_image,
                regenerated_side_valid,
                normalization=image_normalization,
            ),
            regenerated_side_valid,
            regenerated_source.line_xy,
            regenerated_source.start_control_point_xy,
            regenerated_source.target_control_point_xy,
            title=f"regenerated side input {title_suffix}",
            control_points_xy=regenerated_side_control_points,
            control_point_labels=control_point_labels,
            line_width=1,
        ),
        _draw_trace_panel(
            _presence_to_u8(regenerated_side_presence, regenerated_side_presence_valid),
            regenerated_side_presence_valid,
            regenerated_source.line_xy,
            regenerated_source.start_control_point_xy,
            regenerated_source.target_control_point_xy,
            title=f"regenerated side 3D presence {title_suffix}",
            control_points_xy=regenerated_side_control_points,
            control_point_labels=control_point_labels,
            line_width=1,
        ),
        _draw_trace_panel(
            _image_to_u8(
                regenerated_top_image,
                regenerated_top_valid,
                normalization=image_normalization,
            ),
            regenerated_top_valid,
            regenerated_source.line_xy,
            regenerated_source.start_control_point_xy,
            regenerated_source.target_control_point_xy,
            title=f"regenerated top input {title_suffix}",
            control_points_xy=regenerated_top_control_points,
            control_point_labels=control_point_labels,
            line_width=1,
        ),
        _draw_trace_panel(
            _presence_to_u8(regenerated_top_presence, regenerated_top_presence_valid),
            regenerated_top_presence_valid,
            regenerated_source.line_xy,
            regenerated_source.start_control_point_xy,
            regenerated_source.target_control_point_xy,
            title=f"regenerated top 3D presence {title_suffix}",
            control_points_xy=regenerated_top_control_points,
            control_point_labels=control_point_labels,
            line_width=1,
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
        "cp_start_direction": "sampled_model_direction_aligned_to_cp_tangent",
    }


def _native_trace_normal_principal_axis_method(raw_config: dict[str, Any]) -> str:
    trace_cfg = raw_config.get("native_trace2cp", {})
    if not isinstance(trace_cfg, dict):
        return "eigh"
    method = str(trace_cfg.get("normal_principal_axis_method", "eigh"))
    if method not in _NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS:
        raise ValueError(
            "unsupported native_trace2cp.normal_principal_axis_method: "
            f"{method!r}"
        )
    return method


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
    render_visualization: bool = False,
    profile: bool = False,
    normal_sampler_mode: str = "sparse-corner-principal",
    normal_principal_axis_method: str = "config",
    debug_compare_normal_sampler: str | None = None,
    debug_normal_angle_threshold_degrees: float = 1.0,
    output_stem: str = "trace2cp_native_3d",
    prediction_adapter: FiberTrace3DPredictAdapter | None = None,
    model: torch.nn.Module | None = None,
    whole_fiber_start_cp_index: int = 0,
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
    whole_start_cp = int(whole_fiber_start_cp_index)
    if not whole_mode and whole_start_cp != 0:
        raise ValueError(
            "--whole-fiber-start-cp-index only applies to whole-fiber --fiber-json mode"
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
    if prediction_adapter is None:
        prediction_adapter = FiberTrace3DPredictAdapter(raw_config, checkpoint=checkpoint)
    if model is None:
        model = prediction_adapter.load_model(device=device)
    profiler = _NativeTraceProfiler() if bool(profile) else None
    normal_sampler_mode = str(normal_sampler_mode)
    if normal_sampler_mode not in _NATIVE_NORMAL_SAMPLER_MODES:
        raise ValueError(
            f"unsupported native 3D Trace2CP normal sampler mode: {normal_sampler_mode!r}"
        )
    normal_principal_axis_method = str(normal_principal_axis_method)
    if normal_principal_axis_method == "config":
        normal_principal_axis_method = _native_trace_normal_principal_axis_method(raw_config)
    if normal_principal_axis_method not in _NATIVE_NORMAL_PRINCIPAL_AXIS_METHODS:
        raise ValueError(
            "unsupported native 3D Trace2CP normal principal-axis method: "
            f"{normal_principal_axis_method!r}"
        )
    cache = NativeTraceFieldCache(
        record=record,
        prediction_adapter=prediction_adapter,
        model=model,
        patch_shape_zyx=cfg.inference_patch_shape_zyx,
        core_margin_voxels=cfg.core_margin_voxels,
        inference_scaledown_power=int(cfg.inference_scaledown_power),
        inference_blur_sigma_voxels=float(cfg.inference_blur_sigma_voxels),
        device=device,
        max_cached_bytes=_cache_bytes_from_gib(float(cfg.max_cached_inference_gib)),
        inference_block_batch_size=int(cfg.inference_block_batch_size),
        profiler=profiler,
    )
    baseline_normal_sampler: NativeTraceNormalSampler | None = None
    if normal_sampler_mode == "baseline" or debug_compare_normal_sampler is not None:
        baseline_normal_sampler = _NativeLasagnaNormalSampler(
            geometry_loader=geometry_loader,
            trace_record=record,
            normal_record=_native_trace_geometry_normal_record(geometry_loader, record),
            profiler=profiler,
        )
    if debug_compare_normal_sampler is not None:
        if baseline_normal_sampler is None:
            raise RuntimeError("debug normal comparison requires a baseline sampler")
        normal_sampler: NativeTraceNormalSampler = _make_debug_normal_comparison_sampler(
            primary=baseline_normal_sampler,
            trace_record=record,
            raw_config=raw_config,
            device=device,
            mode=str(debug_compare_normal_sampler),
            principal_axis_method=normal_principal_axis_method,
            angle_threshold_degrees=float(debug_normal_angle_threshold_degrees),
            profiler=profiler,
        )
    elif normal_sampler_mode == "sparse-corner-principal":
        normal_sampler: NativeTraceNormalSampler = _make_sparse_corner_normal_sampler(
            trace_record=record,
            raw_config=raw_config,
            device=device,
            principal_axis_method=normal_principal_axis_method,
            profiler=profiler,
        )
    elif baseline_normal_sampler is not None:
        normal_sampler = baseline_normal_sampler
    else:  # pragma: no cover - guarded by mode validation above.
        raise ValueError(
            f"unsupported native 3D Trace2CP normal sampler mode: {normal_sampler_mode!r}"
        )
    effective_normal_sampler = (
        "sparse-corner-principal+baseline-compare"
        if debug_compare_normal_sampler is not None
        else normal_sampler_mode
    )
    cfg = _native_trace_cfg_with_effective_smoothness(cfg, normal_sampler=normal_sampler)
    print(
        "native_trace2cp_3d input "
        f"base_volume_scale={getattr(record, 'volume_scale', 'unknown')} "
        f"volume_spacing_base={float(getattr(record, 'volume_spacing_base', 1.0)):.6g} "
        f"image_normalization={loader_config.image_normalization} "
        f"inference_scaledown_power={int(cfg.inference_scaledown_power)} "
        f"inference_scaledown_factor={int(cache.inference_scaledown_factor)} "
        f"inference_blur_sigma_voxels={float(cfg.inference_blur_sigma_voxels):.3f} "
        f"normal_sampler={effective_normal_sampler} "
        f"normal_principal_axis_method={normal_principal_axis_method} "
        f"debug_compare_normal_sampler={debug_compare_normal_sampler or 'off'} "
        f"debug_normal_angle_threshold_degrees={float(debug_normal_angle_threshold_degrees):.6g} "
        f"sampler={type(getattr(record, 'sampler', None)).__name__} "
        f"blocking={getattr(getattr(record, 'sampler', None), 'blocking', 'n/a')}",
        flush=True,
    )
    if whole_mode:
        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = str(output_stem)
        image_path = out_dir / f"{stem}_vis.jpg"
        closed_panel_blocks: list[tuple[Any, ...]] = []
        closed_span_count = 0
        last_export_paths: list[Path] = []
        active_segments: list[NativeWholeFiberSegmentResult] = []
        active_start_cp_index = 0
        if render_visualization:
            last_export_paths = _save_whole_fiber_panel_pages(
                [],
                image_path=image_path,
                status_text="initializing whole-fiber trace",
                quality=90,
            )
            print(f"native whole-fiber partial={image_path} initializing", flush=True)

        def render_span_blocks(
            span: _NativeWholeFiberVisualSpan,
            *,
            split_inside_span: bool,
        ) -> list[tuple[Any, ...]]:
            spans = (
                _split_native_whole_fiber_visual_span(
                    span,
                    max_width_px=_NATIVE_WHOLE_FIBER_VIS_JPEG_SAFE_PX,
                    trace2cp_rf_margin_px=float(trace2cp_cfg.rf_margin_px),
                )
                if split_inside_span
                else (span,)
            )
            return [
                _render_native_whole_fiber_span_panels(
                    geometry_loader,
                    span=subspan,
                    trace2cp_rf_margin_px=float(trace2cp_cfg.rf_margin_px),
                    cache=cache,
                    image_normalization=loader_config.image_normalization,
                    strip_cross_width_px=64,
                )
                for subspan in spans
            ]

        def on_segment(
            segment: NativeWholeFiberSegmentResult,
            partial: NativeWholeFiberResult | None,
        ) -> None:
            nonlocal active_start_cp_index, active_segments, closed_panel_blocks
            nonlocal closed_span_count, last_export_paths
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
            active_panel_blocks = render_span_blocks(
                active_span,
                split_inside_span=(
                    _native_whole_fiber_visual_span_estimated_width_px(
                        active_span,
                        trace2cp_rf_margin_px=float(trace2cp_cfg.rf_margin_px),
                    )
                    >= float(_NATIVE_WHOLE_FIBER_VIS_JPEG_SAFE_PX)
                ),
            )
            restarts = 0 if partial is None else int(partial.restart_count)
            restarts_per_kvx = 0.0 if partial is None else float(partial.restarts_per_kvx)
            physical_status = ""
            if (
                partial is not None
                and partial.restarts_per_meter is not None
                and partial.reference_length_meters is not None
            ):
                physical_status = (
                    " "
                    + _format_trace2cp_meter_rate(
                        partial.restarts_per_meter,
                        restart_count=int(partial.restart_count),
                        reference_length_meters=partial.reference_length_meters,
                    )
                )
            all_panel_blocks = [*closed_panel_blocks, *active_panel_blocks]
            last_export_paths = _save_whole_fiber_panel_pages(
                all_panel_blocks,
                image_path=image_path,
                status_text=(
                    f"segments={len(partial.segments) if partial is not None else 0} "
                    f"spans={closed_span_count + 1} restarts={restarts} "
                    f"err/kvx={_format_trace2cp_kvx_rate(restarts_per_kvx)}"
                    f"{physical_status}"
                ),
                quality=90,
            )
            print(
                "native whole-fiber partial="
                f"{last_export_paths[0]} pages={len(last_export_paths)} "
                f"segment={segment.start_cp_index}->{segment.target_cp_index}",
                flush=True,
            )
            if not bool(segment.success):
                closed_panel_blocks.extend(active_panel_blocks)
                closed_span_count += len(active_panel_blocks)
                active_segments = []
                active_start_cp_index = int(segment.target_cp_index)

        cp_count = int(record.fiber.control_points_zyx.shape[0])
        final_cp_index = cp_count - 1
        print(
            "native_trace2cp_3d whole_fiber "
            f"fiber_path={'' if record.fiber.path is None else record.fiber.path} "
            f"control_points={cp_count} start_cp={whole_start_cp} "
            f"target_cp={final_cp_index} segments={max(0, final_cp_index - whole_start_cp)} "
            f"threshold_voxels={float(cfg.whole_fiber_error_threshold_voxels):.3f}",
            flush=True,
        )
        if profiler is not None:
            profiler.restart_total()
        trace_wall_start = time.perf_counter()
        trace_cpu_start = time.process_time()
        whole = trace_native_3d_whole_fiber(
            cache,
            record=record,
            cfg=cfg,
            error_threshold_voxels=float(cfg.whole_fiber_error_threshold_voxels),
            start_cp_index=int(whole_start_cp),
            progress=True,
            segment_callback=on_segment if render_visualization else None,
            normal_sampler=normal_sampler,
            profiler=profiler,
        )
        trace_wall_seconds = float(time.perf_counter() - trace_wall_start)
        trace_cpu_seconds = float(time.process_time() - trace_cpu_start)
        if profiler is not None:
            profiler.finish_total()
        summary = {
            "mode": "whole_fiber",
            "fiber_path": "" if record.fiber.path is None else str(record.fiber.path),
            "control_point_count": int(cp_count),
            "start_control_point_index": int(whole_start_cp),
            "target_control_point_index": int(final_cp_index),
            "segment_count": int(whole.segment_count),
            "restart_count": int(whole.restart_count),
            "native_trace2cp_fiber_restarts_per_kvx": float(whole.restarts_per_kvx),
            "native_trace2cp_fiber_restarts_per_meter": None
            if whole.restarts_per_meter is None
            else float(whole.restarts_per_meter),
            "reference_length_voxels": float(whole.reference_length_voxels),
            "reference_length_meters": None
            if whole.reference_length_meters is None
            else float(whole.reference_length_meters),
            "restart_fraction_per_segment": float(
                whole.restart_count / max(1, whole.segment_count)
            ),
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
            "inference_scaledown_power": int(cfg.inference_scaledown_power),
            "inference_scaledown_factor": int(cache.inference_scaledown_factor),
            "inference_blur_sigma_voxels": float(cfg.inference_blur_sigma_voxels),
            "inference_block_batch_size": int(cfg.inference_block_batch_size),
            "inferred_blocks": int(cache.total_inferred_blocks),
            "resident_inferred_blocks": int(len(cache._blocks)),
            "evicted_inferred_blocks": int(cache.evicted_inferred_blocks),
            "resident_inferred_block_bytes": int(cache.resident_inferred_block_bytes),
            "resident_inferred_block_gib": float(
                cache.resident_inferred_block_bytes / float(_GIB)
            ),
            "max_cached_inference_gib": float(cfg.max_cached_inference_gib),
            "max_cached_inference_bytes": _cache_bytes_from_gib(
                float(cfg.max_cached_inference_gib)
            ),
            "trace_wall_seconds": trace_wall_seconds,
            "trace_cpu_seconds": trace_cpu_seconds,
            "trace_profile": None if profiler is None else profiler.summary(),
            "export": str(last_export_paths[0]) if render_visualization and last_export_paths else None,
            "exports": [str(path) for path in last_export_paths] if render_visualization else [],
            "visualization_enabled": bool(render_visualization),
            "segments": [
                {
                    "start_control_point_index": int(segment.start_cp_index),
                    "target_control_point_index": int(segment.target_cp_index),
                    "success": bool(segment.success),
                    "restart": bool(segment.restart),
                    "reason": segment.reason,
                    "reached_target_plane": bool(segment.reached_target_plane),
                    "in_plane_error_voxels": float(segment.in_plane_error_voxels),
                    "selected_target_plane_name": segment.selected_target_plane_name,
                    "selected_target_plane_crossing_zyx": None
                    if segment.selected_target_plane_crossing_zyx is None
                    else [
                        float(v)
                        for v in np.asarray(
                            segment.selected_target_plane_crossing_zyx,
                            dtype=np.float32,
                        )
                    ],
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
        summary_path = out_dir / f"{stem}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(
            "native_trace2cp_fiber "
            f"err/kvx={_format_trace2cp_kvx_rate(whole.restarts_per_kvx)} restarts={whole.restart_count} "
            f"segments={whole.segment_count}",
            flush=True,
        )
        if whole.restarts_per_meter is not None and whole.reference_length_meters is not None:
            print(
                "native_trace2cp_fiber "
                + _format_trace2cp_meter_rate(
                    whole.restarts_per_meter,
                    restart_count=int(whole.restart_count),
                    reference_length_meters=whole.reference_length_meters,
                ),
                flush=True,
            )
        print(
            "native_trace2cp_timing "
            f"trace_wall_s={trace_wall_seconds:.3f} "
            f"trace_cpu_s={trace_cpu_seconds:.3f}",
            flush=True,
        )
        if profiler is not None:
            profiler.print_table()
        print(
            "native_trace2cp_3d whole_fiber "
            f"blocks={len(cache._blocks)} inferred={cache.total_inferred_blocks} "
            f"evicted={cache.evicted_inferred_blocks} "
            f"cache_gib={cache.resident_inferred_block_bytes / float(_GIB):.3f} "
            f"summary={summary_path}"
            + (
                f" export={last_export_paths[0]} exports={len(last_export_paths)}"
                if render_visualization and last_export_paths
                else ""
            ),
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
    if profiler is not None:
        profiler.restart_total()
    forward_target_planes = _fiber_target_planes_zyx(
        cache,
        record,
        target_control_point_index=int(selection.target_cp_index),
        target_zyx=target_zyx,
        inference_reference_direction_zyx=reverse_initial_direction,
    )
    reverse_target_planes = _fiber_target_planes_zyx(
        cache,
        record,
        target_control_point_index=int(selection.start_cp_index),
        target_zyx=start_zyx,
        inference_reference_direction_zyx=forward_initial_direction,
    )
    trace_wall_start = time.perf_counter()
    trace_cpu_start = time.process_time()
    result = trace_native_3d_pair(
        cache,
        start_zyx=start_zyx,
        target_zyx=target_zyx,
        forward_initial_direction_zyx=forward_initial_direction,
        reverse_initial_direction_zyx=reverse_initial_direction,
        cfg=cfg,
        forward_target_planes_zyx=forward_target_planes,
        reverse_target_planes_zyx=reverse_target_planes,
        progress=True,
        normal_sampler=normal_sampler,
        profiler=profiler,
    )
    trace_wall_seconds = float(time.perf_counter() - trace_wall_start)
    trace_cpu_seconds = float(time.process_time() - trace_cpu_start)
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(output_stem)
    image_path = out_dir / f"{stem}_vis.jpg"
    visualization_cross_strip_height_px: int | None = None
    visualization_max_cross_strip_height_px: int | None = None
    if render_visualization:
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

        with (
            profiler.measure("single_visualization")
            if profiler is not None
            else _NullNativeTraceProfileSpan()
        ):
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
            visualization_cross_strip_height_px = int(source.source_shape_hw[0])
            visualization_max_cross_strip_height_px = int(max_source.source_shape_hw[0])
            sheet = _make_native_trace_visualization(
                geometry_loader,
                source,
                result,
                cache=cache,
                image_normalization=loader_config.image_normalization,
                partial_output_path=image_path,
            )
            sheet.convert("RGB").save(image_path, quality=95)
    if profiler is not None:
        profiler.finish_total()
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
        "inference_scaledown_power": int(cfg.inference_scaledown_power),
        "inference_scaledown_factor": int(cache.inference_scaledown_factor),
        "inference_blur_sigma_voxels": float(cfg.inference_blur_sigma_voxels),
        "inference_block_batch_size": int(cfg.inference_block_batch_size),
        "visualization_enabled": bool(render_visualization),
        "visualization_cross_strip_height_px": visualization_cross_strip_height_px,
        "visualization_max_cross_strip_height_px": visualization_max_cross_strip_height_px,
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
        "inferred_blocks": int(cache.total_inferred_blocks),
        "resident_inferred_blocks": int(len(cache._blocks)),
        "evicted_inferred_blocks": int(cache.evicted_inferred_blocks),
        "resident_inferred_block_bytes": int(cache.resident_inferred_block_bytes),
        "resident_inferred_block_gib": float(
            cache.resident_inferred_block_bytes / float(_GIB)
        ),
        "max_cached_inference_gib": float(cfg.max_cached_inference_gib),
        "max_cached_inference_bytes": _cache_bytes_from_gib(
            float(cfg.max_cached_inference_gib)
        ),
        "trace_wall_seconds": trace_wall_seconds,
        "trace_cpu_seconds": trace_cpu_seconds,
        "trace_profile": None if profiler is None else profiler.summary(),
        "export": str(image_path) if render_visualization else None,
    }
    summary_path = out_dir / f"{stem}_summary.json"
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
        "native_trace2cp_timing "
        f"trace_wall_s={trace_wall_seconds:.3f} "
        f"trace_cpu_s={trace_cpu_seconds:.3f}",
        flush=True,
    )
    if profiler is not None:
        profiler.print_table()
    print(
        "native_trace2cp_3d "
        f"sample_index={selection.sample_index} start_cp={selection.start_cp_index} "
        f"target_cp={selection.target_cp_index} "
        f"forward_reached={result.forward.reached_target_plane} reverse_reached={result.reverse.reached_target_plane} "
        f"blocks={len(cache._blocks)} inferred={cache.total_inferred_blocks} "
        f"evicted={cache.evicted_inferred_blocks} "
        f"cache_gib={cache.resident_inferred_block_bytes / float(_GIB):.3f} "
        f"summary={summary_path}"
        + (f" export={image_path}" if render_visualization else ""),
        flush=True,
    )
    return result


def _indexed_trace2cp_output_stem(index: int, count: int) -> str:
    width = max(3, len(str(max(0, int(count) - 1))))
    return f"trace2cp_native_3d_{int(index):0{width}d}"


def _aggregate_native_whole_fiber_results(
    results: Sequence[NativeWholeFiberResult],
) -> NativeMultiFiberResult:
    if not results:
        raise ValueError("native Trace2CP multi-fiber aggregation requires at least one result")
    restart_count = int(sum(int(result.restart_count) for result in results))
    segment_count = int(sum(int(result.segment_count) for result in results))
    reference_length_voxels = float(
        sum(float(result.reference_length_voxels) for result in results)
    )
    reference_lengths_m = [result.reference_length_meters for result in results]
    reference_length_meters = (
        None
        if any(length is None for length in reference_lengths_m)
        else float(sum(float(length) for length in reference_lengths_m))
    )
    return NativeMultiFiberResult(
        results=tuple(results),
        restart_count=restart_count,
        segment_count=segment_count,
        restarts_per_kvx=_restarts_per_kvx(restart_count, reference_length_voxels),
        reference_length_voxels=reference_length_voxels,
        reference_length_meters=reference_length_meters,
        restarts_per_meter=_restarts_per_meter(restart_count, reference_length_meters),
        run_count=int(len(results) + restart_count),
    )


def run_native_trace2cp_many(
    config_path: str | Path,
    *,
    checkpoint: str | Path,
    export_dir: str | Path,
    fiber_jsons: Sequence[str | Path],
    sample_index: int | None,
    start_cp_index: int | None = None,
    target_cp_index: int | None = None,
    target_offset: int = 1,
    sample_mode: str | None = None,
    native_cfg: NativeTrace2CpConfig | None = None,
    render_visualization: bool = False,
    profile: bool = False,
    normal_sampler_mode: str = "sparse-corner-principal",
    normal_principal_axis_method: str = "config",
    debug_compare_normal_sampler: str | None = None,
    debug_normal_angle_threshold_degrees: float = 1.0,
    whole_fiber_start_cp_index: int = 0,
) -> NativeWholeFiberResult | NativeMultiFiberResult:
    paths = [Path(path) for path in fiber_jsons]
    if not paths:
        raise ValueError("native Trace2CP requires at least one --fiber-json path")
    if len(paths) == 1:
        return run_native_trace2cp(
            config_path,
            checkpoint=checkpoint,
            export_dir=export_dir,
            sample_index=sample_index,
            fiber_json=paths[0],
            start_cp_index=start_cp_index,
            target_cp_index=target_cp_index,
            target_offset=int(target_offset),
            sample_mode=sample_mode,
            native_cfg=native_cfg,
            render_visualization=render_visualization,
            profile=profile,
            normal_sampler_mode=normal_sampler_mode,
            normal_principal_axis_method=normal_principal_axis_method,
            debug_compare_normal_sampler=debug_compare_normal_sampler,
            debug_normal_angle_threshold_degrees=float(debug_normal_angle_threshold_degrees),
            whole_fiber_start_cp_index=int(whole_fiber_start_cp_index),
        )
    if sample_index is not None or start_cp_index is not None or target_cp_index is not None:
        raise ValueError(
            "multiple --fiber-json paths support whole-fiber mode only; "
            "omit --sample-index and explicit CP selectors"
        )
    if sample_mode is not None and str(sample_mode) != "flat":
        raise ValueError("multiple --fiber-json whole-fiber mode requires --sample-mode flat or omitted")

    raw_config = _load_raw_config(config_path)
    training = dict(raw_config.get("training", {}))
    device = _device_from_training(training)
    prediction_adapter = FiberTrace3DPredictAdapter(raw_config, checkpoint=checkpoint)
    model = prediction_adapter.load_model(device=device)
    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[NativeWholeFiberResult] = []
    per_fiber_summaries: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        stem = _indexed_trace2cp_output_stem(index, len(paths))
        print(
            "native_trace2cp_3d multi_fiber "
            f"index={index}/{len(paths)} fiber_json={path}",
            flush=True,
        )
        result = run_native_trace2cp(
            config_path,
            checkpoint=checkpoint,
            export_dir=export_dir,
            sample_index=None,
            fiber_json=path,
            start_cp_index=None,
            target_cp_index=None,
            target_offset=int(target_offset),
            sample_mode="flat" if sample_mode is None else sample_mode,
            native_cfg=native_cfg,
            render_visualization=render_visualization,
            profile=profile,
            normal_sampler_mode=normal_sampler_mode,
            normal_principal_axis_method=normal_principal_axis_method,
            debug_compare_normal_sampler=debug_compare_normal_sampler,
            debug_normal_angle_threshold_degrees=float(debug_normal_angle_threshold_degrees),
            output_stem=stem,
            prediction_adapter=prediction_adapter,
            model=model,
            whole_fiber_start_cp_index=int(whole_fiber_start_cp_index),
        )
        if not isinstance(result, NativeWholeFiberResult):
            raise RuntimeError("multi-fiber native Trace2CP unexpectedly returned a pair result")
        results.append(result)
        per_fiber_summaries.append(
            {
                "index": int(index),
                "fiber_json": str(path),
                "summary": str(out_dir / f"{stem}_summary.json"),
                "export": str(out_dir / f"{stem}_vis.jpg") if render_visualization else None,
                "start_control_point_index": int(whole_fiber_start_cp_index),
                "restart_count": int(result.restart_count),
                "segment_count": int(result.segment_count),
                "native_trace2cp_fiber_restarts_per_kvx": float(result.restarts_per_kvx),
                "native_trace2cp_fiber_restarts_per_meter": None
                if result.restarts_per_meter is None
                else float(result.restarts_per_meter),
                "reference_length_voxels": float(result.reference_length_voxels),
                "reference_length_meters": None
                if result.reference_length_meters is None
                else float(result.reference_length_meters),
            }
        )

    aggregate = _aggregate_native_whole_fiber_results(results)
    summary = {
        "mode": "multi_whole_fiber",
        "fiber_count": int(len(paths)),
        "start_control_point_index": int(whole_fiber_start_cp_index),
        "restart_count": int(aggregate.restart_count),
        "segment_count": int(aggregate.segment_count),
        "run_count": int(aggregate.run_count),
        "native_trace2cp_fiber_restarts_per_kvx": float(aggregate.restarts_per_kvx),
        "native_trace2cp_fiber_restarts_per_meter": None
        if aggregate.restarts_per_meter is None
        else float(aggregate.restarts_per_meter),
        "reference_length_voxels": float(aggregate.reference_length_voxels),
        "reference_length_meters": None
        if aggregate.reference_length_meters is None
        else float(aggregate.reference_length_meters),
        "restart_fraction_per_segment": float(
            aggregate.restart_count / max(1, aggregate.segment_count)
        ),
        "visualization_enabled": bool(render_visualization),
        "fibers": per_fiber_summaries,
    }
    summary_path = out_dir / "trace2cp_native_3d_summary_all.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        "native_trace2cp_fibers "
        f"err/kvx={_format_trace2cp_kvx_rate(aggregate.restarts_per_kvx)} "
        f"restarts={aggregate.restart_count} segments={aggregate.segment_count} "
        f"fibers={len(paths)}",
        flush=True,
    )
    if aggregate.restarts_per_meter is not None and aggregate.reference_length_meters is not None:
        print(
            "native_trace2cp_fibers "
            + _format_trace2cp_meter_rate_for_runs(
                aggregate.restarts_per_meter,
                run_count=int(aggregate.run_count),
                reference_length_meters=aggregate.reference_length_meters,
            ),
            flush=True,
        )
    print(
        "native_trace2cp_3d multi_fiber "
        f"summary={summary_path}"
        + (" exports_indexed=true" if render_visualization else ""),
        flush=True,
    )
    return aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native 3D Trace2CP cone tracer",
        formatter_class=_ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Render Trace2CP JPG visualization. By default only metrics/summary are written.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect and print detailed Trace2CP stage timings.",
    )
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--fiber-json", type=Path, nargs="+", default=None)
    parser.add_argument("--start-cp-index", type=int, default=None)
    parser.add_argument("--target-cp-index", type=int, default=None)
    parser.add_argument("--target-offset", type=int, default=1)
    parser.add_argument(
        "--whole-fiber-start-cp-index",
        type=int,
        default=0,
        help="Whole-fiber --fiber-json start CP index; traces from this CP to the final CP.",
    )
    parser.add_argument("--sample-mode", choices=("random", "flat"), default=None)
    parser.add_argument("--step-voxels", type=float, default=4.0)
    parser.add_argument("--cone-angle-degrees", type=float, default=25.0)
    parser.add_argument("--cone-grid-size", type=int, default=25)
    parser.add_argument("--cone-angle-step-degrees", type=float, default=5.0)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--beam-prune-distance-voxels", type=float, default=1.0)
    parser.add_argument("--beam-lookahead-steps", type=int, default=2)
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
    parser.add_argument("--core-margin-voxels", type=int, default=48)
    parser.add_argument(
        "--inference-scaledown-power",
        type=int,
        default=0,
        help=(
            "Power-of-two Gaussian pyramid scaledown applied to raw native 3D "
            "inference outputs before tracing samples them: 0=1x, 1=0.5x, 2=0.25x."
        ),
    )
    parser.add_argument(
        "--inference-blur-sigma-voxels",
        type=float,
        default=0.0,
        help=(
            "3D Gaussian sigma applied to raw inference outputs after optional "
            "scaledown and before trusted-core cropping, measured in unscaled "
            "selected-level inference voxels."
        ),
    )
    parser.add_argument(
        "--inference-block-batch-size",
        type=int,
        default=2,
        help="Maximum number of missing native 3D inference blocks to forward in one batch.",
    )
    parser.add_argument(
        "--max-cached-inference-gib",
        type=float,
        default=8.0,
        help=(
            "Maximum resident CPU model-output cache size in GiB for native 3D tracing. "
            "Use 0 to disable retention; negative values are rejected."
        ),
    )
    parser.add_argument(
        "--normal-sampler",
        choices=_NATIVE_NORMAL_SAMPLER_MODES,
        default="sparse-corner-principal",
        help=(
            "Lasagna normal sampler for native 3D tracing. sparse-corner-principal "
            "uses sparse chunk reads and baseline-style tensor reconstruction; "
            "baseline uses the geometry-loader sampler."
        ),
    )
    parser.add_argument(
        "--normal-principal-axis-method",
        choices=_NATIVE_NORMAL_PRINCIPAL_AXIS_CLI_CHOICES,
        default="config",
        help=(
            "Principal-axis reconstruction for sparse-corner normals. config "
            "uses native_trace2cp.normal_principal_axis_method or eigh; analytic "
            "is a direct closed-form symmetric tensor experiment."
        ),
    )
    parser.add_argument(
        "--debug-compare-normal-sampler",
        nargs="?",
        const="sparse-corner-principal",
        choices=("sparse-corner-principal",),
        default=None,
        help=(
            "Debug-only: run sparse corner/tensor Lasagna normal sampling in "
            "parallel with the baseline sampler and fail fast on differences. "
            "The tracer uses the sparse sampler after comparison succeeds."
        ),
    )
    parser.add_argument(
        "--debug-normal-angle-threshold-degrees",
        type=float,
        default=1.0,
        help="Angular threshold for --debug-compare-normal-sampler fail-fast checks.",
    )
    parser.add_argument("--whole-fiber-error-threshold-voxels", type=float, default=10.0)
    _fill_missing_argparse_default_help(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    patch_shape = (
        (128, 128, 128)
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
        inference_scaledown_power=int(args.inference_scaledown_power),
        inference_blur_sigma_voxels=float(args.inference_blur_sigma_voxels),
        inference_block_batch_size=int(args.inference_block_batch_size),
        whole_fiber_error_threshold_voxels=float(args.whole_fiber_error_threshold_voxels),
        max_cached_inference_gib=float(args.max_cached_inference_gib),
    )
    if args.fiber_json is None:
        run_native_trace2cp(
            args.config,
            checkpoint=args.checkpoint,
            export_dir=args.export_dir,
            sample_index=None if args.sample_index is None else int(args.sample_index),
            fiber_json=None,
            start_cp_index=args.start_cp_index,
            target_cp_index=args.target_cp_index,
            target_offset=int(args.target_offset),
            sample_mode=args.sample_mode,
            native_cfg=native_cfg,
            render_visualization=bool(args.vis),
            profile=bool(args.profile),
            normal_sampler_mode=str(args.normal_sampler),
            normal_principal_axis_method=str(args.normal_principal_axis_method),
            debug_compare_normal_sampler=args.debug_compare_normal_sampler,
            debug_normal_angle_threshold_degrees=float(args.debug_normal_angle_threshold_degrees),
            whole_fiber_start_cp_index=int(args.whole_fiber_start_cp_index),
        )
    else:
        run_native_trace2cp_many(
            args.config,
            checkpoint=args.checkpoint,
            export_dir=args.export_dir,
            sample_index=None if args.sample_index is None else int(args.sample_index),
            fiber_jsons=list(args.fiber_json),
            start_cp_index=args.start_cp_index,
            target_cp_index=args.target_cp_index,
            target_offset=int(args.target_offset),
            sample_mode=args.sample_mode,
            native_cfg=native_cfg,
            render_visualization=bool(args.vis),
            profile=bool(args.profile),
            normal_sampler_mode=str(args.normal_sampler),
            normal_principal_axis_method=str(args.normal_principal_axis_method),
            debug_compare_normal_sampler=args.debug_compare_normal_sampler,
            debug_normal_angle_threshold_degrees=float(args.debug_normal_angle_threshold_degrees),
            whole_fiber_start_cp_index=int(args.whole_fiber_start_cp_index),
        )


if __name__ == "__main__":
    main()
