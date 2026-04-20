from __future__ import annotations

import json
import os
import sys
from contextlib import nullcontext
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import click
import numpy as np
import torch
import torch.distributed as dist
import zarr
from tqdm.auto import tqdm

from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import (
    _build_target_strip_coords,
    _build_target_strip_positions,
    _sample_from_logits,
    infer_autoreg_mesh,
)
from vesuvius.neural_tracing.autoreg_mesh.model import (
    ATTENTION_SCALING_LEGACY_DOUBLE_SCALED,
    ATTENTION_SCALING_STANDARD,
    GENERATED_TOKEN_TYPE,
    START_TOKEN_TYPE,
    AutoregMeshModel,
    build_pseudo_inference_batch,
)
from vesuvius.neural_tracing.autoreg_mesh.serialization import deserialize_continuation_grid, serialize_split_conditioning_example
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz


Color = tuple[int, int, int]
ORIGINAL_COLOR: Color = (90, 180, 255)
PREDICTED_COLOR: Color = (255, 140, 0)
SEAM_COLOR: Color = (255, 0, 255)
LATTICE_GROW_DIRECTIONS = ("left", "right", "up", "down")
Z_PROJECTED_GROW_DIRECTIONS = ("decreasing-z", "increasing-z")
ALL_GROW_DIRECTIONS = ("auto",) + LATTICE_GROW_DIRECTIONS + Z_PROJECTED_GROW_DIRECTIONS
ATTENTION_SCALING_MODES = (ATTENTION_SCALING_STANDARD, ATTENTION_SCALING_LEGACY_DOUBLE_SCALED)
PLANNER_MODE_FAST = "fast"
PLANNER_MODE_COVERAGE_FIRST = "coverage_first"
PLANNER_MODES = (PLANNER_MODE_FAST, PLANNER_MODE_COVERAGE_FIRST)
DEFAULT_COVERAGE_FIRST_PROMPT_LADDER = (8, 10, 12, 16)
DEFAULT_COVERAGE_FIRST_PREDICT_LADDER = (1, 2)
DEFAULT_COVERAGE_FIRST_WINDOW_LADDER = (16, 24, 32, 48, 64)
DEFAULT_COVERAGE_FIRST_GAP_PROMPT_LADDER = (16, 20)
DEFAULT_COVERAGE_FIRST_GAP_WINDOW_LADDER = (1, 2, 4, 8)


@dataclass(frozen=True)
class ExtensionWindow:
    start: int
    end: int


@dataclass
class ExtensionIterationStats:
    iteration_index: int
    window_count: int
    parent_window_count: int
    child_window_count: int
    deduped_child_window_count: int
    valid_new_vertices: int
    fitted_window_count: int
    skipped_window_count: int
    crop_fit_failed_count: int
    empty_prediction_count: int
    model_stop_count: int
    new_band_frontier_coverage_fraction: float
    new_band_cell_coverage_fraction: float
    new_band_max_gap: int
    new_band_gap_spans: list[tuple[int, int]]
    first_uncovered_frontier_index: int | None
    crop_read_ms: float
    encode_decode_ms: float
    merge_ms: float
    iteration_wall_ms: float
    windows_per_second: float
    peak_batch_size_used: int
    model_only_new_band_frontier_coverage_fraction: float
    geometric_gap_fill_vertex_count: int
    geometric_gap_fill_frontier_count: int


@dataclass(frozen=True)
class ExtensionWindowPayload:
    global_index: int
    window: ExtensionWindow
    sample: dict[str, Any]
    direction: str
    target_grid_shape: tuple[int, int]
    strip_length: int
    num_strips: int
    prompt_strips: int
    predict_strips: int


@dataclass(frozen=True)
class FittedWindowPlan:
    window: ExtensionWindow
    prompt_grid: np.ndarray
    min_corner: np.ndarray
    prompt_strips: int
    predict_strips: int


@dataclass(frozen=True)
class ExtensionWindowCandidate:
    window: ExtensionWindow
    prompt_strips: int
    predict_strips: int


@dataclass(frozen=True)
class PlannerCandidateSpec:
    phase: str
    prompt_strips: int
    predict_strips: int
    window_strip_length: int
    window_overlap: int


@dataclass(frozen=True)
class PlannerAtlasEntry:
    spec: PlannerCandidateSpec
    requested_window: ExtensionWindow
    fitted_plan: FittedWindowPlan | None
    frontier_interval: tuple[int, int] | None = None


@dataclass(frozen=True)
class PlannerSelectionRecord:
    spec: PlannerCandidateSpec
    fitted_plan: FittedWindowPlan
    new_coverage: int
    redundant_overlap: int


@dataclass(frozen=True)
class ExtensionStageResult:
    final_tifxyz_path: Path
    stage_summary: dict[str, Any]


@dataclass
class _DistributedInferRuntime:
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


class ExtensionInferenceRuntime:
    def __init__(
        self,
        model,
        *,
        device: torch.device,
        fast_infer: bool,
        compile_infer: bool,
        amp_dtype: str,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.fast_infer_enabled = bool(fast_infer)
        self.compile_infer_requested = bool(self.fast_infer_enabled and compile_infer and self.device.type == "cuda")
        self.compile_infer_actual = False
        self.compile_infer_failure: str | None = None
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype) if self.fast_infer_enabled else None
        self._encode_conditioning = model.encode_conditioning
        self._forward_from_encoded = model.forward_from_encoded
        if self.compile_infer_requested:
            try:
                self._encode_conditioning = torch.compile(
                    model.encode_conditioning,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                self._forward_from_encoded = torch.compile(
                    model.forward_from_encoded,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True,
                )
                self.compile_infer_actual = True
            except Exception as exc:
                self._disable_compile(exc)

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype | None:
        if self.device.type != "cuda":
            return None
        value = str(amp_dtype).lower()
        if value == "bf16":
            return torch.bfloat16
        if value == "fp16":
            return torch.float16
        raise ValueError(f"unsupported amp_dtype {amp_dtype!r}")

    def _disable_compile(self, exc: Exception | str) -> None:
        self.compile_infer_actual = False
        self.compile_infer_failure = str(exc)
        self._encode_conditioning = self.model.encode_conditioning
        self._forward_from_encoded = self.model.forward_from_encoded

    def _mark_compile_step_begin(self) -> None:
        if self.compile_infer_actual and self.device.type == "cuda" and hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    def _clone_tree(self, value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: self._clone_tree(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [self._clone_tree(inner) for inner in value]
        if isinstance(value, tuple):
            return tuple(self._clone_tree(inner) for inner in value)
        return value

    def autocast_context(self):
        if not self.fast_infer_enabled or self.device.type != "cuda" or self.amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def inference_context(self):
        if self.fast_infer_enabled:
            return torch.inference_mode()
        return nullcontext()

    def encode_conditioning(self, volume, *, vol_tokens=None):
        try:
            self._mark_compile_step_begin()
            outputs = self._encode_conditioning(volume, vol_tokens=vol_tokens)
            return self._clone_tree(outputs) if self.compile_infer_actual else outputs
        except Exception as exc:
            if self.compile_infer_actual:
                self._disable_compile(exc)
                return self._encode_conditioning(volume, vol_tokens=vol_tokens)
            raise

    def forward_from_encoded(self, batch, *, memory_tokens, memory_patch_centers):
        try:
            self._mark_compile_step_begin()
            return self._forward_from_encoded(
                batch,
                memory_tokens=memory_tokens,
                memory_patch_centers=memory_patch_centers,
            )
        except Exception as exc:
            if self.compile_infer_actual:
                self._disable_compile(exc)
                return self._forward_from_encoded(
                    batch,
                    memory_tokens=memory_tokens,
                    memory_patch_centers=memory_patch_centers,
                )
            raise


class VolumeCropCache:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(1, int(max_items))
        self._cache: OrderedDict[tuple[int, int, int, int, int, int], np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key not in self._cache:
            self.misses += 1
            return None
        self.hits += 1
        value = self._cache.pop(key)
        self._cache[key] = value
        return value.copy()

    def put(self, key, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.pop(key)
        self._cache[key] = np.asarray(value, dtype=np.float32)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)


def _initialize_distributed_infer_runtime(
    *,
    enabled: bool,
    device: str | torch.device | None = None,
) -> _DistributedInferRuntime:
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested_device = torch.device(device) if device is not None else None
    if env_world_size <= 1:
        runtime_device = requested_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return _DistributedInferRuntime(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=runtime_device,
            backend=None,
            initialized_process_group=False,
        )
    if not bool(enabled):
        raise RuntimeError("WORLD_SIZE > 1 requires --distributed-infer for extension inference")

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
    return _DistributedInferRuntime(
        is_distributed=True,
        rank=int(dist.get_rank()),
        local_rank=int(local_rank),
        world_size=int(dist.get_world_size()),
        device=runtime_device,
        backend=str(backend),
        initialized_process_group=initialized,
    )


def _destroy_distributed_infer_runtime(runtime: _DistributedInferRuntime) -> None:
    if runtime.is_distributed and runtime.initialized_process_group and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _broadcast_object(obj, *, runtime: _DistributedInferRuntime):
    if not runtime.is_distributed:
        return obj
    payload = [obj if runtime.is_main_process else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _all_gather_objects(obj, *, runtime: _DistributedInferRuntime) -> list[Any]:
    if not runtime.is_distributed:
        return [obj]
    gathered = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def _maybe_barrier(runtime: _DistributedInferRuntime) -> None:
    if runtime.is_distributed:
        dist.barrier()


def _surface_grid_zyx(surface: Tifxyz) -> np.ndarray:
    surface = surface.use_stored_resolution()
    grid = np.stack([surface._z, surface._y, surface._x], axis=-1).astype(np.float32, copy=False)
    valid = np.asarray(surface.valid_vertex_mask, dtype=bool)
    grid = grid.copy()
    grid[~valid] = np.nan
    return grid


def _trim_grid_to_valid_bbox(grid_zyx: np.ndarray, provenance: np.ndarray | None = None):
    valid = np.isfinite(grid_zyx).all(axis=-1)
    if not bool(np.any(valid)):
        raise RuntimeError("surface contains no valid vertices")
    rows = np.where(valid.any(axis=1))[0]
    cols = np.where(valid.any(axis=0))[0]
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    col_slice = slice(int(cols[0]), int(cols[-1]) + 1)
    trimmed_grid = grid_zyx[row_slice, col_slice].copy()
    if provenance is None:
        return trimmed_grid, None, (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))
    trimmed_provenance = provenance[row_slice, col_slice].copy()
    return trimmed_grid, trimmed_provenance, (int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1]))


def _smooth_frontier(grid_zyx: np.ndarray, direction: str, *, max_row_deviation: int = 8) -> np.ndarray:
    grid = grid_zyx.copy()
    valid = np.isfinite(grid).all(axis=-1)
    h, w = valid.shape

    if direction in {"up", "left"}:
        first_valid = np.full(w if direction == "up" else h, -1, dtype=np.int64)
        axis_size = h if direction == "up" else w
        for i in range(w if direction == "up" else h):
            col = valid[:, i] if direction == "up" else valid[i, :]
            indices = np.where(col)[0]
            if len(indices) > 0:
                first_valid[i] = indices[0]
    else:
        first_valid = np.full(w if direction == "down" else h, -1, dtype=np.int64)
        axis_size = h if direction == "down" else w
        for i in range(w if direction == "down" else h):
            col = valid[:, i] if direction == "down" else valid[i, :]
            indices = np.where(col)[0]
            if len(indices) > 0:
                first_valid[i] = indices[-1]

    has_data = first_valid >= 0
    if has_data.sum() < 3:
        return grid

    kernel = 15
    half = kernel // 2
    median_frontier = np.full_like(first_valid, -1)
    for i in range(len(first_valid)):
        if not has_data[i]:
            continue
        lo = max(0, i - half)
        hi = min(len(first_valid), i + half + 1)
        neighbors = first_valid[lo:hi]
        neighbors = neighbors[neighbors >= 0]
        if len(neighbors) > 0:
            median_frontier[i] = int(np.median(neighbors))

    smoothed_count = 0
    for i in range(len(first_valid)):
        if not has_data[i] or median_frontier[i] < 0:
            continue
        dev = abs(int(first_valid[i]) - int(median_frontier[i]))
        if dev > int(max_row_deviation):
            old_row = int(first_valid[i])
            target_row = int(median_frontier[i])
            if direction in {"up", "left"}:
                if old_row < target_row:
                    if direction == "up":
                        grid[old_row:target_row, i, :] = np.nan
                    else:
                        grid[i, old_row:target_row, :] = np.nan
            else:
                if old_row > target_row:
                    if direction == "down":
                        grid[target_row + 1:old_row + 1, i, :] = np.nan
                    else:
                        grid[i, target_row + 1:old_row + 1, :] = np.nan
            smoothed_count += 1

    if smoothed_count > 0:
        print(f"[frontier-smooth] trimmed {smoothed_count} jagged frontier vertices (max_row_deviation={max_row_deviation})")
    return grid


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not str(uri).startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    stripped = str(uri)[5:]
    bucket, _, key = stripped.partition("/")
    if not bucket or not key:
        raise ValueError(f"invalid s3 uri {uri!r}")
    return bucket, key.rstrip("/")


def _open_zarr_volume(volume_uri: str):
    if str(volume_uri).startswith("s3://"):
        import s3fs

        bucket, key = _parse_s3_uri(volume_uri)
        fs = s3fs.S3FileSystem(anon=True)
        store = s3fs.S3Map(root=f"{bucket}/{key}", s3=fs, check=False)
        root = zarr.open(store=store, mode="r")
    else:
        root = zarr.open(str(volume_uri), mode="r")

    if hasattr(root, "shape") and len(root.shape) >= 3:
        return root
    if "0" in root:
        return root["0"]
    numeric_keys = sorted([key for key in root.keys() if str(key).isdigit()], key=lambda value: int(value))
    if numeric_keys:
        return root[numeric_keys[0]]
    raise ValueError(f"could not resolve a volume array from {volume_uri!r}")


def _read_volume_crop(volume_array, min_corner: np.ndarray, crop_size: tuple[int, int, int], *, cache: VolumeCropCache | None = None) -> np.ndarray:
    crop_shape = tuple(int(v) for v in crop_size)
    min_corner = np.asarray(min_corner, dtype=np.int64)
    key = tuple(int(v) for v in (*min_corner.tolist(), *crop_shape))
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    volume_shape = tuple(int(v) for v in volume_array.shape[-3:])
    max_corner = min_corner + np.asarray(crop_shape, dtype=np.int64)
    src_starts = np.maximum(min_corner, 0)
    src_ends = np.minimum(max_corner, np.asarray(volume_shape, dtype=np.int64))
    dst_starts = src_starts - min_corner
    dst_ends = dst_starts + (src_ends - src_starts)
    crop = np.zeros(crop_shape, dtype=np.float32)
    if np.all(src_ends > src_starts):
        src_slices = tuple(slice(int(start), int(end)) for start, end in zip(src_starts, src_ends, strict=True))
        dst_slices = tuple(slice(int(start), int(end)) for start, end in zip(dst_starts, dst_ends, strict=True))
        crop[dst_slices] = np.asarray(volume_array[src_slices], dtype=np.float32)
    if cache is not None:
        cache.put(key, crop)
    return crop


def _direction_axis(direction: str) -> int:
    return 1 if direction in {"left", "right"} else 0


def _direction_sign(direction: str) -> int:
    if direction in {"left", "up"}:
        return 1
    return -1


def _boundary_from_prompt(prompt_grid: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    if direction == "left":
        return prompt_grid[:, -1, :], prompt_grid[:, -2, :]
    if direction == "right":
        return prompt_grid[:, 0, :], prompt_grid[:, 1, :]
    if direction == "up":
        return prompt_grid[-1, :, :], prompt_grid[-2, :, :]
    if direction == "down":
        return prompt_grid[0, :, :], prompt_grid[1, :, :]
    raise ValueError(f"unsupported direction {direction!r}")


def _estimate_extension_points(prompt_grid: np.ndarray, direction: str, predict_strips: int) -> np.ndarray:
    boundary, interior = _boundary_from_prompt(prompt_grid, direction)
    outward = boundary - interior
    points = []
    for step_idx in range(1, int(predict_strips) + 1):
        points.append(boundary + float(step_idx) * outward)
    return np.stack(points, axis=1 if _direction_axis(direction) == 1 else 0)


def _window_ranges(length: int, window_length: int, overlap: int) -> list[ExtensionWindow]:
    if int(length) <= 0:
        return []
    window_length = max(1, min(int(window_length), int(length)))
    overlap = max(0, min(int(overlap), window_length - 1))
    stride = max(1, window_length - overlap)
    windows = []
    cursor = 0
    while cursor < length:
        end = min(length, cursor + window_length)
        start = max(0, end - window_length)
        if windows and start <= windows[-1].start and end <= windows[-1].end:
            break
        windows.append(ExtensionWindow(start=start, end=end))
        if end >= length:
            break
        cursor += stride
    return windows


def _crop_min_corner_for_points(points_zyx: np.ndarray, crop_size: tuple[int, int, int], *, margin: float = 8.0) -> np.ndarray | None:
    finite = np.isfinite(points_zyx).all(axis=-1)
    if not bool(np.any(finite)):
        return None
    valid_points = np.asarray(points_zyx[finite], dtype=np.float32)
    crop = np.asarray(crop_size, dtype=np.float32)

    for _trim_pass in range(3):
        low = valid_points.min(axis=0) - float(margin)
        high = valid_points.max(axis=0) + float(margin)
        extent = high - low
        if not np.any(extent >= crop):
            break
        over = extent - crop
        worst_axis = int(np.argmax(over))
        if over[worst_axis] <= 0:
            break
        axis_vals = valid_points[:, worst_axis]
        med = float(np.median(axis_vals))
        dist = np.abs(axis_vals - med)
        keep = dist <= float(crop[worst_axis]) * 0.45
        if keep.sum() < max(3, int(len(valid_points) * 0.5)):
            return None
        valid_points = valid_points[keep]
    else:
        low = valid_points.min(axis=0) - float(margin)
        high = valid_points.max(axis=0) + float(margin)
        extent = high - low
        if np.any(extent >= crop):
            return None

    center = 0.5 * (low + high)
    min_corner = np.floor(center - 0.5 * crop).astype(np.int64)
    return min_corner


def _score_direction(grid_zyx: np.ndarray, direction: str, *, prompt_strips: int, predict_strips: int, crop_size: tuple[int, int, int]) -> float:
    axis = _direction_axis(direction)
    axis_size = int(grid_zyx.shape[axis])
    if axis_size < int(prompt_strips) + 1:
        return -1e12
    if direction == "left":
        prompt_grid = grid_zyx[:, -int(prompt_strips):, :]
    elif direction == "right":
        prompt_grid = grid_zyx[:, :int(prompt_strips), :]
    elif direction == "up":
        prompt_grid = grid_zyx[-int(prompt_strips):, :, :]
    else:
        prompt_grid = grid_zyx[:int(prompt_strips), :, :]
    boundary, _ = _boundary_from_prompt(prompt_grid, direction)
    valid_count = int(np.isfinite(boundary).all(axis=-1).sum())
    if valid_count <= 1:
        return -1e12
    predicted = _estimate_extension_points(prompt_grid, direction, predict_strips)
    envelope_points = np.concatenate([prompt_grid.reshape(-1, 3), predicted.reshape(-1, 3)], axis=0)
    min_corner = _crop_min_corner_for_points(envelope_points, crop_size)
    if min_corner is None:
        return -1e6 + float(valid_count)
    outward = boundary - _boundary_from_prompt(prompt_grid, direction)[1]
    step_norm = float(np.nanmean(np.linalg.norm(outward, axis=-1)))
    return float(valid_count) + 0.01 * step_norm


def _mean_outward_world_z(grid_zyx: np.ndarray, direction: str, *, prompt_strips: int) -> float | None:
    prompt_grid = _extract_prompt_window(
        grid_zyx,
        direction,
        prompt_strips=int(prompt_strips),
        window=ExtensionWindow(
            0,
            int(grid_zyx.shape[0] if direction in {"left", "right"} else grid_zyx.shape[1]),
        ),
    )
    boundary, interior = _boundary_from_prompt(prompt_grid, direction)
    valid = np.isfinite(boundary).all(axis=-1) & np.isfinite(interior).all(axis=-1)
    if not bool(np.any(valid)):
        return None
    outward = boundary[valid] - interior[valid]
    return float(outward[:, 0].mean())


def resolve_growth_direction(
    grid_zyx: np.ndarray,
    *,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    requested_direction: str | None = None,
) -> tuple[str, np.ndarray]:
    requested = None if requested_direction in {None, "", "auto"} else str(requested_direction)
    if requested is None:
        smoothed = grid_zyx.copy()
        for direction in LATTICE_GROW_DIRECTIONS:
            smoothed = _smooth_frontier(smoothed, direction)
        return choose_growth_direction(
            smoothed,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
            override=None,
        ), smoothed

    if requested in LATTICE_GROW_DIRECTIONS:
        smoothed = _smooth_frontier(grid_zyx.copy(), requested)
        return choose_growth_direction(
            smoothed,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
            override=requested,
        ), smoothed

    if requested not in Z_PROJECTED_GROW_DIRECTIONS:
        raise ValueError(f"unsupported grow_direction {requested!r}")

    candidates: list[tuple[float, float, str, np.ndarray]] = []
    for direction in LATTICE_GROW_DIRECTIONS:
        smoothed = _smooth_frontier(grid_zyx.copy(), direction)
        score = _score_direction(
            smoothed,
            direction,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
        )
        if score <= -1e11:
            continue
        mean_outward_z = _mean_outward_world_z(smoothed, direction, prompt_strips=int(prompt_strips))
        if mean_outward_z is None:
            continue
        candidates.append((float(mean_outward_z), float(score), str(direction), smoothed))

    if not candidates:
        raise RuntimeError(f"could not resolve a valid lattice direction for {requested!r}")

    if requested == "decreasing-z":
        mean_z, _score, direction, smoothed = min(candidates, key=lambda item: (item[0], -item[1], item[2]))
    else:
        mean_z, _score, direction, smoothed = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    if not np.isfinite(mean_z):
        raise RuntimeError(f"resolved non-finite projected Z direction for {requested!r}")
    return str(direction), smoothed


def choose_growth_direction(grid_zyx: np.ndarray, *, prompt_strips: int, predict_strips: int, crop_size: tuple[int, int, int], override: str | None = None) -> str:
    directions = [str(override)] if override is not None else ["left", "right", "up", "down"]
    scores = {
        direction: _score_direction(grid_zyx, direction, prompt_strips=prompt_strips, predict_strips=predict_strips, crop_size=crop_size)
        for direction in directions
    }
    best_direction = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
    if scores[best_direction] <= -1e11:
        raise RuntimeError("could not find a valid growth direction for the provided surface")
    return best_direction


def choose_source_tifxyz(root: str | Path, *, prompt_strips: int, predict_strips: int, crop_size: tuple[int, int, int], limit: int = 16) -> Path:
    root = Path(root)
    candidates = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        raise FileNotFoundError(f"no tifxyz directories found under {root}")
    best_path = None
    best_score = -1e18
    for candidate in candidates[: int(limit)]:
        try:
            surface = read_tifxyz(candidate, load_mask=True, validate=True).use_stored_resolution()
            grid = _surface_grid_zyx(surface)
            grid, _, _ = _trim_grid_to_valid_bbox(grid, np.zeros(grid.shape[:2], dtype=np.uint8))
            score = max(
                _score_direction(grid, direction, prompt_strips=prompt_strips, predict_strips=predict_strips, crop_size=crop_size)
                for direction in ("left", "right", "up", "down")
            )
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_path = candidate
    if best_path is None:
        raise RuntimeError(f"could not select a usable tifxyz from {root}")
    return best_path


def _parse_int_list(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    values = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        parsed = int(part)
        if parsed <= 0:
            raise ValueError("batch sizes must be positive")
        values.append(parsed)
    return values


def _extract_prompt_window(grid_zyx: np.ndarray, direction: str, *, prompt_strips: int, window: ExtensionWindow) -> np.ndarray:
    if direction == "left":
        return grid_zyx[window.start:window.end, -int(prompt_strips):, :]
    if direction == "right":
        return grid_zyx[window.start:window.end, :int(prompt_strips), :]
    if direction == "up":
        return grid_zyx[-int(prompt_strips):, window.start:window.end, :]
    return grid_zyx[:int(prompt_strips), window.start:window.end, :]


def _dummy_target_grid(prompt_grid: np.ndarray, direction: str, predict_strips: int) -> np.ndarray:
    if direction in {"left", "right"}:
        return np.full((prompt_grid.shape[0], int(predict_strips), 3), np.nan, dtype=np.float32)
    return np.full((int(predict_strips), prompt_grid.shape[1], 3), np.nan, dtype=np.float32)


def _window_length(window: ExtensionWindow) -> int:
    return int(window.end) - int(window.start)


def _progress_enabled(show_progress: bool | None) -> bool:
    if show_progress is None:
        return bool(getattr(sys.stderr, "isatty", lambda: False)())
    return bool(show_progress)


def _progress_iter(iterable, *, total: int | None, desc: str, show_progress: bool | None):
    if not _progress_enabled(show_progress):
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def _normalize_planner_mode(value: str | None) -> str:
    mode = PLANNER_MODE_FAST if value is None else str(value)
    if mode not in PLANNER_MODES:
        raise ValueError(f"unsupported planner_mode {value!r}")
    return mode


def _fit_window_for_crop(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    window: ExtensionWindow,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    max_crop_fit_retries: int,
    min_window_strip_length: int = 4,
) -> FittedWindowPlan | None:
    local_prompt_strips = int(prompt_strips)
    local_predict_strips = int(predict_strips)
    local_window = ExtensionWindow(int(window.start), int(window.end))
    min_window_strip_length = max(1, min(int(min_window_strip_length), local_window.end - local_window.start))
    prompt_grid = _extract_prompt_window(grid_zyx, direction, prompt_strips=local_prompt_strips, window=local_window)
    for _retry in range(max(1, int(max_crop_fit_retries) * 8)):
        predicted_envelope = _estimate_extension_points(prompt_grid, direction, local_predict_strips)
        crop_points = np.concatenate([prompt_grid.reshape(-1, 3), predicted_envelope.reshape(-1, 3)], axis=0)
        min_corner = _crop_min_corner_for_points(crop_points, crop_size)
        if min_corner is not None:
            return FittedWindowPlan(
                window=local_window,
                prompt_grid=prompt_grid,
                min_corner=np.asarray(min_corner, dtype=np.int64),
                prompt_strips=local_prompt_strips,
                predict_strips=local_predict_strips,
            )
        window_len = local_window.end - local_window.start
        if window_len > min_window_strip_length:
            shorter = max(min_window_strip_length, window_len - 8)
            local_window = ExtensionWindow(local_window.start, local_window.start + shorter)
            prompt_grid = _extract_prompt_window(grid_zyx, direction, prompt_strips=local_prompt_strips, window=local_window)
            continue
        if local_prompt_strips > 2:
            local_prompt_strips -= 1
            prompt_grid = _extract_prompt_window(grid_zyx, direction, prompt_strips=local_prompt_strips, window=local_window)
            continue
        if local_predict_strips > 1:
            local_predict_strips -= 1
            continue
        break
    return None


def _retile_window_span(window: ExtensionWindow, *, child_width: int, child_stride: int) -> list[ExtensionWindow]:
    parent_length = _window_length(window)
    child_width = max(1, min(int(child_width), parent_length))
    child_stride = max(1, int(child_stride))
    if child_width >= parent_length:
        return [ExtensionWindow(int(window.start), int(window.end))]
    windows: list[ExtensionWindow] = []
    cursor = int(window.start)
    while cursor < int(window.end):
        child_end = min(int(window.end), cursor + child_width)
        child_start = max(int(window.start), child_end - child_width)
        candidate = ExtensionWindow(child_start, child_end)
        if windows and candidate == windows[-1]:
            break
        windows.append(candidate)
        if child_end >= int(window.end):
            break
        cursor += child_stride
    return windows


def _candidate_priority(prompt_strips: int, predict_strips: int) -> tuple[int, int]:
    return (int(predict_strips), int(prompt_strips))


def _split_candidate_window(candidate: ExtensionWindowCandidate, *, min_child_window_strip_length: int = 2) -> list[ExtensionWindowCandidate]:
    fixed_width = _window_length(candidate.window)
    if fixed_width <= int(min_child_window_strip_length):
        return []
    next_width = max(int(min_child_window_strip_length), fixed_width // 2)
    next_stride = max(1, next_width // 2)
    return [
        ExtensionWindowCandidate(
            window=subwindow,
            prompt_strips=int(candidate.prompt_strips),
            predict_strips=int(candidate.predict_strips),
        )
        for subwindow in _retile_window_span(candidate.window, child_width=next_width, child_stride=next_stride)
    ]


def _dedupe_window_candidates(candidates: list[ExtensionWindowCandidate]) -> list[ExtensionWindowCandidate]:
    deduped: dict[tuple[int, int], ExtensionWindowCandidate] = {}
    for candidate in candidates:
        key = (int(candidate.window.start), int(candidate.window.end))
        existing = deduped.get(key)
        if existing is None or _candidate_priority(candidate.prompt_strips, candidate.predict_strips) > _candidate_priority(
            existing.prompt_strips,
            existing.predict_strips,
        ):
            deduped[key] = candidate
    return sorted(deduped.values(), key=lambda item: (int(item.window.start), int(item.window.end)))


def _dedupe_fitted_window_plans(plans: list[FittedWindowPlan]) -> list[FittedWindowPlan]:
    deduped: dict[tuple[int, int], FittedWindowPlan] = {}
    for plan in plans:
        key = (int(plan.window.start), int(plan.window.end))
        existing = deduped.get(key)
        if existing is None or _candidate_priority(plan.prompt_strips, plan.predict_strips) > _candidate_priority(
            existing.prompt_strips,
            existing.predict_strips,
        ):
            deduped[key] = plan
    return sorted(deduped.values(), key=lambda item: (int(item.window.start), int(item.window.end)))


def _fit_child_candidate_recursive(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    candidate: ExtensionWindowCandidate,
    crop_size: tuple[int, int, int],
    max_crop_fit_retries: int,
    min_child_window_strip_length: int = 2,
) -> tuple[list[FittedWindowPlan], int]:
    fixed_width = _window_length(candidate.window)
    fitted_plan = _fit_window_for_crop(
        grid_zyx,
        direction=direction,
        window=candidate.window,
        prompt_strips=int(candidate.prompt_strips),
        predict_strips=int(candidate.predict_strips),
        crop_size=crop_size,
        max_crop_fit_retries=max_crop_fit_retries,
        min_window_strip_length=fixed_width,
    )
    if fitted_plan is not None:
        return [fitted_plan], 0
    if fixed_width <= int(min_child_window_strip_length):
        return [], 1
    fitted_plans: list[FittedWindowPlan] = []
    failed_leaf_count = 0
    for subcandidate in _split_candidate_window(candidate, min_child_window_strip_length=int(min_child_window_strip_length)):
        subplans, subfailures = _fit_child_candidate_recursive(
            grid_zyx,
            direction=direction,
            candidate=subcandidate,
            crop_size=crop_size,
            max_crop_fit_retries=max_crop_fit_retries,
            min_child_window_strip_length=int(min_child_window_strip_length),
        )
        fitted_plans.extend(subplans)
        failed_leaf_count += int(subfailures)
    return _dedupe_fitted_window_plans(fitted_plans), int(failed_leaf_count)


def _shard_fitted_window_plans(
    fitted_window_plans: list[FittedWindowPlan],
    *,
    runtime: _DistributedInferRuntime,
    shard_mode: str = "strided",
) -> list[tuple[int, FittedWindowPlan]]:
    if shard_mode != "strided":
        raise ValueError(f"unsupported distributed shard mode {shard_mode!r}")
    indexed = list(enumerate(fitted_window_plans))
    if not runtime.is_distributed:
        return indexed
    return [(global_index, plan) for global_index, plan in indexed if (int(global_index) % int(runtime.world_size)) == int(runtime.rank)]


def _plan_extension_windows(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    window_strip_length: int,
    window_overlap: int,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    max_crop_fit_retries: int,
    progress_desc: str | None = None,
    show_progress: bool | None = None,
) -> tuple[list[FittedWindowPlan], dict[str, int]]:
    frontier_length = _current_frontier_length(grid_zyx, direction)
    parent_windows = _window_ranges(frontier_length, int(window_strip_length), int(window_overlap))
    raw_child_candidates: list[ExtensionWindowCandidate] = []
    parent_iter = _progress_iter(
        parent_windows,
        total=len(parent_windows),
        desc=progress_desc or "planning windows",
        show_progress=show_progress,
    )
    for parent_window in parent_iter:
        parent_candidate = ExtensionWindowCandidate(
            window=parent_window,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
        )
        parent_fit = _fit_window_for_crop(
            grid_zyx,
            direction=direction,
            window=parent_window,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
            max_crop_fit_retries=max_crop_fit_retries,
            min_window_strip_length=4,
        )
        if parent_fit is None:
            raw_child_candidates.extend(_split_candidate_window(parent_candidate, min_child_window_strip_length=2))
            continue
        fit_width = _window_length(parent_fit.window)
        child_stride = max(1, fit_width // 2)
        child_windows = _retile_window_span(parent_window, child_width=fit_width, child_stride=child_stride)
        raw_child_candidates.extend(
            [
                ExtensionWindowCandidate(
                    window=child_window,
                    prompt_strips=int(parent_fit.prompt_strips),
                    predict_strips=int(parent_fit.predict_strips),
                )
                for child_window in child_windows
            ]
        )
    deduped_candidates = _dedupe_window_candidates(raw_child_candidates)
    fitted_plans: list[FittedWindowPlan] = []
    failed_leaf_count = 0
    for candidate in deduped_candidates:
        child_plans, child_failures = _fit_child_candidate_recursive(
            grid_zyx,
            direction=direction,
            candidate=candidate,
            crop_size=crop_size,
            max_crop_fit_retries=max_crop_fit_retries,
            min_child_window_strip_length=2,
        )
        fitted_plans.extend(child_plans)
        failed_leaf_count += int(child_failures)
    fitted_plans = _dedupe_fitted_window_plans(fitted_plans)
    return fitted_plans, {
        "parent_window_count": int(len(parent_windows)),
        "child_window_count": int(len(raw_child_candidates)),
        "deduped_child_window_count": int(len(deduped_candidates)),
        "crop_fit_failed_count": int(failed_leaf_count),
    }


def _boundary_first_prompt_grid(prompt_grid: np.ndarray, direction: str) -> tuple[np.ndarray, int]:
    if direction == "up":
        return prompt_grid[::-1, :, :].copy(), 0
    if direction == "down":
        return prompt_grid.copy(), 0
    if direction == "left":
        return prompt_grid[:, ::-1, :].copy(), 1
    if direction == "right":
        return prompt_grid.copy(), 1
    raise ValueError(f"unsupported direction {direction!r}")


def _frontier_boundary_and_interior_from_grid(grid_zyx: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(grid_zyx).all(axis=-1)
    if direction in {"up", "down"}:
        frontier_length = int(grid_zyx.shape[1])
        boundary = np.full((frontier_length, 3), np.nan, dtype=np.float32)
        interior = np.full((frontier_length, 3), np.nan, dtype=np.float32)
        for frontier_idx in range(frontier_length):
            rows = np.where(valid[:, frontier_idx])[0]
            if direction == "down":
                if len(rows) > 0:
                    boundary[frontier_idx] = grid_zyx[int(rows[0]), frontier_idx]
                if len(rows) > 1:
                    interior[frontier_idx] = grid_zyx[int(rows[1]), frontier_idx]
            else:
                if len(rows) > 0:
                    boundary[frontier_idx] = grid_zyx[int(rows[-1]), frontier_idx]
                if len(rows) > 1:
                    interior[frontier_idx] = grid_zyx[int(rows[-2]), frontier_idx]
        return boundary, interior
    frontier_length = int(grid_zyx.shape[0])
    boundary = np.full((frontier_length, 3), np.nan, dtype=np.float32)
    interior = np.full((frontier_length, 3), np.nan, dtype=np.float32)
    for frontier_idx in range(frontier_length):
        cols = np.where(valid[frontier_idx, :])[0]
        if direction == "right":
            if len(cols) > 0:
                boundary[frontier_idx] = grid_zyx[frontier_idx, int(cols[-1])]
            if len(cols) > 1:
                interior[frontier_idx] = grid_zyx[frontier_idx, int(cols[-2])]
        else:
            if len(cols) > 0:
                boundary[frontier_idx] = grid_zyx[frontier_idx, int(cols[0])]
            if len(cols) > 1:
                interior[frontier_idx] = grid_zyx[frontier_idx, int(cols[1])]
    return boundary, interior


def _smooth_frontier_scalar_field(values: np.ndarray, valid_mask: np.ndarray, *, radius: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    output = values.copy()
    if int(valid_mask.sum()) <= 0:
        return output
    for idx in range(int(values.shape[0])):
        lo = max(0, idx - int(radius))
        hi = min(int(values.shape[0]), idx + int(radius) + 1)
        local_valid = valid_mask[lo:hi]
        if not bool(np.any(local_valid)):
            continue
        offsets = np.arange(lo, hi, dtype=np.float32) - float(idx)
        weights = np.exp(-np.square(offsets / max(1.0, float(radius)))).astype(np.float32)
        weights = weights[local_valid]
        local_values = values[lo:hi][local_valid]
        output[idx] = float((local_values * weights).sum() / max(1e-8, float(weights.sum())))
    return output


def _smooth_frontier_vector_field(vectors: np.ndarray, valid_mask: np.ndarray, *, radius: int) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    output = vectors.copy()
    if int(valid_mask.sum()) <= 0:
        return output
    for idx in range(int(vectors.shape[0])):
        lo = max(0, idx - int(radius))
        hi = min(int(vectors.shape[0]), idx + int(radius) + 1)
        local_valid = valid_mask[lo:hi]
        if not bool(np.any(local_valid)):
            continue
        offsets = np.arange(lo, hi, dtype=np.float32) - float(idx)
        weights = np.exp(-np.square(offsets / max(1.0, float(radius)))).astype(np.float32)
        local_vectors = vectors[lo:hi][local_valid].copy()
        local_weights = weights[local_valid]
        reference = vectors[idx]
        if not np.isfinite(reference).all() or float(np.linalg.norm(reference)) <= 1e-8:
            reference = local_vectors[0]
        for local_idx in range(local_vectors.shape[0]):
            if float(np.dot(local_vectors[local_idx], reference)) < 0.0:
                local_vectors[local_idx] = -local_vectors[local_idx]
        combined = (local_vectors * local_weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(combined))
        if norm <= 1e-8:
            continue
        output[idx] = (combined / norm).astype(np.float32)
    return output


def _nearest_valid_frontier_index(valid_mask: np.ndarray, idx: int) -> int | None:
    valid_positions = np.where(np.asarray(valid_mask, dtype=bool))[0]
    if len(valid_positions) <= 0:
        return None
    return int(valid_positions[np.argmin(np.abs(valid_positions - int(idx)))])


def _weighted_covariance(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    if int(valid.sum()) <= 0:
        return np.zeros(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    points = points[valid]
    weights = weights[valid]
    normalized = weights / max(float(weights.sum()), 1e-8)
    mean = (points * normalized[:, None]).sum(axis=0)
    centered = points - mean[None, :]
    cov = (centered * normalized[:, None]).T @ centered
    if not np.isfinite(cov).all():
        return mean.astype(np.float32), np.eye(3, dtype=np.float32)
    return mean.astype(np.float32), cov.astype(np.float32)


def _weighted_pca_basis(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, cov = _weighted_covariance(points, weights)
    try:
        eigvals, eigvecs = np.linalg.eigh(np.asarray(cov, dtype=np.float64))
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.eye(3, dtype=np.float32)
    order = np.argsort(eigvals)
    eigvecs = eigvecs[:, order]
    eigvals = eigvals[order]
    return eigvals.astype(np.float32), eigvecs.astype(np.float32)


def _weighted_principal_direction(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    _, eigvecs = _weighted_pca_basis(points, weights)
    direction = eigvecs[:, -1]
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (direction / norm).astype(np.float32)


def _planner_boundary_and_raw_vectors(prompt_grid: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oriented, continuation_axis = _boundary_first_prompt_grid(prompt_grid, direction)
    if continuation_axis == 0:
        boundary = oriented[0, :, :]
        deltas = oriented[:-1, :, :] - oriented[1:, :, :]
        depth_weights = np.exp(-np.arange(max(1, deltas.shape[0]), dtype=np.float32) / max(1.0, float(max(1, deltas.shape[0] - 1))))
        valid = np.isfinite(deltas).all(axis=-1)
        weighted = np.where(valid[..., None], deltas, 0.0)
        raw_vectors = np.tensordot(depth_weights[: deltas.shape[0]], weighted, axes=(0, 0))
        denom = np.tensordot(depth_weights[: deltas.shape[0]], valid.astype(np.float32), axes=(0, 0))
    else:
        boundary = oriented[:, 0, :]
        deltas = oriented[:, :-1, :] - oriented[:, 1:, :]
        depth_weights = np.exp(-np.arange(max(1, deltas.shape[1]), dtype=np.float32) / max(1.0, float(max(1, deltas.shape[1] - 1))))
        valid = np.isfinite(deltas).all(axis=-1)
        weighted = np.where(valid[..., None], deltas, 0.0)
        raw_vectors = np.tensordot(weighted, depth_weights[: deltas.shape[1]], axes=([1], [0]))
        denom = np.tensordot(valid.astype(np.float32), depth_weights[: deltas.shape[1]], axes=([1], [0]))
    denom = np.maximum(np.asarray(denom, dtype=np.float32), 1e-6)
    raw_vectors = np.asarray(raw_vectors, dtype=np.float32) / denom[..., None]
    raw_vectors = np.where(np.isfinite(raw_vectors), raw_vectors, 0.0)
    step_lengths = np.linalg.norm(raw_vectors, axis=-1).astype(np.float32)
    return boundary.astype(np.float32), raw_vectors.astype(np.float32), step_lengths.astype(np.float32)


def _surface_continuation_field(prompt_grid: np.ndarray, direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oriented, continuation_axis = _boundary_first_prompt_grid(prompt_grid, direction)
    boundary, raw_vectors, step_lengths = _planner_boundary_and_raw_vectors(prompt_grid, direction)
    frontier_len = int(boundary.shape[0])
    if frontier_len <= 0:
        return boundary, raw_vectors, step_lengths
    radius = min(6, max(2, frontier_len // 32))
    if continuation_axis == 0:
        depth_count = int(oriented.shape[0])
        depth_weights = np.exp(-np.arange(depth_count, dtype=np.float32) / max(1.0, float(max(1, depth_count - 1))))
    else:
        depth_count = int(oriented.shape[1])
        depth_weights = np.exp(-np.arange(depth_count, dtype=np.float32) / max(1.0, float(max(1, depth_count - 1))))
    direction_vectors = np.zeros_like(boundary, dtype=np.float32)
    positive_steps = step_lengths[step_lengths > 1e-6]
    default_step = float(np.median(positive_steps)) if positive_steps.size > 0 else 1.0
    adjusted_steps = step_lengths.copy()
    for idx in range(frontier_len):
        lo = max(0, idx - radius)
        hi = min(frontier_len, idx + radius + 1)
        frontier_offsets = np.arange(lo, hi, dtype=np.float32) - float(idx)
        frontier_weights = np.exp(-np.square(frontier_offsets / max(1.0, float(radius) / 2.0))).astype(np.float32)
        if continuation_axis == 0:
            local_patch = oriented[:, lo:hi, :]
            point_weights = (depth_weights[:, None] * frontier_weights[None, :]).reshape(-1)
            points = local_patch.reshape(-1, 3)
        else:
            local_patch = oriented[lo:hi, :, :]
            point_weights = (frontier_weights[:, None] * depth_weights[None, :]).reshape(-1)
            points = local_patch.reshape(-1, 3)
        valid_points = np.isfinite(points).all(axis=1) & np.isfinite(point_weights) & (point_weights > 0.0)
        if int(valid_points.sum()) < 3:
            direction_vectors[idx] = raw_vectors[idx]
            if adjusted_steps[idx] <= 1e-6:
                adjusted_steps[idx] = float(default_step)
            continue
        _, eigvecs = _weighted_pca_basis(points[valid_points], point_weights[valid_points])
        normal = eigvecs[:, 0]
        boundary_patch = boundary[lo:hi]
        boundary_valid = np.isfinite(boundary_patch).all(axis=1)
        if int(boundary_valid.sum()) >= 2:
            tangent = _weighted_principal_direction(boundary_patch[boundary_valid], frontier_weights[boundary_valid])
        else:
            tangent = raw_vectors[idx]
        tangent = tangent - normal * float(np.dot(normal, tangent))
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-8:
            tangent = raw_vectors[idx]
            tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm <= 1e-8:
            tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            tangent_norm = 1.0
        tangent = (tangent / tangent_norm).astype(np.float32)
        continuation = np.cross(normal, tangent).astype(np.float32)
        continuation_norm = float(np.linalg.norm(continuation))
        if continuation_norm <= 1e-8:
            continuation = raw_vectors[idx]
            continuation_norm = float(np.linalg.norm(continuation))
        if continuation_norm <= 1e-8:
            continuation = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            continuation_norm = 1.0
        continuation = continuation / continuation_norm
        if float(np.dot(continuation, raw_vectors[idx])) < 0.0:
            continuation = -continuation
        direction_vectors[idx] = continuation.astype(np.float32)
        if adjusted_steps[idx] <= 1e-6:
            adjusted_steps[idx] = float(default_step)
    smooth_radius = min(4, max(1, frontier_len // 48))
    smoothed_vectors = np.zeros_like(direction_vectors, dtype=np.float32)
    smoothed_steps = adjusted_steps.copy()
    for idx in range(frontier_len):
        lo = max(0, idx - smooth_radius)
        hi = min(frontier_len, idx + smooth_radius + 1)
        local_vectors = direction_vectors[lo:hi].copy()
        local_steps = adjusted_steps[lo:hi]
        offsets = np.arange(lo, hi, dtype=np.float32) - float(idx)
        local_weights = np.exp(-np.square(offsets / max(1.0, float(smooth_radius)))).astype(np.float32)
        reference = raw_vectors[idx]
        if float(np.linalg.norm(reference)) <= 1e-8:
            reference = direction_vectors[idx]
        for local_idx in range(local_vectors.shape[0]):
            if float(np.dot(local_vectors[local_idx], reference)) < 0.0:
                local_vectors[local_idx] = -local_vectors[local_idx]
        vector = (local_vectors * local_weights[:, None]).sum(axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            vector = direction_vectors[idx]
            norm = max(1e-8, float(np.linalg.norm(vector)))
        vector = (vector / norm).astype(np.float32)
        if float(np.dot(vector, reference)) < 0.0:
            vector = -vector
        smoothed_vectors[idx] = vector
        smoothed_steps[idx] = float((local_steps * local_weights).sum() / max(1e-8, float(local_weights.sum())))
    return boundary.astype(np.float32), smoothed_vectors.astype(np.float32), smoothed_steps.astype(np.float32)


def _geometric_gap_fill_extension_field(
    working_grid: np.ndarray,
    *,
    direction: str,
    predict_strips: int,
) -> np.ndarray:
    boundary, interior = _frontier_boundary_and_interior_from_grid(working_grid, direction)
    boundary_valid = np.isfinite(boundary).all(axis=-1)
    interior_valid = np.isfinite(interior).all(axis=-1)
    vec_valid = boundary_valid & interior_valid
    raw_vectors = np.where(vec_valid[:, None], boundary - interior, 0.0).astype(np.float32)
    step_lengths = np.linalg.norm(raw_vectors, axis=-1).astype(np.float32)
    positive = step_lengths > 1e-6
    default_step = float(np.median(step_lengths[positive])) if bool(np.any(positive)) else 1.0
    step_lengths = np.where(positive, step_lengths, default_step).astype(np.float32)
    vector_norm = np.linalg.norm(raw_vectors, axis=-1, keepdims=True).astype(np.float32)
    unit_vectors = np.divide(
        raw_vectors,
        np.maximum(vector_norm, 1e-6),
        out=np.zeros_like(raw_vectors, dtype=np.float32),
        where=np.maximum(vector_norm, 1e-6) > 0.0,
    )
    smoothed_boundary = np.where(boundary_valid[:, None], boundary, 0.0).astype(np.float32)
    for axis in range(3):
        smoothed_boundary[:, axis] = _smooth_frontier_scalar_field(smoothed_boundary[:, axis], boundary_valid, radius=8)
    smoothed_vectors = _smooth_frontier_vector_field(unit_vectors, vec_valid & (np.linalg.norm(unit_vectors, axis=-1) > 1e-6), radius=8)
    smoothed_steps = _smooth_frontier_scalar_field(step_lengths, vec_valid, radius=8)
    for idx in range(int(smoothed_boundary.shape[0])):
        if not boundary_valid[idx]:
            nearest = _nearest_valid_frontier_index(boundary_valid, idx)
            if nearest is not None:
                smoothed_boundary[idx] = smoothed_boundary[nearest]
        vector_valid = float(np.linalg.norm(smoothed_vectors[idx])) > 1e-6 and np.isfinite(smoothed_vectors[idx]).all()
        if not vector_valid:
            nearest = _nearest_valid_frontier_index(vec_valid, idx)
            if nearest is not None:
                smoothed_vectors[idx] = smoothed_vectors[nearest]
                smoothed_steps[idx] = smoothed_steps[nearest]
            else:
                smoothed_vectors[idx] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                smoothed_steps[idx] = float(default_step)
    smoothed_steps = np.where(np.isfinite(smoothed_steps) & (smoothed_steps > 1e-6), smoothed_steps, default_step).astype(np.float32)
    field_points = []
    for step_idx in range(1, int(predict_strips) + 1):
        field_points.append(smoothed_boundary + float(step_idx) * smoothed_steps[:, None] * smoothed_vectors)
    if direction in {"up", "down"}:
        return np.stack(field_points, axis=0).astype(np.float32)
    return np.stack(field_points, axis=1).astype(np.float32)


def _apply_geometric_gap_fill(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    working_grid: np.ndarray,
    direction: str,
) -> tuple[int, int]:
    if direction in {"up", "down"}:
        predict_strips = int(counts.shape[0])
        frontier_mask = np.asarray(counts > 0).any(axis=0)
        field = _geometric_gap_fill_extension_field(working_grid, direction=direction, predict_strips=predict_strips)
        missing = counts <= 0
    else:
        predict_strips = int(counts.shape[1])
        frontier_mask = np.asarray(counts > 0).any(axis=1)
        field = _geometric_gap_fill_extension_field(working_grid, direction=direction, predict_strips=predict_strips)
        missing = counts <= 0
    valid = np.isfinite(field).all(axis=-1) & missing
    if not bool(np.any(valid)):
        return 0, 0
    sums[valid] = field[valid].astype(np.float64)
    counts[valid] = 1
    filled_frontier_mask = np.asarray(counts > 0).any(axis=0 if direction in {"up", "down"} else 1)
    geometric_frontier_count = int(np.logical_and(filled_frontier_mask, ~frontier_mask).sum())
    geometric_vertex_count = int(valid.sum())
    return geometric_vertex_count, geometric_frontier_count


def _estimate_extension_points_planner(
    prompt_grid: np.ndarray,
    direction: str,
    predict_strips: int,
    *,
    planner_surrogate: str = "raw",
) -> np.ndarray:
    if str(planner_surrogate) == "raw":
        return _estimate_extension_points(prompt_grid, direction, predict_strips)
    boundary, continuation_vectors, step_lengths = _surface_continuation_field(prompt_grid, direction)
    points = []
    for step_idx in range(1, int(predict_strips) + 1):
        points.append(boundary + (float(step_idx) * step_lengths[:, None] * continuation_vectors))
    axis = 1 if _direction_axis(direction) == 1 else 0
    return np.stack(points, axis=axis).astype(np.float32)


def _frontier_mask_from_window(frontier_length: int, window: ExtensionWindow) -> np.ndarray:
    mask = np.zeros(int(frontier_length), dtype=bool)
    start = max(0, min(int(frontier_length), int(window.start)))
    end = max(0, min(int(frontier_length), int(window.end)))
    if end > start:
        mask[start:end] = True
    return mask


def _true_spans_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if value and start is None:
            start = idx
        elif (not value) and start is not None:
            spans.append((int(start), int(idx)))
            start = None
    return spans


def _window_ranges_for_intervals(
    frontier_length: int,
    window_length: int,
    overlap: int,
    frontier_intervals: list[tuple[int, int]] | None = None,
) -> list[tuple[tuple[int, int] | None, ExtensionWindow]]:
    if frontier_intervals is None:
        return [(None, window) for window in _window_ranges(frontier_length, window_length, overlap)]
    windows: list[tuple[tuple[int, int] | None, ExtensionWindow]] = []
    for interval in frontier_intervals:
        interval_start, interval_end = (int(interval[0]), int(interval[1]))
        local_length = max(0, int(interval_end - interval_start))
        for local_window in _window_ranges(local_length, window_length, overlap):
            windows.append(
                (
                    (interval_start, interval_end),
                    ExtensionWindow(int(interval_start + local_window.start), int(interval_start + local_window.end)),
                )
            )
    return windows


def _coverage_first_overlap_ladder(window_strip_length: int) -> list[int]:
    window_strip_length = int(window_strip_length)
    overlaps = {
        max(0, min(window_strip_length - 1, int(round(float(window_strip_length) * fraction))))
        for fraction in (0.25, 0.5, 0.75)
    }
    return sorted(overlaps)


def _build_coverage_first_candidate_specs(
    *,
    phase: str,
    prompt_ladder: tuple[int, ...] = DEFAULT_COVERAGE_FIRST_PROMPT_LADDER,
    predict_ladder: tuple[int, ...] = DEFAULT_COVERAGE_FIRST_PREDICT_LADDER,
    window_ladder: tuple[int, ...] = DEFAULT_COVERAGE_FIRST_WINDOW_LADDER,
) -> list[PlannerCandidateSpec]:
    specs: list[PlannerCandidateSpec] = []
    for prompt_strips in prompt_ladder:
        for predict_strips in predict_ladder:
            for window_strip_length in window_ladder:
                for window_overlap in _coverage_first_overlap_ladder(window_strip_length):
                    specs.append(
                        PlannerCandidateSpec(
                            phase=str(phase),
                            prompt_strips=int(prompt_strips),
                            predict_strips=int(predict_strips),
                            window_strip_length=int(window_strip_length),
                            window_overlap=int(window_overlap),
                        )
                    )
    return specs


def _planner_candidate_spec_to_dict(spec: PlannerCandidateSpec) -> dict[str, Any]:
    return {
        "phase": str(spec.phase),
        "prompt_strips": int(spec.prompt_strips),
        "predict_strips": int(spec.predict_strips),
        "window_strip_length": int(spec.window_strip_length),
        "window_overlap": int(spec.window_overlap),
    }


def _planner_atlas_entry_to_dict(entry: PlannerAtlasEntry) -> dict[str, Any]:
    fitted = entry.fitted_plan
    return {
        "spec": _planner_candidate_spec_to_dict(entry.spec),
        "requested_window": [int(entry.requested_window.start), int(entry.requested_window.end)],
        "frontier_interval": None if entry.frontier_interval is None else [int(entry.frontier_interval[0]), int(entry.frontier_interval[1])],
        "admissible": bool(fitted is not None),
        "fitted_window": None if fitted is None else [int(fitted.window.start), int(fitted.window.end)],
        "predict_strips": None if fitted is None else int(fitted.predict_strips),
    }


def _planner_selection_record_to_dict(record: PlannerSelectionRecord) -> dict[str, Any]:
    return {
        "spec": _planner_candidate_spec_to_dict(record.spec),
        "fitted_window": [int(record.fitted_plan.window.start), int(record.fitted_plan.window.end)],
        "new_coverage": int(record.new_coverage),
        "redundant_overlap": int(record.redundant_overlap),
    }


def _fit_candidate_window_exact(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    window: ExtensionWindow,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    planner_surrogate: str = "surface_frame",
) -> FittedWindowPlan | None:
    prompt_grid = _extract_prompt_window(grid_zyx, direction, prompt_strips=int(prompt_strips), window=window)
    if prompt_grid.size == 0 or np.isfinite(prompt_grid).all(axis=-1).sum() <= 1:
        return None
    predicted_envelope = _estimate_extension_points_planner(
        prompt_grid,
        direction,
        int(predict_strips),
        planner_surrogate=str(planner_surrogate),
    )
    crop_points = np.concatenate([prompt_grid.reshape(-1, 3), predicted_envelope.reshape(-1, 3)], axis=0)
    min_corner = _crop_min_corner_for_points(crop_points, crop_size)
    if min_corner is None:
        return None
    return FittedWindowPlan(
        window=ExtensionWindow(int(window.start), int(window.end)),
        prompt_grid=prompt_grid,
        min_corner=np.asarray(min_corner, dtype=np.int64),
        prompt_strips=int(prompt_strips),
        predict_strips=int(predict_strips),
    )


def _fit_candidate_window_exact_cached(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    window: ExtensionWindow,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    fit_cache: dict[tuple[Any, ...], FittedWindowPlan | None],
    planner_surrogate: str = "surface_frame",
) -> FittedWindowPlan | None:
    key = (
        str(direction),
        int(prompt_strips),
        int(predict_strips),
        int(window.start),
        int(window.end),
        tuple(int(v) for v in crop_size),
    )
    if key not in fit_cache:
        fit_cache[key] = _fit_candidate_window_exact(
            grid_zyx,
            direction=direction,
            window=window,
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
            planner_surrogate=str(planner_surrogate),
        )
    return fit_cache[key]


def _build_admissibility_atlas(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    candidate_specs: list[PlannerCandidateSpec],
    crop_size: tuple[int, int, int],
    fit_cache: dict[tuple[Any, ...], FittedWindowPlan | None],
    frontier_intervals: list[tuple[int, int]] | None = None,
    planner_surrogate: str = "surface_frame",
) -> tuple[list[PlannerAtlasEntry], dict[str, Any]]:
    frontier_length = _current_frontier_length(grid_zyx, direction)
    entries: list[PlannerAtlasEntry] = []
    config_summaries: list[dict[str, Any]] = []
    total_requested_windows = 0
    total_admissible_windows = 0
    total_failures = 0
    for spec in candidate_specs:
        requested_windows = _window_ranges_for_intervals(
            frontier_length,
            spec.window_strip_length,
            spec.window_overlap,
            frontier_intervals=frontier_intervals,
        )
        admissible_plans: list[FittedWindowPlan] = []
        config_failures = 0
        for interval, window in requested_windows:
            fitted = _fit_candidate_window_exact_cached(
                grid_zyx,
                direction=direction,
                window=window,
                prompt_strips=spec.prompt_strips,
                predict_strips=spec.predict_strips,
                crop_size=crop_size,
                fit_cache=fit_cache,
                planner_surrogate=str(planner_surrogate),
            )
            entries.append(
                PlannerAtlasEntry(
                    spec=spec,
                    requested_window=window,
                    fitted_plan=fitted,
                    frontier_interval=interval,
                )
            )
            total_requested_windows += 1
            if fitted is None:
                config_failures += 1
                total_failures += 1
                continue
            total_admissible_windows += 1
            admissible_plans.append(fitted)
        covered_frontier, covered_spans = _frontier_span_coverage(admissible_plans, frontier_length=frontier_length)
        config_summaries.append(
            {
                "spec": _planner_candidate_spec_to_dict(spec),
                "requested_window_count": int(len(requested_windows)),
                "admissible_window_count": int(len(admissible_plans)),
                "crop_fit_failed_count": int(config_failures),
                "covered_frontier": int(covered_frontier),
                "covered_frontier_fraction": (float(covered_frontier) / float(frontier_length)) if frontier_length > 0 else 0.0,
                "covered_frontier_spans": covered_spans,
            }
        )
    return entries, {
        "frontier_length": int(frontier_length),
        "requested_window_count": int(total_requested_windows),
        "admissible_window_count": int(total_admissible_windows),
        "crop_fit_failed_count": int(total_failures),
        "candidate_specs": config_summaries,
    }


def _select_coverage_first_entries(
    atlas_entries: list[PlannerAtlasEntry],
    *,
    frontier_length: int,
    uncovered_mask: np.ndarray | None = None,
) -> tuple[list[PlannerSelectionRecord], np.ndarray]:
    remaining = [entry for entry in atlas_entries if entry.fitted_plan is not None]
    uncovered = np.ones(int(frontier_length), dtype=bool) if uncovered_mask is None else np.asarray(uncovered_mask, dtype=bool).copy()
    selected: list[PlannerSelectionRecord] = []
    while remaining:
        best_idx: int | None = None
        best_key: tuple[int, int, int, int, int] | None = None
        best_new = 0
        best_overlap = 0
        for idx, entry in enumerate(remaining):
            assert entry.fitted_plan is not None
            coverage_mask = _frontier_mask_from_window(frontier_length, entry.fitted_plan.window)
            new_coverage = int(np.logical_and(coverage_mask, uncovered).sum())
            if new_coverage <= 0:
                continue
            redundant_overlap = int(coverage_mask.sum()) - int(new_coverage)
            key = (
                int(new_coverage),
                -int(redundant_overlap),
                int(entry.spec.prompt_strips),
                int(entry.spec.predict_strips),
                int(entry.spec.window_strip_length),
            )
            if best_key is None or key > best_key:
                best_idx = int(idx)
                best_key = key
                best_new = int(new_coverage)
                best_overlap = int(redundant_overlap)
        if best_idx is None:
            break
        chosen = remaining.pop(best_idx)
        assert chosen.fitted_plan is not None
        uncovered &= ~_frontier_mask_from_window(frontier_length, chosen.fitted_plan.window)
        selected.append(
            PlannerSelectionRecord(
                spec=chosen.spec,
                fitted_plan=chosen.fitted_plan,
                new_coverage=int(best_new),
                redundant_overlap=int(best_overlap),
            )
        )
    return selected, uncovered


def _plan_extension_windows_coverage_first(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    crop_size: tuple[int, int, int],
    frontier_intervals: list[tuple[int, int]] | None = None,
) -> tuple[list[FittedWindowPlan], dict[str, Any], dict[str, Any]]:
    fit_cache: dict[tuple[Any, ...], FittedWindowPlan | None] = {}
    frontier_length = _current_frontier_length(grid_zyx, direction)
    initial_uncovered = np.ones(frontier_length, dtype=bool)
    if frontier_intervals is not None:
        initial_uncovered[:] = False
        for start, end in frontier_intervals:
            initial_uncovered[max(0, int(start)):min(frontier_length, int(end))] = True
    atlas_specs = _build_coverage_first_candidate_specs(phase="atlas")
    atlas_entries, atlas_summary = _build_admissibility_atlas(
        grid_zyx,
        direction=direction,
        candidate_specs=atlas_specs,
        crop_size=crop_size,
        fit_cache=fit_cache,
        frontier_intervals=frontier_intervals,
        planner_surrogate="surface_frame",
    )
    selected_records, uncovered = _select_coverage_first_entries(
        atlas_entries,
        frontier_length=frontier_length,
        uncovered_mask=initial_uncovered,
    )
    initial_gap_spans = _true_spans_from_mask(uncovered)
    gap_specs = _build_coverage_first_candidate_specs(
        phase="gap_rescue",
        prompt_ladder=DEFAULT_COVERAGE_FIRST_GAP_PROMPT_LADDER,
        predict_ladder=DEFAULT_COVERAGE_FIRST_PREDICT_LADDER,
        window_ladder=DEFAULT_COVERAGE_FIRST_GAP_WINDOW_LADDER,
    )
    rescue_entries: list[PlannerAtlasEntry] = []
    rescue_summary: dict[str, Any] | None = None
    if initial_gap_spans:
        rescue_entries, rescue_summary = _build_admissibility_atlas(
            grid_zyx,
            direction=direction,
            candidate_specs=gap_specs,
            crop_size=crop_size,
            fit_cache=fit_cache,
            frontier_intervals=initial_gap_spans,
            planner_surrogate="surface_frame",
        )
        rescue_selected, uncovered = _select_coverage_first_entries(
            rescue_entries,
            frontier_length=frontier_length,
            uncovered_mask=uncovered,
        )
        selected_records.extend(rescue_selected)
    selected_plans = _dedupe_fitted_window_plans([record.fitted_plan for record in selected_records])
    covered_frontier, covered_spans = _frontier_span_coverage(selected_plans, frontier_length=frontier_length)
    gap_spans = _true_spans_from_mask(uncovered)
    planning_stats = {
        "planner_mode": PLANNER_MODE_COVERAGE_FIRST,
        "parent_window_count": int(atlas_summary["requested_window_count"]) + int(0 if rescue_summary is None else rescue_summary["requested_window_count"]),
        "child_window_count": int(atlas_summary["admissible_window_count"]) + int(0 if rescue_summary is None else rescue_summary["admissible_window_count"]),
        "deduped_child_window_count": int(len(selected_plans)),
        "crop_fit_failed_count": int(atlas_summary["crop_fit_failed_count"]) + int(0 if rescue_summary is None else rescue_summary["crop_fit_failed_count"]),
        "selected_candidate_count": int(len(selected_records)),
        "covered_frontier": int(covered_frontier),
        "covered_frontier_fraction": (float(covered_frontier) / float(frontier_length)) if frontier_length > 0 else 0.0,
        "gap_spans": gap_spans,
    }
    diagnostics = {
        "planner_mode": PLANNER_MODE_COVERAGE_FIRST,
        "frontier_length": int(frontier_length),
        "requested_direction": str(direction),
        "atlas": {
            **atlas_summary,
            "entries": [_planner_atlas_entry_to_dict(entry) for entry in atlas_entries],
        },
        "selection": {
            "selected": [_planner_selection_record_to_dict(record) for record in selected_records],
            "covered_frontier": int(covered_frontier),
            "covered_frontier_fraction": (float(covered_frontier) / float(frontier_length)) if frontier_length > 0 else 0.0,
            "covered_frontier_spans": covered_spans,
            "gap_spans": gap_spans,
        },
        "gap_rescue": None if rescue_summary is None else {
            **rescue_summary,
            "entries": [_planner_atlas_entry_to_dict(entry) for entry in rescue_entries],
        },
    }
    return selected_plans, planning_stats, diagnostics


def _plan_extension_windows_for_mode(
    grid_zyx: np.ndarray,
    *,
    direction: str,
    planner_mode: str,
    window_strip_length: int,
    window_overlap: int,
    prompt_strips: int,
    predict_strips: int,
    crop_size: tuple[int, int, int],
    max_crop_fit_retries: int,
    frontier_intervals: list[tuple[int, int]] | None = None,
    progress_desc: str | None = None,
    show_progress: bool | None = None,
) -> tuple[list[FittedWindowPlan], dict[str, Any], dict[str, Any] | None]:
    planner_mode = _normalize_planner_mode(planner_mode)
    if planner_mode == PLANNER_MODE_FAST:
        plans, stats = _plan_extension_windows(
            grid_zyx,
            direction=direction,
            window_strip_length=int(window_strip_length),
            window_overlap=int(window_overlap),
            prompt_strips=int(prompt_strips),
            predict_strips=int(predict_strips),
            crop_size=crop_size,
            max_crop_fit_retries=int(max_crop_fit_retries),
            progress_desc=progress_desc,
            show_progress=show_progress,
        )
        return plans, {**stats, "planner_mode": planner_mode}, None
    plans, stats, diagnostics = _plan_extension_windows_coverage_first(
        grid_zyx,
        direction=direction,
        crop_size=crop_size,
        frontier_intervals=frontier_intervals,
    )
    return plans, stats, diagnostics


def build_extension_sample(
    *,
    prompt_grid_world: np.ndarray,
    direction: str,
    min_corner: np.ndarray,
    crop_size: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    offset_num_bins: tuple[int, int, int],
    frontier_band_width: int,
    predict_strips: int,
    volume_crop: np.ndarray,
    wrap_metadata: dict[str, Any],
) -> dict:
    min_corner = np.asarray(min_corner, dtype=np.float32)
    prompt_grid_local = np.asarray(prompt_grid_world, dtype=np.float32) - min_corner.reshape(1, 1, 3)
    dummy_target_local = _dummy_target_grid(prompt_grid_local, direction, predict_strips)
    frontier_band_width = int(prompt_grid_local.shape[1] if direction in {"left", "right"} else prompt_grid_local.shape[0])
    serialized = serialize_split_conditioning_example(
        cond_zyxs_local=prompt_grid_local,
        masked_zyxs_local=dummy_target_local,
        direction=direction,
        volume_shape=crop_size,
        patch_size=patch_size,
        offset_num_bins=offset_num_bins,
        frontier_band_width=frontier_band_width,
    )
    world_bbox = (
        float(min_corner[0]),
        float(min_corner[0] + crop_size[0]),
        float(min_corner[1]),
        float(min_corner[1] + crop_size[1]),
        float(min_corner[2]),
        float(min_corner[2] + crop_size[2]),
    )
    return {
        "volume": torch.from_numpy(np.asarray(volume_crop, dtype=np.float32)[None, ...]),
        "vol_tokens": None,
        "prompt_tokens": {
            "coarse_ids": torch.from_numpy(serialized["prompt_tokens"]["coarse_ids"]).to(torch.long),
            "offset_bins": torch.from_numpy(serialized["prompt_tokens"]["offset_bins"]).to(torch.long),
            "xyz": torch.from_numpy(serialized["prompt_tokens"]["xyz"]).to(torch.float32),
            "strip_positions": torch.from_numpy(serialized["prompt_tokens"]["strip_positions"]).to(torch.long),
            "strip_coords": torch.from_numpy(serialized["prompt_tokens"]["strip_coords"]).to(torch.float32),
            "valid_mask": torch.from_numpy(serialized["prompt_tokens"]["valid_mask"]).to(torch.bool),
        },
        "prompt_meta": {
            **serialized["prompt_meta"],
            "conditioning_shape": tuple(int(v) for v in serialized["conditioning_grid_local"].shape[:2]),
            "surface_sampling_mode": "stored",
            "spatial_augmented": False,
            "spatial_mirror_axes": [],
            "spatial_axis_order": [0, 1, 2],
        },
        "conditioning_grid_local": torch.from_numpy(serialized["conditioning_grid_local"]).to(torch.float32),
        "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
        "prompt_anchor_valid": torch.tensor(bool(serialized["prompt_anchor_valid"]), dtype=torch.bool),
        "prompt_grid_local": torch.from_numpy(serialized["prompt_grid_local"]).to(torch.float32),
        "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
        "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
        "target_valid_mask": torch.from_numpy(serialized["target_valid_mask"]).to(torch.bool),
        "target_invalid_mask": torch.from_numpy(~serialized["target_valid_mask"]).to(torch.bool),
        "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
        "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
        "target_bin_center_xyz": torch.from_numpy(serialized["target_bin_center_xyz"]).to(torch.float32),
        "target_strip_positions": torch.from_numpy(serialized["target_strip_positions"]).to(torch.long),
        "target_strip_coords": torch.from_numpy(serialized["target_strip_coords"]).to(torch.float32),
        "target_grid_local": torch.from_numpy(serialized["target_grid_local"]).to(torch.float32),
        "target_invalid_fraction": torch.tensor(0.0, dtype=torch.float32),
        "frontier_invalid_fraction": torch.tensor(0.0, dtype=torch.float32),
        "touches_crop_boundary": torch.tensor(False, dtype=torch.bool),
        "direction": str(direction),
        "direction_id": torch.tensor(int(serialized["direction_id"]), dtype=torch.long),
        "strip_length": torch.tensor(int(serialized["strip_length"]), dtype=torch.long),
        "num_strips": torch.tensor(int(serialized["num_strips"]), dtype=torch.long),
        "min_corner": torch.from_numpy(min_corner).to(torch.float32),
        "world_bbox": torch.tensor(world_bbox, dtype=torch.float32),
        "target_grid_shape": torch.tensor(tuple(int(v) for v in serialized["target_grid_shape"]), dtype=torch.long),
        "wrap_metadata": dict(wrap_metadata),
    }


def _initialize_extension_arrays(grid_zyx: np.ndarray, direction: str, predict_strips: int) -> tuple[np.ndarray, np.ndarray]:
    if direction in {"left", "right"}:
        extension_shape = (grid_zyx.shape[0], int(predict_strips), 3)
        seam_shape = (grid_zyx.shape[0], int(predict_strips))
    else:
        extension_shape = (int(predict_strips), grid_zyx.shape[1], 3)
        seam_shape = (int(predict_strips), grid_zyx.shape[1])
    sums = np.zeros(extension_shape, dtype=np.float64)
    counts = np.zeros(seam_shape, dtype=np.int32)
    return sums, counts


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
        elif key in {"conditioning_grid_local", "wrap_metadata", "prompt_meta", "direction", "prompt_grid_local", "target_grid_local"}:
            if key == "conditioning_grid_local":
                moved[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
            else:
                moved[key] = value
        else:
            moved[key] = value
    return moved


def _window_bucket_key(payload: ExtensionWindowPayload) -> tuple[Any, ...]:
    sample = payload.sample
    return (
        payload.direction,
        tuple(int(v) for v in payload.target_grid_shape),
        int(payload.strip_length),
        int(payload.num_strips),
        tuple(int(v) for v in sample["volume"].shape),
    )


def _iter_window_batches(payloads: list[ExtensionWindowPayload], *, window_batch_size: int) -> list[list[ExtensionWindowPayload]]:
    grouped: OrderedDict[tuple[Any, ...], list[ExtensionWindowPayload]] = OrderedDict()
    for payload in payloads:
        grouped.setdefault(_window_bucket_key(payload), []).append(payload)
    batches: list[list[ExtensionWindowPayload]] = []
    for group in grouped.values():
        for start in range(0, len(group), max(1, int(window_batch_size))):
            batches.append(group[start:start + max(1, int(window_batch_size))])
    return batches


def _model_patch_diag(model) -> float | None:
    patch_size = getattr(model, "patch_size", None)
    if patch_size is None:
        return None
    return float(np.linalg.norm(np.asarray(patch_size, dtype=np.float32)))


def _model_input_shape_array(model) -> np.ndarray | None:
    input_shape = getattr(model, "input_shape", None)
    if input_shape is None:
        return None
    return np.asarray(input_shape, dtype=np.float32)


def _supports_cached_window_inference(model) -> bool:
    raw_model = model.module if hasattr(model, "module") else model
    return all(callable(getattr(raw_model, name, None)) for name in ("init_kv_cache", "_build_input_embeddings", "step_from_encoded_cached"))


def _decode_single_step_from_outputs(
    model,
    outputs: dict,
    *,
    sample_idx: int,
    step_idx: int,
    greedy: bool,
) -> tuple[int, list[int], np.ndarray, float]:
    device = outputs["stop_logits"].device
    if str(outputs.get("coarse_prediction_mode", getattr(model, "coarse_prediction_mode", "joint_pointer"))) == "axis_factorized":
        axis_ids = {}
        for axis_name in ("z", "y", "x"):
            axis_logits = outputs["coarse_axis_logits"][axis_name][sample_idx, step_idx].float()
            axis_ids[axis_name] = int(_sample_from_logits(axis_logits, greedy=greedy).item())
        coarse_id = int(
            model._flatten_coarse_axis_ids(
                torch.tensor([[axis_ids["z"]]], dtype=torch.long, device=device),
                torch.tensor([[axis_ids["y"]]], dtype=torch.long, device=device),
                torch.tensor([[axis_ids["x"]]], dtype=torch.long, device=device),
            ).item()
        )
    else:
        coarse_logits = outputs["coarse_logits"][sample_idx, step_idx].float()
        coarse_id = int(_sample_from_logits(coarse_logits, greedy=greedy).item())
    offset_bins = []
    for axis, bins in enumerate(model.offset_num_bins):
        axis_logits = outputs["offset_logits"][sample_idx, step_idx, axis, :bins].float()
        offset_bins.append(int(_sample_from_logits(axis_logits, greedy=greedy).item()))
    offset_tensor = torch.tensor(offset_bins, dtype=torch.long, device=device).view(1, 1, 3)
    coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=device)
    bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().to(torch.float32).cpu().numpy()
    refine_residual = outputs.get("pred_refine_residual")
    if refine_residual is not None:
        res = refine_residual[sample_idx, step_idx].detach().to(torch.float32).cpu().numpy()
        patch_diag = _model_patch_diag(model)
        res_norm = float(np.linalg.norm(res))
        if patch_diag is not None and res_norm > patch_diag:
            res = res * (patch_diag / res_norm)
        sampled_xyz = bin_center_xyz + res
    else:
        sampled_xyz = bin_center_xyz
    input_shape = _model_input_shape_array(model)
    if input_shape is not None:
        crop_max = input_shape - 1e-4
        sampled_xyz = np.clip(sampled_xyz, 0.0, crop_max)
    stop_prob = float(torch.sigmoid(outputs["stop_logits"][sample_idx, step_idx].float()).item())
    return coarse_id, offset_bins, sampled_xyz.astype(np.float32, copy=False), stop_prob


def infer_extension_windows_batched(
    model,
    payloads: list[ExtensionWindowPayload],
    *,
    window_batch_size: int,
    device: torch.device,
    greedy: bool = True,
    stop_probability_threshold: float | None = None,
    progress_desc: str | None = None,
    show_progress: bool | None = None,
    fast_infer: bool = True,
    compile_infer: bool = False,
    amp_dtype: str = "bf16",
    runtime: ExtensionInferenceRuntime | None = None,
) -> tuple[list[dict[str, Any]], float, int]:
    if not payloads:
        return [], 0.0, 0
    results: list[dict[str, Any]] = []
    total_encode_decode_ms = 0.0
    peak_batch_size = 0
    runtime = runtime or ExtensionInferenceRuntime(
        model,
        device=device,
        fast_infer=fast_infer,
        compile_infer=compile_infer,
        amp_dtype=amp_dtype,
    )
    payload_batches = _iter_window_batches(payloads, window_batch_size=window_batch_size)
    batch_iter = _progress_iter(
        payload_batches,
        total=len(payload_batches),
        desc=progress_desc or "infer batches",
        show_progress=show_progress,
    )
    with runtime.inference_context():
        for payload_batch in batch_iter:
            peak_batch_size = max(peak_batch_size, len(payload_batch))
            raw_samples = [payload.sample for payload in payload_batch]
            batch = autoreg_mesh_collate(raw_samples)
            batch = _move_batch_to_device(batch, device)
            t0 = perf_counter()
            with runtime.autocast_context():
                encoded = runtime.encode_conditioning(batch["volume"], vol_tokens=batch.get("vol_tokens"))
            target_shapes = [tuple(int(v) for v in sample["target_grid_shape"].tolist()) for sample in raw_samples]
            total_vertices = [int(shape[0] * shape[1]) for shape in target_shapes]
            max_steps = max(total_vertices)
            all_target_strip_coords = []
            all_target_strip_positions = []
            for shape, direction in zip(target_shapes, batch["direction"], strict=True):
                all_target_strip_coords.append(_build_target_strip_coords(direction, shape, device=device))
                all_target_strip_positions.append(_build_target_strip_positions(direction, shape, device=device))
            generated_coarse = [[] for _ in payload_batch]
            generated_offsets = [[] for _ in payload_batch]
            generated_xyz = [[] for _ in payload_batch]
            generated_stop_probs = [[] for _ in payload_batch]
            active = [True for _ in payload_batch]
            for step_idx in range(max_steps):
                if not any(active):
                    break
                current_len = step_idx + 1
                batch_size = len(payload_batch)
                target_coarse_ids = torch.full((batch_size, current_len), -100, dtype=torch.long, device=device)
                target_offset_bins = torch.full((batch_size, current_len, 3), -100, dtype=torch.long, device=device)
                target_xyz = torch.zeros((batch_size, current_len, 3), dtype=torch.float32, device=device)
                target_strip_positions = torch.zeros((batch_size, current_len, 2), dtype=torch.long, device=device)
                target_strip_coords = torch.zeros((batch_size, current_len, 2), dtype=torch.float32, device=device)
                for batch_idx in range(batch_size):
                    history_len = min(len(generated_coarse[batch_idx]), current_len - 1)
                    if history_len > 0:
                        target_coarse_ids[batch_idx, :history_len] = torch.tensor(generated_coarse[batch_idx], dtype=torch.long, device=device)
                        target_offset_bins[batch_idx, :history_len] = torch.tensor(generated_offsets[batch_idx], dtype=torch.long, device=device)
                        target_xyz[batch_idx, :history_len] = torch.tensor(np.asarray(generated_xyz[batch_idx]), dtype=torch.float32, device=device)
                    target_strip_positions[batch_idx, :current_len] = all_target_strip_positions[batch_idx][:current_len]
                    target_strip_coords[batch_idx, :current_len] = all_target_strip_coords[batch_idx][:current_len]
                pseudo_batch = build_pseudo_inference_batch(
                    prompt_tokens=batch["prompt_tokens"],
                    prompt_anchor_xyz=batch["prompt_anchor_xyz"],
                    direction_id=batch["direction_id"],
                    direction=batch["direction"],
                    conditioning_grid_local=batch["conditioning_grid_local"],
                    strip_length=batch["strip_length"],
                    num_strips=batch["num_strips"],
                    target_coarse_ids=target_coarse_ids,
                    target_offset_bins=target_offset_bins,
                    target_xyz=target_xyz,
                    target_strip_positions=target_strip_positions,
                    target_strip_coords=target_strip_coords,
                )
                with runtime.autocast_context():
                    outputs = runtime.forward_from_encoded(
                        pseudo_batch,
                        memory_tokens=encoded["memory_tokens"],
                        memory_patch_centers=encoded["memory_patch_centers"],
                    )
                for batch_idx, is_active in enumerate(active):
                    if not is_active:
                        continue
                    if step_idx >= total_vertices[batch_idx]:
                        active[batch_idx] = False
                        continue
                    coarse_id, offset_bins, xyz, stop_prob = _decode_single_step_from_outputs(
                        model,
                        outputs,
                        sample_idx=batch_idx,
                        step_idx=current_len - 1,
                        greedy=greedy,
                    )
                    generated_coarse[batch_idx].append(coarse_id)
                    generated_offsets[batch_idx].append(offset_bins)
                    generated_xyz[batch_idx].append(xyz)
                    generated_stop_probs[batch_idx].append(stop_prob)
                    if stop_probability_threshold is not None and stop_prob >= float(stop_probability_threshold):
                        active[batch_idx] = False
                    elif len(generated_xyz[batch_idx]) >= total_vertices[batch_idx]:
                        active[batch_idx] = False
            total_encode_decode_ms += 1000.0 * (perf_counter() - t0)
            for batch_idx, payload in enumerate(payload_batch):
                predicted_xyz_local = np.asarray(generated_xyz[batch_idx], dtype=np.float32)
                padded_xyz = predicted_xyz_local
                if predicted_xyz_local.shape[0] < total_vertices[batch_idx]:
                    padded_xyz = np.full((total_vertices[batch_idx], 3), np.nan, dtype=np.float32)
                    if predicted_xyz_local.shape[0] > 0:
                        padded_xyz[: predicted_xyz_local.shape[0]] = predicted_xyz_local
                continuation_grid_local = deserialize_continuation_grid(
                    padded_xyz,
                    direction=payload.direction,
                    grid_shape=payload.target_grid_shape,
                )
                min_corner = raw_samples[batch_idx]["min_corner"].detach().cpu().numpy().astype(np.float32, copy=False)
                continuation_grid_world = continuation_grid_local.copy()
                finite = np.isfinite(continuation_grid_world).all(axis=-1)
                continuation_grid_world[finite] += min_corner
                results.append(
                    {
                        "global_index": int(payload.global_index),
                        "window": payload.window,
                        "direction": payload.direction,
                        "continuation_grid_world": continuation_grid_world,
                        "predicted_vertex_count": int(np.isfinite(continuation_grid_world).all(axis=-1).sum()),
                        "stop_count": int(sum(1 for value in generated_stop_probs[batch_idx] if stop_probability_threshold is not None and value >= float(stop_probability_threshold))),
                    }
                )
    return results, total_encode_decode_ms, peak_batch_size


def infer_extension_windows_batched_cached(
    model,
    payloads: list[ExtensionWindowPayload],
    *,
    window_batch_size: int,
    device: torch.device,
    greedy: bool = True,
    stop_probability_threshold: float | None = None,
    progress_desc: str | None = None,
    show_progress: bool | None = None,
    fast_infer: bool = True,
    compile_infer: bool = False,
    amp_dtype: str = "bf16",
    runtime: ExtensionInferenceRuntime | None = None,
) -> tuple[list[dict[str, Any]], float, int]:
    if not payloads:
        return [], 0.0, 0
    if not _supports_cached_window_inference(model):
        return infer_extension_windows_batched(
            model,
            payloads,
            window_batch_size=window_batch_size,
            device=device,
            greedy=greedy,
            stop_probability_threshold=stop_probability_threshold,
            progress_desc=progress_desc,
            show_progress=show_progress,
            fast_infer=fast_infer,
            compile_infer=compile_infer,
            amp_dtype=amp_dtype,
            runtime=runtime,
        )
    results: list[dict[str, Any]] = []
    total_encode_decode_ms = 0.0
    peak_batch_size = 0
    runtime = runtime or ExtensionInferenceRuntime(
        model, device=device, fast_infer=fast_infer, compile_infer=compile_infer, amp_dtype=amp_dtype,
    )
    payload_batches = _iter_window_batches(payloads, window_batch_size=window_batch_size)
    batch_iter = _progress_iter(
        payload_batches, total=len(payload_batches),
        desc=progress_desc or "infer batches", show_progress=show_progress,
    )
    raw_model = model.module if hasattr(model, "module") else model
    with runtime.inference_context():
        for payload_batch in batch_iter:
            batch_size = len(payload_batch)
            peak_batch_size = max(peak_batch_size, batch_size)
            raw_samples = [payload.sample for payload in payload_batch]
            batch = autoreg_mesh_collate(raw_samples)
            batch = _move_batch_to_device(batch, device)
            t0 = perf_counter()
            with runtime.autocast_context():
                encoded = runtime.encode_conditioning(batch["volume"], vol_tokens=batch.get("vol_tokens"))
            memory_tokens = encoded["memory_tokens"]
            memory_patch_centers = encoded["memory_patch_centers"]
            with runtime.autocast_context():
                _, cache = raw_model.init_kv_cache(
                    batch, memory_tokens=memory_tokens, memory_patch_centers=memory_patch_centers,
                )
            target_shapes = [tuple(int(v) for v in s["target_grid_shape"].tolist()) for s in raw_samples]
            total_vertices = [int(sh[0] * sh[1]) for sh in target_shapes]
            max_steps = max(total_vertices)
            all_strip_coords = []
            for shape, direction in zip(target_shapes, batch["direction"], strict=True):
                all_strip_coords.append(_build_target_strip_coords(direction, shape, device=device))

            prev_coarse = torch.full((batch_size, 1), -100, dtype=torch.long, device=device)
            prev_offset = torch.full((batch_size, 1, 3), -100, dtype=torch.long, device=device)
            prev_xyz = batch["prompt_anchor_xyz"].unsqueeze(1)
            prev_valid = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            ones = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

            generated_coarse = [[] for _ in range(batch_size)]
            generated_offsets = [[] for _ in range(batch_size)]
            generated_xyz = [[] for _ in range(batch_size)]
            generated_stop_probs = [[] for _ in range(batch_size)]
            active = [True] * batch_size

            for step_idx in range(max_steps):
                if not any(active):
                    break
                token_type = torch.full(
                    (batch_size, 1),
                    START_TOKEN_TYPE if step_idx == 0 else GENERATED_TOKEN_TYPE,
                    dtype=torch.long, device=device,
                )
                step_coords = torch.zeros((batch_size, 1, 2), dtype=torch.float32, device=device)
                for bi in range(batch_size):
                    if step_idx < len(all_strip_coords[bi]):
                        step_coords[bi, 0] = all_strip_coords[bi][step_idx]

                with runtime.autocast_context():
                    embedding, coords = raw_model._build_input_embeddings(
                        coarse_ids=prev_coarse,
                        offset_bins=prev_offset,
                        xyz=prev_xyz,
                        strip_coords=step_coords,
                        direction_id=batch["direction_id"],
                        token_type=token_type,
                        sequence_mask=ones,
                        geometry_valid_mask=prev_valid,
                        memory_tokens=memory_tokens,
                    )
                    outputs, cache = raw_model.step_from_encoded_cached(
                        token_embedding=embedding,
                        token_coords=coords,
                        cache=cache,
                        memory_tokens=memory_tokens,
                    )

                new_coarse_ids = []
                new_offsets = []
                new_xyz_list = []
                for bi, is_active in enumerate(active):
                    if not is_active or step_idx >= total_vertices[bi]:
                        active[bi] = False
                        new_coarse_ids.append(0)
                        new_offsets.append([0, 0, 0])
                        new_xyz_list.append(np.zeros(3, dtype=np.float32))
                        continue
                    coarse_id, offset_bins, xyz, stop_prob = _decode_single_step_from_outputs(
                        raw_model, outputs, sample_idx=bi, step_idx=0, greedy=greedy,
                    )
                    generated_coarse[bi].append(coarse_id)
                    generated_offsets[bi].append(offset_bins)
                    generated_xyz[bi].append(xyz)
                    generated_stop_probs[bi].append(stop_prob)
                    new_coarse_ids.append(coarse_id)
                    new_offsets.append(offset_bins)
                    new_xyz_list.append(xyz)
                    if stop_probability_threshold is not None and stop_prob >= float(stop_probability_threshold):
                        active[bi] = False
                    elif len(generated_xyz[bi]) >= total_vertices[bi]:
                        active[bi] = False

                prev_coarse = torch.tensor(new_coarse_ids, dtype=torch.long, device=device).unsqueeze(1)
                prev_offset = torch.tensor(new_offsets, dtype=torch.long, device=device).unsqueeze(1)
                prev_xyz = torch.tensor(np.array(new_xyz_list), dtype=torch.float32, device=device).unsqueeze(1)
                prev_valid = ones

            total_encode_decode_ms += 1000.0 * (perf_counter() - t0)
            for bi, payload in enumerate(payload_batch):
                predicted_xyz_local = np.asarray(generated_xyz[bi], dtype=np.float32)
                padded_xyz = predicted_xyz_local
                if predicted_xyz_local.shape[0] < total_vertices[bi]:
                    padded_xyz = np.full((total_vertices[bi], 3), np.nan, dtype=np.float32)
                    if predicted_xyz_local.shape[0] > 0:
                        padded_xyz[: predicted_xyz_local.shape[0]] = predicted_xyz_local
                continuation_grid_local = deserialize_continuation_grid(
                    padded_xyz, direction=payload.direction, grid_shape=payload.target_grid_shape,
                )
                min_corner = raw_samples[bi]["min_corner"].detach().cpu().numpy().astype(np.float32, copy=False)
                continuation_grid_world = continuation_grid_local.copy()
                finite = np.isfinite(continuation_grid_world).all(axis=-1)
                continuation_grid_world[finite] += min_corner
                results.append({
                    "global_index": int(payload.global_index),
                    "window": payload.window,
                    "direction": payload.direction,
                    "continuation_grid_world": continuation_grid_world,
                    "predicted_vertex_count": int(np.isfinite(continuation_grid_world).all(axis=-1).sum()),
                    "stop_count": int(sum(1 for v in generated_stop_probs[bi] if stop_probability_threshold is not None and v >= float(stop_probability_threshold))),
                })
    return results, total_encode_decode_ms, peak_batch_size


def merge_window_prediction(
    *,
    sums: np.ndarray,
    counts: np.ndarray,
    pred_grid_world: np.ndarray,
    direction: str,
    window: ExtensionWindow,
) -> None:
    if direction in {"left", "right"}:
        target = pred_grid_world
        valid = np.isfinite(target).all(axis=-1)
        row_slice = slice(int(window.start), int(window.end))
        sums_view = sums[row_slice, :target.shape[1], :]
        counts_view = counts[row_slice, :target.shape[1]]
        sums_view[valid] = sums_view[valid] + target[valid]
        counts_view[valid] = counts_view[valid] + 1
        return
    target = pred_grid_world
    valid = np.isfinite(target).all(axis=-1)
    col_slice = slice(int(window.start), int(window.end))
    sums_view = sums[:target.shape[0], col_slice, :]
    counts_view = counts[:target.shape[0], col_slice]
    sums_view[valid] = sums_view[valid] + target[valid]
    counts_view[valid] = counts_view[valid] + 1


def finalize_iteration_extension(
    *,
    sums: np.ndarray,
    counts: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    extension = np.full_like(sums, np.nan, dtype=np.float32)
    valid = counts > 0
    if np.any(valid):
        extension[valid] = (sums[valid] / counts[valid, None]).astype(np.float32)
    provenance = np.full(valid.shape, 1, dtype=np.uint8)
    if direction in {"left", "right"}:
        provenance[:, 0] = 2
    else:
        provenance[0, :] = 2
    provenance[~valid] = 255
    return extension, provenance


def _max_false_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(mask, dtype=bool).tolist():
        if value:
            current = 0
            continue
        current += 1
        best = max(best, current)
    return int(best)


def _gap_spans_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    gap_start: int | None = None
    for idx, value in enumerate(np.asarray(mask, dtype=bool).tolist() + [True]):
        if not value and gap_start is None:
            gap_start = idx
        elif value and gap_start is not None:
            spans.append((int(gap_start), int(idx)))
            gap_start = None
    return spans


def _compute_new_band_coverage_metrics(
    counts: np.ndarray,
    *,
    direction: str,
) -> tuple[float, float, int, list[tuple[int, int]], int | None]:
    valid_cells = np.asarray(counts, dtype=np.int32) > 0
    if valid_cells.size == 0:
        return 0.0, 0.0, 0, [], None
    if direction in {"left", "right"}:
        frontier_mask = valid_cells.any(axis=1)
    else:
        frontier_mask = valid_cells.any(axis=0)
    gap_spans = _gap_spans_from_mask(frontier_mask)
    return (
        float(frontier_mask.mean()) if frontier_mask.size > 0 else 0.0,
        float(valid_cells.mean()),
        _max_false_run(frontier_mask),
        gap_spans,
        int(gap_spans[0][0]) if gap_spans else None,
    )


def _flatten_gathered_window_results(gathered_payloads: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[int]]:
    flattened: list[dict[str, Any]] = []
    per_rank_counts: list[int] = []
    fallback_index = 0
    for payload in gathered_payloads:
        rank_results = list(payload.get("results", []))
        per_rank_counts.append(int(payload.get("fitted_window_count", len(rank_results))))
        for result in rank_results:
            enriched = dict(result)
            enriched.setdefault("global_index", int(fallback_index))
            fallback_index += 1
            flattened.append(enriched)
    flattened.sort(key=lambda item: int(item["global_index"]))
    return flattened, per_rank_counts


def append_extension_to_grid(
    *,
    working_grid: np.ndarray,
    working_provenance: np.ndarray,
    extension_grid: np.ndarray,
    extension_provenance: np.ndarray,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    if direction == "left":
        return (
            np.concatenate([working_grid, extension_grid], axis=1),
            np.concatenate([working_provenance, extension_provenance], axis=1),
        )
    if direction == "right":
        return (
            np.concatenate([extension_grid, working_grid], axis=1),
            np.concatenate([extension_provenance, working_provenance], axis=1),
        )
    if direction == "up":
        return (
            np.concatenate([working_grid, extension_grid], axis=0),
            np.concatenate([working_provenance, extension_provenance], axis=0),
        )
    return (
        np.concatenate([extension_grid, working_grid], axis=0),
        np.concatenate([extension_provenance, working_provenance], axis=0),
    )


def demote_previous_seam(provenance: np.ndarray) -> None:
    provenance[provenance == 2] = 1


def _current_frontier_length(grid_zyx: np.ndarray, direction: str) -> int:
    return int(grid_zyx.shape[0]) if direction in {"left", "right"} else int(grid_zyx.shape[1])


def _color_from_provenance_code(code: int) -> Color:
    if int(code) == 0:
        return ORIGINAL_COLOR
    if int(code) == 2:
        return SEAM_COLOR
    return PREDICTED_COLOR


def grid_to_colored_mesh(grid_zyx: np.ndarray, provenance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(grid_zyx).all(axis=-1)
    vertex_indices = -np.ones(valid.shape, dtype=np.int64)
    vertices = []
    colors = []
    cursor = 0
    for row_idx in range(valid.shape[0]):
        for col_idx in range(valid.shape[1]):
            if not valid[row_idx, col_idx]:
                continue
            vertex_indices[row_idx, col_idx] = cursor
            vertices.append(grid_zyx[row_idx, col_idx, ::-1])  # XYZ
            colors.append(_color_from_provenance_code(int(provenance[row_idx, col_idx])))
            cursor += 1
    faces = []
    for row_idx in range(valid.shape[0] - 1):
        for col_idx in range(valid.shape[1] - 1):
            quad_valid = valid[row_idx:row_idx + 2, col_idx:col_idx + 2]
            if not bool(np.all(quad_valid)):
                continue
            v00 = int(vertex_indices[row_idx, col_idx])
            v01 = int(vertex_indices[row_idx, col_idx + 1])
            v10 = int(vertex_indices[row_idx + 1, col_idx])
            v11 = int(vertex_indices[row_idx + 1, col_idx + 1])
            faces.append((v00, v01, v10))
            faces.append((v11, v10, v01))
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        np.asarray(colors, dtype=np.uint8),
    )


def write_colored_ply(path: str | Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {int(vertices.shape[0])}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {int(faces.shape[0])}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(vertices, colors, strict=True):
            f.write(f"{float(x)} {float(y)} {float(z)} {int(r)} {int(g)} {int(b)}\n")
        for face in faces:
            f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
    return path


def _draw_line(canvas: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: Color) -> None:
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    while True:
        if 0 <= r0 < canvas.shape[0] and 0 <= c0 < canvas.shape[1]:
            canvas[r0, c0] = np.asarray(color, dtype=np.uint8)
        if r0 == r1 and c0 == c1:
            break
        err2 = 2 * err
        if err2 > -dr:
            err -= dr
            c0 += sc
        if err2 < dc:
            err += dc
            r0 += sr


def _render_projection(grid_zyx: np.ndarray, provenance: np.ndarray, *, plane: str, size: int = 1024) -> np.ndarray:
    assert plane in {"xy", "xz", "yz"}
    valid = np.isfinite(grid_zyx).all(axis=-1)
    points = grid_zyx[valid]
    if points.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    if plane == "xy":
        coords = points[:, [1, 2]]
    elif plane == "xz":
        coords = points[:, [0, 2]]
    else:
        coords = points[:, [0, 1]]
    low = coords.min(axis=0)
    high = coords.max(axis=0)
    span = np.maximum(high - low, 1.0)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    def _project(zyx: np.ndarray) -> tuple[int, int]:
        if plane == "xy":
            uv = zyx[[1, 2]]
        elif plane == "xz":
            uv = zyx[[0, 2]]
        else:
            uv = zyx[[0, 1]]
        scaled = (uv - low) / span
        row = int(round((1.0 - float(scaled[0])) * (size - 1)))
        col = int(round(float(scaled[1]) * (size - 1)))
        return row, col

    for row_idx in range(grid_zyx.shape[0]):
        for col_idx in range(grid_zyx.shape[1] - 1):
            if not (valid[row_idx, col_idx] and valid[row_idx, col_idx + 1]):
                continue
            color = _color_from_provenance_code(max(int(provenance[row_idx, col_idx]), int(provenance[row_idx, col_idx + 1])))
            r0, c0 = _project(grid_zyx[row_idx, col_idx])
            r1, c1 = _project(grid_zyx[row_idx, col_idx + 1])
            _draw_line(canvas, r0, c0, r1, c1, color)
    for row_idx in range(grid_zyx.shape[0] - 1):
        for col_idx in range(grid_zyx.shape[1]):
            if not (valid[row_idx, col_idx] and valid[row_idx + 1, col_idx]):
                continue
            color = _color_from_provenance_code(max(int(provenance[row_idx, col_idx]), int(provenance[row_idx + 1, col_idx])))
            r0, c0 = _project(grid_zyx[row_idx, col_idx])
            r1, c1 = _project(grid_zyx[row_idx + 1, col_idx])
            _draw_line(canvas, r0, c0, r1, c1, color)
    return canvas


def _save_projection_images(grid_zyx: np.ndarray, provenance: np.ndarray, out_dir: Path) -> list[str]:
    from PIL import Image

    paths = []
    for plane in ("xy", "xz", "yz"):
        image = _render_projection(grid_zyx, provenance, plane=plane)
        path = out_dir / f"mesh_{plane}.png"
        Image.fromarray(image).save(path)
        paths.append(str(path))
    return paths


def _build_tifxyz_from_grid(grid_zyx: np.ndarray, *, uuid: str, scale: tuple[float, float]) -> Tifxyz:
    valid = np.isfinite(grid_zyx).all(axis=-1)
    safe = np.where(valid[..., None], grid_zyx, -1.0).astype(np.float32)
    return Tifxyz(
        _x=safe[..., 2],
        _y=safe[..., 1],
        _z=safe[..., 0],
        uuid=uuid,
        _scale=tuple(float(v) for v in scale),
        _mask=valid,
        resolution="stored",
    )


def _load_autoreg_model(
    *,
    dino_backbone: str,
    autoreg_checkpoint: str,
    device: torch.device,
    attention_scaling_mode: str | None = None,
) -> tuple[AutoregMeshModel, dict]:
    ckpt = torch.load(Path(autoreg_checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    cfg["dinov2_backbone"] = str(dino_backbone)
    cfg["load_ckpt"] = None
    cfg["wandb_project"] = None
    cfg["save_final_checkpoint"] = False
    cfg["cache_vol_tokens"] = False
    if attention_scaling_mode is not None:
        cfg["attention_scaling_mode"] = str(attention_scaling_mode)
    model = AutoregMeshModel(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, cfg


def extend_tifxyz_mesh(
    *,
    tifxyz_path: str | Path,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    out_dir: str | Path,
    device: str = "cuda",
    grow_direction: str | None = None,
    prompt_strips: int = 8,
    predict_strips_per_iter: int = 8,
    window_strip_length: int = 64,
    window_overlap: int = 16,
    window_batch_size: int = 4,
    max_extension_iters: int = 4,
    max_crop_fit_retries: int = 3,
    show_progress: bool | None = None,
    fast_infer: bool = True,
    compile_infer: bool = False,
    amp_dtype: str = "bf16",
    distributed_infer: bool = False,
    distributed_shard_mode: str = "strided",
    distributed_gather_mode: str = "object",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    planner_mode: str = PLANNER_MODE_FAST,
) -> dict[str, Any]:
    planner_mode = _normalize_planner_mode(planner_mode)
    if str(distributed_shard_mode) != "strided":
        raise ValueError(f"unsupported distributed_shard_mode {distributed_shard_mode!r}")
    if str(distributed_gather_mode) != "object":
        raise ValueError(f"unsupported distributed_gather_mode {distributed_gather_mode!r}")
    runtime = _initialize_distributed_infer_runtime(enabled=bool(distributed_infer), device=device)
    local_show_progress = bool(show_progress) if show_progress is not None else None
    if runtime.is_distributed and not runtime.is_main_process:
        local_show_progress = False
    device_obj = runtime.device
    out_path = Path(out_dir)
    if runtime.is_main_process:
        out_path.mkdir(parents=True, exist_ok=True)

    total_started = perf_counter()
    timings: dict[str, float] = {}
    crop_size = (128, 128, 128)
    cache = VolumeCropCache(max_items=8)
    total_windows = 0
    total_fitted_windows = 0
    total_skipped_windows = 0
    total_parent_windows = 0
    total_child_windows = 0
    total_deduped_child_windows = 0
    total_gather_ms = 0.0
    total_rank_merge_input_count = 0
    per_rank_fitted_window_counts_total = [0 for _ in range(runtime.world_size)]
    summary: dict[str, Any] | None = None
    cached_infer_available = False
    planner_diagnostics_payload: dict[str, Any] | None = None
    planner_diagnostics_path: Path | None = None

    try:
        t0 = perf_counter()
        if runtime.is_main_process:
            surface = read_tifxyz(tifxyz_path, load_mask=True, validate=True).use_stored_resolution()
            grid_zyx = _surface_grid_zyx(surface)
            surface_scale = tuple(float(v) for v in surface.get_scale_tuple())
            surface_uuid = str(surface.uuid or Path(tifxyz_path).name)
            provenance = np.zeros(grid_zyx.shape[:2], dtype=np.uint8)
            grid_zyx, provenance, trimmed_bbox_rc = _trim_grid_to_valid_bbox(grid_zyx, provenance)
            requested_direction = None if grow_direction in {None, ""} else str(grow_direction)
            direction, grid_zyx = resolve_growth_direction(
                grid_zyx,
                prompt_strips=int(prompt_strips),
                predict_strips=int(predict_strips_per_iter),
                crop_size=crop_size,
                requested_direction=requested_direction,
            )
            root_state = {
                "surface_scale": surface_scale,
                "surface_uuid": surface_uuid,
                "trimmed_bbox_rc": trimmed_bbox_rc,
                "direction": direction,
                "requested_direction": requested_direction,
                "original_vertex_count": int(np.isfinite(grid_zyx).all(axis=-1).sum()),
            }
        else:
            grid_zyx = None
            provenance = None
            root_state = None
        root_state = _broadcast_object(root_state, runtime=runtime)
        if runtime.is_main_process:
            timings["load_surface_ms"] = 1000.0 * (perf_counter() - t0)
            working_grid = grid_zyx.copy()
            working_provenance = provenance.copy()
        else:
            working_grid = None
            working_provenance = None
        surface_scale = tuple(float(v) for v in root_state["surface_scale"])
        surface_uuid = str(root_state["surface_uuid"])
        trimmed_bbox_rc = tuple(int(v) for v in root_state["trimmed_bbox_rc"])
        direction = str(root_state["direction"])
        requested_direction = root_state.get("requested_direction")
        original_vertex_count = int(root_state["original_vertex_count"])

        if runtime.is_main_process and planner_mode == PLANNER_MODE_COVERAGE_FIRST:
            planner_diagnostics_payload = {
                "planner_mode": str(planner_mode),
                "surface_uuid": str(surface_uuid),
                "requested_direction": requested_direction,
                "resolved_lattice_direction": str(direction),
                "initial_preflight": _evaluate_extension_planner_on_grid(
                    working_grid,
                    tifxyz_path=str(tifxyz_path),
                    requested_direction=str(requested_direction),
                    resolved_direction=str(direction),
                    trimmed_bbox_rc=trimmed_bbox_rc,
                    planner_mode=str(planner_mode),
                    prompt_strips=int(prompt_strips),
                    predict_strips_per_iter=int(predict_strips_per_iter),
                    window_strip_length=int(window_strip_length),
                    window_overlap=int(window_overlap),
                    max_crop_fit_retries=int(max_crop_fit_retries),
                    crop_size=crop_size,
                ),
                "iterations": [],
            }
            planner_diagnostics_path = out_path / "planner_diagnostics.json"
            planner_diagnostics_path.write_text(json.dumps(planner_diagnostics_payload, indent=2))

        volume = _open_zarr_volume(volume_uri)
        volume_shape = tuple(int(v) for v in volume.shape[-3:])

        t0 = perf_counter()
        try:
            model, model_cfg = _load_autoreg_model(
                dino_backbone=str(dino_backbone),
                autoreg_checkpoint=str(autoreg_checkpoint),
                device=device_obj,
                attention_scaling_mode=str(attention_scaling_mode),
            )
        except TypeError as exc:
            if "attention_scaling_mode" not in str(exc):
                raise
            model, model_cfg = _load_autoreg_model(
                dino_backbone=str(dino_backbone),
                autoreg_checkpoint=str(autoreg_checkpoint),
                device=device_obj,
            )
        timings["load_model_ms"] = 1000.0 * (perf_counter() - t0) if runtime.is_main_process else 0.0
        cached_infer_available = _supports_cached_window_inference(model)
        infer_runtime = ExtensionInferenceRuntime(
            model,
            device=device_obj,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
        )

        iteration_stats: list[ExtensionIterationStats] = []
        coverage_frontier_intervals: list[tuple[int, int]] | None = None
        stop_reason = "max_extension_iters"
        for iteration_idx in range(int(max_extension_iters)):
            iteration_started = perf_counter()
            if runtime.is_main_process:
                if iteration_idx > 0:
                    demote_previous_seam(working_provenance)
                fitted_window_plans, planning_stats, planning_diagnostics = _plan_extension_windows_for_mode(
                    working_grid,
                    direction=direction,
                    planner_mode=str(planner_mode),
                    window_strip_length=int(window_strip_length),
                    window_overlap=int(window_overlap),
                    prompt_strips=int(prompt_strips),
                    predict_strips=int(predict_strips_per_iter),
                    crop_size=crop_size,
                    max_crop_fit_retries=int(max_crop_fit_retries),
                    frontier_intervals=coverage_frontier_intervals if planner_mode == PLANNER_MODE_COVERAGE_FIRST else None,
                    progress_desc=f"iter {iteration_idx + 1} plan",
                    show_progress=local_show_progress,
                )
                planning_state = {
                    "fitted_window_plans": fitted_window_plans,
                    "planning_stats": planning_stats,
                    "planning_diagnostics": planning_diagnostics,
                    "stop_reason": None,
                }
                if int(planning_stats["parent_window_count"]) <= 0:
                    planning_state["stop_reason"] = "no_windows"
                elif not fitted_window_plans:
                    planning_state["stop_reason"] = "all_windows_crop_fit_failed"
            else:
                planning_state = None
            planning_state = _broadcast_object(planning_state, runtime=runtime)
            parent_window_count = int(planning_state["planning_stats"]["parent_window_count"])
            child_window_count = int(planning_state["planning_stats"]["child_window_count"])
            deduped_child_window_count = int(planning_state["planning_stats"]["deduped_child_window_count"])
            crop_fit_failed_count = int(planning_state["planning_stats"]["crop_fit_failed_count"])
            stop_reason_candidate = planning_state.get("stop_reason")
            if stop_reason_candidate is not None:
                stop_reason = str(stop_reason_candidate)
                break

            fitted_window_plans = list(planning_state["fitted_window_plans"])
            sharded_plans = _shard_fitted_window_plans(
                fitted_window_plans,
                runtime=runtime,
                shard_mode=str(distributed_shard_mode),
            )
            max_predict_strips_seen = max(int(plan.predict_strips) for plan in fitted_window_plans)
            crop_read_ms = 0.0
            payloads: list[ExtensionWindowPayload] = []

            def _build_one_payload(args):
                global_index, fitted_plan = args
                t1 = perf_counter()
                volume_crop = _read_volume_crop(volume, fitted_plan.min_corner, crop_size, cache=cache)
                read_ms = 1000.0 * (perf_counter() - t1)
                sample = build_extension_sample(
                    prompt_grid_world=fitted_plan.prompt_grid,
                    direction=direction,
                    min_corner=fitted_plan.min_corner,
                    crop_size=crop_size,
                    patch_size=tuple(int(v) for v in model_cfg["patch_size"]),
                    offset_num_bins=tuple(int(v) for v in model_cfg["offset_num_bins"]),
                    frontier_band_width=int(model_cfg.get("frontier_band_width", 4)),
                    predict_strips=fitted_plan.predict_strips,
                    volume_crop=volume_crop,
                    wrap_metadata={"segment_uuid": surface_uuid, "source_tifxyz": str(tifxyz_path)},
                )
                return ExtensionWindowPayload(
                    global_index=int(global_index),
                    window=fitted_plan.window,
                    sample=sample,
                    direction=direction,
                    target_grid_shape=tuple(int(v) for v in sample["target_grid_shape"].tolist()),
                    strip_length=int(sample["strip_length"].item()),
                    num_strips=int(sample["num_strips"].item()),
                    prompt_strips=fitted_plan.prompt_strips,
                    predict_strips=fitted_plan.predict_strips,
                ), read_ms

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(8, max(1, len(sharded_plans)))) as pool:
                build_results = list(_progress_iter(
                    pool.map(_build_one_payload, sharded_plans),
                    total=len(sharded_plans),
                    desc=f"iter {iteration_idx + 1} build",
                    show_progress=local_show_progress,
                ))
            for payload, read_ms in build_results:
                crop_read_ms += read_ms
                payloads.append(payload)

            local_fitted_window_count = int(len(payloads))
            local_peak_batch_size = 0
            local_results: list[dict[str, Any]] = []
            encode_decode_ms = 0.0
            if payloads:
                local_results, encode_decode_ms, local_peak_batch_size = infer_extension_windows_batched_cached(
                    model,
                    payloads,
                    window_batch_size=int(window_batch_size),
                    device=device_obj,
                    greedy=True,
                    stop_probability_threshold=None,
                    progress_desc=f"iter {iteration_idx + 1} infer",
                    show_progress=local_show_progress,
                    fast_infer=bool(fast_infer),
                    compile_infer=bool(compile_infer),
                    amp_dtype=str(amp_dtype),
                    runtime=infer_runtime,
                )

            gather_started = perf_counter()
            gathered_payloads = _all_gather_objects(
                {
                    "rank": int(runtime.rank),
                    "fitted_window_count": int(local_fitted_window_count),
                    "peak_batch_size": int(local_peak_batch_size),
                    "crop_read_ms": float(crop_read_ms),
                    "encode_decode_ms": float(encode_decode_ms),
                    "results": local_results,
                },
                runtime=runtime,
            )
            gather_ms = 1000.0 * (perf_counter() - gather_started) if runtime.is_distributed else 0.0
            if runtime.is_main_process:
                merged_results, per_rank_counts = _flatten_gathered_window_results(gathered_payloads)
                per_rank_fitted_window_counts_total = [
                    int(prev) + int(curr)
                    for prev, curr in zip(per_rank_fitted_window_counts_total, per_rank_counts, strict=True)
                ]
                total_gather_ms += float(gather_ms)
                total_rank_merge_input_count += int(len(merged_results))
                fitted_window_count = int(sum(per_rank_counts))
                skipped_window_count = int(crop_fit_failed_count)
                peak_batch_size = int(max(int(payload["peak_batch_size"]) for payload in gathered_payloads))
                crop_read_ms_total = float(sum(float(payload["crop_read_ms"]) for payload in gathered_payloads))
                encode_decode_ms_total = float(sum(float(payload["encode_decode_ms"]) for payload in gathered_payloads))
                sums, counts = _initialize_extension_arrays(working_grid, direction, max_predict_strips_seen)
                merge_ms = 0.0
                empty_prediction_count = 0
                model_stop_count = 0
                for result in merged_results:
                    if int(result["predicted_vertex_count"]) <= 0:
                        empty_prediction_count += 1
                        continue
                    if int(result["stop_count"]) > 0:
                        model_stop_count += 1
                    t1 = perf_counter()
                    merge_window_prediction(
                        sums=sums,
                        counts=counts,
                        pred_grid_world=np.asarray(result["continuation_grid_world"], dtype=np.float32),
                        direction=direction,
                        window=result["window"],
                    )
                    merge_ms += 1000.0 * (perf_counter() - t1)

                (
                    model_only_frontier_coverage_fraction,
                    model_only_cell_coverage_fraction,
                    model_only_max_gap,
                    model_only_gap_spans,
                    model_only_first_uncovered_frontier_index,
                ) = _compute_new_band_coverage_metrics(
                    counts,
                    direction=direction,
                )
                geometric_gap_fill_vertex_count = 0
                geometric_gap_fill_frontier_count = 0
                if planner_mode == PLANNER_MODE_COVERAGE_FIRST and model_only_frontier_coverage_fraction < 1.0:
                    geometric_gap_fill_vertex_count, geometric_gap_fill_frontier_count = _apply_geometric_gap_fill(
                        sums,
                        counts,
                        working_grid=working_grid,
                        direction=direction,
                    )
                extension_grid, extension_provenance = finalize_iteration_extension(
                    sums=sums,
                    counts=counts,
                    direction=direction,
                )
                (
                    new_band_frontier_coverage_fraction,
                    new_band_cell_coverage_fraction,
                    new_band_max_gap,
                    new_band_gap_spans,
                    first_uncovered_frontier_index,
                ) = _compute_new_band_coverage_metrics(
                    counts,
                    direction=direction,
                )
                valid_new_vertices = int(np.isfinite(extension_grid).all(axis=-1).sum())
                if valid_new_vertices <= 0:
                    post_state = {"stop_reason": "zero_growth_iteration", "iteration_stat": None}
                else:
                    working_grid, working_provenance = append_extension_to_grid(
                        working_grid=working_grid,
                        working_provenance=working_provenance,
                        extension_grid=extension_grid,
                        extension_provenance=extension_provenance,
                        direction=direction,
                    )
                    stat = ExtensionIterationStats(
                        iteration_index=iteration_idx,
                        window_count=parent_window_count,
                        parent_window_count=parent_window_count,
                        child_window_count=child_window_count,
                        deduped_child_window_count=deduped_child_window_count,
                        valid_new_vertices=valid_new_vertices,
                        fitted_window_count=fitted_window_count,
                        skipped_window_count=skipped_window_count,
                        crop_fit_failed_count=crop_fit_failed_count,
                        empty_prediction_count=empty_prediction_count,
                        model_stop_count=model_stop_count,
                        new_band_frontier_coverage_fraction=new_band_frontier_coverage_fraction,
                        new_band_cell_coverage_fraction=new_band_cell_coverage_fraction,
                        new_band_max_gap=new_band_max_gap,
                        new_band_gap_spans=new_band_gap_spans,
                        first_uncovered_frontier_index=first_uncovered_frontier_index,
                        crop_read_ms=crop_read_ms_total,
                        encode_decode_ms=encode_decode_ms_total,
                        merge_ms=merge_ms,
                        iteration_wall_ms=1000.0 * (perf_counter() - iteration_started),
                        windows_per_second=(float(fitted_window_count) * 1000.0 / max(1e-6, (crop_read_ms_total + encode_decode_ms_total + merge_ms))),
                        peak_batch_size_used=peak_batch_size,
                        model_only_new_band_frontier_coverage_fraction=model_only_frontier_coverage_fraction,
                        geometric_gap_fill_vertex_count=int(geometric_gap_fill_vertex_count),
                        geometric_gap_fill_frontier_count=int(geometric_gap_fill_frontier_count),
                    )
                    post_state = {"stop_reason": None, "iteration_stat": vars(stat)}
                    iteration_stats.append(stat)
                    total_windows += int(parent_window_count)
                    total_fitted_windows += int(fitted_window_count)
                    total_skipped_windows += int(skipped_window_count)
                    total_parent_windows += int(parent_window_count)
                    total_child_windows += int(child_window_count)
                    total_deduped_child_windows += int(deduped_child_window_count)
            else:
                post_state = None
            post_state = _broadcast_object(post_state, runtime=runtime)
            if post_state["stop_reason"] is not None:
                stop_reason = str(post_state["stop_reason"])
                break
            if runtime.is_main_process and planner_diagnostics_payload is not None:
                iteration_diag = dict(planning_state.get("planning_diagnostics") or {})
                iteration_diag["iteration_index"] = int(iteration_idx)
                iteration_diag["frontier_intervals"] = None if coverage_frontier_intervals is None else [list(span) for span in coverage_frontier_intervals]
                iteration_diag["post_iteration_gap_spans"] = list(iteration_stats[-1].new_band_gap_spans)
                iteration_diag["post_iteration_coverage_fraction"] = float(iteration_stats[-1].new_band_frontier_coverage_fraction)
                iteration_diag["model_only_frontier_coverage_fraction"] = float(iteration_stats[-1].model_only_new_band_frontier_coverage_fraction)
                iteration_diag["geometric_gap_fill_vertex_count"] = int(iteration_stats[-1].geometric_gap_fill_vertex_count)
                iteration_diag["geometric_gap_fill_frontier_count"] = int(iteration_stats[-1].geometric_gap_fill_frontier_count)
                planner_diagnostics_payload["iterations"].append(iteration_diag)
                planner_diagnostics_path.write_text(json.dumps(planner_diagnostics_payload, indent=2))
            if planner_mode == PLANNER_MODE_COVERAGE_FIRST and runtime.is_main_process:
                coverage_frontier_intervals = [tuple(int(v) for v in span) for span in iteration_stats[-1].new_band_gap_spans]

        if runtime.is_main_process:
            final_predicted_nonseam_vertex_count = int((working_provenance == 1).sum())
            final_seam_vertex_count = int((working_provenance == 2).sum())
            final_band_frontier_coverage_fraction = float(iteration_stats[-1].new_band_frontier_coverage_fraction) if iteration_stats else 0.0
            final_band_cell_coverage_fraction = float(iteration_stats[-1].new_band_cell_coverage_fraction) if iteration_stats else 0.0
            final_band_max_gap = int(iteration_stats[-1].new_band_max_gap) if iteration_stats else 0
            final_band_gap_spans = list(iteration_stats[-1].new_band_gap_spans) if iteration_stats else []
            final_first_uncovered_frontier_index = iteration_stats[-1].first_uncovered_frontier_index if iteration_stats else None

            vertices, faces, colors = grid_to_colored_mesh(working_grid, working_provenance)
            mesh_path = write_colored_ply(out_path / f"{surface_uuid}_merged.ply", vertices, faces, colors)
            preview_paths = _save_projection_images(working_grid, working_provenance, out_path)
            tifxyz_path_out = write_tifxyz(
                out_path / f"{surface_uuid}_merged_tifxyz",
                _build_tifxyz_from_grid(working_grid, uuid=f"{surface_uuid}_merged", scale=surface_scale),
                overwrite=True,
            )

            summary = {
                "surface_uuid": surface_uuid,
                "source_tifxyz_path": str(tifxyz_path),
                "direction": direction,
                "requested_direction": requested_direction,
                "volume_uri": str(volume_uri),
                "dino_backbone": str(dino_backbone),
                "autoreg_checkpoint": str(autoreg_checkpoint),
                "attention_scaling_mode": str(model_cfg.get("attention_scaling_mode", ATTENTION_SCALING_STANDARD)),
                "planner_mode": str(planner_mode),
                "original_vertex_count": int(original_vertex_count),
                "final_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum()),
                "predicted_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum() - int(original_vertex_count)),
                "cumulative_predicted_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum() - int(original_vertex_count)),
                "final_predicted_nonseam_vertex_count": final_predicted_nonseam_vertex_count,
                "final_seam_vertex_count": final_seam_vertex_count,
                "mesh_path": str(mesh_path),
                "preview_paths": preview_paths,
                "tifxyz_path": str(tifxyz_path_out),
                "timings_ms": timings,
                "iteration_stats": [vars(item) for item in iteration_stats],
                "window_batch_size": int(window_batch_size),
                "total_wall_ms": 1000.0 * (perf_counter() - total_started),
                "total_windows": int(total_windows),
                "total_fitted_windows": int(total_fitted_windows),
                "total_skipped_windows": int(total_skipped_windows),
                "parent_window_count": int(total_parent_windows),
                "child_window_count": int(total_child_windows),
                "deduped_child_window_count": int(total_deduped_child_windows),
                "new_band_frontier_coverage_fraction": final_band_frontier_coverage_fraction,
                "new_band_cell_coverage_fraction": final_band_cell_coverage_fraction,
                "new_band_max_gap": final_band_max_gap,
                "new_band_gap_spans": final_band_gap_spans,
                "first_uncovered_frontier_index": final_first_uncovered_frontier_index,
                "model_only_new_band_frontier_coverage_fraction": (
                    float(iteration_stats[-1].model_only_new_band_frontier_coverage_fraction) if iteration_stats else 0.0
                ),
                "geometric_gap_fill_vertex_count": int(sum(item.geometric_gap_fill_vertex_count for item in iteration_stats)),
                "geometric_gap_fill_frontier_count": int(sum(item.geometric_gap_fill_frontier_count for item in iteration_stats)),
                "iterations_completed": int(len(iteration_stats)),
                "stop_reason": stop_reason,
                "windows_per_second_overall": (
                    float(total_fitted_windows) * 1000.0 / max(1e-6, 1000.0 * (perf_counter() - total_started))
                ),
                "fast_infer_enabled": bool(infer_runtime.fast_infer_enabled),
                "compile_infer_requested": bool(infer_runtime.compile_infer_requested),
                "compile_infer_actual": bool(infer_runtime.compile_infer_actual),
                "compile_infer_failure": infer_runtime.compile_infer_failure,
                "cached_infer_available": bool(cached_infer_available),
                "cached_infer_fallback_used": bool(not cached_infer_available),
                "amp_dtype": (
                    "bf16" if infer_runtime.amp_dtype == torch.bfloat16 else "fp16" if infer_runtime.amp_dtype == torch.float16 else None
                ),
                "encode_decode_ms_per_fitted_window": (
                    float(sum(item.encode_decode_ms for item in iteration_stats)) / float(total_fitted_windows)
                    if total_fitted_windows > 0
                    else None
                ),
                "trimmed_bbox_rc": list(trimmed_bbox_rc),
                "crop_cache_hits": int(cache.hits),
                "crop_cache_misses": int(cache.misses),
                "volume_shape": list(volume_shape),
                "distributed_infer_enabled": bool(runtime.is_distributed),
                "distributed_world_size": int(runtime.world_size),
                "distributed_backend": runtime.backend,
                "distributed_shard_mode": str(distributed_shard_mode),
                "distributed_gather_mode": str(distributed_gather_mode),
                "per_rank_fitted_window_counts": list(per_rank_fitted_window_counts_total),
                "gather_ms": float(total_gather_ms),
                "rank_merge_input_count": int(total_rank_merge_input_count),
            }
            if planner_diagnostics_path is not None:
                summary["planner_diagnostics_path"] = str(planner_diagnostics_path)
            summary_path = out_path / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2))
            summary["summary_path"] = str(summary_path)
        summary = _broadcast_object(summary, runtime=runtime)
        return summary
    finally:
        _destroy_distributed_infer_runtime(runtime)


def _compact_extension_summary(summary: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "direction",
        "requested_direction",
        "attention_scaling_mode",
        "planner_mode",
        "predicted_vertex_count",
        "cumulative_predicted_vertex_count",
        "final_predicted_nonseam_vertex_count",
        "final_seam_vertex_count",
        "new_band_frontier_coverage_fraction",
        "new_band_cell_coverage_fraction",
        "new_band_max_gap",
        "new_band_gap_spans",
        "first_uncovered_frontier_index",
        "iterations_completed",
        "stop_reason",
        "window_batch_size",
        "model_only_new_band_frontier_coverage_fraction",
        "geometric_gap_fill_vertex_count",
        "geometric_gap_fill_frontier_count",
        "encode_decode_ms_per_fitted_window",
        "crop_cache_hits",
        "crop_cache_misses",
        "total_wall_ms",
        "fast_infer_enabled",
        "compile_infer_requested",
        "compile_infer_actual",
        "compile_infer_failure",
        "cached_infer_available",
        "cached_infer_fallback_used",
        "amp_dtype",
    ]
    compact = {key: summary.get(key) for key in keys}
    compact["summary_path"] = summary.get("summary_path")
    compact["tifxyz_path"] = summary.get("tifxyz_path")
    return compact


def run_extension_to_exhaustion(
    *,
    input_tifxyz_path: str | Path,
    output_dir: str | Path,
    grow_direction: str,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    device: str = "cuda",
    prompt_strips: int = 8,
    predict_strips_per_iter: int = 8,
    window_strip_length: int = 64,
    window_overlap: int = 16,
    window_batch_size: int = 4,
    max_extension_iters_per_call: int = 4,
    max_crop_fit_retries: int = 3,
    max_calls: int | None = None,
    show_progress: bool | None = None,
    fast_infer: bool = True,
    compile_infer: bool = False,
    amp_dtype: str = "bf16",
    distributed_infer: bool = False,
    distributed_shard_mode: str = "strided",
    distributed_gather_mode: str = "object",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    planner_mode: str = PLANNER_MODE_FAST,
    call_validator=None,
    extend_impl=None,
) -> ExtensionStageResult:
    current_input = Path(input_tifxyz_path)
    output_dir = Path(output_dir)
    call_idx = 0
    call_summaries: list[dict[str, Any]] = []
    stage_stop_reason = "completed"
    resolved_direction: str | None = None
    while True:
        call_dir = output_dir / f"{grow_direction}_call_{call_idx:03d}"
        requested_direction = str(grow_direction if call_idx == 0 or resolved_direction is None else resolved_direction)
        extend_fn = extend_tifxyz_mesh if extend_impl is None else extend_impl
        summary = extend_fn(
            tifxyz_path=current_input,
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            out_dir=call_dir,
            device=device,
            grow_direction=requested_direction,
            prompt_strips=int(prompt_strips),
            predict_strips_per_iter=int(predict_strips_per_iter),
            window_strip_length=int(window_strip_length),
            window_overlap=int(window_overlap),
            window_batch_size=int(window_batch_size),
            max_extension_iters=int(max_extension_iters_per_call),
            max_crop_fit_retries=int(max_crop_fit_retries),
            show_progress=show_progress,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
            distributed_infer=bool(distributed_infer),
            distributed_shard_mode=str(distributed_shard_mode),
            distributed_gather_mode=str(distributed_gather_mode),
            attention_scaling_mode=str(attention_scaling_mode),
            planner_mode=str(planner_mode),
        )
        compact_summary = _compact_extension_summary(summary)
        call_summaries.append(compact_summary)
        if resolved_direction is None:
            resolved_direction = str(summary["direction"])
        if callable(call_validator):
            call_validator(compact_summary, call_idx)
        current_input = Path(summary["tifxyz_path"])
        if str(summary["stop_reason"]) != "max_extension_iters":
            stage_stop_reason = str(summary["stop_reason"])
            break
        call_idx += 1
        if max_calls is not None and call_idx >= int(max_calls):
            stage_stop_reason = "max_calls_reached"
            break
    stage_summary = {
        "requested_direction": str(grow_direction),
        "direction": str(resolved_direction or grow_direction),
        "call_count": int(len(call_summaries)),
        "final_tifxyz_path": str(current_input),
        "stage_stop_reason": str(stage_stop_reason),
        "calls": call_summaries,
        "final": call_summaries[-1],
    }
    return ExtensionStageResult(final_tifxyz_path=current_input, stage_summary=stage_summary)


def _frontier_span_coverage(plans: list[FittedWindowPlan], *, frontier_length: int) -> tuple[int, list[tuple[int, int]]]:
    if frontier_length <= 0 or not plans:
        return 0, []
    spans = sorted((int(plan.window.start), int(plan.window.end)) for plan in plans)
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    covered = sum(max(0, end - start) for start, end in merged)
    return int(covered), merged


def _evaluate_extension_planner_on_grid(
    grid_zyx: np.ndarray,
    *,
    tifxyz_path: str | Path,
    requested_direction: str,
    resolved_direction: str,
    trimmed_bbox_rc: tuple[int, int, int, int],
    planner_mode: str,
    prompt_strips: int,
    predict_strips_per_iter: int,
    window_strip_length: int,
    window_overlap: int,
    max_crop_fit_retries: int = 3,
    crop_size: tuple[int, int, int] = (128, 128, 128),
    frontier_intervals: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    plans, stats, diagnostics = _plan_extension_windows_for_mode(
        grid_zyx,
        direction=str(resolved_direction),
        planner_mode=str(planner_mode),
        window_strip_length=int(window_strip_length),
        window_overlap=int(window_overlap),
        prompt_strips=int(prompt_strips),
        predict_strips=int(predict_strips_per_iter),
        crop_size=crop_size,
        max_crop_fit_retries=int(max_crop_fit_retries),
        frontier_intervals=frontier_intervals,
    )
    frontier_length = _current_frontier_length(grid_zyx, resolved_direction)
    covered_frontier, merged_spans = _frontier_span_coverage(plans, frontier_length=frontier_length)
    coverage_mask = np.zeros(int(frontier_length), dtype=bool)
    for plan in plans:
        coverage_mask |= _frontier_mask_from_window(frontier_length, plan.window)
    if frontier_intervals is not None:
        active_mask = np.zeros(int(frontier_length), dtype=bool)
        for start, end in frontier_intervals:
            active_mask[max(0, int(start)):min(int(frontier_length), int(end))] = True
        uncovered_mask = active_mask & ~coverage_mask
    else:
        uncovered_mask = ~coverage_mask
    result = {
        "tifxyz_path": str(tifxyz_path),
        "planner_mode": str(planner_mode),
        "requested_direction": str(requested_direction),
        "direction": str(resolved_direction),
        "prompt_strips": int(prompt_strips),
        "predict_strips_per_iter": int(predict_strips_per_iter),
        "window_strip_length": int(window_strip_length),
        "window_overlap": int(window_overlap),
        "trimmed_bbox_rc": list(trimmed_bbox_rc),
        "frontier_length": int(frontier_length),
        "frontier_intervals": None if frontier_intervals is None else [(int(start), int(end)) for start, end in frontier_intervals],
        "fitted_plans": int(len(plans)),
        "fitted_plan_spans": [(int(plan.window.start), int(plan.window.end)) for plan in plans],
        "covered_frontier": int(covered_frontier),
        "covered_frontier_fraction": (float(covered_frontier) / float(frontier_length)) if frontier_length > 0 else 0.0,
        "covered_frontier_spans": merged_spans,
        "gap_spans": _true_spans_from_mask(uncovered_mask),
        "planning_stats": stats,
    }
    if diagnostics is not None:
        result["planner_diagnostics"] = diagnostics
    return result


def evaluate_extension_planner(
    *,
    tifxyz_path: str | Path,
    grow_direction: str,
    prompt_strips: int,
    predict_strips_per_iter: int,
    window_strip_length: int,
    window_overlap: int,
    max_crop_fit_retries: int = 3,
    crop_size: tuple[int, int, int] = (128, 128, 128),
    planner_mode: str = PLANNER_MODE_FAST,
    frontier_intervals: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    planner_mode = _normalize_planner_mode(planner_mode)
    surface = read_tifxyz(tifxyz_path, load_mask=True, validate=True).use_stored_resolution()
    grid_zyx = _surface_grid_zyx(surface)
    grid_zyx, _, trimmed_bbox_rc = _trim_grid_to_valid_bbox(grid_zyx, np.zeros(grid_zyx.shape[:2], dtype=np.uint8))
    direction, resolved_grid = resolve_growth_direction(
        grid_zyx,
        prompt_strips=int(prompt_strips),
        predict_strips=int(predict_strips_per_iter),
        crop_size=crop_size,
        requested_direction=str(grow_direction),
    )
    return _evaluate_extension_planner_on_grid(
        resolved_grid,
        tifxyz_path=tifxyz_path,
        requested_direction=str(grow_direction),
        resolved_direction=str(direction),
        trimmed_bbox_rc=tuple(int(v) for v in trimmed_bbox_rc),
        planner_mode=str(planner_mode),
        prompt_strips=int(prompt_strips),
        predict_strips_per_iter=int(predict_strips_per_iter),
        window_strip_length=int(window_strip_length),
        window_overlap=int(window_overlap),
        max_crop_fit_retries=int(max_crop_fit_retries),
        crop_size=crop_size,
        frontier_intervals=frontier_intervals,
    )


def run_extension_benchmark_suite(
    *,
    tifxyz_path: str | Path,
    volume_uri: str,
    dino_backbone: str,
    autoreg_checkpoint: str,
    out_dir: str | Path,
    device: str,
    prompt_strips: int,
    predict_strips_per_iter: int,
    window_strip_length: int,
    window_overlap: int,
    window_batch_sizes: list[int],
    long_rollout_iters: int,
    max_crop_fit_retries: int,
    grow_direction: str | None = None,
    show_progress: bool | None = None,
    fast_infer: bool = True,
    compile_infer: bool = False,
    amp_dtype: str = "bf16",
    distributed_infer: bool = False,
    distributed_shard_mode: str = "strided",
    distributed_gather_mode: str = "object",
    attention_scaling_mode: str = ATTENTION_SCALING_STANDARD,
    planner_mode: str = PLANNER_MODE_FAST,
) -> dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    batch_sizes = list(dict.fromkeys(int(v) for v in window_batch_sizes))
    if not batch_sizes:
        raise ValueError("window_batch_sizes must be non-empty for benchmark suite")
    serial_runs = []
    for batch_size in batch_sizes:
        run_dir = out_path / f"batch_{batch_size}_iter1"
        summary = extend_tifxyz_mesh(
            tifxyz_path=tifxyz_path,
            volume_uri=volume_uri,
            dino_backbone=dino_backbone,
            autoreg_checkpoint=autoreg_checkpoint,
            out_dir=run_dir,
            device=device,
            grow_direction=grow_direction,
            prompt_strips=prompt_strips,
            predict_strips_per_iter=predict_strips_per_iter,
            window_strip_length=window_strip_length,
            window_overlap=window_overlap,
            window_batch_size=batch_size,
            max_extension_iters=1,
            max_crop_fit_retries=max_crop_fit_retries,
            show_progress=show_progress,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
            distributed_infer=bool(distributed_infer),
            distributed_shard_mode=str(distributed_shard_mode),
            distributed_gather_mode=str(distributed_gather_mode),
            attention_scaling_mode=str(attention_scaling_mode),
            planner_mode=str(planner_mode),
        )
        serial_runs.append(summary)

    candidate_runs = [run for run in serial_runs if int(run.get("predicted_vertex_count", 0)) > 0]
    best_run = max(
        candidate_runs or serial_runs,
        key=lambda item: (
            float(item.get("windows_per_second_overall", 0.0)),
            int(item.get("predicted_vertex_count", 0)),
            -int(item.get("window_batch_size", 1)),
        ),
    )
    long_rollout_dir = out_path / f"batch_{int(best_run['window_batch_size'])}_iter{int(long_rollout_iters)}"
    long_rollout = extend_tifxyz_mesh(
        tifxyz_path=tifxyz_path,
        volume_uri=volume_uri,
        dino_backbone=dino_backbone,
        autoreg_checkpoint=autoreg_checkpoint,
        out_dir=long_rollout_dir,
        device=device,
        grow_direction=grow_direction,
        prompt_strips=prompt_strips,
        predict_strips_per_iter=predict_strips_per_iter,
        window_strip_length=window_strip_length,
        window_overlap=window_overlap,
        window_batch_size=int(best_run["window_batch_size"]),
        max_extension_iters=int(long_rollout_iters),
        max_crop_fit_retries=max_crop_fit_retries,
        show_progress=show_progress,
        fast_infer=bool(fast_infer),
        compile_infer=bool(compile_infer),
        amp_dtype=str(amp_dtype),
        distributed_infer=bool(distributed_infer),
        distributed_shard_mode=str(distributed_shard_mode),
        distributed_gather_mode=str(distributed_gather_mode),
        attention_scaling_mode=str(attention_scaling_mode),
        planner_mode=str(planner_mode),
    )
    suite = {
        "tifxyz_path": str(tifxyz_path),
        "serial_baselines": serial_runs,
        "best_batch_run": best_run,
        "long_rollout": long_rollout,
    }
    suite_path = out_path / "benchmark_suite.json"
    suite_path.write_text(json.dumps(suite, indent=2))
    suite["benchmark_suite_path"] = str(suite_path)
    return suite


@click.command()
@click.option("--tifxyz-path", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--tifxyz-root", type=click.Path(exists=True, path_type=Path), required=False)
@click.option("--volume-uri", type=str, required=True)
@click.option("--dinov2-backbone", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--autoreg-ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option("--device", type=str, default="cuda", show_default=True)
@click.option("--grow-direction", type=click.Choice(list(ALL_GROW_DIRECTIONS)), default="auto", show_default=True)
@click.option("--prompt-strips", type=int, default=8, show_default=True)
@click.option("--predict-strips-per-iter", type=int, default=8, show_default=True)
@click.option("--window-strip-length", type=int, default=64, show_default=True)
@click.option("--window-overlap", type=int, default=16, show_default=True)
@click.option("--window-batch-size", type=int, default=4, show_default=True)
@click.option("--max-extension-iters", type=int, default=4, show_default=True)
@click.option("--run-to-exhaustion/--single-call", default=False, show_default=True)
@click.option("--max-calls", type=int, default=None)
@click.option("--max-crop-fit-retries", type=int, default=3, show_default=True)
@click.option("--fast-infer/--no-fast-infer", default=True, show_default=True)
@click.option("--compile-infer/--no-compile-infer", default=False, show_default=True)
@click.option("--amp-dtype", type=click.Choice(["bf16", "fp16"]), default="bf16", show_default=True)
@click.option("--attention-scaling-mode", type=click.Choice(list(ATTENTION_SCALING_MODES)), default=ATTENTION_SCALING_STANDARD, show_default=True)
@click.option("--planner-mode", type=click.Choice(list(PLANNER_MODES)), default=PLANNER_MODE_FAST, show_default=True)
@click.option("--show-progress/--no-show-progress", default=None)
@click.option("--distributed-infer/--no-distributed-infer", default=False, show_default=True)
@click.option("--distributed-shard-mode", type=click.Choice(["strided"]), default="strided", show_default=True)
@click.option("--distributed-gather-mode", type=click.Choice(["object"]), default="object", show_default=True)
@click.option("--benchmark-window-batch-sizes", type=str, default=None)
@click.option("--benchmark-long-rollout-iters", type=int, default=3, show_default=True)
def main(
    tifxyz_path: Path | None,
    tifxyz_root: Path | None,
    volume_uri: str,
    dinov2_backbone: Path,
    autoreg_ckpt: Path,
    out_dir: Path,
    device: str,
    grow_direction: str,
    prompt_strips: int,
    predict_strips_per_iter: int,
    window_strip_length: int,
    window_overlap: int,
    window_batch_size: int,
    max_extension_iters: int,
    run_to_exhaustion: bool,
    max_calls: int | None,
    max_crop_fit_retries: int,
    fast_infer: bool,
    compile_infer: bool,
    amp_dtype: str,
    attention_scaling_mode: str,
    planner_mode: str,
    show_progress: bool | None,
    distributed_infer: bool,
    distributed_shard_mode: str,
    distributed_gather_mode: str,
    benchmark_window_batch_sizes: str | None,
    benchmark_long_rollout_iters: int,
) -> None:
    if (tifxyz_path is None) == (tifxyz_root is None):
        raise click.UsageError("provide exactly one of --tifxyz-path or --tifxyz-root")
    selected_tifxyz = tifxyz_path
    if selected_tifxyz is None:
        selected_tifxyz = choose_source_tifxyz(
            tifxyz_root,
            prompt_strips=prompt_strips,
            predict_strips=predict_strips_per_iter,
            crop_size=(128, 128, 128),
        )
    benchmark_batch_sizes = _parse_int_list(benchmark_window_batch_sizes)
    if benchmark_batch_sizes:
        result = run_extension_benchmark_suite(
            tifxyz_path=selected_tifxyz,
            volume_uri=volume_uri,
            dino_backbone=str(dinov2_backbone),
            autoreg_checkpoint=str(autoreg_ckpt),
            out_dir=out_dir,
            device=device,
            grow_direction=None if grow_direction == "auto" else grow_direction,
            prompt_strips=prompt_strips,
            predict_strips_per_iter=predict_strips_per_iter,
            window_strip_length=window_strip_length,
            window_overlap=window_overlap,
            window_batch_sizes=benchmark_batch_sizes,
            long_rollout_iters=benchmark_long_rollout_iters,
            max_crop_fit_retries=max_crop_fit_retries,
            show_progress=show_progress,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
            distributed_infer=bool(distributed_infer),
            distributed_shard_mode=str(distributed_shard_mode),
            distributed_gather_mode=str(distributed_gather_mode),
            attention_scaling_mode=str(attention_scaling_mode),
            planner_mode=str(planner_mode),
        )
    elif bool(run_to_exhaustion):
        result = run_extension_to_exhaustion(
            input_tifxyz_path=selected_tifxyz,
            output_dir=out_dir,
            grow_direction=str(grow_direction),
            volume_uri=volume_uri,
            dino_backbone=str(dinov2_backbone),
            autoreg_checkpoint=str(autoreg_ckpt),
            device=device,
            prompt_strips=int(prompt_strips),
            predict_strips_per_iter=int(predict_strips_per_iter),
            window_strip_length=int(window_strip_length),
            window_overlap=int(window_overlap),
            window_batch_size=int(window_batch_size),
            max_extension_iters_per_call=int(max_extension_iters),
            max_crop_fit_retries=int(max_crop_fit_retries),
            max_calls=max_calls,
            show_progress=show_progress,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
            distributed_infer=bool(distributed_infer),
            distributed_shard_mode=str(distributed_shard_mode),
            distributed_gather_mode=str(distributed_gather_mode),
            attention_scaling_mode=str(attention_scaling_mode),
            planner_mode=str(planner_mode),
        )
        result = result.stage_summary
    else:
        result = extend_tifxyz_mesh(
            tifxyz_path=selected_tifxyz,
            volume_uri=volume_uri,
            dino_backbone=str(dinov2_backbone),
            autoreg_checkpoint=str(autoreg_ckpt),
            out_dir=out_dir,
            device=device,
            grow_direction=None if grow_direction == "auto" else grow_direction,
            prompt_strips=prompt_strips,
            predict_strips_per_iter=predict_strips_per_iter,
            window_strip_length=window_strip_length,
            window_overlap=window_overlap,
            window_batch_size=window_batch_size,
            max_extension_iters=max_extension_iters,
            max_crop_fit_retries=max_crop_fit_retries,
            show_progress=show_progress,
            fast_infer=bool(fast_infer),
            compile_infer=bool(compile_infer),
            amp_dtype=str(amp_dtype),
            distributed_infer=bool(distributed_infer),
            distributed_shard_mode=str(distributed_shard_mode),
            distributed_gather_mode=str(distributed_gather_mode),
            attention_scaling_mode=str(attention_scaling_mode),
            planner_mode=str(planner_mode),
        )
    if (not bool(distributed_infer)) or int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
