from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import click
import numpy as np
import torch
import zarr

from vesuvius.neural_tracing.autoreg_mesh.dataset import autoreg_mesh_collate
from vesuvius.neural_tracing.autoreg_mesh.infer import (
    _build_target_strip_coords,
    _build_target_strip_positions,
    _sample_from_logits,
    infer_autoreg_mesh,
)
from vesuvius.neural_tracing.autoreg_mesh.model import AutoregMeshModel, build_pseudo_inference_batch
from vesuvius.neural_tracing.autoreg_mesh.serialization import deserialize_continuation_grid, serialize_split_conditioning_example
from vesuvius.tifxyz import Tifxyz, read_tifxyz, write_tifxyz


Color = tuple[int, int, int]
ORIGINAL_COLOR: Color = (90, 180, 255)
PREDICTED_COLOR: Color = (255, 140, 0)
SEAM_COLOR: Color = (255, 0, 255)


@dataclass(frozen=True)
class ExtensionWindow:
    start: int
    end: int


@dataclass
class ExtensionIterationStats:
    iteration_index: int
    window_count: int
    valid_new_vertices: int
    fitted_window_count: int
    skipped_window_count: int
    crop_fit_failed_count: int
    empty_prediction_count: int
    model_stop_count: int
    crop_read_ms: float
    encode_decode_ms: float
    merge_ms: float
    iteration_wall_ms: float
    windows_per_second: float
    peak_batch_size_used: int


@dataclass(frozen=True)
class ExtensionWindowPayload:
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
    low = valid_points.min(axis=0) - float(margin)
    high = valid_points.max(axis=0) + float(margin)
    extent = high - low
    crop = np.asarray(crop_size, dtype=np.float32)
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


def _decode_single_step_from_outputs(
    model,
    outputs: dict,
    *,
    sample_idx: int,
    step_idx: int,
    greedy: bool,
) -> tuple[int, list[int], np.ndarray, float]:
    if str(outputs.get("coarse_prediction_mode", getattr(model, "coarse_prediction_mode", "joint_pointer"))) == "axis_factorized":
        coarse_axis_ids = {}
        for axis_name in ("z", "y", "x"):
            axis_logits = outputs["coarse_axis_logits"][axis_name][sample_idx, step_idx]
            coarse_axis_ids[axis_name] = int(_sample_from_logits(axis_logits, greedy=greedy).item())
        coarse_id = int(
            model._flatten_coarse_axis_ids(
                torch.tensor(coarse_axis_ids["z"], dtype=torch.long, device=outputs["stop_logits"].device),
                torch.tensor(coarse_axis_ids["y"], dtype=torch.long, device=outputs["stop_logits"].device),
                torch.tensor(coarse_axis_ids["x"], dtype=torch.long, device=outputs["stop_logits"].device),
            ).item()
        )
    else:
        coarse_logits = outputs["coarse_logits"][sample_idx, step_idx]
        coarse_id = int(_sample_from_logits(coarse_logits, greedy=greedy).item())
    offset_bins = []
    for axis, bins in enumerate(model.offset_num_bins):
        axis_logits = outputs["offset_logits"][sample_idx, step_idx, axis, :bins]
        offset_bins.append(int(_sample_from_logits(axis_logits, greedy=greedy).item()))
    offset_tensor = torch.tensor(offset_bins, dtype=torch.long, device=outputs["stop_logits"].device).view(1, 1, 3)
    coarse_tensor = torch.tensor([[coarse_id]], dtype=torch.long, device=outputs["stop_logits"].device)
    bin_center_xyz = model.decode_local_xyz(coarse_tensor, offset_tensor)[0, 0].detach().cpu().numpy()
    refine_residual = outputs.get("pred_refine_residual")
    sampled_xyz = bin_center_xyz + refine_residual[sample_idx, step_idx].detach().cpu().numpy() if refine_residual is not None else bin_center_xyz
    stop_prob = float(torch.sigmoid(outputs["stop_logits"][sample_idx, step_idx]).item())
    return coarse_id, offset_bins, sampled_xyz.astype(np.float32, copy=False), stop_prob


def infer_extension_windows_batched(
    model,
    payloads: list[ExtensionWindowPayload],
    *,
    window_batch_size: int,
    device: torch.device,
    greedy: bool = True,
    stop_probability_threshold: float | None = 1.1,
) -> tuple[list[dict[str, Any]], float, int]:
    if not payloads:
        return [], 0.0, 0
    results: list[dict[str, Any]] = []
    total_encode_decode_ms = 0.0
    peak_batch_size = 0
    for payload_batch in _iter_window_batches(payloads, window_batch_size=window_batch_size):
        peak_batch_size = max(peak_batch_size, len(payload_batch))
        raw_samples = [payload.sample for payload in payload_batch]
        batch = autoreg_mesh_collate(raw_samples)
        batch = _move_batch_to_device(batch, device)
        t0 = perf_counter()
        encoded = model.encode_conditioning(batch["volume"], vol_tokens=batch.get("vol_tokens"))
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
            outputs = model.forward_from_encoded(
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
                    "window": payload.window,
                    "direction": payload.direction,
                    "continuation_grid_world": continuation_grid_world,
                    "predicted_vertex_count": int(np.isfinite(continuation_grid_world).all(axis=-1).sum()),
                    "stop_count": int(sum(1 for value in generated_stop_probs[batch_idx] if stop_probability_threshold is not None and value >= float(stop_probability_threshold))),
                }
            )
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


def _load_autoreg_model(*, dino_backbone: str, autoreg_checkpoint: str, device: torch.device) -> tuple[AutoregMeshModel, dict]:
    ckpt = torch.load(Path(autoreg_checkpoint), map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    cfg["dinov2_backbone"] = str(dino_backbone)
    cfg["load_ckpt"] = None
    cfg["wandb_project"] = None
    cfg["save_final_checkpoint"] = False
    cfg["cache_vol_tokens"] = False
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
) -> dict[str, Any]:
    device_obj = torch.device(device)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total_started = perf_counter()
    timings: dict[str, float] = {}
    t0 = perf_counter()
    surface = read_tifxyz(tifxyz_path, load_mask=True, validate=True).use_stored_resolution()
    grid_zyx = _surface_grid_zyx(surface)
    surface_scale = tuple(float(v) for v in surface.get_scale_tuple())
    surface_uuid = str(surface.uuid or Path(tifxyz_path).name)
    provenance = np.zeros(grid_zyx.shape[:2], dtype=np.uint8)
    grid_zyx, provenance, trimmed_bbox_rc = _trim_grid_to_valid_bbox(grid_zyx, provenance)
    timings["load_surface_ms"] = 1000.0 * (perf_counter() - t0)

    volume = _open_zarr_volume(volume_uri)
    volume_shape = tuple(int(v) for v in volume.shape[-3:])
    crop_size = (128, 128, 128)
    cache = VolumeCropCache(max_items=8)

    t0 = perf_counter()
    model, model_cfg = _load_autoreg_model(
        dino_backbone=str(dino_backbone),
        autoreg_checkpoint=str(autoreg_checkpoint),
        device=device_obj,
    )
    timings["load_model_ms"] = 1000.0 * (perf_counter() - t0)

    direction = choose_growth_direction(
        grid_zyx,
        prompt_strips=int(prompt_strips),
        predict_strips=int(predict_strips_per_iter),
        crop_size=crop_size,
        override=None if grow_direction in {None, "", "auto"} else str(grow_direction),
    )

    iteration_stats: list[ExtensionIterationStats] = []
    working_grid = grid_zyx.copy()
    working_provenance = provenance.copy()
    total_windows = 0
    total_fitted_windows = 0
    total_skipped_windows = 0
    stop_reason = "max_extension_iters"

    for iteration_idx in range(int(max_extension_iters)):
        iteration_started = perf_counter()
        if iteration_idx > 0:
            demote_previous_seam(working_provenance)
        frontier_length = _current_frontier_length(working_grid, direction)
        windows = _window_ranges(frontier_length, int(window_strip_length), int(window_overlap))
        if not windows:
            stop_reason = "no_windows"
            break
        max_predict_strips_seen = 0
        crop_read_ms = 0.0
        crop_fit_failed_count = 0
        payloads: list[ExtensionWindowPayload] = []
        for window in windows:
            fitted_plan = _fit_window_for_crop(
                working_grid,
                direction=direction,
                window=window,
                prompt_strips=int(prompt_strips),
                predict_strips=int(predict_strips_per_iter),
                crop_size=crop_size,
                max_crop_fit_retries=int(max_crop_fit_retries),
                min_window_strip_length=4,
            )
            if fitted_plan is None:
                crop_fit_failed_count += 1
                continue
            max_predict_strips_seen = max(max_predict_strips_seen, int(fitted_plan.predict_strips))
            t1 = perf_counter()
            volume_crop = _read_volume_crop(volume, fitted_plan.min_corner, crop_size, cache=cache)
            crop_read_ms += 1000.0 * (perf_counter() - t1)
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
            payloads.append(
                ExtensionWindowPayload(
                    window=fitted_plan.window,
                    sample=sample,
                    direction=direction,
                    target_grid_shape=tuple(int(v) for v in sample["target_grid_shape"].tolist()),
                    strip_length=int(sample["strip_length"].item()),
                    num_strips=int(sample["num_strips"].item()),
                    prompt_strips=fitted_plan.prompt_strips,
                    predict_strips=fitted_plan.predict_strips,
                )
            )
        if not payloads:
            stop_reason = "all_windows_crop_fit_failed"
            break
        fitted_window_count = len(payloads)
        skipped_window_count = len(windows) - fitted_window_count
        sums, counts = _initialize_extension_arrays(working_grid, direction, max_predict_strips_seen)
        merge_ms = 0.0
        batched_results, encode_decode_ms, peak_batch_size = infer_extension_windows_batched(
            model,
            payloads,
            window_batch_size=int(window_batch_size),
            device=device_obj,
            greedy=True,
            stop_probability_threshold=1.1,
        )
        empty_prediction_count = 0
        model_stop_count = 0
        for result in batched_results:
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

        extension_grid, extension_provenance = finalize_iteration_extension(
            sums=sums,
            counts=counts,
            direction=direction,
        )
        valid_new_vertices = int(np.isfinite(extension_grid).all(axis=-1).sum())
        if valid_new_vertices <= 0:
            stop_reason = "zero_growth_iteration"
            break
        working_grid, working_provenance = append_extension_to_grid(
            working_grid=working_grid,
            working_provenance=working_provenance,
            extension_grid=extension_grid,
            extension_provenance=extension_provenance,
            direction=direction,
        )
        iteration_stats.append(
            ExtensionIterationStats(
                iteration_index=iteration_idx,
                window_count=len(windows),
                valid_new_vertices=valid_new_vertices,
                fitted_window_count=fitted_window_count,
                skipped_window_count=skipped_window_count,
                crop_fit_failed_count=crop_fit_failed_count,
                empty_prediction_count=empty_prediction_count,
                model_stop_count=model_stop_count,
                crop_read_ms=crop_read_ms,
                encode_decode_ms=encode_decode_ms,
                merge_ms=merge_ms,
                iteration_wall_ms=1000.0 * (perf_counter() - iteration_started),
                windows_per_second=(float(fitted_window_count) * 1000.0 / max(1e-6, (crop_read_ms + encode_decode_ms + merge_ms))),
                peak_batch_size_used=int(peak_batch_size),
            )
        )
        total_windows += len(windows)
        total_fitted_windows += fitted_window_count
        total_skipped_windows += skipped_window_count

    final_predicted_nonseam_vertex_count = int((working_provenance == 1).sum())
    final_seam_vertex_count = int((working_provenance == 2).sum())

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
        "volume_uri": str(volume_uri),
        "dino_backbone": str(dino_backbone),
        "autoreg_checkpoint": str(autoreg_checkpoint),
        "original_vertex_count": int(np.isfinite(grid_zyx).all(axis=-1).sum()),
        "final_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum()),
        "predicted_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum() - np.isfinite(grid_zyx).all(axis=-1).sum()),
        "cumulative_predicted_vertex_count": int(np.isfinite(working_grid).all(axis=-1).sum() - np.isfinite(grid_zyx).all(axis=-1).sum()),
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
        "iterations_completed": int(len(iteration_stats)),
        "stop_reason": stop_reason,
        "windows_per_second_overall": (
            float(total_fitted_windows) * 1000.0 / max(1e-6, 1000.0 * (perf_counter() - total_started))
        ),
        "trimmed_bbox_rc": list(trimmed_bbox_rc),
        "crop_cache_hits": int(cache.hits),
        "crop_cache_misses": int(cache.misses),
        "volume_shape": list(volume_shape),
    }
    summary_path = out_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    summary["summary_path"] = str(summary_path)
    return summary


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
@click.option("--grow-direction", type=click.Choice(["auto", "left", "right", "up", "down"]), default="auto", show_default=True)
@click.option("--prompt-strips", type=int, default=8, show_default=True)
@click.option("--predict-strips-per-iter", type=int, default=8, show_default=True)
@click.option("--window-strip-length", type=int, default=64, show_default=True)
@click.option("--window-overlap", type=int, default=16, show_default=True)
@click.option("--window-batch-size", type=int, default=4, show_default=True)
@click.option("--max-extension-iters", type=int, default=4, show_default=True)
@click.option("--max-crop-fit-retries", type=int, default=3, show_default=True)
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
    max_crop_fit_retries: int,
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
        )
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
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
