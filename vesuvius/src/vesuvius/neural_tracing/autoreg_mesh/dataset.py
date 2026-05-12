from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    DIRECTION_TO_ID,
    IGNORE_INDEX,
    downsample_surface_grid,
    extract_frontier_prompt_band,
    serialize_split_conditioning_example,
)
from vesuvius.neural_tracing.datasets.common import _read_volume_crop_from_patch, _trim_to_world_bbox
from vesuvius.neural_tracing.datasets.dataset_defaults import (
    setdefault_rowcol_cond_dataset_config,
    validate_rowcol_cond_dataset_config,
)
from vesuvius.neural_tracing.datasets.dataset_rowcol_cond import EdtSegDataset
from vesuvius.neural_tracing.datasets.triplet_resampling import choose_replacement_index


def _lookup_cached_vol_tokens(cache: Any, sample_key):
    if cache is None:
        return None
    if callable(cache):
        return cache(sample_key)
    if hasattr(cache, "get"):
        return cache.get(sample_key, None)
    return None


def _spatial_axes_equal(crop_size: tuple[int, int, int], axes: list[int]) -> bool:
    if len(axes) < 2:
        return False
    reference = int(crop_size[axes[0]])
    return all(int(crop_size[axis]) == reference for axis in axes[1:])


def _normalize_surface_downsample_factor(surface_downsample_factor) -> tuple[int, int]:
    if isinstance(surface_downsample_factor, int):
        factor = max(1, int(surface_downsample_factor))
        return factor, factor
    if not isinstance(surface_downsample_factor, (list, tuple)) or len(surface_downsample_factor) != 2:
        raise ValueError(
            "surface_downsample_factor must be an int or length-2 sequence when used in dataset planning, "
            f"got {surface_downsample_factor!r}"
        )
    return max(1, int(surface_downsample_factor[0])), max(1, int(surface_downsample_factor[1]))


def _maybe_collapse_downsample_factor(factor_row: int, factor_col: int) -> int | tuple[int, int]:
    if int(factor_row) == int(factor_col):
        return int(factor_row)
    return (int(factor_row), int(factor_col))


def _sample_key(patch_idx: int, wrap_idx: int) -> tuple[int, int]:
    return int(patch_idx), int(wrap_idx)


def _reachable_conditioning_counts(axis_size: int, low: float, high: float) -> tuple[int, ...]:
    axis_size = int(axis_size)
    if axis_size < 2:
        return tuple()
    p_low = float(min(low, high))
    p_high = float(max(low, high))
    counts = []
    for count in range(1, axis_size):
        interval_low = max(p_low, (float(count) - 0.5) / float(axis_size))
        interval_high = min(p_high, (float(count) + 0.5) / float(axis_size))
        if interval_low <= interval_high:
            counts.append(int(count))
    if not counts:
        default_count = min(max(int(round(axis_size * p_low)), 1), axis_size - 1)
        counts.append(int(default_count))
    return tuple(counts)


def _split_surface_grid(
    surface_zyx: np.ndarray,
    *,
    direction: str,
    conditioning_count: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    surface = np.asarray(surface_zyx, dtype=np.float32)
    h_grid, w_grid = surface.shape[:2]
    conditioning_count = int(conditioning_count)
    if direction == "left":
        split_col = min(max(conditioning_count, 1), w_grid - 1)
        return surface[:, :split_col, :], surface[:, split_col:, :]
    if direction == "right":
        split_col = w_grid - min(max(conditioning_count, 1), w_grid - 1)
        return surface[:, split_col:, :], surface[:, :split_col, :]
    if direction == "up":
        split_row = min(max(conditioning_count, 1), h_grid - 1)
        return surface[:split_row, :, :], surface[split_row:, :, :]
    if direction == "down":
        split_row = h_grid - min(max(conditioning_count, 1), h_grid - 1)
        return surface[split_row:, :, :], surface[:split_row, :, :]
    return None, None


def _in_bounds_vertex_mask(grid_local: np.ndarray, *, volume_shape: tuple[int, int, int]) -> np.ndarray:
    grid = np.asarray(grid_local, dtype=np.float32)
    valid = np.isfinite(grid).all(axis=-1)
    for axis, size in enumerate(volume_shape):
        valid &= grid[..., axis] >= 0.0
        valid &= grid[..., axis] < float(size)
    return valid


def _frontier_band_mask(mask: np.ndarray, *, direction: str, band_width: int = 1) -> np.ndarray:
    band = max(1, int(band_width))
    if direction == "left":
        band = min(band, int(mask.shape[1]))
        return mask[:, :band]
    if direction == "right":
        band = min(band, int(mask.shape[1]))
        return mask[:, -band:]
    if direction == "up":
        band = min(band, int(mask.shape[0]))
        return mask[:band, :]
    if direction == "down":
        band = min(band, int(mask.shape[0]))
        return mask[-band:, :]
    raise ValueError(f"unsupported direction {direction!r}")


def _compute_target_boundary_stats(
    target_grid_local: np.ndarray,
    *,
    volume_shape: tuple[int, int, int],
    direction: str,
    frontier_band_width: int = 1,
    prompt_valid_fraction: float | None = None,
) -> dict[str, float | bool]:
    valid_mask = _in_bounds_vertex_mask(target_grid_local, volume_shape=volume_shape)
    invalid_mask = ~valid_mask
    frontier_invalid = _frontier_band_mask(invalid_mask, direction=str(direction), band_width=int(frontier_band_width))
    frontier_invalid_fraction = float(frontier_invalid.mean()) if frontier_invalid.size > 0 else 1.0
    target_invalid_fraction = float(invalid_mask.mean()) if invalid_mask.size > 0 else 1.0
    return {
        "target_invalid_fraction": target_invalid_fraction,
        "frontier_invalid_fraction": frontier_invalid_fraction,
        "touches_crop_boundary": bool(invalid_mask.any()),
        "prompt_valid_fraction": None if prompt_valid_fraction is None else float(prompt_valid_fraction),
    }


def _should_reject_boundary_stats(
    boundary_stats: dict[str, float | bool],
    *,
    max_target_invalid_fraction: float = 0.75,
    max_frontier_invalid_fraction: float = 0.5,
) -> tuple[bool, str | None]:
    prompt_valid_fraction = boundary_stats.get("prompt_valid_fraction")
    if prompt_valid_fraction is not None and float(prompt_valid_fraction) <= 0.0:
        return True, "no_valid_prompt"
    if float(boundary_stats["target_invalid_fraction"]) >= 1.0:
        return True, "target_all_invalid"
    if float(boundary_stats["target_invalid_fraction"]) > float(max_target_invalid_fraction):
        return True, "target_invalid_fraction"
    if float(boundary_stats["frontier_invalid_fraction"]) > float(max_frontier_invalid_fraction):
        return True, "frontier_invalid_fraction"
    return False, None


def _sample_frontier_band_width(config: dict) -> int:
    choices = config.get("frontier_band_width_choices")
    if choices is None:
        return int(config["frontier_band_width"])
    return int(random.choice(list(choices)))


def _restore_single_prompt_strip(
    corrupted_cond_local: np.ndarray,
    original_cond_local: np.ndarray,
    *,
    direction: str,
    frontier_band_width: int,
) -> np.ndarray:
    restored = np.asarray(corrupted_cond_local, dtype=np.float32).copy()
    if direction == "left":
        strip_idx = max(0, restored.shape[1] - int(frontier_band_width))
        restored[:, strip_idx, :] = original_cond_local[:, strip_idx, :]
    elif direction == "right":
        strip_idx = min(restored.shape[1] - 1, int(frontier_band_width) - 1)
        restored[:, strip_idx, :] = original_cond_local[:, strip_idx, :]
    elif direction == "up":
        strip_idx = max(0, restored.shape[0] - int(frontier_band_width))
        restored[strip_idx, :, :] = original_cond_local[strip_idx, :, :]
    elif direction == "down":
        strip_idx = min(restored.shape[0] - 1, int(frontier_band_width) - 1)
        restored[strip_idx, :, :] = original_cond_local[strip_idx, :, :]
    else:
        raise ValueError(f"unsupported direction {direction!r}")
    return restored


def _apply_ragged_frontier_augmentation(
    cond_local: np.ndarray,
    *,
    direction: str,
    frontier_band_width: int,
    ragged_frontier_prob: float,
    ragged_frontier_max_inset: int,
    ragged_frontier_gap_length_choices: list[int] | None,
    ragged_frontier_lowfreq_sigma: float,
) -> tuple[np.ndarray, bool]:
    cond = np.asarray(cond_local, dtype=np.float32).copy()
    if float(ragged_frontier_prob) <= 0.0 or int(ragged_frontier_max_inset) <= 0:
        return cond, False
    if random.random() >= float(ragged_frontier_prob):
        return cond, False
    original = cond.copy()
    if direction in {"left", "right"}:
        frontier_len = int(cond.shape[0])
        depth = int(cond.shape[1])
    else:
        frontier_len = int(cond.shape[1])
        depth = int(cond.shape[0])
    if frontier_len <= 0 or depth <= 1:
        return cond, False

    field = np.random.rand(frontier_len).astype(np.float32)
    field = ndimage.gaussian_filter1d(field, sigma=float(ragged_frontier_lowfreq_sigma), mode="nearest")
    field = field - float(field.min())
    field = field / max(float(field.max()), 1e-6)
    inset = np.rint(field * float(ragged_frontier_max_inset)).astype(np.int64)
    inset = np.clip(inset, 0, max(0, depth - 1))

    gap_choices = list(ragged_frontier_gap_length_choices or [])
    if gap_choices:
        span_count = 1 + int(random.random() < 0.35)
        for _ in range(span_count):
            span = int(random.choice(gap_choices))
            if span <= 0:
                continue
            start = random.randint(0, max(0, frontier_len - 1))
            end = min(frontier_len, start + span)
            inset[start:end] = np.maximum(inset[start:end], max(1, min(depth - 1, int(ragged_frontier_max_inset))))

    if direction == "left":
        for row_idx, amount in enumerate(inset.tolist()):
            if amount > 0:
                cond[row_idx, depth - int(amount):, :] = np.nan
    elif direction == "right":
        for row_idx, amount in enumerate(inset.tolist()):
            if amount > 0:
                cond[row_idx, :int(amount), :] = np.nan
    elif direction == "up":
        for col_idx, amount in enumerate(inset.tolist()):
            if amount > 0:
                cond[depth - int(amount):, col_idx, :] = np.nan
    elif direction == "down":
        for col_idx, amount in enumerate(inset.tolist()):
            if amount > 0:
                cond[:int(amount), col_idx, :] = np.nan
    else:
        raise ValueError(f"unsupported direction {direction!r}")

    prompt_grid = extract_frontier_prompt_band(cond, direction=str(direction), frontier_band_width=int(frontier_band_width))
    if not bool(np.isfinite(prompt_grid).all(axis=-1).any()):
        cond = _restore_single_prompt_strip(
            cond,
            original,
            direction=str(direction),
            frontier_band_width=int(frontier_band_width),
        )
    return cond, True


def _prefilter_prompt_valid_fraction(
    cond_local: np.ndarray,
    *,
    direction: str,
    frontier_band_width: int,
    volume_shape: tuple[int, int, int],
) -> float:
    prompt_grid = extract_frontier_prompt_band(
        cond_local,
        direction=str(direction),
        frontier_band_width=int(frontier_band_width),
    )
    prompt_valid_mask = _in_bounds_vertex_mask(prompt_grid, volume_shape=volume_shape)
    return float(prompt_valid_mask.mean()) if prompt_valid_mask.size > 0 else 0.0


def _evaluate_prefilter_trial(
    cond_local: np.ndarray,
    masked_local: np.ndarray,
    *,
    direction: str,
    frontier_band_width: int,
    volume_shape: tuple[int, int, int],
    max_frontier_invalid_fraction: float,
    max_target_invalid_fraction: float,
) -> tuple[bool, dict[str, float | bool], str | None]:
    prompt_valid_fraction = _prefilter_prompt_valid_fraction(
        cond_local,
        direction=str(direction),
        frontier_band_width=int(frontier_band_width),
        volume_shape=volume_shape,
    )
    if prompt_valid_fraction <= 0.0:
        stats = _compute_target_boundary_stats(
            masked_local,
            volume_shape=volume_shape,
            direction=str(direction),
            frontier_band_width=int(frontier_band_width),
            prompt_valid_fraction=0.0,
        )
        reject, reason = _should_reject_boundary_stats(
            stats,
            max_target_invalid_fraction=float(max_target_invalid_fraction),
            max_frontier_invalid_fraction=float(max_frontier_invalid_fraction),
        )
        return False, stats, reason or "no_valid_prompt"
    target_valid_fraction = float(_in_bounds_vertex_mask(masked_local, volume_shape=volume_shape).mean()) if masked_local.size > 0 else 0.0
    stats = _compute_target_boundary_stats(
        masked_local,
        volume_shape=volume_shape,
        direction=str(direction),
        frontier_band_width=int(frontier_band_width),
        prompt_valid_fraction=prompt_valid_fraction,
    )
    if target_valid_fraction <= 0.0:
        reject, reason = _should_reject_boundary_stats(
            stats,
            max_target_invalid_fraction=float(max_target_invalid_fraction),
            max_frontier_invalid_fraction=float(max_frontier_invalid_fraction),
        )
        if reason == "target_all_invalid":
            return False, stats, "target_all_invalid"
        return False, stats, "no_valid_target" if bool(reject) else None
    reject, reason = _should_reject_boundary_stats(
        stats,
        max_target_invalid_fraction=float(max_target_invalid_fraction),
        max_frontier_invalid_fraction=float(max_frontier_invalid_fraction),
    )
    return not bool(reject), stats, reason


def _tested_frontier_band_widths(config: dict) -> list[int]:
    choices = config.get("frontier_band_width_choices")
    if choices is None:
        return [int(config["frontier_band_width"])]
    return sorted({int(v) for v in choices})


def _normalize_prefilter_sampling_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(float(v) for v in weights.values()))
    return {str(key): float(value) / max(total, 1e-8) for key, value in weights.items()}


def _difficulty_bucket_from_stats(
    *,
    tested_widths: list[int],
    valid_prompt_widths_clean: list[int],
    valid_prompt_widths_ragged: list[int],
    ragged_valid_count: int,
    prefilter_ragged_trials: int,
) -> str:
    tested_widths = sorted({int(width) for width in tested_widths})
    baseline_frontier_band_width = min(tested_widths) if tested_widths else None
    valid_prompt_widths_clean = sorted({int(width) for width in valid_prompt_widths_clean})
    valid_prompt_widths_ragged = sorted({int(width) for width in valid_prompt_widths_ragged})
    clean_any = len(valid_prompt_widths_clean) > 0
    ragged_any = len(valid_prompt_widths_ragged) > 0
    if (not clean_any) and (not ragged_any):
        return "impossible"
    if ragged_any and int(prefilter_ragged_trials) > 0 and int(ragged_valid_count) < int(prefilter_ragged_trials):
        return "hard"
    if not clean_any:
        return "hard"
    if (
        baseline_frontier_band_width is not None and
        baseline_frontier_band_width in valid_prompt_widths_clean and
        len(valid_prompt_widths_clean) == len(tested_widths)
    ):
        return "easy"
    return "medium"


def _mean_or_one(values: list[float], *, default: float = 1.0) -> float:
    if not values:
        return float(default)
    return float(sum(float(v) for v in values) / float(len(values)))


def _fast_prefilter_plan(
    surface_local: np.ndarray,
    *,
    direction: str,
    conditioning_count: int,
    frontier_band_width: int,
    surface_downsample_factor,
    volume_shape: tuple[int, int, int],
    max_frontier_invalid_fraction: float = 0.5,
    max_target_invalid_fraction: float = 0.75,
) -> tuple[bool, dict[str, float | bool] | None, str | None]:
    cond_local, masked_local = _split_surface_grid(
        surface_local,
        direction=str(direction),
        conditioning_count=int(conditioning_count),
    )
    if cond_local is None or masked_local is None:
        return False, None, "split_failed"
    if cond_local.size == 0 or masked_local.size == 0:
        return False, None, "empty_split"

    cond_local = downsample_surface_grid(
        cond_local,
        direction=str(direction),
        surface_downsample_factor=surface_downsample_factor,
    )
    masked_local = downsample_surface_grid(
        masked_local,
        direction=str(direction),
        surface_downsample_factor=surface_downsample_factor,
    )
    if cond_local.size == 0 or masked_local.size == 0:
        return False, None, "empty_downsampled_split"

    is_valid, boundary_stats, reject_reason = _evaluate_prefilter_trial(
        cond_local,
        masked_local,
        direction=str(direction),
        volume_shape=volume_shape,
        frontier_band_width=int(frontier_band_width),
        max_frontier_invalid_fraction=float(max_frontier_invalid_fraction),
        max_target_invalid_fraction=float(max_target_invalid_fraction),
    )
    return bool(is_valid), boundary_stats, reject_reason


def _apply_volume_only_augmentation(volume: np.ndarray, augmentation_cfg: dict, *, enabled: bool) -> np.ndarray:
    volume_np = np.asarray(volume, dtype=np.float32)
    if not bool(enabled):
        return volume_np.astype(np.float32, copy=True)

    vol = volume_np.astype(np.float32, copy=True)
    finite = np.isfinite(vol)
    if not bool(finite.any()):
        return vol

    def _value_range() -> tuple[float, float, float]:
        low = float(vol[finite].min())
        high = float(vol[finite].max())
        return low, high, max(high - low, 1e-6)

    if random.random() < float(augmentation_cfg.get("contrast_prob", 0.0)):
        mean = float(vol[finite].mean())
        factor = random.uniform(0.75, 1.25)
        vol = (vol - mean) * factor + mean

    if random.random() < float(augmentation_cfg.get("mult_brightness_prob", 0.0)):
        vol = vol * random.uniform(0.8, 1.2)

    if random.random() < float(augmentation_cfg.get("add_brightness_prob", 0.0)):
        _low, _high, value_range = _value_range()
        vol = vol + random.uniform(-0.1, 0.1) * value_range

    if random.random() < float(augmentation_cfg.get("gamma_prob", 0.0)):
        low, _high, value_range = _value_range()
        normalized = np.clip((vol - low) / value_range, 0.0, 1.0)
        gamma = random.uniform(0.7, 1.5)
        vol = low + (normalized ** gamma) * value_range

    if random.random() < float(augmentation_cfg.get("gaussian_noise_prob", 0.0)):
        _low, _high, value_range = _value_range()
        noise_std = random.uniform(0.01, 0.04) * value_range
        vol = vol + np.random.normal(0.0, noise_std, size=vol.shape).astype(np.float32)

    if random.random() < float(augmentation_cfg.get("gaussian_blur_prob", 0.0)):
        sigma = random.uniform(0.35, 0.9)
        vol = ndimage.gaussian_filter(vol, sigma=sigma, mode="nearest").astype(np.float32, copy=False)

    if random.random() < float(augmentation_cfg.get("slice_illumination_prob", 0.0)) and int(vol.shape[0]) > 1:
        depth = int(vol.shape[0])
        anchors = min(4, depth)
        anchor_z = np.linspace(0.0, float(depth - 1), num=anchors, dtype=np.float32)
        anchor_scale = 1.0 + np.random.uniform(-0.15, 0.15, size=anchors).astype(np.float32)
        slice_scale = np.interp(np.arange(depth, dtype=np.float32), anchor_z, anchor_scale).astype(np.float32)
        vol = vol * slice_scale[:, None, None]

    if random.random() < float(augmentation_cfg.get("lowres_prob", 0.0)):
        scale = random.uniform(0.5, 0.85)
        tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)
        lowres_shape = [max(2, int(round(size * scale))) for size in vol.shape]
        tensor = F.interpolate(tensor, size=lowres_shape, mode="trilinear", align_corners=False)
        tensor = F.interpolate(tensor, size=list(vol.shape), mode="trilinear", align_corners=False)
        vol = tensor.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)

    return np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _apply_spatial_augmentation(
    volume: np.ndarray,
    *,
    cond_local: np.ndarray,
    masked_local: np.ndarray,
    crop_size: tuple[int, int, int],
    augmentation_cfg: dict,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    volume_np = np.asarray(volume, dtype=np.float32)
    cond_np = np.asarray(cond_local, dtype=np.float32)
    masked_np = np.asarray(masked_local, dtype=np.float32)
    metadata = {
        "spatial_augmented": False,
        "spatial_mirror_axes": [],
        "spatial_axis_order": [0, 1, 2],
    }
    if not bool(enabled):
        return volume_np.astype(np.float32, copy=True), cond_np.copy(), masked_np.copy(), metadata

    cond_shape = cond_np.shape[:2]
    masked_shape = masked_np.shape[:2]
    cond_count = int(np.prod(cond_shape))
    keypoints = torch.from_numpy(
        np.concatenate([cond_np.reshape(-1, 3), masked_np.reshape(-1, 3)], axis=0).astype(np.float32)
    )
    data_dict = {
        "image": torch.from_numpy(volume_np[None].astype(np.float32)),
        "keypoints": keypoints,
        "crop_shape": crop_size,
    }

    mirror_prob = float(augmentation_cfg.get("mirror_prob", 0.5))
    mirror_axes = [int(axis) for axis in augmentation_cfg.get("mirror_axes", [0, 1, 2])]
    if mirror_prob > 0.0 and torch.rand(1).item() < mirror_prob:
        mirror_transform = MirrorTransform(tuple(mirror_axes))
        params = mirror_transform.get_parameters(**data_dict)
        data_dict = mirror_transform.apply(data_dict, **params)
        metadata["spatial_augmented"] = metadata["spatial_augmented"] or bool(params["axes"])
        metadata["spatial_mirror_axes"] = [int(axis) for axis in params["axes"]]

    transpose_prob = float(augmentation_cfg.get("transpose_prob", 0.25))
    transpose_axes = [int(axis) for axis in augmentation_cfg.get("transpose_axes", [0, 1, 2])]
    if transpose_prob > 0.0 and torch.rand(1).item() < transpose_prob and _spatial_axes_equal(crop_size, transpose_axes):
        transpose_transform = TransposeAxesTransform(set(transpose_axes))
        params = transpose_transform.get_parameters(**data_dict)
        data_dict = transpose_transform.apply(data_dict, **params)
        axis_order = [int(axis) - 1 for axis in params["axis_order"][1:]]
        metadata["spatial_augmented"] = metadata["spatial_augmented"] or axis_order != [0, 1, 2]
        metadata["spatial_axis_order"] = axis_order

    keypoints_out = data_dict["keypoints"].detach().cpu().numpy().astype(np.float32, copy=False)
    cond_out = keypoints_out[:cond_count].reshape(*cond_shape, 3)
    masked_out = keypoints_out[cond_count:].reshape(*masked_shape, 3)
    volume_out = data_dict["image"].squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return volume_out, cond_out, masked_out, metadata


# Direction remap under a UV-grid (rows<->cols) transpose. Volume axes are
# UNCHANGED by this augmentation; only the 2D UV parametrization layout
# flips, which permutes how the model interprets the band edge in
# extract_frontier_prompt_band (serialization.py):
#   - "left"  reads cond[:, -band:, :] (rightmost UV cols)
#   - "right" reads cond[:, :band, :]  (leftmost UV cols)
#   - "up"    reads cond[-band:, :, :] (bottom UV rows)
#   - "down"  reads cond[:band, :, :]  (top UV rows)
# After a UV transpose (cond.swapaxes(0,1)), what were "cols" become "rows",
# so left <-> up and right <-> down. This is a 4-cycle on the direction
# token; left+up and right+down swap their roles bijectively.
_UV_TRANSPOSE_DIRECTION_REMAP = {
    "left": "up",
    "right": "down",
    "up": "left",
    "down": "right",
}


def _apply_uv_grid_augmentation(
    cond_local: np.ndarray,
    masked_local: np.ndarray,
    *,
    direction: str,
    augmentation_cfg: dict,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, object]]:
    """UV-grid (2D parametrization) augmentation with consistent direction remap.

    Currently supports a single UV operation: rows<->cols transpose. The
    volume and the world-coord channels of `cond_local`/`masked_local` are
    NOT touched here (those are handled in `_apply_spatial_augmentation`).
    Only the UV layout flips, so we remap the direction token accordingly.

    Skipped when the implied full UV grid (cond+masked stacked along the
    split axis) is non-square, since the transpose would produce an
    incompatible shape for the existing model code paths that consume the
    band along a specific axis.
    """
    metadata: dict[str, object] = {"uv_transposed": False}
    direction_in = str(direction).lower()
    if not bool(enabled):
        return cond_local, masked_local, direction_in, metadata
    uv_transpose_prob = float(augmentation_cfg.get("uv_transpose_prob", 0.0))
    if uv_transpose_prob <= 0.0 or torch.rand(1).item() >= uv_transpose_prob:
        return cond_local, masked_local, direction_in, metadata
    if direction_in not in _UV_TRANSPOSE_DIRECTION_REMAP:
        return cond_local, masked_local, direction_in, metadata
    cond_arr = np.asarray(cond_local)
    masked_arr = np.asarray(masked_local)
    if cond_arr.ndim != 3 or cond_arr.shape[-1] != 3:
        return cond_local, masked_local, direction_in, metadata
    if masked_arr.ndim != 3 or masked_arr.shape[-1] != 3:
        return cond_local, masked_local, direction_in, metadata
    # Reconstruct the full UV shape: cond and masked are split along the
    # axis implied by direction. For left/right the split is on UV-W (cols);
    # for up/down it is on UV-H (rows). We require the FULL surface
    # (cond + masked along the split axis) to be square so swapaxes(0,1)
    # produces a meaningful direction in the swapped frame.
    if direction_in in {"left", "right"}:
        full_h = int(cond_arr.shape[0])
        full_w = int(cond_arr.shape[1]) + int(masked_arr.shape[1])
    else:  # up / down
        full_h = int(cond_arr.shape[0]) + int(masked_arr.shape[0])
        full_w = int(cond_arr.shape[1])
    if full_h != full_w:
        return cond_local, masked_local, direction_in, metadata
    new_cond = cond_arr.swapaxes(0, 1).copy()
    new_masked = masked_arr.swapaxes(0, 1).copy()
    new_direction = _UV_TRANSPOSE_DIRECTION_REMAP[direction_in]
    metadata["uv_transposed"] = True
    return new_cond, new_masked, new_direction, metadata


def extract_wrap_world_surface_stored(patch, wrap: dict, *, require_all_valid: bool = True) -> np.ndarray | None:
    """Extract one wrap directly on the stored tifxyz lattice in world ZYX coordinates."""
    seg = wrap["segment"]
    r_min, r_max, c_min, c_max = wrap["bbox_2d"]

    seg_h, seg_w = seg._valid_mask.shape
    r_min = max(0, int(r_min))
    r_max = min(seg_h - 1, int(r_max))
    c_min = max(0, int(c_min))
    c_max = min(seg_w - 1, int(c_max))
    if r_max < r_min or c_max < c_min:
        return None

    seg.use_stored_resolution()
    x_s, y_s, z_s, valid_s = seg[r_min:r_max + 1, c_min:c_max + 1]
    if x_s.size == 0:
        return None
    if valid_s is not None:
        if require_all_valid and not bool(valid_s.all()):
            return None
        if not require_all_valid and not bool(valid_s.any()):
            return None

    trimmed = _trim_to_world_bbox(
        np.asarray(x_s, dtype=np.float32),
        np.asarray(y_s, dtype=np.float32),
        np.asarray(z_s, dtype=np.float32),
        patch.world_bbox,
    )
    if trimmed is None:
        return None
    x_trim, y_trim, z_trim = trimmed
    return np.stack([z_trim, y_trim, x_trim], axis=-1).astype(np.float32, copy=False)


def create_split_conditioning_from_surface_grid(
    dataset,
    *,
    idx: int,
    patch_idx: int,
    wrap_idx: int,
    patch,
    surface_zyx: np.ndarray,
    conditioning_percent: float | None = None,
    conditioning_count: int | None = None,
    cond_direction: str | None = None,
) -> dict | None:
    """Mirror create_split_conditioning using a caller-provided surface grid."""
    _ = idx
    wrap = patch.wraps[wrap_idx]
    seg = wrap["segment"]
    r_min, r_max, c_min, c_max = wrap["bbox_2d"]

    seg_h, seg_w = seg._valid_mask.shape
    r_min = max(0, int(r_min))
    r_max = min(seg_h - 1, int(r_max))
    c_min = max(0, int(c_min))
    c_max = min(seg_w - 1, int(c_max))
    if r_max < r_min or c_max < c_min:
        return None

    surface = np.asarray(surface_zyx, dtype=np.float32)
    if surface.ndim != 3 or surface.shape[-1] != 3:
        raise ValueError(f"surface_zyx must have shape [H, W, 3], got {surface.shape!r}")

    h_grid, w_grid = surface.shape[:2]
    if h_grid < 2 and w_grid < 2:
        return None

    valid_directions = []
    if w_grid >= 2:
        valid_directions.extend(["left", "right"])
    if h_grid >= 2:
        valid_directions.extend(["up", "down"])
    if not valid_directions:
        return None

    if cond_direction is None:
        cond_direction = random.choice(valid_directions)
    cond_direction = str(cond_direction).lower()
    if cond_direction not in valid_directions:
        return None

    if conditioning_count is None:
        if conditioning_percent is None:
            conditioning_percent = random.uniform(dataset._cond_percent_min, dataset._cond_percent_max)
        conditioning_percent = float(conditioning_percent)
        r_cond = int(round(h_grid * conditioning_percent))
        c_cond = int(round(w_grid * conditioning_percent))
        if h_grid >= 2:
            r_cond = min(max(r_cond, 1), h_grid - 1)
        if w_grid >= 2:
            c_cond = min(max(c_cond, 1), w_grid - 1)
    else:
        conditioning_count = int(conditioning_count)
        if cond_direction in {"left", "right"}:
            c_cond = min(max(conditioning_count, 1), w_grid - 1)
            r_cond = min(max(int(round(h_grid * float(dataset._cond_percent_min))), 1), max(h_grid - 1, 1)) if h_grid >= 2 else 0
            conditioning_percent = float(c_cond) / float(max(w_grid, 1))
        else:
            r_cond = min(max(conditioning_count, 1), h_grid - 1)
            c_cond = min(max(int(round(w_grid * float(dataset._cond_percent_min))), 1), max(w_grid - 1, 1)) if w_grid >= 2 else 0
            conditioning_percent = float(r_cond) / float(max(h_grid, 1))

    if cond_direction == "left":
        cond_zyxs = surface[:, :c_cond, :]
        masked_zyxs = surface[:, c_cond:, :]
    elif cond_direction == "right":
        split_col = w_grid - c_cond
        cond_zyxs = surface[:, split_col:, :]
        masked_zyxs = surface[:, :split_col, :]
    elif cond_direction == "up":
        cond_zyxs = surface[:r_cond, :, :]
        masked_zyxs = surface[r_cond:, :, :]
    elif cond_direction == "down":
        split_row = h_grid - r_cond
        cond_zyxs = surface[split_row:, :, :]
        masked_zyxs = surface[:split_row, :, :]
    else:
        return None

    if cond_zyxs.size == 0 or masked_zyxs.size == 0:
        return None

    crop_size = dataset.crop_size
    z_min, _, y_min, _, x_min, _ = patch.world_bbox
    min_corner = np.round([z_min, y_min, x_min]).astype(np.int64)
    max_corner = min_corner + np.asarray(crop_size, dtype=np.int64)

    return {
        "wrap": wrap,
        "seg": seg,
        "r_min": r_min,
        "r_max": r_max,
        "c_min": c_min,
        "c_max": c_max,
        "conditioning_percent": conditioning_percent,
        "cond_direction": cond_direction,
        "cond_zyxs_unperturbed": cond_zyxs.copy(),
        "masked_zyxs": masked_zyxs.copy(),
        "min_corner": min_corner,
        "max_corner": max_corner,
    }


_active_prefilter_dataset: "AutoregMeshDataset | None" = None


def _prefilter_one_worker(sample_key):
    """Worker entry: returns (sample_key, plan_list) for one sample.

    Relies on `_active_prefilter_dataset` being set in the parent before fork
    so each worker inherits the dataset's read-only state via copy-on-write.
    """
    return sample_key, _active_prefilter_dataset._compute_one_plan_list(sample_key)


def _prefilter_stats_chunk_worker(keys_chunk):
    """Worker: compute partial stats over a chunk of valid_plans keys.

    Reads `_active_prefilter_dataset._valid_split_plans` (fork-inherited) and
    returns a partial-aggregate dict the parent reduces into the global stats.
    """
    ds = _active_prefilter_dataset
    plans_dict = ds._valid_split_plans
    difficulty_bucket_counts = {"easy": 0, "medium": 0, "hard": 0}
    clean_only_plan_count = 0
    ragged_only_plan_count = 0
    multi_width_plan_count = 0
    total_valid_width_count = 0
    total_plan_count = 0
    for key in keys_chunk:
        plans = plans_dict.get(key, ())
        for plan in plans:
            bucket = str(plan.get("difficulty_bucket", "medium"))
            if bucket in difficulty_bucket_counts:
                difficulty_bucket_counts[bucket] += 1
            clean_widths = list(plan.get("valid_prompt_widths_clean", []))
            ragged_widths = list(plan.get("valid_prompt_widths_ragged", []))
            if clean_widths and not ragged_widths:
                clean_only_plan_count += 1
            if ragged_widths and not clean_widths:
                ragged_only_plan_count += 1
            valid_width_count = len(set(clean_widths + ragged_widths))
            if valid_width_count > 1:
                multi_width_plan_count += 1
            total_valid_width_count += int(valid_width_count)
            total_plan_count += 1
    return {
        "chunk_size": len(keys_chunk),
        "difficulty_bucket_counts": difficulty_bucket_counts,
        "clean_only_plan_count": clean_only_plan_count,
        "ragged_only_plan_count": ragged_only_plan_count,
        "multi_width_plan_count": multi_width_plan_count,
        "total_valid_width_count": total_valid_width_count,
        "total_plan_count": total_plan_count,
    }


class AutoregMeshDataset(Dataset):
    """Build autoregressive wrap-completion examples from split conditioning."""

    def __init__(
        self,
        config: dict,
        *,
        patch_metadata=None,
        apply_augmentation: bool = False,
        apply_perturbation: bool = False,
        apply_spatial_augmentation: bool | None = None,
        apply_volume_only_augmentation: bool | None = None,
        max_resample_attempts: int = 8,
    ) -> None:
        self.config = validate_autoreg_mesh_config(config)
        self.max_resample_attempts = int(max_resample_attempts)
        self.apply_spatial_augmentation = bool(
            self.config.get("spatial_augmentation", {}).get("enabled", False)
            if apply_spatial_augmentation is None
            else apply_spatial_augmentation
        )
        self.apply_volume_only_augmentation = bool(
            self.config.get("volume_only_augmentation", {}).get("enabled", False)
            if apply_volume_only_augmentation is None
            else apply_volume_only_augmentation
        )

        base_config = deepcopy(config)
        base_config["sample_mode"] = "wrap"
        setdefault_rowcol_cond_dataset_config(base_config)
        validate_rowcol_cond_dataset_config(base_config)

        _t_init_start = time.perf_counter()
        _t = _t_init_start
        self._base_dataset = EdtSegDataset(
            base_config,
            apply_augmentation=bool(apply_augmentation),
            apply_perturbation=bool(apply_perturbation),
            patch_metadata=patch_metadata,
        )
        self.patches = self._base_dataset.patches
        self.sample_index = list(self._base_dataset.sample_index)
        self.crop_size = tuple(int(v) for v in self._base_dataset.crop_size)
        print(
            f"[autoreg_mesh dataset] base dataset ready in {time.perf_counter() - _t:.1f}s "
            f"({len(self.sample_index)} pre-filter samples)",
            flush=True,
        )
        _t = time.perf_counter()
        precomputed_plans = None
        if isinstance(patch_metadata, dict):
            precomputed_plans = patch_metadata.get("autoreg_mesh_valid_split_plans")
        sidecar_cache_dir: Path | None = None
        sidecar_cache_key: str | None = None
        if precomputed_plans is None:
            sidecar_cache_dir = self._prefilter_cache_path()
            sidecar_cache_key = self._prefilter_cache_key()
            self._valid_split_plans = self._build_valid_split_plans()
        else:
            self._valid_split_plans = precomputed_plans
        print(
            f"[autoreg_mesh dataset] valid_split_plans ready in {time.perf_counter() - _t:.1f}s "
            f"({len(self._valid_split_plans)} keys)",
            flush=True,
        )
        _t = time.perf_counter()
        self.sample_index = [
            _sample_key(*sample_key)
            for sample_key in self.sample_index
            if len(self._valid_split_plans.get(_sample_key(*sample_key), ())) > 0
        ]
        print(
            f"[autoreg_mesh dataset] sample_index post-filter in {time.perf_counter() - _t:.1f}s "
            f"({len(self.sample_index)} kept)",
            flush=True,
        )
        if not self.sample_index:
            raise RuntimeError("autoreg_mesh dataset prefilter removed every sample; no valid wrap splits remain")
        precomputed_stats: dict | None = None
        if isinstance(patch_metadata, dict):
            maybe_stats = patch_metadata.get("autoreg_mesh_prefilter_stats")
            if isinstance(maybe_stats, dict):
                precomputed_stats = dict(maybe_stats)
        self._prefilter_stats = self._load_or_compute_prefilter_stats(
            self._valid_split_plans,
            n_post_filter_samples=len(self.sample_index),
            cache_dir=sidecar_cache_dir,
            cache_key=sidecar_cache_key,
            precomputed_stats=precomputed_stats,
        )
        self._prefilter_stats["prefilter_wall_seconds"] = float(getattr(self, "_prefilter_wall_seconds", 0.0))
        print(
            f"[autoreg_mesh dataset] __init__ total {time.perf_counter() - _t_init_start:.1f}s "
            f"({len(self.sample_index)} samples)",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def export_patch_metadata(self):
        metadata = self._base_dataset.export_patch_metadata()
        metadata["autoreg_mesh_valid_split_plans"] = self._valid_split_plans
        metadata["autoreg_mesh_prefilter_stats"] = self._prefilter_stats
        return metadata

    def _extract_surface_for_plan(self, patch, wrap: dict, *, use_stored_surface: bool) -> np.ndarray | None:
        if bool(use_stored_surface):
            return extract_wrap_world_surface_stored(
                patch,
                wrap,
                require_all_valid=True,
            )
        return self._base_dataset._extract_wrap_world_surface(patch, wrap, require_all_valid=True)

    def _compute_one_plan_list(self, sample_key) -> list[dict]:
        patch_idx, wrap_idx = sample_key
        patch = self.patches[int(patch_idx)]
        wrap = patch.wraps[int(wrap_idx)]
        use_stored_surface, effective_surface_downsample_factor = self._resolve_surface_sampling_plan(wrap)
        surface_zyx = self._extract_surface_for_plan(patch, wrap, use_stored_surface=use_stored_surface)
        if surface_zyx is None:
            return []
        min_corner = np.round([patch.world_bbox[0], patch.world_bbox[2], patch.world_bbox[4]]).astype(np.float32)
        surface_local = np.asarray(surface_zyx, dtype=np.float32) - min_corner[None, None, :]
        h_grid, w_grid = surface_zyx.shape[:2]
        plan_list: list[dict] = []
        tested_widths = _tested_frontier_band_widths(self.config)
        directions: list[str] = []
        if w_grid >= 2:
            directions.extend(["left", "right"])
        if h_grid >= 2:
            directions.extend(["up", "down"])
        max_frontier_invalid = float(self.config.get("prefilter_max_frontier_invalid_fraction", 0.5))
        max_target_invalid = float(self.config.get("prefilter_max_target_invalid_fraction", 0.75))
        ragged_trials = int(self.config.get("prefilter_ragged_trials", 0))
        ragged_prob = float(self.config.get("ragged_frontier_prob", 0.0))
        ragged_max_inset = int(self.config.get("ragged_frontier_max_inset", 0))
        ragged_gap_choices = self.config.get("ragged_frontier_gap_length_choices")
        ragged_lowfreq_sigma = float(self.config.get("ragged_frontier_lowfreq_sigma", 12.0))
        for direction in directions:
            axis_size = w_grid if direction in {"left", "right"} else h_grid
            for conditioning_count in _reachable_conditioning_counts(
                axis_size,
                self._base_dataset._cond_percent_min,
                self._base_dataset._cond_percent_max,
            ):
                cond_local, masked_local = _split_surface_grid(
                    surface_local,
                    direction=str(direction),
                    conditioning_count=int(conditioning_count),
                )
                if cond_local is None or masked_local is None or cond_local.size == 0 or masked_local.size == 0:
                    continue
                cond_local = downsample_surface_grid(
                    cond_local,
                    direction=str(direction),
                    surface_downsample_factor=effective_surface_downsample_factor,
                )
                masked_local = downsample_surface_grid(
                    masked_local,
                    direction=str(direction),
                    surface_downsample_factor=effective_surface_downsample_factor,
                )
                if cond_local.size == 0 or masked_local.size == 0:
                    continue

                clean_frontier_values: list[float] = []
                clean_target_values: list[float] = []
                ragged_frontier_values: list[float] = []
                ragged_target_values: list[float] = []
                clean_reject_reason_counts: dict[str, int] = {}
                ragged_reject_reason_counts: dict[str, int] = {}
                valid_prompt_widths_clean: list[int] = []
                valid_prompt_widths_ragged: list[int] = []
                clean_valid_count = 0
                ragged_valid_count = 0

                for frontier_band_width in tested_widths:
                    clean_valid, clean_stats, clean_reason = _evaluate_prefilter_trial(
                        cond_local,
                        masked_local,
                        direction=str(direction),
                        frontier_band_width=int(frontier_band_width),
                        volume_shape=self.crop_size,
                        max_frontier_invalid_fraction=max_frontier_invalid,
                        max_target_invalid_fraction=max_target_invalid,
                    )
                    if clean_stats is not None:
                        clean_frontier_values.append(float(clean_stats["frontier_invalid_fraction"]))
                        clean_target_values.append(float(clean_stats["target_invalid_fraction"]))
                    if clean_valid:
                        valid_prompt_widths_clean.append(int(frontier_band_width))
                        clean_valid_count += 1
                    elif clean_reason is not None:
                        clean_reject_reason_counts[str(clean_reason)] = clean_reject_reason_counts.get(str(clean_reason), 0) + 1

                    if ragged_prob > 0.0 and ragged_trials > 0:
                        for _ in range(ragged_trials):
                            ragged_cond_local, ragged_applied = _apply_ragged_frontier_augmentation(
                                cond_local,
                                direction=str(direction),
                                frontier_band_width=int(frontier_band_width),
                                ragged_frontier_prob=1.0,
                                ragged_frontier_max_inset=ragged_max_inset,
                                ragged_frontier_gap_length_choices=ragged_gap_choices,
                                ragged_frontier_lowfreq_sigma=ragged_lowfreq_sigma,
                            )
                            if not bool(ragged_applied):
                                continue
                            ragged_valid, ragged_stats, ragged_reason = _evaluate_prefilter_trial(
                                ragged_cond_local,
                                masked_local,
                                direction=str(direction),
                                frontier_band_width=int(frontier_band_width),
                                volume_shape=self.crop_size,
                                max_frontier_invalid_fraction=max_frontier_invalid,
                                max_target_invalid_fraction=max_target_invalid,
                            )
                            if ragged_stats is not None:
                                ragged_frontier_values.append(float(ragged_stats["frontier_invalid_fraction"]))
                                ragged_target_values.append(float(ragged_stats["target_invalid_fraction"]))
                            if ragged_valid:
                                valid_prompt_widths_ragged.append(int(frontier_band_width))
                                ragged_valid_count += 1
                            elif ragged_reason is not None:
                                ragged_reject_reason_counts[str(ragged_reason)] = ragged_reject_reason_counts.get(str(ragged_reason), 0) + 1

                valid_prompt_widths_clean = sorted(set(valid_prompt_widths_clean))
                valid_prompt_widths_ragged = sorted(set(valid_prompt_widths_ragged))
                difficulty_bucket = _difficulty_bucket_from_stats(
                    tested_widths=tested_widths,
                    valid_prompt_widths_clean=valid_prompt_widths_clean,
                    valid_prompt_widths_ragged=valid_prompt_widths_ragged,
                    ragged_valid_count=int(ragged_valid_count),
                    prefilter_ragged_trials=ragged_trials,
                )
                if difficulty_bucket == "impossible":
                    continue
                plan_list.append(
                    {
                        "direction": str(direction),
                        "conditioning_count": int(conditioning_count),
                        "conditioning_percent": float(conditioning_count) / float(max(axis_size, 1)),
                        "use_stored_surface": bool(use_stored_surface),
                        "effective_surface_downsample_factor": effective_surface_downsample_factor,
                        "difficulty_bucket": str(difficulty_bucket),
                        "clean_frontier_invalid_fraction": min(clean_frontier_values) if clean_frontier_values else 1.0,
                        "clean_target_invalid_fraction": min(clean_target_values) if clean_target_values else 1.0,
                        "ragged_trial_frontier_invalid_fraction": _mean_or_one(ragged_frontier_values),
                        "ragged_trial_target_invalid_fraction": _mean_or_one(ragged_target_values),
                        "valid_prompt_widths_clean": valid_prompt_widths_clean,
                        "valid_prompt_widths_ragged": valid_prompt_widths_ragged,
                        "valid_prompt_width_range": (
                            [min(valid_prompt_widths_clean + valid_prompt_widths_ragged), max(valid_prompt_widths_clean + valid_prompt_widths_ragged)]
                            if (valid_prompt_widths_clean or valid_prompt_widths_ragged) else []
                        ),
                        "clean_valid_count": int(clean_valid_count),
                        "ragged_valid_count": int(ragged_valid_count),
                        "clean_reject_reason_counts": clean_reject_reason_counts,
                        "ragged_reject_reason_counts": ragged_reject_reason_counts,
                    }
                )
        return plan_list

    def _prefilter_cache_path(self) -> Path:
        first_segments_path = self.config["datasets"][0]["segments_path"]
        return Path(first_segments_path) / ".patch_cache"

    def _load_or_compute_prefilter_stats(
        self,
        valid_plans: dict,
        *,
        n_post_filter_samples: int,
        cache_dir: Path | None,
        cache_key: str | None,
        precomputed_stats: dict | None = None,
    ) -> dict:
        if precomputed_stats is not None:
            stats = dict(precomputed_stats)
            stats["num_samples_with_valid_plans"] = int(n_post_filter_samples)
            return stats
        sidecar_path: Path | None = None
        if cache_dir is not None and cache_key is not None:
            sidecar_path = cache_dir / f"valid_plans_stats_{cache_key}.json"
            if sidecar_path.exists():
                try:
                    with open(sidecar_path) as f:
                        cached = json.load(f)
                    if isinstance(cached, dict):
                        cached["num_samples_with_valid_plans"] = int(n_post_filter_samples)
                        print(
                            f"[autoreg_mesh dataset] prefilter stats sidecar HIT key={cache_key}",
                            flush=True,
                        )
                        return cached
                except (json.JSONDecodeError, OSError) as e:
                    print(
                        f"[autoreg_mesh dataset] prefilter stats sidecar unreadable, recomputing: {e}",
                        flush=True,
                    )
        n_workers_cfg = int(self.config.get("prefilter_num_workers", 0))
        if n_workers_cfg <= 0:
            n_workers_cfg = max(1, (os.cpu_count() or 4) // 2)
        keys = list(valid_plans.keys())
        n_workers = max(1, min(n_workers_cfg, len(keys)))
        print(
            f"[autoreg_mesh dataset] computing prefilter stats over {len(keys)} keys "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        t_stats = time.perf_counter()
        difficulty_bucket_counts = {"easy": 0, "medium": 0, "hard": 0}
        clean_only_plan_count = 0
        ragged_only_plan_count = 0
        multi_width_plan_count = 0
        total_valid_width_count = 0
        total_plan_count = 0
        show_progress = bool(self.config.get("prefilter_show_progress", True)) and len(keys) > 0
        if n_workers <= 1 or len(keys) <= 1024:
            iterator: Any = (
                tqdm(keys, desc="autoreg_mesh prefilter stats", unit="sample", leave=False)
                if show_progress
                else keys
            )
            # The chunk worker reads `_active_prefilter_dataset._valid_split_plans`;
            # set it for the single-process path too (the fork-pool branch below
            # sets it inside its own block, so this is symmetric).
            global _active_prefilter_dataset
            previous_active_sp = _active_prefilter_dataset
            _active_prefilter_dataset = self
            try:
                partials_iter = (_prefilter_stats_chunk_worker([key]) for key in iterator)
                for partial in partials_iter:
                    for k, v in partial["difficulty_bucket_counts"].items():
                        difficulty_bucket_counts[k] += int(v)
                    clean_only_plan_count += int(partial["clean_only_plan_count"])
                    ragged_only_plan_count += int(partial["ragged_only_plan_count"])
                    multi_width_plan_count += int(partial["multi_width_plan_count"])
                    total_valid_width_count += int(partial["total_valid_width_count"])
                    total_plan_count += int(partial["total_plan_count"])
            finally:
                _active_prefilter_dataset = previous_active_sp
        else:
            chunk_size = max(1, len(keys) // (n_workers * 8))
            chunks = [keys[i : i + chunk_size] for i in range(0, len(keys), chunk_size)]
            previous_active = _active_prefilter_dataset
            _active_prefilter_dataset = self
            try:
                try:
                    from threadpoolctl import threadpool_limits as _threadpool_limits
                    _tpl_ctx = _threadpool_limits(limits=1)
                except ImportError:
                    _tpl_ctx = None
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(n_workers) as pool:
                    iterator = pool.imap_unordered(_prefilter_stats_chunk_worker, chunks)
                    if show_progress:
                        pbar = tqdm(
                            total=len(keys),
                            desc=f"autoreg_mesh prefilter stats (n_workers={n_workers})",
                            unit="sample",
                            leave=False,
                        )
                    else:
                        pbar = None
                    for partial in iterator:
                        for k, v in partial["difficulty_bucket_counts"].items():
                            difficulty_bucket_counts[k] += int(v)
                        clean_only_plan_count += int(partial["clean_only_plan_count"])
                        ragged_only_plan_count += int(partial["ragged_only_plan_count"])
                        multi_width_plan_count += int(partial["multi_width_plan_count"])
                        total_valid_width_count += int(partial["total_valid_width_count"])
                        total_plan_count += int(partial["total_plan_count"])
                        if pbar is not None:
                            pbar.update(int(partial["chunk_size"]))
                            pbar.set_postfix(plans=total_plan_count)
                    if pbar is not None:
                        pbar.close()
            finally:
                _active_prefilter_dataset = previous_active
                if _tpl_ctx is not None:
                    _tpl_ctx.unregister()
        stats = {
            "num_samples_with_valid_plans": int(n_post_filter_samples),
            "num_total_valid_split_plans": int(total_plan_count),
            "difficulty_bucket_counts": difficulty_bucket_counts,
            "clean_only_plan_count": int(clean_only_plan_count),
            "ragged_only_plan_count": int(ragged_only_plan_count),
            "multi_width_plan_count": int(multi_width_plan_count),
            "avg_valid_prompt_width_count": (
                float(total_valid_width_count) / float(total_plan_count) if total_plan_count > 0 else 0.0
            ),
        }
        elapsed = time.perf_counter() - t_stats
        print(
            f"[autoreg_mesh dataset] prefilter stats computed in {elapsed:.1f}s",
            flush=True,
        )
        if sidecar_path is not None:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp = sidecar_path.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(stats, f, indent=2)
                os.replace(tmp, sidecar_path)
                print(
                    f"[autoreg_mesh dataset] prefilter stats sidecar saved: {sidecar_path}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[autoreg_mesh dataset] prefilter stats sidecar save FAILED: {e}",
                    flush=True,
                )
        return stats

    def _prefilter_cache_key(self) -> str:
        tested_widths = _tested_frontier_band_widths(self.config)
        sample_signature: list[tuple] = []
        for patch_idx, wrap_idx in sorted(self.sample_index):
            patch = self.patches[int(patch_idx)]
            wrap = patch.wraps[int(wrap_idx)]
            seg = wrap.get("segment")
            seg_uuid = getattr(seg, "uuid", None) if seg is not None else None
            sample_signature.append((int(patch_idx), int(wrap_idx), int(wrap.get("wrap_id", -1)), str(seg_uuid)))
        samples_md5 = hashlib.md5(json.dumps(sample_signature, sort_keys=True).encode()).hexdigest()
        cache_key_data = {
            "method": "autoreg_mesh_valid_plans_v1",
            "tested_widths": list(tested_widths),
            "prefilter_max_frontier_invalid_fraction": float(self.config.get("prefilter_max_frontier_invalid_fraction", 0.5)),
            "prefilter_max_target_invalid_fraction": float(self.config.get("prefilter_max_target_invalid_fraction", 0.75)),
            "prefilter_ragged_trials": int(self.config.get("prefilter_ragged_trials", 0)),
            "ragged_frontier_prob": float(self.config.get("ragged_frontier_prob", 0.0)),
            "ragged_frontier_max_inset": int(self.config.get("ragged_frontier_max_inset", 0)),
            "ragged_frontier_gap_length_choices": list(self.config.get("ragged_frontier_gap_length_choices") or []),
            "ragged_frontier_lowfreq_sigma": float(self.config.get("ragged_frontier_lowfreq_sigma", 12.0)),
            "cond_percent_min": float(self._base_dataset._cond_percent_min),
            "cond_percent_max": float(self._base_dataset._cond_percent_max),
            "crop_size": list(self.crop_size),
            "surface_downsample_factor": self.config.get("surface_downsample_factor"),
            "use_stored_resolution_only": bool(self.config.get("use_stored_resolution_only", True)),
            "n_samples": len(self.sample_index),
            "samples_md5": samples_md5,
        }
        return hashlib.md5(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()

    def _build_valid_split_plans(self) -> dict[tuple[int, int], tuple[dict, ...]]:
        prefilter_started = time.perf_counter()
        cache_dir = self._prefilter_cache_path()
        cache_key = self._prefilter_cache_key()
        cache_file = cache_dir / f"valid_plans_{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict):
                    self._prefilter_wall_seconds = float(time.perf_counter() - prefilter_started)
                    print(
                        f"[autoreg_mesh prefilter] cache hit key={cache_key} "
                        f"samples={len(cached)} path={cache_file}"
                    )
                    return cached
            except (pickle.UnpicklingError, EOFError, OSError) as e:
                print(f"[autoreg_mesh prefilter] cache unreadable, recomputing: {e}")
        print(f"[autoreg_mesh prefilter] cache miss key={cache_key}; recomputing...")

        valid_plans: dict[tuple[int, int], tuple[dict, ...]] = {}
        show_progress = bool(self.config.get("prefilter_show_progress", True)) and len(self.sample_index) > 0
        n_workers_cfg = int(self.config.get("prefilter_num_workers", 0))
        if n_workers_cfg <= 0:
            n_workers_cfg = max(1, (os.cpu_count() or 4) // 2)
        n_workers = max(1, min(n_workers_cfg, len(self.sample_index)))

        running_plan_count = 0
        if n_workers <= 1:
            iterator: Any = self.sample_index
            if show_progress:
                iterator = tqdm(self.sample_index, desc="autoreg_mesh prefilter", unit="wrap", leave=False)
            for sample_key in iterator:
                plan_list = self._compute_one_plan_list(sample_key)
                if plan_list:
                    valid_plans[sample_key] = tuple(plan_list)
                    running_plan_count += len(plan_list)
                if show_progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(kept=len(valid_plans), plans=running_plan_count)
            if show_progress and hasattr(iterator, "close"):
                iterator.close()
        else:
            global _active_prefilter_dataset
            _active_prefilter_dataset = self
            try:
                # Workers inherit the parent's BLAS thread-pool sizes via fork.
                # Cap them to 1 so workers don't oversubscribe with N x BLAS threads.
                try:
                    from threadpoolctl import threadpool_limits as _threadpool_limits
                    _tpl_ctx = _threadpool_limits(limits=1)
                except ImportError:
                    _tpl_ctx = None
                chunksize = max(1, len(self.sample_index) // (n_workers * 32))
                ctx = multiprocessing.get_context("fork")
                with ctx.Pool(n_workers) as pool:
                    iterator = pool.imap_unordered(
                        _prefilter_one_worker,
                        self.sample_index,
                        chunksize=chunksize,
                    )
                    if show_progress:
                        iterator = tqdm(
                            iterator,
                            total=len(self.sample_index),
                            desc=f"autoreg_mesh prefilter (n_workers={n_workers})",
                            unit="wrap",
                            leave=False,
                        )
                    for sample_key, plan_list in iterator:
                        if plan_list:
                            valid_plans[sample_key] = tuple(plan_list)
                            running_plan_count += len(plan_list)
                        if show_progress and hasattr(iterator, "set_postfix"):
                            iterator.set_postfix(kept=len(valid_plans), plans=running_plan_count)
            finally:
                _active_prefilter_dataset = None
                if _tpl_ctx is not None:
                    _tpl_ctx.unregister()

        self._prefilter_wall_seconds = float(time.perf_counter() - prefilter_started)

        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_file = cache_file.with_suffix(".pkl.tmp")
            with open(tmp_file, "wb") as f:
                pickle.dump(valid_plans, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_file, cache_file)
            print(
                f"[autoreg_mesh prefilter] cache saved key={cache_key} "
                f"samples={len(valid_plans)} path={cache_file}"
            )
        except Exception as e:
            print(f"[autoreg_mesh prefilter] cache save FAILED: {e}")

        return valid_plans

    def _resolve_surface_sampling_plan(self, wrap: dict) -> tuple[bool, int | tuple[int, int]]:
        factor_row, factor_col = _normalize_surface_downsample_factor(self.config["surface_downsample_factor"])
        seg = wrap.get("segment")
        scale = getattr(seg, "_scale", None)
        if scale is None:
            return bool(self.config.get("use_stored_resolution_only", False)), _maybe_collapse_downsample_factor(
                factor_row,
                factor_col,
            )

        row_scale = float(scale[0])
        col_scale = float(scale[1])
        use_stored_only = bool(self.config.get("use_stored_resolution_only", False))
        can_approximate_requested_full_factor_from_stored = (
            (factor_row * row_scale) >= 1.0 and
            (factor_col * col_scale) >= 1.0
        )
        use_stored_surface = use_stored_only or can_approximate_requested_full_factor_from_stored
        if not use_stored_surface:
            return False, _maybe_collapse_downsample_factor(factor_row, factor_col)

        stored_factor_row = max(1, int(round(factor_row * row_scale)))
        stored_factor_col = max(1, int(round(factor_col * col_scale)))
        return True, _maybe_collapse_downsample_factor(stored_factor_row, stored_factor_col)

    def _build_example(self, idx: int) -> dict | None:
        patch_idx, wrap_idx = self.sample_index[int(idx)]
        sample_key = _sample_key(patch_idx, wrap_idx)
        split_plans = self._valid_split_plans.get(sample_key, ())
        if not split_plans:
            return None
        patch = self.patches[patch_idx]
        wrap = patch.wraps[wrap_idx]
        plans_by_bucket = {"easy": [], "medium": [], "hard": []}
        for plan in split_plans:
            bucket = str(plan.get("difficulty_bucket", "medium"))
            if bucket in plans_by_bucket:
                plans_by_bucket[bucket].append(plan)
        available_buckets = [bucket for bucket, plans in plans_by_bucket.items() if plans]
        if not available_buckets:
            return None
        weight_cfg = _normalize_prefilter_sampling_weights(dict(self.config.get("prefilter_difficulty_sampling_weights", {})))
        weighted_buckets = [bucket for bucket in available_buckets if float(weight_cfg.get(bucket, 0.0)) > 0.0]
        if weighted_buckets:
            bucket_weights = np.asarray([float(weight_cfg[bucket]) for bucket in weighted_buckets], dtype=np.float64)
            bucket_weights = bucket_weights / np.maximum(bucket_weights.sum(), 1e-8)
            chosen_bucket = random.choices(weighted_buckets, weights=bucket_weights.tolist(), k=1)[0]
        else:
            chosen_bucket = random.choice(available_buckets)
        split_plan = random.choice(plans_by_bucket[chosen_bucket])
        use_stored_surface = bool(split_plan["use_stored_surface"])
        effective_surface_downsample_factor = split_plan["effective_surface_downsample_factor"]
        surface_zyx = self._extract_surface_for_plan(patch, wrap, use_stored_surface=use_stored_surface)
        if surface_zyx is None:
            return None
        conditioning = create_split_conditioning_from_surface_grid(
            self._base_dataset,
            idx=0,
            patch_idx=int(patch_idx),
            wrap_idx=int(wrap_idx),
            patch=patch,
            surface_zyx=surface_zyx,
            cond_direction=str(split_plan["direction"]),
            conditioning_count=int(split_plan["conditioning_count"]),
        )
        if conditioning is None:
            return None

        min_corner = np.asarray(conditioning["min_corner"], dtype=np.int64)
        max_corner = np.asarray(conditioning["max_corner"], dtype=np.int64)
        vol_crop = _read_volume_crop_from_patch(patch, self.crop_size, min_corner, max_corner)
        cond_local = np.asarray(conditioning["cond_zyxs_unperturbed"], dtype=np.float32) - min_corner[None, None, :]
        masked_local = np.asarray(conditioning["masked_zyxs"], dtype=np.float32) - min_corner[None, None, :]
        vol_crop, cond_local, masked_local, spatial_aug_metadata = _apply_spatial_augmentation(
            vol_crop,
            cond_local=cond_local,
            masked_local=masked_local,
            crop_size=self.crop_size,
            augmentation_cfg=self.config.get("spatial_augmentation", {}),
            enabled=self.apply_spatial_augmentation,
        )
        # UV-grid augmentation (rows<->cols) operates ONLY on the 2D UV
        # parametrization layout; volume and world-coord channels are
        # untouched. Direction token must be remapped consistently
        # (left<->up, right<->down) since the frontier band semantics in
        # serialization.extract_frontier_prompt_band index UV axes.
        cond_local, masked_local, cond_direction, uv_aug_metadata = _apply_uv_grid_augmentation(
            cond_local,
            masked_local,
            direction=str(conditioning["cond_direction"]).lower(),
            augmentation_cfg=self.config.get("spatial_augmentation", {}),
            enabled=self.apply_spatial_augmentation,
        )
        spatial_aug_metadata = {**spatial_aug_metadata, **uv_aug_metadata}
        valid_prompt_widths_clean = list(split_plan.get("valid_prompt_widths_clean", []))
        valid_prompt_widths_ragged = list(split_plan.get("valid_prompt_widths_ragged", []))
        width_candidates = valid_prompt_widths_clean if valid_prompt_widths_clean else valid_prompt_widths_ragged
        frontier_band_width = int(random.choice(width_candidates)) if width_candidates else _sample_frontier_band_width(self.config)
        cond_local, ragged_frontier_applied = _apply_ragged_frontier_augmentation(
            cond_local,
            direction=cond_direction,
            frontier_band_width=int(frontier_band_width),
            ragged_frontier_prob=float(self.config.get("ragged_frontier_prob", 0.0)),
            ragged_frontier_max_inset=int(self.config.get("ragged_frontier_max_inset", 0)),
            ragged_frontier_gap_length_choices=self.config.get("ragged_frontier_gap_length_choices"),
            ragged_frontier_lowfreq_sigma=float(self.config.get("ragged_frontier_lowfreq_sigma", 12.0)),
        )
        vol_crop = _apply_volume_only_augmentation(
            vol_crop,
            self.config.get("volume_only_augmentation", {}),
            enabled=self.apply_volume_only_augmentation,
        )
        serialized = serialize_split_conditioning_example(
            cond_zyxs_local=cond_local,
            masked_zyxs_local=masked_local,
            direction=cond_direction,
            volume_shape=self.crop_size,
            patch_size=self.config["patch_size"],
            offset_num_bins=self.config["offset_num_bins"],
            frontier_band_width=int(frontier_band_width),
            surface_downsample_factor=effective_surface_downsample_factor,
            use_stored_resolution_only=use_stored_surface,
        )

        if not bool(np.any(serialized["prompt_tokens"]["valid_mask"])):
            return None
        if not bool(np.any(serialized["target_valid_mask"])):
            return None
        boundary_stats = _compute_target_boundary_stats(
            serialized["target_grid_local"],
            volume_shape=self.crop_size,
            direction=str(serialized["direction"]),
            frontier_band_width=int(frontier_band_width),
        )
        reject, _reject_reason = _should_reject_boundary_stats(
            boundary_stats,
            max_frontier_invalid_fraction=float(self.config.get("prefilter_max_frontier_invalid_fraction", 0.5)),
            max_target_invalid_fraction=float(self.config.get("prefilter_max_target_invalid_fraction", 0.75)),
        )
        if (not bool(ragged_frontier_applied)) and bool(reject):
            return None

        sample_key = (int(patch_idx), int(wrap_idx))
        vol_tokens = None
        if bool(self.config.get("cache_vol_tokens", False)):
            vol_tokens = _lookup_cached_vol_tokens(self.config.get("vol_token_cache"), sample_key)
            if vol_tokens is not None:
                vol_tokens = np.asarray(vol_tokens, dtype=np.float32)

        wrap = conditioning["wrap"]
        wrap_metadata = {
            "patch_idx": int(patch_idx),
            "wrap_idx": int(wrap_idx),
            "wrap_id": int(wrap.get("wrap_id", -1)),
            "segment_idx": int(wrap.get("segment_idx", -1)),
            "bbox_2d": tuple(int(v) for v in wrap["bbox_2d"]),
            "segment_uuid": getattr(wrap.get("segment"), "uuid", ""),
            "surface_sampling_mode": "stored" if use_stored_surface else "full",
        }

        return {
            "volume": torch.from_numpy(vol_crop[None, ...]).to(torch.float32),
            "vol_tokens": None if vol_tokens is None else torch.from_numpy(vol_tokens).to(torch.float32),
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
                "sample_key": sample_key,
                "surface_sampling_mode": "stored" if use_stored_surface else "full",
                "frontier_band_width": int(frontier_band_width),
                "difficulty_bucket": str(split_plan.get("difficulty_bucket", "medium")),
                "valid_prompt_widths_clean": list(valid_prompt_widths_clean),
                "valid_prompt_widths_ragged": list(valid_prompt_widths_ragged),
                "ragged_frontier_applied": bool(ragged_frontier_applied),
                "spatial_augmented": bool(spatial_aug_metadata["spatial_augmented"]),
                "spatial_mirror_axes": list(spatial_aug_metadata["spatial_mirror_axes"]),
                "spatial_axis_order": list(spatial_aug_metadata["spatial_axis_order"]),
                "uv_transposed": bool(spatial_aug_metadata.get("uv_transposed", False)),
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
            "target_invalid_fraction": torch.tensor(float(boundary_stats["target_invalid_fraction"]), dtype=torch.float32),
            "frontier_invalid_fraction": torch.tensor(float(boundary_stats["frontier_invalid_fraction"]), dtype=torch.float32),
            "touches_crop_boundary": torch.tensor(bool(boundary_stats["touches_crop_boundary"]), dtype=torch.bool),
            "direction": str(serialized["direction"]),
            "direction_id": torch.tensor(int(serialized["direction_id"]), dtype=torch.long),
            "strip_length": torch.tensor(int(serialized["strip_length"]), dtype=torch.long),
            "num_strips": torch.tensor(int(serialized["num_strips"]), dtype=torch.long),
            "min_corner": torch.from_numpy(min_corner.astype(np.float32)).to(torch.float32),
            "world_bbox": torch.tensor(tuple(float(v) for v in patch.world_bbox), dtype=torch.float32),
            "target_grid_shape": torch.tensor(tuple(int(v) for v in serialized["target_grid_shape"]), dtype=torch.long),
            "wrap_metadata": wrap_metadata,
        }

    def __getitem__(self, idx: int) -> dict:
        attempted = {int(idx)}
        current_idx = int(idx)
        for _ in range(self.max_resample_attempts):
            example = self._build_example(current_idx)
            if example is not None:
                return example
            replacement = choose_replacement_index(self.sample_index, attempted_indices=attempted)
            if replacement is None:
                break
            current_idx = int(replacement)
            attempted.add(current_idx)
        raise RuntimeError(f"Failed to build an autoreg_mesh example after {self.max_resample_attempts} attempts")


def _pad_1d_long(items: list[torch.Tensor], *, pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(item.shape[0]) for item in items)
    padded = torch.full((len(items), max_len), int(pad_value), dtype=torch.long)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for batch_idx, item in enumerate(items):
        length = int(item.shape[0])
        if length == 0:
            continue
        padded[batch_idx, :length] = item
        mask[batch_idx, :length] = True
    return padded, mask


def _pad_1d_bool(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(item.shape[0]) for item in items)
    padded = torch.zeros((len(items), max_len), dtype=torch.bool)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for batch_idx, item in enumerate(items):
        length = int(item.shape[0])
        if length == 0:
            continue
        padded[batch_idx, :length] = item.to(torch.bool)
        mask[batch_idx, :length] = True
    return padded, mask


def _pad_2d_long(items: list[torch.Tensor], *, pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(item.shape[0]) for item in items)
    feat_dim = int(items[0].shape[1])
    padded = torch.full((len(items), max_len, feat_dim), int(pad_value), dtype=torch.long)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for batch_idx, item in enumerate(items):
        length = int(item.shape[0])
        if length == 0:
            continue
        padded[batch_idx, :length] = item
        mask[batch_idx, :length] = True
    return padded, mask


def _pad_2d_float(items: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(int(item.shape[0]) for item in items)
    feat_dim = int(items[0].shape[1])
    padded = torch.zeros((len(items), max_len, feat_dim), dtype=torch.float32)
    mask = torch.zeros((len(items), max_len), dtype=torch.bool)
    for batch_idx, item in enumerate(items):
        length = int(item.shape[0])
        if length == 0:
            continue
        padded[batch_idx, :length] = item
        mask[batch_idx, :length] = True
    return padded, mask


def autoreg_mesh_collate(batch: list[dict]) -> dict:
    volumes = torch.stack([item["volume"] for item in batch], dim=0)
    result = {
        "volume": volumes,
        "direction": [item["direction"] for item in batch],
        "direction_id": torch.stack([item["direction_id"] for item in batch], dim=0),
        "strip_length": torch.stack([item["strip_length"] for item in batch], dim=0),
        "num_strips": torch.stack([item["num_strips"] for item in batch], dim=0),
        "min_corner": torch.stack([item["min_corner"] for item in batch], dim=0),
        "world_bbox": torch.stack([item["world_bbox"] for item in batch], dim=0),
        "target_grid_shape": torch.stack([item["target_grid_shape"] for item in batch], dim=0),
        "prompt_anchor_xyz": torch.stack([item["prompt_anchor_xyz"] for item in batch], dim=0),
        "prompt_anchor_valid": torch.stack([item["prompt_anchor_valid"] for item in batch], dim=0),
        "prompt_meta": [item["prompt_meta"] for item in batch],
        "wrap_metadata": [item["wrap_metadata"] for item in batch],
        "conditioning_grid_local": [item["conditioning_grid_local"] for item in batch],
        "prompt_grid_local": [item["prompt_grid_local"] for item in batch],
        "target_grid_local": [item["target_grid_local"] for item in batch],
    }

    if all(item.get("vol_tokens") is not None for item in batch):
        result["vol_tokens"] = torch.stack([item["vol_tokens"] for item in batch], dim=0)

    prompt_coarse_ids, prompt_padding_mask = _pad_1d_long(
        [item["prompt_tokens"]["coarse_ids"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    prompt_offset_bins, _ = _pad_2d_long(
        [item["prompt_tokens"]["offset_bins"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    prompt_xyz, _ = _pad_2d_float([item["prompt_tokens"]["xyz"] for item in batch])
    prompt_strip_positions, _ = _pad_2d_long(
        [item["prompt_tokens"]["strip_positions"] for item in batch],
        pad_value=0,
    )
    prompt_strip_coords, _ = _pad_2d_float([item["prompt_tokens"]["strip_coords"] for item in batch])
    prompt_valid_mask, _ = _pad_1d_bool([item["prompt_tokens"]["valid_mask"] for item in batch])

    result["prompt_tokens"] = {
        "coarse_ids": prompt_coarse_ids,
        "offset_bins": prompt_offset_bins,
        "xyz": prompt_xyz,
        "strip_positions": prompt_strip_positions,
        "strip_coords": prompt_strip_coords,
        "mask": prompt_padding_mask,
        "valid_mask": prompt_valid_mask & prompt_padding_mask,
    }

    target_coarse_ids, target_padding_mask = _pad_1d_long(
        [item["target_coarse_ids"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_offset_bins, _ = _pad_2d_long(
        [item["target_offset_bins"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_xyz, _ = _pad_2d_float([item["target_xyz"] for item in batch])
    target_bin_center_xyz, _ = _pad_2d_float([item["target_bin_center_xyz"] for item in batch])
    target_stop, _ = _pad_2d_float([item["target_stop"].unsqueeze(-1) for item in batch])
    target_strip_positions, _ = _pad_2d_long(
        [item["target_strip_positions"] for item in batch],
        pad_value=0,
    )
    target_strip_coords, _ = _pad_2d_float([item["target_strip_coords"] for item in batch])
    target_valid_mask, _ = _pad_1d_bool([item["target_valid_mask"] for item in batch])

    result["target_coarse_ids"] = target_coarse_ids
    result["target_offset_bins"] = target_offset_bins
    result["target_xyz"] = target_xyz
    result["target_bin_center_xyz"] = target_bin_center_xyz
    result["target_stop"] = target_stop.squeeze(-1)
    result["target_strip_positions"] = target_strip_positions
    result["target_strip_coords"] = target_strip_coords
    result["target_mask"] = target_padding_mask
    result["target_valid_mask"] = target_valid_mask & target_padding_mask
    result["target_invalid_mask"] = (~result["target_valid_mask"]) & target_padding_mask
    result["target_supervision_mask"] = result["target_valid_mask"]
    result["target_lengths"] = target_padding_mask.sum(dim=1)
    result["target_invalid_fraction"] = torch.stack(
        [item.get("target_invalid_fraction", torch.tensor(0.0, dtype=torch.float32)) for item in batch],
        dim=0,
    )
    result["frontier_invalid_fraction"] = torch.stack(
        [item.get("frontier_invalid_fraction", torch.tensor(0.0, dtype=torch.float32)) for item in batch],
        dim=0,
    )
    result["touches_crop_boundary"] = torch.stack(
        [item.get("touches_crop_boundary", torch.tensor(False, dtype=torch.bool)) for item in batch],
        dim=0,
    )
    result["direction_id"] = torch.stack([torch.tensor(DIRECTION_TO_ID[item["direction"]], dtype=torch.long) for item in batch])
    return result
