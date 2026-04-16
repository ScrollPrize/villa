from __future__ import annotations

from copy import deepcopy
from typing import Any
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    DIRECTION_TO_ID,
    IGNORE_INDEX,
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
    }


def _should_reject_boundary_stats(
    boundary_stats: dict[str, float | bool],
    *,
    max_target_invalid_fraction: float = 0.25,
) -> bool:
    return (
        float(boundary_stats["frontier_invalid_fraction"]) > 0.0 or
        float(boundary_stats["target_invalid_fraction"]) > float(max_target_invalid_fraction)
    )


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


class AutoregMeshDataset(Dataset):
    """Build autoregressive wrap-completion examples from split conditioning."""

    def __init__(
        self,
        config: dict,
        *,
        patch_metadata=None,
        apply_augmentation: bool = False,
        apply_perturbation: bool = False,
        apply_volume_only_augmentation: bool | None = None,
        max_resample_attempts: int = 8,
    ) -> None:
        self.config = validate_autoreg_mesh_config(config)
        self.max_resample_attempts = int(max_resample_attempts)
        self.apply_volume_only_augmentation = bool(
            self.config.get("volume_only_augmentation", {}).get("enabled", False)
            if apply_volume_only_augmentation is None
            else apply_volume_only_augmentation
        )

        base_config = deepcopy(config)
        base_config["sample_mode"] = "wrap"
        setdefault_rowcol_cond_dataset_config(base_config)
        validate_rowcol_cond_dataset_config(base_config)

        self._base_dataset = EdtSegDataset(
            base_config,
            apply_augmentation=bool(apply_augmentation),
            apply_perturbation=bool(apply_perturbation),
            patch_metadata=patch_metadata,
        )
        self.patches = self._base_dataset.patches
        self.sample_index = list(self._base_dataset.sample_index)
        self.crop_size = tuple(int(v) for v in self._base_dataset.crop_size)
        precomputed_plans = None
        if isinstance(patch_metadata, dict):
            precomputed_plans = patch_metadata.get("autoreg_mesh_valid_split_plans")
        if precomputed_plans is None:
            self._valid_split_plans = self._build_valid_split_plans()
        else:
            self._valid_split_plans = {
                _sample_key(*sample_key): tuple(dict(plan) for plan in plans)
                for sample_key, plans in dict(precomputed_plans).items()
            }
        self.sample_index = [
            _sample_key(*sample_key)
            for sample_key in self.sample_index
            if len(self._valid_split_plans.get(_sample_key(*sample_key), ())) > 0
        ]
        self._prefilter_stats = {
            "num_samples_with_valid_plans": len(self.sample_index),
            "num_total_valid_split_plans": int(sum(len(plans) for plans in self._valid_split_plans.values())),
        }
        if not self.sample_index:
            raise RuntimeError("autoreg_mesh dataset prefilter removed every sample; no valid wrap splits remain")

    def __len__(self) -> int:
        return len(self.sample_index)

    def export_patch_metadata(self):
        metadata = self._base_dataset.export_patch_metadata()
        metadata["autoreg_mesh_valid_split_plans"] = {
            sample_key: tuple(dict(plan) for plan in plans)
            for sample_key, plans in self._valid_split_plans.items()
        }
        metadata["autoreg_mesh_prefilter_stats"] = dict(self._prefilter_stats)
        return metadata

    def _extract_surface_for_plan(self, patch, wrap: dict, *, use_stored_surface: bool) -> np.ndarray | None:
        if bool(use_stored_surface):
            return extract_wrap_world_surface_stored(
                patch,
                wrap,
                require_all_valid=True,
            )
        return self._base_dataset._extract_wrap_world_surface(patch, wrap, require_all_valid=True)

    def _serialize_candidate_plan(
        self,
        *,
        patch_idx: int,
        wrap_idx: int,
        patch,
        surface_zyx: np.ndarray,
        direction: str,
        conditioning_count: int,
        effective_surface_downsample_factor,
        use_stored_surface: bool,
    ) -> tuple[dict | None, dict | None]:
        conditioning = create_split_conditioning_from_surface_grid(
            self._base_dataset,
            idx=0,
            patch_idx=patch_idx,
            wrap_idx=wrap_idx,
            patch=patch,
            surface_zyx=surface_zyx,
            cond_direction=str(direction),
            conditioning_count=int(conditioning_count),
        )
        if conditioning is None:
            return None, None
        min_corner = np.asarray(conditioning["min_corner"], dtype=np.int64)
        cond_local = np.asarray(conditioning["cond_zyxs_unperturbed"], dtype=np.float32) - min_corner[None, None, :]
        masked_local = np.asarray(conditioning["masked_zyxs"], dtype=np.float32) - min_corner[None, None, :]
        serialized = serialize_split_conditioning_example(
            cond_zyxs_local=cond_local,
            masked_zyxs_local=masked_local,
            direction=str(conditioning["cond_direction"]).lower(),
            volume_shape=self.crop_size,
            patch_size=self.config["patch_size"],
            offset_num_bins=self.config["offset_num_bins"],
            frontier_band_width=int(self.config["frontier_band_width"]),
            surface_downsample_factor=effective_surface_downsample_factor,
            use_stored_resolution_only=use_stored_surface,
        )
        return conditioning, serialized

    def _build_valid_split_plans(self) -> dict[tuple[int, int], tuple[dict, ...]]:
        valid_plans: dict[tuple[int, int], tuple[dict, ...]] = {}
        for patch_idx, wrap_idx in self.sample_index:
            sample_key = _sample_key(patch_idx, wrap_idx)
            patch = self.patches[int(patch_idx)]
            wrap = patch.wraps[int(wrap_idx)]
            use_stored_surface, effective_surface_downsample_factor = self._resolve_surface_sampling_plan(wrap)
            surface_zyx = self._extract_surface_for_plan(patch, wrap, use_stored_surface=use_stored_surface)
            if surface_zyx is None:
                continue
            h_grid, w_grid = surface_zyx.shape[:2]
            plan_list: list[dict] = []
            directions = []
            if w_grid >= 2:
                directions.extend(["left", "right"])
            if h_grid >= 2:
                directions.extend(["up", "down"])
            for direction in directions:
                axis_size = w_grid if direction in {"left", "right"} else h_grid
                for conditioning_count in _reachable_conditioning_counts(
                    axis_size,
                    self._base_dataset._cond_percent_min,
                    self._base_dataset._cond_percent_max,
                ):
                    conditioning, serialized = self._serialize_candidate_plan(
                        patch_idx=int(patch_idx),
                        wrap_idx=int(wrap_idx),
                        patch=patch,
                        surface_zyx=surface_zyx,
                        direction=direction,
                        conditioning_count=int(conditioning_count),
                        effective_surface_downsample_factor=effective_surface_downsample_factor,
                        use_stored_surface=use_stored_surface,
                    )
                    if conditioning is None or serialized is None:
                        continue
                    if not bool(np.any(serialized["prompt_tokens"]["valid_mask"])):
                        continue
                    if not bool(np.any(serialized["target_valid_mask"])):
                        continue
                    boundary_stats = _compute_target_boundary_stats(
                        serialized["target_grid_local"],
                        volume_shape=self.crop_size,
                        direction=str(serialized["direction"]),
                    )
                    if _should_reject_boundary_stats(boundary_stats):
                        continue
                    plan_list.append(
                        {
                            "direction": str(direction),
                            "conditioning_count": int(conditioning_count),
                            "conditioning_percent": float(conditioning["conditioning_percent"]),
                            "use_stored_surface": bool(use_stored_surface),
                            "effective_surface_downsample_factor": effective_surface_downsample_factor,
                        }
                    )
            if plan_list:
                valid_plans[sample_key] = tuple(plan_list)
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
        split_plan = random.choice(split_plans)
        use_stored_surface = bool(split_plan["use_stored_surface"])
        effective_surface_downsample_factor = split_plan["effective_surface_downsample_factor"]
        surface_zyx = self._extract_surface_for_plan(patch, wrap, use_stored_surface=use_stored_surface)
        if surface_zyx is None:
            return None
        conditioning, serialized = self._serialize_candidate_plan(
            patch_idx=int(patch_idx),
            wrap_idx=int(wrap_idx),
            patch=patch,
            surface_zyx=surface_zyx,
            direction=str(split_plan["direction"]),
            conditioning_count=int(split_plan["conditioning_count"]),
            effective_surface_downsample_factor=effective_surface_downsample_factor,
            use_stored_surface=use_stored_surface,
        )
        if conditioning is None or serialized is None:
            return None

        min_corner = np.asarray(conditioning["min_corner"], dtype=np.int64)
        max_corner = np.asarray(conditioning["max_corner"], dtype=np.int64)
        vol_crop = _read_volume_crop_from_patch(patch, self.crop_size, min_corner, max_corner)
        vol_crop = _apply_volume_only_augmentation(
            vol_crop,
            self.config.get("volume_only_augmentation", {}),
            enabled=self.apply_volume_only_augmentation,
        )

        if not bool(np.any(serialized["prompt_tokens"]["valid_mask"])):
            return None
        if not bool(np.any(serialized["target_valid_mask"])):
            return None
        boundary_stats = _compute_target_boundary_stats(
            serialized["target_grid_local"],
            volume_shape=self.crop_size,
            direction=str(serialized["direction"]),
        )
        if _should_reject_boundary_stats(boundary_stats):
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
