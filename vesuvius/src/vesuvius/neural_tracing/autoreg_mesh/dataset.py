from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from vesuvius.neural_tracing.autoreg_mesh.config import validate_autoreg_mesh_config
from vesuvius.neural_tracing.autoreg_mesh.serialization import (
    DIRECTION_TO_ID,
    IGNORE_INDEX,
    serialize_split_conditioning_example,
)
from vesuvius.neural_tracing.datasets.common import _read_volume_crop_from_patch
from vesuvius.neural_tracing.datasets.conditioning import create_split_conditioning
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


class AutoregMeshDataset(Dataset):
    """Build autoregressive wrap-completion examples from split conditioning."""

    def __init__(
        self,
        config: dict,
        *,
        patch_metadata=None,
        apply_augmentation: bool = False,
        apply_perturbation: bool = False,
        max_resample_attempts: int = 8,
    ) -> None:
        self.config = validate_autoreg_mesh_config(config)
        self.max_resample_attempts = int(max_resample_attempts)

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

    def __len__(self) -> int:
        return len(self.sample_index)

    def export_patch_metadata(self):
        return self._base_dataset.export_patch_metadata()

    def _resolve_surface_downsample_factor(self, conditioning: dict) -> int | tuple[int, int]:
        factor = self.config["surface_downsample_factor"]
        if not bool(self.config.get("use_stored_resolution_only", False)):
            return factor
        seg = conditioning.get("seg")
        scale = getattr(seg, "_scale", None)
        if scale is None:
            return factor
        row_scale = float(scale[0])
        col_scale = float(scale[1])
        row_factor = max(1, int(round(1.0 / max(row_scale, 1e-6))))
        col_factor = max(1, int(round(1.0 / max(col_scale, 1e-6))))
        factor_row, factor_col = row_factor, col_factor
        if isinstance(factor, int):
            factor_row = max(factor_row, int(factor))
            factor_col = max(factor_col, int(factor))
        else:
            factor_row = max(factor_row, int(factor[0]))
            factor_col = max(factor_col, int(factor[1]))
        return (factor_row, factor_col)

    def _build_example(self, idx: int) -> dict | None:
        patch_idx, wrap_idx = self.sample_index[int(idx)]
        patch = self.patches[patch_idx]
        conditioning = create_split_conditioning(self._base_dataset, int(idx), patch_idx, wrap_idx, patch)
        if conditioning is None:
            return None

        min_corner = np.asarray(conditioning["min_corner"], dtype=np.int64)
        max_corner = np.asarray(conditioning["max_corner"], dtype=np.int64)
        vol_crop = _read_volume_crop_from_patch(patch, self.crop_size, min_corner, max_corner)

        cond_local = (np.asarray(conditioning["cond_zyxs_unperturbed"], dtype=np.float32) - min_corner[None, None, :])
        masked_local = (np.asarray(conditioning["masked_zyxs"], dtype=np.float32) - min_corner[None, None, :])

        serialized = serialize_split_conditioning_example(
            cond_zyxs_local=cond_local,
            masked_zyxs_local=masked_local,
            direction=str(conditioning["cond_direction"]).lower(),
            volume_shape=self.crop_size,
            patch_size=self.config["patch_size"],
            offset_num_bins=self.config["offset_num_bins"],
            frontier_band_width=int(self.config["frontier_band_width"]),
            surface_downsample_factor=self._resolve_surface_downsample_factor(conditioning),
            use_stored_resolution_only=bool(self.config.get("use_stored_resolution_only", False)),
        )

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
                "conditioning_shape": tuple(int(v) for v in cond_local.shape[:2]),
                "sample_key": sample_key,
            },
            "prompt_anchor_xyz": torch.from_numpy(serialized["prompt_anchor_xyz"]).to(torch.float32),
            "prompt_grid_local": torch.from_numpy(serialized["prompt_grid_local"]).to(torch.float32),
            "target_coarse_ids": torch.from_numpy(serialized["target_coarse_ids"]).to(torch.long),
            "target_offset_bins": torch.from_numpy(serialized["target_offset_bins"]).to(torch.long),
            "target_stop": torch.from_numpy(serialized["target_stop"]).to(torch.float32),
            "target_xyz": torch.from_numpy(serialized["target_xyz"]).to(torch.float32),
            "target_strip_positions": torch.from_numpy(serialized["target_strip_positions"]).to(torch.long),
            "target_strip_coords": torch.from_numpy(serialized["target_strip_coords"]).to(torch.float32),
            "target_grid_local": torch.from_numpy(serialized["target_grid_local"]).to(torch.float32),
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
        "prompt_meta": [item["prompt_meta"] for item in batch],
        "wrap_metadata": [item["wrap_metadata"] for item in batch],
        "prompt_grid_local": [item["prompt_grid_local"] for item in batch],
        "target_grid_local": [item["target_grid_local"] for item in batch],
    }

    if all(item.get("vol_tokens") is not None for item in batch):
        result["vol_tokens"] = torch.stack([item["vol_tokens"] for item in batch], dim=0)

    prompt_coarse_ids, prompt_mask = _pad_1d_long(
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

    result["prompt_tokens"] = {
        "coarse_ids": prompt_coarse_ids,
        "offset_bins": prompt_offset_bins,
        "xyz": prompt_xyz,
        "strip_positions": prompt_strip_positions,
        "strip_coords": prompt_strip_coords,
        "mask": prompt_mask,
    }

    target_coarse_ids, target_mask = _pad_1d_long(
        [item["target_coarse_ids"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_offset_bins, _ = _pad_2d_long(
        [item["target_offset_bins"] for item in batch],
        pad_value=IGNORE_INDEX,
    )
    target_xyz, _ = _pad_2d_float([item["target_xyz"] for item in batch])
    target_stop, _ = _pad_2d_float([item["target_stop"].unsqueeze(-1) for item in batch])
    target_strip_positions, _ = _pad_2d_long(
        [item["target_strip_positions"] for item in batch],
        pad_value=0,
    )
    target_strip_coords, _ = _pad_2d_float([item["target_strip_coords"] for item in batch])

    result["target_coarse_ids"] = target_coarse_ids
    result["target_offset_bins"] = target_offset_bins
    result["target_xyz"] = target_xyz
    result["target_stop"] = target_stop.squeeze(-1)
    result["target_strip_positions"] = target_strip_positions
    result["target_strip_coords"] = target_strip_coords
    result["target_mask"] = target_mask
    result["target_lengths"] = target_mask.sum(dim=1)
    result["direction_id"] = torch.stack([torch.tensor(DIRECTION_TO_ID[item["direction"]], dtype=torch.long) for item in batch])
    return result
