# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in LICENSES/Apache-2.0.txt.
#
# Modified for Dinovol's three-dimensional training pipeline.

from __future__ import annotations

from functools import partial
import math
import random
from typing import Any, Mapping

import torch

from .masking import MaskingGenerator3d
from .point_embeddings import interpolation_support_indices


def _as_3tuple(value: int | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    return tuple(int(v) for v in value)


def collate_dino_ibot_batch(
    samples: list[Mapping[str, Any]],
    *,
    mask_ratio_min_max: tuple[float, float],
    mask_sample_probability: float,
    n_tokens: int,
    mask_generator: MaskingGenerator3d,
    patch_size: tuple[int, int, int],
    feature_map_size: tuple[int, int, int],
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    n_global_views = len(samples[0]["global_views"])
    n_local_views = len(samples[0]["local_views"])

    global_crops = torch.stack(
        [sample["global_views"][i] for i in range(n_global_views) for sample in samples]
    ).to(dtype)

    if n_local_views:
        local_crops = torch.stack(
            [sample["local_views"][i] for i in range(n_local_views) for sample in samples]
        ).to(dtype)
    else:
        local_shape = samples[0]["global_views"][0].shape
        local_crops = torch.empty((0, *local_shape), dtype=dtype)

    if samples[0].get("gram_teacher_views"):
        collated_gram_teacher_crops = torch.stack(
            [sample["gram_teacher_views"][i] for i in range(n_global_views) for sample in samples]
        ).to(dtype)
    else:
        collated_gram_teacher_crops = None

    n_masked_samples = int(global_crops.shape[0] * mask_sample_probability)
    masks_list: list[torch.Tensor] = []
    upperbound = 0

    if n_masked_samples:
        probs = torch.linspace(mask_ratio_min_max[0], mask_ratio_min_max[1], n_masked_samples + 1)
        for i in range(n_masked_samples):
            ratio = random.uniform(float(probs[i]), float(probs[i + 1]))
            n_masked = min(int(math.floor(n_tokens * ratio)), n_tokens)
            masks_list.append(torch.from_numpy(mask_generator(n_masked)).bool())
            upperbound += int(math.ceil(n_tokens * float(probs[i + 1])))

    for _ in range(global_crops.shape[0] - n_masked_samples):
        masks_list.append(torch.from_numpy(mask_generator(0)).bool())

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    point_coordinates_parts: list[torch.Tensor] = []
    point_type_parts: list[torch.Tensor] = []
    point_row_parts: list[torch.Tensor] = []
    for view_index in range(n_global_views):
        for sample_index, sample in enumerate(samples):
            coordinates_by_view = sample.get("global_point_coordinates")
            types_by_view = sample.get("global_point_type_ids")
            if coordinates_by_view is None:
                coordinates = torch.empty((0, 3), dtype=torch.float32)
                type_ids = torch.empty((0,), dtype=torch.long)
            else:
                coordinates = torch.as_tensor(coordinates_by_view[view_index], dtype=torch.float32)
                type_ids = torch.as_tensor(types_by_view[view_index], dtype=torch.long)
            if coordinates.ndim != 2 or coordinates.shape[1:] != (3,) or type_ids.shape != (coordinates.shape[0],):
                raise ValueError("Each global view must provide matching Nx3 point coordinates and N type IDs.")
            if coordinates.shape[0]:
                row_index = view_index * len(samples) + sample_index
                point_coordinates_parts.append(coordinates)
                point_type_parts.append(type_ids)
                point_row_parts.append(torch.full((coordinates.shape[0],), row_index, dtype=torch.long))

    if point_coordinates_parts:
        collated_point_coordinates = torch.cat(point_coordinates_parts)
        collated_point_type_ids = torch.cat(point_type_parts)
        collated_point_rows = torch.cat(point_row_parts)
        support = interpolation_support_indices(
            collated_point_coordinates,
            patch_size,
            feature_map_size,
        )
        collated_masks[collated_point_rows[:, None], support] = False
    else:
        collated_point_coordinates = torch.empty((0, 3), dtype=torch.float32)
        collated_point_type_ids = torch.empty((0,), dtype=torch.long)
        collated_point_rows = torch.empty((0,), dtype=torch.long)

    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    tokens_per_sample = collated_masks.shape[1]
    inverse_mask_counts = 1.0 / collated_masks.sum(-1).clamp(min=1.0)
    masked_sample_indices = torch.div(mask_indices_list, tokens_per_sample, rounding_mode="floor")
    masks_weight = inverse_mask_counts.index_select(0, masked_sample_indices)

    batch = {
        "collated_global_crops": global_crops,
        "collated_local_crops": local_crops,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.tensor([mask_indices_list.numel()], dtype=torch.long),
        "n_global_views": n_global_views,
        "n_local_views": n_local_views,
        "batch_size": len(samples),
        "collated_point_coordinates": collated_point_coordinates,
        "collated_point_type_ids": collated_point_type_ids,
        "collated_point_rows": collated_point_rows,
    }
    if collated_gram_teacher_crops is not None:
        batch["collated_gram_teacher_crops"] = collated_gram_teacher_crops
    return batch


def build_dino_ibot_collate_fn(config: Mapping[str, Any]) -> partial:
    global_crop_size = _as_3tuple(config["global_crop_size"])
    patch_size = _as_3tuple(config["patch_size"])
    feature_map_size = tuple(size // patch for size, patch in zip(global_crop_size, patch_size))
    n_tokens = math.prod(feature_map_size)
    mask_generator = MaskingGenerator3d(feature_map_size)
    return partial(
        collate_dino_ibot_batch,
        mask_ratio_min_max=tuple(config.get("mask_ratio_min_max", (0.1, 0.5))),
        mask_sample_probability=float(config.get("mask_sample_probability", 0.5)),
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        patch_size=patch_size,
        feature_map_size=feature_map_size,
        dtype=config.get("dtype", torch.float32),
    )
