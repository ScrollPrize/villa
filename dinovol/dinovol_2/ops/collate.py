from __future__ import annotations

from functools import partial
import math
import random
from typing import Any, Mapping

import torch

from .masking import MaskingGenerator3d


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
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    tokens_per_sample = collated_masks.shape[1]
    inverse_mask_counts = 1.0 / collated_masks.sum(-1).clamp(min=1.0)
    masked_sample_indices = torch.div(mask_indices_list, tokens_per_sample, rounding_mode="floor")
    masks_weight = inverse_mask_counts.index_select(0, masked_sample_indices)

    return {
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
    }


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
        dtype=config.get("dtype", torch.float32),
    )
