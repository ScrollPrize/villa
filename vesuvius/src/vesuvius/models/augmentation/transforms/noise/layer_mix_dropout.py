import random

import numpy as np
import torch

from vesuvius.models.augmentation.transforms.base.basic_transform import ImageOnlyTransform


def layer_mix_dropout(
    image: torch.Tensor,
    min_crop_ratio: float = 0.25,
    max_crop_ratio: float = 0.75,
    cutout_max_count: int = 0,
    cutout_probability: float = 0.0,
) -> torch.Tensor:
    """Relocate a contiguous channel block into an otherwise zeroed tensor."""
    in_channels = int(image.shape[0])

    min_crop = max(1, int(np.ceil(in_channels * float(min_crop_ratio))))
    max_crop = max(1, int(np.floor(in_channels * float(max_crop_ratio))))
    max_crop = min(max_crop, in_channels)
    min_crop = min(min_crop, in_channels)
    max_crop = max(min_crop, max_crop)

    crop_size = random.randint(min_crop, max_crop)
    max_start = max(0, in_channels - crop_size)
    crop_start = random.randint(0, max_start)
    paste_start = random.randint(0, max_start)

    image_out = torch.zeros_like(image)
    crop_slice = slice(crop_start, crop_start + crop_size)
    paste_slice = slice(paste_start, paste_start + crop_size)
    image_out[paste_slice, ...] = image[crop_slice, ...]

    shuffled_indices = torch.arange(
        paste_start,
        paste_start + crop_size,
        device=image.device,
    )
    if crop_size > 1:
        shuffled_indices = shuffled_indices[torch.randperm(crop_size, device=image.device)]

    cutout_count = random.randint(0, min(int(cutout_max_count), crop_size))
    if random.random() < float(cutout_probability) and cutout_count > 0:
        image_out[shuffled_indices[:cutout_count], ...] = 0

    return image_out


class LayerMixDropoutTransform(ImageOnlyTransform):
    """Apply ``layer_mix_dropout`` to channel-first tensors."""

    def __init__(
        self,
        min_crop_ratio: float = 0.25,
        max_crop_ratio: float = 0.75,
        cutout_max_count: int = 0,
        cutout_probability: float = 0.0,
    ):
        if min_crop_ratio <= 0:
            raise ValueError(f"min_crop_ratio must be > 0, got {min_crop_ratio}")
        if max_crop_ratio <= 0:
            raise ValueError(f"max_crop_ratio must be > 0, got {max_crop_ratio}")
        if min_crop_ratio > max_crop_ratio:
            raise ValueError(
                f"min_crop_ratio ({min_crop_ratio}) must be <= max_crop_ratio ({max_crop_ratio})"
            )
        if cutout_max_count < 0:
            raise ValueError(f"cutout_max_count must be >= 0, got {cutout_max_count}")
        if not 0.0 <= cutout_probability <= 1.0:
            raise ValueError(
                f"cutout_probability must be in [0, 1], got {cutout_probability}"
            )

        super().__init__()
        self.min_crop_ratio = float(min_crop_ratio)
        self.max_crop_ratio = float(max_crop_ratio)
        self.cutout_max_count = int(cutout_max_count)
        self.cutout_probability = float(cutout_probability)
        self._perf_name = 'layer_mix_dropout'

    def _apply_to_image(self, img: torch.Tensor, **params) -> torch.Tensor:
        return layer_mix_dropout(
            img,
            min_crop_ratio=self.min_crop_ratio,
            max_crop_ratio=self.max_crop_ratio,
            cutout_max_count=self.cutout_max_count,
            cutout_probability=self.cutout_probability,
        )
