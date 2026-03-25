from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ink.recipes.data.normalization import build_normalization_transform


def _resize_mask_for_loss(mask, *, patch_size: int):
    """Resize masks to the quarter-resolution grid expected by the loss path."""
    import torch
    import torch.nn.functional as F

    if not torch.is_floating_point(mask):
        mask = mask.float()
    else:
        mask = mask.to(dtype=torch.float32)
    if mask.numel() > 0 and float(mask.max().detach().item()) > 1.0:
        mask = mask / 255.0
    return F.interpolate(mask.unsqueeze(0), (int(patch_size) // 4, int(patch_size) // 4)).squeeze(0)

