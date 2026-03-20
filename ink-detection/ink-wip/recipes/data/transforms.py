from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from ink.recipes.data.normalization import build_normalization_transform


def _default_valid_mask_for_label(label):
    import numpy as np

    return np.ones_like(label, dtype=np.uint8)


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


def _apply_joint_transform(transform, image, label, *, patch_size: int, valid_mask=None):
    valid_mask = _default_valid_mask_for_label(label) if valid_mask is None else valid_mask
    data = transform(image=image, mask=label, valid_mask=valid_mask)
    image = data["image"].unsqueeze(0)
    label = _resize_mask_for_loss(data["mask"], patch_size=patch_size)
    valid_mask = _resize_mask_for_loss(data["valid_mask"], patch_size=patch_size)
    return image, label, valid_mask


def build_joint_transform(split: str, *, augment, normalization, patch_size: int, in_channels: int):
    """Build the shared Albumentations pipeline for the requested split."""
    split = str(split).strip().lower()
    patch_size = int(patch_size)
    in_channels = int(in_channels)
    transforms = [A.Resize(patch_size, patch_size)]
    if split == "train":
        transforms.extend(augment.build_train_ops())
    elif split == "valid":
        transforms.extend(augment.build_valid_ops())
    else:
        raise ValueError(f"Unknown augmentation split: {split!r}. Expected 'train' or 'valid'.")
    transforms.append(build_normalization_transform(normalization, in_channels=in_channels))
    transforms.append(ToTensorV2(transpose_mask=True))
    return A.Compose(transforms, additional_targets={"valid_mask": "mask"})


def apply_train_sample_transforms(
    image,
    label,
    *,
    augment,
    patch_size,
    transform,
    valid_mask=None,
):
    """Apply train-only image ops before the joint image/label/mask transform."""
    has_valid_mask = valid_mask is not None
    patch_size = int(patch_size)
    image = augment.apply_train_image(image)
    image, label, valid_mask = _apply_joint_transform(
        transform,
        image,
        label,
        patch_size=patch_size,
        valid_mask=valid_mask,
    )
    if not has_valid_mask:
        return image, label
    return image, label, valid_mask


def apply_eval_sample_transforms(
    image,
    label,
    *,
    patch_size,
    transform,
    valid_mask=None,
):
    """Apply the eval transform path without train-only image augmentation."""
    has_valid_mask = valid_mask is not None
    patch_size = int(patch_size)
    image, label, valid_mask = _apply_joint_transform(
        transform,
        image,
        label,
        patch_size=patch_size,
        valid_mask=valid_mask,
    )
    if not has_valid_mask:
        return image, label
    return image, label, valid_mask


__all__ = [
    "apply_eval_sample_transforms",
    "apply_train_sample_transforms",
    "build_joint_transform",
]
