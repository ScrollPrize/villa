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


def _apply_joint_transform(transform, image, label, *, patch_size: int):
    data = transform(image=image, mask=label)
    image = data["image"].unsqueeze(0)
    label = _resize_mask_for_loss(data["mask"], patch_size=patch_size)
    return image, label


def _apply_joint_transform_with_valid_mask(transform, image, label, *, patch_size: int, valid_mask):
    data = transform(image=image, mask=label, valid_mask=valid_mask)
    image = data["image"].unsqueeze(0)
    label = _resize_mask_for_loss(data["mask"], patch_size=patch_size)
    valid_mask = _resize_mask_for_loss(data["valid_mask"], patch_size=patch_size)
    return image, label, valid_mask


def _apply_image_only_transform(transform, image):
    data = transform(image=image)
    return data["image"].unsqueeze(0)


def _build_common_transform_ops(split: str, *, augment, normalization, patch_size: int, in_channels: int):
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
    return transforms


def build_joint_transform(split: str, *, augment, normalization, patch_size: int, in_channels: int):
    """Build the shared Albumentations pipeline for the requested split."""
    transforms = _build_common_transform_ops(
        split,
        augment=augment,
        normalization=normalization,
        patch_size=patch_size,
        in_channels=in_channels,
    )
    transforms.append(ToTensorV2(transpose_mask=True))
    return A.Compose(transforms, additional_targets={"valid_mask": "mask"})


def build_image_transform(split: str, *, augment, normalization, patch_size: int, in_channels: int):
    """Build the image-only Albumentations pipeline for inference-style datasets."""
    transforms = _build_common_transform_ops(
        split,
        augment=augment,
        normalization=normalization,
        patch_size=patch_size,
        in_channels=in_channels,
    )
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


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
    patch_size = int(patch_size)
    image = augment.apply_train_image(image)
    if valid_mask is None:
        return _apply_joint_transform(
            transform,
            image,
            label,
            patch_size=patch_size,
        )
    return _apply_joint_transform_with_valid_mask(
        transform,
        image,
        label,
        patch_size=patch_size,
        valid_mask=valid_mask,
    )


def apply_eval_sample_transforms(
    image,
    label,
    *,
    patch_size,
    transform,
    valid_mask=None,
):
    """Apply the eval transform path without train-only image augmentation."""
    patch_size = int(patch_size)
    if valid_mask is None:
        return _apply_joint_transform(
            transform,
            image,
            label,
            patch_size=patch_size,
        )
    return _apply_joint_transform_with_valid_mask(
        transform,
        image,
        label,
        patch_size=patch_size,
        valid_mask=valid_mask,
    )


def apply_infer_sample_transforms(
    image,
    *,
    transform,
):
    """Apply the eval transform path for image-only inference samples."""
    return _apply_image_only_transform(
        transform,
        image,
    )


__all__ = [
    "apply_eval_sample_transforms",
    "apply_infer_sample_transforms",
    "apply_train_sample_transforms",
    "build_image_transform",
    "build_joint_transform",
]
