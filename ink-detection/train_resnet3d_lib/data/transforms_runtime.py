import random

import albumentations as A
import numpy as np
import torch
import torch.nn.functional as F

from train_resnet3d_lib.config import CFG


def get_transforms(data, cfg):
    if data == "train":
        aug = A.Compose(cfg.train_aug_list)
    elif data == "valid":
        aug = A.Compose(cfg.valid_aug_list)
    else:
        raise ValueError(f"unknown transform split: {data!r}")
    return aug


def _resize_label_for_loss(label, cfg):
    if not torch.is_floating_point(label):
        label = label.float()
    else:
        label = label.to(dtype=torch.float32)
    if label.numel() > 0 and float(label.max().detach().item()) > 1.0:
        label = label / 255.0
    return F.interpolate(label.unsqueeze(0), (cfg.size // 4, cfg.size // 4)).squeeze(0)


def _apply_joint_transform(transform, image, label, cfg):
    if transform is None:
        return image, label
    data = transform(image=image, mask=label)
    image = data["image"].unsqueeze(0)
    label = _resize_label_for_loss(data["mask"], cfg)
    return image, label


def _apply_image_transform(transform, image):
    if transform is None:
        return image
    data = transform(image=image)
    return data["image"].unsqueeze(0)


def _xy_to_bounds(xy):
    x1, y1, x2, y2 = [int(v) for v in xy]
    return x1, y1, x2, y2


def _fourth_augment(image, cfg):
    in_chans = int(cfg.in_chans)
    if in_chans <= 0:
        raise ValueError(f"in_chans must be > 0 for fourth augment, got {in_chans}")
    if image.shape[-1] != in_chans:
        raise ValueError(
            f"fourth augment expected image with {in_chans} channels, got shape {tuple(image.shape)}"
        )

    min_crop_ratio = float(cfg.fourth_augment_min_crop_ratio)
    max_crop_ratio = float(cfg.fourth_augment_max_crop_ratio)
    if not (0.0 < min_crop_ratio <= max_crop_ratio <= 1.0):
        raise ValueError(
            "fourth augment crop ratios must satisfy 0 < min_crop_ratio <= max_crop_ratio <= 1, "
            f"got min={min_crop_ratio}, max={max_crop_ratio}"
        )

    image_tmp = np.zeros_like(image)
    min_crop = max(1, int(np.ceil(in_chans * min_crop_ratio)))
    max_crop = max(1, int(np.floor(in_chans * max_crop_ratio)))
    if min_crop > max_crop:
        raise ValueError(
            f"invalid fourth augment crop window for in_chans={in_chans}: min_crop={min_crop}, max_crop={max_crop}"
        )
    cropping_num = random.randint(min_crop, max_crop)

    max_start = max(0, in_chans - cropping_num)
    start_idx = random.randint(0, max_start)
    crop_indices = np.arange(start_idx, start_idx + cropping_num)

    start_paste_idx = random.randint(0, max_start)

    tmp = np.arange(start_paste_idx, start_paste_idx + cropping_num)
    np.random.shuffle(tmp)

    cutout_max_count = int(cfg.fourth_augment_cutout_max_count)
    if cutout_max_count < 0:
        raise ValueError(f"fourth_augment_cutout_max_count must be >= 0, got {cutout_max_count}")
    cutout_idx = random.randint(0, min(cutout_max_count, cropping_num))
    temporal_random_cutout_idx = tmp[:cutout_idx]

    image_tmp[..., start_paste_idx:start_paste_idx + cropping_num] = image[..., crop_indices]

    cutout_p = float(cfg.fourth_augment_cutout_p)
    if not (0.0 <= cutout_p <= 1.0):
        raise ValueError(f"fourth_augment_cutout_p must be in [0, 1], got {cutout_p}")
    if random.random() < cutout_p:
        image_tmp[..., temporal_random_cutout_idx] = 0
    return image_tmp


def _maybe_fourth_augment(image, cfg):
    p = float(cfg.fourth_augment_p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"fourth_augment_p must be in [0, 1], got {p}")
    if random.random() < p:
        return _fourth_augment(image, cfg)
    return image


def _invert_augment(image):
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy array, got {type(image).__name__}")
    if image.size == 0:
        raise ValueError("image must be non-empty for invert augment")

    min_value = image.min()
    max_value = image.max()
    if np.issubdtype(image.dtype, np.integer):
        image_i64 = image.astype(np.int64, copy=False)
        inverted = (np.int64(min_value) + np.int64(max_value)) - image_i64
        return inverted.astype(image.dtype, copy=False)
    if np.issubdtype(image.dtype, np.floating):
        inverted = (float(min_value) + float(max_value)) - image
        return inverted.astype(image.dtype, copy=False)
    raise TypeError(f"unsupported image dtype for invert augment: {image.dtype}")


def _maybe_invert_augment(image, cfg):
    p = float(cfg.invert_augment_p)
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"invert_augment_p must be in [0, 1], got {p}")
    if random.random() < p:
        return _invert_augment(image)
    return image


