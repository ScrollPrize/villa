import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


def _as_int(value, default=None):
    if value is None:
        value = default
    return int(value)


def _as_float(value, default=None):
    if value is None:
        value = default
    return float(value)


def _apply_fold_label_foreground_percentile_clip_zscore(image, *, stats):
    if not isinstance(stats, dict):
        raise ValueError(
            "CFG.fold_label_foreground_percentile_clip_zscore_stats must be set before using "
            "normalization_mode='train_fold_fg_clip_zscore'"
        )
    lower_bound = float(stats["percentile_00_5"])
    upper_bound = float(stats["percentile_99_5"])
    mean_intensity = float(stats["mean"])
    std_intensity = float(stats["std"])
    image = image.astype(np.float32, copy=True)
    np.clip(image, lower_bound, upper_bound, out=image)
    image -= mean_intensity
    image /= std_intensity
    return image


def _apply_fold_label_foreground_percentile_clip_robust_zscore(image, *, stats):
    if not isinstance(stats, dict):
        raise ValueError(
            "CFG.fold_label_foreground_percentile_clip_zscore_stats must be set before using "
            "normalization_mode='train_fold_fg_clip_robust_zscore'"
        )
    lower_bound = float(stats["percentile_00_5"])
    upper_bound = float(stats["percentile_99_5"])
    median_intensity = float(stats["median"])
    robust_scale = float(stats["robust_scale"])
    image = image.astype(np.float32, copy=True)
    np.clip(image, lower_bound, upper_bound, out=image)
    image -= median_intensity
    image /= robust_scale
    return image


def build_intensity_normalization_transform(cfg, *, in_chans):
    mode = str(getattr(cfg, "normalization_mode", "clip_max_div255")).strip().lower()
    if mode == "clip_max_div255":
        return A.Normalize(mean=[0] * in_chans, std=[1] * in_chans)
    if mode == "train_fold_fg_clip_zscore":
        return A.Lambda(
            image=lambda image, **kwargs: _apply_fold_label_foreground_percentile_clip_zscore(
                image,
                stats=getattr(cfg, "fold_label_foreground_percentile_clip_zscore_stats", None),
            ),
            p=1.0,
        )
    if mode == "train_fold_fg_clip_robust_zscore":
        return A.Lambda(
            image=lambda image, **kwargs: _apply_fold_label_foreground_percentile_clip_robust_zscore(
                image,
                stats=getattr(cfg, "fold_label_foreground_percentile_clip_zscore_stats", None),
            ),
            p=1.0,
        )
    raise ValueError(f"Unsupported normalization_mode: {mode!r}")


def rebuild_augmentations(cfg, augmentation_cfg=None):
    augmentation_cfg = dict(augmentation_cfg or {})
    size = int(getattr(cfg, "size"))
    in_chans = int(getattr(cfg, "in_chans"))

    fourth_cfg = dict(augmentation_cfg.get("fourth_augment") or {})
    invert_cfg = dict(augmentation_cfg.get("invert") or {})

    cfg.fourth_augment_p = _as_float(fourth_cfg.get("p"), getattr(cfg, "fourth_augment_p", 0.6))
    cfg.fourth_augment_min_crop_ratio = _as_float(
        fourth_cfg.get("min_crop_ratio"),
        getattr(cfg, "fourth_augment_min_crop_ratio", 0.9),
    )
    cfg.fourth_augment_max_crop_ratio = _as_float(
        fourth_cfg.get("max_crop_ratio"),
        getattr(cfg, "fourth_augment_max_crop_ratio", 1.0),
    )
    cfg.fourth_augment_cutout_max_count = _as_int(
        fourth_cfg.get("cutout_max_count"),
        getattr(cfg, "fourth_augment_cutout_max_count", 2),
    )
    cfg.fourth_augment_cutout_p = _as_float(
        fourth_cfg.get("cutout_p"),
        getattr(cfg, "fourth_augment_cutout_p", 0.6),
    )
    cfg.invert_augment_p = _as_float(invert_cfg.get("p"), getattr(cfg, "invert_augment_p", 0.0))

    shift_scale_rotate = dict(augmentation_cfg.get("shift_scale_rotate") or {})
    blur_cfg = dict(augmentation_cfg.get("blur") or {})
    coarse_dropout_cfg = dict(augmentation_cfg.get("coarse_dropout") or {})

    raw_blur_types = blur_cfg.get("types", ["GaussNoise", "GaussianBlur", "MotionBlur"])
    if isinstance(raw_blur_types, str):
        raw_blur_types = [raw_blur_types]
    blur_types = {str(x) for x in raw_blur_types}

    blur_transforms = []
    if "GaussNoise" in blur_types:
        gauss_noise_std_range = (
            float(np.sqrt(10.0) / 255.0),
            float(np.sqrt(50.0) / 255.0),
        )
        try:
            blur_transforms.append(A.GaussNoise(std_range=gauss_noise_std_range))
        except TypeError:
            blur_transforms.append(A.GaussNoise(var_limit=(10, 50)))
    if "GaussianBlur" in blur_types:
        blur_transforms.append(A.GaussianBlur())
    if "MotionBlur" in blur_types:
        blur_transforms.append(A.MotionBlur())

    max_holes = _as_int(coarse_dropout_cfg.get("max_holes"), 2)
    max_width_ratio = _as_float(coarse_dropout_cfg.get("max_width_ratio"), 0.2)
    max_height_ratio = _as_float(coarse_dropout_cfg.get("max_height_ratio"), 0.2)
    max_width = max(1, _as_int(size * max_width_ratio, int(size * 0.2)))
    max_height = max(1, _as_int(size * max_height_ratio, int(size * 0.2)))

    try:
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, max(1, max_holes)),
            hole_height_range=(1, max_height),
            hole_width_range=(1, max_width),
            fill=0,
            fill_mask=0,
            p=_as_float(coarse_dropout_cfg.get("p"), 0.5),
        )
    except TypeError:
        coarse_dropout = A.CoarseDropout(
            max_holes=max(1, max_holes),
            max_width=max_width,
            max_height=max_height,
            mask_fill_value=0,
            p=_as_float(coarse_dropout_cfg.get("p"), 0.5),
        )

    cfg.train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=_as_float(augmentation_cfg.get("horizontal_flip"), 0.5)),
        A.VerticalFlip(p=_as_float(augmentation_cfg.get("vertical_flip"), 0.5)),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(
            rotate_limit=_as_int(shift_scale_rotate.get("rotate_limit"), 360),
            shift_limit=_as_float(shift_scale_rotate.get("shift_limit"), 0.15),
            scale_limit=_as_float(shift_scale_rotate.get("scale_limit"), 0.1),
            p=_as_float(shift_scale_rotate.get("p"), 0.75),
        ),
        A.OneOf(
            blur_transforms or [A.GaussianBlur(), A.MotionBlur()],
            p=_as_float(blur_cfg.get("p"), 0.4),
        ),
        coarse_dropout,
        build_intensity_normalization_transform(cfg, in_chans=in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    cfg.valid_aug_list = [
        A.Resize(size, size),
        build_intensity_normalization_transform(cfg, in_chans=in_chans),
        ToTensorV2(transpose_mask=True),
    ]
