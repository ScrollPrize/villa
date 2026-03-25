from __future__ import annotations

import random
from dataclasses import dataclass, replace

import albumentations as A
import numpy as np

_DEFAULT_GAUSS_NOISE_STD_RANGE = (
    0.012403473458920844,
    0.027729677693590096,
)


def _require_size(size) -> int:
    if size is None:
        raise ValueError("augment size must be set on the recipe or provided by the data recipe")
    return int(size)


def _build_blur_transforms(recipe):
    blur_transforms = []
    if recipe.use_gauss_noise:
        blur_transforms.append(A.GaussNoise(std_range=recipe.gauss_noise_std_range))
    if recipe.use_gaussian_blur:
        blur_transforms.append(A.GaussianBlur())
    if recipe.use_motion_blur:
        blur_transforms.append(A.MotionBlur())
    return blur_transforms or [A.GaussianBlur(), A.MotionBlur()]


def _fourth_augment(image, recipe):
    in_channels = int(image.shape[-1])
    min_crop = max(1, int(np.ceil(in_channels * float(recipe.fourth_augment_min_crop_ratio))))
    max_crop = max(1, int(np.floor(in_channels * float(recipe.fourth_augment_max_crop_ratio))))
    crop_size = random.randint(min_crop, max_crop)

    max_start = max(0, in_channels - crop_size)
    crop_start = random.randint(0, max_start)
    paste_start = random.randint(0, max_start)

    image_out = np.zeros_like(image)
    crop_indices = np.arange(crop_start, crop_start + crop_size)
    shuffled_indices = np.arange(paste_start, paste_start + crop_size)
    np.random.shuffle(shuffled_indices)
    image_out[..., paste_start:paste_start + crop_size] = image[..., crop_indices]

    cutout_count = random.randint(0, min(int(recipe.fourth_augment_cutout_max_count), crop_size))
    if random.random() < float(recipe.fourth_augment_cutout_p):
        image_out[..., shuffled_indices[:cutout_count]] = 0
    return image_out


def _invert_augment(image):
    if np.issubdtype(image.dtype, np.integer):
        min_value = np.int64(image.min())
        max_value = np.int64(image.max())
        return ((min_value + max_value) - image.astype(np.int64, copy=False)).astype(image.dtype, copy=False)
    if np.issubdtype(image.dtype, np.floating):
        min_value = float(image.min())
        max_value = float(image.max())
        return ((min_value + max_value) - image).astype(image.dtype, copy=False)
    raise TypeError(f"unsupported image dtype for invert augment: {image.dtype}")

@dataclass(frozen=True)
class TrainAugment:
    size: int | None = None
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    brightness_contrast_p: float = 0.75
    brightness_limit: float | tuple[float, float] = 0.2
    contrast_limit: float | tuple[float, float] = 0.2
    brightness_by_max: bool = True
    random_gamma_p: float = 0.0
    random_gamma_limit: tuple[int, int] = (80, 120)
    multiplicative_noise_p: float = 0.0
    multiplicative_noise_multiplier: tuple[float, float] = (0.9, 1.1)
    multiplicative_noise_per_channel: bool = False
    multiplicative_noise_elementwise: bool = False
    shift_scale_rotate_p: float = 0.75
    rotate_limit: int = 360
    shift_limit: float = 0.15
    scale_limit: float = 0.1
    blur_p: float = 0.4
    use_gauss_noise: bool = True
    use_gaussian_blur: bool = True
    use_motion_blur: bool = True
    gauss_noise_std_range: tuple[float, float] = _DEFAULT_GAUSS_NOISE_STD_RANGE
    coarse_dropout_p: float = 0.5
    coarse_dropout_max_holes: int = 2
    coarse_dropout_max_width_ratio: float = 0.2
    coarse_dropout_max_height_ratio: float = 0.2
    fourth_augment_p: float = 0.6
    fourth_augment_min_crop_ratio: float = 0.9
    fourth_augment_max_crop_ratio: float = 1.0
    fourth_augment_cutout_max_count: int = 2
    fourth_augment_cutout_p: float = 0.6
    invert_p: float = 0.0

    def build(self, *, patch_size: int | None = None, runtime=None):
        del runtime
        if self.size is not None or patch_size is None:
            return self
        return replace(self, size=int(patch_size))

    def build_train_ops(self):
        size = _require_size(self.size)
        max_width = max(1, int(size * float(self.coarse_dropout_max_width_ratio)))
        max_height = max(1, int(size * float(self.coarse_dropout_max_height_ratio)))
        return [
            A.HorizontalFlip(p=float(self.horizontal_flip_p)),
            A.VerticalFlip(p=float(self.vertical_flip_p)),
            A.RandomBrightnessContrast(
                brightness_limit=self.brightness_limit,
                contrast_limit=self.contrast_limit,
                brightness_by_max=bool(self.brightness_by_max),
                p=float(self.brightness_contrast_p),
            ),
            A.RandomGamma(gamma_limit=self.random_gamma_limit, p=float(self.random_gamma_p)),
            A.MultiplicativeNoise(
                multiplier=self.multiplicative_noise_multiplier,
                per_channel=bool(self.multiplicative_noise_per_channel),
                elementwise=bool(self.multiplicative_noise_elementwise),
                p=float(self.multiplicative_noise_p),
            ),
            A.ShiftScaleRotate(
                rotate_limit=int(self.rotate_limit),
                shift_limit=float(self.shift_limit),
                scale_limit=float(self.scale_limit),
                p=float(self.shift_scale_rotate_p),
            ),
            A.OneOf(_build_blur_transforms(self), p=float(self.blur_p)),
            A.CoarseDropout(
                num_holes_range=(1, max(1, int(self.coarse_dropout_max_holes))),
                hole_height_range=(1, max_height),
                hole_width_range=(1, max_width),
                fill=0,
                fill_mask=0,
                p=float(self.coarse_dropout_p),
            ),
        ]

    def build_valid_ops(self):
        return []

    def apply_train_image(self, image):
        if random.random() < float(self.fourth_augment_p):
            image = _fourth_augment(image, self)
        if random.random() < float(self.invert_p):
            image = _invert_augment(image)
        return image


__all__ = ["TrainAugment"]

