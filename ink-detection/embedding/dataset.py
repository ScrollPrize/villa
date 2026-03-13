from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from albumentations import (
    Affine,
    Compose,
    HorizontalFlip,
    Lambda,
    RandomBrightnessContrast,
    RandomGamma,
    VerticalFlip,
)
from scipy.interpolate import Rbf
from torch.utils.data import Dataset


SUPPORTED_SUFFIXES = {".png"}
TRAIN_AFFINE_SCALE = (0.7, 1.4)
TRAIN_AFFINE_TRANSLATE_PERCENT = 0.08
TRAIN_AFFINE_ROTATE = (-30, 30)
TRAIN_AFFINE_SHEAR = (-8, 8)


def image_to_float32(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.size > 0 and float(np.max(image)) > 1.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def load_grayscale_image(path: Path, downsample_factor: int = 1) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read grayscale PNG: {path}")
    if downsample_factor > 1:
        image = cv2.resize(
            image,
            (
                max(1, image.shape[1] // downsample_factor),
                max(1, image.shape[0] // downsample_factor),
            ),
            interpolation=cv2.INTER_AREA,
        )
    return image_to_float32(image)


def list_pngs(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


class ThinPlateSplineWarp:
    def __init__(
        self,
        p: float = 0.35,
        num_control_points: int = 3,
        max_displacement_ratio: float = 0.06,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> None:
        self.p = float(p)
        self.num_control_points = int(num_control_points)
        self.max_displacement_ratio = float(max_displacement_ratio)
        self.interpolation = interpolation

    def __call__(self, image: np.ndarray, **_: Any) -> np.ndarray:
        if random.random() >= self.p:
            return image

        height, width = image.shape[:2]
        side = float(min(height, width))
        displacement = side * self.max_displacement_ratio
        if displacement <= 0:
            return image

        xs = np.linspace(0.0, width - 1.0, self.num_control_points, dtype=np.float32)
        ys = np.linspace(0.0, height - 1.0, self.num_control_points, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        dest_points = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=1)
        source_points = dest_points.copy()

        jitter = np.random.uniform(-displacement, displacement, size=source_points.shape).astype(np.float32)
        source_points += jitter
        source_points[:, 0] = np.clip(source_points[:, 0], 0.0, width - 1.0)
        source_points[:, 1] = np.clip(source_points[:, 1], 0.0, height - 1.0)

        corner_indices = np.array(
            [
                0,
                self.num_control_points - 1,
                self.num_control_points * (self.num_control_points - 1),
                self.num_control_points * self.num_control_points - 1,
            ],
            dtype=np.int64,
        )
        source_points[corner_indices] = dest_points[corner_indices]

        rbf_x = Rbf(dest_points[:, 0], dest_points[:, 1], source_points[:, 0], function="thin_plate", smooth=1e-3)
        rbf_y = Rbf(dest_points[:, 0], dest_points[:, 1], source_points[:, 1], function="thin_plate", smooth=1e-3)

        map_x, map_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        remap_x = np.asarray(rbf_x(map_x, map_y), dtype=np.float32)
        remap_y = np.asarray(rbf_y(map_x, map_y), dtype=np.float32)
        return cv2.remap(
            image,
            remap_x,
            remap_y,
            interpolation=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )


def additive_gaussian_noise(image: np.ndarray, sigma_min: float = 0.01, sigma_max: float = 0.05) -> np.ndarray:
    sigma = random.uniform(sigma_min, sigma_max)
    noisy = image + np.random.normal(loc=0.0, scale=sigma, size=image.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def build_train_augmentations(crop_size: int) -> Compose:
    return Compose(
        [
            Affine(
                scale=TRAIN_AFFINE_SCALE,
                translate_percent=(-TRAIN_AFFINE_TRANSLATE_PERCENT, TRAIN_AFFINE_TRANSLATE_PERCENT),
                rotate=TRAIN_AFFINE_ROTATE,
                shear=TRAIN_AFFINE_SHEAR,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                cval=0,
                p=0.95,
            ),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            Lambda(image=ThinPlateSplineWarp(p=0.45, max_displacement_ratio=0.06)),
            RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            Lambda(image=lambda image, **_: additive_gaussian_noise(image), p=0.55),
            Lambda(image=lambda image, **_: np.clip(image, 0.0, 1.0)),
            Lambda(image=lambda image, **_: cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)),
        ]
    )


def build_eval_augmentations(crop_size: int) -> Compose:
    return Compose([Lambda(image=lambda image, **_: cv2.resize(image, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR))])


@dataclass
class DatasetConfig:
    image_dir: Path
    split: str
    seed: int
    crop_size: int
    downsample_factor: int
    samples_per_epoch: int
    min_foreground_fraction: float
    max_crop_attempts: int
    test_fraction: float
    foreground_threshold: float
    cache_images: bool


class _BaseInkCropDataset(Dataset):
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        all_paths = list_pngs(config.image_dir)
        if len(all_paths) < 2:
            raise ValueError(f"Need at least 2 PNG files under {config.image_dir}, found {len(all_paths)}")

        rng = random.Random(config.seed)
        shuffled = list(all_paths)
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * config.test_fraction)))
        test_count = min(test_count, len(shuffled) - 1)
        train_count = len(shuffled) - test_count
        if config.split == "train":
            self.paths = shuffled[:train_count]
        elif config.split == "test":
            self.paths = shuffled[train_count:]
        else:
            raise ValueError(f"Unknown split {config.split!r}")

        self.samples_per_epoch = int(config.samples_per_epoch)
        self.min_foreground_pixels = max(1, int(math.ceil(config.min_foreground_fraction * config.crop_size * config.crop_size)))
        self.foreground_threshold = float(config.foreground_threshold)
        self.max_crop_attempts = int(config.max_crop_attempts)

        self.images: list[np.ndarray] = []
        self.foreground_coords: list[np.ndarray] = []
        self.image_nonzero_weights: list[float] = []
        retained_paths: list[Path] = []
        for path in self.paths:
            image = load_grayscale_image(path, downsample_factor=config.downsample_factor)
            foreground = np.argwhere(image > self.foreground_threshold)
            if foreground.shape[0] < self.min_foreground_pixels:
                continue
            retained_paths.append(path)
            self.images.append(image if config.cache_images else image)
            self.foreground_coords.append(foreground)
            self.image_nonzero_weights.append(float(np.count_nonzero(image)))

        self.paths = retained_paths
        if not self.paths:
            raise ValueError(
                f"No usable images in split {config.split!r} after foreground filtering. "
                f"Lower --min-foreground-fraction or --foreground-threshold."
            )
        self.total_image_nonzero_weight = float(sum(self.image_nonzero_weights))
        if self.total_image_nonzero_weight <= 0.0:
            raise ValueError(f"No nonzero pixels available for weighted sampling in split {config.split!r}.")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _crop_at(
        self,
        image: np.ndarray,
        center_y: int,
        center_x: int,
        rng: random.Random | None = None,
    ) -> np.ndarray:
        height, width = image.shape
        crop_size = self.config.crop_size
        top = int(np.clip(center_y - crop_size // 2, 0, max(0, height - crop_size)))
        left = int(np.clip(center_x - crop_size // 2, 0, max(0, width - crop_size)))
        crop = image[top : top + crop_size, left : left + crop_size]
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            padded = np.zeros((crop_size, crop_size), dtype=np.float32)
            random_source = rng or random
            offset_y = 0 if crop.shape[0] == crop_size else random_source.randrange(crop_size - crop.shape[0] + 1)
            offset_x = 0 if crop.shape[1] == crop_size else random_source.randrange(crop_size - crop.shape[1] + 1)
            padded[offset_y : offset_y + crop.shape[0], offset_x : offset_x + crop.shape[1]] = crop
            crop = padded
        return crop

    def _sample_crop_with_rng(
        self,
        image: np.ndarray,
        foreground: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape
        crop_size = self.config.crop_size

        for _ in range(self.max_crop_attempts):
            if foreground.shape[0] > 0 and rng.random() < 0.8:
                center_y, center_x = foreground[rng.randrange(foreground.shape[0])]
                center_y = int(center_y)
                center_x = int(center_x)
            else:
                center_y = rng.randrange(height)
                center_x = rng.randrange(width)

            top = int(np.clip(center_y - crop_size // 2, 0, max(0, height - crop_size)))
            left = int(np.clip(center_x - crop_size // 2, 0, max(0, width - crop_size)))
            crop = self._crop_at(image, center_y, center_x, rng=rng)
            foreground_pixels = int((crop > self.foreground_threshold).sum())
            if foreground_pixels >= self.min_foreground_pixels:
                return crop, foreground_pixels / float(crop_size * crop_size), top, left

        center_y, center_x = foreground[rng.randrange(foreground.shape[0])]
        top = int(np.clip(int(center_y) - crop_size // 2, 0, max(0, height - crop_size)))
        left = int(np.clip(int(center_x) - crop_size // 2, 0, max(0, width - crop_size)))
        crop = self._crop_at(image, int(center_y), int(center_x), rng=rng)
        return crop, float((crop > self.foreground_threshold).mean()), top, left

    def _sample_image_index_with_rng(self, rng: random.Random) -> int:
        target = rng.random() * self.total_image_nonzero_weight
        cumulative = 0.0
        for image_index, weight in enumerate(self.image_nonzero_weights):
            cumulative += weight
            if target < cumulative:
                return image_index
        return len(self.image_nonzero_weights) - 1


class InkCropDataset(_BaseInkCropDataset):
    def __init__(self, config: DatasetConfig, augmentation: Compose) -> None:
        super().__init__(config)
        self.augmentation = augmentation

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_index = self._sample_image_index_with_rng(random)
        image = self.images[image_index]
        foreground = self.foreground_coords[image_index]
        crop, foreground_fraction, _top, _left = self._sample_crop_with_rng(image, foreground, random)
        view1 = self.augmentation(image=crop)["image"].astype(np.float32)
        view2 = self.augmentation(image=crop)["image"].astype(np.float32)
        crop = np.clip(crop, 0.0, 1.0)
        view1 = np.clip(view1, 0.0, 1.0)
        view2 = np.clip(view2, 0.0, 1.0)
        return {
            "view1": torch.from_numpy(view1).unsqueeze(0),
            "view2": torch.from_numpy(view2).unsqueeze(0),
            "base": torch.from_numpy(crop).unsqueeze(0),
            "foreground_fraction": torch.tensor(foreground_fraction, dtype=torch.float32),
            "image_index": torch.tensor(image_index, dtype=torch.long),
        }


class IndexedInkCropDataset(_BaseInkCropDataset):
    def __init__(self, config: DatasetConfig, augmentation: Compose) -> None:
        super().__init__(config)
        self.augmentation = augmentation

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = random.Random(self.config.seed + index)
        image_index = self._sample_image_index_with_rng(rng)
        image = self.images[image_index]
        foreground = self.foreground_coords[image_index]
        crop, foreground_fraction, top, left = self._sample_crop_with_rng(image, foreground, rng)
        augmented = self.augmentation(image=crop)["image"].astype(np.float32)
        crop = np.clip(crop, 0.0, 1.0)
        augmented = np.clip(augmented, 0.0, 1.0)
        path = self.paths[image_index]
        return {
            "base": torch.from_numpy(crop).unsqueeze(0),
            "image": torch.from_numpy(augmented).unsqueeze(0),
            "foreground_fraction": torch.tensor(foreground_fraction, dtype=torch.float32),
            "image_index": torch.tensor(image_index, dtype=torch.long),
            "sample_index": torch.tensor(index, dtype=torch.long),
            "top": torch.tensor(top, dtype=torch.long),
            "left": torch.tensor(left, dtype=torch.long),
            "path": str(path),
        }
