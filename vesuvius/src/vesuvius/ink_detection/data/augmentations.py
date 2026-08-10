"""Ink training augmentation presets and native-crop translation."""

from __future__ import annotations

import numpy as np

from vesuvius.models.augmentation.transforms.intensity.brightness import (
    BrightnessAdditiveTransform,
    MultiplicativeBrightnessTransform,
)
from vesuvius.models.augmentation.transforms.intensity.contrast import (
    BGContrast,
    ContrastTransform,
)
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import (
    GaussianNoiseTransform,
)
from vesuvius.models.augmentation.transforms.intensity.illumination import (
    InhomogeneousSliceIlluminationTransform,
)
from vesuvius.models.augmentation.transforms.intensity.inversion import (
    InvertImageTransform,
)
from vesuvius.models.augmentation.transforms.local import (
    BrightnessGradientAdditiveTransform,
)
from vesuvius.models.augmentation.transforms.noise import SharpeningTransform
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import (
    BlankRectangleTransform,
    SmearTransform,
)
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import (
    GaussianBlurTransform,
)
from vesuvius.models.augmentation.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
from vesuvius.models.augmentation.transforms.spatial.transpose import (
    TransposeAxesTransform,
)
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform


NATIVE_CROP_TRANSLATION_PROBABILITY = 0.30
NATIVE_CROP_TRANSLATION_MIN_VOXELS = 10
NATIVE_CROP_TRANSLATION_MAX_VOXELS = 40
NATIVE_CROP_TRANSLATION_KEEP_FRACTION = 1.0 / 3.0
NATIVE_CROP_TRANSLATION_MAX_AXES = 2
NATIVE_CROP_TRANSLATION_MAX_ATTEMPTS = 24


def _mirror_axes(dimension: int) -> tuple[int, ...]:
    if dimension == 2:
        return 0, 1
    if dimension == 3:
        return 1, 2
    raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3")


def _rotation_axes(
    patch_size: tuple[int, ...], requested: tuple[int, ...] | None
) -> set[int] | None:
    if len(patch_size) == 2:
        return {0, 1} if patch_size[0] == patch_size[1] else None
    planes = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    allowed: set[int] = set()
    for rotation_axis in ({0} if requested is None else set(requested)):
        plane = planes.get(rotation_axis)
        if plane is None:
            continue
        axis_a, axis_b = sorted(plane)
        if patch_size[axis_a] == patch_size[axis_b]:
            allowed.update(plane)
    return allowed or None


def create_spatial_only_transforms(
    patch_size: tuple[int, ...], rotation_axes: tuple[int, ...] | None = None
) -> ComposeTransforms:
    """Create label-safe flips and rotations without intensity edits."""
    dimension = len(patch_size)
    transforms = [MirrorTransform(allowed_axes=_mirror_axes(dimension))]
    allowed = _rotation_axes(patch_size, rotation_axes)
    if allowed is not None:
        transforms.append(
            RandomTransform(
                Rot90Transform(
                    num_axis_combinations=1,
                    num_rot_per_combination=(1, 2, 3),
                    allowed_axes=allowed,
                ),
                apply_probability=0.3,
            )
        )
    return ComposeTransforms(transforms)


def create_spatial_intensity_no_clip_transforms(
    patch_size: tuple[int, ...], rotation_axes: tuple[int, ...] | None = None
) -> ComposeTransforms:
    """Create the bounded no-clipping augmentation preset."""
    transforms = list(
        create_spatial_only_transforms(patch_size, rotation_axes).transforms
    )
    transforms.extend(
        [
            RandomTransform(
                InvertImageTransform(
                    p_invert_image=1.0,
                    p_synchronize_channels=1.0,
                    p_per_channel=1.0,
                ),
                apply_probability=0.2,
            ),
            RandomTransform(
                BrightnessAdditiveTransform(
                    mu=(-0.15, 0.15),
                    sigma=0.0,
                    synchronize_channels=True,
                    p_per_channel=1.0,
                ),
                apply_probability=0.3,
            ),
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0.005, 0.03),
                    p_per_channel=1.0,
                    synchronize_channels=True,
                ),
                apply_probability=0.25,
            ),
        ]
    )
    return ComposeTransforms(transforms)


def create_default_training_transforms(
    patch_size: tuple[int, ...],
) -> ComposeTransforms:
    """Compose the default ink training transform graph."""
    dimension = len(patch_size)
    if dimension not in (2, 3):
        raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3")
    transforms = []
    transpose_axes: set[int] = set()
    if dimension == 3:
        for axis_a, axis_b in ((0, 1), (0, 2), (1, 2)):
            if patch_size[axis_a] == patch_size[axis_b]:
                transpose_axes.update((axis_a, axis_b))
    if dimension == 2:
        transforms.append(MirrorTransform(allowed_axes=(0, 1)))
    else:
        allowed = set()
        if patch_size[1] == patch_size[2]:
            allowed.update((1, 2))
        if patch_size[0] == patch_size[2]:
            allowed.update((0, 2))
        if patch_size[0] == patch_size[1]:
            allowed.update((0, 1))
        if allowed:
            transforms.append(
                RandomTransform(
                    Rot90Transform(
                        num_axis_combinations=1,
                        num_rot_per_combination=(1, 2, 3),
                        allowed_axes=allowed,
                    ),
                    apply_probability=0.5,
                )
            )

    blank_rectangle = RandomTransform(
        BlankRectangleTransform(
            rectangle_size=tuple(
                (max(1, size // 6), size // 3) for size in patch_size
            ),
            rectangle_value=np.mean,
            num_rectangles=(1, 5),
            force_square=False,
            p_per_sample=0.4,
            p_per_channel=0.5,
        ),
        apply_probability=0.1,
    )

    common = [
        OneOfTransform(
            [
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.3, 1.5),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=False,
                    ),
                    apply_probability=0.3,
                )
            ]
        ),
        RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.2),
                p_per_channel=0.5,
                synchronize_channels=True,
            ),
            apply_probability=0.3,
        ),
        RandomTransform(
            SharpeningTransform(
                strength=(0.1, 1.5),
                p_same_for_each_channel=0.5,
                p_per_channel=0.5,
                p_clamp_intensities=0.5,
            ),
            apply_probability=0.2,
        ),
    ]
    common.extend(
        [
            OneOfTransform(
                [
                    RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.75, 1.25)),
                            preserve_range=True,
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        MultiplicativeBrightnessTransform(
                            multiplier_range=BGContrast((0.75, 1.25)),
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                ]
            ),
            RandomTransform(
                BrightnessAdditiveTransform(
                    mu=0,
                    sigma=0.5,
                    synchronize_channels=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.1,
            ),
        ]
    )
    common.append(
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.25, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=0.4,
        )
    )
    common.extend(
        [
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.2,
            ),
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.7, 1.5)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.4,
            ),
            RandomTransform(
                InvertImageTransform(
                    p_invert_image=1,
                    p_synchronize_channels=0.5,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            ),
        ]
    )
    if dimension == 2:
        transforms.append(blank_rectangle)
        transforms.extend(common)
        return ComposeTransforms(transforms)
    if len(transpose_axes) >= 2:
        transforms.append(
            RandomTransform(
                TransposeAxesTransform(allowed_axes=transpose_axes),
                apply_probability=0.2,
            )
        )
    transforms.append(blank_rectangle)
    transforms.append(
        RandomTransform(
            SmearTransform(shift=(5, 0), alpha=0.2, num_prev_slices=3, smear_axis=3),
            apply_probability=0.3,
        )
    )
    transforms.extend(
        [
            RandomTransform(
                InhomogeneousSliceIlluminationTransform(
                    num_defects=(2, 5),
                    defect_width=(25, 50),
                    mult_brightness_reduction_at_defect=(0.3, 1.5),
                    base_p=(0.2, 0.4),
                    base_red=(0.5, 0.9),
                    p_per_sample=1.0,
                    per_channel=True,
                    p_per_channel=0.5,
                ),
                apply_probability=0.4,
            ),
            RandomTransform(
                BrightnessGradientAdditiveTransform(
                    scale=(min(patch_size) / 6, min(patch_size) / 2),
                    loc=(-0.5, 1.5),
                    max_strength=0.5,
                    same_for_all_channels=False,
                    mean_centered=True,
                    clip_intensities=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            ),
        ]
    )
    transforms.extend(common)
    return ComposeTransforms(transforms)


def build_augmentations(
    preset: str,
    patch_size: tuple[int, ...],
    *,
    rotation_axes: tuple[int, ...] | None,
) -> ComposeTransforms | None:
    """Build one configured preset, returning no transform for `none`."""

    if preset == "none":
        return None
    if preset == "spatial_only":
        return create_spatial_only_transforms(patch_size, rotation_axes)
    if preset == "spatial_intensity_no_clip":
        return create_spatial_intensity_no_clip_transforms(patch_size, rotation_axes)
    if preset == "default":
        return create_default_training_transforms(patch_size)
    raise ValueError(f"unknown augmentation preset {preset!r}")


def maybe_translate_crop_bbox(
    crop_bbox_zyx,
    positions_zyx,
    valid_mask,
    supervision_flat,
    *,
    rng=None,
):
    """Translate a native crop while retaining the required supervised points."""
    rng = np.random if rng is None else rng
    if rng.random() >= NATIVE_CROP_TRANSLATION_PROBABILITY:
        return crop_bbox_zyx
    constrained = np.zeros(np.asarray(valid_mask).shape, dtype=bool)
    supervision_flat = np.asarray(supervision_flat)
    overlap = tuple(
        slice(0, min(a, b)) for a, b in zip(constrained.shape, supervision_flat.shape)
    )
    constrained[overlap] = supervision_flat[overlap] != 0
    constrained &= np.asarray(valid_mask, dtype=bool)
    constrained &= np.isfinite(positions_zyx).all(axis=-1)
    points = np.asarray(positions_zyx)[constrained].astype(np.int64, copy=False)
    if points.shape[0] == 0:
        return crop_bbox_zyx
    required = max(
        1,
        int(np.ceil(points.shape[0] * NATIVE_CROP_TRANSLATION_KEEP_FRACTION)),
    )
    axes = np.array([1, 2], dtype=np.int64)
    for _ in range(NATIVE_CROP_TRANSLATION_MAX_ATTEMPTS):
        selected = axes.copy()
        rng.shuffle(selected)
        count = (
            2
            if NATIVE_CROP_TRANSLATION_MAX_AXES >= 2 and rng.random() < 0.5
            else 1
        )
        translation = np.zeros(3, dtype=np.int64)
        for axis in selected[:count]:
            magnitude = int(
                rng.randint(
                    NATIVE_CROP_TRANSLATION_MIN_VOXELS,
                    NATIVE_CROP_TRANSLATION_MAX_VOXELS + 1,
                )
            )
            translation[int(axis)] = -magnitude if rng.random() < 0.5 else magnitude
        starts = np.asarray(crop_bbox_zyx[:3]) + translation
        stops = np.asarray(crop_bbox_zyx[3:]) + translation
        within = np.all((points >= starts) & (points < stops), axis=1)
        if int(within.sum()) >= required:
            return tuple(int(value) for value in np.concatenate([starts, stops]))
    return crop_bbox_zyx
