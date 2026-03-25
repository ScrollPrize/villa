from typing import Dict, List, Optional, Tuple

import numpy as np

from vesuvius.models.augmentation.transforms.intensity.brightness import (
    BrightnessAdditiveTransform,
)
from vesuvius.models.augmentation.transforms.intensity.contrast import (
    BGContrast,
    ContrastTransform,
)
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import (
    GaussianNoiseTransform,
)
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import (
    BlankRectangleTransform,
)
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import (
    GaussianBlurTransform,
)
from vesuvius.models.augmentation.transforms.noise.layer_mix_dropout import (
    LayerMixDropoutTransform,
)
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform

_DEFAULT_GAUSS_NOISE_STD_RANGE = (
    0.012403473458920844,
    0.027729677693590096,
)
_BRIGHTNESS_LIMIT_RANGE = (-0.2, 0.2)
_CONTRAST_MULTIPLIER_RANGE = (0.8, 1.2)
_GAUSSIAN_BLUR_SIGMA_RANGE = (0.5, 3.0)
_ROTATION_RANGE_RADIANS = (-2.0 * np.pi, 2.0 * np.pi)
_SCALE_RANGE = (0.9, 1.1)


def _mirror_axes_for_dimension(dimension: int) -> Tuple[int, ...]:
    if dimension == 2:
        return (0, 1)
    if dimension == 3:
        # TrainAugment flips only the in-plane spatial axes; z/depth acts like channels.
        return (1, 2)
    raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")


def _rotation_axes_for_dimension(
    dimension: int,
    allowed_rotation_axes: Optional[List[int]],
) -> Optional[Tuple[int, ...]]:
    if allowed_rotation_axes is not None:
        return tuple(sorted({int(axis) for axis in allowed_rotation_axes}))
    if dimension == 3:
        # Rotate around z only to match 2D ShiftScaleRotate behavior on y/x slices.
        return (0,)
    return None


def _coarse_dropout_size_ranges(
    patch_size: Tuple[int, ...],
) -> Tuple[Tuple[int, int], ...]:
    if len(patch_size) == 2:
        return tuple((1, max(1, int(size * 0.2))) for size in patch_size)
    return (
        (int(patch_size[0]), int(patch_size[0])),
        (1, max(1, int(patch_size[1] * 0.2))),
        (1, max(1, int(patch_size[2] * 0.2))),
    )


def create_training_transforms(
    patch_size: Tuple[int, ...],
    no_spatial: bool = False,
    no_scaling: bool = False,
    only_spatial_and_intensity: bool = False,
    allowed_rotation_axes: Optional[List[int]] = None,
    skeleton_targets: Optional[List[str]] = None,
    skeleton_ignore_values: Optional[Dict[str, int]] = None,
) -> ComposeTransforms:
    dimension = len(patch_size)
    if dimension not in (2, 3):
        raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")

    transforms = []

    if not no_spatial:
        transforms.append(MirrorTransform(allowed_axes=_mirror_axes_for_dimension(dimension)))
        transforms.append(
            RandomTransform(
                SpatialTransform(
                    patch_size=patch_size,
                    patch_center_dist_from_border=0,
                    random_crop=False,
                    p_elastic_deform=0,
                    p_rotation=1.0,
                    rotation=_ROTATION_RANGE_RADIANS,
                    p_scaling=0.0 if no_scaling else 1.0,
                    scaling=_SCALE_RANGE,
                    p_synchronize_scaling_across_axes=1.0,
                    bg_style_seg_sampling=False,
                    mode_seg="nearest",
                    allowed_rotation_axes=_rotation_axes_for_dimension(
                        dimension,
                        allowed_rotation_axes,
                    ),
                ),
                apply_probability=0.75,
            )
        )

    transforms.append(
        RandomTransform(
            LayerMixDropoutTransform(
                min_crop_ratio=0.9,
                max_crop_ratio=1.0,
                cutout_max_count=2,
                cutout_probability=0.6,
            ),
            apply_probability=0.6,
        )
    )

    transforms.append(
        RandomTransform(
            ComposeTransforms(
                [
                    BrightnessAdditiveTransform(
                        mu=_BRIGHTNESS_LIMIT_RANGE,
                        sigma=0,
                        synchronize_channels=True,
                        p_per_channel=1.0,
                    ),
                    ContrastTransform(
                        contrast_range=BGContrast(_CONTRAST_MULTIPLIER_RANGE),
                        preserve_range=False,
                        synchronize_channels=True,
                        p_per_channel=1.0,
                    ),
                ]
            ),
            apply_probability=0.75,
        )
    )

    if not only_spatial_and_intensity:
        # No direct MotionBlur equivalent exists in the Vesuvius library.
        transforms.append(
            RandomTransform(
                OneOfTransform(
                    [
                        GaussianNoiseTransform(
                            noise_variance=_DEFAULT_GAUSS_NOISE_STD_RANGE,
                            p_per_channel=1.0,
                            synchronize_channels=True,
                        ),
                        GaussianBlurTransform(
                            blur_sigma=_GAUSSIAN_BLUR_SIGMA_RANGE,
                            synchronize_channels=True,
                            synchronize_axes=False,
                            p_per_channel=1.0,
                            benchmark=True,
                        ),
                    ]
                ),
                apply_probability=0.4,
            )
        )
        transforms.append(
            RandomTransform(
                BlankRectangleTransform(
                    rectangle_size=_coarse_dropout_size_ranges(patch_size),
                    rectangle_value=0,
                    num_rectangles=(1, 2),
                    force_square=False,
                    p_per_sample=1.0,
                    p_per_channel=1.0,
                ),
                apply_probability=0.5,
            )
        )

    if skeleton_targets:
        from vesuvius.models.augmentation.transforms.utils.skeleton_transform import (
            MedialSurfaceTransform,
        )

        transforms.append(
            MedialSurfaceTransform(
                do_tube=False,
                target_keys=skeleton_targets,
                ignore_values=skeleton_ignore_values or None,
            )
        )

    return ComposeTransforms(transforms)


def create_validation_transforms(
    skeleton_targets: Optional[List[str]] = None,
    skeleton_ignore_values: Optional[Dict[str, int]] = None,
) -> Optional[ComposeTransforms]:
    if not skeleton_targets:
        return None

    from vesuvius.models.augmentation.transforms.utils.skeleton_transform import (
        MedialSurfaceTransform,
    )

    transforms = [
        MedialSurfaceTransform(
            do_tube=False,
            target_keys=skeleton_targets,
            ignore_values=skeleton_ignore_values or None,
        )
    ]

    return ComposeTransforms(transforms)
