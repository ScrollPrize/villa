from typing import Dict, List, Optional, Set, Tuple

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
from vesuvius.models.augmentation.transforms.noise import SharpeningTransform
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import (
    BlankRectangleTransform,
)
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import (
    GaussianBlurTransform,
)
from vesuvius.models.augmentation.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from vesuvius.models.augmentation.transforms.spatial.mirroring import (
    MirrorTransform,
)
from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.utils.oneoftransform import (
    OneOfTransform,
)
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform

_DEFAULT_GAUSS_NOISE_STD_RANGE = (
    0.012403473458920844,
    0.027729677693590096,
)
_BRIGHTNESS_LIMIT_RANGE = (-0.2, 0.2)
_CONTRAST_MULTIPLIER_RANGE = (0.8, 1.2)
_GAUSSIAN_BLUR_SIGMA_RANGE = (0.5, 3.0)
_SCALE_RANGE = (0.9, 1.1)


def _mirror_axes_for_dimension(dimension: int) -> Tuple[int, ...]:
    if dimension == 2:
        return (0, 1)
    if dimension == 3:
        # TrainAugment flips only the in-plane spatial axes; z/depth acts like channels.
        return (1, 2)
    raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")


def _rot90_allowed_axes_for_patch(
    patch_size: Tuple[int, ...],
    allowed_rotation_axes: Optional[List[int]],
) -> Optional[Set[int]]:
    dimension = len(patch_size)
    if dimension == 2:
        return {0, 1} if patch_size[0] == patch_size[1] else None

    if dimension == 3:
        rotation_axis_to_plane = {
            0: {1, 2},
            1: {0, 2},
            2: {0, 1},
        }
        requested_axes = (
            {int(axis) for axis in allowed_rotation_axes}
            if allowed_rotation_axes is not None
            else {0}
        )
        allowed_axes: Set[int] = set()
        for rotation_axis in requested_axes:
            plane_axes = rotation_axis_to_plane.get(rotation_axis)
            if plane_axes is None:
                continue
            axis_a, axis_b = sorted(plane_axes)
            if patch_size[axis_a] == patch_size[axis_b]:
                allowed_axes.update(plane_axes)
        return allowed_axes or None

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
        rot90_allowed_axes = _rot90_allowed_axes_for_patch(
            patch_size,
            allowed_rotation_axes,
        )
        if rot90_allowed_axes is not None:
            transforms.append(
                RandomTransform(
                    Rot90Transform(
                        num_axis_combinations=1,
                        num_rot_per_combination=(1, 2, 3),
                        allowed_axes=rot90_allowed_axes,
                    ),
                    apply_probability=0.3,
                )
            )
        # Disabled because SpatialTransform currently leaves keypoints/vectors
        # unchanged, which breaks normal-pooled 3D geometry alignment.

    transforms.append(
        RandomTransform(
            SharpeningTransform(
                strength=(0.1, 1.5),
                p_same_for_each_channel=0.5,
                p_per_channel=0.5,
                p_clamp_intensities=0.5,
            ),
            apply_probability=0.2,
        )
    )
    transforms.append(
        RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=0.5,
            ),
            apply_probability=0.2,
        )
    )
    transforms.append(
        RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.25, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            ),
            apply_probability=0.1,
        )
    )
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.2,
        )
    )
    transforms.append(
        RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=0.3,
        )
    )
    transforms.append(
        RandomTransform(
            InvertImageTransform(
                p_invert_image=1,
                p_synchronize_channels=0.5,
                p_per_channel=0.5,
            ),
            apply_probability=0.2,
        )
    )
    if dimension == 3:
        transforms.append(
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
                apply_probability=0.3,
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
            apply_probability=0.4,
        )
    )
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
            apply_probability=0.2,
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
