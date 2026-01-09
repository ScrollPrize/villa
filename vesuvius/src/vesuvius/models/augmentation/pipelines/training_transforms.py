"""
Training augmentation pipeline configuration.

This module contains the standard training augmentation pipeline used for
volumetric data augmentation in the Vesuvius project.
"""

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from vesuvius.models.augmentation.transforms.utils.random import RandomTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform

# Spatial transforms
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.spatial.rot90 import Rot90Transform
from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform

# Intensity transforms
from vesuvius.models.augmentation.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
    BrightnessAdditiveTransform,
)
from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform, BGContrast
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.intensity.illumination import InhomogeneousSliceIlluminationTransform
from vesuvius.models.augmentation.transforms.intensity.inversion import InvertImageTransform

# Noise transforms
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import (
    BlankRectangleTransform,
    SmearTransform,
)
from vesuvius.models.augmentation.transforms.noise import (
    MedianFilterTransform,
    SharpeningTransform,
)

# Local transforms
from vesuvius.models.augmentation.transforms.local import (
    BrightnessGradientAdditiveTransform,
    LocalGammaTransform,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
ENABLE_BLANK_RECTANGLE = False


def create_training_transforms(
    patch_size: Tuple[int, ...],
    no_spatial: bool = False,
    no_scaling: bool = False,
    only_spatial_and_intensity: bool = False,
    allowed_rotation_axes: Optional[List[int]] = None,
    skeleton_targets: Optional[List[str]] = None,
    skeleton_ignore_values: Optional[Dict[str, int]] = None,
) -> ComposeTransforms:
    """
    Create the training augmentation pipeline.

    Parameters
    ----------
    patch_size : Tuple[int, ...]
        The patch size (z, y, x) for 3D or (y, x) for 2D.
    no_spatial : bool
        If True, disable all spatial transforms (rotations, scaling, mirroring).
    no_scaling : bool
        If True, disable scaling augmentation (which requires padding and can cause
        issues with semi-supervised trainers like mean teacher).
    only_spatial_and_intensity : bool
        If True, only use spatial and basic intensity transforms (skip noise, blur, etc.).
    allowed_rotation_axes : Optional[List[int]]
        Restrict rotations to specific axes. None means all axes.
    skeleton_targets : Optional[List[str]]
        List of target names that need skeleton/medial surface transform.
    skeleton_ignore_values : Optional[Dict[str, int]]
        Ignore values for skeleton targets.

    Returns
    -------
    ComposeTransforms
        The composed training augmentation pipeline.
    """
    dimension = len(patch_size)

    if dimension == 2:
        patch_h, patch_w = patch_size
        patch_d = None
    elif dimension == 3:
        patch_d, patch_h, patch_w = patch_size
    else:
        raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")

    # For 3D anisotropic patches, determine valid rotation and transpose axes
    # Rot90: picks 2 axes from allowed_axes to form a rotation plane
    # We can only rotate in a plane if both dimensions in that plane are equal
    # Transpose: can only swap axes with equal dimensions
    if dimension == 3:
        dims_3d = [patch_d, patch_h, patch_w]
        # Valid rotation planes: can rotate in plane (i,j) if dims[i] == dims[j]
        # Collect all axes that are part of at least one valid rotation plane
        rot90_allowed_axes = set()
        if patch_h == patch_w:  # can rotate in H-W plane (axes 1, 2)
            rot90_allowed_axes.update([1, 2])
        if patch_d == patch_w:  # can rotate in D-W plane (axes 0, 2)
            rot90_allowed_axes.update([0, 2])
        if patch_d == patch_h:  # can rotate in D-H plane (axes 0, 1)
            rot90_allowed_axes.update([0, 1])

        # Valid transpose axes: pairs of axes with equal dimensions
        transpose_allowed_axes = set()
        if patch_d == patch_h:
            transpose_allowed_axes.update([0, 1])
        if patch_d == patch_w:
            transpose_allowed_axes.update([0, 2])
        if patch_h == patch_w:
            transpose_allowed_axes.update([1, 2])

    # Local transform scale parameters (derived from patch size)
    min_patch_dim = min(patch_size)
    _local_transform_scale = (min_patch_dim / 6, min_patch_dim / 2)
    _brightness_gradient_max_strength = 0.5
    _local_gamma_gamma = (0.5, 1.5)

    transforms = []

    # =========================================================================
    # SPATIAL TRANSFORMS
    # =========================================================================
    if not no_spatial:
        # Configure rotation based on patch aspect ratio
        if dimension == 2:
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi)
            else:
                rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
            mirror_axes = (0, 1)
        else:  # 3D
            rotation_for_DA = (-np.pi, np.pi)
            mirror_axes = (0, 1, 2)

        # SpatialTransform for scaling (rotation handled by Rot90 + Transpose)
        # Scaling can be disabled for semi-supervised trainers (mean teacher) since
        # scaling requires padding which causes issues with consistency loss.
        scaling_prob = 0 if no_scaling else 0.2
        transforms.append(
            SpatialTransform(
                patch_size,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0,
                rotation=rotation_for_DA,
                p_scaling=scaling_prob,
                scaling=(0.7, 1.4),
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                elastic_deform_magnitude=(5, 25),
                allowed_rotation_axes=allowed_rotation_axes
            )
        )

        # Rot90 for 3D (only if there are valid rotation axes), Mirror for 2D
        if dimension == 3 and rot90_allowed_axes:
            transforms.append(RandomTransform(
                Rot90Transform(
                    num_axis_combinations=1,
                    num_rot_per_combination=(1, 2, 3),
                    allowed_axes=rot90_allowed_axes,
                ),
                apply_probability=0.5
            ))
        elif dimension == 2:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        # Single-axis low resolution simulation (3D only)
        if dimension == 3:
            transforms.append(
                OneOfTransform([
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 1],  # Only affect axis 2
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[0, 2],  # Only affect axis 1
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        SimulateLowResolutionTransform(
                            scale=(0.1, 0.9),
                            synchronize_channels=True,
                            synchronize_axes=False,
                            ignore_axes=[1, 2],  # Only affect axis 0
                            allowed_channels=None,
                        ),
                        apply_probability=0.3,
                    ),
                ])
            )

    # =========================================================================
    # BLANK RECTANGLE (conditional)
    # =========================================================================
    blank_rectangle = None
    if ENABLE_BLANK_RECTANGLE and not only_spatial_and_intensity:
        blank_rectangle = RandomTransform(
            BlankRectangleTransform(
                rectangle_size=tuple(
                    (max(1, size // 6), size // 3) for size in patch_size
                ),
                rectangle_value=np.mean,
                num_rectangles=(1, 5),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5
            ),
            apply_probability=0.1
        )

    # =========================================================================
    # COMMON INTENSITY/NOISE TRANSFORMS
    # =========================================================================
    common_transforms = []

    if not only_spatial_and_intensity:
        # Blur OR Median filter (one of)
        common_transforms.append(
            OneOfTransform([
                RandomTransform(
                    MedianFilterTransform(
                        filter_size=(2, 8),
                        p_same_for_each_channel=0.5,
                        p_per_channel=0.5,
                    ),
                    apply_probability=0.0,
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.3, 1.5),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.3,
                ),
            ])
        )

        # Gaussian noise
        common_transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.2),
                p_per_channel=0.5,
                synchronize_channels=True
            ),
            apply_probability=0.3
        ))

        # Sharpening
        common_transforms.append(RandomTransform(
            SharpeningTransform(
                strength=(0.1, 1.5),
                p_same_for_each_channel=0.5,
                p_per_channel=0.5,
                p_clamp_intensities=0.5,
            ),
            apply_probability=0.2
        ))

    # Contrast OR Multiplicative Brightness (one of)
    common_transforms.append(
        OneOfTransform([
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
        ])
    )

    # Additive brightness
    common_transforms.append(RandomTransform(
        BrightnessAdditiveTransform(
            mu=0,
            sigma=0.5,
            synchronize_channels=False,
            p_per_channel=0.5,
        ),
        apply_probability=0.1
    ))

    if not only_spatial_and_intensity:
        # All-axis low resolution
        common_transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.25, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            ),
            apply_probability=0.4
        ))

    # Gamma (with inversion)
    common_transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ),
        apply_probability=0.2
    ))

    # Gamma (without inversion)
    common_transforms.append(RandomTransform(
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ),
        apply_probability=0.4
    ))

    # Invert image
    common_transforms.append(RandomTransform(
        InvertImageTransform(
            p_invert_image=1,
            p_synchronize_channels=0.5,
            p_per_channel=0.5,
        ),
        apply_probability=0.2
    ))

    # =========================================================================
    # DIMENSION-SPECIFIC ASSEMBLY
    # =========================================================================
    if dimension == 2:
        if blank_rectangle is not None:
            transforms.append(blank_rectangle)
        transforms.extend(common_transforms)
    else:
        # 3D-specific transforms
        # Transpose only between axes with equal dimensions (need at least 2 axes)
        if not no_spatial and len(transpose_allowed_axes) >= 2:
            transforms.append(RandomTransform(
                TransposeAxesTransform(allowed_axes=transpose_allowed_axes),
                apply_probability=0.2
            ))

        if blank_rectangle is not None:
            transforms.append(blank_rectangle)

        if not only_spatial_and_intensity:
            transforms.append(RandomTransform(
                SmearTransform(
                    shift=(5, 0),
                    alpha=0.2,
                    num_prev_slices=3,
                    smear_axis=3
                ),
                apply_probability=0.3
            ))

        transforms.append(RandomTransform(
            InhomogeneousSliceIlluminationTransform(
                num_defects=(2, 5),
                defect_width=(25, 50),
                mult_brightness_reduction_at_defect=(0.3, 1.5),
                base_p=(0.2, 0.4),
                base_red=(0.5, 0.9),
                p_per_sample=1.0,
                per_channel=True,
                p_per_channel=0.5
            ),
            apply_probability=0.4
        ))

        # Local brightness gradient (3D only)
        transforms.append(RandomTransform(
            BrightnessGradientAdditiveTransform(
                scale=_local_transform_scale,
                loc=(-0.5, 1.5),
                max_strength=_brightness_gradient_max_strength,
                same_for_all_channels=False,
                mean_centered=True,
                clip_intensities=False,
                p_per_channel=0.5,
            ),
            apply_probability=0.2
        ))

        # Local gamma (3D only)
        transforms.append(RandomTransform(
            LocalGammaTransform(
                scale=_local_transform_scale,
                loc=(-0.5, 1.5),
                gamma=_local_gamma_gamma,
                same_for_all_channels=False,
                p_per_channel=0.5,
            ),
            apply_probability=0.2
        ))

        transforms.extend(common_transforms)

    # =========================================================================
    # SKELETON TRANSFORMS (if needed)
    # =========================================================================
    if skeleton_targets:
        from vesuvius.models.augmentation.transforms.utils.skeleton_transform import MedialSurfaceTransform
        transforms.append(
            MedialSurfaceTransform(
                do_tube=False,
                target_keys=skeleton_targets,
                ignore_values=skeleton_ignore_values or None,
            )
        )
        print(f"Added MedialSurfaceTransform to training pipeline for targets: {', '.join(skeleton_targets)}")

    if no_spatial:
        print("Spatial transformations disabled (no_spatial=True)")

    if no_scaling:
        print("Scaling augmentation disabled (no_scaling=True)")

    if only_spatial_and_intensity:
        print("Only spatial and intensity augmentations enabled (only_spatial_and_intensity=True)")

    return ComposeTransforms(transforms)
