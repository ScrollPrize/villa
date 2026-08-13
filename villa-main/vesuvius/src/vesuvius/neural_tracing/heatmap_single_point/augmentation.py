import numpy as np

from vesuvius.models.augmentation.transforms.spatial.transpose import TransposeAxesTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
from vesuvius.models.augmentation.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from vesuvius.models.augmentation.transforms.intensity.contrast import ContrastTransform, BGContrast
from vesuvius.models.augmentation.transforms.intensity.gamma import GammaTransform
from vesuvius.models.augmentation.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from vesuvius.models.augmentation.transforms.spatial.spatial import SpatialTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.noise.extranoisetransforms import BlankRectangleTransform, RicianNoiseTransform, SmearTransform
from vesuvius.models.augmentation.transforms.intensity.illumination import InhomogeneousSliceIlluminationTransform


def get_training_augmentations(crop_size, allow_transposes, allow_mirroring, only_spatial_and_intensity):
    """
    Create training transforms using custom batchgeneratorsv2.
    Returns None for validation (no augmentations).
    """
        
    patch_d, patch_h, patch_w = crop_size, crop_size, crop_size

    transforms = []

    if allow_transposes:  # diverges from vesuvius!
        # Only add transpose transform if all three dimensions are equal
        if patch_d == patch_h == patch_w:
            transforms.append(RandomTransform(
                TransposeAxesTransform(allowed_axes={0, 1, 2}, normal_keys={'normals'}),
                apply_probability=0.5  # diverges from vesuvius!
            ))

    if allow_mirroring:
        transforms.append(RandomTransform(
            MirrorTransform(allowed_axes=(0, 1, 2), normal_keys={'normals'}),
            apply_probability=0.5
        ))
    # Always add intensity transforms
    one_of_intensity = OneOfTransform([
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast((0.75, 1.25)),
            synchronize_channels=False,
            p_per_channel=1
        ),
        ContrastTransform(
            contrast_range=BGContrast((0.50, 1.50)),
            preserve_range=True,
            synchronize_channels=False,
            p_per_channel=1
        ),
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=0,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ),
        GammaTransform(
            gamma=BGContrast((0.7, 1.5)),
            p_invert_image=1,
            synchronize_channels=False,
            p_per_channel=1,
            p_retain_stats=1
        ),
        InhomogeneousSliceIlluminationTransform(
            num_defects=(2, 5),
            defect_width=(5, 20),
            mult_brightness_reduction_at_defect=(0.3, 0.7),
            base_p=(0.2, 0.4),
            base_red=(0.5, 0.9),
            p_per_sample=1.0,
            per_channel=True,
            p_per_channel=0.5
        )
    ])
    
    transforms.append(RandomTransform(
        one_of_intensity,
        apply_probability=0.2
    ))
    
    # Only add noise/blur/rectangle transforms if not in only_spatial_and_intensity mode
    if not only_spatial_and_intensity:
        one_of_noise = OneOfTransform([
            GaussianNoiseTransform(
                noise_variance=(0, 0.20),
                p_per_channel=1,
                synchronize_channels=True
            ),
            RicianNoiseTransform(
                noise_variance=(0, 0.1),
            ),
            SmearTransform(
                shift=(10, 0),
                alpha=0.5,
                num_prev_slices=1,
                smear_axis=1
            )
        ])
        one_of_blur = OneOfTransform([
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=True,
                synchronize_axes=False,
                p_per_channel=1.0
            ),
            SimulateLowResolutionTransform(
                scale=(0.3, 1.5),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5
            )
        ])

        transforms.append(RandomTransform(
            one_of_noise,
            apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            one_of_blur,
            apply_probability=0.2
        ))

        rectangle_sizes_3d = tuple(
            (max(1, crop_size // 10), crop_size // 3) for _ in range(3)
        )
        transforms.append(RandomTransform(
            BlankRectangleTransform(
                rectangle_size=rectangle_sizes_3d,
                rectangle_value=np.mean,
                num_rectangles=(1, 3),
                force_square=False,
                p_per_sample=0.4,
                p_per_channel=0.5
            ), apply_probability=0.3
        ))

    return ComposeTransforms(transforms)
