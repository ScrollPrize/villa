"""
Training augmentation pipeline configuration.

The standard training augmentation pipeline used for volumetric data in the
Vesuvius project, as a vc_reader transform pipeline.
"""

import itertools
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from vc_reader import transforms as A

ENABLE_BLANK_RECTANGLE = False

_draws = itertools.count()


def sample_seed(index: int) -> int:
    """A seed for one draw of sample ``index``: differs per worker, epoch and draw."""
    return A.seed(torch.initial_seed(), int(index), next(_draws))


def _targets(label_names: Iterable[str], extra_targets: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    targets = {"image": A.Image(), "padding_mask": A.ValidMask()}
    targets.update({name: A.Label() for name in label_names})
    targets.update(extra_targets or {})
    return targets


def _skeleton(skeleton_targets, skeleton_ignore_values):
    return A.MedialSurface(
        on=list(skeleton_targets), tube=False, ignore=dict(skeleton_ignore_values or {})
    )


def create_training_transforms(
    patch_size: Tuple[int, ...],
    label_names: Iterable[str] = (),
    no_spatial: bool = False,
    no_scaling: bool = False,
    only_spatial_and_intensity: bool = False,
    allowed_rotation_axes: Optional[List[int]] = None,
    skeleton_targets: Optional[List[str]] = None,
    skeleton_ignore_values: Optional[Dict[str, int]] = None,
    backend: str = "cpu",
    extra_targets: Optional[Dict[str, object]] = None,
) -> A.Pipeline:
    """
    Create the training augmentation pipeline.

    Parameters
    ----------
    patch_size : Tuple[int, ...]
        The patch size (z, y, x) for 3D or (y, x) for 2D.
    label_names : Iterable[str]
        The label arrays of a sample: they follow its geometry.
    no_spatial : bool
        If True, disable all spatial transforms (rotations, scaling, mirroring).
    no_scaling : bool
        Accepted for callers; the pipeline has no scaling.
    only_spatial_and_intensity : bool
        If True, only use spatial and basic intensity transforms (skip noise, blur, etc.).
    allowed_rotation_axes : Optional[List[int]]
        Accepted for callers; the pipeline has no free rotation.
    skeleton_targets : Optional[List[str]]
        List of target names that need skeleton/medial surface transform.
    skeleton_ignore_values : Optional[Dict[str, int]]
        Ignore values for skeleton targets.
    backend : str
        "cpu" for dataset workers, "cuda" for batches on the model device.
    extra_targets : Optional[Dict[str, object]]
        Other sample arrays and their vc_reader target roles.
    """
    dimension = len(patch_size)
    if dimension not in (2, 3):
        raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3.")
    has_equal_axes = dimension == 3 and len(set(patch_size)) < 3

    min_patch_dim = min(patch_size)
    local_scale = A.Uniform(min_patch_dim / 6, min_patch_dim / 2)
    local_loc = A.Uniform(-0.5, 1.5)

    transforms = []

    if not no_spatial:
        if has_equal_axes:
            transforms.append(A.Rotate90(turns=(1, 2, 3), p=0.5))
        elif dimension == 2:
            transforms.append(A.Flip(axes="yx", p_axis=0.5))

    blank_rectangle = None
    if ENABLE_BLANK_RECTANGLE and not only_spatial_and_intensity:
        blank_rectangle = A.BlankRectangle(
            size=[A.Uniform(max(1, size // 6), size // 3) for size in patch_size],
            value="mean",
            count=A.Uniform(1, 5),
            p=0.1 * 0.4,
            p_channel=0.5,
        )

    common = []
    if not only_spatial_and_intensity:
        common.append(A.GaussianBlur(sigma=A.Uniform(0.3, 1.5), shared_axes=False, p=0.3, p_channel=0.5))
        common.append(A.GaussianNoise(std=A.Uniform(0, 0.2), shared=True, shared_noise=True, p=0.3, p_channel=0.5))
        common.append(A.Sharpen(strength=A.Uniform(0.1, 1.5), p_clamp=0.5, p=0.2, p_channel=0.5))
    common.append(
        A.OneOf(
            transforms=[
                A.Contrast(factor=A.Uniform(0.75, 1.25), preserve_range=True, p=0.3, p_channel=0.5),
                A.BrightnessMultiply(factor=A.Uniform(0.75, 1.25), p=0.3, p_channel=0.5),
            ]
        )
    )
    common.append(A.BrightnessAdd(shift=A.Normal(0, 0.5), p=0.1, p_channel=0.5))
    if not only_spatial_and_intensity:
        common.append(A.LowResolution(scale=A.Uniform(0.25, 1), shared_axes=True, p=0.4, p_channel=0.5))
    for p_invert, p in ((1.0, 0.2), (0.0, 0.4)):
        # Half the draws below 1, half above, whatever the range's midpoint.
        common.append(
            A.OneOf(
                transforms=[
                    A.Gamma(gamma=A.Uniform(0.7, 1), p_invert=p_invert, p_retain_stats=1),
                    A.Gamma(gamma=A.Uniform(1, 1.5), p_invert=p_invert, p_retain_stats=1),
                ],
                p=p,
            )
        )
    common.append(A.OneOf(transforms=[A.Invert(shared=True), A.Invert(p_channel=0.5)], p=0.2))

    if dimension == 3 and not no_spatial and has_equal_axes:
        transforms.append(A.Transpose(p=0.2))
    if blank_rectangle is not None:
        transforms.append(blank_rectangle)
    if dimension == 3:
        if not only_spatial_and_intensity:
            transforms.append(A.Smear(axis="x", shift=[5, 0], alpha=0.2, previous=3, p=0.3))
        transforms.append(
            A.SliceIllumination(
                axis="z",
                defects=A.Uniform(2, 5),
                width=A.Uniform(25, 50),
                reduction=A.Uniform(0.3, 1.5),
                base_p=A.Uniform(0.2, 0.4),
                base_reduction=A.Uniform(0.5, 0.9),
                p=0.4,
                p_channel=0.5,
            )
        )
        transforms.append(
            A.BrightnessGradient(
                scale=local_scale, loc=local_loc, strength=0.5, mean_centered=True, clip=False, p=0.2, p_channel=0.5
            )
        )
        transforms.append(A.LocalGamma(scale=local_scale, loc=local_loc, gamma=A.Uniform(0.5, 1.5), p=0.2, p_channel=0.5))
    transforms.extend(common)

    if skeleton_targets:
        transforms.append(_skeleton(skeleton_targets, skeleton_ignore_values))
        print(f"Added MedialSurface to training pipeline for targets: {', '.join(skeleton_targets)}")

    if no_spatial:
        print("Spatial transformations disabled (no_spatial=True)")
    if only_spatial_and_intensity:
        print("Only spatial and intensity augmentations enabled (only_spatial_and_intensity=True)")

    return A.Pipeline(transforms, targets=_targets(label_names, extra_targets), backend=backend)


def create_validation_transforms(
    label_names: Iterable[str] = (),
    skeleton_targets: Optional[List[str]] = None,
    skeleton_ignore_values: Optional[Dict[str, int]] = None,
) -> Optional[A.Pipeline]:
    """Skeleton generation for validation when needed, without augmentation; None otherwise."""
    if not skeleton_targets:
        return None
    return A.Pipeline([_skeleton(skeleton_targets, skeleton_ignore_values)], targets=_targets(label_names))
