"""The augmentation of one training view, as vc_reader transform pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from vc_reader import transforms as A


@dataclass(frozen=True)
class ViewAugmentation:
    """Two pipelines, called with one seed.

    A pipeline draws each transform's randomness from the seed and the transform's position, not
    from what came before it, so ``geometry`` (the spatial transforms at the same positions as in
    ``full``) moves a view the way ``full`` did, at any resolution with the same axes of equal size.
    """

    full: A.Pipeline
    geometry: A.Pipeline


def create_training_transforms(
    patch_size: tuple[int, int, int],
    gaussian_blur: bool = True,
    dino_only: bool = False,
) -> ViewAugmentation:
    """The pipelines for a (z, y, x) view.

    ``gaussian_blur=False`` leaves out the blur. ``dino_only`` keeps only flips, right-angle
    rotations, axis transposes and brightness/contrast jitter (the dataset applies the random
    resized crops).
    """
    if len(patch_size) != 3:
        raise ValueError(f"Invalid patch size dimension: {len(patch_size)}. Expected 3.")
    min_patch_dim = min(patch_size)
    local_scale = A.Uniform(min_patch_dim / 6, min_patch_dim / 2)
    local_loc = A.Uniform(-0.5, 1.5)

    # Simulated scanner artifacts act along fixed tensor axes. They go before the rotations, flips
    # and transposes so they follow the volume's axes (as real scan artifacts do) and are reoriented
    # with the content, instead of always lying along the same axes of the augmented view.
    scanner_artifacts = [
        A.Smear(axis="x", shift=[5, 0], alpha=0.2, previous=3, p=0.3),
        A.SliceIllumination(
            axis="z",
            defects=A.Uniform(2, 5),
            width=A.Uniform(25, 50),
            reduction=A.Uniform(0.3, 1.5),
            base_p=A.Uniform(0.2, 0.4),
            base_reduction=A.Uniform(0.5, 0.9),
            p=0.4,
            p_channel=0.5,
        ),
    ]
    # Rotations and transposes use the planes and axes of equal size.
    spatial = [
        A.Rotate90(turns=(1, 2, 3), p=0.5),
        A.Flip(axes="zyx", p_axis=0.5, p=0.5),
        A.Transpose(p=0.2),
    ]
    local = [
        A.BrightnessGradient(
            scale=local_scale, loc=local_loc, strength=0.5, mean_centered=True, clip=False, p=0.2, p_channel=0.5
        ),
        A.LocalGamma(scale=local_scale, loc=local_loc, gamma=A.Uniform(0.5, 1.5), p=0.2, p_channel=0.5),
    ]
    blur = [A.GaussianBlur(sigma=A.Uniform(0.3, 1.5), shared_axes=False, p=0.3, p_channel=0.5)] if gaussian_blur else []
    noise = [
        A.GaussianNoise(std=A.Uniform(0, 0.2), shared=True, shared_noise=True, p=0.3, p_channel=0.5),
        A.Sharpen(strength=A.Uniform(0.1, 1.5), p_clamp=0.5, p=0.2, p_channel=0.5),
    ]
    jitter = [
        A.OneOf(
            transforms=[
                A.Contrast(factor=A.Uniform(0.75, 1.25), preserve_range=True, p=0.3, p_channel=0.5),
                A.BrightnessMultiply(factor=A.Uniform(0.75, 1.25), p=0.3, p_channel=0.5),
            ]
        ),
        A.BrightnessAdd(shift=A.Normal(0, 0.5), p=0.1, p_channel=0.5),
    ]
    low_resolution = [A.LowResolution(scale=A.Uniform(0.25, 1), shared_axes=True, p=0.4, p_channel=0.5)]
    # Half the gamma draws below 1, half above, whatever the range's midpoint.
    gamma = [
        A.OneOf(
            transforms=[
                A.Gamma(gamma=A.Uniform(0.7, 1), p_invert=p_invert, p_retain_stats=1),
                A.Gamma(gamma=A.Uniform(1, 1.5), p_invert=p_invert, p_retain_stats=1),
            ],
            p=p,
        )
        for p_invert, p in ((1.0, 0.2), (0.0, 0.4))
    ]
    invert = [A.OneOf(transforms=[A.Invert(shared=True), A.Invert(p_channel=0.5)], p=0.2)]

    if dino_only:
        before, after = [], jitter
    else:
        before = scanner_artifacts
        after = local + blur + noise + jitter + low_resolution + gamma + invert

    image = A.Image(valid=None)
    return ViewAugmentation(
        full=A.Pipeline(
            before + spatial + after,
            targets={"image": image, "keypoints": A.Points(order="zyx", outside="keep")},
        ),
        geometry=A.Pipeline([A.Invert(p=0) for _ in before] + spatial, targets={"image": image}),
    )
