from vc_reader import transforms as A


def get_training_augmentations(crop_size, allow_transposes, allow_mirroring, only_spatial_and_intensity):
    """Training transforms for cubic crops of ``crop_size``. The sample's dist_map stays out: it does not follow geometry."""
    transforms = []

    if allow_transposes:  # diverges from vesuvius!
        transforms.append(A.Transpose(p=0.5))  # diverges from vesuvius!

    if allow_mirroring:
        transforms.append(A.Flip(axes="zyx", p_axis=0.5, p=0.5))

    # Always add intensity transforms; half the gamma draws below 1, half above
    transforms.append(A.OneOf(transforms=[
        A.BrightnessMultiply(factor=A.Uniform(0.75, 1.25)),
        A.Contrast(factor=A.Uniform(0.50, 1.50), preserve_range=True),
        A.OneOf(transforms=[
            A.Gamma(gamma=A.Uniform(0.7, 1), p_retain_stats=1),
            A.Gamma(gamma=A.Uniform(1, 1.5), p_retain_stats=1),
        ]),
        A.OneOf(transforms=[
            A.Gamma(gamma=A.Uniform(0.7, 1), p_invert=1, p_retain_stats=1),
            A.Gamma(gamma=A.Uniform(1, 1.5), p_invert=1, p_retain_stats=1),
        ]),
        A.SliceIllumination(
            axis="z",
            defects=A.Uniform(2, 5),
            width=A.Uniform(5, 20),
            reduction=A.Uniform(0.3, 0.7),
            base_p=A.Uniform(0.2, 0.4),
            base_reduction=A.Uniform(0.5, 0.9),
            p_channel=0.5,
        ),
    ], p=0.2))

    # Only add noise/blur/rectangle transforms if not in only_spatial_and_intensity mode
    if not only_spatial_and_intensity:
        transforms.append(A.OneOf(transforms=[
            A.GaussianNoise(std=A.Uniform(0, 0.20), shared=True, shared_noise=True),
            A.RicianNoise(std=A.Uniform(0, 0.1 ** 0.5)),
            A.Smear(axis="z", shift=[10, 0], alpha=0.5, previous=1),
        ], p=0.2))
        transforms.append(A.OneOf(transforms=[
            A.GaussianBlur(sigma=A.Uniform(0.5, 1.0), shared=True, shared_axes=False),
            A.LowResolution(scale=A.Uniform(0.3, 1), shared_axes=True, p_channel=0.5),
        ], p=0.2))
        transforms.append(A.BlankRectangle(
            size=[A.Uniform(max(1, crop_size // 10), crop_size // 3)] * 3,
            value="mean",
            count=A.Uniform(1, 3),
            p=0.3 * 0.4,
            p_channel=0.5,
        ))

    return A.Pipeline(transforms, targets={
        "image": A.Image(),
        "regression_target": A.Label(),
        "segmentation": A.Label(),
        "normals": A.NormalField(components="zyx"),
        "normals_mask": A.Label(),
    })
