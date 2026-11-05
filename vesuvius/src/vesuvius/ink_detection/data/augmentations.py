"""Ink training augmentation presets and native-crop translation."""

from __future__ import annotations

import numpy as np

from vc_reader import transforms as A


NATIVE_CROP_TRANSLATION_PROBABILITY = 0.30
NATIVE_CROP_TRANSLATION_MIN_VOXELS = 10
NATIVE_CROP_TRANSLATION_MAX_VOXELS = 40
NATIVE_CROP_TRANSLATION_KEEP_FRACTION = 1.0 / 3.0
NATIVE_CROP_TRANSLATION_MAX_AXES = 2
NATIVE_CROP_TRANSLATION_MAX_ATTEMPTS = 24

_TARGETS = {
    "image": A.Image(),
    "inklabels": A.Label(),
    "supervision_mask": A.Label(),
    "surface_mask": A.ScalarField(),
    "image_mask_for_label": A.Label(),
}


def _rotation_planes(
    patch_size: tuple[int, ...], requested: tuple[int, ...] | None
) -> list[str]:
    if len(patch_size) == 2:
        return ["yx"] if patch_size[0] == patch_size[1] else []
    planes = {0: (1, 2, "yx"), 1: (0, 2, "zx"), 2: (0, 1, "zy")}
    chosen = []
    for rotation_axis in sorted({0} if requested is None else set(requested)):
        plane = planes.get(rotation_axis)
        if plane is not None and patch_size[plane[0]] == patch_size[plane[1]]:
            chosen.append(plane[2])
    return chosen


def _spatial(patch_size, rotation_axes) -> list:
    """Label-safe flips and rotations."""
    if len(patch_size) not in (2, 3):
        raise ValueError(f"Invalid patch size dimension: {len(patch_size)}. Expected 2 or 3")
    transforms = [A.Flip(axes="yx", p_axis=0.5)]
    planes = _rotation_planes(patch_size, rotation_axes)
    if planes:
        transforms.append(A.Rotate90(planes=planes, turns=(1, 2, 3), p=0.3))
    return transforms


def _no_clip_intensity() -> list:
    return [
        A.Invert(shared=True, p=0.2),
        A.BrightnessAdd(shift=A.Uniform(-0.15, 0.15), shared=True, p=0.3),
        A.GaussianNoise(std=A.Uniform(0.005, 0.03), shared=True, shared_noise=True, p=0.25),
    ]


def _default(patch_size) -> tuple[list, list]:
    """The default ink training transforms: those changing geometry, then the others."""
    dimension = len(patch_size)
    if dimension not in (2, 3):
        raise ValueError(f"Invalid patch size dimension: {dimension}. Expected 2 or 3")
    geometric = []
    if dimension == 2:
        geometric.append(A.Flip(axes="yx", p_axis=0.5))
    elif len(set(patch_size)) < 3:
        geometric.append(A.Rotate90(turns=(1, 2, 3), p=0.5))
        geometric.append(A.Transpose(p=0.2))
    photometric = [
        A.BlankRectangle(
            size=[A.Uniform(max(1, size // 6), size // 3) for size in patch_size],
            value="mean",
            count=A.Uniform(1, 5),
            p=0.1 * 0.4,
            p_channel=0.5,
        )
    ]
    if dimension == 3:
        photometric += [
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
            A.BrightnessGradient(
                scale=A.Uniform(min(patch_size) / 6, min(patch_size) / 2),
                loc=A.Uniform(-0.5, 1.5),
                strength=0.5,
                mean_centered=True,
                clip=False,
                p=0.2,
                p_channel=0.5,
            ),
        ]
    photometric += [
        A.GaussianBlur(sigma=A.Uniform(0.3, 1.5), shared_axes=False, p=0.3, p_channel=0.5),
        A.GaussianNoise(std=A.Uniform(0, 0.2), shared=True, shared_noise=True, p=0.3, p_channel=0.5),
        A.Sharpen(strength=A.Uniform(0.1, 1.5), p_clamp=0.5, p=0.2, p_channel=0.5),
        A.OneOf(transforms=[
            A.Contrast(factor=A.Uniform(0.75, 1.25), preserve_range=True, p=0.3, p_channel=0.5),
            A.BrightnessMultiply(factor=A.Uniform(0.75, 1.25), p=0.3, p_channel=0.5),
        ]),
        A.BrightnessAdd(shift=A.Normal(0, 0.5), p=0.1, p_channel=0.5),
        A.LowResolution(scale=A.Uniform(0.25, 1), shared_axes=True, p=0.4, p_channel=0.5),
    ]
    # Half the gamma draws below 1, half above.
    for p_invert, p in ((1.0, 0.2), (0.0, 0.4)):
        photometric.append(A.OneOf(transforms=[
            A.Gamma(gamma=A.Uniform(0.7, 1), p_invert=p_invert, p_retain_stats=1),
            A.Gamma(gamma=A.Uniform(1, 1.5), p_invert=p_invert, p_retain_stats=1),
        ], p=p))
    photometric.append(A.OneOf(transforms=[A.Invert(shared=True), A.Invert(p_channel=0.5)], p=0.2))
    return geometric, photometric


def _preset(preset: str, patch_size, rotation_axes) -> tuple[list, list] | None:
    if preset == "none":
        return None
    if preset == "spatial_only":
        return _spatial(patch_size, rotation_axes), []
    if preset == "spatial_intensity_no_clip":
        return _spatial(patch_size, rotation_axes), _no_clip_intensity()
    if preset == "default":
        return _default(patch_size)
    raise ValueError(f"unknown augmentation preset {preset!r}")


def build_augmentations(
    preset: str,
    patch_size: tuple[int, ...],
    *,
    rotation_axes: tuple[int, ...] | None,
) -> A.Pipeline | None:
    """Build one configured preset, returning no pipeline for `none`."""
    parts = _preset(preset, patch_size, rotation_axes)
    return None if parts is None else A.Pipeline(parts[0] + parts[1], targets=_TARGETS)


def build_split_augmentations(
    preset: str,
    patch_size: tuple[int, ...],
    *,
    rotation_axes: tuple[int, ...] | None,
) -> tuple[A.Pipeline, A.Pipeline] | None:
    """One preset split around the clean pseudo-label image boundary: geometry, then intensity."""
    parts = _preset(preset, patch_size, rotation_axes)
    return None if parts is None else (A.Pipeline(parts[0], targets=_TARGETS), A.Pipeline(parts[1], targets=_TARGETS))


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
