from __future__ import annotations

from itertools import combinations
from typing import Sequence

from vesuvius.models.augmentation.pipelines.training_transforms import create_training_transforms
from vesuvius.models.augmentation.transforms.noise.gaussian_blur import GaussianBlurTransform
from vesuvius.models.augmentation.transforms.spatial.mirroring import MirrorTransform
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform


def compute_equal_length_mirror_axes(patch_size: Sequence[int]) -> tuple[int, ...]:
    dims = tuple(int(v) for v in patch_size)
    if len(dims) != 3:
        raise ValueError(f"Expected a 3D patch size, got {dims!r}")

    allowed_axes: set[int] = set()
    for left in range(len(dims)):
        for right in range(left + 1, len(dims)):
            if dims[left] == dims[right]:
                allowed_axes.add(left)
                allowed_axes.add(right)
    return tuple(sorted(allowed_axes))


def iter_mirror_axes(allowed_axes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    axes = tuple(int(axis) for axis in allowed_axes)
    variants: list[tuple[int, ...]] = []
    for count in range(len(axes) + 1):
        variants.extend(tuple(combo) for combo in combinations(axes, count))
    return tuple(variants)


def create_tifxyz_training_transforms(patch_size: Sequence[int]):
    transforms = create_training_transforms(tuple(int(v) for v in patch_size))
    allowed_axes = compute_equal_length_mirror_axes(patch_size)

    retained = []
    insert_at = 0
    for transform in transforms.transforms:
        inner = getattr(transform, "transform", None)
        if isinstance(inner, GaussianBlurTransform):
            # Avoid the fft_conv_pytorch backend here: its internal list-based
            # multidimensional indexing emits a PyTorch 2.9 deprecation warning.
            inner.benchmark = False
        if isinstance(inner, MirrorTransform):
            continue
        retained.append(transform)
        if type(inner).__name__ in {"Rot90Transform", "TransposeAxesTransform"}:
            insert_at = len(retained)

    if allowed_axes:
        retained.insert(
            insert_at,
            RandomTransform(
                MirrorTransform(allowed_axes=allowed_axes),
                apply_probability=1.0,
            ),
        )

    transforms.transforms = retained
    return transforms
