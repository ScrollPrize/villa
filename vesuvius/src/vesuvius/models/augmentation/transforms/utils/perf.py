from __future__ import annotations

from typing import List

from vesuvius.models.augmentation.transforms.base.basic_transform import BasicTransform
from vesuvius.models.augmentation.transforms.utils.compose import ComposeTransforms
from vesuvius.models.augmentation.transforms.utils.oneoftransform import OneOfTransform
from vesuvius.models.augmentation.transforms.utils.random import RandomTransform


def collect_augmentation_names(transform: BasicTransform | None) -> List[str]:
    if transform is None:
        return []

    names: List[str] = []

    def walk(t: BasicTransform) -> None:
        if t is None:
            return
        if isinstance(t, ComposeTransforms):
            for child in t.transforms:
                walk(child)
            return
        if isinstance(t, RandomTransform):
            walk(t.transform)
            return
        if isinstance(t, OneOfTransform):
            for child in t.list_of_transforms:
                walk(child)
            return

        perf_name = getattr(t, "_perf_name", type(t).__name__)
        names.append(perf_name)

    walk(transform)

    seen = set()
    unique_names: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names
