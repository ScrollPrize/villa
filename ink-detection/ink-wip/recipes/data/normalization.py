from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial

import albumentations as A
import numpy as np

from ink.core.types import DataBundle


def _stats_from_bundle(data: DataBundle):
    stats = data.extras.get("normalization_stats")
    if stats is None:
        return None
    return dict(stats)


def _apply_fold_foreground_clip_zscore(image, *, normalization, **kwargs):
    del kwargs
    stats = normalization.stats
    if stats is None:
        raise ValueError("normalization stats must be set before using FoldForegroundClipZScoreNormalization")
    image = image.astype(np.float32, copy=True)
    np.clip(image, float(stats["percentile_00_5"]), float(stats["percentile_99_5"]), out=image)
    image -= float(stats["mean"])
    image /= float(stats["std"])
    return image


def _apply_fold_foreground_clip_robust_zscore(image, *, normalization, **kwargs):
    del kwargs
    stats = normalization.stats
    if stats is None:
        raise ValueError("normalization stats must be set before using FoldForegroundClipRobustZScoreNormalization")
    image = image.astype(np.float32, copy=True)
    np.clip(image, float(stats["percentile_00_5"]), float(stats["percentile_99_5"]), out=image)
    image -= float(stats["median"])
    image /= float(stats["robust_scale"])
    return image


@dataclass(frozen=True)
class ClipMaxDiv255Normalization:
    def build(self, *, data: DataBundle):
        del data
        return self


@dataclass(frozen=True)
class FoldForegroundClipZScoreNormalization:
    stats: dict[str, float] | None = None

    def build(self, *, data: DataBundle):
        if self.stats is not None:
            return self
        return replace(self, stats=_stats_from_bundle(data))


@dataclass(frozen=True)
class FoldForegroundClipRobustZScoreNormalization:
    stats: dict[str, float] | None = None

    def build(self, *, data: DataBundle):
        if self.stats is not None:
            return self
        return replace(self, stats=_stats_from_bundle(data))


def build_normalization_transform(normalization, *, in_channels: int):
    if normalization is None or isinstance(normalization, ClipMaxDiv255Normalization):
        return A.Normalize(mean=[0] * int(in_channels), std=[1] * int(in_channels))
    if isinstance(normalization, FoldForegroundClipZScoreNormalization):
        return A.Lambda(image=partial(_apply_fold_foreground_clip_zscore, normalization=normalization), p=1.0)
    if isinstance(normalization, FoldForegroundClipRobustZScoreNormalization):
        return A.Lambda(image=partial(_apply_fold_foreground_clip_robust_zscore, normalization=normalization), p=1.0)
    raise TypeError(f"unsupported normalization recipe: {normalization!r}")


__all__ = [
    "ClipMaxDiv255Normalization",
    "FoldForegroundClipRobustZScoreNormalization",
    "FoldForegroundClipZScoreNormalization",
    "build_normalization_transform",
]
