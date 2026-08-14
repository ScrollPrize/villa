"""Ink image normalization and flat-window masking semantics."""

from __future__ import annotations

import numpy as np

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.image_proc.intensity.normalization import normalize_robust


def normalize_image(
    image: np.ndarray, config: NormalizationConfig
) -> np.ndarray:
    """Normalize one crop as float32 using the selected mode."""
    image_f32 = np.asarray(image).astype(np.float32, copy=False)
    if image_f32.size == 0:
        return image_f32
    if config.mode == "none":
        return image_f32
    if config.mode == "divide":
        image_f32 /= config.divisor
        return image_f32
    if config.mode == "clip_zscore":
        np.clip(image_f32, config.clip_min, config.clip_max, out=image_f32)
        image_f32 -= config.mean
        image_f32 /= config.std
        np.nan_to_num(
            image_f32, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        return image_f32
    if config.mode == "robust_mad":
        return normalize_robust(
            image_f32,
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
        )

    lower = 0.0 if config.mode == "minmax" else config.percentile_lower
    upper = 100.0 if config.mode == "minmax" else config.percentile_upper
    lower_value = float(np.percentile(image_f32, lower))
    upper_value = float(np.percentile(image_f32, upper))
    np.clip(image_f32, lower_value, upper_value, out=image_f32)
    if config.mode == "robust_percentile_span":
        median = float(np.median(image_f32))
        scale = 0.5 * (upper_value - lower_value)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        image_f32 -= median
    else:
        scale = upper_value - lower_value
        if not np.isfinite(scale) or scale < 1e-6:
            image_f32.fill(0.0)
            return image_f32
        image_f32 -= lower_value
    image_f32 /= scale
    np.nan_to_num(
        image_f32, copy=False, nan=0.0, posinf=0.0, neginf=0.0
    )
    return image_f32


def exclude_validation_voxels(
    supervision: np.ndarray,
    validation: np.ndarray | None,
) -> np.ndarray:
    """Remove held-out voxels from training supervision without mutating input."""
    if validation is None:
        return supervision
    supervision = np.asarray(supervision)
    validation = np.asarray(validation)
    if supervision.shape != validation.shape:
        raise ValueError(
            "supervision and validation must have matching shapes, "
            f"got {supervision.shape!r} and {validation.shape!r}"
        )
    if supervision.size == 0 or not np.any(validation):
        return supervision
    masked = np.array(supervision, copy=True)
    masked[validation > 0] = 0
    return masked
