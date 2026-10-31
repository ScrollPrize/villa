"""Ink image normalization and flat-window masking semantics."""

from __future__ import annotations

import numpy as np

from vesuvius.ink_detection.config import NormalizationConfig
from vesuvius.image_proc.intensity.normalization import apply, normalize_robust


def normalize_image(
    image: np.ndarray, config: NormalizationConfig
) -> np.ndarray:
    """Normalize one crop as float32 using the selected mode."""
    image_f32 = np.asarray(image).astype(np.float32, copy=False)
    if image_f32.size == 0 or config.mode == "none":
        return image_f32
    if config.mode == "divide":
        return apply(image_f32, "fixed", scale=float(config.divisor))
    if config.mode == "clip_zscore":
        return apply(image_f32, "fixed", center=float(config.mean), scale=float(config.std),
                     bounds=(float(config.clip_min), float(config.clip_max)))
    if config.mode == "robust_mad":
        return normalize_robust(
            image_f32,
            percentile_lower=config.percentile_lower,
            percentile_upper=config.percentile_upper,
        )
    if config.mode == "minmax":
        return apply(image_f32, "percentile", lower=0.0, upper=100.0)
    return apply(image_f32, "percentile", lower=float(config.percentile_lower), upper=float(config.percentile_upper),
                 median=config.mode == "robust_percentile_span")


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
