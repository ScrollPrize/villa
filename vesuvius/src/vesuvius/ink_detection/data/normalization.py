"""Ink image normalization and flat-window masking semantics."""

from __future__ import annotations

import numpy as np

from vesuvius.ink_detection.config import NormalizationConfig
import vc_reader.transforms as A
from functools import lru_cache


@lru_cache(maxsize=32)
def _pipeline(mode, options):
    return A.Pipeline([A.Normalize(mode, **dict(options))], backend="cpu")


def normalize_image(
    image: np.ndarray, config: NormalizationConfig
) -> np.ndarray:
    """Normalize one crop as float32 using the selected mode."""
    image_f32 = np.asarray(image).astype(np.float32, copy=False)
    if image_f32.size == 0 or config.mode == "none":
        return image_f32
    if config.mode == "divide":
        mode, options = "fixed", dict(scale=float(config.divisor))
    elif config.mode == "clip_zscore":
        mode, options = "fixed", dict(center=float(config.mean), scale=float(config.std),
                                     bounds=(float(config.clip_min), float(config.clip_max)))
    elif config.mode == "robust_mad":
        mode, options = "robust", dict(lower=float(config.percentile_lower), upper=float(config.percentile_upper))
    elif config.mode == "minmax":
        mode, options = "percentile", dict(lower=0.0, upper=100.0)
    else:
        mode, options = "percentile", dict(lower=float(config.percentile_lower), upper=float(config.percentile_upper),
                                           median=config.mode == "robust_percentile_span")
    pipeline = _pipeline(mode, tuple(sorted(options.items())))
    return pipeline({"image": image_f32.reshape(1, 1, -1)}, seed=0)["image"].reshape(image_f32.shape)


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
