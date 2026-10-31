"""Normalization schemes as ``vc_reader.transforms.Normalize``, with statistics over the whole input."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np
import vc_reader.transforms as A

__all__ = ["normalize_minmax", "normalize_zscore", "normalize_ct", "normalize_robust", "NORMALIZATION_FUNCTIONS", "apply"]

EPS = 1e-8


@lru_cache(maxsize=None)
def _pipeline(mode: str, options: tuple) -> A.Pipeline:
    return A.Pipeline([A.Normalize(mode, **dict(options))], backend="cpu")


def apply(image: np.ndarray, mode: str, **options) -> np.ndarray:
    """``A.Normalize(mode, **options)`` of any array as one channel, as float32, in its shape."""
    values = np.asarray(image, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    flat = values.reshape(1, 1, -1)
    return _pipeline(mode, tuple(sorted(options.items())))({"image": flat}, seed=0)["image"].reshape(values.shape)


def normalize_minmax(image: np.ndarray) -> np.ndarray:
    return apply(image, "minmax")


def normalize_zscore(image: np.ndarray) -> np.ndarray:
    return apply(image, "zscore", nonzero=False, eps=EPS)


def normalize_ct(image: np.ndarray, *, intensity_properties: Mapping[str, float] | None) -> np.ndarray:
    """Clip to global percentiles, then z-score with the global mean and standard deviation."""
    if intensity_properties is None:
        raise ValueError("normalize_ct requires intensity_properties")
    required_keys = ("mean", "std", "percentile_00_5", "percentile_99_5")
    missing = [key for key in required_keys if key not in intensity_properties]
    if missing:
        raise ValueError(
            f"normalize_ct missing intensity properties: {missing}. "
            f"Provided keys: {list(intensity_properties.keys())}"
        )
    return apply(
        image, "fixed", center=float(intensity_properties["mean"]), scale=max(float(intensity_properties["std"]), EPS),
        bounds=(float(intensity_properties["percentile_00_5"]), float(intensity_properties["percentile_99_5"])))


def normalize_robust(image: np.ndarray, *, percentile_lower: float = 1.0, percentile_upper: float = 99.0,
                     clip_values: bool = True) -> np.ndarray:
    """Median and MAD; the fallback scales are in ``A.Normalize``."""
    return apply(image, "robust", lower=float(percentile_lower), upper=float(percentile_upper), clip=bool(clip_values))


def _normalize_identity(image: np.ndarray, **_: object) -> np.ndarray:
    return np.array(image, dtype=np.float32)


NORMALIZATION_FUNCTIONS = {
    "zscore": normalize_zscore,
    "ct": normalize_ct,
    "rescale_to_01": normalize_minmax,
    "minmax": normalize_minmax,
    "robust": normalize_robust,
    "none": _normalize_identity,
}
