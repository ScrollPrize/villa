"""Normalization schemes named in dataset configs, for images of any shape."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np

from vesuvius.image_proc.intensity.normalization import normalize_ct, normalize_minmax, normalize_robust, normalize_zscore


def get_normalization(scheme: str, intensityproperties: dict | None = None) -> Callable[[np.ndarray], np.ndarray] | None:
    """The function normalizing an image under a scheme: ``zscore``, ``ct``, ``rescale_to_01``,
    ``minmax``, ``robust`` or ``none`` (None).
    """
    if scheme == "none":
        return None
    if scheme == "zscore":
        return normalize_zscore
    if scheme == "ct":
        required = ("mean", "std", "percentile_00_5", "percentile_99_5")
        assert intensityproperties and all(key in intensityproperties for key in required), \
            "CT normalization requires 'mean', 'std', 'percentile_00_5', and 'percentile_99_5' in intensity properties"
        return partial(normalize_ct, intensity_properties=intensityproperties)
    if scheme in ("rescale_to_01", "minmax"):
        return normalize_minmax
    if scheme == "robust":
        return normalize_robust
    raise ValueError(f"Unknown normalization scheme: {scheme}. "
                     f"Available schemes: ['zscore', 'ct', 'rescale_to_01', 'minmax', 'robust', 'none']")
