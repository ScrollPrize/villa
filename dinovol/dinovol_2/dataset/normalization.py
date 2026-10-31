"""Normalization schemes as ``vc_reader.transforms.Normalize``, with statistics over the whole input."""

from __future__ import annotations

import numpy as np
from vc_reader import transforms as A

EPS = 1e-8


class ImageNormalization:
    """One ``A.Normalize`` for images of any shape, as one channel."""

    def __init__(self, mode: str, **options):
        self._pipeline = A.Pipeline([A.Normalize(mode, **options)], targets={"image": A.Image(valid=None)})

    def run(self, image: np.ndarray) -> np.ndarray:
        values = np.asarray(image, dtype=np.float32)
        if values.size == 0:
            return values.copy()
        return self._pipeline({"image": values.reshape(1, 1, -1)}, seed=0)["image"].reshape(values.shape)


def _ct(intensityproperties: dict | None) -> ImageNormalization:
    """Clip to global percentiles, then z-score with the global mean and standard deviation."""
    properties = intensityproperties or {}
    missing = [key for key in ("mean", "std", "percentile_00_5", "percentile_99_5") if key not in properties]
    if missing:
        raise ValueError(f"CT normalization missing intensity properties: {missing}. Provided keys: {list(properties)}")
    return ImageNormalization(
        "fixed",
        center=float(properties["mean"]),
        scale=max(float(properties["std"]), EPS),
        bounds=(float(properties["percentile_00_5"]), float(properties["percentile_99_5"])),
    )


# Mapping from names to normalizers; "none" is no normalization.
NORMALIZATION_SCHEMES = {
    "zscore": lambda _: ImageNormalization("zscore", nonzero=False, eps=EPS),
    "ct": _ct,
    "rescale_to_01": lambda _: ImageNormalization("minmax"),
    "minmax": lambda _: ImageNormalization("minmax"),
    "robust": lambda _: ImageNormalization("robust"),
    "none": None,
}


def get_normalization(scheme: str, intensityproperties: dict | None = None) -> ImageNormalization | None:
    """The normalizer of a scheme, ``None`` for ``none``; ``intensityproperties`` are for ``ct``."""
    if scheme not in NORMALIZATION_SCHEMES:
        raise ValueError(f"Unknown normalization scheme: {scheme}. Available schemes: {list(NORMALIZATION_SCHEMES)}")
    build = NORMALIZATION_SCHEMES[scheme]
    return None if build is None else build(intensityproperties)
