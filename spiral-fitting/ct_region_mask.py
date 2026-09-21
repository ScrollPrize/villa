"""Per-slice CT envelope used by mask_zarr_from_ct.py.

Defaults reproduce the reviewed level-2 CT masks, in that array's voxels.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class MaskSettings:
    threshold: float = 65.0
    ct_sigma: float = 1.0
    seed_radius: int = 3
    closing_radius: int = 12
    boundary_sigma: float = 4.0

    def __post_init__(self):
        values = (self.threshold, self.ct_sigma, self.seed_radius,
                  self.closing_radius, self.boundary_sigma)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('mask settings must be finite')
        if any(v < 0 for v in values[1:]):
            raise ValueError('sigmas and radii must be nonnegative')
        if any(int(v) != v for v in (self.seed_radius, self.closing_radius)):
            raise ValueError('closing radii must be integers')


def largest_component(mask):
    """Keep the largest 4-connected component; ties follow raster label order."""
    labels, count = ndi.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == sizes.argmax()


def binary_close(mask, radius):
    if not radius:
        return mask.copy()
    pad = radius + 1
    padded = np.pad(mask.astype(np.uint8), pad)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)[pad:-pad, pad:-pad] > 0


def fill_holes(mask):
    """4-connected exterior flood, equivalent to ndi.binary_fill_holes.

    A zero frame connects every image edge to the flood seed, including
    foreground touching corners. Only exterior background becomes 2.
    """
    flooded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
    cv2.floodFill(flooded, None, (0, 0), 2, flags=4)
    return flooded[1:-1, 1:-1] != 2


def make_ct_mask(image, settings=MaskSettings()):
    """Return zero or one connected region, with the dim rim excluded.

    Removing satellites before the wider closing avoids absorbing them back
    into the main region. Internal cavities are filled. Nonfinite CT samples
    are treated as zero before segmentation. Intensities are not normalized.
    """
    if image.ndim != 2:
        raise ValueError('CT mask requires a 2D slice')
    image = np.asarray(image, dtype=np.float32)
    image = np.where(np.isfinite(image), image, 0)
    support = image > 0
    if not support.any():
        return np.zeros(image.shape, dtype=bool)
    seed = ndi.gaussian_filter(image, settings.ct_sigma) >= settings.threshold
    seed = largest_component(fill_holes(
        binary_close(seed, settings.seed_radius)))
    if not seed.any():
        return seed
    region = fill_holes(binary_close(seed, settings.closing_radius))
    # Bound distance-transform memory without changing the reviewed algorithm.
    yy, xx = np.nonzero(region)
    pad = int(np.ceil(4 * settings.boundary_sigma)) + 2
    y0, y1 = max(0, yy.min() - pad), min(region.shape[0], yy.max() + pad + 1)
    x0, x1 = max(0, xx.min() - pad), min(region.shape[1], xx.max() + pad + 1)
    cropped = np.pad(region[y0:y1, x0:x1], pad)
    distance = ndi.distance_transform_edt(cropped) - ndi.distance_transform_edt(~cropped)
    smoothed = ndi.gaussian_filter(distance, settings.boundary_sigma) > 0
    result = np.zeros(region.shape, dtype=bool)
    result[y0:y1, x0:x1] = smoothed[pad:-pad, pad:-pad]
    # Constrain the outer boundary first, then fill: zero-valued internal CT
    # cavities belong to the envelope too, not just positive-intensity gaps.
    return largest_component(fill_holes(result & support))
