"""Full-resolution canvas extent of a tifxyz grid, as vc_render_tifxyz sizes it.

Dependency-free on purpose: ``vesuvius.tifxyz`` (OpenCV) and
``vesuvius.tifxyz_label_transfer`` (SciPy) both need this and neither should
import the other for it.
"""

from __future__ import annotations

import math

import numpy as np


def full_resolution_extent(stored: int, scale: float) -> int:
    """Stored grid extent -> full-resolution canvas extent.

    ``vc_render_tifxyz.cpp`` ("Compute render scale") does::

        double sx = render_scale / surf->_scale[0];      // _scale is cv::Vec2f
        full_size.width = std::max(1, int(std::lround(full_size.width * sx)));

    i.e. the stored scale is a float32, the reciprocal is taken in double, the
    product is rounded half away from zero, and the result is at least 1.
    Tracer meshes store ``scale`` as float32 0.05 (``0.05000000074505806``);
    ``stored / scale`` in float64 then lands just below the integer
    (``152 / 0.05000000074505806 == 3039.99995...``), so truncating it is one
    pixel short of the canvas the renderer and the published surface volumes
    use. Plain ``round()`` is not equivalent either: Python rounds half to even,
    ``std::lround`` half away from zero, and exact halves occur with integer
    scales (7 x 5 at scale 2.0 renders as 4 x 3).

    A scale that is not a positive finite float32 (including one that
    underflows float32 to 0) falls back to the stored extent.
    """
    stored = int(stored)
    scale32 = float(np.float32(scale))
    if not math.isfinite(scale32) or scale32 <= 0.0:
        return max(1, stored)
    quotient = float(stored) * (1.0 / scale32)
    return max(1, int(math.floor(quotient + 0.5)))
