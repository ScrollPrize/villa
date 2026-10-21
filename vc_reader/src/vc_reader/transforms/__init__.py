"""Image and volume transforms with CPU and optional CUDA execution.

Python constructs configurations and adapts dictionaries/array containers.
Parameter sampling, target routing, transforms, and replay run without the GIL.
"""
import numpy as np

from .. import _transforms
from . import _config
from ._pipeline import Pipeline, Plan

globals().update({name: getattr(_config, name) for name in _config._constructors})
seed = _transforms.seed


def translate_crop_box(box, points, *, seed, **kwargs):
    """Translate a ZYX start/stop box, retaining supervised (N, 3) points.

    Filter invalid/unsupervised points before calling, then read the returned box.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return tuple(_transforms.translate_crop_box(box, points.ravel().tolist(), seed, **kwargs))


__all__ = [*_config._constructors, "Pipeline", "Plan", "seed", "translate_crop_box"]
