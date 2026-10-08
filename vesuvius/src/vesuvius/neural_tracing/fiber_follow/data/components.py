"""Clearance rule and crop indices for neighboring-fiber geometry.

All coordinates are crop-local trace voxels (a, b, c) = (u, v, forward); crop
volumes are laid out (c, b, a) like ``crop_local_grid``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ComponentRule:
    """Pair support/clearance settings; the old component sampler is retired."""
    own_radius: float = 1.5  # minimum target clearance
    lateral_max: float = 12.  # lateral support radius; crop support still applies
    along_window: float = 2.  # negatives within this arc of their positive


def crop_indices(crop, local):
    """Continuous volume indices (c, b, a) for crop-local (a, b, c) points."""
    local = np.asarray(local, np.float64).reshape(-1, 3)
    return np.c_[local[:, 2]/crop.spacing+crop.behind, local[:, 1]/crop.spacing+(crop.width-1)/2,
                 local[:, 0]/crop.spacing+(crop.width-1)/2]
