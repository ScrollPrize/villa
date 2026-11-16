"""OME-Zarr output of reconstructed volumes: the streaming writer (helirec's C++ OmeZarrWriter, zarr.hpp) and the
uint8 conversion.

    w = OmeZarrWriter(path, shape, np.uint8, chunks=(128, 128, 128), voxel_size_um=8.64, levels=3,
                      compression="vc-delta3d")
    w.write(z0, slices)                                  # or SliceReconstructor.reconstruct(uint8_to=w, z_to=z0)
    w.close()
"""
from __future__ import annotations

import numpy as np

from ._core import OmeZarrWriter  # noqa: F401


def to_uint8(img, lo=-0.03, hi=0.145, disk=None, mask=None):
    """float slice(s) (..., ny, nx) -> uint8 with the window [lo, hi] (default: the reference uint8 window).
    disk = (n, x0, y0): the slices are the ROI at (x0, y0) of n x n slices; the pixels outside their reconstruction
    disk are set to 0 (so empty chunks stay empty). mask: the result is 0 where it is 0: a uint8 array of img's
    shape, or a helirec OutputMask over the slices (possibly coarser).
    The kernel behind SliceReconstructor.reconstruct(uint8=...)."""
    from . import _core
    if mask is not None and not isinstance(mask, _core.OutputMask):
        mask = np.ascontiguousarray(mask, np.uint8)
    return _core.to_uint8(np.ascontiguousarray(img, np.float32), lo, hi, disk, mask=mask)
