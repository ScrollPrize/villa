"""Shared decoder for Lasagna's uint8 surface-normal channels."""

from __future__ import annotations

import numpy as np


def decode_normals(
    nx_u8: np.ndarray, ny_u8: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode Lasagna uint8 ``(nx, ny)`` normals into float32 ``(nx, ny, nz)``.

    ``nx`` and ``ny`` are stored with 128 at zero and a scale of 127. ``nz`` is
    the non-negative hemisphere component ``sqrt(max(0, 1 - nx^2 - ny^2))``.
    This is the single implementation of the decoder that used to live in
    ``lasagna.omezarr_pyramid._decode_normals``.
    """
    nx = (nx_u8.astype(np.float32, copy=False) - 128.0) / 127.0
    ny = (ny_u8.astype(np.float32, copy=False) - 128.0) / 127.0
    nz = np.sqrt(np.maximum(0.0, 1.0 - nx * nx - ny * ny)).astype(np.float32, copy=False)
    return nx, ny, nz
