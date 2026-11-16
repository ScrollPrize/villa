"""Air levels of reconstructed slices against the distance to the object (calibrate_sld.py: air must be zero).
NumPy only."""
from __future__ import annotations

import numpy as np

BANDS = (0, 16, 32, 64, 128, 256, 512, 1024)   # distance to the object beyond the guard (pixels), band edges


def box_bin(a, b):
    n0, n1 = a.shape[0] // b * b, a.shape[1] // b * b
    return a[:n0, :n1].reshape(n0 // b, b, n1 // b, b).mean((1, 3))


def dilate(mask, n):
    """n steps of 3 x 3 dilation: chessboard distance <= n"""
    m = mask.copy()
    for _ in range(n):
        p = np.pad(m, 1)
        m = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:] | p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return m


def air_bands(ref, b, threshold, guard, disk=True):
    """label image of the air's distance bands of a reference slice `ref` box-binned by b: 0 not air, 1 .. NB open
    air by distance band, NB + 1 .. 2 NB enclosed air (NB = len(BANDS) - 1). Object: ref > threshold; air keeps
    `guard` pixels clear of it and, with `disk` (whole slices), of the reconstruction disk's edge. Open air is what
    can be reached from the disk's (or the crop's) edge without crossing the object and its guard."""
    nb = len(BANDS) - 1
    yy, xx = np.indices(ref.shape)
    if disk:
        n = ref.shape[0]
        inside = np.hypot(yy - (n - 1) / 2, xx - (n - 1) / 2) < n / 2 - guard / b
    else:
        inside = np.ones(ref.shape, bool)
        inside[[0, -1], :] = inside[:, [0, -1]] = False
    near = dilate(ref > threshold, max(1, round(guard / b)))
    labels = np.zeros(ref.shape, np.uint8)
    prev = near
    for i in range(1, nb + 1):
        cur = dilate(prev, max(1, round((BANDS[i] - BANDS[i - 1]) / b)))
        labels[cur & ~prev & inside] = i
        prev = cur
    free = ~near
    open_air = free & ~inside
    while True:
        grown = dilate(open_air, 1) & free
        if (grown == open_air).all():
            break
        open_air = grown
    labels[(labels > 0) & ~open_air] += nb
    return labels


def band_levels(img, labels):
    """median air level per band: open bands, then enclosed bands (nan: a band of fewer than 50 pixels)"""
    nb = len(BANDS) - 1
    return [float(np.median(img[labels == i])) if (labels == i).sum() >= 50 else float("nan") for i in range(1, 2 * nb + 1)]

