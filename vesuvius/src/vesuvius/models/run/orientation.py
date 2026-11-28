"""Principal axes of orientation tensors, for numpy (blending) and torch (TTA) arrays.

Kept free of torch so the `blending` extra can blend axis heads without it.
"""
import math

import numpy as np

# An axis is written in the half-space 4x + 2y + z >= 0 of the input frame (x, y, z weights), so
# the output has one sign convention. No coordinate axis or axis diagonal lies on its boundary.
_HALF_SPACE = (4.0, 2.0, 1.0)
# The orientation tensors of an axis head are reduced this many voxels (whole rows) at a time,
# so their float32 temporaries have a fixed size instead of growing with the patch or chunk.
_SLAB_VOXELS = 1 << 18
# Squarings in `principal_axis`: the axis error falls like (l2 / l1) ** (2 ** _SQUARINGS).
_SQUARINGS = 6


def _pairs(d):
    return [(i, j) for i in range(d) for j in range(i, d)]


def _row_slabs(rows, row_voxels):
    """Slices of whole rows (the first spatial dim) holding about `_SLAB_VOXELS` voxels each."""
    step = max(1, _SLAB_VOXELS // max(1, int(row_voxels)))
    return [slice(s, min(s + step, rows)) for s in range(0, rows, step)]


def _reciprocal_or_zero(x, xp):
    positive = x > 0
    return xp.where(positive, 1 / xp.where(positive, x, 1), 0)


def principal_axis(m, d, xp):
    """The principal axis of per-voxel orientation tensors.

    `m` maps (i, j), i <= j < d, to float32 arrays of the module `xp` (numpy or torch): the
    components of the (weighted) mean of v v^T over the predictions v of each voxel. Returns the
    d components of the eigenvector of its largest eigenvalue, scaled to the square root of that
    eigenvalue and put in the half-space `_HALF_SPACE`. Predictions that agree up to sign give
    their common vector (with its length); in general the length is the root mean square of
    their projections on the axis. The result does not depend on the predictions' signs or order.
    """
    def at(p, i, j):
        return p[(i, j) if i <= j else (j, i)]

    def normalized(p):
        inv = _reciprocal_or_zero(sum(p[(i, i)] for i in range(d)), xp)
        return {key: value * inv for key, value in p.items()}

    # Repeated squaring: (M / tr M) ** (2 ** k), renormalised, tends to the projector on the axis.
    p = normalized(m)
    for _ in range(_SQUARINGS):
        p = normalized({(i, j): sum(at(p, i, k) * at(p, k, j) for k in range(d)) for i, j in _pairs(d)})
    # Its column with the largest diagonal entry is a nonzero multiple of the axis (unless M = 0).
    column, diagonal = [at(p, i, 0) for i in range(d)], p[(0, 0)]
    for j in range(1, d):
        take = p[(j, j)] > diagonal
        diagonal = xp.where(take, p[(j, j)], diagonal)
        column = [xp.where(take, at(p, i, j), c) for i, c in enumerate(column)]
    inv = _reciprocal_or_zero(xp.sqrt(sum(c * c for c in column)), xp)
    axis = [c * inv for c in column]
    eigenvalue = sum(at(m, i, j) * axis[i] * axis[j] for i in range(d) for j in range(d))
    length = xp.sqrt(eigenvalue.clip(min=0))
    side = sum(w * c for w, c in zip(_HALF_SPACE, axis))
    length = xp.where(side < 0, -length, length)
    return [c * length for c in axis]


def axis_head_channels(target_info, spatial_dims=3):
    """The (x, y[, z]) channels of each head marked `sign_invariant` in `target_info`."""
    heads = sorted((target_info or {}).values(), key=lambda info: info['start_channel'])
    return [tuple(range(int(info['start_channel']), int(info['start_channel']) + spatial_dims))
            for info in heads if info.get('sign_invariant', False)]


def orientation_layouts(axis_channels, num_channels):
    """Where a blend accumulates the orientation tensors v v^T of its axis heads.

    `axis_channels` holds the (x, y, z) channels of each axis head (see `axis_head_channels`).
    The diagonal xx, yy, zz goes to the head's own channels, the off-diagonal xy, xz, yz to
    three extra channels after the `num_channels` output channels. Returns one {(i, j): channel}
    per head; the accumulator needs `num_channels + 3 * len(axis_channels)` channels (3D).
    """
    layouts, extra = [], int(num_channels)
    for channels in axis_channels:
        layout = {}
        for i, j in _pairs(len(channels)):
            if i == j:
                layout[(i, i)] = int(channels[i])
            else:
                layout[(i, j)], extra = extra, extra + 1
        layouts.append(layout)
    return layouts


def accumulator_channels(layouts, num_channels):
    """Channel count of a blend accumulator with these `orientation_layouts`."""
    return max([int(num_channels)] + [c + 1 for layout in layouts for c in layout.values()])


def reduce_orientation_tensors(sums, layouts) -> None:
    """In blended means `sums` (C_acc, Z, Y, X, numpy), replace the orientation tensor of each axis
    head (`orientation_layouts`) by its principal axis, in the head's x, y, z channels."""
    row_voxels = math.prod(sums.shape[2:])
    for layout in layouts:
        d = sum(1 for i, j in layout if i == j)
        channels = [layout[(i, i)] for i in range(d)]
        for rows in _row_slabs(sums.shape[1], row_voxels):
            axis = principal_axis({key: sums[c, rows] for key, c in layout.items()}, d, np)
            for c, component in zip(channels, axis):
                sums[c, rows] = component
