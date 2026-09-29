"""Fast scalar crops for the direct follower.

Read only each oriented crop's tight source block. ChunkedArray memory-maps
uncompressed Zarr chunks; the fused Numba kernel interpolates them.
"""
import numpy as np
import torch

from .data import _grid_flat, read_tight_blocks
from .fast_sample import sample_scalar_crop


def scalar_crops(items, vol, crop, pool=None, *, presence=False, out=None):
    grid = _grid_flat(crop)
    raw, starts = read_tight_blocks(items, vol, crop, pool, presence=presence)
    scale = 1. if presence else vol.input_scale
    shape = (len(items), 1, crop.depth, crop.width, crop.width)
    result = np.empty(shape, np.float32) if out is None else out
    if result.shape != shape or result.dtype != np.float32:
        raise ValueError('Scalar crop destination must have the expected shape and float32 dtype')
    if any(not result[j].flags.c_contiguous for j in range(len(items))):
        raise ValueError('Each scalar crop destination must be contiguous')
    for j, item in enumerate(items):
        sample_scalar_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale,
                           grid, result[j])
    return torch.from_numpy(result)
