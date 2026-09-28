"""Fast scalar crops for the direct follower.

Read only each oriented crop's tight source block. ChunkedArray memory-maps
uncompressed Zarr chunks; the fused Numba kernel interpolates them.
"""
import numpy as np
import torch

from .data import _grid_flat, read_tight_blocks
from .fast_sample import sample_scalar_crop


def scalar_crops(items, vol, crop, pool=None, *, presence=False):
    grid = _grid_flat(crop)
    raw, starts = read_tight_blocks(items, vol, crop, pool, presence=presence)
    scale = 1. if presence else vol.input_scale
    result = np.empty((len(items), 1, crop.depth, crop.width, crop.width), np.float32)
    for j, item in enumerate(items):
        sample_scalar_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale,
                           grid, result[j])
    return torch.from_numpy(result)
