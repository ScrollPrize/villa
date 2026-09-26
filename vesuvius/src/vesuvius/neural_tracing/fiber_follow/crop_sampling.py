"""Fast scalar crops shared by the direct follower and the learned beam.

Read only each oriented crop's tight source block. ChunkedArray memory-maps
uncompressed Zarr chunks; the fused Numba kernel interpolates and optionally
renders observed history in the same pass.
"""
import numpy as np
import torch

from .data import _grid_flat, read_tight_blocks, render_count
from .fast_sample import sample_crop


def scalar_crops(items, vol, crop, pool=None, *, presence=False, history=False):
    grid = _grid_flat(crop)
    empty = np.empty((0, 3), np.float32)
    mask = np.empty(0, np.float32)
    raw, starts = read_tight_blocks(items, vol, crop, pool, presence=presence)
    scale = 1. if presence else vol.input_scale
    channels = 2 if history else 1
    result = np.empty((len(items), channels, crop.depth, crop.width, crop.width), np.float32)
    for j, item in enumerate(items):
        hist = item['hist_local'][:render_count(crop)] if history else empty
        hmask = item['hmask'][:render_count(crop)] if history else mask
        sampled = sample_crop(raw[j], starts[j], item['pos']*scale, item['frame']*scale,
                              grid, False, hist, hmask, 2, crop.history_sigma, crop.history_render)
        result[j] = sampled[:channels].reshape(channels, crop.depth, crop.width, crop.width)
    return torch.from_numpy(result)
