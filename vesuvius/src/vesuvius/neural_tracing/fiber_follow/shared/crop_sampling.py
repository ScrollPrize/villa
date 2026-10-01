"""Fast scalar crops for the direct follower.

Read only each oriented crop's tight source block. ChunkedArray memory-maps
uncompressed Zarr chunks; the fused Numba kernel interpolates them.
"""
import numpy as np
import torch

from .data import _grid_flat, read_tight_blocks
from .fast_sample import sample_scalar_crop


def empty_image_batch(shape):
    """Write worker images into their final IPC storage, avoiding a full copy.

    This is the same allocation used by PyTorch's default tensor collator.
    Keep the returned tensor as the batch value: wrapping its NumPy view in a
    new tensor would lose the shared-storage handle and reintroduce the copy.
    """
    tensor = torch.empty(0, dtype=torch.float32)
    if torch.utils.data.get_worker_info() is not None:
        storage = tensor._typed_storage()._new_shared(int(np.prod(shape)), device=tensor.device)
        return tensor.new(storage).resize_(shape)
    return tensor.resize_(shape)


def unique_crop_indices(items):
    """First row for each identical sampling geometry, in input order.

    Matched decisions share a current crop but keep distinct history/labels.
    This cache lasts for one call; output rows remain independently writable.
    """
    seen = {}
    sources = []
    for j, item in enumerate(items):
        pos, frame = np.asarray(item['pos']), np.asarray(item['frame'])
        # Scalar crops scale before converting to float64, so source dtype can
        # affect rounding even when the unscaled coordinates compare equal.
        key = (pos.dtype.str, pos.shape, pos.tobytes(), frame.dtype.str, frame.shape, frame.tobytes())
        sources.append(seen.setdefault(key, j))
    return sources


def scalar_crops(items, vol, crop, pool=None, *, presence=False, out=None):
    grid = _grid_flat(crop)
    sources = unique_crop_indices(items)
    unique = [j for j, source in enumerate(sources) if j == source]
    raw, starts = read_tight_blocks([items[j] for j in unique], vol, crop, pool, presence=presence)
    scale = 1. if presence else vol.input_scale
    shape = (len(items), 1, crop.depth, crop.width, crop.width)
    result = np.empty(shape, np.float32) if out is None else out
    if result.shape != shape or result.dtype != np.float32:
        raise ValueError('Scalar crop destination must have the expected shape and float32 dtype')
    if any(not result[j].flags.c_contiguous for j in range(len(items))):
        raise ValueError('Each scalar crop destination must be contiguous')
    for k, j in enumerate(unique):
        item = items[j]
        sample_scalar_crop(raw[k], starts[k], item['pos']*scale, item['frame']*scale,
                           grid, result[j])
    for j, source in enumerate(sources):
        if j != source:
            result[j] = result[source]
    return torch.from_numpy(result)
