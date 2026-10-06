"""Fast oriented scalar crops.

Read only each oriented crop's tight source block. ChunkedArray memory-maps
uncompressed Zarr chunks; the fused Numba kernel interpolates them.
"""
import numpy as np
import torch

from vesuvius.neural_tracing.tracers.ct.volume import FiberVolume
from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec, crop_local_grid
from vesuvius.neural_tracing.tracers.ct.fast_sample import sample_scalar_crop
from vesuvius.neural_tracing.tracers.ct.normalization import normalize_ct


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


def scalar_crops(items, vol, crop, pool=None, *, out=None, normalize=True):
    grid = _grid_flat(crop)
    sources = unique_crop_indices(items)
    unique = [j for j, source in enumerate(sources) if j == source]
    raw, starts = read_tight_blocks([items[j] for j in unique], vol, crop, pool)
    scale = vol.input_scale
    shape = (len(items), 1, crop.depth, crop.width, crop.width)
    result = np.empty(shape, np.float32) if out is None else out
    if result.shape != shape or result.dtype != np.float32:
        raise ValueError('Scalar crop destination must have the expected shape and float32 dtype')
    if any(not result[j].flags.c_contiguous for j in range(len(items))):
        raise ValueError('Each scalar crop destination must be contiguous')
    def sample(k):
        j = unique[k]
        item = items[j]
        sample_scalar_crop(raw[k], starts[k], item['pos']*scale, item['frame']*scale,
                           grid, result[j])
        if normalize:
            normalize_ct(result[j, 0], vol.spec.ct_normalization)
    # Rows are independent and the Numba kernel releases the GIL, so a pool only
    # changes scheduling; each row's values are identical to the serial loop.
    if pool is None:
        for k in range(len(unique)):
            sample(k)
    else:
        list(pool.map(sample, range(len(unique))))
    for j, source in enumerate(sources):
        if j != source:
            result[j] = result[source]
    return torch.from_numpy(result)


_GRID_CACHE: dict = {}


def _grid_flat(crop: CropSpec) -> np.ndarray:
    g = _GRID_CACHE.get(crop)
    if g is None:
        g = _GRID_CACHE[crop] = crop_local_grid(crop).reshape(-1, 3).astype(np.float64)
    return g


_CORNER_CACHE: dict = {}


def crop_corners(crop: CropSpec) -> np.ndarray:
    """Eight local (a, b, c) corners; the crop's sample points are their convex hull."""
    corners = _CORNER_CACHE.get(crop)
    if corners is None:
        lc, fc = crop.lateral_coords, crop.forward_coords
        corners = _CORNER_CACHE[crop] = np.array([(a, b, c) for a in (lc[0], lc[-1])
                                                  for b in (lc[0], lc[-1]) for c in (fc[0], fc[-1])], np.float64)
    return corners


def tight_block(pos, frame, crop: CropSpec, scale: float = 1.):
    """zyx start and size of the smallest axis-aligned block around one oriented crop.

    Source-array coordinates (``scale`` source voxels per trace voxel). The block
    holds every sample point plus the trilinear/nearest support voxel above it
    and one guard voxel on each side, so a per-sample sampler that zero-pads
    beyond the block reads exactly the values it reads from the larger
    rotation-invariant ``block_start`` block (arrays fill outside with zero too).
    Unlike ``block_start`` the footprint depends on the frame, so the holdout
    guard keeps using the rotation-invariant block.
    """
    world = np.asarray(pos, np.float64)*scale + crop_corners(crop) @ (np.asarray(frame, np.float64)*scale).T
    zyx = world[:, ::-1]
    lo = np.floor(zyx.min(0)).astype(np.int64) - 1
    hi = np.floor(zyx.max(0)).astype(np.int64) + 3
    return lo, hi - lo


def read_tight_blocks(items, vol: FiberVolume, crop: CropSpec, pool=None):
    """Per-item minimal blocks for per-sample CPU sampling: (list of raw, list of zyx starts).

    Same values as the rotation-invariant ``block_start`` block at every sample point, reading roughly a
    third of the voxels and touching about half the chunks per item.
    """
    scale = getattr(vol, 'input_scale', 1.)
    bounds = [tight_block(it["pos"], it["frame"], crop, scale) for it in items]
    read = lambda b: vol.raw_block(b[0], b[1])
    raw = list(map(read, bounds) if pool is None else pool.map(read, bounds))
    return raw, [start for start, _ in bounds]
