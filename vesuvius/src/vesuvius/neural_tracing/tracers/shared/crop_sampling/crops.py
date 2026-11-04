"""Oriented scalar crops of uint8 CT, sampled by vc_reader.

A crop (``CropSpec``, frame columns ``u, v, f``) is laid out ``(D, H, W)`` = forward (f), v, u; sample ``(i, j, k)``
lies at CT-array xyz ``pos*scale + (frame*scale) @ (lateral[k], vertical[j], forward[i])``. ``crop_grids`` turns that
into the origin and axes of ``ZarrReader.read_affine_many`` (CPU) and ``CudaReader.read_affine_batch`` (CUDA): the
same trilinear values, CT/255, with the array's fill value outside it.
"""
import numpy as np
import torch

from vesuvius.neural_tracing.tracers.shared.geometry import CropSpec
from vesuvius.neural_tracing.tracers.shared.crop_sampling.normalization import normalize_ct, zscore_crops


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


def crop_grids(items, crop: CropSpec, scale: float):
    """Output shape (D, H, W), zyx origins (B, 3) and zyx axes (B, 3, 3; columns: steps along f, v, u) of the items'
    crops in CT-array voxels (``scale`` CT voxels per trace voxel)."""
    frames = np.stack([np.asarray(item['frame'], np.float64) for item in items])*scale  # rows xyz, columns u, v, f
    pos = np.stack([np.asarray(item['pos'], np.float64) for item in items])*scale
    first = np.array([crop.lateral_coords[0], crop.vertical_coords[0], crop.forward_coords[0]])
    origins = (pos + frames @ first)[:, ::-1]
    axes = frames[:, ::-1, ::-1]*crop.spacing
    return (crop.depth, crop.height or crop.width, crop.width), np.ascontiguousarray(origins), np.ascontiguousarray(axes)


def scalar_crops(items, ct, spec, crop, *, out=None, normalize=True):
    """(B, 1, D, H, W) float32 crops of the uint8 CT ``ct`` (a vc_reader ``ZarrReader``): CT/255, then normalized
    with ``spec.ct_normalization`` unless ``normalize`` is false."""
    if ct.dtype != np.uint8:
        raise ValueError('CT intensity normalization currently requires uint8 data')
    sources = unique_crop_indices(items)
    unique = [j for j, source in enumerate(sources) if j == source]
    shape = (len(items), 1, crop.depth, crop.height or crop.width, crop.width)
    result = np.empty(shape, np.float32) if out is None else out
    if result.shape != shape or result.dtype != np.float32:
        raise ValueError('Scalar crop destination must have the expected shape and float32 dtype')
    if any(not result[j].flags.c_contiguous for j in range(len(items))):
        raise ValueError('Each scalar crop destination must be contiguous')
    if unique:
        size, origins, axes = crop_grids([items[j] for j in unique], crop, spec.input_scale)
        ct.read_affine_many([(size, o, a) for o, a in zip(origins, axes)], out=[result[j, 0] for j in unique])
    for j in unique:
        result[j] *= np.float32(1/255)
        if normalize:
            normalize_ct(result[j, 0], spec.ct_normalization)
    for j, source in enumerate(sources):
        if j != source:
            result[j] = result[source]
    return torch.from_numpy(result)


def cuda_scalar_crops(items, cuda, spec, crop) -> torch.Tensor:
    """``scalar_crops`` on the GPU from a vc_reader ``CudaReader`` (``ZarrReader.cuda()``): (B, 1, D, H, W) float32
    crops on its device, z-scored per crop (``zscore_crops``)."""
    out = torch.empty((len(items), crop.depth, crop.height or crop.width, crop.width), dtype=torch.float32,
                      device=torch.device('cuda', cuda.device))
    if items:
        size, origins, axes = crop_grids(items, crop, spec.input_scale)
        cuda.read_affine_batch(size, origins, axes, order=1, out=out)
    return zscore_crops(out[:, None]/255.)
