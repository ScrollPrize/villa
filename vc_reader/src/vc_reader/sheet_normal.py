"""Anisotropic CT sheet normals used by fiber-strip linking.

Recipe ct-anisotropic-1-24-4-grid3-r48-v4: full-resolution Gaussian
derivatives (seed sigma 2, refinement sigma 1), global 3-voxel lattice,
radius 48, tangent sigma 24, normal sigma 4, three refinements, and
median-floored inverse gradient magnitude weighting. Normals are unsigned
XYZ unit vectors; column 3 is the relative principal eigengap. Invalid
normals (flat/ambiguous tensor or insufficient volume support) are zero.
"""
from __future__ import annotations

import numpy as np
from . import _native

RECIPE = "ct-anisotropic-1-24-4-grid3-r48-v4"


def sheet_normals(patches, offsets_xyz=None, *, valid=None, out=None):
    """Estimate (B,4) normal XYZ + gap from (B,109,109,109) uint8/float32 CT.

    CT intensities are in [0,255]. Patches start at floor(query/3)*3 - 54
    in each axis. offsets_xyz is query - floor(query/3)*3, in [0,3).
    Array axes are BZYX; vectors are XYZ. CPU input returns NumPy; CUDA
    DLPack input returns DeviceArray, without a download or Torch dependency.
    CUDA calls complete before returning; finish external writes first.
    Only small host offsets and validity metadata are uploaded.
    out optionally supplies writable contiguous (B,4) float32 storage on the
    same device. The supplied object is returned, without a temporary output.
    """
    from .cuda import is_device_array, wrap, DeviceArray
    device = is_device_array(patches)
    patches = wrap(patches) if device else np.asarray(patches)
    if len(patches.shape) != 4 or tuple(patches.shape[1:]) != (109, 109, 109):
        raise ValueError("patches must have shape (B,109,109,109)")
    n = patches.shape[0]
    offsets = np.zeros((n, 3), np.float32) if offsets_xyz is None else np.asarray(offsets_xyz, np.float32)
    if offsets.shape != (n, 3) or not np.isfinite(offsets).all() or np.any((offsets < 0) | (offsets >= 3)):
        raise ValueError("offsets_xyz must be finite (B,3) values in [0,3)")
    centers = np.ones((n, 4), np.float32)
    centers[:, :3] = offsets + np.float32(48)
    # Preserve the open upper bound after float32 addition near a lattice edge.
    np.minimum(centers[:, :3], np.nextafter(np.float32(51), np.float32(48)), out=centers[:, :3])
    if valid is not None:
        mask = np.asarray(valid, bool)
        if mask.shape != (n,):
            raise ValueError("valid must have shape (B,)")
        centers[:, 3] = mask
    if device:
        if out is not None:
            if not is_device_array(out):
                raise TypeError('out must be a CUDA array for CUDA patches')
            destination = wrap(out)
            _native._sheet_normals_cuda(patches._native, centers, destination._native)
            return out
        return DeviceArray(_native._sheet_normals_cuda(patches._native, centers), (n, 4), np.float32, patches.device)
    if out is not None:
        if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != (n,4) or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('out must be writable contiguous (B,4) float32 NumPy storage')
        if np.shares_memory(patches, out):
            raise ValueError('out must not overlap input patches')
    return _native._sheet_normals_cpu(np.ascontiguousarray(patches), centers, out)


def sheet_normal_regions(points_xyz, shape_zyx):
    """Return integer ZYX starts, fractional XYZ offsets, and support validity.

    Global coordinates retain float64 until the integer origin is removed;
    this prevents loss of subvoxel precision in large scrolls. Image work and
    local offsets use float32.
    """
    points = np.asarray(points_xyz, dtype=np.float64)
    shape = np.asarray(shape_zyx)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points_xyz must be finite (B,3)")
    if shape.shape != (3,) or np.any(shape < 0):
        raise ValueError("shape_zyx must have three nonnegative dimensions")
    if np.any(np.abs(points) >= 2**52):
        raise ValueError("query coordinates exceed exact integer lattice range")
    lattice = np.floor(points / 3).astype(np.int64) * 3
    starts = np.ascontiguousarray(lattice[:, ::-1] - 54)
    valid = np.all((starts >= 0) & (starts + 109 <= shape), axis=1)
    return starts, (points - lattice).astype(np.float32), valid


def sample_sheet_normals(reader, points_xyz, *, out=None):
    """Read one bounded batch and estimate normals with a ZarrReader/CudaReader.

    CUDA uses source='chunks' with no CPU sampling fallback. Batch requests
    should be grouped by the caller (typically 4–32 queries) to bound memory.
    All 109³ context voxels must be inside the volume; padding is invalid.
    """
    from .reader.cuda import CudaReader
    cuda = isinstance(reader, CudaReader)
    shape = reader.reader.shape if cuda else reader.shape
    starts, offsets, valid = sheet_normal_regions(points_xyz, shape)
    if cuda:
        axes = np.broadcast_to(np.eye(3, dtype=np.float32), (len(starts), 3, 3))
        patches = reader.read_affine_batch((109, 109, 109), starts, axes, order=0, source="chunks")
    else:
        patches = np.stack([reader.read_region(start, (109, 109, 109)) for start in starts]) if len(starts) else np.empty((0,109,109,109),np.uint8)
    return sheet_normals(patches, offsets, valid=valid, out=out)
