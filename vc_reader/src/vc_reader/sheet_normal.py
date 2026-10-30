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
from . import _reader

RECIPE = _reader.SHEET_NORMAL_RECIPE


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
    if offsets.shape != (n, 3):
        raise ValueError("offsets_xyz must be finite (B,3) values in [0,3)")
    mask = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    if mask.shape != (n,):
        raise ValueError("valid must have shape (B,)")
    centers = _reader._sheet_normal_centers(np.ascontiguousarray(offsets), np.ascontiguousarray(mask))
    if device:
        if out is not None:
            if not is_device_array(out):
                raise TypeError('out must be a CUDA array for CUDA patches')
            destination = wrap(out)
            _reader._sheet_normals_cuda(patches._array, centers, destination._array)
            return out
        return DeviceArray(_reader._sheet_normals_cuda(patches._array, centers), (n, 4), np.float32, patches.device)
    if out is not None:
        if not isinstance(out, np.ndarray) or out.dtype != np.float32 or out.shape != (n,4) or not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError('out must be writable contiguous (B,4) float32 NumPy storage')
        if np.shares_memory(patches, out):
            raise ValueError('out must not overlap input patches')
    return _reader._sheet_normals_cpu(np.ascontiguousarray(patches), centers, out)


def sheet_normal_regions(points_xyz, shape_zyx):
    """Return integer ZYX starts, fractional XYZ offsets, and support validity.

    Global coordinates retain float64 until the integer origin is removed;
    this prevents loss of subvoxel precision in large scrolls. Image work and
    local offsets use float32.
    """
    points = np.ascontiguousarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be finite (B,3)")
    return _reader._sheet_normal_regions(points, tuple(int(n) for n in shape_zyx))


def sample_sheet_normals(reader, points_xyz, *, out=None):
    """Read one bounded batch and estimate normals with a ZarrReader/CudaReader.

    CUDA uses source='chunks' with no CPU sampling fallback. Batch requests
    should be grouped by the caller (typically 4–32 queries) to bound memory.
    All 109³ context voxels must be inside the volume; padding is invalid.
    """
    from .cuda import DeviceArray
    from .reader.cuda import CudaReader
    points = np.ascontiguousarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must be finite (B,3)")
    if isinstance(reader, CudaReader):
        reader._ready()
        shape = tuple(reader.reader.shape)
        return DeviceArray(_reader._sample_sheet_normals_cuda(reader._engine, shape, points),
                           (len(points), 4), np.float32, reader.device)
    reader._ready()
    return _reader._sample_sheet_normals(reader._opened, points)
