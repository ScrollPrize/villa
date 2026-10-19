"""Batched float32 curve geometry and sparse voxel occupancy on CPU or CUDA.

Polyline coordinates use XYZ; voxel coordinates use ZYX. Remove a suitable
coordinate origin before converting large world coordinates to float32.
CUDA queries and results stay on device; calls finish on vc_reader's selected
stream before returning. Finish external writes before supplying CUDA arrays.
Static curve indexing is built on CPU once, then uploaded by ``cuda()``.
"""
import operator
import os

import numpy as np

from . import _native
from . import cuda as _cuda


def _device(value):
    return _cuda.wrap(value).device if _cuda.is_device_array(value) else -1


def _input(value, dtype, device):
    if device >= 0:
        if not _cuda.is_device_array(value):
            raise TypeError('CUDA geometry queries require CUDA arrays; upload explicitly')
        value = _cuda.wrap(value)
        if value.device != device or value.dtype != np.dtype(dtype):
            raise ValueError(f'expected contiguous {np.dtype(dtype)} on CUDA device {device}')
        return value
    if _cuda.is_device_array(value):
        raise TypeError('CPU geometry queries require host arrays')
    value = np.asarray(value)
    if np.dtype(dtype).kind in 'iu' and value.dtype.kind not in 'iu' and value.size:
        raise TypeError('indices must have an integer dtype')
    return np.ascontiguousarray(value, dtype=dtype)


def _native_array(value):
    return value._native if isinstance(value, _cuda.DeviceArray) else value


def _empty(shape, dtype, device):
    return np.empty(shape, dtype) if device < 0 else _cuda.empty(shape, dtype, device=device)


def _output(out, shape, dtype, device):
    if out is None:
        out = _empty(shape, dtype, device)
    view = _cuda.wrap(out) if _cuda.is_device_array(out) else out
    if device < 0:
        if not isinstance(view, np.ndarray) or not view.flags.c_contiguous or not view.flags.writeable:
            raise ValueError('CPU out must be writable contiguous NumPy storage')
    elif not isinstance(view, _cuda.DeviceArray) or view.device != device:
        raise ValueError('out must be on the geometry CUDA device')
    if tuple(view.shape) != tuple(shape) or np.dtype(view.dtype) != np.dtype(dtype):
        raise ValueError(f'out must have shape {shape} and dtype {np.dtype(dtype)}')
    return out, _native_array(view)


def _outputs(out, specification, device):
    if out is not None and (not isinstance(out, dict) or set(out) != set(specification)):
        raise ValueError(f'out requires fields {tuple(specification)}')
    result, native = {}, []
    for key, (shape, dtype) in specification.items():
        result[key], view = _output(None if out is None else out[key], shape, dtype, device)
        native.append(view)
    return result if out is None else out, native


def _points(value, device):
    value = _input(value, np.float32, device)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError('points must have shape (Q,3)')
    return value


def _ids(value, n, device):
    if value is None:
        return _empty((0,), np.int64, device)
    value = _input(value, np.int64, device)
    if value.shape != (n,):
        raise ValueError('IDs must have shape (Q,)')
    return value


class _Resident:
    def _ready(self):
        if self.device >= 0 and self._pid != os.getpid():
            raise RuntimeError('CUDA geometry cannot be used after fork')

    @property
    def nbytes(self):
        """Bytes of persistent native geometry/index storage."""
        self._ready()
        return self._native.nbytes

    def __getstate__(self):
        raise TypeError('Geometry indexes cannot be pickled; construct them in the worker')


class PolylineSet(_Resident):
    """Immutable packed curves, with BVHs for vertices and line segments.

    points: (N,3), offsets: (C+1,) starting at zero and ending at N. Every
    curve must have at least one vertex; offsets=[0] represents an empty set.
    Optional arclengths: (N,) finite nondecreasing coordinates within each
    curve. Otherwise float32 Euclidean cumulative lengths start at zero.
    Duplicate vertices are supported. Construction copies the inputs.
    Coordinates must be finite and within +/-1e15, keeping squared segment
    lengths representable in float32.

    ``cuda(device=0)`` uploads the immutable geometry and indexes once. Both
    versions accept batches and optional caller-owned output dictionaries.
    Invalid queries set valid=False, coordinates/arclengths to NaN, indices
    to -1 and nearest distances to infinity. Nearest ties select the lowest
    packed vertex/segment-start index. No trainer-specific labels are built.
    """
    def __init__(self, points, offsets, *, arclengths=None):
        points = _points(points, -1)
        offsets = _input(offsets, np.int64, -1)
        if offsets.ndim != 1 or not len(offsets):
            raise ValueError('offsets must have shape (C+1,)')
        arcs = np.empty(0, np.float32) if arclengths is None else _input(arclengths, np.float32, -1)
        if arcs.ndim != 1 or (arclengths is not None and arcs.shape != (len(points),)):
            raise ValueError('arclengths must have shape (N,)')
        self._native = _native.GeometryCurves(points, offsets, arcs)
        self.device, self._pid = -1, os.getpid()
        self.n_curves, self.n_points = len(offsets)-1, len(points)

    def cuda(self, device=0):
        """Return an independently owned CUDA index; no implicit query uploads."""
        self._ready()
        device = operator.index(device)
        if device < 0:
            raise ValueError('CUDA device must be nonnegative')
        if self.device >= 0:
            if self.device != device:
                raise ValueError('construct from the CPU index to change CUDA devices')
            return self
        if not _cuda.is_available():
            raise RuntimeError('CUDA geometry requires a CUDA-enabled build and device')
        result = object.__new__(type(self))
        result._native = _native.GeometryCudaCurves(self._native, device)
        result.device, result._pid = device, os.getpid()
        result.n_curves, result.n_points = self.n_curves, self.n_points
        return result

    def gather(self, indices, *, segments=False, out=None):
        """Fetch packed vertices or segment endpoints on the index's device.

        indices is (Q,) int64, such as flattened query_boxes results. Returns
        points (Q,3), arclengths (Q,), curve_ids and valid. With segments=True,
        points is (Q,2,3) and arclengths is (Q,2). Singleton curves return a
        degenerate segment; other final vertices are invalid segment starts.
        Negative/padded and out-of-range indices are invalid.
        """
        self._ready()
        ids = _input(indices, np.int64, self.device)
        if ids.ndim != 1:
            raise ValueError('indices must have shape (Q,)')
        n = ids.shape[0]
        point_shape, arc_shape = ((n,2,3), (n,2)) if segments else ((n,3), (n,))
        result, arrays = _outputs(out, dict(points=(point_shape,np.float32), arclengths=(arc_shape,np.float32),
            curve_ids=((n,),np.int64), valid=((n,),np.bool_)), self.device)
        _native._geometry_gather(self._native, _native_array(ids), bool(segments), *arrays)
        return result

    def interpolate(self, curve_ids, distances, *, out=None):
        """Evaluate arclengths, clipping finite values to curve endpoints.

        curve_ids and distances are (Q,) int64 and float32. valid indicates
        that the original distance was within the curve's arclength range.
        Returns points (Q,3) float32 and valid (Q,) bool.
        """
        self._ready()
        ids = _input(curve_ids, np.int64, self.device)
        s = _input(distances, np.float32, self.device)
        if ids.ndim != 1 or s.shape != ids.shape:
            raise ValueError('curve_ids and distances must have shape (Q,)')
        n = ids.shape[0]
        result, arrays = _outputs(out, dict(points=((n,3),np.float32), valid=((n,),np.bool_)), self.device)
        _native._geometry_interpolate(self._native, _native_array(ids), _native_array(s), *arrays)
        return result

    def intersect_planes(self, curve_ids, planes, *, intervals=None, direction=1, out=None):
        """First directed crossing of n·p=d for each (curve, plane) query.

        planes: (Q,4), normal XYZ followed by d; IDs: (Q,). intervals: optional
        (Q,2) arclength start/stop, clipped to the curve. Decreasing intervals
        traverse backwards. Default traverses the whole curve forwards.
        direction=1 selects negative→nonnegative signed distances, -1 selects
        positive→nonpositive, 0 selects either. A starting point on the plane
        and coplanar segments do not count as crossings.

        Returns points, arclengths, indices (packed segment-start index), valid.
        Combine plane queries in one batch to share the curve index.
        """
        self._ready()
        ids = _input(curve_ids, np.int64, self.device)
        p = _input(planes, np.float32, self.device)
        if ids.ndim != 1 or p.shape != (ids.shape[0],4):
            raise ValueError('expected (Q,) IDs and (Q,4) planes')
        n = ids.shape[0]
        spans = _empty((0,2), np.float32, self.device) if intervals is None else _input(intervals, np.float32, self.device)
        if intervals is not None and spans.shape != (n,2):
            raise ValueError('intervals must have shape (Q,2)')
        direction = operator.index(direction)
        result, arrays = _outputs(out, dict(points=((n,3),np.float32), arclengths=((n,),np.float32),
            indices=((n,),np.int64), valid=((n,),np.bool_)), self.device)
        _native._geometry_intersect(self._native, _native_array(ids), _native_array(p), _native_array(spans), direction, *arrays)
        return result

    def nearest(self, points, *, curve_ids=None, vertices=False, max_distance=np.inf, out=None):
        """Exact BVH nearest queries against segments, or vertices=True.

        Optional (Q,) curve_ids restrict each query to one curve; -1 searches
        all curves. max_distance is an inclusive Euclidean bound. Segment
        indices are packed start-vertex indices; singleton curves act as points.
        Candidates with unrepresentable float32 squared distances are invalid.
        Returns points, distances, arclengths, indices, curve_ids and valid.
        """
        self._ready()
        p = _points(points, self.device)
        n = p.shape[0]
        ids = _ids(curve_ids, n, self.device)
        result, arrays = _outputs(out, dict(points=((n,3),np.float32), distances=((n,),np.float32),
            arclengths=((n,),np.float32), indices=((n,),np.int64), curve_ids=((n,),np.int64),
            valid=((n,),np.bool_)), self.device)
        _native._geometry_nearest(self._native, _native_array(p), _native_array(ids), bool(vertices), float(max_distance), *arrays)
        return result


    def query_boxes(self, lower, upper, *, capacity, curve_ids=None, vertices=False, out=None):
        """Find primitive AABBs overlapping closed XYZ boxes (Q,3).

        Returns indices (Q,capacity), counts (Q,) int64 and valid (Q,) bool.
        Entries are packed vertex/segment-start indices in BVH traversal order,
        padded with -1. counts includes all matches; counts>capacity indicates
        truncation. Invalid/nonfinite boxes have valid=False and count=0.
        These are bounding-box candidates for subsequent geometric tests.
        Fixed capacity keeps allocation and overflow detection on the GPU.
        """
        self._ready()
        lower, upper = _points(lower, self.device), _points(upper, self.device)
        if lower.shape != upper.shape:
            raise ValueError('lower and upper must have the same shape')
        capacity = operator.index(capacity)
        if capacity < 1 or capacity > np.iinfo(np.int32).max:
            raise ValueError('capacity must be a positive int32')
        n = lower.shape[0]
        ids = _ids(curve_ids, n, self.device)
        result, arrays = _outputs(out, dict(indices=((n,capacity),np.int64), counts=((n,),np.int64),
            valid=((n,),np.bool_)), self.device)
        _native._geometry_boxes(self._native, _native_array(lower), _native_array(upper), _native_array(ids), bool(vertices), capacity, *arrays)
        return result


def transform_points(points, matrices, *, vectors=False, out=None):
    """Apply one (3,4) affine, or (Q,3,4) affines, to (Q,3) float32 points.

    Uses column-vector convention p'=R@p+t. vectors=True omits translation;
    it does not normalize or apply the inverse transpose for surface normals.
    Reuses the CPU/CUDA surface transform arithmetic. Inputs share a device.
    """
    device = _device(points)
    p = _points(points, device)
    matrices = _input(matrices, np.float32, device)
    if matrices.shape == (3,4):
        matrices = matrices.reshape(1,3,4)
    if matrices.ndim != 3 or matrices.shape[1:] != (3,4) or matrices.shape[0] not in (1,p.shape[0]):
        raise ValueError('matrices must have shape (3,4), (1,3,4), or (Q,3,4)')
    result, output = _output(out, p.shape, np.float32, device)
    _native._geometry_transform(_native_array(p), _native_array(matrices), bool(vectors), output)
    return result


class VoxelSet(_Resident):
    """Sparse binary occupancy from (N,3) continuous ZYX voxel coordinates.

    shape is (D,H,W), each dimension <=2**24. Points round to nearest-even
    cells. Nonfinite/out-of-bounds points and invalid batch IDs are ignored.
    batch_ids optionally separates points into ``batches`` independent grids.
    Construction and queries work on CPU or CUDA, inferred from the points.
    CUDA construction sorts keys on device and retains duplicates to avoid a
    count download; membership and dense outputs remain unique per cell.
    Memory scales with supplied points, not grid volume. Use to_dense only
    when a consumer requires a dense mask.
    """
    def __init__(self, points, shape, *, batch_ids=None, batches=1):
        self.device, self._pid = _device(points), os.getpid()
        p = _points(points, self.device)
        ids = _ids(batch_ids, p.shape[0], self.device)
        self.shape = tuple(operator.index(n) for n in shape)
        self.batches = operator.index(batches)
        if len(self.shape) != 3:
            raise ValueError('shape must be (D,H,W)')
        self._native = _native._geometry_voxelize(_native_array(p), _native_array(ids), self.shape, self.batches)

    def cuda(self, device=0):
        """Upload a CPU occupancy index once, retaining sparse storage."""
        self._ready()
        device = operator.index(device)
        if device < 0:
            raise ValueError('CUDA device must be nonnegative')
        if self.device >= 0:
            if self.device != device:
                raise ValueError('construct from CPU storage to change CUDA devices')
            return self
        if not _cuda.is_available():
            raise RuntimeError('CUDA geometry requires a CUDA-enabled build and device')
        result = object.__new__(type(self))
        result._native = _native._geometry_voxels_cuda(self._native, device)
        result.device, result._pid = device, os.getpid()
        result.shape, result.batches = self.shape, self.batches
        return result

    def contains(self, points, *, batch_ids=None, out=None):
        """Return (Q,) bool occupancy at nearest-even voxel cells."""
        self._ready()
        p = _points(points, self.device)
        ids = _ids(batch_ids, p.shape[0], self.device)
        result, output = _output(out, (p.shape[0],), np.bool_, self.device)
        _native._geometry_contains(self._native, _native_array(p), _native_array(ids), output)
        return result

    def sorted_keys(self, *, capacity=None, out=None):
        """Copy sorted int64 occupancy keys into caller-owned storage.

        Keys encode ``((batch*D+z)*H+y)*W+x``. CUDA storage can contain
        duplicate keys and INT64_MAX for invalid points; padding also uses
        INT64_MAX, which must never match a query. CPU storage is deduplicated.
        Capacity defaults to the stored count (nbytes/8), and cannot be smaller.
        A fixed capacity allows tensor consumers to compile stable input shapes
        without downloading the number of occupied cells. No storage is shared.
        """
        self._ready()
        count = self.nbytes // 8
        capacity = count if capacity is None else operator.index(capacity)
        if capacity < count:
            raise ValueError('key output capacity is too small')
        result, output = _output(out, (capacity,), np.int64, self.device)
        _native._geometry_keys(self._native, output)
        return result

    def to_dense(self, *, out=None):
        """Explicitly materialize (B,D,H,W) uint8 occupancy on this device."""
        self._ready()
        result, output = _output(out, (self.batches,*self.shape), np.uint8, self.device)
        _native._geometry_dense(self._native, output)
        return result
