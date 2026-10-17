"""Optional CUDA backend for 3-D uint8 CT readers."""
import operator
import os
import numpy as np
from .. import _native
from ..cuda import DeviceArray, is_available
from ._sampling import affine32, points32, shape3


def _integer_points(points):
    pts = np.asarray(points)
    if pts.ndim == 0 or pts.shape[-1] != 3 or pts.dtype.kind != "i" or pts.dtype.itemsize not in (4, 8):
        raise TypeError("integer points must have shape (..., 3) and int32 or int64 dtype")
    return np.ascontiguousarray(pts.reshape(-1, 3), dtype=pts.dtype.newbyteorder("=")), pts.shape[:-1]


class PointRequest:
    """Immutable point coordinates/planning retained for repeated device reads."""
    def __init__(self, reader, native, shape, dtype):
        self.reader, self._native, self.shape, self.dtype = reader, native, shape, dtype

    def read(self, *, source="auto"):
        self.reader._ready()
        native = self.reader._native.read_plan(self._native, source)
        return DeviceArray(native, (int(np.prod(self.shape)),), self.dtype, self.reader.device).reshape(self.shape)

    def __getstate__(self):
        raise TypeError("CUDA point requests cannot be pickled")


class CudaReader:
    """Create with reader.cuda(). Outputs stay on device.

    source='auto' uses CPU crop/upload when fewer bytes must cross to the GPU,
    and falls back to CPU sampling/upload if the request exceeds cache_bytes.
    source='chunks' requires device sampling and a sufficient chunk budget.
    source='cpu' explicitly samples/crops on CPU and uploads the result.
    The cache is private to this reader adapter and device; clear_cache() drops
    resident source chunks. Arrays retain independent ownership after close().
    """
    def __init__(self, reader, *, cache_bytes=512 << 20, device=0):
        reader._ready()
        if reader.ndim != 3 or reader.dtype != np.uint8:
            raise TypeError("CUDA sampling supports 3-D uint8 CT")
        cache_bytes, device = operator.index(cache_bytes), operator.index(device)
        if cache_bytes < 0 or device < 0:
            raise ValueError("cache_bytes and device must be nonnegative")
        if not is_available():
            raise RuntimeError("CUDA reader unavailable: build with VC_READER_ENABLE_CUDA=ON and use a CUDA device")
        self.reader, self.device, self._pid = reader, device, os.getpid()
        self._native = _native.CudaReader(reader._array, cache_bytes, device)

    def _ready(self):
        if self._pid != os.getpid():
            raise RuntimeError("CUDA reader cannot be used after fork; create it in the worker")
        if self._native is None:
            raise ValueError("CUDA reader is closed")
        self.reader._ready()

    def clear_cache(self):
        self._ready()
        self._native.clear()

    @property
    def stats(self):
        self._ready()
        return dict(zip(("cache_bytes", "cache_budget", "cache_hits", "chunk_uploads", "uploaded_bytes", "cpu_routes", "gpu_routes"), self._native.stats()))

    def close(self):
        if self._pid == os.getpid() and self._native is not None:
            self._native.clear()
            self._native = None

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
    def __getstate__(self):
        raise TypeError("CUDA readers cannot be pickled; construct one in each worker")

    def _grids(self, grids, kind, order, dtype, source):
        self._ready()
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("uint8"), np.dtype("float32")) or (kind == 1 and order == 1 and dtype != np.float32):
            raise TypeError("CUDA crops support uint8 or float32; linear sampling requires float32")
        shapes, bases, origins, axes = zip(*grids) if grids else ((), (), (), ())
        native = self._native.grids(shapes, bases, [list(o) for o in origins], [np.asarray(a).ravel().tolist() for a in axes], kind, order, dtype == np.float32, source)
        total = sum(int(np.prod(s)) for s in shapes)
        flat = DeviceArray(native, (total,), dtype, self.device)
        result, offset = [], 0
        for shape in shapes:
            result.append(flat._view(shape, offset))
            offset += int(np.prod(shape))
        return result

    def read(self, start, size, *, dtype=None, source="auto"):
        return self.read_many([(start, size)], dtype=dtype, source=source)[0]

    def read_many(self, regions, *, dtype=None, source="auto"):
        grids = []
        for start, size in regions:
            start = tuple(operator.index(n) for n in start)
            if len(start) != 3: raise ValueError("start must have three integers")
            grids.append((shape3(size), start, (0, 0, 0), np.eye(3, dtype=np.float32)))
        return self._grids(grids, 0, 0, np.uint8 if dtype is None else dtype, source)

    def read_affine(self, shape, origin, axes, *, order=1, dtype=None, coordinate_origin=None, source="auto"):
        grid = affine32(shape, origin, axes, coordinate_origin)
        dtype = (np.float32 if order else np.uint8) if dtype is None else dtype
        return self._grids([grid], 1, order, dtype, source)[0]

    def read_affine_many(self, grids, *, order=1, dtype=None, coordinate_origin=None, source="auto"):
        grids = [affine32(*g, coordinate_origin) for g in grids]
        dtype = (np.float32 if order else np.uint8) if dtype is None else dtype
        return self._grids(grids, 1, order, dtype, source)

    def sample(self, points, *, order=1, dtype=np.float32, coordinate_origin=None, source="auto"):
        self._ready()
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        if order and np.dtype(dtype) != np.float32: raise TypeError("CUDA interpolation requires float32 output")
        local, base, shape = points32(points, coordinate_origin)
        native = self._native.points(local, base, 2, order, source)
        return DeviceArray(native, (len(local),), np.float32 if order else np.uint8, self.device).reshape(shape)

    def read_points(self, points, *, source="auto"):
        self._ready()
        flat, shape = _integer_points(points)
        native = self._native.points(flat, [0, 0, 0], 3, 0, source)
        return DeviceArray(native, (len(flat),), np.uint8, self.device).reshape(shape)

    def prepare_points(self, points, *, integer=False, order=1, coordinate_origin=None):
        """Plan/upload coordinates once; request.read() reuses them on device.

        Source chunks still obey the cache budget and may be evicted between
        reads. Caller mutations of points do not change the prepared request.
        """
        self._ready()
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        if integer:
            if coordinate_origin is not None: raise ValueError("integer points use absolute coordinates")
            local, shape = _integer_points(points)
            base, kind, order, dtype = [0, 0, 0], 3, 0, np.uint8
        else:
            local, base, shape = points32(points, coordinate_origin)
            kind, dtype = 2, np.float32 if order else np.uint8
        plan = self._native.prepare_points(local, base, kind, order)
        return PointRequest(self, plan, shape, dtype)
