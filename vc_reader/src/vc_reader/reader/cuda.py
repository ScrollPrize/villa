"""Optional CUDA backend for 3-D uint8 CT readers."""
import operator
import os
import numpy as np
from .. import _native
from ..cuda import DeviceArray, is_available, is_device_array, wrap
from ._sampling import affine32, origin3, points32, shape3
from ._indexing import parse_index


def _integer_points(points):
    pts = np.asarray(points)
    if pts.ndim == 0 or pts.shape[-1] != 3 or pts.dtype.kind != "i" or pts.dtype.itemsize not in (4, 8):
        raise TypeError("integer points must have shape (..., 3) and int32 or int64 dtype")
    return np.ascontiguousarray(pts.reshape(-1, 3), dtype=pts.dtype.newbyteorder("=")), pts.shape[:-1]


def _host_input(value):
    """Host copy of a CUDA array (an explicit synchronizing download), else NumPy."""
    return wrap(value).numpy() if is_device_array(value) else np.asarray(value)


def _device_input(value, device, dtype=None):
    """Contiguous DeviceArray on `device` from a CUDA array or host data."""
    from ..cuda import asarray
    array = wrap(value) if is_device_array(value) else asarray(np.ascontiguousarray(value, dtype=dtype), device=device)
    if array.device != device:
        raise ValueError("array is on another CUDA device")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise TypeError(f"expected {np.dtype(dtype)} CUDA data")
    return array


def _device_points(points, device, dtype):
    array = _device_input(points, device, dtype)
    if array.ndim == 0 or array.shape[-1] != 3:
        raise ValueError("points must have shape (..., 3)")
    if dtype is None and (array.dtype.kind != "i" or array.dtype.itemsize not in (4, 8)):
        raise TypeError("integer points must be int32 or int64")
    return array.reshape(-1, 3), array.shape[:-1]


class PointRequest:
    """Immutable point coordinates/planning retained for repeated device reads."""
    def __init__(self, reader, native, shape, dtype):
        self.reader, self._native, self.shape, self.dtype = reader, native, shape, dtype

    def read(self, *, source="auto", valid=False):
        self.reader._ready()
        native = self.reader._native.read_plan(self._native, source, valid)
        return self.reader._point_outputs(native, self.dtype, self.shape, valid)

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

    @staticmethod
    def _split(flat, shapes):
        result, offset = [], 0
        for shape in shapes:
            result.append(flat._view(shape, offset))
            offset += int(np.prod(shape))
        return result

    def _grids(self, grids, kind, order, dtype, source, valid=False, out=None):
        """Samples (and validity) as one flat device allocation plus per-crop views."""
        self._ready()
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype("uint8"), np.dtype("float32")) or (kind == 1 and order == 1 and dtype != np.float32):
            raise TypeError("CUDA crops support uint8 or float32; linear sampling requires float32")
        if source == "resident":
            raise ValueError("source='resident' needs device frames; use read_affine_batch with CUDA arrays")
        shapes, bases, origins, axes = zip(*grids) if grids else ((), (), (), ())
        total = sum(int(np.prod(s)) for s in shapes)
        if out is not None and not is_device_array(out):
            raise TypeError('out must be a CUDA array')
        destination = None if out is None else _device_input(out, self.device, dtype)
        if destination is not None and destination.size != total:
            raise ValueError('output size mismatch')
        native = self._native.grids(shapes, bases, [list(o) for o in origins], [np.asarray(a).ravel().tolist() for a in axes],
                                    kind, order, dtype == np.float32, source, valid,
                                    None if destination is None else destination._native)
        if destination is not None:
            masks = DeviceArray(native, (total,), np.uint8, self.device) if valid else None
            return destination.reshape(total), masks, shapes
        native, mask = native if valid else (native, None)
        flat = DeviceArray(native, (total,), dtype, self.device)
        masks = DeviceArray(mask, (total,), np.uint8, self.device) if valid else None
        return flat, masks, shapes

    def _crops(self, grids, kind, order, dtype, source, valid):
        flat, masks, shapes = self._grids(grids, kind, order, dtype, source, valid)
        crops = self._split(flat, shapes)
        return (crops, self._split(masks, shapes)) if valid else crops

    def __getitem__(self, key):
        """Read a basic NumPy-style selection without downloading the result."""
        return self.read(key)

    def read(self, selection, *, dtype=None, source="auto", valid=False):
        """Read a NumPy-style selection, keeping output and optional masks on device."""
        output = self.read_many([selection], dtype=dtype, source=source, valid=valid)
        return (output[0][0], output[1][0]) if valid else output[0]

    def read_many(self, selections, *, dtype=None, source="auto", valid=False):
        """Read slice selections as one batch with shared chunk uploads."""
        self._ready()
        plans = [parse_index(selection, self.reader.shape) for selection in selections]
        if not plans:
            return ([], []) if valid else []
        output = self.read_regions([(start, size) for start, size, _ in plans],
                                   dtype=dtype, source=source, valid=valid)
        shapes = [tuple(n for n, drop in zip(size, squeeze) if not drop) for _, size, squeeze in plans]
        def reshape(values):
            return [value.reshape(shape) for value, shape in zip(values, shapes)]
        return (reshape(output[0]), reshape(output[1])) if valid else reshape(output)

    def read_region(self, start, size, *, dtype=None, source="auto", valid=False):
        """Read an explicitly sized region, with fill outside the array."""
        out = self.read_regions([(start, size)], dtype=dtype, source=source, valid=valid)
        return (out[0][0], out[1][0]) if valid else out[0]

    def read_regions(self, regions, *, dtype=None, source="auto", valid=False):
        """valid=True also returns uint8 masks: 1 inside the volume (a padding mask)."""
        grids = []
        for start, size in regions:
            start = tuple(operator.index(n) for n in start)
            if len(start) != 3: raise ValueError("start must have three integers")
            grids.append((shape3(size), start, (0, 0, 0), np.eye(3, dtype=np.float32)))
        return self._crops(grids, 0, 0, np.uint8 if dtype is None else dtype, source, valid)

    def read_affine(self, shape, origin, axes, *, order=1, dtype=None, coordinate_origin=None, source="auto", valid=False):
        out = self.read_affine_many([(shape, origin, axes)], order=order, dtype=dtype,
                                    coordinate_origin=coordinate_origin, source=source, valid=valid)
        return (out[0][0], out[1][0]) if valid else out[0]

    def read_affine_many(self, grids, *, order=1, dtype=None, coordinate_origin=None, source="auto", valid=False):
        """valid=True also returns uint8 masks: 1 where every neighbour read lies in the volume."""
        grids = [affine32(*g, coordinate_origin) for g in grids]
        dtype = (np.float32 if order else np.uint8) if dtype is None else dtype
        return self._crops(grids, 1, order, dtype, source, valid)

    def read_affine_batch(self, shape, origins, axes, *, order=1, dtype=None, coordinate_origin=None,
                          source="auto", valid=False, out=None):
        """Crops of one shape as a single (B, *shape) array.

        origins (B, 3) and axes (B, 3, 3) may be host arrays or CUDA arrays
        (DeviceArray, torch, ...). source="resident" samples device frames from
        already cached bricks with no planning or coordinate transfer; bricks
        that are not resident read as fill and are invalid in the mask. Other
        sources copy device frames to the host to plan brick loads.

        out may be a contiguous CUDA array of the exact output shape/dtype.
        Writes complete before return and the supplied object is returned.
        Consumer-owned storage avoids exporting a native allocation or copying
        a temporary output. out requires source='auto', 'chunks', or 'cpu'.
        """
        self._ready()
        shape = shape3(shape)
        dtype = np.dtype((np.float32 if order else np.uint8) if dtype is None else dtype)
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        if source == "resident":
            if out is not None:
                raise ValueError("out requires a host-planned source")
            if dtype not in (np.dtype("uint8"), np.dtype("float32")) or (order == 1 and dtype != np.float32):
                raise TypeError("CUDA crops support uint8 or float32; linear sampling requires float32")
            origins = _device_input(origins, self.device)
            axes = _device_input(axes, self.device, np.float32)
            base = None if coordinate_origin is None else list(origin3(coordinate_origin))
            result = self._native.affine_resident(origins._native, axes._native, list(shape), base, order,
                                                  dtype == np.float32, valid)
            samples, masks = result if valid else (result, None)
            count = origins.shape[0]
            out = DeviceArray(samples, (count * int(np.prod(shape)),), dtype, self.device).reshape((count, *shape))
            if not valid: return out
            return out, DeviceArray(masks, (out.size,), np.uint8, self.device).reshape(out.shape)
        origins, axes = _host_input(origins), _host_input(axes)
        if origins.ndim != 2 or origins.shape[1] != 3 or axes.shape != (len(origins), 3, 3):
            raise ValueError("origins must be (B, 3) and axes (B, 3, 3)")
        grids = [affine32(shape, o, a, coordinate_origin) for o, a in zip(origins, axes)]
        output_shape = (len(grids), *shape)
        if out is not None and tuple(out.shape) != output_shape:
            raise ValueError(f'out must have shape {output_shape}')
        flat, masks, _ = self._grids(grids, 1, order, dtype, source, valid, out)
        result = flat.reshape(output_shape) if out is None else out
        return (result, masks.reshape(output_shape)) if valid else result

    def warm(self, regions):
        """Load the bricks of voxel boxes [(start, size), ...] without output."""
        self._ready()
        shapes, bases = [], []
        for start, size in regions:
            start = tuple(operator.index(n) for n in start)
            if len(start) != 3: raise ValueError("start must have three integers")
            shapes.append(shape3(size)); bases.append(start)
        self._native.warm(shapes, bases)

    def sample(self, points, *, order=1, dtype=np.float32, coordinate_origin=None, source="auto", valid=False):
        """points may be a CUDA array: source="resident" samples it in place from
        cached bricks (coordinates absolute float32 unless coordinate_origin is
        given); other sources copy it to the host first."""
        self._ready()
        if order not in (0, 1): raise ValueError("order must be 0 or 1")
        if order and np.dtype(dtype) != np.float32: raise TypeError("CUDA interpolation requires float32 output")
        out_dtype = np.float32 if order else np.uint8
        if source == "resident":
            flat, shape = _device_points(points, self.device, np.float32)
            base = [0, 0, 0] if coordinate_origin is None else list(origin3(coordinate_origin))
            return self._point_outputs(self._native.points_resident(flat._native, base, 2, order, valid), out_dtype, shape, valid)
        local, base, shape = points32(_host_input(points), coordinate_origin)
        return self._point_outputs(self._native.points(local, base, 2, order, source, valid), out_dtype, shape, valid)

    def _point_outputs(self, native, dtype, shape, valid):
        samples, mask = native if valid else (native, None)
        count = int(np.prod(shape))
        out = DeviceArray(samples, (count,), dtype, self.device).reshape(shape)
        return (out, DeviceArray(mask, (count,), np.uint8, self.device).reshape(shape)) if valid else out

    def read_points(self, points, *, source="auto", valid=False):
        self._ready()
        if source == "resident":
            flat, shape = _device_points(points, self.device, None)
            return self._point_outputs(self._native.points_resident(flat._native, [0, 0, 0], 3, 0, valid), np.uint8, shape, valid)
        flat, shape = _integer_points(_host_input(points))
        return self._point_outputs(self._native.points(flat, [0, 0, 0], 3, 0, source, valid), np.uint8, shape, valid)

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
