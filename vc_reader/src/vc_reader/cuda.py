"""Owned CUDA arrays shared by the optional reader and augmentation backends.

Calls complete before returning. DLPack exports are consequently ready on any
consumer stream. External writers must finish before handing a shared array back
(or before calling numpy()). No implicit host download is performed.
"""
import math
import operator
import os
import numpy as np
from . import _native


def is_available():
    return bool(getattr(_native, "reader_cuda_available", lambda: False)())


class DeviceArray:
    """Contiguous device storage. Use numpy() for an explicit host copy.

    DLPack consumers (e.g. torch.from_dlpack) share storage and its lifetime.
    reshape() is a view; astype(float32) converts uint8 on device.
    """
    def __init__(self, native, shape, dtype, device=0):
        storage_type = getattr(_native, "CudaStorage", None)
        if storage_type is not None and isinstance(native, storage_type):
            # Native reader results own storage; kernels consume its array view.
            # This synchronous public adapter can export views through DLPack,
            # so keep destruction safe for consumers on external streams.
            native.export()
            native = _native._cuda_view(native.array(), shape, 0)
        self._native = native
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.device = operator.index(device)
        self._pid = os.getpid()

    def _ready(self):
        if self._pid != os.getpid():
            raise RuntimeError("CUDA arrays cannot be used after fork")

    @property
    def ndim(self): return len(self.shape)
    @property
    def size(self): return math.prod(self.shape)
    @property
    def nbytes(self): return self.size * self.dtype.itemsize

    def numpy(self):
        self._ready()
        out = np.empty(self.shape, self.dtype)
        _native._cuda_download(self._native, out.reshape(-1).view(np.uint8))
        return out

    def _view(self, shape, offset=0):
        self._ready()
        return DeviceArray(_native._cuda_view(self._native, shape, offset), shape, self.dtype, self.device)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        shape = tuple(operator.index(n) for n in shape)
        if shape.count(-1) > 1 or any(n < -1 for n in shape):
            raise ValueError("invalid reshape dimensions")
        if -1 in shape:
            known = math.prod(n for n in shape if n != -1)
            if not known or self.size % known:
                raise ValueError("cannot infer reshape dimension")
            shape = tuple(self.size // known if n == -1 else n for n in shape)
        if math.prod(shape) != self.size:
            raise ValueError("reshape changes array size")
        return self._view(shape)

    def astype(self, dtype):
        self._ready()
        dtype = np.dtype(dtype)
        if dtype == self.dtype:
            return self
        if self.dtype != np.uint8 or dtype != np.float32:
            raise TypeError("device conversion supports uint8 to float32")
        return DeviceArray(_native._cuda_cast(self._native), (self.size,), dtype, self.device).reshape(self.shape)

    def __dlpack__(self, *, stream=None, **kwargs):
        self._ready()
        return self._native.__dlpack__(stream=stream, **kwargs)

    def __dlpack_device__(self):
        self._ready()
        return (2, self.device)

    def __array__(self, *args, **kwargs):
        raise TypeError("use DeviceArray.numpy() for an explicit host copy")

    def __getstate__(self):
        raise TypeError("CUDA arrays cannot be pickled; create them in the worker")

    def __repr__(self):
        return f"DeviceArray(shape={self.shape}, dtype={self.dtype}, device={self.device})"


def is_device_array(value):
    """True for DeviceArray or any CUDA DLPack producer (e.g. a CUDA torch tensor)."""
    if isinstance(value, DeviceArray):
        return True
    describe = getattr(value, "__dlpack_device__", None)
    try:
        return describe is not None and describe()[0] == 2
    except Exception:
        return False


def wrap(value):
    """DeviceArray view sharing a CUDA DLPack producer's storage (no copy)."""
    if isinstance(value, DeviceArray):
        value._ready()
        return value
    shape, dtype, device = _native._cuda_describe(value)
    return DeviceArray(_native._cuda_view(value, shape, 0), shape, np.dtype(dtype), device)


def asarray(array, *, device=0):
    """Upload a native-endian NumPy array; no Torch/CuPy dependency."""
    if not is_available():
        raise RuntimeError("CUDA reader unavailable: build with VC_READER_ENABLE_CUDA=ON")
    if isinstance(array, DeviceArray):
        array._ready()
        if array.device != device:
            raise ValueError("device-to-device migration is not supported")
        return array
    array = np.asarray(array)
    if not array.dtype.isnative or array.dtype.kind not in "buif" or array.dtype.itemsize not in (1, 2, 4, 8):
        raise TypeError("unsupported device array dtype")
    array = np.require(array, requirements=["C", "A"])
    return DeviceArray(_native._cuda_upload(array, operator.index(device)), array.shape, array.dtype, device)
