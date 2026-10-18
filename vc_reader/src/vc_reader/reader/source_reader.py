"""Spatial source crops with exact encodings and explicitly owned decode caches."""

from contextlib import contextmanager
import itertools
import math
import operator
import os

import numpy as np

from ..sources import Source
from .decoded_cache import DecodedCache, DEFAULT_DECODE_BYTES
from .image_files import ImageFile


def _fill(value, dtype):
    if dtype.kind in 'iub':
        low, high = (0, 1) if dtype.kind == 'b' else (np.iinfo(dtype).min, np.iinfo(dtype).max)
        if not isinstance(value, (int, float)) or not low <= value <= high or int(value) != value:
            raise ValueError(f'fill value {value!r} is not representable in {dtype}')
    converted = np.asarray(value, dtype=dtype).item()
    if dtype.kind == 'f' and np.isfinite(value) and not np.isfinite(converted):
        raise ValueError(f'fill value {value!r} is not representable in {dtype}')
    return converted


class _Lookup:
    """Sorted, exact integer/color lookup; no label passes through float32."""

    def __init__(self, encoding, input_dtype, channels):
        if input_dtype.kind not in 'iub':
            raise ValueError('integer/color encoding requires integer decoded samples')
        self.encoding = encoding
        self.dtype = np.dtype(encoding.dtype)
        self.color = encoding.kind == 'color'
        low, high = (0, 1) if input_dtype.kind == 'b' else (np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        pairs = []
        for key, value in encoding.pairs:
            items = key if self.color else (key,)
            if self.color and len(items) != channels:
                raise ValueError('color lookup tuple length must match selected channels')
            if all(low <= item <= high for item in items):
                pairs.append((key, value))
        pairs.sort(key=lambda pair: pair[0])
        self.key_dtype = (np.dtype([(f'c{i}', input_dtype) for i in range(channels)])
                          if self.color else input_dtype)
        self.keys = np.array([p[0] for p in pairs], dtype=self.key_dtype)
        self.values = np.array([p[1] for p in pairs], dtype=self.dtype)

    def apply(self, array, channel_axis):
        if self.color:
            array = np.ascontiguousarray(np.moveaxis(array, channel_axis, -1))
            shape = array.shape[:-1]
            query = array.view(self.key_dtype).reshape(-1)
        else:
            shape = array.shape
            query = array.reshape(-1)
        fallback = self.encoding.fallback
        result = np.full(query.shape, 0 if fallback is None else fallback, dtype=self.dtype)
        if len(self.keys):
            indices = np.searchsorted(self.keys, query)
            np.minimum(indices, len(self.keys) - 1, out=indices)
            found = self.keys[indices] == query
            result[found] = self.values[indices[found]]
        else:
            found = np.zeros(query.shape, dtype=bool)
        if fallback is None and not found.all():
            unknown = query[np.flatnonzero(~found)[0]]
            raise ValueError(f'unknown encoding value {unknown!r}; declare a fallback to accept it')
        return result.reshape(shape)


class SourceReader:
    """Local PNG/JPEG/TIFF and explicit-stack reader.

    ``read(start, size)`` takes YX or ZYX coordinates. Output preserves storage
    channel position and order; time selection removes the time axis. Encoding
    consumes a color channel axis and reports the encoded shape/axes/dtype.
    ``decoded_cache`` shares both memory budgets across readers in each process.
    Do not also supply byte-budget arguments when passing a shared cache.

    Remote file caching and Zarr storage resolution are separate plan-02 work.
    Unsupported storage raises explicitly instead of guessing a relationship.
    """

    def __init__(self, source, *, decoded_cache=None, decoded_cache_bytes=None,
                 max_decode_bytes=None):
        if not isinstance(source, Source):
            raise TypeError('source must be a Source')
        if decoded_cache is not None:
            if not isinstance(decoded_cache, DecodedCache):
                raise TypeError('decoded_cache must be a DecodedCache')
            if decoded_cache_bytes is not None or max_decode_bytes is not None:
                raise ValueError('configure byte budgets on the supplied decoded_cache object')
        self.source = source
        self._owns_cache = decoded_cache is None
        self.decoded_cache = decoded_cache if decoded_cache is not None else DecodedCache(
            DEFAULT_DECODE_BYTES if decoded_cache_bytes is None else decoded_cache_bytes,
            DEFAULT_DECODE_BYTES if max_decode_bytes is None else max_decode_bytes)
        self._closed = False
        self._pid = os.getpid()
        self._files = {}
        self._lookup = None
        with self._errors():
            self._open()

    @contextmanager
    def _errors(self):
        try:
            yield
        except (ValueError, OSError, MemoryError, NotImplementedError) as error:
            if str(error).startswith(self.source.id + ':'):
                raise
            raise type(error)(f'{self.source.id}: {error}') from error

    def _open(self):
        source = self.source
        storage = source.storage
        self._stack = storage.format == 'stack'
        if storage.format not in (None, 'stack', 'png', 'jpeg', 'tiff'):
            raise NotImplementedError(f'Source.open for format {storage.format!r} is not implemented yet')
        if storage.location is None and not self._stack:
            raise ValueError('storage.location is required')
        if storage.alternatives:
            raise NotImplementedError('alternative storage origin resolution is not implemented yet')
        if source.companions:
            raise NotImplementedError('companion storage validation is not implemented yet')
        registration = source.registration
        if registration is not None and registration.declared_grid != storage.selection:
            raise NotImplementedError('registration for a changed selection requires storage-grid resolution')
        if registration is not None and getattr(registration, 'mode', None) == 'storage':
            raise NotImplementedError('storage-frame registration resolution is not implemented yet')
        self._locations = storage.locations if self._stack else (storage.location,)
        declared = source.axes
        if self._stack and declared is not None:
            if declared[0] != 'z':
                raise ValueError('an ordered file stack requires a leading z axis')
            declared = declared[1:]
        self._file_axes = declared
        first = self._file(0)
        info = first.info
        if self._stack and ('z' in info.axes or 't' in info.axes or first.prefix_shape):
            raise ValueError('stack entries must be individual 2D images, optionally with channels')
        self._raw_axes = (('z',) if self._stack else ()) + info.axes
        self._raw_shape = ((len(self._locations),) if self._stack else ()) + info.shape
        if source.shape is not None and source.shape != self._raw_shape:
            raise ValueError(f'declared shape {source.shape} contradicts storage shape {self._raw_shape}')
        if source.dtype is not None and np.dtype(source.dtype) != info.dtype:
            raise ValueError(f'declared dtype {source.dtype} contradicts storage dtype {info.dtype}')
        selection = storage.selection
        self._time = None
        if 't' in self._raw_axes:
            ntime = self._raw_shape[self._raw_axes.index('t')]
            if selection.time is None and ntime != 1:
                raise ValueError('time selection is required for a multi-time source')
            self._time = self._index(selection.time if selection.time is not None else 0, ntime, 'time')
        elif selection.time is not None:
            raise ValueError('time was selected but storage has no time axis')
        self._channels = None
        if 'c' in self._raw_axes:
            nchannel = self._raw_shape[self._raw_axes.index('c')]
            values = selection.channels if selection.channels is not None else range(nchannel)
            self._channels = tuple(self._index(v, nchannel, 'channel') for v in values)
        elif selection.channels is not None:
            raise ValueError('channels were selected but storage has no channel axis')
        self._selected_axes = tuple(a for a in self._raw_axes if a != 't')
        shape = tuple(len(self._channels) if a == 'c' else n
                      for a, n in zip(self._raw_axes, self._raw_shape) if a != 't')
        self.axes, self.shape, self.dtype = self._selected_axes, shape, info.dtype
        if source.encoding is not None:
            if source.encoding.kind == 'color' and self._channels is None:
                raise ValueError('color encoding requires a decoded color channel axis; palette PNG uses indices')
            table_bytes = len(source.encoding.pairs) * (info.dtype.itemsize * (len(self._channels or ()) or 1) + 8) * 2
            with self.decoded_cache.working(table_bytes):
                self._lookup = _Lookup(source.encoding, info.dtype, len(self._channels or ()))
            self.dtype = self._lookup.dtype
            if self._lookup.color:
                self.shape = tuple(n for a, n in zip(self.axes, self.shape) if a != 'c')
                self.axes = tuple(a for a in self.axes if a != 'c')
        self.spatial_axes = tuple(a for a in ('z', 'y', 'x') if a in self.axes)
        self.fill_value = _fill(0 if source.fill_value is None else source.fill_value, self.dtype)
        self.palette = first.palette
        # Missing placement remains unknown; raw reads do not invent alignment.
        self.registration = source.registration

    @staticmethod
    def _index(value, length, name):
        if isinstance(value, str):
            raise ValueError(f'{name} names are not present in this file layout; select by integer index')
        value = operator.index(value)
        if not 0 <= value < length:
            raise ValueError(f'{name} index {value} is outside [0, {length})')
        return value

    def _file(self, index):
        if index not in self._files:
            image = ImageFile(self._locations[index], None if self._stack else self.source.storage.format,
                              self.source.storage.selection, self._file_axes)
            if self._files:
                first = self._files[0].info
                if (image.info.shape, image.info.axes, image.info.dtype) != (first.shape, first.axes, first.dtype):
                    raise ValueError(f'stack plane {index} shape/dtype/axes differs from the first plane')
            self._files[index] = image
        return self._files[index]

    def _ready(self):
        if self._closed:
            raise ValueError('reader is closed')
        if self._pid != os.getpid() or not self._files:
            self._files = {}
            self._lookup = None
            self._pid = os.getpid()
            self._open()

    def __getattr__(self, name):
        if name in ('shape', 'axes', 'dtype', 'spatial_axes', 'fill_value', 'palette', 'registration'):
            with self._errors():
                self._ready()
            return self.__dict__[name]
        raise AttributeError(name)

    def read(self, start, size):
        """Return one owned NumPy crop, padded in the encoded output domain."""
        with self._errors():
            self._ready()
            start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
            if len(start) != len(self.spatial_axes) or len(size) != len(start) or any(n < 0 for n in size):
                raise ValueError(f'start/size must have {len(self.spatial_axes)} spatial entries and nonnegative sizes')
            origin = dict(zip(self.spatial_axes, start))
            requested = dict(zip(self.spatial_axes, size))
            dims = dict(zip(self._raw_axes, self._raw_shape))
            bounds = {a: (max(0, origin[a]), min(dims[a], origin[a] + requested[a])) for a in self.spatial_axes}
            shape = tuple(requested.get(a, n) for a, n in zip(self.axes, self.shape))
            output = np.full(shape, self.fill_value, dtype=self.dtype)
            if any(hi <= lo for lo, hi in bounds.values()):
                return output
            first = self._files[0]
            prefix_axes = (('z',) if self._stack else ()) + first.info.axes[:len(first.prefix_shape)]
            choices = []
            for axis in prefix_axes:
                if axis == 't':
                    choices.append((self._time,))
                elif axis == 'c':
                    if self._lookup is not None and self._lookup.color:
                        raise NotImplementedError('color encoding across separate TIFF pages is not implemented yet')
                    choices.append(self._channels)
                else:
                    choices.append(range(*bounds[axis]))
            for prefix in itertools.product(*choices):
                fixed = dict(zip(prefix_axes, prefix))
                image = self._file(fixed['z'] if self._stack else 0)
                index = 0
                for value, extent in zip(prefix[1:] if self._stack else prefix, image.prefix_shape):
                    index = index * extent + value
                slices, page_axes = [], []
                for axis in image.info.page_axes:
                    if axis == 't':
                        slices.append(self._time)
                    else:
                        slices.append(slice(*bounds[axis]) if axis in bounds else slice(None))
                        page_axes.append(axis)
                # Reserve selected-channel copies, lookup indices, masks and
                # encoded temporary outputs. The destination is caller-owned.
                count = math.prod(hi-lo for a, (lo, hi) in bounds.items() if a in page_axes)
                channels = len(self._channels or ()) or 1
                scratch = (count * (64 + channels * image.info.dtype.itemsize * 3)
                           if self._lookup is not None or 'c' in page_axes else 0)
                with image.page(index, self.decoded_cache, scratch) as array:
                    region = array[tuple(slices)]
                    if 'c' in page_axes:
                        region = np.take(region, self._channels, axis=page_axes.index('c'))
                    if self._lookup is not None:
                        region = self._lookup.apply(region, page_axes.index('c') if 'c' in page_axes else None)
                    target = []
                    for axis in self.axes:
                        if axis in fixed:
                            target.append(self._channels.index(fixed[axis]) if axis == 'c' else fixed[axis]-origin[axis])
                        elif axis in bounds:
                            lo, hi = bounds[axis]
                            target.append(slice(lo-origin[axis], hi-origin[axis]))
                        else:
                            target.append(slice(None))
                    output[tuple(target)] = region
            return output

    def read_many(self, regions):
        return [self.read(start, size) for start, size in regions]

    def close(self):
        if not self._closed:
            self._files = {}
            self._lookup = None
            if self._owns_cache:
                self.decoded_cache.clear()
            self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        return self.source, self.decoded_cache, self._owns_cache, self._closed

    def __setstate__(self, state):
        self.source, self.decoded_cache, self._owns_cache, self._closed = state
        self._pid = os.getpid()
        self._files = {}
        self._lookup = None
