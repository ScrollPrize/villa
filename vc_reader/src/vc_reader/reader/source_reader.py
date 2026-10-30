"""NumPy indexing, conversion and pickling over the native source reader (``vcr::sources::SourceReader``)."""

import operator
import os

import numpy as np

from .. import _reader
from ..sources import Source
from ._indexing import parse_index
from .decoded_cache import DEFAULT_DECODE_BYTES, DecodedCache
from .zarr_reader import ZarrReader, _make_settings, _memory_cache, _open_options

_ZARR_OPTIONS = {'auth', 'cache_dir', 'cache_writes', 'cache_max_bytes', 'offline', 'memory_cache', 'compress_cache',
                 'allow_lossy_cache', 'threads', 'connections', 'read_mode'}


class _ZarrBackend:
    """The opened Zarr array as a :class:`ZarrReader`, for samplers that read it directly."""

    def __init__(self, native, options, cache):
        self._native = native
        self.fill_value = native.fill_value
        opened = native.zarr_array()
        settings, _ = _make_settings(opened.location, **{k: v for k, v in options.items() if k != 'memory_cache'})
        self.reader = ZarrReader(settings, cache, opened=opened)


class SourceReader:
    """Zarr, PNG/JPEG/TIFF and explicit-stack reader over local/HTTP/S3 storage.

    Index with integers, unit-step slices and ellipsis in ``axes`` order.
    Output preserves storage channel position and order; time selection removes
    the time axis. Encoding consumes a color channel axis and reports the encoded
    shape/axes/dtype. ``decoded_cache`` shares both memory budgets across readers
    in each process. Do not also supply byte-budget arguments when passing a
    shared cache.

    Resolved storage metadata and placement are exposed through ``resolved``.
    """

    def __init__(self, source, *, decoded_cache=None, decoded_cache_bytes=None, max_decode_bytes=None, **reader_options):
        if not isinstance(source, Source):
            raise TypeError('source must be a Source')
        if decoded_cache is not None:
            if not isinstance(decoded_cache, DecodedCache):
                raise TypeError('decoded_cache must be a DecodedCache')
            if decoded_cache_bytes is not None or max_decode_bytes is not None:
                raise ValueError('configure byte budgets on the supplied decoded_cache object')
        unknown = reader_options.keys() - _ZARR_OPTIONS
        if unknown:
            raise TypeError(f'unknown reader options: {sorted(unknown)}')
        self.source = source
        self._options = dict(reader_options)
        self._owns_cache = decoded_cache is None
        self.decoded_cache = decoded_cache if decoded_cache is not None else DecodedCache(
            DEFAULT_DECODE_BYTES if decoded_cache_bytes is None else decoded_cache_bytes,
            DEFAULT_DECODE_BYTES if max_decode_bytes is None else max_decode_bytes)
        self._closed = False
        self._start()
        self._native = _reader.SourceReader(source, self._options_for, self.decoded_cache)

    def _start(self):
        self._pid = os.getpid()
        self._memory = _memory_cache(self._options.get('memory_cache'))
        self._native = None
        self._backend_cache = None

    def _options_for(self, location):
        """The native options of one location, in this process."""
        settings, _ = _make_settings(location, **{k: v for k, v in self._options.items() if k != 'memory_cache'})
        return _open_options(settings, self._memory)

    def _engine(self):
        if self._closed:
            raise ValueError('reader is closed')
        if self._native is None:
            self._native = _reader.SourceReader(self.source, self._options_for, self.decoded_cache)
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._backend_cache = None
        return self._native

    @classmethod
    def _wrap(cls, native, parent):
        reader = object.__new__(cls)
        reader.source = native.source
        reader._options = parent._options
        reader._owns_cache = False
        reader.decoded_cache = parent.decoded_cache
        reader._closed = False
        reader._pid = os.getpid()
        reader._memory = parent._memory
        reader._native = native
        reader._backend_cache = None
        return reader

    @property
    def shape(self):
        return self._engine().shape

    @property
    def axes(self):
        return self._engine().axes

    @property
    def dtype(self):
        return np.dtype(self._engine().dtype)

    @property
    def spatial_axes(self):
        return self._engine().spatial_axes

    @property
    def fill_value(self):
        return self._engine().fill_value

    @property
    def palette(self):
        return self._engine().palette

    @property
    def registration(self):
        return self._engine().registration

    @property
    def resolved(self):
        return self._engine().resolved

    @property
    def companions(self):
        engine = self._engine()
        return {name: SourceReader._wrap(engine.companion(name), self) for name in engine.companion_names}

    @property
    def _is_direct(self):
        return self._engine().is_direct

    @property
    def _backend(self):
        engine = self._engine()
        if not engine.is_zarr:
            return None
        if self._backend_cache is None:
            self._backend_cache = _ZarrBackend(engine, self._options, self._memory)
        return self._backend_cache

    def __getitem__(self, key):
        """Read a NumPy-style selection of the selected, encoded source array."""
        engine = self._engine()
        shape, axes, spatial_axes = engine.shape, engine.axes, engine.spatial_axes
        start, size, squeeze = parse_index(key, shape)
        if not all(size):
            return np.empty(tuple(n for n, drop in zip(size, squeeze) if not drop), dtype=self.dtype)
        spatial = [axes.index(axis) for axis in spatial_axes]
        output = self.read(tuple(start[i] for i in spatial), tuple(size[i] for i in spatial))
        # read() already crops spatial axes; channel indexing uses its output
        # positions, after source selection and any color encoding.
        index = tuple((start[i] if squeeze[i] else slice(start[i], start[i] + size[i]))
                      if axis == 'c' else (0 if squeeze[i] else slice(None))
                      for i, axis in enumerate(axes))
        return output[index]

    def read(self, start, size):
        """Return one owned NumPy crop, padded in the encoded output domain."""
        return self._engine().read(list(map(operator.index, start)), list(map(operator.index, size)))

    def read_many(self, regions):
        return self._engine().read_many([(list(map(operator.index, start)), list(map(operator.index, size))) for start, size in regions])

    def stats(self):
        engine = self._engine()
        result = engine.stats()
        if engine.is_zarr:
            result['zarr'] = self._backend.reader.stats()
        else:
            del result['zarr_cache_bytes'], result['zarr_chunk_bytes']
        return result

    def close(self):
        if not self._closed:
            self._backend_cache = None
            if self._native is not None:
                self._native.close()
            self._native = None
            self._closed = True

    def __enter__(self):
        self._engine()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        return self.source, self.decoded_cache, self._owns_cache, self._closed, self._options

    def __setstate__(self, state):
        self.source, self.decoded_cache, self._owns_cache, self._closed, self._options = state
        self._start()
