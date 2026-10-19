"""Spatial source crops with exact encodings and explicitly owned decode caches."""

from contextlib import contextmanager
import itertools
import math
import operator
import os

import numpy as np

from ..errors import CacheMissError
from ..sources import Source, Selection, VolumeRegistration
from ..affine import Affine
from dataclasses import replace
from .encoded_files import EncodedFiles
from .source_resolution import ResolvedSource, resolve_registration, spatial_selection, matches_reference
from .decoded_cache import DecodedCache, DEFAULT_DECODE_BYTES
from .image_files import ImageFile
from ._indexing import parse_index


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
    """Zarr, PNG/JPEG/TIFF and explicit-stack reader over local/HTTP/S3 storage.

    Index with integers, unit-step slices and ellipsis in ``axes`` order.
    Output preserves storage
    channel position and order; time selection removes the time axis. Encoding
    consumes a color channel axis and reports the encoded shape/axes/dtype.
    ``decoded_cache`` shares both memory budgets across readers in each process.
    Do not also supply byte-budget arguments when passing a shared cache.

    Resolved storage metadata and placement are exposed through ``resolved``.
    """

    def __init__(self, source, *, decoded_cache=None, decoded_cache_bytes=None,
                 max_decode_bytes=None, **reader_options):
        if not isinstance(source, Source):
            raise TypeError('source must be a Source')
        if decoded_cache is not None:
            if not isinstance(decoded_cache, DecodedCache):
                raise TypeError('decoded_cache must be a DecodedCache')
            if decoded_cache_bytes is not None or max_decode_bytes is not None:
                raise ValueError('configure byte budgets on the supplied decoded_cache object')
        self.source = source
        allowed = {'auth', 'cache_dir', 'cache_writes', 'cache_max_bytes', 'offline', 'memory_cache',
                   'compress_cache', 'allow_lossy_cache', 'threads', 'connections', 'read_mode', 'prefetcher'}
        unknown = reader_options.keys() - allowed
        if unknown:
            raise TypeError(f'unknown reader options: {sorted(unknown)}')
        self._options = dict(reader_options)
        if self._options.get('threads') is None:
            self._options['threads'] = 1
        self._backend = None
        self.companions = {}
        self._encoded = EncodedFiles(**{k: v for k, v in reader_options.items()
                                       if k in {'auth', 'cache_dir', 'cache_writes', 'cache_max_bytes', 'offline'}})
        self._owns_cache = decoded_cache is None
        self.decoded_cache = decoded_cache if decoded_cache is not None else DecodedCache(
            DEFAULT_DECODE_BYTES if decoded_cache_bytes is None else decoded_cache_bytes,
            DEFAULT_DECODE_BYTES if max_decode_bytes is None else max_decode_bytes)
        self._closed = False
        self._pid = os.getpid()
        self._files = {}
        self._lookup = None
        try:
            with self._errors():
                self._open()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _errors(self):
        try:
            yield
        except (ValueError, OSError, MemoryError, RuntimeError, KeyError, IndexError) as error:
            if str(error).startswith(self.source.id + ':'):
                raise
            raise type(error)(f'{self.source.id}: {error}') from error

    def _open(self):
        source = self.source
        storage = source.storage
        self._stack = storage.format == 'stack'
        self._is_zarr = storage.format == 'zarr' or (storage.format is None and
                        storage.location is not None and storage.location.rstrip('/').endswith('.zarr'))
        if storage.location is None and not self._stack:
            raise ValueError('storage.location is required')
        self._locations = storage.locations if self._stack else (storage.location,)
        self._file_axes = source.axes
        self._image_selection = storage.selection
        if self._stack and source.axes is not None:
            if source.axes[0] != 'z':
                raise ValueError('an ordered file stack requires a leading z axis')
            self._file_axes = source.axes[1:]
        failures = []
        for location in (storage.location, *storage.alternatives):
            try:
                if self._is_zarr:
                    from .zarr_source import ZarrSource
                    selected = replace(source, storage=replace(storage, location=location, alternatives=()))
                    self._backend = ZarrSource(selected, self._options)
                    info = self._backend.info
                else:
                    if not self._stack:
                        self._locations = (location,)
                    self._files = {}
                    if storage.selection.corresponding is not None:
                        self._image_selection = self._corresponding_image()
                    first = self._file(0)
                    info = first.info
                    if self._stack and ('z' in info.axes or 't' in info.axes):
                        raise ValueError('stack entries must be individual 2D images, optionally with channels')
                break
            except (FileNotFoundError, CacheMissError) as error:
                failures.append(error)
        else:
            raise failures[-1]
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
            self._time = self._index(selection.time if selection.time is not None else 0, ntime, 'time', getattr(self._backend or first, 'time_names', ()))
        elif selection.time is not None:
            raise ValueError('time was selected but storage has no time axis')
        self._channels = None
        if 'c' in self._raw_axes:
            nchannel = self._raw_shape[self._raw_axes.index('c')]
            values = selection.channels if selection.channels is not None else range(nchannel)
            self._channels = tuple(self._index(v, nchannel, 'channel', getattr(self._backend or first, 'channel_names', ())) for v in values)
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
        if source.fill_value is not None:
            fill = source.fill_value
        elif self._is_zarr:
            fill = self._backend.fill_value
            if self._lookup is not None:
                values = np.full((len(self._channels),) if self._lookup.color else (1,), fill, dtype=info.dtype)
                fill = self._lookup.apply(values, 0 if self._lookup.color else None).item()
        else:
            fill = 0
        self.fill_value = _fill(fill, self.dtype)
        self.palette = None if self._is_zarr else first.palette
        if self._is_zarr:
            self.registration = self._backend.registration
            self.resolved = replace(self._backend.resolved, axes=self.axes, shape=self.shape, dtype=str(self.dtype))
        else:
            current = self._image_selection
            original = source.registration.declared_grid if source.registration is not None else None
            placement_source = source
            old_transform = old_units = None
            if original is not None and spatial_selection(original) != spatial_selection(current):
                old = ImageFile(self._locations[0], first.format, original, self._file_axes, self._encoded)
                if old.info.series != first.info.series:
                    raise ValueError('selected TIFF series has no declared connection to the original series; supply placement')
                if old.info.level == first.info.level:
                    placement_source = replace(source, registration=replace(source.registration, declared_grid=current))
                old_transform, old_units = old.storage_from_array, old.storage_units
            relative = None
            if isinstance(source.registration, VolumeRegistration) and source.registration.origin == 'catalog-fallback' and (
                    first.storage_from_array is None and old_transform is None and original is not None and
                    original.level is not None and current.level is not None):
                relative = Affine.scale(2**(current.level-original.level))
            self.registration = resolve_registration(placement_source, current, first.storage_from_array, first.storage_units,
                                                       old_transform, old_units, relative)
            self.resolved = ResolvedSource(self._locations[0], current, self.axes, self.shape, str(self.dtype),
                                           self.registration, first.storage_from_array, first.storage_units)
        self._open_companions()

    def _open_companions(self):
        for companion in self.source.companions:
            reader = SourceReader(companion.source, decoded_cache=self.decoded_cache, **self._options)
            self.companions[companion.name] = reader
            if 'c' in reader.axes and reader.shape[reader.axes.index('c')] != 1:
                raise ValueError(f'companion {companion.name} must have one channel')
            if companion.same_grid:
                parent_shape = {a: n for a, n in zip(self.axes, self.shape) if a in 'zyx'}
                child_shape = {a: n for a, n in zip(reader.axes, reader.shape) if a in 'zyx'}
                if parent_shape != child_shape:
                    raise ValueError(f'companion {companion.name} same_grid shape differs from its parent')
                a, b = self.resolved, reader.resolved
                if a.storage_from_array is not None and b.storage_from_array is not None and (
                        a.storage_units != b.storage_units or a.storage_from_array.matrix != b.storage_from_array.matrix):
                    raise ValueError(f'companion {companion.name} storage metadata contradicts same_grid binding')
                reader.registration = self.registration
                reader.resolved = replace(reader.resolved, registration=self.registration)

    @staticmethod
    def _index(value, length, name, names=()):
        if isinstance(value, str):
            matches = [i for i, candidate in enumerate(names) if candidate == value]
            if len(matches) != 1:
                raise ValueError(f'{name} name {value!r} is absent or ambiguous; select by integer index')
            value = matches[0]
        value = operator.index(value)
        if not 0 <= value < length:
            raise ValueError(f'{name} index {value} is outside [0, {length})')
        return value

    def _corresponding_image(self):
        request = self.source.storage.selection.corresponding
        reg = self.source.registration
        if self._stack or not isinstance(reg, VolumeRegistration):
            raise ValueError('corresponding image levels require a registered TIFF pyramid')
        selection = replace(self.source.storage.selection, corresponding=None, level=0)
        base = ImageFile(self._locations[0], self.source.storage.format, selection, self._file_axes, self._encoded)
        if base.format != 'tiff':
            raise ValueError('corresponding levels require a TIFF pyramid or Zarr group')
        original = reg.declared_grid
        if original is None or original.corresponding is not None:
            raise ValueError('corresponding levels need an explicitly placed base grid')
        old = ImageFile(self._locations[0], base.format, original, self._file_axes, self._encoded)
        matches = []
        for level in range(base.level_count):
            selected = replace(selection, level=level)
            image = ImageFile(self._locations[0], base.format, selected, self._file_axes, self._encoded)
            relative = None
            if image.storage_from_array is None and old.storage_from_array is None:
                if reg.origin != 'catalog-fallback' or original.level is None:
                    continue
                relative = Affine.scale(2**(level-original.level))
            placement = resolve_registration(self.source, selected, image.storage_from_array, image.storage_units,
                                               old.storage_from_array, old.storage_units, relative)
            if matches_reference(self.source, placement, request.reference, image.info.axes):
                matches.append(selected)
        if len(matches) != 1:
            raise ValueError(f'requested reference level {request.reference.level} has {len(matches)} exact TIFF levels; select a level explicitly')
        return matches[0]

    def _file(self, index):
        if index not in self._files:
            image = ImageFile(self._locations[index], None if self._stack else self.source.storage.format,
                              self._image_selection, self._file_axes, self._encoded)
            if self._files:
                first = self._files[0].info
                if (image.info.shape, image.info.axes, image.info.dtype) != (first.shape, first.axes, first.dtype):
                    raise ValueError(f'stack plane {index} shape/dtype/axes differs from the first plane')
            self._files[index] = image
        return self._files[index]

    def _ready(self):
        if self._closed:
            raise ValueError('reader is closed')
        if self._pid != os.getpid() or (not self._files and self._backend is None):
            if self._backend is not None:
                self._backend.close()
            for child in self.companions.values():
                child.close()
            self._backend = None
            self.companions = {}
            self._files = {}
            self._lookup = None
            self._pid = os.getpid()
            self._open()

    def __getattr__(self, name):
        if name in ('shape', 'axes', 'dtype', 'spatial_axes', 'fill_value', 'palette', 'registration', 'resolved'):
            with self._errors():
                self._ready()
            return self.__dict__[name]
        raise AttributeError(name)

    def __getitem__(self, key):
        """Read a NumPy-style selection of the selected, encoded source array."""
        self._ready()
        start, size, squeeze = parse_index(key, self.shape)
        if not all(size):
            return np.empty(tuple(n for n, drop in zip(size, squeeze) if not drop), dtype=self.dtype)
        spatial = [self.axes.index(axis) for axis in self.spatial_axes]
        output = self.read(tuple(start[i] for i in spatial), tuple(size[i] for i in spatial))
        # read() already crops spatial axes; channel indexing uses its output
        # positions, after source selection and any color encoding.
        index = tuple((start[i] if squeeze[i] else slice(start[i], start[i] + size[i]))
                      if axis == 'c' else (0 if squeeze[i] else slice(None))
                      for i, axis in enumerate(self.axes))
        return output[index]

    def read(self, start, size):
        """Return one owned NumPy crop, padded in the encoded output domain."""
        with self._errors():
            self._ready()
            start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
            if len(start) != len(self.spatial_axes) or len(size) != len(start) or any(n < 0 for n in size):
                raise ValueError(f'start/size must have {len(self.spatial_axes)} spatial entries and nonnegative sizes')
            if (self._is_zarr and self.axes == self.spatial_axes and self._lookup is None
                    and self.fill_value == self._backend.fill_value):
                return self._backend.reader.read_region(start, size)
            origin = dict(zip(self.spatial_axes, start))
            requested = dict(zip(self.spatial_axes, size))
            dims = dict(zip(self._raw_axes, self._raw_shape))
            bounds = {a: (max(0, origin[a]), min(dims[a], origin[a] + requested[a])) for a in self.spatial_axes}
            shape = tuple(requested.get(a, n) for a, n in zip(self.axes, self.shape))
            output = np.full(shape, self.fill_value, dtype=self.dtype)
            if any(hi <= lo for lo, hi in bounds.values()):
                return output
            if self._is_zarr:
                count = math.prod(hi-lo for lo, hi in bounds.values())
                channels = len(self._channels or ()) or 1
                scratch = count * (channels*self._backend.info.dtype.itemsize*3 +
                                   (64 + self.dtype.itemsize if self._lookup is not None else 0))
                with self.decoded_cache.working(scratch):
                    self._read_zarr(bounds, origin, output)
                return output
            first = self._files[0]
            prefix_axes = (('z',) if self._stack else ()) + first.info.axes[:len(first.prefix_shape)]
            choices = []
            for axis in prefix_axes:
                if axis == 't':
                    choices.append((self._time,))
                elif axis == 'c':
                    choices.append((None,) if self._lookup is not None and self._lookup.color else self._channels)
                else:
                    choices.append(range(*bounds[axis]))
            for prefix in itertools.product(*choices):
                fixed = dict(zip(prefix_axes, prefix))
                image = self._file(fixed['z'] if self._stack else 0)
                index = 0
                for value, extent in zip(prefix[1:] if self._stack else prefix, image.prefix_shape):
                    index = index * extent + (0 if value is None else value)
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
                target = []
                for axis in self.axes:
                    if axis in fixed:
                        target.append(self._channels.index(fixed[axis]) if axis == 'c' else fixed[axis]-origin[axis])
                    elif axis in bounds:
                        lo, hi = bounds[axis]
                        target.append(slice(lo-origin[axis], hi-origin[axis]))
                    else:
                        target.append(slice(None))
                if 'c' in fixed and fixed['c'] is None:
                    page_shape = tuple((bounds[a][1]-bounds[a][0]) for a in page_axes)
                    combined_bytes = math.prod(page_shape)*len(self._channels)*image.info.dtype.itemsize
                    with self.decoded_cache.working(combined_bytes + scratch):
                        combined = np.empty((len(self._channels), *page_shape), dtype=image.info.dtype)
                        for out_channel, channel in enumerate(self._channels):
                            ordinal = 0
                            values = [channel if a == 'c' else fixed[a] for a in prefix_axes]
                            for value, extent in zip(values[1:] if self._stack else values, image.prefix_shape):
                                ordinal = ordinal*extent + value
                            with image.region(ordinal, tuple(slices), self.decoded_cache, 0) as region:
                                combined[out_channel] = region
                        output[tuple(target)] = self._lookup.apply(combined, 0)
                else:
                    with image.region(index, tuple(slices), self.decoded_cache, scratch) as region:
                        if 'c' in page_axes:
                            region = np.take(region, self._channels, axis=page_axes.index('c'))
                        if self._lookup is not None:
                            region = self._lookup.apply(region, page_axes.index('c') if 'c' in page_axes else None)
                        output[tuple(target)] = region
            return output

    def _read_zarr(self, bounds, origin, output):
        # Read only selected channels/time. ZarrReader supplies source chunk
        # planning, authentication, missing-chunk fill and cache ownership.
        starts, sizes = [], []
        for axis, length in zip(self._raw_axes, self._raw_shape):
            if axis in bounds:
                lo, hi = bounds[axis]
            elif axis == 't':
                lo, hi = self._time, self._time+1
            elif axis == 'c':
                lo, hi = 0, 1
            starts.append(lo)
            sizes.append(hi-lo)
        regions = []
        for channel in self._channels or (None,):
            start = list(starts)
            if channel is not None:
                start[self._raw_axes.index('c')] = channel
            regions.append((start, sizes))
        arrays = self._backend.reader.read_regions(regions)
        raw = np.concatenate(arrays, axis=self._raw_axes.index('c')) if len(arrays) > 1 else arrays[0]
        if 't' in self._raw_axes:
            raw = np.squeeze(raw, axis=self._raw_axes.index('t'))
        if self._lookup is not None:
            raw = self._lookup.apply(raw, self._selected_axes.index('c') if 'c' in self._selected_axes else None)
        target = tuple(slice(bounds[a][0]-origin[a], bounds[a][1]-origin[a]) if a in bounds else slice(None)
                       for a in self.axes)
        output[target] = raw

    def read_many(self, regions):
        with self._errors():
            self._ready()
            if (self._is_zarr and self.axes == self.spatial_axes and self._lookup is None
                    and self.fill_value == self._backend.fill_value):
                return self._backend.reader.read_regions(regions)
            return [self.read(start, size) for start, size in regions]

    def stats(self):
        self._ready()
        result = dict(decoded_cache=self.decoded_cache.stats(), remote_bytes=self._encoded.bytes_read,
                      encoded_cache_hits=self._encoded.cache_hits,
                      image_payload_bytes=sum(f.bytes_read for f in self._files.values()))
        if self._backend is not None:
            result['zarr'] = self._backend.reader.stats()
            cache = self._backend.reader.memory_cache
            result['zarr_cache_bytes'] = cache.nbytes if cache is not None else 0
            result['zarr_chunk_bytes'] = self._backend.reader.meta.chunk_nbytes
        return result

    def close(self):
        if not self._closed:
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            for reader in self.companions.values():
                reader.close()
            self.companions = {}
            self._files = {}
            self._lookup = None
            self._encoded.close()
            if self._owns_cache:
                self.decoded_cache.clear()
            self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        return self.source, self.decoded_cache, self._owns_cache, self._closed, self._options

    def __setstate__(self, state):
        self.source, self.decoded_cache, self._owns_cache, self._closed, self._options = state
        self._pid = os.getpid()
        self._files = {}
        self._lookup = self._backend = None
        self.companions = {}
        self._encoded = EncodedFiles(**{k: v for k, v in self._options.items()
                                       if k in {'auth', 'cache_dir', 'cache_writes', 'cache_max_bytes', 'offline'}})
