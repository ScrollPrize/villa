"""Aligned, bounded CPU/CUDA reads in a caller-selected reference array grid.

Importing this module does not import NumPy, native extensions, or PyTorch.
"""
from dataclasses import asdict, replace
import math
import operator
import os
from collections import OrderedDict
from threading import RLock

from .affine import Affine
from .catalog import VolumeLevel
from .sources import Source, VolumeRegistration, SampleResult

# Forked readers may contain native locks held by vanished parent threads.
_INHERITED = []


def open_sample(reference, targets=None, *, backend='cpu', catalog=None, **reader_options):
    """Open a reusable aligned reader; reads return unbatched CZYX SampleResults.

    ``sampling_cache_bytes`` (64 MiB) bounds retained CPU/device tiles;
    ``sampling_work_bytes`` (64 MiB) bounds tiles held by one native request.
    ``plan_cache_bytes`` (8 MiB) bounds reusable affine plans. ``device`` selects
    the CUDA device. Other options are passed to Source.open().
    """
    return SampleReader(reference, targets, backend=backend, catalog=catalog, **reader_options)


def _frame(reg):
    return reg.sample_id, reg.volume_id


def _placement(reg):
    if not isinstance(reg, VolumeRegistration):
        raise ValueError('volume registration is required; surface targets require plan 04 methods')
    if reg.plane is not None:
        p = reg.plane
        affine = Affine.from_rows(tuple((x, y, n, o) for x, y, n, o in
                                       zip(p.basis_x, p.basis_y, p.thickness, p.origin)))
    else:
        affine = reg.affine
    if affine is None or reg.mode != 'array':
        raise ValueError('complete resolved array placement is required')
    inverse = affine.inverse()
    if not all(math.isfinite(v) for row in (*affine.matrix, *inverse.matrix) for v in row):
        raise ValueError('registration and its inverse must be finite')
    return affine


def _region(start, size):
    start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
    if len(start) != 3 or len(size) != 3 or any(n < 0 or n > 2**24 for n in size):
        raise ValueError('start/size must be ZYX triples, with sizes in [0, 2**24]')
    if any(abs(n) > 2**40 for n in start):
        raise ValueError('crop starts must be within +/- 2**40')
    return start, size


class _Bounded:
    def __init__(self, limit):
        self.limit = limit
        self.items = OrderedDict()
        self.bytes = self.hits = 0

    def get(self, key):
        item = self.items.get(key)
        if item is not None:
            self.items.move_to_end(key)
            self.hits += 1
            return item[0]

    def put(self, key, value, size):
        if size > self.limit:
            return
        if key in self.items:
            self.bytes -= self.items.pop(key)[1]
        while self.items and self.bytes + size > self.limit:
            self.bytes -= self.items.popitem(last=False)[1][1]
        self.items[key] = value, size
        self.bytes += size


class _SampleTarget:
    """One aligned target, indexed in CZYX order on its reader's reference grid."""

    def __init__(self, reader, name):
        self._reader, self._name = reader, name

    @property
    def shape(self):
        self._reader._ready()
        return (self._reader._targets[self._name]['channels'], *self._reader.shape)

    def __getitem__(self, key):
        from .reader._indexing import parse_index
        reader = self._reader
        reader._ready()
        with reader._lock:
            start, size, squeeze = parse_index(key, self.shape)
            target = reader._targets[self._name]
            # Transform complete geometric components before selecting channels:
            # one output component may depend on every input component.
            spatial_size = tuple(size[1:]) if size[0] else (0, 0, 0)
            array, _ = reader._sample_many(target, [(tuple(start[1:]), spatial_size)])[0]
            shape = tuple(n for n, drop in zip(size, squeeze) if not drop)
            if reader.backend == 'cuda':
                offset = start[0] * math.prod(size[1:]) if all(size) else 0
                return array._view(shape, offset)
            array = array[start[0]:start[0] + size[0]]
            array = array.reshape(shape)
            return array[()] if not shape else array


class SampleReader:
    def __init__(self, reference, targets=None, *, backend='cpu', catalog=None,
                 sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20,
                 plan_cache_bytes=8 << 20, device=0, **reader_options):
        import numpy as np
        from . import _native
        from .reader import DecodedCache, MemoryCache
        if backend not in ('cpu', 'cuda'):
            raise ValueError("backend must be 'cpu' or 'cuda'")
        self.backend, self.device = backend, operator.index(device)
        limits = tuple(map(operator.index, (sampling_cache_bytes, sampling_work_bytes, plan_cache_bytes)))
        if min(limits) < 0 or limits[1] == 0 or self.device < 0:
            raise ValueError('cache/device settings must be nonnegative and sampling_work_bytes positive')
        if backend == 'cuda':
            from .cuda import is_available
            if not is_available():
                raise RuntimeError('CUDA sampling requires a configured CUDA build and device')
            # Never create a CUDA context in a PyTorch DataLoader worker.
            import sys
            torch = sys.modules.get('torch')
            if torch is not None and torch.utils.data.get_worker_info() is not None:
                raise RuntimeError('CUDA sampling belongs in the device-owning process, not a DataLoader worker')
        self._configuration = (reference, dict(targets or {}), backend, catalog, limits, self.device, dict(reader_options))
        self._pid, self._closed, self._lock = os.getpid(), False, RLock()
        self._readers, self._opened, self._targets = {}, [], {}
        self._tiles, self._plans = _Bounded(limits[0]), _Bounded(limits[2])
        self._work_bytes = limits[1]
        self._uploads = self._tile_reads = self._peak_work = 0
        self._catalog = reference.volume.sample.catalog if isinstance(reference, VolumeLevel) else catalog
        if isinstance(reference, VolumeLevel) and catalog is not None and catalog is not self._catalog:
            raise ValueError('VolumeLevel and explicit catalog must belong to the same catalog')
        options = dict(reader_options)
        self._own_decoded = 'decoded_cache' not in options
        self._own_memory = not isinstance(options.get('memory_cache'), MemoryCache)
        if self._own_decoded:
            options['decoded_cache'] = DecodedCache(max_bytes=options.pop('decoded_cache_bytes', 256 << 20),
                                                   max_decode_bytes=options.pop('max_decode_bytes', 256 << 20))
        if self._own_memory:
            options['memory_cache'] = MemoryCache(options.get('memory_cache') if options.get('memory_cache') is not None else 64 << 20)
        self._options = options
        try:
            if isinstance(reference, VolumeLevel):
                reference = self._volume_source(reference)
            if not isinstance(reference, Source) or reference.role != 'image':
                raise ValueError('reference must be a VolumeLevel or registered volumetric image Source')
            self.reference = reference
            ref = self._open(reference)
            self._reference_map = _placement(ref.registration)
            if ref.spatial_axes != ('z', 'y', 'x') or ref.registration.plane is not None:
                raise ValueError('reference must have three volume spatial axes')
            self._reference_reader = ref
            self._reference_map = _placement(ref.registration)
            self._reference_frame = _frame(ref.registration)
            targets = dict(targets or {})
            if 'image' in targets or any(not isinstance(k, str) or not k for k in targets):
                raise ValueError('target names must be nonempty strings; image is reserved')
            for name, source in {'image': reference, **targets}.items():
                if not isinstance(source, Source):
                    raise TypeError(f'{name}: target must be a Source')
                reader = self._open(source)
                self._targets[name] = self._prepare(source, reader)
                companions = {}
                for companion in source.companions:
                    child = reader.companions[companion.name]
                    companions[companion.name] = self._prepare(companion.source, child)
                self._targets[name]['companions'] = companions
            self.metadata = {'value_frames': {name: target['metadata'] for name, target in self._targets.items()
                                              if target['metadata'] is not None}}
            if not self.metadata['value_frames']:
                self.metadata = None
        except BaseException:
            self.close()
            raise

    def _volume_source(self, level):
        # Reuse discovery's adapter and storage-resolution contract.
        volume = level.volume
        candidates = [s for s in volume.sample.catalog.sources if s.sample_id == volume.sample.id
                      and s.owner_id == volume.id and s.role == 'image' and not s.derived]
        if len(candidates) != 1:
            raise ValueError(f'{volume.id}: expected one reference volume image source, found {len(candidates)}')
        source = candidates[0]
        if source.registration.affine is None:
            source = replace(source, registration=replace(source.registration, affine=Affine.identity(), origin='catalog-fallback'))
        return source.with_selection(level=level.level)

    def _open(self, source):
        # IDs alone do not identify selected arrays, encodings, or placement.
        import json
        key = json.dumps(source.to_dict(), sort_keys=True, separators=(',', ':'))
        if key not in self._readers:
            self._readers[key] = source.open(**self._options)
            self._opened.append(self._readers[key])
        return self._readers[key]

    def _between(self, source, frame):
        destination = self._reference_frame
        declared = None
        for descriptor in (source, self.reference):
            q, reg = descriptor.query, descriptor.registration
            if q is None:
                continue
            native = _frame(reg)
            qframe = (q.reference.sample_id, q.reference.volume_id)
            if self._catalog is not None:
                expected = self._catalog.sample(reg.sample_id).transform(reg.volume_id, q.reference.volume_id)
                if reg.sample_id != q.reference.sample_id or expected is None or expected.matrix != q.transform.matrix:
                    raise ValueError(f'{descriptor.id}: query context conflicts with catalog registration')
            candidate = None
            if native == frame and qframe == destination:
                candidate = q.transform
            elif native == destination and qframe == frame:
                candidate = q.transform.inverse()
            if candidate is not None:
                if declared is not None and candidate.matrix != declared.matrix:
                    raise ValueError(f'{source.id}: conflicting descriptor frame relationships')
                declared = candidate
        if frame == destination:
            known = Affine.identity()
        elif self._catalog is not None and frame[0] == destination[0]:
            known = self._catalog.sample(frame[0]).transform(frame[1], destination[1])
        else:
            known = None
        if known is not None and declared is not None and known.matrix != declared.matrix:
            raise ValueError(f'{source.id}: conflicting frame relationships')
        result = known if known is not None else declared
        if result is None:
            raise ValueError(f'{source.id}: unresolved relationship {frame} -> {destination}; supply catalog=')
        result.inverse()
        return result

    def _prepare(self, source, reader):
        import numpy as np
        from . import _native
        try:
            placement = _placement(reader.registration)
            if reader.spatial_axes not in (('z', 'y', 'x'), ('y', 'x')):
                raise ValueError('sampling requires ZYX or YX spatial axes')
            plane = reader.registration.plane is not None
            if (reader.spatial_axes == ('y', 'x')) != plane:
                raise ValueError('2D sources require an explicit Plane, 3D sources require a volume affine')
            if source.role not in ('image', 'scalar', 'label', 'vector', 'normal', 'coordinate'):
                raise ValueError('an explicit sampling role is required')
            dtype = np.dtype(reader.dtype)
            if dtype.kind not in 'buif' or (dtype.kind == 'f' and dtype.itemsize not in (4, 8)):
                raise TypeError('sampling requires bool, integer, float32 or float64 values')
            if source.role == 'label' and dtype.kind not in 'bui':
                raise TypeError('categorical sampling requires an integer/bool dtype or explicit encoding')
            between = self._between(source, _frame(reader.registration))
            mapping = (Affine.identity() if placement.matrix == self._reference_map.matrix
                       and between.matrix == Affine.identity().matrix else
                       placement.inverse() @ between.inverse() @ self._reference_map)
            shape = dict(zip(reader.axes, reader.shape))
            shape3 = tuple(shape.get(a, 1) for a in ('z', 'y', 'x'))
            channels = shape.get('c', 1)
            field = metadata = None
            if source.role in ('vector', 'normal', 'coordinate'):
                if channels != 3 or source.component_order is None or source.value_frame is None:
                    raise ValueError('geometric fields require three channels, component_order and value_frame')
                if source.role == 'vector' and source.normalize is None:
                    raise ValueError('vectors require an explicit normalize boolean')
                value = source.value_frame
                if value.kind == 'external':
                    if source.role != 'coordinate':
                        raise ValueError('external value frames are supported only for coordinates')
                    transform = Affine.identity()
                    metadata = dict(kind='external', name=value.name, units=value.units)
                else:
                    value_map = value.affine
                    if value_map is None:
                        if value.level == 0:
                            value_map = Affine.identity()
                        elif value.level is not None and self._catalog is not None:
                            volume = self._catalog.sample(value.sample_id).volumes[value.volume_id]
                            value_map = _placement(self._open(self._volume_source(volume.level(value.level))).registration)
                        else:
                            raise ValueError('value frame needs an affine or a resolvable volume level')
                    value_map.inverse()
                    transform = self._reference_map.inverse() @ self._between(source, (value.sample_id, value.volume_id)) @ value_map
                    metadata = dict(kind='volume', sample_id=self._reference_frame[0], volume_id=self._reference_frame[1],
                                    array_to_level0=self._reference_map.matrix, reference_id=self.reference.id,
                                    selection=asdict(self._reference_reader.resolved.selection))
                matrix = transform.linear
                if source.role == 'normal':
                    inv = transform.inverse().linear
                    matrix = tuple(tuple(inv[j][i] for j in range(3)) for i in range(3))
                translation = transform.translation if source.role == 'coordinate' else (0., 0., 0.)
                field = (tuple(v for row in matrix for v in row), translation,
                         tuple('xyz'.index(c) for c in source.component_order),
                         source.role == 'normal' or bool(source.normalize), source.role == 'normal')
                metadata['component_order'] = source.component_order
                if value.kind == 'external':
                    field = None
            # Native tile kernels operate in native byte order; categorical CPU
            # outputs are converted back to the declared byte order at return.
            storage_key = (reader.resolved.location, repr(reader.resolved.selection), reader.axes,
                           str(dtype), repr(source.encoding), reader.fill_value)
            direct = not plane and mapping.linear == Affine.identity().linear and all(v.is_integer() for v in mapping.translation)
            return dict(source=source, reader=reader, mapping=mapping, shape=shape3, channels=channels,
                        dtype=dtype, native_dtype=dtype.newbyteorder('='), plane=plane, storage_key=storage_key,
                        order=int(source.role != 'label'), direct=direct, field=field, metadata=metadata,
                        native_field=_native._aligned_make_field(*field) if field else _native.AlignedField())
        except (ValueError, TypeError) as error:
            raise type(error)(f'{source.id}: {error}') from error

    def _ready(self):
        if self._closed:
            raise ValueError('sample reader is closed')
        if self._pid != os.getpid():
            if self.backend == 'cuda':
                raise RuntimeError('CUDA sample readers cannot be used after fork')
            _INHERITED.append((self._opened, self._options, self._tiles, self._plans))
            self._lazy = True
        if getattr(self, '_lazy', False):
            state = self._configuration
            self._lazy = False
            self._restore(state)

    @staticmethod
    def _canonical(array, reader):
        import numpy as np
        axes = list(reader.axes)
        if 'c' not in axes:
            array = np.expand_dims(array, 0)
            axes.insert(0, 'c')
        if 'z' not in axes:
            array = np.expand_dims(array, 0)
            axes.insert(0, 'z')
        return np.ascontiguousarray(np.transpose(array, [axes.index(a) for a in ('c', 'z', 'y', 'x')]), dtype=array.dtype.newbyteorder('='))

    def _blocks(self, size):
        # Bounded affine work even when a request contains millions of voxels.
        for z in range(0, size[0], 16):
            for y in range(0, size[1], 32):
                for x in range(0, size[2], 64):
                    offset = (z, y, x)
                    yield offset, tuple(min(n-o, step) for n, o, step in zip(size, offset, (16, 32, 64)))

    def _plan(self, target, start, size, offset, block, tiles=True):
        from . import _native
        mapping = target['mapping']
        # Only this small matrix/origin calculation uses Python double. Splitting
        # before float32 preserves e.g. the subvoxel part of 2**26 + .25.
        origin = mapping.apply(tuple(reversed(start)))
        origin = tuple(reversed(origin))
        base = tuple(math.floor(v) if not (target['plane'] and a == 0) else 0 for a, v in enumerate(origin))
        local = tuple(v-b for v, b in zip(origin, base))
        axes = tuple(mapping.linear[a][b] for a in (2, 1, 0) for b in (2, 1, 0))
        limits = (1, 32, 32) if target['plane'] else (16, 16, 16)
        tile = tuple(max(1, min(extent, limit)) for extent, limit in zip(target['shape'], limits))
        key = (target['shape'], block, size, offset, base, local, axes, tile,
               target['channels'], target['order'], target['plane'], tiles)
        plan = self._plans.get(key)
        if plan is None:
            plan = _native.AlignedPlan(*key)
            self._plans.put(key, plan, plan.nbytes)
        return plan, tile

    def _empty(self, shape, dtype):
        import numpy as np
        from . import _native
        if self.backend == 'cpu':
            return np.empty(shape, dtype=dtype)
        return _native._aligned_empty(math.prod(shape), dtype.kind, dtype.itemsize, self.device)

    @staticmethod
    def _bytes(array):
        import numpy as np
        return array.reshape(-1).view(np.uint8)

    def _tile_inputs(self, target, plan, tile):
        from . import _native
        reader = target['reader']
        starts = [tuple(p) for p in plan.starts()]
        tile_bytes = math.prod(tile)*target['channels']*target['dtype'].itemsize
        # Source results, canonical copies, and CUDA tiles can overlap.
        working = len(starts)*tile_bytes*3
        if working > self._work_bytes:
            return None
        self._peak_work = max(self._peak_work, working)
        inputs, missing, positions = [], [], []
        # Selected output values are shared only when the reader/selection matches.
        for start in starts:
            key = (target['storage_key'], start, tile)
            value = self._tiles.get(key)
            inputs.append(value)
            if value is None:
                missing.append((start[1:] if target['plane'] else start, tile[1:] if target['plane'] else tile))
                positions.append(len(inputs)-1)
        arrays = reader.read_many(missing) if missing else []
        self._tile_reads += len(arrays)
        for position, array in zip(positions, arrays):
            array = self._canonical(array, reader)
            value = array
            if self.backend == 'cuda':
                value = _native._cuda_upload(array, self.device)
                self._uploads += array.nbytes
            inputs[position] = value
            self._tiles.put((target['storage_key'], starts[position], tile), value, tile_bytes)
        return inputs

    def _execute(self, target, start, size, offset, block, output, valid):
        import numpy as np
        from . import _native
        plan, tile = self._plan(target, start, size, offset, block)
        inputs = self._tile_inputs(target, plan, tile)
        if inputs is None:
            axis = max(range(3), key=lambda a: block[a])
            if block[axis] <= 1:
                raise MemoryError(f"{target['source'].id}: one sampling stencil exceeds sampling_work_bytes")
            left = list(block); left[axis] //= 2
            right = list(block); right[axis] -= left[axis]
            after = list(offset); after[axis] += left[axis]
            self._execute(target, start, size, offset, tuple(left), output, valid)
            self._execute(target, start, size, tuple(after), tuple(right), output, valid)
            return
        dtype = target['native_dtype']
        fill = np.array([target['reader'].fill_value], dtype=dtype).view(np.uint8)
        if self.backend == 'cpu':
            plan.cpu([self._bytes(a) for a in inputs], fill, dtype.kind, dtype.itemsize, self._bytes(output), self._bytes(valid), target['native_field'])
        else:
            _native._aligned_cuda(plan, inputs, fill, dtype.kind, dtype.itemsize, output, valid, target['native_field'])

    def _cuda_blocks(self, target, start, size, output, valid):
        import numpy as np
        from . import _native
        dtype = target['native_dtype']
        fill = np.array([target['reader'].fill_value], dtype=dtype).view(np.uint8)
        plans, inputs, working = [], [], 0

        def flush():
            if plans:
                _native._aligned_cuda_many(plans, inputs, fill, dtype.kind, dtype.itemsize,
                                           output, valid, target['native_field'])
                plans.clear()
                inputs.clear()

        for offset, block in self._blocks(size):
            plan, tile = self._plan(target, start, size, offset, block)
            needed = len(plan.starts()) * math.prod(tile) * target['channels'] * dtype.itemsize * 3
            if needed + working > self._work_bytes or len(plans) == 64:
                flush()
                working = 0
            tiles = self._tile_inputs(target, plan, tile)
            if tiles is None:
                self._execute(target, start, size, offset, block, output, valid)
            else:
                plans.append(plan)
                inputs.append(tiles)
                working += needed
                self._peak_work = max(self._peak_work, working)
        flush()

    def _sample_many(self, target, regions):
        import numpy as np
        from . import _native
        reader = target['reader']
        arrays = None
        if target['direct']:
            shift = tuple(reversed(tuple(int(v) for v in target['mapping'].translation)))
            arrays = reader.read_many([(tuple(s+d for s, d in zip(start, shift)), size) for start, size in regions])
        affine_arrays = None
        if (self.backend == 'cpu' and not target['direct'] and not target['plane']
                and target['dtype'].name == 'uint8' and reader.axes == ('z', 'y', 'x')
                and reader.source.encoding is None and reader._is_zarr
                and reader.fill_value == reader._backend.fill_value):
            grids = [(size, tuple(reversed(target['mapping'].apply(tuple(reversed(start))))),
                      tuple(tuple(target['mapping'].linear[a][b] for b in (2, 1, 0)) for a in (2, 1, 0)))
                     for start, size in regions]
            if all(0 <= v <= 2**30 for _, origin, _ in grids for v in origin):
                affine_arrays = reader._backend.reader.read_affine_many(grids, order=target['order'])
        results = []
        for index, (start, size) in enumerate(regions):
            dtype = target['native_dtype'] if target['direct'] or not target['order'] else np.dtype('float32')
            if target['metadata'] is not None:
                dtype = np.dtype('float32')
            shape = (target['channels'], *size)
            valid = self._empty((1, *size), np.dtype('bool'))
            if arrays is not None:
                output = self._canonical(arrays[index], reader).astype(dtype, copy=False)
                if self.backend == 'cuda':
                    self._uploads += output.nbytes
                    output = _native._cuda_upload(output, self.device)
                at = tuple(s+d for s, d in zip(start, shift))
                if self.backend == 'cpu':
                    valid.fill(False)
                    bounds = tuple(slice(max(0, -s), max(0, min(n, extent-s)))
                                   for s, n, extent in zip(at, size, target['shape']))
                    valid[(0, *bounds)] = True
                else:
                    _native._aligned_box_valid_cuda(target['shape'], at, size, valid)

            elif affine_arrays is not None:
                output = affine_arrays[index].reshape(shape)
                for offset, block in self._blocks(size):
                    plan, _ = self._plan(target, start, size, offset, block, False)
                    plan.valid_cpu(self._bytes(valid))
            else:
                output = self._empty(shape, dtype)
                if self.backend == 'cuda':
                    self._cuda_blocks(target, start, size, output, valid)
                else:
                    for offset, block in self._blocks(size):
                        self._execute(target, start, size, offset, block, output, valid)
            if target['field'] is not None and (arrays is not None or affine_arrays is not None):
                if self.backend == 'cpu':
                    _native._aligned_field_cpu(self._bytes(output), self._bytes(valid), *target['field'])
                else:
                    _native._aligned_field_cuda(output, valid, *target['field'])
            if self.backend == 'cuda':
                from .cuda import DeviceArray
                output = DeviceArray(output, shape, dtype, self.device)
                valid = DeviceArray(valid, (1, *size), np.bool_, self.device)
            elif not target['order']:
                output = output.astype(target['dtype'], copy=False)
            results.append((output, valid))
        return results

    @property
    def shape(self):
        """ZYX shape of the selected reference grid, independent of channels."""
        self._ready()
        return self._targets['image']['shape']

    def __getitem__(self, key):
        """Index shared ZYX space, or select a named target for CZYX indexing."""
        from .reader._indexing import parse_index
        if isinstance(key, str):
            self._ready()
            if key not in self._targets:
                raise KeyError(key)
            return _SampleTarget(self, key)
        start, size, squeeze = parse_index(key, self.shape)
        result = self.read(start, size)
        if any(squeeze):
            spatial = tuple(n for n, drop in zip(size, squeeze) if not drop)
            # Contiguous singleton-axis removal is a view on CPU and CUDA.
            # Coordinate values and their metadata stay in the reference frame.
            for values in (result.data, result.valid, *result.masks.values()):
                for name, array in values.items():
                    values[name] = array.reshape((array.shape[0], *spatial))
        return result

    def read(self, start, size):
        return self.read_many([(start, size)])[0]

    def read_many(self, regions):
        import copy
        self._ready()
        with self._lock:
            if self._closed:
                raise ValueError('sample reader is closed')
            regions = [_region(*region) for region in regions]
            results = [SampleResult({}, {}, {}, copy.deepcopy(self.metadata)) for _ in regions]
            for name, target in self._targets.items():
                for result, (array, valid) in zip(results, self._sample_many(target, regions)):
                    result.data[name], result.valid[name], result.masks[name] = array, valid, {}
                for mask, child in target['companions'].items():
                    for result, (array, _) in zip(results, self._sample_many(child, regions)):
                        result.masks[name][mask] = array
            return results

    def stats(self):
        self._ready()
        return dict(tile_cache_bytes=self._tiles.bytes, tile_cache_hits=self._tiles.hits,
                    plan_cache_bytes=self._plans.bytes, plan_cache_hits=self._plans.hits,
                    tile_reads=self._tile_reads, uploaded_bytes=self._uploads, peak_tile_work_bytes=self._peak_work,
                    sources={name: target['reader'].stats() for name, target in self._targets.items()})

    def close(self):
        if getattr(self, '_closed', True):
            return
        if self._pid != os.getpid() and not getattr(self, '_lazy', False):
            _INHERITED.append((self._opened, self._options, self._tiles, self._plans))
            self._closed = True
            return
        if getattr(self, '_lazy', False):
            self._closed = True
            return
        with self._lock:
            if self._closed:
                return
            for reader in self._opened:
                reader.close()
            self._opened.clear()
            self._tiles.items.clear()
            self._plans.items.clear()
            if self._own_decoded:
                self._options['decoded_cache'].clear()
            if self._own_memory:
                self._options['memory_cache'].clear()
            self._closed = True

    def __enter__(self):
        self._ready()
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):
        if self.backend == 'cuda':
            raise TypeError('CUDA sample readers cannot be pickled')
        if self._closed:
            raise ValueError('sample reader is closed')
        return self._configuration

    def __setstate__(self, state):
        self._configuration = state
        self.backend = state[2]
        self._closed, self._lazy, self._pid = False, True, os.getpid()

    def _restore(self, state):
        reference, targets, backend, catalog, limits, device, options = state
        self.__init__(reference, targets, backend=backend, catalog=catalog, sampling_cache_bytes=limits[0],
                      sampling_work_bytes=limits[1], plan_cache_bytes=limits[2], device=device, **options)
