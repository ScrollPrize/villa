"""Shared bounded affine execution and reader lifecycle for 2D/3D workflows."""
import math
import operator
import os
from collections import OrderedDict
from threading import RLock
from .affine import Affine

# Retain inherited C++ locks rather than destroying them after fork.
_INHERITED = []

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
        self.items[key] = (value, size)
        self.bytes += size

class SamplingEngine:
    def _value_placement(self, value):
        """Resolve a declared geometric value grid without guessing its level."""
        from .reader.source_resolution import array_placement
        if value.affine is not None:
            return value.affine
        if value.level == 0:
            return Affine.identity()
        if value.level is not None and getattr(self, '_catalog', None) is not None:
            volume = self._catalog.samples[value.sample_id].volumes[value.volume_id]
            return array_placement(self._open(volume.level(value.level).source).registration)
        raise ValueError('value frame needs an affine or a resolvable volume level')

    def _initialize_runtime(self, backend, device, limits, reader_options):
        from .reader import DecodedCache, MemoryCache
        if backend not in ('cpu', 'cuda'):
            raise ValueError("backend must be 'cpu' or 'cuda'")
        self.backend, self.device = backend, operator.index(device)
        if min(limits) < 0 or limits[1] == 0 or self.device < 0:
            raise ValueError('cache/device settings must be nonnegative and sampling_work_bytes positive')
        if backend == 'cuda':
            from .cuda import is_available
            if not is_available():
                raise RuntimeError('CUDA sampling requires a configured CUDA build and device')
            import sys
            torch = sys.modules.get('torch')
            if torch is not None and torch.utils.data.get_worker_info() is not None:
                raise RuntimeError('CUDA sampling belongs in the device-owning process, not a DataLoader worker')
        self._pid, self._closed, self._lock = os.getpid(), False, RLock()
        self._readers, self._opened, self._targets = {}, [], {}
        self._tiles, self._plans = _Bounded(limits[0]), _Bounded(limits[2])
        self._work_bytes = limits[1]
        self._uploads = self._tile_reads = self._peak_work = 0
        self._surface_metrics = dict(geometry_read_bytes=0, geometry_upload_bytes=0, geometry_download_bytes=0,
                                     candidate_upload_bytes=0, preparation_seconds=0., candidate_index_seconds=0.,
                                     raster_call_seconds=0., output_allocation_seconds=0.)
        options = dict(reader_options)
        self._own_decoded = 'decoded_cache' not in options
        self._own_memory = not isinstance(options.get('memory_cache'), MemoryCache)
        if self._own_decoded:
            options['decoded_cache'] = DecodedCache(max_bytes=options.pop('decoded_cache_bytes', 256 << 20),
                                                   max_decode_bytes=options.pop('max_decode_bytes', 256 << 20))
        if self._own_memory:
            options['memory_cache'] = MemoryCache(options.get('memory_cache') if options.get('memory_cache') is not None else 64 << 20)
        self._options = options

    def _sampling_target(self, source, reader, mapping, *, plane=False, planar_grid=False, field=None, metadata=None):
        import numpy as np
        from . import _reader
        dtype = np.dtype(reader.dtype)
        if dtype.kind not in 'buif' or (dtype.kind == 'f' and dtype.itemsize not in (4, 8)):
            raise TypeError('sampling requires bool, integer, float32 or float64 values')
        if source.role == 'label' and dtype.kind not in 'bui':
            raise TypeError('categorical sampling requires an integer/bool dtype or explicit encoding')
        shape = dict(zip(reader.axes, reader.shape))
        storage_key = (reader.resolved.location, repr(reader.resolved.selection), reader.axes,
                       str(dtype), repr(source.encoding), reader.fill_value)
        direct = (not plane or planar_grid) and mapping.linear == Affine.identity().linear and all(v.is_integer() for v in mapping.translation)
        return dict(source=source, reader=reader, mapping=mapping,
                    shape=tuple(shape.get(a, 1) for a in ('z', 'y', 'x')), channels=shape.get('c', 1),
                    dtype=dtype, native_endian_dtype=dtype.newbyteorder('='), plane=plane, storage_key=storage_key,
                    order=int(source.role != 'label'), direct=direct, field=field, metadata=metadata,
                    field_state=_reader._aligned_make_field(*field) if field else _reader.AlignedField())

    def _open(self, source):
        # IDs alone do not identify selected arrays, encodings, or placement.
        import json
        key = json.dumps(source.to_dict(), sort_keys=True, separators=(',', ':'))
        if key not in self._readers:
            self._readers[key] = source.open(**self._options)
            self._opened.append(self._readers[key])
        return self._readers[key]

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
        from . import _reader
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
            plan = _reader.AlignedPlan(*key)
            self._plans.put(key, plan, plan.nbytes)
        return plan, tile

    def _empty(self, shape, dtype):
        import numpy as np
        from . import _reader
        if self.backend == 'cpu':
            return np.empty(shape, dtype=dtype)
        return _reader._aligned_empty(math.prod(shape), dtype.kind, dtype.itemsize, self.device)

    @staticmethod
    def _bytes(array):
        import numpy as np
        return array.reshape(-1).view(np.uint8)

    def _tile_inputs(self, target, plan, tile):
        from . import _reader
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
                value = _reader._cuda_upload(array, self.device)
                self._uploads += array.nbytes
            inputs[position] = value
            self._tiles.put((target['storage_key'], starts[position], tile), value, tile_bytes)
        return inputs

    def _execute(self, target, start, size, offset, block, output, valid):
        import numpy as np
        from . import _reader
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
        dtype = target['native_endian_dtype']
        fill = np.array([target['reader'].fill_value], dtype=dtype).view(np.uint8)
        if self.backend == 'cpu':
            plan.cpu([self._bytes(a) for a in inputs], fill, dtype.kind, dtype.itemsize, self._bytes(output), self._bytes(valid), target['field_state'])
        else:
            _reader._aligned_cuda(plan, inputs, fill, dtype.kind, dtype.itemsize, output, valid, target['field_state'])

    def _cuda_blocks(self, target, start, size, output, valid):
        import numpy as np
        from . import _reader
        dtype = target['native_endian_dtype']
        fill = np.array([target['reader'].fill_value], dtype=dtype).view(np.uint8)
        plans, inputs, working = [], [], 0

        def flush():
            if plans:
                _reader._aligned_cuda_many(plans, inputs, fill, dtype.kind, dtype.itemsize,
                                           output, valid, target['field_state'])
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
        from . import _reader
        reader = target['reader']
        arrays = None
        if target['direct']:
            shift = tuple(reversed(tuple(int(v) for v in target['mapping'].translation)))
            shifted = [(tuple(s+d for s, d in zip(start, shift)), size) for start, size in regions]
            arrays = reader.read_many([(start[1:], size[1:]) if target['plane'] else (start, size) for start, size in shifted])
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
            dtype = target['native_endian_dtype'] if target['direct'] or not target['order'] else np.dtype('float32')
            if target['metadata'] is not None:
                dtype = np.dtype('float32')
            shape = (target['channels'], *size)
            valid = self._empty((1, *size), np.dtype('bool'))
            if arrays is not None:
                output = self._canonical(arrays[index], reader).astype(dtype, copy=False)
                if self.backend == 'cuda':
                    self._uploads += output.nbytes
                    output = _reader._cuda_upload(output, self.device)
                at = tuple(s+d for s, d in zip(start, shift))
                if self.backend == 'cpu':
                    valid.fill(False)
                    bounds = tuple(slice(max(0, -s), max(0, min(n, extent-s)))
                                   for s, n, extent in zip(at, size, target['shape']))
                    valid[(0, *bounds)] = True
                else:
                    _reader._aligned_box_valid_cuda(target['shape'], at, size, valid)

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
                    _reader._aligned_field_cpu(self._bytes(output), self._bytes(valid), *target['field'])
                else:
                    _reader._aligned_field_cuda(output, valid, *target['field'])
            if self.backend == 'cuda':
                from .cuda import DeviceArray
                output = DeviceArray(output, shape, dtype, self.device)
                valid = DeviceArray(valid, (1, *size), np.bool_, self.device)
            elif not target['order']:
                output = output.astype(target['dtype'], copy=False)
            results.append((output, valid))
        return results

    def stats(self):
        self._ready()
        result = dict(tile_cache_bytes=self._tiles.bytes, tile_cache_hits=self._tiles.hits,
                    plan_cache_bytes=self._plans.bytes, plan_cache_hits=self._plans.hits,
                    tile_reads=self._tile_reads, uploaded_bytes=self._uploads, peak_tile_work_bytes=self._peak_work,
                    sources={name: target['reader'].stats() for name, target in self._targets.items()})
        if getattr(self, '_geometries', None) or any(self._surface_metrics.values()):
            result['surface'] = dict(self._surface_metrics)
            result['sources'].update({name: target['reader'].stats() for name,target in getattr(self,'_surface_targets',{}).items()})
        return result

    def _point_block(self, target, points, base):
        """Sample one local-coordinate block using the shared tile kernels."""
        import numpy as np
        from . import _reader
        n = len(points)
        limits = (1, 32, 32) if target['plane'] else (16, 16, 16)
        tile = tuple(max(1, min(s, t)) for s, t in zip(target['shape'], limits))
        plan = _reader.PointPlan(target['shape'], base, points, tile,
                                 target['channels'], target['order'], target['plane'])
        inputs = self._tile_inputs(target, plan, tile)
        if inputs is None:
            if n <= 1:
                raise MemoryError('one point stencil exceeds sampling_work_bytes')
            yield from self._point_block(target, points[:n//2], base)
            yield from self._point_block(target, points[n//2:], base)
            return
        dtype = np.dtype('float32') if target['order'] else target['native_endian_dtype']
        out = self._empty((target['channels'], n), dtype)
        valid = self._empty((1, n), np.dtype('bool'))
        fill = np.array([target['reader'].fill_value], dtype=target['native_endian_dtype']).view(np.uint8)
        if self.backend == 'cpu':
            plan.cpu([self._bytes(a) for a in inputs], fill, target['dtype'].kind,
                     target['dtype'].itemsize, self._bytes(out), self._bytes(valid), target['field_state'])
        else:
            _reader._aligned_cuda(plan, inputs, fill, target['dtype'].kind, target['dtype'].itemsize,
                                  out, valid, target['field_state'])
            from .cuda import DeviceArray
            out = DeviceArray(out, (target['channels'], n), dtype, self.device)
            valid = DeviceArray(valid, (1, n), np.bool_, self.device)
        yield out, valid

    def _sample_points(self, target, points, base=(0, 0, 0)):
        """ZYX local float32 points plus an integer source-grid origin."""
        import numpy as np
        n = len(points)
        dtype = np.dtype('float32') if target['order'] else target['native_endian_dtype']
        if self.backend == 'cpu' and not target['order']:
            dtype = target['dtype']
        out, valid = self._owned_empty((target['channels'], n), dtype), self._owned_empty((1, n), np.dtype('bool'))
        cursor = 0
        for start in range(0, n, 32768):
            block = np.ascontiguousarray(points[start:start+32768], dtype=np.float32)
            for values, support in self._point_block(target, block, base):
                self._copy_columns(values, out, cursor)
                self._copy_columns(support, valid, cursor)
                cursor += values.shape[-1]
        return out, valid

    def _owned_empty(self, shape, dtype):
        out = self._empty(shape, dtype)
        if self.backend == 'cuda':
            from .cuda import DeviceArray
            out = DeviceArray(out, shape, dtype, self.device)
        return out

    def _copy_columns(self, source, destination, offset):
        """Copy a contiguous CxN block without downloading device values."""
        if self.backend == 'cpu':
            destination[:, offset:offset+source.shape[1]] = source
        else:
            from . import _reader
            _reader._surface_copy_cuda(source._array, destination._array, source.shape[0],
                                      source.shape[1], destination.shape[1], offset, source.dtype.itemsize)

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
            if hasattr(self, '_geometries'):
                self._geometries.clear()
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

