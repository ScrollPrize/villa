"""Aligned, bounded CPU/CUDA reads in a caller-selected reference array grid.

Importing this module does not import NumPy, native extensions, or PyTorch.
"""
from dataclasses import asdict, replace
import math
import operator
import os

from .affine import Affine
from .browsing import VolumeLevel
from .sources import Source, VolumeRegistration, SurfaceRegistration, SampleResult, TargetResult

from ._sampling_engine import SamplingEngine
from .reader.source_resolution import array_placement


def open_sample(reference, targets=None, *, backend='cpu', catalog=None, **reader_options):
    """Open a reusable aligned reader; reads return unbatched CZYX SampleResults.

    ``sampling_cache_bytes`` (64 MiB) bounds retained CPU/device tiles;
    ``sampling_work_bytes`` (64 MiB) bounds tiles held by one native request.
    ``plan_cache_bytes`` (8 MiB) bounds reusable affine plans. ``device`` selects
    the CUDA device. ``surface_output_bytes`` (512 MiB) limits dense surface
    results; projection can instead yield bounded tiles. Other options are
    passed to Source.open().
    """
    return SampleReader(reference, targets, backend=backend, catalog=catalog, **reader_options)


def _frame(reg):
    return reg.sample_id, reg.volume_id



def _region(start, size):
    start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
    if len(start) != 3 or len(size) != 3 or any(n < 0 or n > 2**24 for n in size):
        raise ValueError('start/size must be ZYX triples, with sizes in [0, 2**24]')
    if any(abs(n) > 2**40 for n in start):
        raise ValueError('crop starts must be within +/- 2**40')
    return start, size




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


class SampleReader(SamplingEngine):
    def __init__(self, reference, targets=None, *, backend='cpu', catalog=None,
                 sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20,
                 plan_cache_bytes=8 << 20, device=0, **reader_options):
        limits = tuple(map(operator.index, (sampling_cache_bytes, sampling_work_bytes, plan_cache_bytes)))
        self._configuration = (reference, dict(targets or {}), backend, catalog, limits, device, dict(reader_options))
        self._catalog = reference.volume.scroll.catalog if isinstance(reference, VolumeLevel) else catalog
        if isinstance(reference, VolumeLevel) and catalog is not None and catalog is not self._catalog:
            raise ValueError('VolumeLevel and explicit catalog must belong to the same catalog')
        self._initialize_runtime(backend, device, limits, reader_options)
        self._surface_targets, self._geometries = {}, {}
        try:
            self._surface_output_bytes = operator.index(self._options.pop('surface_output_bytes', 512 << 20))
            if self._surface_output_bytes <= 0:
                raise ValueError('surface_output_bytes must be positive')
            if isinstance(reference, VolumeLevel):
                reference = self._volume_source(reference)
            if not isinstance(reference, Source) or reference.role != 'image':
                raise ValueError('reference must be a VolumeLevel or registered volumetric image Source')
            self.reference = reference
            ref = self._open(reference)
            self._reference_map = array_placement(ref.registration)
            if ref.spatial_axes != ('z', 'y', 'x') or ref.registration.plane is not None:
                raise ValueError('reference must have three volume spatial axes')
            self._reference_reader = ref
            self._reference_map = array_placement(ref.registration)
            self._reference_frame = _frame(ref.registration)
            targets = dict(targets or {})
            if 'image' in targets or any(not isinstance(k, str) or not k for k in targets):
                raise ValueError('target names must be nonempty strings; image is reserved')
            for name, source in {'image': reference, **targets}.items():
                if not isinstance(source, Source):
                    raise TypeError(f'{name}: target must be a Source')
                reader = self._open(source)
                if isinstance(reader.registration, SurfaceRegistration):
                    from .surface import prepare_target
                    self._surface_targets[name] = prepare_target(self, source, reader)
                    continue
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
        return level.source


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
                expected = self._catalog.samples[reg.sample_id].transform(reg.volume_id, q.reference.volume_id)
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
            known = self._catalog.samples[frame[0]].transform(frame[1], destination[1])
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
            placement = array_placement(reader.registration)
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
                    value_map = self._value_placement(value)
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
            return self._sampling_target(source, reader, mapping, plane=plane, field=field, metadata=metadata)
        except (ValueError, TypeError) as error:
            raise type(error)(f'{source.id}: {error}') from error











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
            for target in result.targets.values():
                target.data = target.data.reshape((target.data.shape[0], *spatial))
                target.valid = target.valid.reshape((target.valid.shape[0], *spatial))
                for name, array in target.masks.items():
                    target.masks[name] = array.reshape((array.shape[0], *spatial))
        return result

    def read(self, start, size):
        return self.read_many([(start, size)])[0]

    def read_many(self, regions):
        import copy
        self._ready()
        if self._surface_targets:
            raise ValueError('surface targets require explicit render_surface or project_surface')
        with self._lock:
            if self._closed:
                raise ValueError('sample reader is closed')
            regions = [_region(*region) for region in regions]
            results = [SampleResult({}, copy.deepcopy(self.metadata)) for _ in regions]
            for name, target in self._targets.items():
                for result, (array, valid) in zip(results, self._sample_many(target, regions)):
                    result.targets[name] = TargetResult(array, valid)
                for mask, child in target['companions'].items():
                    for result, (array, _) in zip(results, self._sample_many(child, regions)):
                        result[name].masks[mask] = array
            return results

    def render_surface(self, surface, start, size, *, offsets_um=(0,)):
        """Sample physical-offset layers over a YX patch in native surface UV."""
        from .surface import render_surface
        self._ready()
        with self._lock:
            return render_surface(self, surface, start, size, offsets_um)

    def project_surface(self, surface, start, size, *, label_target, thickness_um=0, padding_um=0):
        """Derive and read a 3D crop from a YX patch of a registered label."""
        from .surface import project_surface
        self._ready()
        with self._lock:
            return project_surface(self, surface, start, size, label_target, thickness_um, padding_um)

    def iter_project_surface(self, surface, start, size, *, label_target, tile_size,
                             thickness_um=0, padding_um=0):
        """Yield bounded CZYX results, with each tile's origin in metadata."""
        from .surface import iter_project_surface
        return iter_project_surface(self, surface, start, size, label_target, tile_size, thickness_um, padding_um)







    def _restore(self, state):
        reference, targets, backend, catalog, limits, device, options = state
        self.__init__(reference, targets, backend=backend, catalog=catalog, sampling_cache_bytes=limits[0],
                      sampling_work_bytes=limits[1], plan_cache_bytes=limits[2], device=device, **options)
