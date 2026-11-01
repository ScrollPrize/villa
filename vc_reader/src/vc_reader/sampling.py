"""Aligned, bounded CPU/CUDA reads in a caller-selected reference array grid (``vcr::sources::SampleReader``)."""
import math
import operator

from . import _reader
from ._sampling_engine import SamplingEngine
from .browsing import VolumeLevel
from .reader._indexing import parse_index
from .sources import SurfaceRegistration, Surface


def open_sample(reference, targets=None, *, backend='cpu', catalog=None, **reader_options):
    """Open a reusable aligned reader; reads return unbatched CZYX SampleResults.

    ``sampling_cache_bytes`` (64 MiB) bounds retained CPU/device tiles;
    ``sampling_work_bytes`` (64 MiB) bounds tiles held by one reader request.
    ``plan_cache_bytes`` (8 MiB) bounds reusable affine plans. ``device`` selects
    the CUDA device. ``surface_output_bytes`` (512 MiB) limits dense surface
    results; projection can instead yield bounded tiles. Other options are
    passed to Source.open().
    """
    return SampleReader(reference, targets, backend=backend, catalog=catalog, **reader_options)


def _select(array, backend, start, size, squeeze):
    # Complete geometric components are transformed before channels are selected: one
    # output component may depend on every input component.
    shape = tuple(n for n, drop in zip(size, squeeze) if not drop)
    if backend == 'cuda':
        offset = start[0] * math.prod(size[1:]) if all(size) else 0
        return array._view(shape, offset)
    array = array[start[0]:start[0] + size[0]].reshape(shape)
    return array[()] if not shape else array


class _SampleTarget:
    """One aligned target, indexed in CZYX order on its reader's reference grid."""

    def __init__(self, reader, name):
        self._reader, self._name = reader, name

    @property
    def shape(self):
        native = self._reader._ready()
        return (native.target_channels(self._name), *native.shape)

    def __getitem__(self, key):
        native = self._reader._ready()
        start, size, squeeze = parse_index(key, self.shape)
        spatial_size = tuple(size[1:]) if size[0] else (0, 0, 0)
        array, _ = native.sample_target(self._name, tuple(start[1:]), spatial_size)
        return _select(array, self._reader.backend, start, size, squeeze)


class SampleReader(SamplingEngine):
    def __init__(self, reference, targets=None, *, backend='cpu', catalog=None,
                 sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20,
                 plan_cache_bytes=8 << 20, device=0, surface_output_bytes=512 << 20, **reader_options):
        limits = tuple(map(operator.index, (sampling_cache_bytes, sampling_work_bytes, plan_cache_bytes)))
        self._surface_output_bytes = operator.index(surface_output_bytes)
        if self._surface_output_bytes <= 0:
            raise ValueError('surface_output_bytes must be positive')
        self._configuration = (reference, dict(targets or {}), backend, catalog, limits, device,
                               self._surface_output_bytes, dict(reader_options))
        self.reference = reference.source if isinstance(reference, VolumeLevel) else reference
        self._reference, self._targets_declared, self._catalog, self._limits = reference, dict(targets or {}), catalog, limits
        self._setup(backend, device, limits, reader_options)
        try:
            self._build()
        except BaseException:
            self._closed = True
            raise

    def _build(self):
        self._native = _reader.SampleReader(self._reference, self._targets_declared, self._catalog, self.backend, self.device, *self._limits,
                                            self._surface_output_bytes, self._options_for, self._decoded)

    def _restore(self, state):
        reference, targets, backend, catalog, limits, device, surface_output, options = state
        self.__init__(reference, targets, backend=backend, catalog=catalog, sampling_cache_bytes=limits[0],
                      sampling_work_bytes=limits[1], plan_cache_bytes=limits[2], device=device,
                      surface_output_bytes=surface_output, **options)

    @property
    def reference_map(self):
        return self._ready().reference_map

    @property
    def shape(self):
        """ZYX shape of the selected reference grid, independent of channels."""
        return tuple(self._ready().shape)

    def __getitem__(self, key):
        """Index shared ZYX space, or select a named target for CZYX indexing."""
        native = self._ready()
        if isinstance(key, str):
            if not native.has_target(key):
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
        native = self._ready()
        normalized = []
        for start, size in regions:
            start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
            if len(start) != 3 or len(size) != 3:
                raise ValueError('start/size must be ZYX triples, with sizes in [0, 2**24]')
            normalized.append((start, size))
        return native.read_many(normalized)

    def _surface(self, surface):
        if isinstance(surface, Surface):
            return surface
        source = getattr(surface, 'source', surface)
        registration = getattr(source, 'registration', None)
        if isinstance(registration, SurfaceRegistration):
            return registration.surface
        return self._ready().surface_named(getattr(surface, 'id', surface))

    def render_surface(self, surface, start, size, *, offsets_um=(0,), interpolation='linear'):
        """Sample physical-offset layers over a YX patch in full-resolution surface UV.

        ``interpolation`` is ``'linear'`` (bilinear patches) or ``'smooth'`` (bicubic).
        """
        native = self._ready()
        start, size = _yx(start, size)
        return native.render_surface(self._surface(surface), start, size, [float(v) for v in offsets_um], interpolation)

    def project_surface(self, surface, start, size, *, label_target, thickness_um=0, padding_um=0):
        """Use a label YX patch to bound a crop, then project its sheet across it.

        Geometric coverage includes sheet outside the selecting patch. Label
        validity additionally requires annotation support at the surface point.
        """
        native = self._ready()
        start, size = _yx(start, size)
        projection = native.projection(self._surface(surface), start, size, label_target, float(thickness_um), float(padding_um))
        return projection.read_crop(projection.origin, projection.shape)

    def iter_project_surface(self, surface, start, size, *, label_target, tile_size, thickness_um=0, padding_um=0):
        """Yield bounded CZYX results, with each tile's origin in metadata."""
        tile_size = tuple(map(operator.index, tile_size))
        if len(tile_size) != 3:
            raise ValueError('tile_size must be three positive ZYX dimensions')
        native = self._ready()
        start, size = _yx(start, size)
        projection = native.projection(self._surface(surface), start, size, label_target, float(thickness_um), float(padding_um))
        for origin, shape in projection.regions(tile_size):
            self._ready()
            yield projection.read_crop(origin, shape)


def _yx(start, size):
    start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
    if len(start) != 2 or len(size) != 2:
        raise ValueError('surface start/size must be YX pairs with positive sizes <= 2**24')
    return start, size
