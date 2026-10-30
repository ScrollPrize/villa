"""Read published or explicitly registered flattened image/label pairs (``vcr::sources::PairedReader``)."""
import math
import operator

from . import _reader
from ._sampling_engine import SamplingEngine
from .reader._indexing import parse_index


class _PairedTarget:
    def __init__(self, reader, name):
        self.reader, self.name = reader, name

    @property
    def shape(self):
        native = self.reader._ready()
        depth = (native.layer_count,) if self.name == 'image' and not native.drop_layer else ()
        return (native.target_channels(self.name), *depth, *native.shape)

    def __getitem__(self, key):
        native = self.reader._ready()
        start, size, squeeze = parse_index(key, self.shape)
        z = native.layer_start if self.name == 'image' else 0
        depth = native.layer_count if self.name == 'image' else 1
        if self.name == 'image' and not native.drop_layer:
            z += start[1]
            depth = size[1]
        # A zero channel request needs no I/O either.
        region_size = (depth, *size[-2:]) if size[0] else (0, *size[-2:])
        array, _ = native.sample_target(self.name, (z, *start[-2:]), region_size)
        shape = tuple(n for n, drop in zip(size, squeeze) if not drop)
        if self.reader.backend == 'cuda':
            offset = start[0] * math.prod(region_size) if all(size) else 0
            return array._view(shape, offset)
        array = array[start[0]:start[0]+size[0]].reshape(shape)
        return array[()] if not shape else array


class PairedReader(SamplingEngine):
    """Shared indexing is YX on the selected image, with independent layer depth.

    label/mask grids are selected independently. A matching stored grid is used
    when available; otherwise categorical values use exact nearest-neighbor
    sampling. CPU workers reopen descriptors; CUDA readers remain process-local.
    """
    def __init__(self, label, image, pairing, *, level=0, layers=slice(None), backend='cpu',
                 sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20,
                 plan_cache_bytes=8 << 20, device=0, **reader_options):
        limits = tuple(map(operator.index, (sampling_cache_bytes, sampling_work_bytes, plan_cache_bytes)))
        self._configuration = (label, image, backend, pairing, level, layers, limits, device, dict(reader_options))
        self._pair, self._level, self._layers, self._limits = (label, image, pairing), level, layers, limits
        self._setup(backend, device, limits, reader_options)
        try:
            self._build()
        except BaseException:
            self._closed = True
            raise

    def _choose_layers(self, depth):
        start, size, drop = parse_index(self._layers, (depth,))
        return start[0], size[0], drop[0]

    def _build(self):
        self._native = _reader.PairedReader(*self._pair, self._level, self._choose_layers, self.backend, self.device, *self._limits,
                                            self._options_for, self._decoded)

    def _restore(self, state):
        label, image, backend, pairing, level, layers, limits, device, options = state
        self.__init__(label, image, pairing, backend=backend, level=level, layers=layers,
                      sampling_cache_bytes=limits[0], sampling_work_bytes=limits[1], plan_cache_bytes=limits[2],
                      device=device, **options)

    @property
    def shape(self):
        return tuple(self._ready().shape)

    def __getitem__(self, key):
        native = self._ready()
        if isinstance(key, str):
            if not native.has_target(key):
                raise KeyError(key)
            return _PairedTarget(self, key)
        start, size, squeeze = parse_index(key, self.shape)
        result = self.read(start, size)
        if any(squeeze):
            spatial = tuple(n for n, drop in zip(size, squeeze) if not drop)
            for target in result.targets.values():
                target.data = target.data.reshape((*target.data.shape[:-2], *spatial))
                target.valid = target.valid.reshape((*target.valid.shape[:-2], *spatial))
                for name, array in target.masks.items():
                    target.masks[name] = array.reshape((*array.shape[:-2], *spatial))
        return result

    def read(self, start, size):
        return self.read_many([(start, size)])[0]

    def read_many(self, regions):
        native = self._ready()
        normalized = []
        for start, size in regions:
            start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
            if len(start) != 2 or len(size) != 2:
                raise ValueError('paired regions require YX start/size, nonnegative bounded sizes')
            normalized.append((start, size))
        return native.read_many(normalized)
