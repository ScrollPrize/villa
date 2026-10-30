"""Geometry-only surface interpolation (``vcr::sources::SurfaceGeometry``)."""
import operator

from . import _reader
from ._sampling_engine import SamplingEngine


class SurfaceGeometry(SamplingEngine):
    """Reusable geometry-only UV interpolation, including the benchmark path.

    ``read_points`` returns XYZ float32 points in the surface coordinate-value frame
    and their validity; offsets are physical micrometers. It does not sample a volume.
    """
    def __init__(self, surface, *, sampling_cache_bytes=64 << 20, sampling_work_bytes=64 << 20, **reader_options):
        limits = (operator.index(sampling_cache_bytes), operator.index(sampling_work_bytes), 0)
        self._configuration = (surface, dict(reader_options), limits)
        self._surface, self._limits = surface, limits
        self._setup('cpu', 0, limits, reader_options)
        try:
            self._build()
        except BaseException:
            self._closed = True
            raise

    def _build(self):
        self._native = _reader.SurfaceGeometry(self._surface, *self._limits[:2], self._options_for, self._decoded)

    def _restore(self, state):
        surface, options, limits = state
        self.__init__(surface, sampling_cache_bytes=limits[0], sampling_work_bytes=limits[1], **options)

    @property
    def shape(self):
        """Rows and columns of the stored coordinate grid."""
        return tuple(self._ready().shape)

    def read_points(self, start, size, *, offsets_um=(0,)):
        start, size = tuple(map(operator.index, start)), tuple(map(operator.index, size))
        if len(start) != 2 or len(size) != 2:
            raise ValueError('surface start/size must be YX pairs with positive sizes <= 2**24')
        return self._ready().read_points(start, size, [float(v) for v in offsets_um])
