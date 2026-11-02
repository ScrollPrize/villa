"""Scroll-centered discovery and bounded image/label reading."""

from __future__ import annotations

from .catalog import DEFAULT_METADATA_URL, default_cache_dir
from .affine import Affine
from .artifacts import AccessRoot, Artifact, ArtifactType, Origin, to_https, to_s3
from .catalog import DEFAULT_RESOLUTION_REL_TOL, PYRAMID_LEVELS, Catalog, License, Model, Scan, VolumeCoverage
from .browsing import ScrollSource, Collection, Asset, Volume, VolumeLevel, Segment, ResolutionCandidate, VolumeMapping
from .errors import AmbiguousMatchError, CacheMissError, LowDiskSpaceError, NotFoundError, VcReaderError
from .sources import (
    Companion, Encoding, Frame, LevelRequest, Plane, QueryContext, SampleResult, TargetResult,
    Selection, Source, Storage, Surface, SurfaceRegistration, ValueFrame,
    VolumeRegistration,
)

__version__ = "0.1.0"

load = Catalog.load

__all__ = [
    "shortest_path",
    "DEFAULT_METADATA_URL",
    "DEFAULT_RESOLUTION_REL_TOL",
    "PYRAMID_LEVELS",
    "AccessRoot",
    "Affine",
    "AmbiguousMatchError",
    "CacheMissError",
    "LowDiskSpaceError",
    "Artifact",
    "ArtifactType",
    "Catalog",
    "ScrollSource",
    "Collection",
    "Asset",
    "ResolutionCandidate",
    "VolumeMapping",
    "License",
    "Model",
    "NotFoundError",
    "Origin",
    "Scan",
    "Segment",
    "VcReaderError",
    "Volume",
    "VolumeCoverage",
    "VolumeLevel",
    "default_cache_dir",
    "load",
    "to_https",
    "to_s3",
    "Companion",
    "Encoding",
    "Frame",
    "LevelRequest",
    "Plane",
    "QueryContext",
    "SampleResult",
    "TargetResult",
    "Selection",
    "Source",
    "Storage",
    "Surface",
    "SurfaceRegistration",
    "ValueFrame",
    "VolumeRegistration",
]


_NATIVE_EXPORTS = {
    'shortest_path', 'write_image', 'encode_png', 'encode_jpeg', 'snapshot',
    'window_level_lut', 'palette_lut', 'overlay_lut', 'display_to_uint8',
    'apply_lut', 'blend', 'blend_argb',
}
__all__ += sorted(_NATIVE_EXPORTS - set(__all__))


def __getattr__(name):
    if name in _NATIVE_EXPORTS:
        from . import _reader
        return getattr(_reader, name)
    raise AttributeError(name)
