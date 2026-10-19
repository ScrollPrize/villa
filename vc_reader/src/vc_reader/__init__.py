"""Scroll-centered discovery and bounded image/label reading."""

from __future__ import annotations

from ._fetch import DEFAULT_METADATA_URL, default_cache_dir
from .affine import Affine
from .artifacts import AccessRoot, Artifact, ArtifactType, Origin, to_https, to_s3
from .catalog import DEFAULT_RESOLUTION_REL_TOL, PYRAMID_LEVELS, Catalog, License, Model, Scan, VolumeCoverage
from .browsing import ScrollSource, Collection, Asset, Volume, VolumeLevel, Segment, ResolutionCandidate, VolumeMapping
from .errors import AmbiguousMatchError, CacheMissError, NotFoundError, VcReaderError
from .sources import (
    Companion, Encoding, Frame, LevelRequest, Plane, QueryContext, SampleResult, TargetResult,
    Selection, Source, Storage, Surface, SurfaceRegistration, ValueFrame,
    VolumeRegistration,
)

__version__ = "0.1.0"

load = Catalog.load

__all__ = [
    "DEFAULT_METADATA_URL",
    "DEFAULT_RESOLUTION_REL_TOL",
    "PYRAMID_LEVELS",
    "AccessRoot",
    "Affine",
    "AmbiguousMatchError",
    "CacheMissError",
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
