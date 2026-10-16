"""Find and locate Vesuvius Challenge open data from its published metadata.json.

    >>> import vc_reader
    >>> cat = vc_reader.load()
    >>> vol = cat.sample("pherc paris 4").volume(resolution=2.4, energy_kev=78)
    >>> lvl = vol.level(um=9.6)                      # pyramid level 2
    >>> labels = vol.ink_labels()                    # direct + mappable via transforms
"""

from __future__ import annotations

from ._fetch import DEFAULT_METADATA_URL, default_cache_dir
from .affine import Affine
from .artifacts import AccessRoot, Artifact, ArtifactType, Origin, to_https, to_s3
from .catalog import (
    DEFAULT_RESOLUTION_REL_TOL,
    PYRAMID_LEVELS,
    Catalog,
    InkLabelMatch,
    InkLabelSet,
    License,
    Model,
    PairedArtifact,
    PairedData,
    Sample,
    Scan,
    Segment,
    SegmentOnVolume,
    Volume,
    VolumeCoverage,
    VolumeLevel,
)
from .errors import AmbiguousMatchError, NotFoundError, VcReaderError

__version__ = "0.1.0"

load = Catalog.load

__all__ = [
    "DEFAULT_METADATA_URL",
    "DEFAULT_RESOLUTION_REL_TOL",
    "PYRAMID_LEVELS",
    "AccessRoot",
    "Affine",
    "AmbiguousMatchError",
    "Artifact",
    "ArtifactType",
    "Catalog",
    "InkLabelMatch",
    "InkLabelSet",
    "License",
    "Model",
    "NotFoundError",
    "Origin",
    "PairedArtifact",
    "PairedData",
    "Sample",
    "Scan",
    "Segment",
    "SegmentOnVolume",
    "VcReaderError",
    "Volume",
    "VolumeCoverage",
    "VolumeLevel",
    "default_cache_dir",
    "load",
    "to_https",
    "to_s3",
]
