"""Immutable descriptions of storage, values, and registration (``vcr::sources``).

Constructing or deserializing these objects never opens storage. Unknown fields
remain None; ``unresolved_fields`` is a declaration diagnostic, not an I/O probe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import _catalog
from ._catalog import (Companion, Encoding, Frame, LevelRequest, Plane, QueryContext, Selection, Source, Storage, Surface,
                       SurfaceRegistration, ValueFrame, VolumeRegistration)


def _collection_types():
    return (_catalog.ScrollCollection, _catalog.VolumeCollection, _catalog.SegmentCollection, _catalog.AssetCollection,
            _catalog.VolumeMappingCollection, _catalog.ResolutionCollection)


@dataclass(slots=True)
class TargetResult:
    """One sampled target's arrays, sharing the requested spatial grid.

    ``valid`` describes geometric support; ``masks`` retains companion values
    without applying a training policy. Arrays retain their CPU/CUDA storage.
    """

    data: Any
    valid: Any
    masks: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleResult:
    """One unbatched crop, grouped by target name.

    Access targets with ``sample[name]``; ``sample.image`` and ``sample.label``
    select the corresponding named targets. Slice-only volume data are CZYX;
    integer indexing removes spatial axes. Valid/mask arrays have one channel.
    For default DataLoader collation, return a dictionary of the needed arrays
    from the dataset. Metadata is optional and may require explicit conversion.
    """

    targets: dict[str, TargetResult]
    metadata: dict | None = None

    def __getitem__(self, name: str) -> TargetResult:
        return self.targets[name]

    @property
    def image(self) -> TargetResult:
        return self.targets['image']

    @property
    def label(self) -> TargetResult:
        return self.targets['label']
