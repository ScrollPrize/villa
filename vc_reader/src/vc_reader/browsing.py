"""Scroll-centered views over shared metadata records and storage descriptors (``vcr::catalog``)."""
from abc import ABC

from . import _catalog
from ._catalog import Asset, Pairing, ResolutionCandidate, ScrollSource, Segment, Volume, VolumeLevel, VolumeMapping
from .sources import _collection_types


class Collection(ABC):
    """Deterministic views with explicit filtering and unambiguous selection.

    Iteration hides only explicitly superseded entities. ID lookup also finds
    historical entities; filter(include_superseded=True) includes them in queries.
    """


for _type in _collection_types():
    Collection.register(_type)

__all__ = ["Asset", "Collection", "Pairing", "ResolutionCandidate", "ScrollSource", "Segment", "Volume", "VolumeLevel", "VolumeMapping"]
