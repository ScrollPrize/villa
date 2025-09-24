"""Base abstractions for mesh data sources."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Sequence

if TYPE_CHECKING:
    from ..adapters.base_io import AdapterConfig
from .handles import MeshHandle
from .types import MeshMetadata


@dataclass(frozen=True)
class LoadedMesh:
    """Container pairing mesh metadata with a mesh handle."""

    metadata: MeshMetadata
    handle: MeshHandle


class MeshDataSourceAdapter(ABC):
    """Base class for mesh providers that integrate with the dataset pipeline."""

    def __init__(self, config: "AdapterConfig", *, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._discovered: Sequence[MeshMetadata] = ()

    @abstractmethod
    def discover(self) -> Sequence[MeshMetadata]:
        """Locate mesh assets and return discovered metadata."""

    def prepare(self, discovered: Sequence[MeshMetadata]) -> None:  # pragma: no cover - optional override
        self._discovered = discovered

    @abstractmethod
    def iter_meshes(self) -> Iterator[LoadedMesh]:
        """Yield fully prepared mesh handles."""

    def run(self) -> Iterator[LoadedMesh]:
        self.logger.debug("Starting mesh adapter discovery")
        discovered = self.discover()
        self.logger.debug("Discovered %d meshes", len(discovered))
        self.prepare(discovered)
        return self.iter_meshes()
