"""Mesh metadata and payload structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class MeshMetadata:
    """Describes mesh alignment and provenance information."""

    mesh_id: str
    path: Path
    source_volume_id: Optional[str] = None
    transform: Optional[np.ndarray] = None  # Homogeneous 4x4 transform to global frame
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeshPayload:
    """Concrete mesh buffers returned by mesh handles."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    uv_faces: Optional[np.ndarray] = None
