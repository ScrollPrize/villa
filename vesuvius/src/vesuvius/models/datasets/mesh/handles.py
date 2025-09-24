"""Lazy mesh handle implementation."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Callable, Optional

import numpy as np

from .types import MeshMetadata, MeshPayload


class MeshHandle:
    """Lazy mesh loader with cached payload data."""

    def __init__(
        self,
        path: Path,
        *,
        metadata: MeshMetadata,
        loader: Optional[Callable[[Path], MeshPayload]] = None,
    ) -> None:
        self._path = Path(path)
        self._metadata = metadata
        self._loader = loader or self._default_loader
        self._payload: Optional[MeshPayload] = None
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def metadata(self) -> MeshMetadata:
        return self._metadata

    def read(self) -> MeshPayload:
        with self._lock:
            if self._payload is None:
                self._payload = self._loader(self._path)
            return self._payload

    def raw(self) -> MeshPayload:
        return self.read()

    def close(self) -> None:
        with self._lock:
            self._payload = None

    @staticmethod
    def _default_loader(path: Path) -> MeshPayload:
        try:
            import trimesh  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency managed externally
            raise RuntimeError(
                "MeshHandle requires 'trimesh' to load meshes; provide a custom loader to avoid this dependency."
            ) from exc

        mesh = trimesh.load(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):  # pragma: no cover - defensive
            mesh = mesh.dump().sum()

        uv_faces: Optional[np.ndarray] = None
        uv: Optional[np.ndarray] = None

        if mesh.visual.kind == "texture" and getattr(mesh.visual, "uv", None) is not None:
            uv = np.asarray(mesh.visual.uv, dtype=np.float32)
            if getattr(mesh.visual, "uv_faces", None) is not None:
                uv_faces = np.asarray(mesh.visual.uv_faces, dtype=np.int64)

        normals: Optional[np.ndarray]
        if mesh.vertex_normals is not None and len(mesh.vertex_normals):
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        else:
            normals = None

        return MeshPayload(
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int64),
            normals=normals,
            uv=uv,
            uv_faces=uv_faces,
        )
