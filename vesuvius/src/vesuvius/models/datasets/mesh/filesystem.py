"""Filesystem-backed mesh adapter for .obj and .ply assets."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from ..adapters.base_io import AdapterConfig
from .base import LoadedMesh, MeshDataSourceAdapter
from .handles import MeshHandle
from .types import MeshMetadata, MeshPayload

try:  # Optional YAML support for manifests
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ManifestEntry:
    mesh_id: str
    path: Optional[Path]
    source_volume: Optional[str]
    transform: Optional[np.ndarray]
    attributes: Mapping[str, object]


class MeshAdapter(MeshDataSourceAdapter):
    """Loads mesh assets and exposes them as :class:`MeshHandle` objects."""

    def __init__(
        self,
        config: AdapterConfig,
        *,
        logger: Optional[logging.Logger] = None,
        metadata_path: Optional[Path] = None,
        metadata: Optional[Mapping[str, object]] = None,
        default_source_volume: Optional[str] = None,
        source_map: Optional[Mapping[str, str]] = None,
        transform_map: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> None:
        super().__init__(config, logger=logger)
        self._metadata_path = Path(metadata_path) if metadata_path else None
        self._external_metadata = metadata
        self._default_source_volume = default_source_volume
        self._source_map = dict(source_map or {})
        self._transform_map = self._normalise_transform_map(transform_map or {})
        self._validated: Dict[str, MeshMetadata] = {}

    # MeshDataSourceAdapter overrides -------------------------------------------------------------

    def discover(self) -> Tuple[MeshMetadata, ...]:
        mesh_dir = self.config.data_path / self.config.mesh_dirname
        if not mesh_dir.exists():
            raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

        files = [path for path in mesh_dir.glob("**/*") if path.is_file() and self._is_mesh_file(path)]
        if not files:
            raise ValueError(f"No mesh files with extensions {self._supported_extensions()} found in {mesh_dir}")

        manifest_entries = self._load_manifest(mesh_dir)
        path_lookup: Dict[Path, Path] = {path.resolve(): path for path in files}
        for entry in manifest_entries.values():
            if entry.path is None:
                continue
            resolved = entry.path.resolve(strict=False)
            if resolved.exists() and self._is_mesh_file(resolved):
                path_lookup.setdefault(resolved, resolved)
            elif entry.path.exists():
                raise ValueError(
                    f"Mesh manifest entry '{entry.mesh_id}' references unsupported file extension: {entry.path}"
                )
            else:
                raise FileNotFoundError(
                    f"Mesh manifest entry '{entry.mesh_id}' references missing file: {entry.path}"
                )

        files = sorted(path_lookup.values(), key=lambda p: str(p))
        discovered: List[MeshMetadata] = []

        for path in sorted(files):
            mesh_id = self._resolve_mesh_id(path, manifest_entries)
            entry = manifest_entries.get(mesh_id)
            transform = None
            attributes: Mapping[str, object] = {}
            source_volume: Optional[str] = None

            if entry is not None:
                transform = entry.transform
                attributes = dict(entry.attributes)
                source_volume = entry.source_volume

            if mesh_id in self._source_map:
                source_volume = self._source_map[mesh_id]
            if source_volume is None:
                source_volume = self._default_source_volume

            override_transform = self._transform_map.get(mesh_id)
            if override_transform is not None:
                transform = override_transform

            metadata = MeshMetadata(
                mesh_id=mesh_id,
                path=path,
                source_volume_id=source_volume,
                transform=transform,
                attributes=attributes,
            )
            discovered.append(metadata)
            self._validated[mesh_id] = metadata

        return tuple(discovered)

    def prepare(self, discovered: Tuple[MeshMetadata, ...]) -> None:
        self._discovered = discovered
        self._validate_source_volumes(discovered)

    def iter_meshes(self) -> Iterator[LoadedMesh]:
        if not getattr(self, "_discovered", None):
            raise RuntimeError("MeshAdapter.iter_meshes called before discover()")

        for metadata in self._discovered:
            loader = self._build_loader(metadata)
            handle = MeshHandle(metadata.path, metadata=metadata, loader=loader)
            yield LoadedMesh(metadata=metadata, handle=handle)

    # Internal helpers ---------------------------------------------------------------------------

    def _supported_extensions(self) -> Tuple[str, ...]:
        return tuple(ext.lower() for ext in self.config.mesh_extensions)

    def _is_mesh_file(self, path: Path) -> bool:
        return path.suffix.lower() in self._supported_extensions()

    def _load_manifest(self, mesh_dir: Path) -> Dict[str, _ManifestEntry]:
        manifest_data: Optional[Mapping[str, object]] = None

        if self._external_metadata is not None:
            manifest_data = self._external_metadata
        elif self._metadata_path is not None and self._metadata_path.exists():
            manifest_data = self._read_manifest_file(self._metadata_path)
        else:
            filename = self.config.mesh_metadata_filename
            if filename:
                candidate = mesh_dir / filename
                if candidate.exists():
                    manifest_data = self._read_manifest_file(candidate)

        if manifest_data is None:
            return {}

        entries: Dict[str, _ManifestEntry] = {}

        if "meshes" in manifest_data and isinstance(manifest_data["meshes"], list):
            mesh_entries = manifest_data["meshes"]
        else:
            mesh_entries = []
            for key, value in manifest_data.items():
                if isinstance(value, Mapping):
                    mesh_entries.append(dict(value, mesh_id=key))

        for raw_entry in mesh_entries:
            entry = self._normalise_manifest_entry(mesh_dir, raw_entry)
            entries[entry.mesh_id] = entry

        return entries

    def _read_manifest_file(self, path: Path) -> Mapping[str, object]:
        suffix = path.suffix.lower()
        with open(path, "r", encoding="utf-8") as fh:
            if suffix in {".yaml", ".yml"}:
                if yaml is None:
                    raise RuntimeError("PyYAML is required to read YAML mesh manifests")
                return yaml.safe_load(fh) or {}
            return json.load(fh)

    def _normalise_manifest_entry(self, root: Path, raw: Mapping[str, object]) -> _ManifestEntry:
        mesh_id = str(raw.get("mesh_id") or raw.get("id") or raw.get("name"))
        if not mesh_id:
            raise ValueError("Mesh manifest entries must define a 'mesh_id'")

        path_value = raw.get("path") or raw.get("file")
        if path_value is None:
            path = None
        else:
            path = Path(path_value)

        if path is not None:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = (root / candidate).resolve()
        else:
            candidate = None

        source_volume = raw.get("source_volume") or raw.get("volume_id")

        transform_value = raw.get("transform")
        transform = self._coerce_transform(transform_value)

        attributes = raw.get("attributes") or {}
        if not isinstance(attributes, Mapping):
            raise TypeError(f"Mesh manifest attributes for '{mesh_id}' must be a mapping")

        return _ManifestEntry(
            mesh_id=mesh_id,
            path=candidate,
            source_volume=source_volume,
            transform=transform,
            attributes=dict(attributes),
        )

    def _resolve_mesh_id(self, path: Path, manifest_entries: Mapping[str, _ManifestEntry]) -> str:
        stem = path.stem
        if stem in manifest_entries:
            return stem
        target = path.resolve()
        for entry in manifest_entries.values():
            entry_path = entry.path
            if entry_path is None:
                continue
            if entry_path.resolve() == target:
                return entry.mesh_id
        return stem

    def _validate_source_volumes(self, discovered: Tuple[MeshMetadata, ...]) -> None:
        missing: List[str] = []
        for metadata in discovered:
            if metadata.source_volume_id is None:
                continue
            # Validation is deferred to orchestrator where volume IDs are known.
            if metadata.source_volume_id.strip() == "":
                missing.append(metadata.mesh_id)
        if missing:
            raise ValueError(f"Meshes {missing} specify empty source_volume_id")

    def _build_loader(self, metadata: MeshMetadata):
        transform = metadata.transform

        if transform is None:
            return None  # type: ignore[arg-type]

        matrix = np.asarray(transform, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(
                f"Transform for mesh '{metadata.mesh_id}' must be 4x4 homogeneous; got shape {matrix.shape}"
            )

        rotation = matrix[:3, :3]
        translation = matrix[:3, 3]

        def loader(path: Path) -> MeshPayload:
            payload = MeshHandle._default_loader(path)
            vertices = (rotation @ payload.vertices.T).T + translation

            normals = payload.normals
            if normals is not None:
                rot = rotation
                normals = (rot @ normals.T).T
                norms = np.linalg.norm(normals, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                normals = normals / norms

            return MeshPayload(
                vertices=vertices.astype(np.float32, copy=False),
                faces=payload.faces,
                normals=None if normals is None else normals.astype(np.float32, copy=False),
                uv=payload.uv,
                uv_faces=payload.uv_faces,
            )

        return loader

    def _normalise_transform_map(
        self, transform_map: Mapping[str, object]
    ) -> Dict[str, np.ndarray]:
        matrices: Dict[str, np.ndarray] = {}
        for mesh_id, value in transform_map.items():
            result = self._coerce_transform(value)
            if result is not None:
                matrices[mesh_id] = result
        return matrices

    def _coerce_transform(self, value: Optional[object]) -> Optional[np.ndarray]:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)

        if array.ndim == 2:
            if array.shape == (4, 4):
                return array
            if array.shape == (3, 4):
                return np.vstack([array, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])

        flat = array.flatten()
        if flat.size == 16:
            return flat.reshape(4, 4)
        if flat.size == 12:
            promoted = flat.reshape(3, 4)
            return np.vstack([promoted, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])

        raise ValueError("Transforms must contain 12 (3x4) or 16 (4x4) values")
