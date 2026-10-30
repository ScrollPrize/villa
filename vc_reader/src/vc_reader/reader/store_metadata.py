"""What a volume store says about itself: its ``meta.json`` / ``metadata.json``, read
once when the store is opened. Remote documents are kept in the disk cache, so later
opens and offline opens read them from there; a cached copy is trusted like a chunk.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .auth import Access, is_remote
from .disk_cache import VC3D, atomic_write, cached_node_dirs, decoded_cache_path
from .transport import cache_url, read_node_file


@dataclass(frozen=True)
class StoreMetadata:
    voxel_size_um: float | None = None  # level 0
    energy_kev: float | None = None
    detector_distance_mm: float | None = None
    beamline: str = ""
    scan_name: str = ""
    acquisition_date: str = ""
    # The source values the stored range spans, and the source Z range kept (zarr_export).
    value_window: tuple[float, float] | None = None
    z_crop: tuple[int, int] | None = None
    documents: dict[str, Any] = field(default_factory=dict)  # file name -> document as published


def read_store_metadata(location: str, access: Access, spec: dict[str, Any]) -> StoreMetadata:
    """The metadata of the store at ``location``, or empty metadata when it says nothing."""
    from .. import _reader

    remote = is_remote(location)
    cache_dirs = spec.get("cache_dirs") or []
    offline = bool(spec.get("offline")) and remote

    def read(node: str, name: str) -> str | None:
        if remote and cache_dirs:
            for directory in cached_node_dirs(cache_dirs, cache_url(node, access)):
                path = Path(directory, name)
                if path.is_file():
                    return path.read_text()
        if offline:
            return None
        data = read_node_file(node, name, access)
        if data is None:
            return None
        if remote and spec.get("cache_writes", True):
            root = next((d for d in cache_dirs if os.fspath(d) != VC3D), None)
            if root is not None:
                target = Path(decoded_cache_path(Path(root).expanduser(), cache_url(node, access)))
                try:
                    target.mkdir(parents=True, exist_ok=True)
                    atomic_write(target / name, data)
                except OSError:
                    pass  # a read-only cache directory: the store still opens
        return data.decode()

    values = json.loads(_reader.store_metadata_json(location, read))
    for key in ("value_window", "z_crop"):
        if values[key] is not None:
            values[key] = tuple(values[key])
    return StoreMetadata(**values)


def read_group_attrs(group: str, access: Access, spec: dict[str, Any]) -> dict[str, Any]:
    """The ``.zattrs`` of a store's root (empty when it has none), from the disk caches first."""
    remote = is_remote(group)
    try:
        if remote:
            for directory in cached_node_dirs(spec.get("cache_dirs") or [], cache_url(group, access)):
                path = Path(directory, ".zattrs")
                if path.is_file():
                    return json.loads(path.read_text())
            if spec.get("offline"):
                return {}
        data = read_node_file(group, ".zattrs", access)
        return json.loads(data) if data else {}
    except (OSError, ValueError):
        return {}


def level_scale(attrs: dict[str, Any], key: str) -> tuple[float, float, float]:
    """How many level-0 voxels one voxel of the level at path ``key`` spans (Z, Y, X):
    from the OME multiscales when they list it, otherwise 2^level for a numbered level."""
    multiscales = attrs.get("multiscales") or (attrs.get("ome") or {}).get("multiscales") or []
    datasets = multiscales[0].get("datasets", []) if multiscales else []

    def scale_of(dataset: dict[str, Any]) -> list[float] | None:
        for transform in dataset.get("coordinateTransformations", ()):
            if transform.get("type") == "scale":
                return [float(s) for s in transform["scale"]][-3:]
        return None

    finest = scale_of(datasets[0]) if datasets else None
    for dataset in datasets:
        if str(dataset.get("path")).strip("/") == key:
            scale = scale_of(dataset)
            if scale is not None and finest is not None and len(scale) == 3 and len(finest) == 3:
                return (scale[0] / finest[0], scale[1] / finest[1], scale[2] / finest[2])
    if key.isdigit():
        return (2.0 ** int(key),) * 3
    return (1.0, 1.0, 1.0)
