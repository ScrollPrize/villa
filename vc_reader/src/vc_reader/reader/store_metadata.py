"""What is known about a volume store: its own ``meta.json`` / ``metadata.json`` and the
Open Data catalog, merged once when the store is first opened and kept from then on.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .auth import Access, is_remote
from .disk_cache import VC3D, atomic_write, cached_node_dirs, decoded_cache_path
from .transport import cache_url, read_node_file


CATALOG_FIELDS = ("sample_id", "volume_id", "scan_id", "name", "created_at", "data_format")
_METADATA_FILE = ".vc_reader_metadata.json"


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
    # From the Open Data catalog.
    sample_id: str = ""
    volume_id: str = ""
    scan_id: str = ""
    name: str = ""
    created_at: str = ""
    data_format: str = ""
    # The sources as published: file names (.zattrs too), and "catalog".
    documents: dict[str, Any] = field(default_factory=dict)


# What this process already knows, by store: later opens consult nothing.
_known: dict[str, StoreMetadata] = {}


@lru_cache(maxsize=1)
def _catalog_manifest(offline: bool):
    """The Open Data catalog's native manifest (from its cache file alone when offline);
    ``None`` when it is unavailable."""
    from ..catalog import Catalog
    from .. import _catalog
    try:
        if offline:
            return _catalog.cached(os.fspath(Catalog.cache_file()))
        return Catalog.load(timeout=15)._native
    except Exception:
        return None


def _catalog_entry(location: str, offline: bool) -> tuple[dict[str, Any] | None, bool]:
    """The catalog's entry for a remote store, and whether the catalog could answer. Stores
    outside the public channels are not in it, so the catalog is not loaded for them."""
    from .. import _catalog

    if _catalog.to_s3(location) is None:
        return None, True
    manifest = _catalog_manifest(offline)
    if manifest is None:
        return None, False
    return manifest.volume_at(location), True


def _merged(sidecars: dict[str, Any], catalog: dict[str, Any] | None) -> StoreMetadata:
    """One value per field: the store's own documents first, the catalog for what they omit."""
    values = dict(sidecars)
    for key in ("value_window", "z_crop"):
        if values[key] is not None:
            values[key] = tuple(values[key])
    if catalog:
        for key in ("voxel_size_um", "energy_kev", "detector_distance_mm"):
            if values[key] is None:
                values[key] = catalog.get(key)
        for key in CATALOG_FIELDS:
            values[key] = catalog.get(key) or ""
        values["documents"] = {**values["documents"], "catalog": catalog.get("raw")}
    return StoreMetadata(**values)


def read_store_metadata(location: str, access: Access, spec: dict[str, Any]) -> StoreMetadata:
    """The metadata of the store at ``location``: its sidecar documents and, for a remote
    store, the catalog, consulted only the first time: a remote store's merged result is
    kept in the disk cache (or for this process, without one), and a kept copy is trusted
    like a chunk. A local store's documents are cheap to read at each open."""
    from .. import _reader

    remote = is_remote(location)
    key = cache_url(location, access) if remote else location
    if remote and key in _known:
        return _known[key]
    cache_dirs = spec.get("cache_dirs") or []
    offline = bool(spec.get("offline")) and remote
    cached = cached_node_dirs(cache_dirs, key) if remote and cache_dirs else []

    if remote:
        for directory in cached:
            path = Path(directory, _METADATA_FILE)
            if path.is_file():
                try:
                    values = json.loads(path.read_text())
                    for name in ("value_window", "z_crop"):
                        if values.get(name) is not None:
                            values[name] = tuple(values[name])
                    _known[key] = StoreMetadata(**values)
                    return _known[key]
                except (OSError, ValueError, TypeError):
                    pass  # a damaged copy: read the sources again

    def read(node: str, name: str) -> str | None:
        for directory in cached_node_dirs(cache_dirs, cache_url(node, access)) if remote and cache_dirs else []:
            path = Path(directory, name)
            if path.is_file():
                return path.read_text()
        if offline:
            return None
        data = read_node_file(node, name, access)
        return None if data is None else data.decode()

    sidecars = json.loads(_reader.store_metadata_json(location, read))
    try:
        attrs = read(location, ".zattrs")
        if attrs:
            sidecars["documents"][".zattrs"] = json.loads(attrs)
    except (OSError, ValueError):
        pass
    catalog, answered = _catalog_entry(location, offline) if remote else (None, True)
    metadata = _merged(sidecars, catalog)
    # Keep it only once both sources have answered, so a later open can fill what is missing.
    if remote and answered and not offline:
        _known[key] = metadata
        root = next((d for d in cache_dirs if os.fspath(d) != VC3D), None)
        if remote and root is not None and spec.get("cache_writes", True):
            target = Path(decoded_cache_path(Path(root).expanduser(), key))
            try:
                target.mkdir(parents=True, exist_ok=True)
                atomic_write(target / _METADATA_FILE, json.dumps(asdict(metadata)).encode())
            except OSError:
                pass  # a read-only cache directory: this process still remembers it
    return metadata


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
