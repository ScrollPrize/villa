from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import ManagerConfig


@dataclass(frozen=True)
class CatalogCache:
    document: dict[str, Any]
    metadata: dict[str, Any]
    warning: str | None = None


@dataclass(frozen=True)
class VolumeRecord:
    sample_id: str
    volume_id: str
    long_id: str
    shape: tuple[int, ...]
    pixel_size_um: float | None
    data_format: str | None
    license: dict[str, Any] | None
    origins: tuple[dict[str, Any], ...]
    selected_origin: dict[str, Any] | None
    catalog_sha256: str
    catalog_fetched_at: str | None
    catalog_metadata: dict[str, Any]
    raw: dict[str, Any]

    @property
    def selector(self) -> str:
        return f"{self.sample_id}/{self.long_id}"

    @property
    def s3_url(self) -> str | None:
        if self.selected_origin is None:
            return None
        for root in self.selected_origin.get("access_roots") or ():
            if root.get("type") == "s3" and root.get("url"):
                return root["url"].rstrip("/") + "/" + self.selected_origin.get("path", "").lstrip("/")
        return None


def cache_paths(config: ManagerConfig) -> tuple[Path, Path]:
    root = config.resolved_path("cache_dir", required=True)
    assert root is not None
    return root / "catalog" / "metadata.json", root / "catalog" / "metadata.cache.json"


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_cache(config: ManagerConfig) -> CatalogCache | None:
    document_path, metadata_path = cache_paths(config)
    if not document_path.is_file() or not metadata_path.is_file():
        return None
    try:
        raw = document_path.read_bytes()
        document = json.loads(raw)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(metadata, dict):
            return None
        if hashlib.sha256(raw).hexdigest() != metadata.get("sha256"):
            return None
        return CatalogCache(document, metadata)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_catalog(
    config: ManagerConfig,
    *,
    force_refresh: bool = False,
    allow_network: bool = True,
    now: float | None = None,
    timeout: float = 30.0,
) -> CatalogCache:
    cached = _read_cache(config)
    now = time.time() if now is None else now
    validated = float((cached.metadata if cached else {}).get("validated_unix", 0.0))
    stale = cached is None or now - validated >= config.catalog_max_age_seconds
    if not force_refresh and (not stale or not allow_network):
        if cached is None:
            raise FileNotFoundError("catalog is not cached; run 'las_manager fetch'")
        return cached
    headers = {"Accept-Encoding": "gzip", "User-Agent": "las_manager/0.1"}
    if cached:
        if cached.metadata.get("etag"):
            headers["If-None-Match"] = cached.metadata["etag"]
        if cached.metadata.get("last_modified"):
            headers["If-Modified-Since"] = cached.metadata["last_modified"]
    try:
        try:
            response = urlopen(Request(config.catalog_url, headers=headers), timeout=timeout)
        except HTTPError as error:
            if error.code != 304 or cached is None:
                raise
            metadata = dict(cached.metadata)
            metadata.update(validated_at=_utc_now(), validated_unix=now, last_refresh_error=None)
            _atomic_bytes(cache_paths(config)[1], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
            return CatalogCache(cached.document, metadata)
        with response:
            raw = response.read()
            if response.headers.get("Content-Encoding", "").lower() == "gzip" or raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            document = json.loads(raw)
            if not isinstance(document, dict) or not isinstance(document.get("samples"), dict):
                raise ValueError("catalog root must contain an object-valued 'samples' field")
            canonical = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            metadata = {
                "schema_version": 1,
                "url": config.catalog_url,
                "fetched_at": _utc_now(),
                "validated_at": _utc_now(),
                "validated_unix": now,
                "sha256": hashlib.sha256(canonical).hexdigest(),
                "etag": response.headers.get("ETag"),
                "last_modified": response.headers.get("Last-Modified"),
                "last_refresh_error": None,
            }
            document_path, metadata_path = cache_paths(config)
            _atomic_bytes(document_path, canonical)
            _atomic_bytes(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
            return CatalogCache(document, metadata)
    except Exception as error:
        if cached is None:
            raise RuntimeError(f"catalog refresh failed and no valid cache exists: {error}") from error
        warning = f"catalog refresh failed; using cached catalog: {error}"
        metadata = dict(cached.metadata)
        metadata["last_refresh_error"] = str(error)
        try:
            _atomic_bytes(cache_paths(config)[1], (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
        except OSError:
            pass
        return CatalogCache(cached.document, metadata, warning)


def _iter_values(value: Any) -> Iterable[dict[str, Any]]:
    values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return (item for item in values if isinstance(item, dict))


def index_volumes(cache: CatalogCache) -> list[VolumeRecord]:
    records: list[VolumeRecord] = []
    for sample_key, sample_entry in sorted(cache.document.get("samples", {}).items()):
        if not isinstance(sample_entry, dict):
            continue
        sample_meta = sample_entry.get("sample") if isinstance(sample_entry.get("sample"), dict) else {}
        sample_id = str(sample_meta.get("id") or sample_key)
        for volume in _iter_values(sample_entry.get("volumes")):
            data_entries = list(_iter_values(volume.get("data")))
            ome_entries = [entry for entry in data_entries if entry.get("type") == "ome-zarr"]
            origins = tuple(
                origin
                for entry in ome_entries
                for origin in _iter_values(entry.get("origins"))
            )
            selected = next((origin for origin in origins if any(root.get("type") == "s3" for root in _iter_values(origin.get("access_roots")))), None)
            properties = volume.get("properties") if isinstance(volume.get("properties"), dict) else {}
            license_value = properties.get("license")
            records.append(VolumeRecord(
                sample_id=str(volume.get("sample_id") or sample_id),
                volume_id=str(volume.get("id") or ""),
                long_id=str(volume.get("long_id") or volume.get("id") or ""),
                shape=tuple(int(v) for v in properties.get("shape", ()) if isinstance(v, (int, float))),
                pixel_size_um=float(properties["pixel_size_um"]) if properties.get("pixel_size_um") is not None else None,
                data_format=str(properties["data_format"]) if properties.get("data_format") is not None else None,
                license=dict(license_value) if isinstance(license_value, dict) else None,
                origins=origins,
                selected_origin=selected,
                catalog_sha256=str(cache.metadata.get("sha256", "")),
                catalog_fetched_at=cache.metadata.get("fetched_at"),
                catalog_metadata=dict(cache.metadata),
                raw=volume,
            ))
    return sorted(records, key=lambda record: (record.sample_id, record.long_id))


def resolve_volume(records: list[VolumeRecord], selector: str) -> VolumeRecord:
    candidates: dict[str, list[VolumeRecord]] = {}
    for record in records:
        for value in (record.selector, record.long_id, record.volume_id):
            candidates.setdefault(value, []).append(record)
    exact = candidates.get(selector, [])
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(_ambiguity("volume", selector, exact))
    matches = {record.selector: record for key, values in candidates.items() if key.startswith(selector) for record in values}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if not matches:
        raise ValueError(f"no volume matches {selector!r}")
    raise ValueError(_ambiguity("volume", selector, matches.values()))


def _ambiguity(kind: str, selector: str, records: Iterable[VolumeRecord]) -> str:
    choices = ", ".join(sorted({record.selector for record in records}))
    return f"ambiguous {kind} selector {selector!r}; matches: {choices}"
