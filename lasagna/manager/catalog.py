from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
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


@dataclass(frozen=True)
class LasagnaPredictionRecord:
    sample_id: str
    volume_id: str
    volume_long_id: str
    base_shape_zyx: tuple[int, ...]
    license: dict[str, Any] | None
    model_id: str
    source_level: int | None
    lasagna_version: int | None
    source_to_base: float | None
    output_channels: tuple[str, ...]
    role: str
    validation_error: str | None
    origins: tuple[dict[str, Any], ...]
    selected_origin: dict[str, Any] | None
    selected_access_root: dict[str, Any] | None
    catalog_sha256: str
    catalog_fetched_at: str | None
    raw: dict[str, Any]

    @property
    def selector(self) -> str:
        level = "?" if self.source_level is None else str(self.source_level)
        return f"atlas:{self.model_id or 'unknown'}@L{level}"

    @property
    def s3_url(self) -> str | None:
        value = self.source_url
        if value is None or not (value.startswith("s3://") or value.startswith("s3+")):
            return None
        return value

    @property
    def source_url(self) -> str | None:
        if self.selected_origin is None or self.selected_access_root is None:
            return None
        root = str(self.selected_access_root.get("url") or "")
        path = str(
            self.selected_origin.get("path")
            or self.selected_origin.get("url")
            or self.selected_origin.get("uri")
            or ""
        )
        if not root:
            return None
        if path.startswith(("s3://", "s3+", "http://", "https://")):
            return path
        return root.rstrip("/") + "/" + path.lstrip("/") if path else root

    @property
    def artifact_url(self) -> str | None:
        """Return the exact URL identity used by VC3D's open-data cache."""
        value = self.source_url
        if value is None:
            return None
        open_data = "s3://vesuvius-challenge-open-data/"
        challenge = "s3://vesuvius-challenge/"
        if value.startswith(open_data):
            return (
                "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
                + value[len(open_data):]
            ).rstrip("/")
        if value.startswith(challenge):
            return (
                "https://data.aws.ash2txt.org/samples/"
                + value[len(challenge):]
            ).rstrip("/")
        if value.startswith("s3://") or value.startswith("s3+"):
            if value.startswith("s3://"):
                region = "us-east-1"
                bucket_and_key = value[len("s3://"):]
            else:
                scheme, separator, bucket_and_key = value.partition("://")
                if not separator:
                    return value.rstrip("/")
                region = scheme[len("s3+"):]
            bucket, separator, key = bucket_and_key.partition("/")
            result = f"https://{bucket}.s3.{region}.amazonaws.com"
            if separator:
                result += f"/{key}"
            return result.rstrip("/")
        return value.rstrip("/")


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
            shape_value = properties.get("shape")
            shape_values = shape_value if isinstance(shape_value, (list, tuple)) else ()
            records.append(VolumeRecord(
                sample_id=str(volume.get("sample_id") or sample_id),
                volume_id=str(volume.get("id") or ""),
                long_id=str(volume.get("long_id") or volume.get("id") or ""),
                shape=tuple(int(v) for v in shape_values if isinstance(v, (int, float))),
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


def index_lasagna_predictions(cache: CatalogCache) -> list[LasagnaPredictionRecord]:
    """Index Atlas Lasagna entries and classify their model-declared role."""
    records: list[LasagnaPredictionRecord] = []
    models = cache.document.get("models")
    models = models if isinstance(models, dict) else {}
    for sample_key, sample_entry in sorted(cache.document.get("samples", {}).items()):
        if not isinstance(sample_entry, dict):
            continue
        sample_meta = sample_entry.get("sample") if isinstance(sample_entry.get("sample"), dict) else {}
        sample_id = str(sample_meta.get("id") or sample_key)
        for volume in _iter_values(sample_entry.get("volumes")):
            properties = volume.get("properties") if isinstance(volume.get("properties"), dict) else {}
            shape = properties.get("shape")
            shape_values = shape if isinstance(shape, (list, tuple)) else ()
            license_value = properties.get("license")
            for entry in _iter_values(volume.get("data")):
                if str(entry.get("type") or "").lower() != "lasagna":
                    continue
                parameters = entry.get("parameters") if isinstance(entry.get("parameters"), dict) else {}
                creation = entry.get("creation_info") if isinstance(entry.get("creation_info"), dict) else {}
                model_id = str(parameters.get("model_id") or "")
                level_value = parameters.get("level")
                source_level = (
                    int(level_value)
                    if isinstance(level_value, int) and not isinstance(level_value, bool)
                    and 0 <= level_value <= 5
                    else None
                )
                version_value = creation.get("lasagna_version", creation.get("lasagnaVersion"))
                try:
                    lasagna_version = (
                        int(float(version_value))
                        if isinstance(version_value, (int, float, str))
                        and not isinstance(version_value, bool)
                        else None
                    )
                except (TypeError, ValueError, OverflowError):
                    lasagna_version = None
                source_to_base_value = creation.get("source_to_base", creation.get("sourceToBase"))
                try:
                    source_to_base = (
                        float(source_to_base_value)
                        if isinstance(source_to_base_value, (int, float, str))
                        and not isinstance(source_to_base_value, bool)
                        else None
                    )
                except (TypeError, ValueError, OverflowError):
                    source_to_base = None
                if source_to_base is not None and (
                    not math.isfinite(source_to_base) or source_to_base <= 0
                ):
                    source_to_base = None
                model = models.get(model_id) if isinstance(models.get(model_id), dict) else None
                model_properties = model.get("properties") if isinstance(model, dict) and isinstance(model.get("properties"), dict) else {}
                raw_channels = model_properties.get("output_channels")
                channels = tuple(str(value) for value in raw_channels) if (
                    isinstance(raw_channels, list) and all(isinstance(value, str) for value in raw_channels)
                ) else ()
                channel_set = set(channels)
                error = None
                if not model_id:
                    error = "parameters.model_id is missing"
                elif model is None:
                    error = f"model {model_id!r} is missing from the exact catalogue model index"
                elif source_level is None:
                    error = "parameters.level must be an integer from 0 through 5"
                elif not (
                    len(shape_values) == 3
                    and all(
                        isinstance(value, int) and not isinstance(value, bool) and value > 0
                        for value in shape_values
                    )
                ):
                    error = "parent volume properties.shape must contain three positive integers"
                elif ("lasagna_version" in creation or "lasagnaVersion" in creation) and lasagna_version is None:
                    error = "creation_info.lasagna_version is invalid"
                elif ("source_to_base" in creation or "sourceToBase" in creation) and source_to_base is None:
                    error = "creation_info.source_to_base is invalid"
                elif not channels:
                    error = f"model {model_id!r} has no valid output_channels"
                if {"presence", "nx", "ny"} <= channel_set:
                    role = "fiber"
                elif {"grad_mag", "nx", "ny"} <= channel_set and "presence" not in channel_set:
                    role = "normal"
                else:
                    role = "invalid"
                    if error is None:
                        error = (
                            f"model {model_id!r} channels do not identify Fiber "
                            "presence/nx/ny or regular Lasagna grad_mag/nx/ny"
                        )
                origins = tuple(_iter_values(entry.get("origins")))
                selected = None
                selected_root = None
                for origin in origins:
                    for root in _iter_values(origin.get("access_roots")):
                        if (
                            str(root.get("usage") or "").lower() == "public-read"
                            and root.get("url")
                        ):
                            selected = origin
                            selected_root = root
                            break
                    if selected is not None:
                        break
                if selected is None and error is None:
                    error = "no public-read origin is available"
                    role = "invalid"
                records.append(LasagnaPredictionRecord(
                    sample_id=str(volume.get("sample_id") or sample_id),
                    volume_id=str(volume.get("id") or ""),
                    volume_long_id=str(volume.get("long_id") or volume.get("id") or ""),
                    base_shape_zyx=tuple(int(value) for value in shape_values if isinstance(value, (int, float))),
                    license=dict(license_value) if isinstance(license_value, dict) else None,
                    model_id=model_id,
                    source_level=source_level,
                    lasagna_version=lasagna_version,
                    source_to_base=source_to_base,
                    output_channels=channels,
                    role=role,
                    validation_error=error,
                    origins=origins,
                    selected_origin=selected,
                    selected_access_root=selected_root,
                    catalog_sha256=str(cache.metadata.get("sha256", "")),
                    catalog_fetched_at=cache.metadata.get("fetched_at"),
                    raw=entry,
                ))
    return sorted(records, key=lambda record: (
        record.sample_id, record.volume_id, record.role,
        record.model_id, -1 if record.source_level is None else record.source_level,
    ))


def normal_predictions_for_volume(
    records: Iterable[LasagnaPredictionRecord], *, sample_id: str, volume_id: str,
) -> list[LasagnaPredictionRecord]:
    return [
        record for record in records
        if record.sample_id == sample_id and record.volume_id == volume_id
        and record.role == "normal" and record.validation_error is None
    ]


def resolve_lasagna_prediction(
    records: Iterable[LasagnaPredictionRecord], selector: str, *,
    sample_id: str, volume_id: str, role: str = "normal",
) -> LasagnaPredictionRecord:
    scoped = [
        record for record in records
        if record.sample_id == sample_id and record.volume_id == volume_id
        and record.role == role and record.validation_error is None
    ]
    exact = [record for record in scoped if record.selector == selector]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(_prediction_ambiguity(selector, exact))
    matches = [record for record in scoped if record.selector.startswith(selector)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            f"no published {role} Lasagna prediction matches {selector!r} for "
            f"{sample_id}/{volume_id}"
        )
    raise ValueError(_prediction_ambiguity(selector, matches))


def _prediction_ambiguity(
    selector: str, records: Iterable[LasagnaPredictionRecord],
) -> str:
    choices = ", ".join(sorted({
        f"{record.selector} ({record.s3_url})" for record in records
    }))
    return f"ambiguous published Lasagna selector {selector!r}; matches: {choices}"


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
