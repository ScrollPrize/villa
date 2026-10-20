"""Managed whole-volume Fiberlet preprocessing jobs."""
from __future__ import annotations

import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys
import tempfile
from typing import Any, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .catalog import (
    CatalogCache, LasagnaPredictionRecord, index_lasagna_predictions,
    normal_predictions_for_volume, resolve_lasagna_prediction,
)
from .config import ManagerConfig, config_path
from .runs import (
    SCHEMA_VERSION, _git_revision, _process_matches, _runtime_info,
    _runtime_python, atomic_json, launch_recorded_job, read_runs,
    reserve_managed_run, resolve_run, utc_now,
)
from .tmux import Tmux


_MANAGER_OWNED_OPTIONS = {
    "--anchor-cache",
    "--normal-manifest",
    "--source-context",
    "--remote-cache-dir",
}


def _validate_native_options(values: Sequence[str], *, source: str) -> None:
    for value in values:
        option = str(value).split("=", 1)[0]
        if option in _MANAGER_OWNED_OPTIONS:
            raise ValueError(
                f"{source} cannot override manager-owned Fiberlet option {option}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _manifest_channels(manifest: dict[str, Any]) -> set[str]:
    return {
        str(channel)
        for group in manifest.get("groups", {}).values()
        if isinstance(group, dict)
        for channel in group.get("channels", ())
        if isinstance(channel, str)
    }


def _validate_prediction_manifest(
    manifest: dict[str, Any], *, expected_kind: str, label: str,
) -> None:
    channels = _manifest_channels(manifest)
    required = (
        {"presence", "nx", "ny"}
        if expected_kind == "fiber3d-prediction"
        else {"grad_mag", "nx", "ny"}
    )
    missing = sorted(required - channels)
    if missing:
        raise ValueError(f"{label} is missing channels: {', '.join(missing)}")
    if expected_kind == "lasagna" and "presence" in channels:
        raise ValueError(f"{label} is a Fiber prediction, not regular Lasagna normals")
    crops = manifest.get("crops") or (
        [] if manifest.get("crop_xyzwhd") is None else [manifest["crop_xyzwhd"]]
    )
    if crops:
        raise ValueError("whole-volume Fiberlet preprocessing does not accept cropped prediction bundles")
    shape = manifest.get("base_shape_zyx")
    if not isinstance(shape, list) or len(shape) != 3 or not all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in shape
    ):
        raise ValueError(f"{label} has no valid base_shape_zyx")
    source_to_base = manifest.get("source_to_base")
    if (
        not isinstance(source_to_base, (int, float))
        or isinstance(source_to_base, bool)
        or not math.isfinite(float(source_to_base))
        or source_to_base <= 0
    ):
        raise ValueError(f"{label} has no valid source_to_base")


def _completed_dependency(
    config: ManagerConfig, selector: str, *, expected_kind: str,
) -> dict[str, Any]:
    run_dir, record = resolve_run(config, selector)
    phase = str(record.get("active_lifecycle_phase") or "inference")
    if record.get("status") != "completed" or record.get("lifecycle", {}).get(phase) != "completed":
        raise ValueError(f"input run {record.get('run_name', selector)!r} is not completed")
    if record.get("artifact_kind") != expected_kind:
        raise ValueError(
            f"input run {record.get('run_name', selector)!r} has artifact_kind "
            f"{record.get('artifact_kind')!r}, expected {expected_kind!r}"
        )
    artifacts = record.get("artifacts") if isinstance(record.get("artifacts"), dict) else {}
    bundle = run_dir / str(artifacts.get("root", "artifacts"))
    provenance_path = run_dir / str(artifacts.get("provenance", "artifacts/inference.json"))
    provenance = _load_json(provenance_path, "portable provenance")
    if provenance.get("status") != "completed" or provenance.get("artifact_kind") != expected_kind:
        raise ValueError(f"input run {record.get('run_name', selector)!r} has incomplete portable provenance")
    manifest_entries = [
        entry for entry in provenance.get("artifacts", ())
        if isinstance(entry, dict) and entry.get("kind") == "manifest"
    ]
    if len(manifest_entries) != 1 or not isinstance(manifest_entries[0].get("path"), str):
        raise ValueError(f"input run {record.get('run_name', selector)!r} must contain exactly one manifest")
    relative = Path(manifest_entries[0]["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("input manifest path is not portable")
    manifest_path = bundle / relative
    manifest = _load_json(manifest_path, "Lasagna manifest")
    _validate_prediction_manifest(
        manifest, expected_kind=expected_kind,
        label=f"input run {record.get('run_name', selector)!r}",
    )
    inference = provenance.get("inference") if isinstance(provenance.get("inference"), dict) else {}
    if inference.get("crop_xyzwhd_base") is not None:
        raise ValueError("whole-volume Fiberlet preprocessing does not accept cropped prediction bundles")
    source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    model = provenance.get("model") if isinstance(provenance.get("model"), dict) else {}
    source_scale = provenance.get("source_scale") if isinstance(provenance.get("source_scale"), dict) else {}
    return {
        "run_dir": run_dir,
        "run_name": str(record.get("run_name") or run_dir.name),
        "run_uuid": str(record.get("run_uuid") or ""),
        "dependency_kind": "manager-run",
        "artifact_kind": expected_kind,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "source": source,
        "model": model,
        "source_scale": source_scale,
    }


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _remote_bytes(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "las_manager/0.1"}), timeout=30.0) as response:
        return response.read()


def _discover_published_manifest(
    record: LasagnaPredictionRecord,
) -> str:
    from lasagna.scripts.download_omezarr import list_s3_prefix, parse_s3_uri

    artifact_s3 = record.s3_url
    artifact_url = record.artifact_url
    if not artifact_s3 or not artifact_url:
        raise ValueError(f"published prediction {record.selector} has no usable S3 origin")
    bucket, prefix = parse_s3_uri(artifact_s3)
    prefix = prefix.rstrip("/") + "/"
    try:
        entries = list_s3_prefix(
            bucket, prefix, anon=True, region="us-east-1",
        )
    except Exception as error:
        raise RuntimeError(
            f"could not list published prediction {record.selector} at {artifact_s3}: {error}"
        ) from error
    keys = []
    for key in entries:
        relative = key[len(prefix):] if key.startswith(prefix) else ""
        if relative and "/" not in relative and relative.endswith(".lasagna.json"):
            keys.append(key)
    keys = sorted(set(keys))
    if len(keys) != 1:
        raise ValueError(
            f"published prediction {record.selector} must contain exactly one root "
            f".lasagna.json; found {len(keys)} below {artifact_s3}"
        )
    return artifact_url.rstrip("/") + "/" + keys[0][len(prefix):]


def _safe_cache_component(value: str) -> str:
    value = "".join(
        character if character.isascii() and (
            character.isalnum() or character in "-_ ."
        ) else "_"
        for character in value
    ).replace(" ", "_")
    return value.lstrip("._") or "unnamed"


def _fnv1a64(value: str) -> str:
    result = 14695981039346656037
    for byte in value.encode("utf-8"):
        result ^= byte
        result = (result * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def _vc_lasagna_cache_dir(
    config: ManagerConfig, *, sample_id: str, volume_id: str,
    artifact_url: str, model_id: str, lasagna_version: int | None,
    source_to_base: float | None, base_shape_zyx: Sequence[int],
) -> Path:
    if (
        not sample_id or not volume_id or not artifact_url or not model_id
        or not isinstance(lasagna_version, (int, type(None)))
        or isinstance(lasagna_version, bool)
        or not isinstance(source_to_base, (int, float, type(None)))
        or isinstance(source_to_base, bool)
        or len(base_shape_zyx) != 3
        or not all(
            isinstance(extent, int) and not isinstance(extent, bool) and extent > 0
            for extent in base_shape_zyx
        )
    ):
        raise ValueError("published Lasagna cache identity is incomplete or invalid")
    cache_root = config.resolved_path("cache_dir", required=True)
    assert cache_root is not None
    identity = "\n".join([
        artifact_url,
        model_id,
        str(-1 if lasagna_version is None else lasagna_version),
        "-1" if source_to_base is None else format(source_to_base, ".17g"),
        *(str(int(extent)) for extent in base_shape_zyx),
    ])
    return (
        cache_root / "open_data" / "lasagna" /
        _safe_cache_component(sample_id) / _safe_cache_component(volume_id) /
        _fnv1a64(identity)
    )


def _published_cache_values(record: LasagnaPredictionRecord) -> dict[str, Any]:
    artifact_url = record.artifact_url
    if artifact_url is None:
        raise ValueError(f"published prediction {record.selector} has no usable remote origin")
    return {
        "sample_id": record.sample_id,
        "volume_id": record.volume_id,
        "artifact_url": artifact_url,
        "model_id": record.model_id,
        "source_level": record.source_level,
        "lasagna_version": record.lasagna_version,
        "source_to_base": record.source_to_base,
        "base_shape_zyx": list(record.base_shape_zyx),
    }


def _validate_published_manifest(
    manifest: dict[str, Any], *, cache_values: dict[str, Any], label: str,
) -> None:
    _validate_prediction_manifest(manifest, expected_kind="lasagna", label=label)
    if manifest.get("version") != 2:
        raise ValueError(f"{label} must use Lasagna manifest version 2")
    outer_version = cache_values.get("lasagna_version")
    if outer_version is not None and manifest.get("version") != outer_version:
        raise ValueError(f"{label} disagrees with catalogue creation_info.lasagna_version")
    outer_source_to_base = cache_values.get("source_to_base")
    if outer_source_to_base is not None and abs(
        float(manifest["source_to_base"]) - float(outer_source_to_base)
    ) > 1.0e-12:
        raise ValueError(f"{label} disagrees with catalogue creation_info.source_to_base")
    outer_shape = cache_values.get("base_shape_zyx")
    if not isinstance(outer_shape, list) or len(outer_shape) != 3 or any(
        abs(int(left) - int(right)) > 1
        for left, right in zip(outer_shape, manifest["base_shape_zyx"], strict=True)
    ):
        raise ValueError(f"{label} base_shape_zyx does not match its catalogue parent volume")


def _decode_published_manifest(raw: bytes, *, url: str) -> dict[str, Any]:
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"published normal manifest is invalid JSON: {url}: {error}") from error
    if not isinstance(manifest, dict):
        raise ValueError(f"published normal manifest must contain an object: {url}")
    return manifest


def _validate_remote_group_descriptors(
    manifest: dict[str, Any], *, directory: Path, artifact_url: str,
) -> None:
    base_shape = manifest["base_shape_zyx"]
    source_to_base = float(manifest["source_to_base"])
    seen: set[str] = set()
    for name, group in manifest.get("groups", {}).items():
        if not isinstance(group, dict):
            raise ValueError(f"published normal group {name!r} must be an object")
        channels = group.get("channels")
        if not (
            isinstance(channels, list) and len(channels) == 1
            and isinstance(channels[0], str)
        ):
            raise ValueError(f"published normal group {name!r} must describe exactly one channel")
        relative = group.get("zarr")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"published normal group {name!r} has no Zarr location")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"published normal group {name!r} has an unsafe Zarr location")
        key = relative_path.as_posix().strip("/")
        if key in seen:
            continue
        seen.add(key)
        descriptor_path = directory / key / ".zarray"
        descriptor_url = artifact_url.rstrip("/") + "/" + key + "/.zarray"
        if descriptor_path.is_file():
            raw = descriptor_path.read_bytes()
        else:
            try:
                raw = _remote_bytes(descriptor_url)
            except Exception as error:
                raise RuntimeError(
                    f"cannot resolve published normal Zarr descriptor {descriptor_url}: {error}"
                ) from error
            _atomic_bytes(descriptor_path, raw)
        try:
            descriptor = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"published normal Zarr descriptor is invalid JSON: {descriptor_url}: {error}"
            ) from error
        if not isinstance(descriptor, dict) or descriptor.get("dtype") != "|u1":
            raise ValueError(f"published normal Zarr descriptor is not uint8: {descriptor_url}")
        shape = descriptor.get("shape")
        chunks = descriptor.get("chunks")
        if not all(
            isinstance(values, list) and len(values) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values)
            for values in (shape, chunks)
        ):
            raise ValueError(f"published normal Zarr descriptor is not a valid 3D array: {descriptor_url}")
        scaledown = group.get("scaledown")
        if (
            not isinstance(scaledown, int) or isinstance(scaledown, bool)
            or not 0 <= scaledown <= 30
        ):
            raise ValueError(f"published normal group {name!r} has an invalid scaledown")
        spacing = float(1 << scaledown) * source_to_base
        for extent, actual, chunk in zip(base_shape, shape, chunks, strict=True):
            expected = math.ceil(float(extent) / spacing)
            if actual < expected or actual > expected + chunk:
                raise ValueError(
                    f"published normal Zarr shape is incompatible with base_shape_zyx: {descriptor_url}"
                )


def _prepare_published_manifest(
    config: ManagerConfig, *, cache_values: dict[str, Any],
    record: LasagnaPredictionRecord | None = None,
    manifest_url: str | None = None, expected_sha256: str | None = None,
) -> tuple[Path, str, bytes]:
    source_level = cache_values.get("source_level")
    if (
        not isinstance(source_level, int) or isinstance(source_level, bool)
        or not 0 <= source_level <= 5
    ):
        raise ValueError("published Lasagna source coordinate level is incomplete or invalid")
    artifact_url = str(cache_values.get("artifact_url") or "").rstrip("/")
    directory = _vc_lasagna_cache_dir(
        config,
        sample_id=str(cache_values.get("sample_id") or ""),
        volume_id=str(cache_values.get("volume_id") or ""),
        artifact_url=artifact_url,
        model_id=str(cache_values.get("model_id") or ""),
        lasagna_version=cache_values.get("lasagna_version"),
        source_to_base=cache_values.get("source_to_base"),
        base_shape_zyx=cache_values.get("base_shape_zyx") or (),
    )
    marker_path = directory / "lasagna-remote.json"
    if marker_path.is_file():
        marker = _load_json(marker_path, "VC3D Lasagna remote marker")
        filename = marker.get("manifest_file")
        if not (
            marker.get("artifact_url") == artifact_url
            and marker.get("sample_id") == cache_values["sample_id"]
            and marker.get("volume_id") == cache_values["volume_id"]
            and marker.get("model_id") == cache_values["model_id"]
            and isinstance(filename, str)
            and Path(filename).name == filename
        ):
            raise ValueError(
                f"VC3D Lasagna remote marker does not match catalogue identity: {marker_path}"
            )
        cached_path = directory / filename
        if cached_path.is_file():
            raw = cached_path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if expected_sha256 is not None and digest != expected_sha256:
                raise ValueError(
                    "cached published normal manifest differs from the recorded "
                    f"dependency: expected {expected_sha256}, got {digest}"
                )
            cached_url = artifact_url + "/" + filename
            if manifest_url is not None and cached_url != manifest_url:
                raise ValueError("cached published normal manifest URL differs from the recorded dependency")
            cached_manifest = _decode_published_manifest(raw, url=cached_url)
            _validate_published_manifest(
                cached_manifest, cache_values=cache_values,
                label="cached published normal manifest",
            )
            _validate_remote_group_descriptors(
                cached_manifest, directory=directory, artifact_url=artifact_url,
            )
            marker_changed = False
            if marker.get("anonymous") is not True:
                marker["anonymous"] = True
                marker_changed = True
            if marker.get("source_coordinate_level") != cache_values["source_level"]:
                marker["source_coordinate_level"] = cache_values["source_level"]
                marker_changed = True
            if marker_changed:
                atomic_json(marker_path, marker)
            return cached_path, cached_url, raw

    if manifest_url is None:
        if record is None:
            raise ValueError("cannot discover a published manifest without its catalogue record")
        manifest_url = _discover_published_manifest(record)
    raw = _remote_bytes(manifest_url)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"published normal manifest changed: expected {expected_sha256}, got {digest}"
        )
    filename = Path(urlparse(manifest_url).path).name
    if not filename.endswith(".lasagna.json"):
        raise ValueError(f"published normal manifest URL has an invalid filename: {manifest_url}")
    downloaded_manifest = _decode_published_manifest(raw, url=manifest_url)
    _validate_published_manifest(
        downloaded_manifest, cache_values=cache_values,
        label="downloaded published normal manifest",
    )
    _validate_remote_group_descriptors(
        downloaded_manifest, directory=directory, artifact_url=artifact_url,
    )
    manifest_path = directory / filename
    _atomic_bytes(manifest_path, raw)
    marker = {
        "version": 1,
        "anonymous": True,
        "artifact_url": artifact_url,
        "sample_id": cache_values["sample_id"],
        "volume_id": cache_values["volume_id"],
        "model_id": cache_values["model_id"],
        "source_coordinate_level": cache_values["source_level"],
        "manifest_file": filename,
    }
    if cache_values.get("lasagna_version") is not None:
        marker["lasagna_version"] = cache_values["lasagna_version"]
    if cache_values.get("source_to_base") is not None:
        marker["source_to_base"] = cache_values["source_to_base"]
    if cache_values.get("base_shape_zyx"):
        marker["base_shape_zyx"] = cache_values["base_shape_zyx"]
    atomic_json(marker_path, marker)
    return manifest_path, manifest_url, raw


def _published_dependency(
    config: ManagerConfig, record: LasagnaPredictionRecord,
) -> dict[str, Any]:
    cache_values = _published_cache_values(record)
    manifest_path, manifest_url, raw = _prepare_published_manifest(
        config, cache_values=cache_values, record=record,
    )
    manifest = _decode_published_manifest(raw, url=manifest_url)
    _validate_published_manifest(
        manifest, cache_values=cache_values,
        label=f"published prediction {record.selector}",
    )
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    return {
        "run_dir": None,
        "run_name": record.selector,
        "dependency_kind": "atlas",
        "artifact_kind": "lasagna",
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest": manifest,
        "source": {
            "sample_id": record.sample_id,
            "volume_id": record.volume_id,
            "license": record.license,
            "requested_group": record.source_level,
        },
        "model": {"atlas_model_id": record.model_id},
        "source_scale": {"requested_group": record.source_level},
        "catalog": {
            "sha256": record.catalog_sha256,
            "fetched_at": record.catalog_fetched_at,
        },
        "artifact_url": cache_values["artifact_url"],
        "manifest_url": manifest_url,
        "lasagna_version": record.lasagna_version,
        "source_to_base": record.source_to_base,
        "base_shape_zyx": list(record.base_shape_zyx),
    }


def _restore_published_dependency(
    config: ManagerConfig, dependency: dict[str, Any], source: dict[str, Any],
) -> dict[str, Any]:
    manifest_url = str(dependency.get("manifest_url") or "")
    artifact_url = str(dependency.get("artifact_url") or "")
    expected_sha256 = str(dependency.get("manifest_sha256") or "")
    if not manifest_url or not artifact_url or len(expected_sha256) != 64:
        raise ValueError("recorded Atlas normal dependency is incomplete")
    cache_values = {
        "sample_id": source.get("sample_id"),
        "volume_id": source.get("volume_id"),
        "artifact_url": artifact_url,
        "model_id": dependency.get("model_id"),
        "source_level": dependency.get("requested_group"),
        "lasagna_version": dependency.get("lasagna_version"),
        "source_to_base": dependency.get("source_to_base"),
        "base_shape_zyx": dependency.get("base_shape_zyx"),
    }
    manifest_path, _resolved_url, raw = _prepare_published_manifest(
        config, cache_values=cache_values, manifest_url=manifest_url,
        expected_sha256=expected_sha256,
    )
    manifest = json.loads(raw)
    _validate_prediction_manifest(
        manifest, expected_kind="lasagna", label="recorded published normal prediction",
    )
    model_id = dependency.get("model_id")
    requested_group = dependency.get("requested_group")
    return {
        "run_dir": None,
        "run_name": str(dependency.get("run_name") or f"atlas:{model_id}@L{requested_group}"),
        "dependency_kind": "atlas",
        "artifact_kind": "lasagna",
        "manifest_path": manifest_path,
        "manifest_sha256": expected_sha256,
        "manifest": manifest,
        "source": source,
        "model": {"atlas_model_id": model_id},
        "source_scale": {"requested_group": requested_group},
        "catalog": dependency.get("catalog") if isinstance(dependency.get("catalog"), dict) else {},
        "artifact_url": artifact_url,
        "manifest_url": manifest_url,
        "lasagna_version": dependency.get("lasagna_version"),
        "source_to_base": dependency.get("source_to_base"),
        "base_shape_zyx": dependency.get("base_shape_zyx"),
    }


def _select_published_normal(
    config: ManagerConfig, cache: CatalogCache, fiber: dict[str, Any],
    selector: str | None,
) -> dict[str, Any]:
    source = fiber["source"]
    sample_id = str(source.get("sample_id") or "")
    volume_id = str(source.get("volume_id") or "")
    records = index_lasagna_predictions(cache)
    if selector is not None:
        record = resolve_lasagna_prediction(
            records, selector, sample_id=sample_id, volume_id=volume_id,
        )
    else:
        candidates = normal_predictions_for_volume(
            records, sample_id=sample_id, volume_id=volume_id,
        )
        if len(candidates) > 1:
            choices = ", ".join(record.selector for record in candidates)
            raise ValueError(
                f"multiple published regular Lasagna normal predictions match "
                f"{sample_id}/{volume_id}; pass one explicitly: {choices}"
            )
        if not candidates:
            invalid = [
                record.validation_error for record in records
                if record.sample_id == sample_id and record.volume_id == volume_id
                and record.validation_error
            ]
            detail = f" Invalid catalogue entries: {'; '.join(sorted(set(invalid)))}" if invalid else ""
            raise ValueError(
                f"no published regular Lasagna normal prediction exists for "
                f"{sample_id}/{volume_id}; Fiberlet processing requires "
                f"grad_mag/nx/ny from a regular Lasagna model.{detail}"
            )
        record = candidates[0]
    return _published_dependency(config, record)


def resolve_fiberlet_inputs(
    config: ManagerConfig, fiber_selector: str, normal_selector: str | None = None,
    *, catalog_cache: CatalogCache | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fiber = _completed_dependency(config, fiber_selector, expected_kind="fiber3d-prediction")
    if normal_selector is not None and not normal_selector.startswith("atlas:"):
        normal = _completed_dependency(config, normal_selector, expected_kind="lasagna")
    else:
        if catalog_cache is None:
            from .catalog import get_catalog

            catalog_cache = get_catalog(config)
            if catalog_cache.warning:
                print(f"warning: {catalog_cache.warning}", file=sys.stderr)
        normal = _select_published_normal(
            config, catalog_cache, fiber, normal_selector,
        )
    _validate_fiberlet_pair(fiber, normal)
    return fiber, normal


def _validate_fiberlet_pair(
    fiber: dict[str, Any], normal: dict[str, Any],
) -> None:
    for field in ("sample_id", "volume_id"):
        if not fiber["source"].get(field) or fiber["source"].get(field) != normal["source"].get(field):
            raise ValueError(f"Fiber and normal inputs must identify the same source {field}")
    fiber_shape = fiber["manifest"].get("base_shape_zyx")
    normal_shape = normal["manifest"].get("base_shape_zyx")
    if not fiber_shape or not normal_shape or any(
        abs(int(left) - int(right)) > 1
        for left, right in zip(fiber_shape, normal_shape, strict=True)
    ):
        raise ValueError(
            "Fiber and normal manifests must identify base_shape_zyx values "
            "that differ by at most one voxel per axis"
        )


def _resolve_binary(config: ManagerConfig) -> Path:
    configured = config.fiberlet_binary.strip()
    if configured:
        candidate = Path(os.path.expandvars(configured)).expanduser()
        if not candidate.is_absolute():
            candidate = config_path().parent / candidate
        candidate = candidate.resolve()
    else:
        venv = config.resolved_path("venv")
        installed = venv / "bin" / "vc_fiberlets" if venv is not None else None
        found = (
            str(installed)
            if installed is not None and installed.is_file()
            else shutil.which("vc_fiberlets")
        )
        if found is None:
            raise FileNotFoundError(
                "vc_fiberlets is not installed in the configured venv or on PATH; "
                "set fiberlet_binary in the las_manager config"
            )
        candidate = Path(found).resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(f"configured Fiberlet executable is not executable: {candidate}")
    return candidate


def _dependency_identity(value: dict[str, Any]) -> dict[str, Any]:
    model = value["model"]
    identity = {
        "dependency_kind": value.get("dependency_kind", "manager-run"),
        "artifact_kind": value["artifact_kind"],
        "manifest_sha256": value["manifest_sha256"],
        "model_id": model.get("atlas_model_id"),
        "model_sha256": model.get("sha256"),
        "model_run": model.get("run"),
        "snapshot": model.get("snapshot"),
        "requested_group": value["source_scale"].get("requested_group", value["source"].get("requested_group")),
        **({
            "catalog": value.get("catalog", {}),
            "artifact_url": value.get("artifact_url"),
            "manifest_url": value.get("manifest_url"),
            "lasagna_version": value.get("lasagna_version"),
            "source_to_base": value.get("source_to_base"),
            "base_shape_zyx": value.get("base_shape_zyx"),
        } if value.get("dependency_kind") == "atlas" else {}),
    }
    if value.get("dependency_kind") != "atlas":
        identity["run_uuid"] = value["run_uuid"]
    return identity


def launch_fiberlet(
    config: ManagerConfig,
    fiber_selector: str,
    normal_selector: str | None = None,
    *,
    original_argv: Sequence[str],
    extra_args: Sequence[str] = (),
    tmux: Tmux | None = None,
    catalog_cache: CatalogCache | None = None,
) -> Path:
    fiber, normal = resolve_fiberlet_inputs(
        config, fiber_selector, normal_selector, catalog_cache=catalog_cache,
    )
    _validate_native_options(config.fiberlet_params, source="fiberlet_params")
    _validate_native_options(extra_args, source="explicit arguments")
    binary = _resolve_binary(config)
    client = tmux or Tmux()
    sample_id = str(fiber["source"]["sample_id"])
    volume_id = str(fiber["source"]["volume_id"])
    run_uuid, run_name, session, run_dir = reserve_managed_run(
        config,
        name_stem=f"{sample_id}-{volume_id}-fiberlets",
        session_prefix="flt",
        tmux=client,
    )
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    cache = run_dir / "cache"
    cache.mkdir()
    output = artifacts / "fiberlets.zarr"
    context_path = run_dir / "source_context.json"
    fiber_identity = _dependency_identity(fiber)
    normal_identity = _dependency_identity(normal)
    source_context = {
        "source_volume": {"sample_id": sample_id, "volume_id": volume_id},
        "fiber_prediction": fiber_identity,
        "normal_prediction": normal_identity,
    }
    atomic_json(context_path, source_context)
    effective_args = (
        "--threads", str(config.fiberlet_threads),
        *config.fiberlet_params,
        *extra_args,
    )
    command = [
        str(binary), "preprocess-volume", str(fiber["manifest_path"]), str(output),
        "--normal-manifest", str(normal["manifest_path"]),
        "--anchor-cache", str(cache / "fiberlets.anchors.zarr"),
        "--source-context", str(context_path),
        *effective_args,
    ]
    python = _runtime_python(config)
    source_license = fiber["source"].get("license")
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job_kind": "fiberlet",
        "active_lifecycle_phase": "fiberlet_preprocess",
        "run_uuid": run_uuid,
        "run_name": run_name,
        "backend": "vc_fiberlets",
        "artifact_kind": "fiberlets",
        "status": "created",
        "created_at": utc_now(),
        "started_at": None,
        "ended_at": None,
        "exit_code": None,
        "pid": None,
        "process_start_time": None,
        "tmux_session": session,
        "tmux_window_id": None,
        "private": {"hostname": socket.gethostname(), "user": getpass.getuser()},
        "manager": {"version": "0.1", **_git_revision(Path(__file__).resolve().parents[2])},
        "runtime": _runtime_info(python),
        "source": {
            "sample_id": sample_id,
            "volume_id": volume_id,
            "license": source_license,
        },
        "dependencies": {
            "fiber_prediction": {**fiber_identity, "run_name": fiber["run_name"]},
            "normal_prediction": {**normal_identity, "run_name": normal["run_name"]},
        },
        "processor": {
            "path": str(binary),
            "sha256": _sha256(binary),
            "arguments": list(effective_args),
        },
        "source_context_path": "source_context.json",
        "command_path": "command.json",
        "log_path": "run.log",
        "artifacts": {
            "root": "artifacts",
            "provenance": "artifacts/fiberlets.json",
            "dataset": "artifacts/fiberlets.zarr",
            "inventory": [],
        },
        "lifecycle": {
            "fiberlet_preprocess": "created",
            "staging_upload": "not_started",
            "atlas_ingest": "not_started",
            "atlas_publication": "not_started",
        },
    }
    command_record = {
        "schema_version": 1,
        "original_argv": list(original_argv),
        "resolved_argv": command,
        "display": " ".join(shlex.quote(value) for value in command),
        "cwd": str(run_dir),
        "venv_activation": f"source {config.resolved_path('venv', required=True)}/bin/activate",
        "prefetch": None,
    }
    return launch_recorded_job(
        config,
        run_dir=run_dir,
        record=record,
        command_record=command_record,
        window_name=f"flt-{sample_id}-{run_uuid[:4]}"[:24],
        tmux=client,
    )


def finalize_fiberlet_provenance(run_dir: Path, record: dict[str, Any]) -> None:
    dataset = run_dir / str(record["artifacts"]["dataset"])
    for relative in (".zgroup", "anchors/.zarray", "prefix/.zarray", "routes/.zarray"):
        if not (dataset / relative).is_file():
            raise ValueError(f"native Fiberlet output is missing {relative}")
    attrs = _load_json(dataset / ".zattrs", "Fiberlet dataset metadata")
    if (
        attrs.get("vc_format") != "fiberlet_dataset"
        or attrs.get("format_version") != 2
        or attrs.get("dataset_kind") != "combined"
    ):
        raise ValueError("native output is not a combined Fiberlet dataset")
    if not isinstance(attrs.get("processing"), dict):
        raise ValueError("native Fiberlet output has no processing contract")
    dataset_fingerprint = str(attrs.get("dataset_fingerprint") or "")
    if len(dataset_fingerprint) != 64 or any(
        value not in "0123456789abcdef" for value in dataset_fingerprint.lower()
    ):
        raise ValueError("native Fiberlet output has an invalid dataset fingerprint")
    sources = attrs.get("sources") if isinstance(attrs.get("sources"), dict) else {}
    for name in ("fiber_prediction", "normal_prediction"):
        stored = sources.get(name) if isinstance(sources.get(name), dict) else {}
        expected = record["dependencies"][name]
        for field in ("run_uuid", "manifest_sha256"):
            if stored.get(field) != expected.get(field):
                raise ValueError(
                    f"native Fiberlet output {name}.{field} differs from the recorded input"
                )
    document = {
        "schema_version": 1,
        "artifact_kind": "fiberlets",
        "status": "completed",
        "run_uuid": record["run_uuid"],
        "generated_at": utc_now(),
        "source": record["source"],
        "dependencies": record["dependencies"],
        "processor": {
            "sha256": record["processor"]["sha256"],
            "arguments": record["processor"]["arguments"],
        },
        "fiberlet_dataset": {
            "dataset_fingerprint": attrs.get("dataset_fingerprint"),
            "algorithm_fingerprint": attrs.get("algorithm_fingerprint"),
            "format_version": attrs.get("format_version"),
        },
        "artifacts": [{
            "kind": "fiberlet-zarr",
            "path": "fiberlets.zarr",
            "metadata_sha256": _sha256(dataset / ".zattrs"),
        }],
    }
    atomic_json(run_dir / str(record["artifacts"]["provenance"]), document)


def resume_fiberlet(
    config: ManagerConfig, selector: str, *, tmux: Tmux | None = None,
) -> Path:
    run_dir, record = resolve_run(config, selector)
    if record.get("job_kind") != "fiberlet":
        raise ValueError(f"run {selector!r} is not a Fiberlet job")
    if _process_matches(record):
        raise ValueError(f"Fiberlet run {record.get('run_name')!r} is still active")
    dependencies = record.get("dependencies", {})
    fiber = dependencies.get("fiber_prediction", {})
    normal = dependencies.get("normal_prediction", {})
    resolved_fiber = _completed_dependency(
        config, str(fiber.get("run_name") or fiber.get("run_uuid")),
        expected_kind="fiber3d-prediction",
    )
    if normal.get("dependency_kind") == "atlas":
        normal_source = {
            **(record.get("source") if isinstance(record.get("source"), dict) else {}),
            "requested_group": normal.get("requested_group"),
        }
        resolved_normal = _restore_published_dependency(
            config, normal, normal_source,
        )
    else:
        resolved_normal = _completed_dependency(
            config, str(normal.get("run_name") or normal.get("run_uuid")),
            expected_kind="lasagna",
        )
    _validate_fiberlet_pair(resolved_fiber, resolved_normal)
    if _dependency_identity(resolved_fiber) != {key: fiber.get(key) for key in _dependency_identity(resolved_fiber)}:
        raise ValueError("Fiber prediction dependency changed since the original run")
    if _dependency_identity(resolved_normal) != {key: normal.get(key) for key in _dependency_identity(resolved_normal)}:
        raise ValueError("normal prediction dependency changed since the original run")
    client = tmux or Tmux()
    session = str(record.get("tmux_session") or "")
    if session and client.has_session(session):
        raise ValueError(f"Fiberlet run {record.get('run_name')!r} still has a live tmux session")
    record.update(status="created", started_at=None, ended_at=None, exit_code=None, pid=None, process_start_time=None, tmux_window_id=None)
    record.setdefault("lifecycle", {})["fiberlet_preprocess"] = "created"
    command_record = _load_json(run_dir / "command.json", "recorded command")
    return launch_recorded_job(
        config,
        run_dir=run_dir,
        record=record,
        command_record=command_record,
        window_name=str(record.get("tmux_window_name") or f"flt-{record['run_uuid'][:4]}"),
        tmux=client,
    )


def fiberlet_runs(config: ManagerConfig) -> list[tuple[Path, dict[str, Any]]]:
    return [item for item in read_runs(config) if item[1].get("job_kind") == "fiberlet"]
