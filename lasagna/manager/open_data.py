"""Validate and atomically stage portable inference bundles for Atlas ingestion."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Iterable, Protocol
from urllib.parse import urlparse

try:
    from inference_provenance import validate_portable_bundle
except ImportError:  # pragma: no cover - monorepo package import mode.
    from lasagna.inference_provenance import validate_portable_bundle

from .config import ManagerConfig
from .runs import atomic_json, resolve_run, utc_now
from .snapshots import SnapshotRecord, index_snapshots


SUPPORTED_LICENSE = "CC BY-NC 4.0"
SUPPORTED_ARTIFACTS = {"fiber3d-prediction", "lasagna"}
INCOMPLETE_MARKER = "_INCOMPLETE"
_RUN_TIMESTAMP = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")


class ObjectStore(Protocol):
    def put_file(self, key: str, path: Path) -> None: ...
    def put_bytes(self, key: str, value: bytes) -> None: ...
    def delete(self, key: str) -> None: ...


@dataclass(frozen=True)
class UploadPlan:
    run_dir: Path
    bundle_dir: Path
    provenance: dict[str, Any]
    model_id: str
    bucket: str
    prefix: str
    files: tuple[str, ...]


class S3ObjectStore:
    def __init__(
        self,
        bucket: str,
        *,
        client: Any | None = None,
        rclone_params: Iterable[str] = (),
    ) -> None:
        if client is None:
            try:
                import boto3
            except ImportError as error:
                raise RuntimeError("staging upload requires boto3") from error
            client = boto3.client("s3")
        self.bucket = bucket
        self.client = client
        self.rclone_params = tuple(rclone_params)

    def put_file(self, key: str, path: Path) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def put_files(self, prefix: str, bundle: Path, files: Iterable[str]) -> None:
        """Upload a fixed bundle inventory concurrently with rclone."""
        paths = tuple(files)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as file_list:
            file_list.write("\n".join(paths))
            file_list.write("\n")
            file_list.flush()
            destination = f":s3:{self.bucket}/{prefix.strip('/')}/"
            command = [
                "rclone", "copy", str(bundle), destination,
                *self.rclone_params,
                "--files-from-raw", file_list.name,
            ]
            subprocess.run(command, check=True)

    def put_bytes(self, key: str, value: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=value)

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_files(bundle: Path) -> tuple[str, ...]:
    files: list[str] = []
    for path in sorted((item for item in bundle.rglob("*") if item.is_file()), key=lambda p: p.as_posix()):
        relative = path.relative_to(bundle).as_posix()
        if relative == INCOMPLETE_MARKER:
            raise ValueError(f"reserved upload filename in bundle: {relative}")
        files.append(relative)
    return tuple(files)


def _parse_staging(value: str, run_uuid: str) -> tuple[str, str]:
    parsed = urlparse(value)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError("upload_staging_s3 must be an s3://bucket[/prefix] URL")
    root = parsed.path.strip("/")
    prefix = "/".join(part for part in (root, "inference", run_uuid) if part)
    return parsed.netloc, prefix


def _matching_snapshot(config: ManagerConfig, model: dict[str, Any]) -> SnapshotRecord:
    expected_hash = str(model.get("sha256") or "").lower()
    if len(expected_hash) != 64:
        raise ValueError("inference model.sha256 is required for Atlas model resolution")
    candidates = [
        record for record in index_snapshots(config, write_cache=False)
        if record.sha256.lower() == expected_hash
    ]
    verified: list[SnapshotRecord] = []
    for record in candidates:
        path = Path(record.path)
        if path.is_file() and _sha256(path) == expected_hash:
            verified.append(record)
    if not verified:
        raise ValueError("no configured snapshot matches inference model.sha256")
    normalized = {
        (record.backend, record.run, record.model_creation_utc)
        for record in verified
    }
    if len(normalized) != 1:
        raise ValueError("matching snapshot copies have conflicting identity metadata")
    expected_run = model.get("run")
    expected_checkpoint = model.get("checkpoint")
    named = [record for record in verified if not expected_checkpoint or record.checkpoint == str(expected_checkpoint)]
    if expected_checkpoint and not named:
        raise ValueError("inference model.checkpoint does not match any byte-identical snapshot copy")
    chosen = sorted(named or verified, key=lambda record: record.path)[0]
    if expected_run and str(expected_run) != chosen.run:
        raise ValueError("inference model.run does not match the hashed snapshot")
    expected_snapshot = f"{chosen.run}/snapshots/{chosen.checkpoint}"
    if model.get("snapshot") is not None and str(model["snapshot"]) != expected_snapshot:
        raise ValueError("inference model.snapshot does not match the hashed snapshot")
    expected_architecture = "fiber3d/unet" if chosen.backend == "fiber3d" else "unet"
    if model.get("architecture") is not None and str(model["architecture"]) != expected_architecture:
        raise ValueError("inference model.architecture does not match the hashed snapshot backend")
    if chosen.atlas_model_id and model.get("atlas_model_id"):
        if str(chosen.atlas_model_id)[:14] != str(model["atlas_model_id"])[:14]:
            raise ValueError("inference Atlas model ID conflicts with trusted snapshot metadata")
    if chosen.model_creation_utc and model.get("model_creation_utc"):
        if str(chosen.model_creation_utc) != str(model["model_creation_utc"]):
            raise ValueError("inference model creation time conflicts with trusted snapshot metadata")
    return chosen


def _canonical_model_id(record: SnapshotRecord, model: dict[str, Any]) -> str:
    carried = record.atlas_model_id
    if carried:
        value = str(carried)
        if not re.match(r"^\d{14}(?:-|$)", value):
            raise ValueError(f"invalid checkpoint Atlas model ID: {value!r}")
        return value if "-" in value else f"{value}-lasagna"
    creation = record.model_creation_utc
    if creation:
        try:
            from datetime import datetime, timezone

            stamp = datetime.fromisoformat(str(creation).replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"invalid model creation UTC: {creation!r}") from error
        if stamp.tzinfo is None:
            raise ValueError("model creation time must include a UTC offset")
        stamp = stamp.astimezone(timezone.utc)
        canonical = stamp.strftime("%Y%m%d%H%M%S")
    else:
        matches = _RUN_TIMESTAMP.findall(record.run)
        if len(matches) != 1:
            raise ValueError("snapshot run must contain exactly one YYYYMMDD_HHMMSS UTC timestamp")
        canonical = matches[0].replace("_", "")
    return f"{canonical}-lasagna"


def resolve_model_id(config: ManagerConfig, provenance: dict[str, Any]) -> str:
    model = provenance.get("model") if isinstance(provenance.get("model"), dict) else {}
    record = _matching_snapshot(config, model)
    return _canonical_model_id(record, model)


def validate_inference(
    config: ManagerConfig,
    selector: str,
) -> UploadPlan:
    run_dir, record = resolve_run(config, selector)
    if record.get("status") != "completed" or record.get("lifecycle", {}).get("inference") != "completed":
        raise ValueError(f"inference {record.get('run_name', selector)!r} is not completed")
    bundle = run_dir / str(record.get("artifacts", {}).get("root", "artifacts"))
    provenance = validate_portable_bundle(bundle)
    if provenance.get("artifact_kind") not in SUPPORTED_ARTIFACTS:
        raise ValueError(
            f"unsupported inference artifact {provenance.get('artifact_kind')!r}; "
            f"expected one of {sorted(SUPPORTED_ARTIFACTS)}"
        )
    if provenance.get("run_uuid") != record.get("run_uuid"):
        raise ValueError("run UUID differs between durable record and inference.json")
    source = provenance.get("source") if isinstance(provenance.get("source"), dict) else {}
    license_value = source.get("license") if isinstance(source.get("license"), dict) else {}
    if license_value.get("name") != SUPPORTED_LICENSE:
        raise ValueError(f"publication requires source license {SUPPORTED_LICENSE!r}")
    resolved_model = resolve_model_id(config, provenance)
    bucket, prefix = _parse_staging(config.upload_staging_s3, str(record["run_uuid"]))
    files = _bundle_files(bundle)
    if not files:
        raise ValueError("portable artifact bundle is empty")
    return UploadPlan(
        run_dir=run_dir,
        bundle_dir=bundle,
        provenance=provenance,
        model_id=str(resolved_model),
        bucket=bucket,
        prefix=prefix,
        files=files,
    )


def _key(plan: UploadPlan, relative: str) -> str:
    safe = PurePosixPath(relative)
    if safe.is_absolute() or ".." in safe.parts:
        raise ValueError(f"unsafe upload path: {relative}")
    return f"{plan.prefix}/{safe.as_posix()}"


def stage_upload(plan: UploadPlan, store: ObjectStore) -> str:
    """Upload through rclone, with a marker guarding manager-side commit."""
    marker_key = _key(plan, INCOMPLETE_MARKER)
    store.put_bytes(marker_key, b"")
    bulk_put = getattr(store, "put_files", None)
    if callable(bulk_put):
        bulk_put(plan.prefix, plan.bundle_dir, plan.files)
    else:
        for relative in plan.files:
            store.put_file(_key(plan, relative), plan.bundle_dir / relative)
    store.delete(marker_key)
    return f"s3://{plan.bucket}/{plan.prefix}/"


def _load_atlas_api(config: ManagerConfig):
    atlas = config.resolved_path("atlas_dir", required=True)
    assert atlas is not None
    source = atlas / "vesuvius-atlas-py" / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"Atlas Python source does not exist: {source}")
    import sys
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("vesuvius_atlas.inference_bundle")
        return module.validate_inference_bundle_for_atlas, module.ingest_inference_bundle
    finally:
        try:
            sys.path.remove(str(source))
        except ValueError:
            pass


def upload_inference(
    config: ManagerConfig,
    selector: str,
    *,
    store: ObjectStore | None = None,
    validator: Any | None = None,
    ingester: Any | None = None,
) -> dict[str, Any]:
    plan = validate_inference(config, selector)
    if validator is None or ingester is None:
        atlas_validator, atlas_ingester = _load_atlas_api(config)
        validator = validator or atlas_validator
        ingester = ingester or atlas_ingester
    atlas_dir = config.resolved_path("atlas_dir", required=True)
    atlas_preflight = validator(
        atlas_dir=atlas_dir, bundle_dir=plan.bundle_dir, model_id=plan.model_id,
    )
    record_path = plan.run_dir / "metadata.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    lifecycle = record.setdefault("lifecycle", {})
    lifecycle["staging_upload"] = "uploading"
    record["updated_at"] = utc_now()
    atomic_json(record_path, record)
    try:
        staging_url = stage_upload(
            plan,
            store or S3ObjectStore(
                plan.bucket,
                rclone_params=config.rclone_params,
            ),
        )
        lifecycle["staging_upload"] = "completed"
        record["upload"] = {
            "staging_url": staging_url,
        }
        lifecycle["atlas_ingest"] = "ingesting"
        atomic_json(record_path, record)
        result = ingester(
            atlas_dir=atlas_dir,
            bundle_dir=plan.bundle_dir,
            staging_url=staging_url,
            model_id=plan.model_id,
        )
        lifecycle["atlas_ingest"] = "completed"
        record["atlas"] = result
        record["atlas_preflight"] = atlas_preflight
        record["updated_at"] = utc_now()
        atomic_json(record_path, record)
        return record
    except Exception as error:
        if lifecycle.get("staging_upload") == "uploading":
            lifecycle["staging_upload"] = "failed"
        elif lifecycle.get("atlas_ingest") == "ingesting":
            lifecycle["atlas_ingest"] = "failed"
        record["upload_error"] = str(error)
        record["updated_at"] = utc_now()
        atomic_json(record_path, record)
        raise


def validate_atlas_inference(
    config: ManagerConfig,
    selector: str,
    *,
    validator: Any | None = None,
) -> tuple[UploadPlan, dict[str, Any]]:
    plan = validate_inference(config, selector)
    if validator is None:
        validator, _ingester = _load_atlas_api(config)
    atlas_dir = config.resolved_path("atlas_dir", required=True)
    result = validator(
        atlas_dir=atlas_dir, bundle_dir=plan.bundle_dir, model_id=plan.model_id,
    )
    return plan, result
