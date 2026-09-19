"""Session-input uploads: transfer, validation, and publication.

``UploadManager`` owns the staging area, the per-upload transfer state, the
content validation for every input kind, and the content-addressed store of
uploaded resume checkpoints. It never sees ``ServiceState``: everything it
needs from the lifecycle orchestrator arrives through the small callback set
in ``UploadEnvironment`` (where the output directory is, whether a session is
loaded and which checkpoint the session is using).

The manager does not know about revision bookkeeping, status
snapshots, the event buffer, artifacts, or the fit session itself; finalize
hands the caller a record and the caller decides what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
from http import HTTPStatus
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import threading
import time
from typing import Callable, Optional

from service_http import (ApiError, TRANSFER_CHUNK_BYTES,
                          is_safe_relative_name)
from fit_session import (PCL_ROLE_CONVENTIONS,
                         validate_checkpoint_container)
from vc3d_fiber_format_adapter import parse_vc3d_fiber_format


MAX_UPLOAD_FILES = 256
UPLOAD_GC_SECONDS = 3600.0
UPLOADED_CHECKPOINTS_KEPT = 3
UPLOADED_CHECKPOINTS_DIRNAME = "uploaded-checkpoints"
MAX_CHECKPOINT_UPLOAD_BYTES = int(os.environ.get(
    "SPIRAL_CHECKPOINT_UPLOAD_MAX_BYTES", 64 * 1024 * 1024 * 1024))
UPLOAD_KINDS = ("patch", "fiber", "pcl", "checkpoint")
STAGING_DIRNAME = ".spiral-upload-staging"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

PCL_ROLE_FILES = {
    role.value: filename for role, filename in PCL_ROLE_CONVENTIONS}


class Upload:
    def __init__(self, upload_id, session_id, kind, role, input_id, manifest, staging_dir):
        self.upload_id = upload_id
        self.session_id = session_id
        self.kind = kind
        self.role = role
        self.input_id = input_id
        self.manifest = manifest
        self.staging_dir = staging_dir
        self.received = {}
        self.record = None
        self.created = time.time()
        self.lock = threading.Lock()
        self.cancelled = False
        self.offsets = {}

    def declared_bytes(self):
        return sum(entry["size"] for entry in self.manifest.values())


def _validate_upload_manifest(value):
    files = value.get("files")
    if not isinstance(files, list) or not files:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Upload manifest lists no files")
    if len(files) > MAX_UPLOAD_FILES:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Upload manifest lists too many files")
    manifest = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Malformed upload manifest entry")
        name = entry.get("name")
        if not is_safe_relative_name(name):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Unsafe upload file name: {name!r}")
        try:
            size = int(entry.get("size"))
            digest = str(entry.get("sha256", "")).lower()
        except (TypeError, ValueError):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Malformed upload manifest entry")
        if size < 0 or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Malformed upload manifest entry")
        if name in manifest:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Duplicate upload file name: {name}")
        manifest[name] = {"size": size, "sha256": digest}
    return manifest


def _validate_patch_content(directory):
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise ApiError(HTTPStatus.BAD_REQUEST, "Patch upload is missing meta.json")
    try:
        with meta_path.open("r", encoding="utf-8") as stream:
            meta = json.load(stream)
    except Exception as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Patch meta.json is invalid JSON: {exc}")
    if meta.get("format") != "tifxyz":
        raise ApiError(HTTPStatus.BAD_REQUEST, "Patch meta.json format must be 'tifxyz'")
    for raster in ("x.tif", "y.tif", "z.tif"):
        if not (directory / raster).is_file():
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Patch upload is missing raster file {raster}")


def _load_single_json(directory, kind):
    json_files = [p for p in directory.rglob("*") if p.is_file()]
    if len(json_files) != 1 or json_files[0].suffix.lower() != ".json":
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       f"A {kind} upload must contain exactly one JSON file")
    try:
        with json_files[0].open("r", encoding="utf-8") as stream:
            return json.load(stream), json_files[0]
    except Exception as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")


def _validate_replacement_document(document, target_collection_id, role=None):
    collections = document.get("collections")
    if not isinstance(collections, dict) \
            or list(collections) != [target_collection_id]:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "A replacement upload must contain exactly its target collection")
    collection = collections[target_collection_id]
    if not isinstance(collection, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "The replacement collection is malformed")
    if ("id" in collection
            and str(collection["id"]) != target_collection_id):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "The replacement collection id does not match its target")
    if collection.get("windings_linked"):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "Linked collections cannot be replaced safely")
    points = collection.get("points")
    if not isinstance(points, dict) or len(points) < 2:
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "A replacement collection needs at least two points")
    expected = [str(index) for index in range(len(points))]
    if set(points) != set(expected):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "Replacement point ids must be contiguous from zero")
    previous_time = None
    for point_id in expected:
        point = points[point_id]
        if not isinstance(point, dict) or point.get("links"):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Linked points cannot be replaced safely")
        position = point.get("p")
        if not isinstance(position, list) or len(position) != 3 \
                or any(not isinstance(value, (int, float))
                       or isinstance(value, bool) or not math.isfinite(value)
                       for value in position):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Replacement point {point_id} has an invalid position")
        creation_time = point.get("creation_time")
        if not isinstance(creation_time, (int, float)) \
                or isinstance(creation_time, bool) \
                or not math.isfinite(creation_time):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Replacement points need numeric creation_time values")
        if previous_time is not None and creation_time <= previous_time:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Replacement creation_time values must strictly increase")
        previous_time = creation_time
        if role == "relative":
            # The fitter drops unannotated points from a partially annotated
            # collection, so a relative replacement must annotate every point.
            winding = point.get("wind_a")
            if not isinstance(winding, (int, float)) \
                    or isinstance(winding, bool) or not math.isfinite(winding):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"Relative-winding replacement point {point_id} needs a "
                    "finite wind_a annotation")


def _validate_upload_content(kind, role, directory):
    if kind == "patch":
        _validate_patch_content(directory)
        return
    if kind == "checkpoint":
        files = [p for p in directory.rglob("*") if p.is_file()]
        if len(files) != 1:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "A checkpoint upload must contain exactly one file")
        try:
            validate_checkpoint_container(files[0])
        except (OSError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid checkpoint: {exc}")
        return
    document, _ = _load_single_json(directory, kind)
    if kind == "fiber":
        if not isinstance(document, dict) or document.get("type") != "vc3d_fiber":
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Fiber uploads must be JSON documents with type 'vc3d_fiber'")
        if document.get("version", 1) == 1:
            return
        try:
            parse_vc3d_fiber_format(document)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Invalid fiber upload: {exc}") from exc
        return
    if kind == "pcl":
        if not isinstance(document, dict) \
                or document.get("vc_pointcollections_json_version") != "1":
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "PCL uploads must be vc_pointcollections_json_version 1 documents")
        if not isinstance(document.get("collections"), dict) or not document["collections"]:
            raise ApiError(HTTPStatus.BAD_REQUEST, "PCL upload contains no collections")
        if role not in PCL_ROLE_FILES:
            raise ApiError(HTTPStatus.BAD_REQUEST, "PCL uploads must declare a valid role")
        return
    raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown input kind {kind!r}")


@dataclass(frozen=True)
class UploadEnvironment:
    """Everything the upload manager may ask the orchestrator about.

    Deliberately absent: the fit session, the status snapshot, the event
    buffer, the artifact registry, and the revision catalog.
    """

    lock: threading.RLock
    #: Root for staging and the uploaded-checkpoint store, or None when the
    #: service has neither a session nor a bound dataset output.
    output_root: Callable[[], Optional[Path]]
    #: Identifier of the loaded session, or None.
    session_id: Callable[[], Optional[str]]
    #: Checkpoint the loaded session resumed from; protected from retention.
    active_checkpoint: Callable[[], str] = lambda: ""
    allowed_kinds: tuple[str, ...] = UPLOAD_KINDS


@dataclass
class FinalizedUpload:
    """Result of finalizing one upload."""

    kind: str
    record: dict
    #: True when this call replayed an already finalized upload.
    replayed: bool = False


class UploadManager:
    """Staging, transfer, validation and publication of session inputs."""

    def __init__(self, environment):
        self.environment = environment
        self.uploads = {}
        # A caller-chosen id survives an ambiguous create response. Keep its
        # manifest and receipt for the entire scope, including cancellation.
        self._begin_requests = {}
        self._begin_results = {}
        self._begin_locks = {}

    @property
    def _lock(self):
        return self.environment.lock

    # ------------------------------------------------------------------
    # Locations
    # ------------------------------------------------------------------

    def staging_root(self):
        root = self.environment.output_root()
        return None if root is None else root / STAGING_DIRNAME

    def checkpoint_root(self):
        root = self.environment.output_root()
        return None if root is None else root / UPLOADED_CHECKPOINTS_DIRNAME

    @staticmethod
    def checkpoint_digest_path(root, digest):
        return root / f"{digest}.ckpt"

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                block = stream.read(TRANSFER_CHUNK_BYTES)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def find_uploaded_checkpoint(self, root, digest, size):
        """Find retained checkpoint content, including pre-v7 named uploads."""
        canonical = self.checkpoint_digest_path(root, digest)
        try:
            if canonical.is_file() and canonical.stat().st_size == size:
                return canonical
        except OSError:
            pass
        if not root.is_dir():
            return None
        for candidate in root.iterdir():
            if candidate == canonical:
                continue
            try:
                if not candidate.is_file() or candidate.stat().st_size != size:
                    continue
                if self._file_sha256(candidate) == digest:
                    return candidate
            except OSError:
                continue
        return None

    @staticmethod
    def checkpoint_record(input_id, path, size, upload_id=None):
        record = {
            "id": input_id,
            "kind": "checkpoint",
            "role": None,
            "path": str(path),
            "bytes": size,
            "state": "uploaded",
        }
        if upload_id is not None:
            record["upload_id"] = upload_id
        return record


    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def begin(self, request):
        request = copy.deepcopy(request)
        upload_id = request.get("upload_id")
        if upload_id is None:
            return self._begin(request)
        if not isinstance(upload_id, str) or not re.fullmatch(r"[0-9a-f]{32}", upload_id):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "upload_id must be 32 lowercase hexadecimal characters")
        try:
            fingerprint = json.dumps(request, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid upload manifest") from exc
        with self._lock:
            previous = self._begin_requests.get(upload_id)
            if previous is None and upload_id in self.uploads:
                raise ApiError(HTTPStatus.CONFLICT, "Upload id already exists")
            if previous is not None and previous != fingerprint:
                raise ApiError(HTTPStatus.CONFLICT,
                               "Upload id was reused with different content")
            self._begin_requests[upload_id] = fingerprint
            lock = self._begin_locks.setdefault(upload_id, threading.Lock())
        # Different transfers remain concurrent; only duplicates wait here.
        with lock:
            with self._lock:
                previous = self._begin_results.get(upload_id)
                if previous is not None:
                    return copy.deepcopy(previous)
            result = self._begin(request)
            with self._lock:
                if result.get("deduplicated"):
                    # Checkpoint reuse still has a reconcilable transfer id
                    # when the caller supplied one. No staging bytes exist.
                    record = dict(result["input"], upload_id=upload_id)
                    upload = Upload(
                        upload_id, self.environment.session_id(), "checkpoint", None,
                        request["id"], _validate_upload_manifest(request),
                        self.staging_root() / upload_id)
                    upload.record = record
                    upload.received = dict.fromkeys(upload.manifest, True)
                    self.uploads[upload_id] = upload
                    result = dict(result, upload_id=upload_id, input=record)
                self._begin_results[upload_id] = copy.deepcopy(result)
            return result

    def _begin(self, request):
        """Start an upload.

        Returns ``{"upload_id": ...}`` for a transfer that must follow, or
        ``{"deduplicated": True, "input": record}`` when identical checkpoint
        content is already retained by the service.
        """
        kind = str(request.get("kind") or "").strip()
        if kind not in UPLOAD_KINDS:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Input kind must be one of patch, fiber, pcl, checkpoint")
        role = request.get("role")
        if kind == "pcl":
            if role not in PCL_ROLE_FILES:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "A PCL upload must declare its role")
        else:
            role = None
        if kind not in self.environment.allowed_kinds:
            raise ApiError(400, "This upload store does not accept that input kind")
        if any(key in request for key in ("operation", "target_collection_id",
                                          "base_source_revision", "base_revision")):
            raise ApiError(400, "Uploads contain bytes only; stage revisions with /session/input-changes")
        if kind != "checkpoint" and not request.get("upload_id"):
            raise ApiError(400, "Input uploads require a stable upload_id")
        input_id = str(request.get("id") or "").strip()
        if not _SAFE_ID.match(input_id):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "The input id must be a single safe path component")
        manifest = _validate_upload_manifest(request)
        declared = sum(entry["size"] for entry in manifest.values())
        if kind == "checkpoint":
            with self._lock:
                # Resume checkpoints are needed before a session exists, so
                # they are service-scoped: allowed whenever an output
                # directory is known (a --dataset launch or a live session).
                output_root = self.environment.output_root()
                if output_root is None:
                    raise ApiError(HTTPStatus.CONFLICT,
                                   "Checkpoint uploads need a --dataset service "
                                   "or an active session")
                if len(manifest) != 1:
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "A checkpoint upload must declare exactly one file")
                if declared > MAX_CHECKPOINT_UPLOAD_BYTES:
                    raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                   "The checkpoint exceeds the upload size limit")
            entry = next(iter(manifest.values()))
            checkpoint_root = output_root / UPLOADED_CHECKPOINTS_DIRNAME
            existing = self.find_uploaded_checkpoint(
                checkpoint_root, entry["sha256"], entry["size"])
            if existing is not None:
                try:
                    os.utime(existing, None)
                except OSError:
                    pass
                return {
                    "accepted": True,
                    "deduplicated": True,
                    "input": self.checkpoint_record(
                        input_id, existing, entry["size"]),
                }
        with self._lock:
            if kind == "checkpoint":
                current_output_root = self.environment.output_root()
                if current_output_root is None or current_output_root != output_root:
                    raise ApiError(HTTPStatus.CONFLICT,
                                   "The checkpoint upload destination changed")
                # Close the race with another request that finalized this
                # digest while the legacy-file scan ran without the state lock.
                canonical = self.checkpoint_digest_path(
                    checkpoint_root, entry["sha256"])
                if canonical.is_file() and canonical.stat().st_size == entry["size"]:
                    os.utime(canonical, None)
                    return {
                        "accepted": True,
                        "deduplicated": True,
                        "input": self.checkpoint_record(
                            input_id, canonical, entry["size"]),
                    }
            upload_id = request.get("upload_id") or secrets.token_hex(16)
            staging = self.staging_root() / upload_id
            upload = Upload(upload_id, self.environment.session_id(), kind,
                            role, input_id, manifest, staging)
            staging.mkdir(parents=True, exist_ok=True)
            self.uploads[upload_id] = upload
        return {"upload_id": upload_id, "accepted": True}

    def get(self, upload_id, *, include_cancelled=False):
        with self._lock:
            upload = self.uploads.get(upload_id)
            # Checkpoint uploads are service-scoped; editable inputs are
            # bound to the workspace they were started for.
            if upload is None or (upload.kind != "checkpoint"
                                  and upload.session_id != self.environment.session_id()):
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown upload")
            if upload.cancelled and not include_cancelled:
                raise ApiError(HTTPStatus.GONE, "Upload was cancelled")
            return upload

    def status(self, upload_id):
        upload = self.get(upload_id, include_cancelled=True)
        with upload.lock:
            return {
                "upload_id": upload_id,
                "state": ("cancelled" if upload.cancelled else
                          "finalized" if upload.record is not None else "transferring"),
                "files": [{"name": name, **entry,
                           "offset": (entry["size"] if name in upload.received
                                      else upload.offsets.get(name, 0)),
                           "received": name in upload.received}
                          for name, entry in upload.manifest.items()],
                "input": copy.deepcopy(upload.record),
            }

    def cancel(self, upload_id):
        upload = self.get(upload_id, include_cancelled=True)
        with upload.lock:
            if upload.record is not None:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A finalized upload cannot be cancelled")
            upload.cancelled = True
            shutil.rmtree(upload.staging_dir, ignore_errors=True)
        return {"upload_id": upload_id, "cancelled": True}

    def receive(self, upload_id, relative_name, stream, length, *, offset=None):
        upload = self.get(upload_id)
        # Finalize/cancel cannot rename or remove the staging directory while
        # a writer still has a file open. Different uploads use different locks.
        with upload.lock:
            if upload.cancelled:
                raise ApiError(HTTPStatus.GONE, "Upload was cancelled")
            if offset is not None:
                return self._receive_chunk(upload, relative_name, stream, length, offset)
            return self._receive(upload, relative_name, stream, length)

    def _receive(self, upload, relative_name, stream, length):
        """Store one declared file.

        The transfer is content addressed, not command addressed: a client
        may repeat a PUT for the same (upload, file) as often as it likes.
        Every attempt is written to a private temporary file, digested, and
        only promoted to the staged name when the bytes match the manifest,
        so a retry converges on exactly the declared content and a truncated
        or corrupted attempt never replaces a good staged file.
        """
        if not is_safe_relative_name(relative_name):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Unsafe upload file name")
        entry = upload.manifest.get(relative_name)
        if entry is None:
            raise ApiError(HTTPStatus.NOT_FOUND,
                           "The upload manifest does not declare this file")
        if upload.record is not None:
            raise ApiError(HTTPStatus.CONFLICT, "The upload is already finalized")
        if length != entry["size"]:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Declared size is {entry['size']} bytes but the request "
                           f"body is {length} bytes")
        destination = upload.staging_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        temp = destination.parent / f".{destination.name}.part-{secrets.token_hex(4)}"
        try:
            with temp.open("wb") as sink:
                remaining = length
                while remaining > 0:
                    block = stream.read(min(TRANSFER_CHUNK_BYTES, remaining))
                    if not block:
                        raise ApiError(HTTPStatus.BAD_REQUEST,
                                       "The request body ended early")
                    digest.update(block)
                    sink.write(block)
                    remaining -= len(block)
            if digest.hexdigest() != entry["sha256"]:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "The uploaded bytes do not match the declared SHA-256")
            os.replace(temp, destination)
        finally:
            temp.unlink(missing_ok=True)
        (destination.parent / f".{destination.name}.partial").unlink(missing_ok=True)
        upload.received[relative_name] = True
        return relative_name

    def _receive_chunk(self, upload, relative_name, stream, length, offset):
        if not is_safe_relative_name(relative_name):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Unsafe upload file name")
        entry = upload.manifest.get(relative_name)
        if entry is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "The upload manifest does not declare this file")
        if upload.record is not None:
            raise ApiError(HTTPStatus.CONFLICT, "The upload is already finalized")
        current = (entry["size"] if relative_name in upload.received
                   else upload.offsets.get(relative_name, 0))
        if (type(offset) is not int or type(length) is not int or offset < 0
                or length < 0 or offset + length > entry["size"]):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid upload byte range")
        # A lost response is reconciled through status. Refuse overlaps rather
        # than appending the same chunk twice or trusting unverified bytes.
        if offset != current:
            raise ApiError(HTTPStatus.CONFLICT, "Upload offset changed",
                           payload={"offset": current})
        if relative_name in upload.received:
            return relative_name
        destination = upload.staging_dir / relative_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.parent / f".{destination.name}.partial"
        try:
            with partial.open("r+b" if partial.exists() else "w+b") as sink:
                sink.seek(current)
                remaining = length
                while remaining:
                    block = stream.read(min(TRANSFER_CHUNK_BYTES, remaining))
                    if not block:
                        raise ApiError(HTTPStatus.BAD_REQUEST, "The request body ended early")
                    sink.write(block)
                    current += len(block)
                    remaining -= len(block)
                sink.flush()
        finally:
            # Bytes written before a disconnect remain resumable. No content
            # becomes final until the complete manifest digest is verified.
            upload.offsets[relative_name] = partial.stat().st_size if partial.exists() else 0
        if current == entry["size"]:
            if self._file_sha256(partial) != entry["sha256"]:
                partial.unlink()
                upload.offsets[relative_name] = 0
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "The uploaded bytes do not match the declared SHA-256")
            os.replace(partial, destination)
            upload.received[relative_name] = True
        return relative_name

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def finalize(self, upload_id):
        """Validate and publish an upload; idempotent per upload ID."""
        upload = self.get(upload_id)
        with upload.lock:
            if upload.cancelled:
                raise ApiError(HTTPStatus.GONE, "Upload was cancelled")
            if upload.record is not None:
                return FinalizedUpload(upload.kind, dict(upload.record),
                                       replayed=True)
            missing = sorted(set(upload.manifest) - set(upload.received))
            if missing:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "The upload is missing declared files",
                               [{"field": name, "message": "File was not uploaded"}
                                for name in missing])
            _validate_upload_content(upload.kind, upload.role, upload.staging_dir)
            if upload.kind != "checkpoint":
                path = (upload.staging_dir if upload.kind == "patch" else
                        next(p for p in upload.staging_dir.rglob("*") if p.is_file()))
                upload.record = {
                    "id": upload.input_id, "kind": upload.kind, "role": upload.role,
                    "path": str(path), "upload_id": upload.upload_id,
                    "bytes": upload.declared_bytes(), "state": "uploaded"}
                return FinalizedUpload(upload.kind, dict(upload.record))
            if upload.kind == "checkpoint":
                record = self._publish_checkpoint(upload)
                upload.record = record
                return FinalizedUpload(upload.kind, dict(record))
    def _publish_checkpoint(self, upload):
        """Move a finalized checkpoint into the service's upload directory.

        The published path lies under the output directory, which the
        dataset-mode load validation already accepts for resume checkpoints.
        """
        root = self.checkpoint_root()
        if root is None:
            raise ApiError(HTTPStatus.CONFLICT,
                           "The service no longer has an output directory for "
                           "uploaded checkpoints")
        root.mkdir(parents=True, exist_ok=True)
        source = next(p for p in upload.staging_dir.rglob("*") if p.is_file())
        entry = next(iter(upload.manifest.values()))
        destination = self.checkpoint_digest_path(root, entry["sha256"])
        with self._lock:
            # A concurrent upload of the same content may have finalized after
            # begin() checked the content-addressed destination.
            if destination.is_file() and destination.stat().st_size == entry["size"]:
                source.unlink(missing_ok=True)
                os.utime(destination, None)
            else:
                os.replace(source, destination)
        shutil.rmtree(upload.staging_dir, ignore_errors=True)
        self.prune_checkpoints(destination)
        return self.checkpoint_record(
            upload.input_id, destination, upload.declared_bytes(),
            upload.upload_id)

    def prune_checkpoints(self, just_published):
        root = self.checkpoint_root()
        if root is None or not root.is_dir():
            return
        active = self.environment.active_checkpoint()
        entries = sorted((path for path in root.iterdir() if path.is_file()),
                         key=lambda path: path.stat().st_mtime, reverse=True)
        kept = 0
        for path in entries:
            protected = path == Path(just_published) or str(path) == active
            if protected or kept < UPLOADED_CHECKPOINTS_KEPT:
                kept += 1
                continue
            path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Removal
    # ------------------------------------------------------------------

    def collect_garbage(self):
        expired = []
        now = time.time()
        with self._lock:
            for upload_id, upload in list(self.uploads.items()):
                if (upload_id not in self._begin_requests and upload.record is None
                        and now - upload.created > UPLOAD_GC_SECONDS):
                    expired.append(upload)
                    del self.uploads[upload_id]
        for upload in expired:
            shutil.rmtree(upload.staging_dir, ignore_errors=True)


def collection_has_affected_links(collections, target_id):
    target = collections[target_id]
    if target.get("windings_linked"):
        return True
    points = target.get("points") or {}
    try:
        target_point_ids = {int(key) for key in points}
    except (TypeError, ValueError):
        # A malformed source cannot be renumbered safely by a local
        # replacement operation.
        return True
    if any(point.get("links") for point in points.values()
           if isinstance(point, dict)):
        return True
    target_numeric = int(target_id)
    for collection_id, collection in collections.items():
        if collection_id == target_id or not isinstance(collection, dict):
            continue
        if target_numeric in (collection.get("windings_linked") or []):
            return True
        for point in (collection.get("points") or {}).values():
            if isinstance(point, dict) and target_point_ids.intersection(
                    point.get("links") or []):
                return True
    return False
