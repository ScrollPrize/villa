"""Session-input uploads: transfer, validation, and publication.

``UploadManager`` owns the staging area, the per-upload transfer state, the
content validation for every input kind, and the content-addressed store of
uploaded resume checkpoints. It never sees ``ServiceState``: everything it
needs from the lifecycle orchestrator arrives through the small callback set
in ``UploadEnvironment`` (where the output directory is, whether a session is
loaded, where that session's ephemeral folder is, which checkpoint the
session is using, and whether an ephemeral reservation is allowed).

In particular the manager does not know about ephemeral bookkeeping, status
snapshots, the event buffer, artifacts, or the fit session itself; finalize
hands the caller a record and the caller decides what to do with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from http import HTTPStatus
import json
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
from fit_session import PCL_ROLE_CONVENTIONS, validate_checkpoint_container
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

# Role -> conventional filename for ephemeral PCL uploads, from the
# declarative fit-input catalog.
PCL_ROLE_FILES = {
    role.value: filename for role, filename in PCL_ROLE_CONVENTIONS}


def _utc_stamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


class Upload:
    __slots__ = ("upload_id", "session_id", "kind", "role", "input_id",
                 "manifest", "staging_dir", "received", "record", "created",
                 "lock", "base_revision", "revision")

    def __init__(self, upload_id, session_id, kind, role, input_id, manifest,
                 staging_dir, base_revision=None, revision=None):
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
        self.base_revision = base_revision
        self.revision = revision

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


def _merge_pcl_documents(existing, incoming):
    """Merge the incoming multi-collection document into the existing one."""
    merged = dict(existing)
    collections = dict(existing.get("collections", {}))
    next_id = max((int(key) for key in collections), default=-1) + 1
    for _, collection in sorted(incoming.get("collections", {}).items(),
                                key=lambda item: int(item[0])):
        collections[str(next_id)] = collection
        next_id += 1
    merged["collections"] = collections
    return merged


def _copy_publish(source, destination, keep_source=False):
    """Publish across filesystems: copy to a temp sibling, rename, and unless
    keep_source is set delete the source (a move)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.parent / f".{destination.name}.incoming-{secrets.token_hex(4)}"
    try:
        if Path(source).is_dir():
            shutil.copytree(source, temp, symlinks=False)
        else:
            shutil.copy2(source, temp)
        os.replace(temp, destination)
    except BaseException:
        if temp.is_dir():
            shutil.rmtree(temp, ignore_errors=True)
        elif temp.exists():
            temp.unlink(missing_ok=True)
        raise
    if keep_source:
        return
    if Path(source).is_dir():
        shutil.rmtree(source, ignore_errors=True)
    else:
        Path(source).unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Ephemeral input ledger
#
# Owned by the lifecycle orchestrator, not by the upload manager: finalize
# produces one record and hands it over, and everything after that (whether
# the fit has taken the input, whether the dataset has a copy of it, when the
# staged bytes may go) is bookkeeping the orchestrator does through here.
# ----------------------------------------------------------------------

#: Has the resident fit taken this input yet?
INCORPORATION_STATES = ("pending", "queued", "incorporated", "error")
#: Does the dataset hold a copy of it yet?
PERSISTENCE_STATES = ("ephemeral", "committed")


@dataclass
class EphemeralInput:
    """One uploaded session input, with its two independent states.

    Incorporation and persistence are not stages of one enum: an input can be
    committed to the dataset before the fit has taken it (commit early, run
    later), and can be incorporated long before anyone commits it. Every
    combination is legal and the pair is what the transitions below move.
    """

    id: str
    kind: str
    role: Optional[str]
    path: str
    bytes: int
    upload_id: Optional[str] = None
    incorporation: str = "pending"
    persistence: str = "ephemeral"
    error: Optional[str] = None
    # Fibers are stable logical inputs. ``revision`` is the newest accepted
    # content, while the incorporated/committed fields describe the resident
    # runtime and dataset independently.
    revision: Optional[str] = None
    incorporated_revision: Optional[str] = None
    committed_revision: Optional[str] = None
    auto_commit: bool = False
    error_revision: Optional[str] = None
    revision_errors: dict = field(default_factory=dict)

    @classmethod
    def from_record(cls, record):
        return cls(
            id=record["id"], kind=record["kind"], role=record.get("role"),
            path=record["path"], bytes=record["bytes"],
            upload_id=record.get("upload_id"),
            revision=record.get("revision"))

    @property
    def committed(self):
        if self.kind == "fiber":
            return (self.revision is not None
                    and self.committed_revision == self.revision)
        return self.persistence == "committed"

    @property
    def incorporated(self):
        return self.incorporation == "incorporated"

    @property
    def settled(self):
        """Committed and incorporated: nothing is left to do with it."""
        # Ordinary inputs can leave the ledger. Fibers remain as reconnectable
        # revision cursors even after their current content settles.
        return self.kind != "fiber" and self.committed and self.incorporated

    def payload(self):
        """The plain record the fitter (and its DDP children) receive."""
        record = {
            "id": self.id, "kind": self.kind, "role": self.role,
            "path": self.path, "bytes": self.bytes,
            "state": self.incorporation,
        }
        if self.upload_id is not None:
            record["upload_id"] = self.upload_id
        if self.kind == "fiber":
            record.update({
                "revision": self.revision,
                "incorporated_revision": self.incorporated_revision,
                "committed_revision": self.committed_revision,
                "auto_commit": self.auto_commit,
                "operation": ("replace" if self.incorporated_revision
                              else "add"),
            })
        return record

    def status_entry(self):
        result = {"id": self.id, "kind": self.kind, "role": self.role,
                  "state": self.incorporation, "bytes": self.bytes,
                  "committed": self.committed}
        if self.error:
            result["error"] = self.error
        if self.kind == "fiber":
            result.update({
                "revision": self.revision,
                "incorporated_revision": self.incorporated_revision,
                "committed_revision": self.committed_revision,
                "auto_commit": self.auto_commit,
            })
            if self.error_revision:
                result["error_revision"] = self.error_revision
            if self.revision_errors:
                result["revision_errors"] = dict(self.revision_errors)
        return result


class EphemeralLedger:
    """The session's ephemeral inputs and every transition they can make.

    All state changes go through this object under the service lock, so
    "committed", "incorporated" and "removed" cannot be spelled three
    different ways in three different call sites.
    """

    def __init__(self, lock):
        self._lock = lock
        self._records = []

    def __iter__(self):
        return iter(list(self._records))

    def __len__(self):
        return len(self._records)

    def __eq__(self, other):
        # Convenience for callers (and tests) comparing against a plain list.
        if isinstance(other, list):
            return list(self._records) == other
        return NotImplemented

    @property
    def records(self):
        with self._lock:
            return list(self._records)

    def clear(self):
        with self._lock:
            self._records = []

    def add(self, record):
        """Register a freshly finalized upload as a pending, ephemeral input."""
        entry = (record if isinstance(record, EphemeralInput)
                 else EphemeralInput.from_record(record))
        with self._lock:
            self._records.append(entry)
        return entry

    def revise_fiber(self, record):
        """Install a CAS-validated newest fiber revision on its logical row."""
        with self._lock:
            entry = self.find("fiber", record["id"])
            if entry is None:
                return self.add(record)
            entry.path = record["path"]
            entry.bytes = record["bytes"]
            entry.upload_id = record.get("upload_id")
            entry.revision = record.get("revision")
            entry.incorporation = "pending"
            entry.error = None
            entry.error_revision = None
            return entry

    def find(self, kind, input_id):
        with self._lock:
            return next((record for record in self._records
                         if record.id == input_id and record.kind == kind),
                        None)

    def contains(self, kind, input_id):
        return self.find(kind, input_id) is not None

    def remove(self, record):
        with self._lock:
            if record in self._records:
                self._records.remove(record)

    def bytes_in_use(self):
        with self._lock:
            return sum(record.bytes for record in self._records)

    # -- transitions ---------------------------------------------------

    def pending(self):
        """Inputs the next run must incorporate."""
        with self._lock:
            return [record for record in self._records
                    if record.incorporation == "pending"]

    def claim_pending(self):
        """Atomically reserve pending inputs in deterministic ledger order."""
        with self._lock:
            records = [record for record in self._records
                       if record.incorporation == "pending"]
            for record in records:
                record.incorporation = "queued"
                record.error = None
            return records

    def return_pending(self, records):
        """Return still-queued claims to the next-Run pool."""
        with self._lock:
            for record in records:
                if record in self._records and record.incorporation == "queued":
                    record.incorporation = "pending"
                    record.error = None

    def uncommitted(self):
        """Inputs the dataset has no copy of yet."""
        with self._lock:
            return [record for record in self._records
                    if not record.committed
                    and record.incorporation in
                    ("pending", "queued", "incorporated")]

    def committed_not_incorporated(self):
        """Committed inputs the resident fit has not taken yet."""
        with self._lock:
            return [record for record in self._records
                    if record.committed and not record.incorporated]

    def resolve(self, payloads):
        """Map payload dicts handed to the fitter back to their records."""
        wanted = {(payload.get("kind"), payload.get("id"))
                  for payload in payloads}
        with self._lock:
            return [record for record in self._records
                    if (record.kind, record.id) in wanted]

    def mark_incorporated(self, records, error=None):
        """Record the outcome of one incorporation attempt."""
        with self._lock:
            for record in records:
                record.incorporation = "error" if error else "incorporated"
                record.error = error
                if record.kind == "fiber":
                    if error:
                        record.error_revision = record.revision
                        record.revision_errors[record.revision] = error
                    else:
                        record.incorporated_revision = record.revision
                        record.error_revision = None
                        record.revision_errors.pop(record.revision, None)
            if error is None:
                self._drop_settled()

    def mark_outcomes(self, outcomes):
        """Apply per-record runtime outcomes keyed by (kind, id)."""
        by_identity = {
            (outcome.get("kind"), outcome.get("id")): outcome
            for outcome in outcomes
        }
        with self._lock:
            for record in self._records:
                outcome = by_identity.get((record.kind, record.id))
                if outcome is None:
                    continue
                state = outcome.get("state")
                if state not in ("incorporated", "error"):
                    continue
                if record.kind == "fiber":
                    outcome_revision = outcome.get("revision", record.revision)
                    if state == "incorporated":
                        record.incorporated_revision = outcome_revision
                        record.revision_errors.pop(outcome_revision, None)
                    else:
                        record.error_revision = outcome_revision
                        record.revision_errors[outcome_revision] = outcome.get("error")
                    # A newer save may have arrived while this revision was in
                    # flight. Its pending/queued state must not be overwritten
                    # by the older outcome.
                    if outcome_revision != record.revision:
                        continue
                record.incorporation = state
                record.error = outcome.get("error") if state == "error" else None
                if record.kind == "fiber" and state == "incorporated":
                    record.error_revision = None
            self._drop_settled()

    def mark_committed(self, records):
        with self._lock:
            for record in records:
                record.persistence = "committed"
                if record.kind == "fiber":
                    record.committed_revision = record.revision
                    record.auto_commit = True
            self._drop_settled()

    def _drop_settled(self):
        """Fully persisted, fully incorporated inputs leave the ledger."""
        self._records = [record for record in self._records
                         if not record.settled]

    # -- presentation --------------------------------------------------

    def status_entries(self):
        with self._lock:
            return [record.status_entry() for record in self._records]


@dataclass(frozen=True)
class UploadEnvironment:
    """Everything the upload manager may ask the orchestrator about.

    Deliberately absent: the fit session, the status snapshot, the event
    buffer, the artifact registry, and the ephemeral input ledger.
    """

    lock: threading.RLock
    #: Root for staging and the uploaded-checkpoint store, or None when the
    #: service has neither a session nor a bound dataset output.
    output_root: Callable[[], Optional[Path]]
    #: Identifier of the loaded session, or None.
    session_id: Callable[[], Optional[str]]
    #: Destination folder for finalized ephemeral inputs.
    ephemeral_dir: Callable[[], Optional[Path]]
    #: Raise ApiError(409) unless a session is loaded.
    require_session: Callable[[], None]
    #: Checkpoint the loaded session resumed from; protected from retention.
    active_checkpoint: Callable[[], str] = lambda: ""
    #: Raise ApiError when an input/revision may not be accepted. Called with
    #: (kind, id, declared bytes, base revision, content revision).
    reserve_ephemeral: Callable[[str, str, int, Optional[str], Optional[str]], None] = \
        lambda kind, input_id, declared, base_revision, revision: None


@dataclass
class FinalizedUpload:
    """Result of finalizing one upload."""

    kind: str
    record: dict
    #: True when this call replayed an already finalized upload.
    replayed: bool = False
    #: Files still staged for this upload, for the caller's bookkeeping.
    extra: dict = field(default_factory=dict)


class UploadManager:
    """Staging, transfer, validation and publication of session inputs."""

    def __init__(self, environment):
        self.environment = environment
        self.uploads = {}

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

    def staged_ephemeral_bytes(self):
        """Declared bytes of ephemeral uploads that are not finalized yet."""
        return sum(upload.declared_bytes() for upload in self.uploads.values()
                   if upload.record is None and upload.kind != "checkpoint")

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    def begin(self, request):
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
        input_id = str(request.get("id") or "").strip()
        if not _SAFE_ID.match(input_id):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "The input id must be a single safe path component")
        manifest = _validate_upload_manifest(request)
        declared = sum(entry["size"] for entry in manifest.values())
        base_revision = request.get("base_revision")
        if base_revision is not None and not isinstance(base_revision, str):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "base_revision must be a string or null")
        if kind != "fiber" and base_revision is not None:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "base_revision is only valid for fiber uploads")
        # Fibers contain exactly one JSON file, so its manifest digest is an
        # opaque, deterministic content revision without another file pass.
        revision = (next(iter(manifest.values()))["sha256"]
                    if kind == "fiber" and len(manifest) == 1 else None)
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
            else:
                self.environment.require_session()
                self.environment.reserve_ephemeral(
                    kind, input_id, declared, base_revision, revision)
            upload_id = secrets.token_hex(16)
            staging = self.staging_root() / upload_id
            upload = Upload(upload_id, self.environment.session_id(), kind,
                            role, input_id, manifest, staging,
                            base_revision=base_revision, revision=revision)
            self.uploads[upload_id] = upload
        staging.mkdir(parents=True, exist_ok=True)
        return {"upload_id": upload_id, "accepted": True}

    def get(self, upload_id):
        with self._lock:
            upload = self.uploads.get(upload_id)
            # Checkpoint uploads are service-scoped; the ephemeral kinds are
            # bound to the session they were started for.
            if upload is None or (upload.kind != "checkpoint"
                                  and upload.session_id != self.environment.session_id()):
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown upload")
            return upload

    def receive(self, upload_id, relative_name, stream, length):
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
        upload = self.get(upload_id)
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
        with upload.lock:
            upload.received[relative_name] = True
        return relative_name

    # ------------------------------------------------------------------
    # Publication
    # ------------------------------------------------------------------

    def finalize(self, upload_id):
        """Validate and publish an upload; idempotent per upload ID."""
        upload = self.get(upload_id)
        with upload.lock:
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
            if upload.kind == "checkpoint":
                record = self._publish_checkpoint(upload)
                upload.record = record
                return FinalizedUpload(upload.kind, dict(record))
            with self._lock:
                self.environment.require_session()
                ephemeral_root = self.environment.ephemeral_dir()
            kind_dir = ephemeral_root / f"{upload.kind}s"
            kind_dir.mkdir(parents=True, exist_ok=True)
            if upload.kind == "patch":
                published = kind_dir / upload.input_id
            elif upload.kind == "fiber":
                revision_dir = kind_dir / upload.input_id
                revision_dir.mkdir(parents=True, exist_ok=True)
                published = revision_dir / f"{upload.revision}.json"
                single = next(p for p in upload.staging_dir.rglob("*") if p.is_file())
            else:
                published = kind_dir / f"{upload.input_id}.json"
                single = next(p for p in upload.staging_dir.rglob("*") if p.is_file())
            if published.exists():
                if upload.kind == "fiber":
                    # A duplicate-content retry through a fresh upload ID is
                    # harmless. Leave publication to the service CAS check.
                    single.unlink(missing_ok=True)
                    shutil.rmtree(upload.staging_dir, ignore_errors=True)
                else:
                    raise ApiError(HTTPStatus.CONFLICT,
                                   "An ephemeral input with this id already exists")
            if upload.kind == "patch":
                os.replace(upload.staging_dir, published)
            elif not published.exists():
                os.replace(single, published)
                shutil.rmtree(upload.staging_dir, ignore_errors=True)
            record = {
                "id": upload.input_id,
                "kind": upload.kind,
                "role": upload.role,
                "path": str(published),
                "bytes": upload.declared_bytes(),
                "state": "pending",
                "upload_id": upload.upload_id,
            }
            if upload.kind == "fiber":
                record["base_revision"] = upload.base_revision
                record["revision"] = upload.revision
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
                if upload.record is None and now - upload.created > UPLOAD_GC_SECONDS:
                    expired.append(upload)
                    del self.uploads[upload_id]
        for upload in expired:
            shutil.rmtree(upload.staging_dir, ignore_errors=True)

    def reset(self):
        """Forget every upload; called when the session scope is replaced."""
        self.uploads = {}
