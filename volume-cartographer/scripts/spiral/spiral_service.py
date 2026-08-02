#!/usr/bin/env python3
"""HTTP service for a persistent interactive Spiral fit.

The service binds to loopback by default. Non-loopback binds are explicit and
always carry bearer authentication; every client — including VC3D talking to a
process it launched itself — uses the same authenticated HTTP protocol.

Generated display data (previews, downloadable
checkpoints) is published as immutable, opaque artifacts and transferred
through ``/artifacts/...`` instead of host filesystem paths. Session inputs
(patches, fibers, PCL documents) can be uploaded into a session-scoped
ephemeral folder and later committed into the dataset.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np
from PIL import Image
import scipy.ndimage
from vc3d_fiber_format_adapter import (
    parse_vc3d_fiber_format,
)

from fit_session import (API_VERSION, PclRole, parse_session_request,
                         resolve_dataset_root, validate_checkpoint_container,
                         validate_session_request)
from config import Config


SERVICE_VERSION = "6.1.0"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256
TRANSFER_CHUNK_BYTES = 1024 * 1024
PREVIEW_ARTIFACTS_KEPT = 3
CHECKPOINT_ARTIFACTS_KEPT = 2
LASAGNA_PREVIEW_OUTPUT_STEP_VX = 20.0
MAX_ARTIFACT_FILES = 4096
MAX_UPLOAD_FILES = 256
UPLOAD_GC_SECONDS = 3600.0
EPHEMERAL_QUOTA_BYTES = int(os.environ.get("SPIRAL_EPHEMERAL_QUOTA_BYTES",
                                           4 * 1024 * 1024 * 1024))
# Uploaded resume checkpoints are service-scoped (usable by future sessions),
# exempt from the ephemeral quota, and bounded by retention instead.
UPLOADED_CHECKPOINTS_KEPT = 3
MAX_CHECKPOINT_UPLOAD_BYTES = int(os.environ.get(
    "SPIRAL_CHECKPOINT_UPLOAD_MAX_BYTES", 64 * 1024 * 1024 * 1024))
UPLOADED_CHECKPOINTS_DIRNAME = "uploaded-checkpoints"
# This buffer is also the reconnect/late-attach history for a remote VC3D
# client.  tqdm produces one entry for each carriage-return redraw, so leave
# enough room for the loading bars and a substantial portion of a long fit.
MAX_LOG_ENTRIES = 20000
MAX_LOG_READ_ENTRIES = 1000
MAX_LOG_ENTRY_CHARS = 8192
DATASET_COMMIT_LOCK_TIMEOUT_SECONDS = 20.0

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@ -]{0,127}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_PCL_ROLE_FILES = {
    PclRole.ABSOLUTE.value: "abs_winding.json",
    PclRole.PATCH_OVERLAP.value: "patch-overlap-pcls.json",
    PclRole.RELATIVE.value: "relative_windings.json",
    PclRole.SAME_WINDING.value: "same_windings.json",
    PclRole.DRAWN_CONTROL_POINTS.value: "drawn_control_points.json",
}

# Base input paths are owned by the service when it was launched with
# --dataset; a load request may then only choose among service-advertised
# values for these keys.
_DATASET_CLIENT_SELECTABLE = ("checkpoint", "tracks_dbm")


class ApiError(Exception):
    def __init__(self, status, message, details=None):
        super().__init__(message)
        self.status = int(status)
        self.message = message
        self.details = details


def parse_gpu_ids(value):
    """Parse a comma-separated list of physical CUDA device indices."""
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise argparse.ArgumentTypeError(
            "--gpus must be a comma-separated list such as 0 or 0,1,2,3")
    try:
        gpu_ids = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpus entries must be non-negative integer device indices") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError(
            "--gpus entries must be non-negative integer device indices")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("--gpus cannot contain duplicate devices")
    return gpu_ids


def parse_session_name(value):
    """Validate a host-owned name which is also used as one path component."""
    name = str(value).strip()
    if not _SAFE_SESSION_NAME.fullmatch(name) or name in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "--session-name must be 1-64 characters, start with a letter or "
            "digit, and contain only letters, digits, '.', '_', or '-'")
    return name


class FileLockUnavailable(RuntimeError):
    pass


class ExclusiveFileLock:
    """Small stdlib-only advisory lock shared by independent service processes."""

    def __init__(self, path):
        self.path = Path(path)
        self._stream = None

    def acquire(self, timeout=0.0):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b")
        if os.name == "nt":
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._stream = stream
                return self
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                    stream.close()
                    raise
                if time.monotonic() >= deadline:
                    stream.close()
                    raise FileLockUnavailable(str(self.path)) from exc
                time.sleep(0.05)

    def release(self):
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self):
        if self._stream is None:
            self.acquire()
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.release()


def _validate_run_influence_config(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "influence_config must be a JSON object")
    allowed = {
        "influence_enabled",
        "influence_z",
        "influence_windings",
        "influence_theta_frac",
        "influence_disable_dt_frac",
        "influence_sigma",
        "sample_count_influence_footprint_points",
        "sample_count_influence_anchor_lattice_points",
        "sample_count_influence_anchor_geometry_points",
        "sample_count_influence_anchor_samples_per_step",
        "influence_anchor_ramp_power",
        "loss_weight_anchor",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       f"Unknown influence configuration keys: {unknown}")
    result = {}
    if "influence_enabled" in value:
        enabled = value["influence_enabled"]
        if not isinstance(enabled, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "interactive_influence_enabled must be boolean")
        result["influence_enabled"] = enabled
    ranges = {
        "influence_z": (1.0, 1_000_000.0),
        "influence_windings": (0.1, 100.0),
        "influence_theta_frac": (0.01, 1.0),
        "influence_disable_dt_frac": (0.0, 1.0),
        "influence_sigma": (0.000001, 10.0),
        "sample_count_influence_footprint_points": (1.0, 1_000_000.0),
        "sample_count_influence_anchor_lattice_points": (1.0, 1_000_000.0),
        "sample_count_influence_anchor_geometry_points": (1.0, 100_000.0),
        "sample_count_influence_anchor_samples_per_step": (1.0, 1_000_000.0),
        "influence_anchor_ramp_power": (0.000001, 100.0),
        "loss_weight_anchor": (0.0, 10_000.0),
    }
    for key, (minimum, maximum) in ranges.items():
        if key not in value:
            continue
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} must be numeric")
        number = float(item)
        if not minimum <= number <= maximum:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"{key} must be between {minimum} and {maximum}")
        result[key] = number
    integer_keys = {
        "sample_count_influence_footprint_points",
        "sample_count_influence_anchor_lattice_points",
        "sample_count_influence_anchor_geometry_points",
        "sample_count_influence_anchor_samples_per_step",
    }
    for key in integer_keys & result.keys():
        if not result[key].is_integer():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} must be an integer")
        result[key] = int(result[key])
    return result


class ServiceLogBuffer:
    """Bounded, incremental copy of the service's stdout and stderr lines."""

    def __init__(self, max_entries=MAX_LOG_ENTRIES):
        self._lock = threading.Lock()
        self._entries = deque(maxlen=max_entries)
        self._pending = {"stdout": "", "stderr": ""}
        self._next_sequence = 1

    def write(self, stream, text):
        if not text:
            return
        # Carriage-return progress displays should still give remote clients
        # useful snapshots even though they overwrite one terminal line.
        text = str(text).replace("\r", "\n")
        with self._lock:
            parts = (self._pending.get(stream, "") + text).split("\n")
            self._pending[stream] = parts.pop()
            for line in parts:
                if not line:
                    continue
                # These high-frequency access lines are still written to the
                # service terminal, but keeping them out of the relay leaves
                # the bounded buffer for useful fitter output.
                if line.startswith('SPIRAL_HTTP "GET /session/status HTTP/') \
                        or line.startswith('SPIRAL_HTTP "GET /logs?after='):
                    continue
                if len(line) > MAX_LOG_ENTRY_CHARS:
                    line = line[:MAX_LOG_ENTRY_CHARS] + " … [truncated]"
                self._entries.append({
                    "sequence": self._next_sequence,
                    "stream": stream,
                    "text": line,
                })
                self._next_sequence += 1

    def read_after(self, after):
        with self._lock:
            latest = self._next_sequence - 1
            cursor_reset = after > latest
            if cursor_reset:
                after = 0
            oldest = self._entries[0]["sequence"] if self._entries else self._next_sequence
            dropped = max(0, oldest - max(0, after + 1))
            entries = [dict(entry) for entry in self._entries
                       if entry["sequence"] > after][:MAX_LOG_READ_ENTRIES]
            next_sequence = entries[-1]["sequence"] if entries else min(after, latest)
        return {
            "entries": entries,
            "next_sequence": next_sequence,
            "latest_sequence": latest,
            "dropped": dropped,
            "cursor_reset": cursor_reset,
        }


class _TeeStream:
    """Preserve normal terminal output while copying complete lines to logs."""

    def __init__(self, stream, logs, name):
        self._stream = stream
        self._logs = logs
        self._name = name

    def write(self, text):
        written = self._stream.write(text)
        self._logs.write(self._name, text)
        return written

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _utc_stamp():
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            block = stream.read(TRANSFER_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _is_safe_relative_name(name):
    """Accept forward-slash relative names made of safe components only."""
    if not isinstance(name, str) or not name or len(name) > 1024:
        return False
    if "\\" in name or name.startswith("/"):
        return False
    parts = name.split("/")
    if len(parts) > 8:
        return False
    for part in parts:
        if part in ("", ".", "..") or not _SAFE_COMPONENT.match(part):
            return False
    return True


def _resolve_inside(root, relative_name):
    """Resolve ``relative_name`` under ``root`` refusing symlink/`..` escapes."""
    root = Path(root).resolve(strict=True)
    candidate = (root / relative_name).resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ApiError(HTTPStatus.FORBIDDEN, "Path escapes the artifact root")
    if candidate.is_symlink() or not candidate.is_file():
        raise ApiError(HTTPStatus.FORBIDDEN, "Not a regular file")
    return candidate


class Artifact:
    __slots__ = ("artifact_id", "kind", "session_id", "generation", "root",
                 "files", "entry_point", "inflight", "pruned",
                 "delete_root_on_prune", "created")

    def __init__(self, artifact_id, kind, session_id, generation, root,
                 files, entry_point, delete_root_on_prune):
        self.artifact_id = artifact_id
        self.kind = kind
        self.session_id = session_id
        self.generation = generation
        self.root = root
        self.files = files
        self.entry_point = entry_point
        self.inflight = 0
        self.pruned = False
        self.delete_root_on_prune = delete_root_on_prune
        self.created = time.time()

    def ref(self):
        return {"id": self.artifact_id, "kind": self.kind,
                "generation": self.generation}

    def manifest(self):
        return {
            "schema_version": 1,
            "id": self.artifact_id,
            "kind": self.kind,
            "session_id": self.session_id,
            "generation": self.generation,
            "entry_point": self.entry_point,
            "files": [
                {"name": name, "size": info["size"], "sha256": info["sha256"]}
                for name, info in sorted(self.files.items())
            ],
        }


class ArtifactRegistry:
    """Immutable generated-data directories addressed by opaque IDs.

    Files are digested once at registration (inside the fitter's pause/export
    window). Pruning never removes an artifact while a download holds an
    in-flight reference; a pruned ID answers ``410 Gone``.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._artifacts = OrderedDict()
        self._pruned_ids = OrderedDict()

    def register_directory(self, kind, session_id, generation, root,
                           entry_point, *, delete_root_on_prune=False,
                           progress=None, hash_workers=1):
        root = Path(root).resolve(strict=True)
        paths = []
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                paths.append(path)
                if len(paths) > MAX_ARTIFACT_FILES:
                    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   "Artifact has too many files to register")

        def digest(path):
            return path, path.stat().st_size, _sha256_file(path)

        files = {}
        workers = max(1, min(int(hash_workers), len(paths) or 1))
        with ThreadPoolExecutor(max_workers=workers,
                                thread_name_prefix="spiral-artifact-hash") as executor:
            for index, (path, size, sha256) in enumerate(
                    executor.map(digest, paths), start=1):
                relative = path.relative_to(root).as_posix()
                files[relative] = {"size": size, "sha256": sha256}
                if progress is not None:
                    progress(index, len(paths), relative)
        if entry_point not in files:
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                           f"Artifact entry point {entry_point!r} was not found")
        artifact_id = f"{kind}-{generation}-{secrets.token_hex(8)}"
        artifact = Artifact(artifact_id, kind, session_id, generation, root,
                            files, entry_point, delete_root_on_prune)
        with self._lock:
            self._artifacts[artifact_id] = artifact
        return artifact.ref()

    def _get(self, artifact_id):
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            if artifact_id in self._pruned_ids:
                raise ApiError(HTTPStatus.GONE, "Artifact has been pruned")
            raise ApiError(HTTPStatus.NOT_FOUND, "Unknown artifact")
        return artifact

    def manifest(self, artifact_id):
        with self._lock:
            return self._get(artifact_id).manifest()

    def acquire_file(self, artifact_id, relative_name):
        """Return ``(artifact, path, info)`` holding an in-flight reference."""
        with self._lock:
            artifact = self._get(artifact_id)
            info = artifact.files.get(relative_name)
            if info is None:
                raise ApiError(HTTPStatus.NOT_FOUND,
                               "The artifact does not contain this file")
            artifact.inflight += 1
        try:
            path = _resolve_inside(artifact.root, relative_name)
        except BaseException:
            self.release(artifact)
            raise
        return artifact, path, info

    def release(self, artifact):
        delete_root = None
        with self._lock:
            artifact.inflight -= 1
            if artifact.pruned and artifact.inflight == 0 and artifact.delete_root_on_prune:
                delete_root = artifact.root
        if delete_root is not None:
            shutil.rmtree(delete_root, ignore_errors=True)

    def prune(self, kind, session_id, keep):
        """Prune all but the newest ``keep`` artifacts of one kind."""
        to_delete = []
        with self._lock:
            matching = [a for a in self._artifacts.values()
                        if a.kind == kind and a.session_id == session_id]
            matching.sort(key=lambda a: a.generation)
            for artifact in matching[:-keep] if keep else matching:
                del self._artifacts[artifact.artifact_id]
                self._pruned_ids[artifact.artifact_id] = True
                while len(self._pruned_ids) > 4096:
                    self._pruned_ids.popitem(last=False)
                artifact.pruned = True
                if artifact.delete_root_on_prune and artifact.inflight == 0:
                    to_delete.append(artifact.root)
        for root in to_delete:
            shutil.rmtree(root, ignore_errors=True)


class Upload:
    __slots__ = ("upload_id", "session_id", "kind", "role", "input_id",
                 "manifest", "staging_dir", "received", "record", "created",
                 "lock")

    def __init__(self, upload_id, session_id, kind, role, input_id, manifest,
                 staging_dir):
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
        if not _is_safe_relative_name(name):
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
        if role not in _PCL_ROLE_FILES:
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


def _find_lasagna_service():
    configured = str(os.environ.get("LASAGNA_SERVICE_PATH") or "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    here = Path(__file__).resolve()
    candidates.extend([
        here.parents[3] / "lasagna" / "fit_service.py",
        Path.home() / "villa" / "lasagna" / "fit_service.py",
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        "Cannot find Lasagna fit_service.py on the Spiral host; "
        "set LASAGNA_SERVICE_PATH")


def _md5_file(path):
    digest = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(TRANSFER_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return f"md5:{digest.hexdigest()}"


def _prepare_cleaned_lasagna_surface(surface_dir, destination,
                                     erosion_cells=3):
    """Stage a Lasagna-only TIFXYZ with ragged/disconnected support removed."""
    surface_dir = Path(surface_dir).resolve(strict=True)
    destination = Path(destination)
    required = ("meta.json", "x.tif", "y.tif", "z.tif")
    missing = [name for name in required if not (surface_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Spiral preview is missing: {', '.join(missing)}")

    coordinates = []
    shape = None
    valid = None
    for name in ("x.tif", "y.tif", "z.tif"):
        with Image.open(surface_dir / name) as image:
            coordinate = np.asarray(image, dtype=np.float32)
        if coordinate.ndim != 2:
            raise RuntimeError(
                f"Spiral preview {name} must be a two-dimensional TIFF")
        if shape is None:
            shape = coordinate.shape
        elif coordinate.shape != shape:
            raise RuntimeError(
                "Spiral preview coordinate TIFF dimensions do not match")
        coordinate = coordinate.copy()
        coordinates.append(coordinate)
        coordinate_valid = coordinate != -1.0
        valid = (coordinate_valid if valid is None
                 else valid | coordinate_valid)

    cleaned = scipy.ndimage.binary_erosion(
        valid, iterations=int(erosion_cells), border_value=0)
    labels, component_count = scipy.ndimage.label(
        cleaned, structure=scipy.ndimage.generate_binary_structure(2, 1))
    if component_count == 0:
        raise RuntimeError(
            f"Lasagna input cleanup removed every valid TIFXYZ vertex "
            f"after {int(erosion_cells)}-cell erosion")
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    cleaned = labels == int(np.argmax(component_sizes))

    shutil.copytree(surface_dir, destination)
    for coordinate, name in zip(coordinates, ("x.tif", "y.tif", "z.tif")):
        coordinate[~cleaned] = -1.0
        Image.fromarray(coordinate).save(destination / name)

    metadata_path = destination / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["bbox"] = [
        [float(coordinate[cleaned].min()) for coordinate in coordinates],
        [float(coordinate[cleaned].max()) for coordinate in coordinates],
    ]
    valid_quad = (
        cleaned[:-1, :-1]
        & cleaned[1:, :-1]
        & cleaned[:-1, 1:]
        & cleaned[1:, 1:]
    )
    scale = metadata.get("scale")
    if (isinstance(scale, list) and len(scale) >= 2
            and all(isinstance(value, (int, float)) and value > 0
                    for value in scale[:2])):
        area_vx2 = float(valid_quad.sum()) / (
            float(scale[0]) * float(scale[1]))
        old_area_vx2 = metadata.get("area_vx2")
        old_area_cm2 = metadata.get("area_cm2")
        metadata["area_vx2"] = area_vx2
        if (isinstance(old_area_vx2, (int, float))
                and old_area_vx2 > 0
                and isinstance(old_area_cm2, (int, float))):
            metadata["area_cm2"] = (
                area_vx2 * float(old_area_cm2) / float(old_area_vx2))
    metadata["lasagna_input_cleanup"] = {
        "erosion_cells": int(erosion_cells),
        "component_connectivity": 4,
        "components_after_erosion": int(component_count),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=4) + "\n", encoding="utf-8")
    return destination


def _prepare_lasagna_surface_object(surface_dir, object_store):
    surface_dir = Path(surface_dir).resolve(strict=True)
    required = ("meta.json", "x.tif", "y.tif", "z.tif")
    missing = [name for name in required if not (surface_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Spiral preview is missing: {', '.join(missing)}")
    lines = []
    for path in sorted(p for p in surface_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(surface_dir).as_posix()
        lines.append(f"{relative}\t{_md5_file(path)}\n")
    manifest = hashlib.md5(
        "".join(lines).encode("utf-8"), usedforsecurity=False).hexdigest()
    ref = {"type": "tifxyz_segment", "name": surface_dir.name,
           "hash": f"md5:{manifest}"}
    destination = (Path(object_store) / ref["type"] / manifest
                   / quote(ref["name"], safe=""))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "segment").symlink_to(surface_dir, target_is_directory=True)
    (destination / "object.json").write_text(
        json.dumps(ref, indent=2) + "\n", encoding="utf-8")
    return ref


def _surface_xyz(surface_dir):
    """Load one TIFXYZ grid as HxWx3 float32 plus its validity mask."""
    coordinates = []
    for name in ("x.tif", "y.tif", "z.tif"):
        with Image.open(Path(surface_dir) / name) as image:
            coordinates.append(np.asarray(image, dtype=np.float32).copy())
    if not coordinates or any(value.shape != coordinates[0].shape
                              for value in coordinates[1:]):
        raise RuntimeError("TIFXYZ coordinate TIFF dimensions do not match")
    xyz = np.stack(coordinates, axis=-1)
    valid = np.isfinite(xyz).all(axis=-1) & ~np.all(xyz == -1.0, axis=-1)
    return xyz, valid


def _validate_tifxyz_output_step(metadata, expected_step):
    """Require exported TIFXYZ scale to match the requested grid step."""
    expected = float(expected_step)
    scale = metadata.get("scale") if isinstance(metadata, dict) else None
    if (not math.isfinite(expected) or expected <= 0.0
            or not isinstance(scale, list) or len(scale) < 2):
        raise RuntimeError(
            "Lasagna output metadata has no valid two-axis scale")
    expected_scale = 1.0 / expected
    values = []
    for raw in scale[:2]:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise RuntimeError(
                "Lasagna output metadata has a non-numeric scale") from None
        if (not math.isfinite(value) or value <= 0.0
                or not math.isclose(
                    value, expected_scale, rel_tol=1.0e-6, abs_tol=1.0e-9)):
            raise RuntimeError(
                "Lasagna output scale does not match requested preview step "
                f"{expected:g}")
        values.append(value)
    return values


def _load_flatten_correspondence(checkpoint_path=None, map_path=None):
    """Read Lasagna's flattened-output -> Spiral-source grid map."""
    if map_path is not None and Path(map_path).is_file():
        mapping = np.load(str(map_path), mmap_mode="r", allow_pickle=False)
        mapping = np.asarray(mapping, dtype=np.float32)
        if mapping.ndim != 3 or mapping.shape[-1] != 2:
            raise RuntimeError(
                "Lasagna output-to-source map must have shape (rows, columns, 2)")
        if not np.isfinite(mapping).all():
            raise RuntimeError(
                "Lasagna output-to-source map contains non-finite values")
        return mapping

    if checkpoint_path is None:
        raise RuntimeError("Lasagna produced no flatten correspondence")
    import torch

    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "flatten_map_flat" not in state:
        raise RuntimeError(
            "Lasagna flatten checkpoint contains no output-to-source map")
    value = state["flatten_map_flat"]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    mapping = np.asarray(value, dtype=np.float32)
    if mapping.ndim != 3 or mapping.shape[-1] != 2:
        raise RuntimeError(
            "Lasagna output-to-source map must have shape (rows, columns, 2)")
    if not np.isfinite(mapping).all():
        raise RuntimeError("Lasagna output-to-source map contains non-finite values")
    return mapping


def _sample_rgba_through_map(source_rgba, source_yx, output_valid, *,
                             executor=None):
    """Bilinearly warp RGBA using premultiplied alpha."""
    source = np.asarray(source_rgba, dtype=np.float32) / 255.0
    if source.ndim != 3 or source.shape[-1] != 4:
        raise RuntimeError("Mapped preview overlay must be RGBA")
    alpha = source[..., 3]
    premultiplied = source[..., :3] * alpha[..., None]
    coordinates = [source_yx[..., 0], source_yx[..., 1]]

    def sample(channel):
        values = alpha if channel == 3 else premultiplied[..., channel]
        return scipy.ndimage.map_coordinates(
            values, coordinates, order=1, mode="constant", cval=0.0,
            prefilter=False)

    if executor is None:
        sampled = [sample(channel) for channel in range(4)]
    else:
        # Each channel is independent and scipy releases the GIL here.  Mapping
        # them concurrently preserves the exact interpolation and output bytes.
        sampled = list(executor.map(sample, range(4)))
    sampled_alpha = sampled[3]
    sampled_rgb = np.stack(sampled[:3], axis=-1)
    nonzero = sampled_alpha > 1.0e-8
    sampled_rgb[nonzero] /= sampled_alpha[nonzero, None]
    sampled_rgb[~nonzero] = 0.0
    sampled_alpha = np.where(output_valid, sampled_alpha, 0.0)
    sampled_rgb = np.where(output_valid[..., None], sampled_rgb, 0.0)
    result = np.empty((*sampled_alpha.shape, 4), dtype=np.uint8)
    result[..., :3] = np.clip(
        np.rint(sampled_rgb * 255.0), 0, 255).astype(np.uint8)
    result[..., 3] = np.clip(
        np.rint(sampled_alpha * 255.0), 0, 255).astype(np.uint8)
    return result


def _mapped_winding_ids(source_manifest, source_shape, source_yx,
                        output_valid):
    """Categorically map source winding membership onto a flattened grid."""
    ranges = source_manifest.get("winding_column_ranges")
    windings = source_manifest.get("winding_ids")
    if (not isinstance(ranges, list) or not isinstance(windings, list)
            or len(ranges) != len(windings) or not ranges):
        raise RuntimeError("Spiral source preview has no winding mapping")
    winding_values = sorted({int(winding) for winding in windings})
    winding_to_dense = {
        winding: index + 1 for index, winding in enumerate(winding_values)
    }
    source_labels = np.zeros(source_shape, dtype=np.int32)
    for bounds, winding in zip(ranges, windings):
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise RuntimeError("Malformed Spiral winding column range")
        begin, end = int(bounds[0]), int(bounds[1])
        if begin < 0 or end <= begin or end > source_shape[1]:
            raise RuntimeError("Spiral winding column range is out of bounds")
        source_labels[:, begin:end] = winding_to_dense[int(winding)]

    rows = np.rint(source_yx[..., 0]).astype(np.int64)
    columns = np.rint(source_yx[..., 1]).astype(np.int64)
    in_bounds = (
        output_valid
        & (rows >= 0) & (rows < source_shape[0])
        & (columns >= 0) & (columns < source_shape[1])
    )
    dense_result = np.zeros(output_valid.shape, dtype=np.int32)
    dense_result[in_bounds] = source_labels[
        rows[in_bounds], columns[in_bounds]]
    bounds = []
    objects = scipy.ndimage.find_objects(
        dense_result, max_label=len(winding_values))
    for dense_label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        row_slice, column_slice = slices
        winding = winding_values[dense_label - 1]
        bounds.append({
            "winding": winding,
            "row_begin": int(row_slice.start),
            "row_end": int(row_slice.stop),
            "column_begin": int(column_slice.start),
            "column_end": int(column_slice.stop),
        })
    if not bounds:
        raise RuntimeError("Lasagna correspondence mapped no preview windings")
    lookup = np.asarray([-1, *winding_values], dtype=np.int32)
    result = lookup[dense_result]
    return result, bounds


def _raw_run_diff_rgba(previous_manifest, current_manifest, *,
                       current_surface_data=None):
    """Build a current-source-grid displacement overlay by winding identity."""
    if current_surface_data is None:
        current_surface = Path(current_manifest["surface_path"])
        current_xyz, current_valid = _surface_xyz(current_surface)
    else:
        current_xyz, current_valid = current_surface_data
    rgba = np.zeros((*current_valid.shape, 4), dtype=np.uint8)
    if previous_manifest is None:
        return rgba, 0
    previous_xyz, previous_valid = _surface_xyz(
        Path(previous_manifest["surface_path"]))
    previous_by_winding = {
        int(winding): bounds
        for winding, bounds in zip(
            previous_manifest.get("winding_ids", []),
            previous_manifest.get("winding_column_ranges", []))
    }
    magnitudes = np.full(current_valid.shape, np.nan, dtype=np.float32)
    for winding, current_bounds in zip(
            current_manifest.get("winding_ids", []),
            current_manifest.get("winding_column_ranges", [])):
        previous_bounds = previous_by_winding.get(int(winding))
        if previous_bounds is None:
            continue
        current_begin, current_end = map(int, current_bounds)
        previous_begin, previous_end = map(int, previous_bounds)
        rows = min(current_xyz.shape[0], previous_xyz.shape[0])
        width = min(current_end - current_begin,
                    previous_end - previous_begin)
        if rows <= 0 or width <= 0:
            continue
        current_region = current_xyz[:rows, current_begin:current_begin + width]
        previous_region = previous_xyz[
            :rows, previous_begin:previous_begin + width]
        valid = (
            current_valid[:rows, current_begin:current_begin + width]
            & previous_valid[:rows, previous_begin:previous_begin + width]
        )
        delta = np.linalg.norm(current_region - previous_region, axis=-1)
        target = magnitudes[:rows, current_begin:current_begin + width]
        target[valid] = delta[valid]
    finite = np.isfinite(magnitudes) & (magnitudes > 1.0e-6)
    if not finite.any():
        return rgba, 0
    display_maximum = float(np.percentile(magnitudes[finite], 95))
    if not math.isfinite(display_maximum) or display_maximum <= 0.0:
        display_maximum = float(np.nanmax(magnitudes))
    intensity = np.zeros_like(magnitudes)
    intensity[finite] = np.clip(
        magnitudes[finite] / display_maximum, 0.0, 1.0)
    # Match VC3D's blue/cyan/yellow/red diagnostic palette closely.
    stops = np.asarray(
        [[32, 64, 220], [20, 210, 235], [255, 220, 35], [255, 45, 20]],
        dtype=np.float32)
    scaled = intensity * (len(stops) - 1)
    lower = np.minimum(scaled.astype(np.int32), len(stops) - 2)
    fraction = (scaled - lower)[..., None]
    rgba[..., :3] = np.clip(
        stops[lower] * (1.0 - fraction)
        + stops[lower + 1] * fraction,
        0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(
        finite,
        np.clip(28.0 + 207.0 * np.sqrt(intensity), 0, 235),
        0).astype(np.uint8)
    return rgba, int(finite.sum())


def _fit_service_json(port, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}{path}", data=data,
        method="GET" if body is None else "POST",
        headers={"X-Fit-Service-API-Version": "2",
                 "Content-Type": "application/json",
                 "X-VC3D-Source": "Spiral host service"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {}
        raise RuntimeError(
            str(payload.get("error") or f"Lasagna HTTP {exc.code}")) from exc
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def _stop_process_group(process):
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


class ServiceState:
    def __init__(self, dataset_root=None, dataset_resolution=None,
                 service_name=None, session_name="", logs=None, gpu_ids=(0,)):
        self.lock = threading.RLock()
        self.session = None
        self.session_id = None
        self.session_paths = None
        self.session_request = None
        self.service_generation = 1
        self.session_generation = 0
        self.command_generation = 0
        self.status_generation = 0
        self.commands = OrderedDict()
        self.inflight_commands = set()
        self.command_condition = threading.Condition(self.lock)
        self.replacing = False
        self.replacement_old_session_released = False
        self.dataset_root = str(dataset_root) if dataset_root else None
        self.dataset_resolution = dataset_resolution
        self.service_name = service_name or socket.gethostname()
        self.session_name = str(session_name or "")
        self.logs = logs if logs is not None else ServiceLogBuffer()
        self.gpu_ids = tuple(gpu_ids)
        self.artifacts = ArtifactRegistry()
        self.uploads = {}
        self.ephemeral_records = []
        self._registered_preview_generation = 0
        self._processed_preview_generation = 0
        self._publishing_preview_generation = 0
        self._preview_artifact = None
        self._preview_publish = None
        self._preview_progress_started = None
        self._preview_publish_error = None
        self._preview_process = None
        self._previous_raw_preview_manifest = None
        self.config_catalog = Config.catalog()
        self.session_revision = 0
        self.run_plans = {}
        self.pending_revision_target = None

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    def _base(self):
        return {
            "api_version": API_VERSION,
            "service_version": SERVICE_VERSION,
            "service_name": self.service_name,
            "session_name": self.session_name,
            "session_id": self.session_id,
            "service_generation": self.service_generation,
            "session_generation": self.session_generation,
            "session_revision": self.session_revision,
            "command_generation": self.command_generation,
            "generation": self.status_generation,
            "session_replacement_in_progress": self.replacing,
            "replacement_old_session_released": self.replacement_old_session_released,
            "gpus": list(self.gpu_ids),
        }

    def _commit_availability(self):
        if self.session is None or self.session_paths is None:
            return False, "No fit session is loaded"
        if not self.ephemeral_records:
            return False, "No ephemeral inputs have been added"
        if not any(record["state"] in ("pending", "incorporated")
                   and not record.get("committed")
                   for record in self.ephemeral_records):
            return False, "Every added input is already committed"
        dataset_root = self.session_paths.dataset_root
        if not dataset_root or not Path(dataset_root).is_dir():
            return False, "The session has no dataset root directory"
        if not os.access(dataset_root, os.W_OK):
            return False, "The dataset root is read-only"
        return True, ""

    def status(self):
        with self.lock:
            response = self._base()
            response.update(self.session.status() if self.session else {
                "state": "Empty", "phase": "No session", "current_iteration": 0,
                "target_iteration": 0, "latest_metrics": {}, "warnings": [],
                "error": None, "preview_manifest_path": None, "preview_generation": 0,
                "progress": None,
            })
            response.setdefault("progress", None)
            response["session_request"] = self.session_request
            response["preview_artifact"] = self._preview_artifact
            response["preview_publish"] = (
                dict(self._preview_publish)
                if self._preview_publish else None)
            response["preview_publish_error"] = self._preview_publish_error
            if self._preview_publish:
                stage_name = str(
                    self._preview_publish.get("stage_name") or "").strip()
                if stage_name:
                    response["phase"] = stage_name
                    step = self._preview_publish.get("step")
                    total = self._preview_publish.get("total_steps")
                    elapsed = (
                        max(
                            0.0,
                            time.monotonic()
                            - self._preview_progress_started)
                        if self._preview_progress_started is not None
                        else 0.0)
                    eta = None
                    if (step is not None and total is not None
                            and int(step) > 0 and int(total) > int(step)
                            and elapsed >= 2.0):
                        eta = elapsed * (
                            int(total) - int(step)) / int(step)
                    elif (step is not None and total is not None
                          and int(total) > 0
                          and int(step) >= int(total)):
                        eta = 0.0
                    response["progress"] = {
                        "operation": "publishing_preview",
                        "stage_name": stage_name,
                        "detail": None,
                        "step": int(step) if step is not None else None,
                        "total_steps": (
                            int(total) if total is not None else None),
                        "unit": "steps",
                        "elapsed_seconds": elapsed,
                        "eta_seconds": eta,
                    }
            response["ephemeral_inputs"] = [
                {"id": record["id"], "kind": record["kind"],
                 "role": record.get("role"), "state": record["state"],
                 "bytes": record["bytes"],
                 "committed": bool(record.get("committed"))}
                for record in self.ephemeral_records
            ]
            available, reason = self._commit_availability()
            response["commit_available"] = available
            response["commit_unavailable_reason"] = reason
            response["dataset_owned"] = self.dataset_resolution is not None
            return response

    def health(self):
        response = self._base()
        response.update({
            "ready": True,
            "process_id": os.getpid(),
            "dataset_owned": self.dataset_resolution is not None,
            "dataset_root": self.dataset_root,
            "cuda_ready": None if not self.session else self.session.status()["state"] != "Error",
        })
        return response

    def configuration_catalog(self):
        return {**self._base(), **self.config_catalog}

    def dataset(self):
        if self.dataset_resolution is None:
            raise ApiError(HTTPStatus.NOT_FOUND,
                           "This service was not launched with --dataset")
        return {**self._base(), **self.dataset_resolution.to_dict()}

    def resolve(self, root_value):
        if self.dataset_resolution is not None:
            requested = str(root_value or "").strip()
            if requested and Path(requested).resolve(strict=False) != \
                    Path(self.dataset_root).resolve(strict=False):
                raise ApiError(HTTPStatus.FORBIDDEN,
                               "This service resolves only the dataset it was launched with")
            return self.dataset()
        return {
            **self._base(),
            **resolve_dataset_root(
                root_value, session_name=self.session_name).to_dict(),
        }

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _dataset_session_request(self, request):
        """Build the load request for a --dataset service from its own resolution."""
        resolution = self.dataset_resolution.to_dict()
        requested_paths = request.get("paths") or {}
        offending = sorted(
            key for key, value in requested_paths.items()
            if key not in _DATASET_CLIENT_SELECTABLE
            and (value or (isinstance(value, list) and value))
        )
        if offending:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "This service owns its base inputs; the load request must not "
                "carry input paths",
                [{"field": key, "message": "Base input paths are owned by the service"}
                 for key in offending])
        paths = {"dataset_root": resolution["root"], "scroll_zarr": ""}
        for key in ("umbilicus", "fibers", "verified_patches", "unverified_patches",
                    "outer_shell", "normal_x", "normal_y", "gradient_magnitude",
                    "surf_sdt", "tracks_dbm", "output_directory", "cache_directory"):
            paths[key] = resolution["resolved"].get(key, "")
        paths["pcls"] = resolution["pcl_inputs"]

        checkpoint = str(requested_paths.get("checkpoint") or "").strip()
        if checkpoint:
            allowed = set(resolution.get("detected_checkpoints", []))
            resolved_checkpoint = str(Path(checkpoint).resolve(strict=False))
            output_root = Path(paths["output_directory"]).resolve(strict=False)
            if resolved_checkpoint not in allowed and \
                    not Path(resolved_checkpoint).is_relative_to(output_root):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "Checkpoint must be one the service advertises or "
                               "one under the session output directory",
                               [{"field": "checkpoint", "message": "Not a service-advertised checkpoint"}])
            paths["checkpoint"] = resolved_checkpoint

        tracks = str(requested_paths.get("tracks_dbm") or "").strip()
        if tracks:
            candidates = set(resolution.get("ambiguities", {}).get("tracks_dbm", []))
            if resolution["resolved"].get("tracks_dbm"):
                candidates.add(resolution["resolved"]["tracks_dbm"])
            if str(Path(tracks).resolve(strict=False)) not in candidates:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "tracks_dbm must be one of the service-advertised candidates",
                               [{"field": "tracks_dbm", "message": "Not a service-advertised candidate"}])
            paths["tracks_dbm"] = str(Path(tracks).resolve(strict=False))

        return {**request, "paths": paths}

    def load(self, request):
        if self.dataset_resolution is not None:
            request = self._dataset_session_request(request)
        paths, run, preview = parse_session_request(request)
        errors = validate_session_request(paths, run)
        if errors:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Session validation failed", errors)
        with self.lock:
            if self.replacing:
                raise ApiError(HTTPStatus.CONFLICT, "A session replacement is already in progress")
            if self.session and self.session.status()["state"] in {
                "Loading", "Running", "Saving", "ExportingPreview"
            }:
                raise ApiError(HTTPStatus.CONFLICT, "The current session is active")
            previous = self.session
            previous_ephemeral = self._session_ephemeral_dir()
            self.replacing = True
            self.replacement_old_session_released = False
        try:
            if previous:
                previous.close()
                with self.lock:
                    # Validation happened before replacement.  Once teardown has
                    # succeeded, report honestly that the previous resident CUDA
                    # session is no longer available even if new loading fails.
                    if self.session is previous:
                        self.session = None
                        self.session_id = None
                        self.session_paths = None
                        self.session_request = None
                    self._reset_session_scope()
                    self.replacement_old_session_released = True
                    self.status_generation += 1
                if previous_ephemeral:
                    shutil.rmtree(previous_ephemeral, ignore_errors=True)
            from spiral_runtime import create_session
            with self.lock:
                self.session_generation += 1
                self.session_id = f"spiral-{self.session_generation}-{secrets.token_hex(5)}"
                self.session_paths = paths
                self.session_request = {
                    "paths": paths.manifest(),
                    "run": run.manifest(),
                    "preview": preview.manifest(),
                }
                self.session_revision += 1
                self.run_plans.clear()
                self._reset_session_scope()
                try:
                    self.session = create_session(
                        paths, run, preview, self._status_changed,
                        gpu_ids=self.gpu_ids)
                except BaseException:
                    self.session_id = None
                    self.session_paths = None
                    self.session_request = None
                    raise
                self.status_generation += 1
                response = self.status()
                response["accepted"] = True
                return response
        finally:
            with self.lock:
                self.replacing = False

    def _reset_session_scope(self):
        previous_raw = self._previous_raw_preview_manifest
        self.ephemeral_records = []
        self.uploads = {}
        self._registered_preview_generation = 0
        self._processed_preview_generation = 0
        self._publishing_preview_generation = 0
        self._preview_artifact = None
        self._preview_publish = None
        self._preview_progress_started = None
        self._preview_publish_error = None
        self._previous_raw_preview_manifest = None
        if previous_raw:
            shutil.rmtree(
                Path(previous_raw).parent, ignore_errors=True)

    def _status_changed(self, status):
        # Runs on the fitter thread inside the pause/export window, so artifact
        # digests are computed while training is stopped.
        try:
            self._maybe_register_artifacts(status)
        except Exception as exc:
            print(f"SPIRAL_ARTIFACT_ERROR {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        with self.lock:
            applied_manifest = status.get("input_manifest")
            if (self.session_paths is not None
                    and isinstance(applied_manifest, dict)
                    and applied_manifest != self.session_paths.manifest()):
                self.session_paths = SpiralInputPaths.from_mapping(
                    applied_manifest)
                if self.session_request is not None:
                    self.session_request["paths"] = \
                        self.session_paths.manifest()
            if (self.pending_revision_target is not None
                    and status.get("state") in {"Ready", "Paused"}
                    and int(status.get("current_iteration") or 0)
                    >= self.pending_revision_target):
                self.session_revision += 1
                self.pending_revision_target = None
            self.status_generation += 1

    def _maybe_register_artifacts(self, status):
        with self.lock:
            session_id = self.session_id
            preview_generation = int(status.get("preview_generation") or 0)
            preview_manifest = status.get("preview_manifest_path")
            publish_preview = (
                preview_manifest
                and preview_generation > self._processed_preview_generation
                and preview_generation != self._publishing_preview_generation)
            if publish_preview:
                self._publishing_preview_generation = preview_generation
                self._preview_publish_error = None
        if not publish_preview:
            return

        try:
            published_manifest = self._publish_flattened_preview(
                session_id, preview_generation, Path(preview_manifest))

            def indexing_progress(current, total, relative):
                self._update_preview_publish(
                    preview_generation, state="indexing",
                    stage_name=(
                        f"Indexing preview files ({current}/{total}): "
                        f"{relative}"),
                    step=current, total_steps=total,
                    overall_progress=(
                        float(current) / float(total) if total else 1.0))

            indexing_started = time.perf_counter()
            self._update_preview_publish(
                preview_generation, state="indexing",
                stage_name="Indexing preview files",
                step=0, total_steps=0, overall_progress=0.0)
            ref = self.artifacts.register_directory(
                "spiral-preview", session_id, preview_generation,
                published_manifest.parent, published_manifest.name,
                delete_root_on_prune=True, progress=indexing_progress,
                hash_workers=4)
            print(
                "SPIRAL_PREVIEW_TIMING "
                f"generation={preview_generation} "
                "stage='Indexing preview files' "
                f"seconds={time.perf_counter() - indexing_started:.6f}",
                flush=True)
            with self.lock:
                if self.session_id == session_id:
                    self._preview_artifact = ref
                    self._registered_preview_generation = preview_generation
                    self._preview_publish_error = None
            self.artifacts.prune(
                "spiral-preview", session_id, PREVIEW_ARTIFACTS_KEPT)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"SPIRAL_PREVIEW_ERROR {error}", file=sys.stderr, flush=True)
            # A failed raw generation is never exposed or retried. Keep only
            # the previous successful raw generation, which is needed to map
            # the next run-difference overlay.
            failed_raw = Path(preview_manifest).parent
            with self.lock:
                retained_raw = self._previous_raw_preview_manifest
            if (not retained_raw
                    or failed_raw != Path(retained_raw).parent):
                shutil.rmtree(failed_raw, ignore_errors=True)
            with self.lock:
                if self.session_id == session_id:
                    self._preview_publish_error = error
        finally:
            with self.lock:
                if self.session_id == session_id:
                    self._processed_preview_generation = max(
                        self._processed_preview_generation,
                        preview_generation)
                    if self._publishing_preview_generation == preview_generation:
                        self._publishing_preview_generation = 0
                    self._preview_publish = None
                    self._preview_progress_started = None
                    self.status_generation += 1

    def _update_preview_publish(self, generation, **values):
        with self.lock:
            if self._publishing_preview_generation != generation:
                return
            current = dict(self._preview_publish or {})
            next_stage = values.get("stage_name", current.get("stage_name"))
            if next_stage != current.get("stage_name"):
                self._preview_progress_started = time.monotonic()
            current.update(values)
            current["generation"] = generation
            self._preview_publish = current
            self.status_generation += 1

    def run(self, request):
        token = request.get("plan_token")
        with self.lock:
            plan = self.run_plans.pop(token, None)
        if not plan or plan["expires"] < time.monotonic():
            raise ApiError(HTTPStatus.CONFLICT, "Run plan is missing or expired")
        if plan["revision"] != self.session_revision:
            raise ApiError(HTTPStatus.CONFLICT, "Run plan is stale")
        if plan["new_fit_required"]:
            raise ApiError(HTTPStatus.CONFLICT,
                           "This plan requires Start New Fit")
        if plan["session_reload_required"]:
            raise ApiError(HTTPStatus.CONFLICT,
                           "This plan requires reloading fit inputs")
        session = self._require_session()
        status = session.status()
        influence_config = _validate_run_influence_config(
            plan["influence"])
        run_config = plan["configuration_changes"]
        with self.lock:
            pending = [record for record in self.ephemeral_records
                       if record["state"] == "pending"]

            def mark_incorporated(records, error=None):
                with self.lock:
                    for record in records:
                        record["state"] = "error" if error else "incorporated"
                        if error:
                            record["error"] = error
                    # Records that are both committed and incorporated are
                    # fully persisted and part of the fit: nothing is left to
                    # do with them, so they leave the ephemeral list.
                    if not error:
                        self.ephemeral_records = [
                            record for record in self.ephemeral_records
                            if not (record.get("committed")
                                    and record["state"] == "incorporated")]
                    self.status_generation += 1

        with self.lock:
            self.pending_revision_target = (
                int(status.get("current_iteration") or 0) + plan["iterations"])
        try:
            run_arguments = {
                "pending_inputs": pending,
                "mark_incorporated": mark_incorporated,
                "influence_config": influence_config,
                "run_config": run_config,
            }
            if plan["path_changes"]:
                run_arguments["path_changes"] = plan["path_changes"]
            target = session.run(plan["iterations"], **run_arguments)
        except BaseException:
            with self.lock:
                self.pending_revision_target = None
            raise
        with self.lock:
            self.run_plans.clear()
            self.status_generation += 1
        return {**self.status(), "accepted": True, "target_iteration": target}

    def plan_run(self, request):
        session = self._require_session()
        if session.status().get("state") not in {"Ready", "Paused"}:
            raise ApiError(HTTPStatus.CONFLICT,
                           "Run planning requires a paused session")
        expected = request.get("expected_session_revision")
        if expected != self.session_revision:
            raise ApiError(HTTPStatus.CONFLICT, "Session revision is stale")
        configuration = request.get("configuration")
        if not isinstance(configuration, dict) or \
                set(configuration) != set(self.config_catalog["defaults"]):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Run planning requires a complete configuration")
        try:
            configuration = Config(configuration).as_dict()
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        iterations = int(request.get("iterations", 0))
        if iterations < 1:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "iterations must be at least 1")
        current = session.status().get("applied_config")
        if current is None:
            current = Config(
                (self.session_request.get("run") or {}).get("config") or {}
            ).as_dict()
        changes = {
            key: value for key, value in configuration.items()
            if current.get(key) != value
        }
        fields = self.config_catalog["schema"]["fields"]
        impacts = {fields[key]["runtime_impact"] for key in changes}
        dependencies = sorted({
            dependency for key in changes
            for dependency in fields[key]["dependencies"]
        })
        current_manifest = self.session_paths.manifest()
        input_manifest = request.get("inputs")
        if input_manifest is None:
            input_manifest = current_manifest
        if not isinstance(input_manifest, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "inputs must be a path manifest object")
        path_changes = {
            key: input_manifest.get(key)
            for key in set(current_manifest) | set(input_manifest)
            if input_manifest.get(key) != current_manifest.get(key)
        }
        path_specs = self.config_catalog["schema"].get("paths", {})
        input_changes = []
        for key, value in sorted(path_changes.items()):
            spec = path_specs.get(key)
            impact = (
                spec["runtime_impact"] if spec is not None
                else "prepared_input_rebuild")
            impacts.add(impact)
            dependencies.extend(
                spec.get("dependencies", []) if spec is not None else [
                    "dense_stores", "patch_pcl", "tracks", "shell",
                    "preview_output",
                ])
            input_changes.append({
                "key": key,
                "before": current_manifest.get(key),
                "after": value,
                "runtime_impact": impact,
            })
        if "outer_shell" in path_changes:
            outer_shell = str(path_changes["outer_shell"] or "").strip()
            if not outer_shell or not Path(outer_shell).is_dir():
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "Outer shell path is not a readable directory",
                    [{"field": "outer_shell",
                      "message": "Path is not a directory"}])
        dependencies = sorted(set(dependencies))
        token = secrets.token_urlsafe(24)
        new_fit = "new_fit" in impacts
        session_reload_required = "prepared_input_rebuild" in impacts
        plan = {
            "revision": self.session_revision,
            "expires": time.monotonic() + 60.0,
            "iterations": iterations,
            "influence": request.get("influence") or {},
            "configuration_changes": changes,
            "path_changes": path_changes,
            "changes": [
                {"key": key, "before": current.get(key), "after": value,
                 "runtime_impact": fields[key]["runtime_impact"]}
                for key, value in changes.items()
            ],
            "affected_prepared_inputs": dependencies,
            "model_state_preserved": not new_fit,
            "optimizer_state_preserved": not new_fit,
            "new_fit_required": new_fit,
            "session_reload_required": session_reload_required,
            "input_changed": bool(path_changes),
            "input_changes": input_changes,
        }
        with self.lock:
            self.run_plans[token] = plan
        return {
            **self._base(), "plan_token": token,
            "expires_in_seconds": 60,
            **{key: value for key, value in plan.items()
               if key not in {"expires", "configuration_changes", "path_changes",
                              "iterations", "influence", "revision"}},
        }

    def stop(self):
        self._require_session().stop()
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True}

    def save_checkpoint(self, request):
        session = self._require_session()
        path = request.get("path")
        if not path:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Checkpoint path is required")
        resolved = Path(path).expanduser().resolve(strict=False)
        if self.dataset_resolution is not None:
            output_root = Path(self.session_paths.output_directory).resolve(strict=False)
            if not resolved.is_relative_to(output_root):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "This service only saves checkpoints under the "
                               "session output directory")
        saved = session.save_checkpoint(str(resolved))
        return {**self.status(), "checkpoint_path": saved}

    def download_checkpoint(self):
        """Create a checkpoint and publish it as a downloadable artifact."""
        session = self._require_session()
        with self.lock:
            session_id = self.session_id
            output_directory = self.session_paths.output_directory
            generation = int(time.time_ns())
        root = Path(output_directory) / ".spiral-artifacts" / f"checkpoint-{secrets.token_hex(6)}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            saved = session.save_checkpoint(str(root / "checkpoint.ckpt"))
        except BaseException:
            shutil.rmtree(root, ignore_errors=True)
            raise
        ref = self.artifacts.register_directory(
            "spiral-checkpoint", session_id, generation, root,
            Path(saved).name, delete_root_on_prune=True)
        self.artifacts.prune("spiral-checkpoint", session_id, CHECKPOINT_ARTIFACTS_KEPT)
        return {**self.status(), "checkpoint_artifact": ref}

    def delete(self):
        with self.lock:
            if not self.session:
                return {**self.status(), "deleted": False}
            if self.session.status()["state"] in {"Loading", "Running", "Saving", "ExportingPreview"}:
                raise ApiError(HTTPStatus.CONFLICT, "Stop and wait for the session to settle before deleting it")
            session = self.session
            ephemeral_dir = self._session_ephemeral_dir()
            self.session = None
            self.session_id = None
            self.session_paths = None
            self.session_request = None
            self.session_generation += 1
            self.status_generation += 1
            self._reset_session_scope()
        session.close()
        if ephemeral_dir:
            shutil.rmtree(ephemeral_dir, ignore_errors=True)
        return {**self.status(), "deleted": True}

    def _require_session(self):
        with self.lock:
            if self.session is None:
                raise ApiError(HTTPStatus.CONFLICT, "No fit session is loaded")
            return self.session

    # ------------------------------------------------------------------
    # Automatic host-owned Lasagna preview publication
    # ------------------------------------------------------------------

    def _publish_flattened_preview(
            self, session_id, generation, preview_manifest_path):
        process = None
        publish_root = None
        timing_stage = None
        timing_started = time.perf_counter()

        def start_stage(state, stage_name, **values):
            nonlocal timing_stage, timing_started
            now = time.perf_counter()
            if timing_stage is not None:
                print(
                    "SPIRAL_PREVIEW_TIMING "
                    f"generation={generation} stage={timing_stage!r} "
                    f"seconds={now - timing_started:.6f}",
                    flush=True)
            timing_stage = stage_name
            timing_started = now
            self._update_preview_publish(
                generation, state=state, stage_name=stage_name, **values)

        try:
            fit_service = _find_lasagna_service()
            config_path = fit_service.parent / "configs" / "flatten_fast_nofilter.json"
            if not config_path.is_file():
                raise RuntimeError(
                    f"Cannot find Lasagna flatten config: {config_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(preview_manifest_path.read_text(encoding="utf-8"))
            surface_path = Path(str(manifest.get("surface_path") or ""))
            if not surface_path.is_dir():
                raise RuntimeError(
                    f"Spiral preview surface does not exist: {surface_path}")

            args = config.get("args")
            if not isinstance(args, dict):
                args = {}
            args["flatten_output_step"] = LASAGNA_PREVIEW_OUTPUT_STEP_VX
            args["flatten_output_margin"] = 0.0
            config["args"] = args
            run_request = (self.session_request or {}).get("run") or {}
            voxel_size = run_request.get("voxel_size_um")
            if isinstance(voxel_size, (int, float)) and float(voxel_size) > 0.0:
                config["voxel_size_um"] = float(voxel_size)

            output_root = Path(self.session_paths.output_directory).resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            publish_parent = output_root / ".spiral-published" / session_id
            publish_parent.mkdir(parents=True, exist_ok=True)
            final_root = publish_parent / f"generation-{generation}"
            if final_root.exists():
                raise RuntimeError(
                    f"Refusing to overwrite published preview: {final_root}")
            publish_root = publish_parent / (
                f".generation-{generation}.incoming-{secrets.token_hex(5)}")
            publish_root.mkdir(parents=True, exist_ok=False)
            source_id = str(manifest.get("surface_id") or "")
            if not source_id:
                raise RuntimeError("Spiral preview manifest has no surface id")
            surface_id = f"{source_id}-lasagna.tifxyz"
            flattened_surface = publish_root / surface_id
            model_output = publish_root / "flatten-model.pt"

            with tempfile.TemporaryDirectory(prefix="spiral_lasagna_") as temporary:
                temporary = Path(temporary)
                object_store = temporary / "objects"
                start_stage(
                    "preparing", "Preparing Lasagna input surface",
                    output_step_vx=LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                metadata = json.loads(
                    (surface_path / "meta.json").read_text(encoding="utf-8"))
                cleanup = metadata.get("lasagna_input_cleanup")
                if (not isinstance(cleanup, dict)
                        or cleanup.get("erosion_cells") != 3
                        or cleanup.get("component_connectivity") != 4
                        or not isinstance(cleanup.get("components_after_erosion"), int)
                        or cleanup["components_after_erosion"] < 1
                        or manifest.get("schema_version") != 2
                        or "components" in metadata):
                    raise RuntimeError(
                        "Spiral preview was not published with authoritative "
                        "connected-surface cleanup")
                surface_ref = _prepare_lasagna_surface_object(
                    surface_path, object_store)
                ready = threading.Event()
                port_holder = {}

                process = subprocess.Popen(
                    [sys.executable, str(fit_service), "--port", "0",
                     "--allow-no-data-dir", "--object-store-dir",
                     str(object_store)],
                    cwd=str(fit_service.parent),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=(os.name == "posix"))
                with self.lock:
                    if (self.session_id == session_id
                            and self._publishing_preview_generation == generation):
                        self._preview_process = process

                def relay_output():
                    pattern = re.compile(r"listening on http://[^:]+:(\d+)")
                    assert process.stdout is not None
                    for line in process.stdout:
                        text = line.rstrip()
                        if text:
                            print(f"SPIRAL_LASAGNA {text}", flush=True)
                        match = pattern.search(text)
                        if match:
                            port_holder["port"] = int(match.group(1))
                            ready.set()
                        if "[fit] peak GPU memory:" in text:
                            self._update_preview_publish(
                                generation, state="saving",
                                stage_name="Saving optimized flatten model",
                                step=0, total_steps=0,
                                overall_progress=0.0)

                relay = threading.Thread(
                    target=relay_output, name="spiral-lasagna-log",
                    daemon=True)
                relay.start()
                if not ready.wait(60):
                    code = process.poll()
                    raise RuntimeError(
                        "Temporary Lasagna service failed to start"
                        + (f" (exit {code})" if code is not None else ""))
                port = port_holder["port"]
                config["external_surfaces"] = [surface_ref]
                request_body = {
                    "config": config,
                    "job_spec": {
                        "config": config,
                        "linked_surfaces": [surface_ref],
                    },
                    "single_segment": True,
                    "config_name": "flatten_fast_nofilter.json",
                    "output_name": surface_id,
                    "output_dir": str(publish_root),
                    "model_output": str(model_output),
                    "embed_job_metadata": False,
                    "omit_model": True,
                    "export_flatten_map": True,
                    "source": "Spiral host service",
                }
                accepted = _fit_service_json(
                    port, "/optimize", request_body, timeout=60)
                fit_job_id = str(accepted.get("job_id") or "")
                if not fit_job_id:
                    raise RuntimeError(
                        "Temporary Lasagna service returned no job id")
                start_stage(
                    "running", "Flattening preview surface",
                    step=0, total_steps=0, overall_progress=0.0)

                while True:
                    with self.lock:
                        if self.session_id != session_id:
                            raise RuntimeError(
                                "The Spiral session changed while publishing its preview")
                    fit_status = _fit_service_json(
                        port, f"/jobs/{fit_job_id}", timeout=15)
                    state = str(fit_status.get("state") or "")
                    self._update_preview_publish(
                        generation, state=state or "running",
                        stage_name=str(
                            fit_status.get("stage_name")
                            or fit_status.get("stage") or "Flattening"),
                        step=int(fit_status.get("step") or 0),
                        total_steps=int(fit_status.get("total_steps") or 0),
                        overall_progress=float(
                            fit_status.get("overall_progress") or 0.0),
                        loss=fit_status.get("loss"))
                    if state == "finished":
                        break
                    if state == "cancelled":
                        raise RuntimeError("Lasagna preview flatten was cancelled")
                    if state == "error":
                        raise RuntimeError(
                            str(fit_status.get("error")
                                or "Lasagna flatten failed"))
                    time.sleep(0.5)

                flattened_metadata_path = flattened_surface / "meta.json"
                if not flattened_metadata_path.is_file():
                    raise RuntimeError(
                        "Lasagna reported success but produced no tifxyz output")
                flatten_map_output = publish_root / ".flatten-map.npy"
                start_stage(
                    "loading", "Loading flattened preview output",
                    step=0, total_steps=0, overall_progress=0.0)
                correspondence = _load_flatten_correspondence(
                    model_output, flatten_map_output)
                flattened_xyz, flattened_valid = _surface_xyz(flattened_surface)
                if correspondence.shape[:2] != flattened_xyz.shape[:2]:
                    raise RuntimeError(
                        "Lasagna correspondence dimensions do not match "
                        "the flattened surface")
                source_xyz, source_valid = _surface_xyz(surface_path)
                start_stage(
                    "mapping", "Mapping preview winding membership",
                    step=0, total_steps=0, overall_progress=0.0)
                winding_ids, winding_bounds = _mapped_winding_ids(
                    manifest, source_xyz.shape[:2],
                    correspondence, flattened_valid)
                winding_map_name = "winding-ids.tif"
                # OpenCV/Qt do not portably decode signed-int TIFF samples.
                # IEEE float32 represents every supported winding id exactly
                # and is converted back to int32 after validation in VC3D.
                Image.fromarray(
                    winding_ids.astype(np.float32), mode="F").save(
                    publish_root / winding_map_name)

                mapped_loss_maps = []
                loss_output = publish_root / "loss-maps"
                loss_output.mkdir(exist_ok=True)
                loss_entries = [
                    entry for entry in manifest.get("loss_maps", [])
                    if isinstance(entry, dict)
                    and (preview_manifest_path.parent
                         / str(entry.get("path") or "")).is_file()
                ]
                remap_total = len(loss_entries)
                start_stage(
                    "mapping", (
                        f"Remapping preview loss maps (0/{remap_total})"
                        if remap_total else "No preview loss maps to remap"),
                    step=0, total_steps=remap_total,
                    overall_progress=0.0)
                with ThreadPoolExecutor(
                        max_workers=4,
                        thread_name_prefix="spiral-overlay-channel") as remap_executor:
                    for loss_index, entry in enumerate(loss_entries, start=1):
                        relative = str(entry.get("path") or "")
                        self._update_preview_publish(
                            generation, state="mapping",
                            stage_name=(
                                f"Remapping preview loss maps "
                                f"({loss_index}/{remap_total}): "
                                f"{Path(relative).name}"),
                            step=loss_index - 1, total_steps=remap_total,
                            overall_progress=(
                                float(loss_index - 1) / float(remap_total)
                                if remap_total else 1.0))
                        source_overlay = preview_manifest_path.parent / relative
                        with Image.open(source_overlay) as image:
                            source_rgba = np.asarray(
                                image.convert("RGBA"), dtype=np.uint8)
                        mapped = _sample_rgba_through_map(
                            source_rgba, correspondence, flattened_valid,
                            executor=remap_executor)
                        destination = loss_output / Path(relative).name
                        Image.fromarray(mapped, mode="RGBA").save(destination)
                        mapped_entry = dict(entry)
                        mapped_entry["path"] = (
                            Path("loss-maps") / destination.name).as_posix()
                        mapped_entry["supported_pixels"] = int(
                            np.count_nonzero(mapped[..., 3]))
                        mapped_loss_maps.append(mapped_entry)

                    start_stage(
                        "mapping", "Building preview run difference",
                        step=0, total_steps=0, overall_progress=0.0)
                    previous_manifest = None
                    with self.lock:
                        previous_path = self._previous_raw_preview_manifest
                    if previous_path and Path(previous_path).is_file():
                        previous_manifest = json.loads(
                            Path(previous_path).read_text(encoding="utf-8"))
                    raw_diff, changed_pixels = _raw_run_diff_rgba(
                        previous_manifest, manifest,
                        current_surface_data=(source_xyz, source_valid))
                    mapped_diff = _sample_rgba_through_map(
                        raw_diff, correspondence, flattened_valid,
                        executor=remap_executor)
                run_diff = None
                if changed_pixels:
                    run_diff_name = "run-diff.png"
                    Image.fromarray(mapped_diff, mode="RGBA").save(
                        publish_root / run_diff_name)
                    run_diff = {
                        "path": run_diff_name,
                        "changed_source_pixels": changed_pixels,
                        "supported_pixels": int(
                            np.count_nonzero(mapped_diff[..., 3])),
                    }

                start_stage(
                    "finalizing", "Finalizing preview metadata",
                    step=0, total_steps=0, overall_progress=0.0)
                flattened_metadata = json.loads(
                    flattened_metadata_path.read_text(encoding="utf-8"))
                _validate_tifxyz_output_step(
                    flattened_metadata, LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                flattened_metadata.pop("components", None)
                flattened_metadata.pop("winding_column_ranges", None)
                flattened_metadata.pop("model_source", None)
                flattened_metadata["uuid"] = surface_id
                flattened_metadata["name"] = surface_id
                flattened_metadata["grid_shape"] = [
                    int(flattened_xyz.shape[0]), int(flattened_xyz.shape[1])]
                flattened_metadata["output_step_vx"] = (
                    LASAGNA_PREVIEW_OUTPUT_STEP_VX)
                flattened_metadata["winding_id_map"] = winding_map_name
                flattened_metadata["winding_id_dtype"] = "float32_integer"
                flattened_metadata["winding_bounds"] = winding_bounds
                flattened_metadata["component_winding_ids"] = [
                    item["winding"] for item in winding_bounds]
                flattened_metadata_path.write_text(
                    json.dumps(flattened_metadata, indent=4) + "\n",
                    encoding="utf-8")

                published = dict(manifest)
                published["schema_version"] = 3
                published["surface_id"] = surface_id
                published["surface_path"] = str(final_root / surface_id)
                published["manifest_path"] = str(final_root / "manifest.json")
                published["output_step_vx"] = LASAGNA_PREVIEW_OUTPUT_STEP_VX
                published["grid_shape"] = [
                    int(flattened_xyz.shape[0]), int(flattened_xyz.shape[1])]
                published["winding_ids"] = [
                    item["winding"] for item in winding_bounds]
                published["winding_id_map"] = winding_map_name
                published["winding_id_dtype"] = "float32_integer"
                published["winding_bounds"] = winding_bounds
                published["loss_maps"] = mapped_loss_maps
                published.pop("winding_column_ranges", None)
                published.pop("components", None)
                if run_diff is None:
                    published.pop("run_diff", None)
                else:
                    published["run_diff"] = run_diff
                (publish_root / "manifest.json").write_text(
                    json.dumps(published, indent=2) + "\n",
                    encoding="utf-8")

                # These are transient transport files.  The interactive Spiral
                # preview never exposes a Lasagna model to VC3D.
                del correspondence
                model_output.unlink(missing_ok=True)
                flatten_map_output.unlink(missing_ok=True)
                (flattened_surface / "model.pt").unlink(missing_ok=True)
                os.replace(publish_root, final_root)
                publish_root = None

                with self.lock:
                    old_raw = self._previous_raw_preview_manifest
                    self._previous_raw_preview_manifest = str(
                        preview_manifest_path)
                if old_raw and old_raw != str(preview_manifest_path):
                    shutil.rmtree(Path(old_raw).parent, ignore_errors=True)
                start_stage(
                    "finalizing", "Preparing preview artifact index",
                    step=0, total_steps=0, overall_progress=0.0)
                return final_root / "manifest.json"
        finally:
            _stop_process_group(process)
            if publish_root is not None:
                shutil.rmtree(publish_root, ignore_errors=True)
            with self.lock:
                if self._preview_process is process:
                    self._preview_process = None
                self.status_generation += 1

    # ------------------------------------------------------------------
    # Session input uploads
    # ------------------------------------------------------------------

    def _output_root(self):
        """Output directory known before any session in dataset mode."""
        if self.session_paths is not None and self.session_paths.output_directory:
            return Path(self.session_paths.output_directory)
        if self.dataset_resolution is not None:
            return Path(self.dataset_resolution.resolved["output_directory"])
        return None

    def _session_ephemeral_dir(self):
        if self.session_paths is None or self.session_id is None:
            return None
        return Path(self.session_paths.output_directory) / ".spiral-ephemeral" / self.session_id

    def _staging_root(self):
        return self._output_root() / ".spiral-upload-staging"

    def _checkpoint_upload_root(self):
        return self._output_root() / UPLOADED_CHECKPOINTS_DIRNAME

    @staticmethod
    def _checkpoint_digest_path(root, digest):
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

    def _find_uploaded_checkpoint(self, root, digest, size):
        """Find retained checkpoint content, including pre-v7 named uploads."""
        canonical = self._checkpoint_digest_path(root, digest)
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
    def _checkpoint_record(input_id, path, size, upload_id=None):
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

    def _ephemeral_bytes_in_use(self):
        total = sum(record["bytes"] for record in self.ephemeral_records)
        total += sum(upload.declared_bytes() for upload in self.uploads.values()
                     if upload.record is None and upload.kind != "checkpoint")
        return total

    def begin_upload(self, request):
        kind = str(request.get("kind") or "").strip()
        if kind not in ("patch", "fiber", "pcl", "checkpoint"):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Input kind must be one of patch, fiber, pcl, checkpoint")
        role = request.get("role")
        if kind == "pcl":
            if role not in _PCL_ROLE_FILES:
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
        if kind == "checkpoint":
            with self.lock:
                # Resume checkpoints are needed before a session exists, so
                # they are service-scoped: allowed whenever an output
                # directory is known (a --dataset launch or a live session).
                output_root = self._output_root()
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
            existing = self._find_uploaded_checkpoint(
                checkpoint_root, entry["sha256"], entry["size"])
            if existing is not None:
                try:
                    os.utime(existing, None)
                except OSError:
                    pass
                record = self._checkpoint_record(
                    input_id, existing, entry["size"])
                return {
                    **self._base(),
                    "accepted": True,
                    "deduplicated": True,
                    "input": record,
                }
        with self.lock:
            if kind == "checkpoint":
                current_output_root = self._output_root()
                if current_output_root is None or current_output_root != output_root:
                    raise ApiError(HTTPStatus.CONFLICT,
                                   "The checkpoint upload destination changed")
                # Close the race with another request that finalized this
                # digest while the legacy-file scan ran without the state lock.
                canonical = self._checkpoint_digest_path(
                    checkpoint_root, entry["sha256"])
                if canonical.is_file() and canonical.stat().st_size == entry["size"]:
                    os.utime(canonical, None)
                    return {
                        **self._base(),
                        "accepted": True,
                        "deduplicated": True,
                        "input": self._checkpoint_record(
                            input_id, canonical, entry["size"]),
                    }
            else:
                self._require_session()
                if any(record["id"] == input_id and record["kind"] == kind
                       for record in self.ephemeral_records):
                    raise ApiError(HTTPStatus.CONFLICT,
                                   f"An ephemeral {kind} named {input_id!r} already exists")
                if self._ephemeral_bytes_in_use() + declared > EPHEMERAL_QUOTA_BYTES:
                    raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                   "The ephemeral input quota is exhausted")
            upload_id = secrets.token_hex(16)
            staging = self._staging_root() / upload_id
            upload = Upload(upload_id, self.session_id, kind, role, input_id,
                            manifest, staging)
            self.uploads[upload_id] = upload
        staging.mkdir(parents=True, exist_ok=True)
        return {**self._base(), "upload_id": upload_id, "accepted": True}

    def _get_upload(self, upload_id):
        with self.lock:
            upload = self.uploads.get(upload_id)
            # Checkpoint uploads are service-scoped; the ephemeral kinds are
            # bound to the session they were started for.
            if upload is None or (upload.kind != "checkpoint"
                                  and upload.session_id != self.session_id):
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown upload")
            return upload

    def receive_upload_file(self, upload_id, relative_name, stream, length):
        if not _is_safe_relative_name(relative_name):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Unsafe upload file name")
        upload = self._get_upload(upload_id)
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
        return {**self._base(), "received": relative_name, "accepted": True}

    def finalize_upload(self, upload_id):
        upload = self._get_upload(upload_id)
        with upload.lock:
            if upload.record is not None:
                # Finalize is idempotent per upload ID.
                return {**self.status(), "input": dict(upload.record), "accepted": True}
            missing = sorted(set(upload.manifest) - set(upload.received))
            if missing:
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "The upload is missing declared files",
                               [{"field": name, "message": "File was not uploaded"}
                                for name in missing])
            _validate_upload_content(upload.kind, upload.role, upload.staging_dir)
            if upload.kind == "checkpoint":
                record = self._publish_checkpoint_upload(upload)
                upload.record = record
                with self.lock:
                    self.status_generation += 1
                return {**self.status(), "input": dict(record), "accepted": True}
            with self.lock:
                self._require_session()
                ephemeral_root = self._session_ephemeral_dir()
            kind_dir = ephemeral_root / f"{upload.kind}s"
            kind_dir.mkdir(parents=True, exist_ok=True)
            if upload.kind == "patch":
                published = kind_dir / upload.input_id
            else:
                published = kind_dir / f"{upload.input_id}.json"
                single = next(p for p in upload.staging_dir.rglob("*") if p.is_file())
            if published.exists():
                raise ApiError(HTTPStatus.CONFLICT,
                               "An ephemeral input with this id already exists")
            if upload.kind == "patch":
                os.replace(upload.staging_dir, published)
            else:
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
            upload.record = record
        with self.lock:
            self.ephemeral_records.append(record)
            self.status_generation += 1
        return {**self.status(), "input": dict(record), "accepted": True}

    def _publish_checkpoint_upload(self, upload):
        """Move a finalized checkpoint into the service's upload directory.

        The published path lies under the output directory, which the
        dataset-mode load validation already accepts for resume checkpoints.
        """
        root = self._checkpoint_upload_root()
        if root is None:
            raise ApiError(HTTPStatus.CONFLICT,
                           "The service no longer has an output directory for "
                           "uploaded checkpoints")
        root.mkdir(parents=True, exist_ok=True)
        source = next(p for p in upload.staging_dir.rglob("*") if p.is_file())
        entry = next(iter(upload.manifest.values()))
        destination = self._checkpoint_digest_path(root, entry["sha256"])
        with self.lock:
            # A concurrent upload of the same content may have finalized after
            # begin_upload checked the content-addressed destination.
            if destination.is_file() and destination.stat().st_size == entry["size"]:
                source.unlink(missing_ok=True)
                os.utime(destination, None)
            else:
                os.replace(source, destination)
        shutil.rmtree(upload.staging_dir, ignore_errors=True)
        self._prune_uploaded_checkpoints(destination)
        return self._checkpoint_record(
            upload.input_id, destination, upload.declared_bytes(),
            upload.upload_id)

    def _prune_uploaded_checkpoints(self, just_published):
        root = self._checkpoint_upload_root()
        if root is None or not root.is_dir():
            return
        with self.lock:
            active = self.session_paths.checkpoint if self.session_paths else ""
        entries = sorted((path for path in root.iterdir() if path.is_file()),
                         key=lambda path: path.stat().st_mtime, reverse=True)
        kept = 0
        for path in entries:
            protected = path == Path(just_published) or str(path) == active
            if protected or kept < UPLOADED_CHECKPOINTS_KEPT:
                kept += 1
                continue
            path.unlink(missing_ok=True)

    def delete_upload(self, upload_id):
        with self.lock:
            upload = self.uploads.get(upload_id)
            if upload is None:
                raise ApiError(HTTPStatus.NOT_FOUND, "Unknown upload")
            if upload.record is not None:
                raise ApiError(HTTPStatus.CONFLICT,
                               "The upload is finalized; it is now a session input")
            del self.uploads[upload_id]
        shutil.rmtree(upload.staging_dir, ignore_errors=True)
        return {**self._base(), "deleted": True}

    def gc_uploads(self):
        expired = []
        now = time.time()
        with self.lock:
            for upload_id, upload in list(self.uploads.items()):
                if upload.record is None and now - upload.created > UPLOAD_GC_SECONDS:
                    expired.append(upload)
                    del self.uploads[upload_id]
        for upload in expired:
            shutil.rmtree(upload.staging_dir, ignore_errors=True)

    def commit_inputs(self):
        with self.lock:
            self._require_session()
            available, reason = self._commit_availability()
            if not available:
                raise ApiError(HTTPStatus.CONFLICT, f"Commit is unavailable: {reason}")
            expected_session_id = self.session_id
            dataset_root = Path(self.session_paths.dataset_root)
        commit_lock = ExclusiveFileLock(dataset_root / ".spiral-commit.lock")
        try:
            commit_lock.acquire(DATASET_COMMIT_LOCK_TIMEOUT_SECONDS)
        except FileLockUnavailable as exc:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Dataset commit is busy in another Spiral session; try again") from exc
        try:
            # Re-check after acquiring the process-wide lock: another request
            # may have completed while this one was waiting.
            with self.lock:
                self._require_session()
                if self.session_id != expected_session_id:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "The Spiral session changed while waiting to commit")
                available, reason = self._commit_availability()
                if not available:
                    raise ApiError(
                        HTTPStatus.CONFLICT, f"Commit is unavailable: {reason}")
                records = [record for record in self.ephemeral_records
                           if record["state"] in ("pending", "incorporated")
                           and not record.get("committed")]
                paths = self.session_paths
            dataset_root = Path(paths.dataset_root)
            patches_dir = Path(paths.verified_patches) if paths.verified_patches \
                else dataset_root / "verified_patches"
            fibers_dir = Path(paths.fibers) if paths.fibers else dataset_root / "fibers"

            # Collision checks and publications share the same dataset lock, so
            # cooperating service processes cannot race an existence check.
            for record in records:
                if record["kind"] == "patch" and (patches_dir / record["id"]).exists():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"A patch named {record['id']!r} already exists in the dataset")
                if record["kind"] == "fiber" and \
                        (fibers_dir / f"{record['id']}.json").exists():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"A fiber named {record['id']!r} already exists in the dataset")

            committed = []
            for record in records:
                source = Path(record["path"])
                # A still-pending record keeps its staged copy: it remains the
                # incorporation source for the next run, so committing never
                # removes an input from the live session's queue.
                keep_source = record["state"] == "pending"
                if record["kind"] == "patch":
                    _copy_publish(source, patches_dir / record["id"], keep_source)
                elif record["kind"] == "fiber":
                    _copy_publish(source, fibers_dir / f"{record['id']}.json", keep_source)
                else:
                    target = dataset_root / _PCL_ROLE_FILES[record["role"]]
                    with source.open("r", encoding="utf-8") as stream:
                        incoming = json.load(stream)
                    if target.exists():
                        backup = target.with_name(f"{target.name}.{_utc_stamp()}.bak")
                        shutil.copy2(target, backup)
                        with target.open("r", encoding="utf-8") as stream:
                            existing = json.load(stream)
                        merged = _merge_pcl_documents(existing, incoming)
                    else:
                        merged = incoming
                    temp = target.with_name(
                        f".{target.name}.incoming-{secrets.token_hex(4)}")
                    with temp.open("w", encoding="utf-8") as stream:
                        json.dump(merged, stream, indent=2)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temp, target)
                    if not keep_source:
                        source.unlink(missing_ok=True)
                committed.append(record["id"])
            with self.lock:
                for record in records:
                    record["committed"] = True
                # Committed records that already joined the resident fit are
                # done; pending ones stay queued for the next run.
                self.ephemeral_records = [
                    record for record in self.ephemeral_records
                    if not (record.get("committed")
                            and record["state"] == "incorporated")
                ]
                if self.dataset_resolution is not None:
                    self.dataset_resolution = resolve_dataset_root(
                        self.dataset_root, session_name=self.session_name)
                self.status_generation += 1
            return {**self.status(), "committed": committed, "accepted": True}
        finally:
            commit_lock.release()

    def remove_input(self, request):
        kind = str(request.get("kind") or "").strip()
        input_id = str(request.get("id") or "").strip()
        with self.lock:
            self._require_session()
            record = next((record for record in self.ephemeral_records
                           if record["id"] == input_id and record["kind"] == kind), None)
            if record is None:
                raise ApiError(HTTPStatus.NOT_FOUND,
                               f"No ephemeral {kind or 'input'} named {input_id!r} exists")
            if record["state"] == "incorporated":
                raise ApiError(HTTPStatus.CONFLICT,
                               "This input already joined the resident fit; removing it "
                               "requires reloading the session")
            self.ephemeral_records.remove(record)
            self.status_generation += 1
        # The staged copy is only deleted when the dataset holds no committed
        # copy; a committed record's file is the user's data now.
        if not record.get("committed"):
            path = Path(record["path"])
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        return {**self.status(), "removed": input_id, "accepted": True}

    # ------------------------------------------------------------------
    # Command deduplication
    # ------------------------------------------------------------------

    def deduplicated(self, command_id, operation):
        if not isinstance(command_id, str) or not command_id.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "A non-empty command_id is required")
        with self.lock:
            while command_id in self.inflight_commands:
                self.command_condition.wait()
            if command_id in self.commands:
                cached = self.commands[command_id]
                self.commands.move_to_end(command_id)
                return cached
            self.inflight_commands.add(command_id)
        try:
            response = operation()
            with self.lock:
                self.command_generation += 1
                response["command_generation"] = self.command_generation
                self.commands[command_id] = response
                while len(self.commands) > MAX_DEDUPLICATED_COMMANDS:
                    self.commands.popitem(last=False)
            return response
        finally:
            with self.lock:
                self.inflight_commands.discard(command_id)
                self.command_condition.notify_all()

    def close(self):
        with self.lock:
            session = self.session
            self.session = None
            process = self._preview_process
        _stop_process_group(process)
        if session:
            session.close()


class SpiralServer(ThreadingHTTPServer):
    daemon_threads = True
    # SO_REUSEADDR is set from main() for explicit ports; the default stays
    # False so an ephemeral auto-launch port can never be hijacked mid-restart.
    allow_reuse_address = False

    def __init__(self, address, credentials, state):
        super().__init__(address, SpiralHandler)
        self.credentials = list(credentials)
        self.state = state
        self.restart_requested = threading.Event()
        self._restart_lock = threading.Lock()
        self._restart_scheduled = False

    def request_restart(self):
        """Acknowledge first, then ask main() to close and re-exec the service."""
        with self._restart_lock:
            if not self._restart_scheduled:
                self._restart_scheduled = True
                timer = threading.Timer(0.1, self.restart_requested.set)
                timer.daemon = True
                timer.start()
        return {**self.state._base(), "restarting": True}


class SpiralHandler(BaseHTTPRequestHandler):
    server_version = "VC3D-Spiral/2"
    # HTTP/1.1 keeps connections alive so multi-file artifact transfers and
    # uploads do not pay a fresh TCP (or tunnel) setup per file.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("SPIRAL_HTTP " + (fmt % args), file=sys.stderr, flush=True)

    def _authorise(self):
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            token = header[len("Bearer "):].strip()
        else:
            # Compatibility alias for the original VC3D-owned local launch.
            token = self.headers.get("X-Spiral-Nonce", "")
        valid = False
        for credential in self.server.credentials:
            if secrets.compare_digest(token, credential):
                valid = True
        if not valid:
            raise ApiError(HTTPStatus.UNAUTHORIZED, "Invalid API key")

    def _body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
        if length < 0 or length > MAX_BODY_BYTES:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")

    def _send(self, status, value, *, close=False):
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(raw)

    def _parse_range(self, size):
        header = self.headers.get("Range")
        if not header:
            return None
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
        if not match or (not match.group(1) and not match.group(2)):
            raise ApiError(HTTPStatus.BAD_REQUEST, "Unsupported Range header")
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else size - 1
        else:
            # suffix form: last N bytes
            start = max(0, size - int(match.group(2)))
            end = size - 1
        if start >= size or end < start:
            raise ApiError(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
                           "Range is not satisfiable")
        return start, min(end, size - 1)

    def _send_artifact_file(self, artifact_id, relative_name):
        registry = self.server.state.artifacts
        artifact, path, info = registry.acquire_file(artifact_id, relative_name)
        try:
            size = info["size"]
            byte_range = self._parse_range(size)
            if byte_range is None:
                status, start, end = HTTPStatus.OK, 0, size - 1
            else:
                status, (start, end) = HTTPStatus.PARTIAL_CONTENT, byte_range
            length = max(0, end - start + 1) if size else 0
            self.send_response(int(status))
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("X-Spiral-Sha256", info["sha256"])
            if byte_range is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining > 0:
                    block = stream.read(min(TRANSFER_CHUNK_BYTES, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)
        finally:
            registry.release(artifact)

    def _dispatch(self):
        self._authorise()
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path).rstrip("/") or "/"
        if "\\" in path or "\x00" in path or "/../" in path + "/":
            raise ApiError(HTTPStatus.FORBIDDEN, "Malformed request path")
        state = self.server.state

        if self.command == "GET":
            if path == "/health":
                return state.health()
            if path == "/configuration":
                return state.configuration_catalog()
            if path == "/session/status":
                return state.status()
            if path == "/logs":
                values = parse_qs(parsed_url.query).get("after", ["0"])
                try:
                    after = int(values[-1])
                except (TypeError, ValueError):
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "The log cursor must be an integer")
                if after < 0:
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "The log cursor must not be negative")
                return state.logs.read_after(after)
            if path == "/dataset":
                return state.dataset()
            match = re.fullmatch(r"/artifacts/([A-Za-z0-9._-]+)/manifest", path)
            if match:
                return state.artifacts.manifest(match.group(1))
            match = re.fullmatch(r"/artifacts/([A-Za-z0-9._-]+)/files/(.+)", path)
            if match:
                if not _is_safe_relative_name(match.group(2)):
                    raise ApiError(HTTPStatus.FORBIDDEN, "Unsafe artifact file name")
                self._send_artifact_file(match.group(1), match.group(2))
                return None

        if self.command == "PUT":
            match = re.fullmatch(r"/session/inputs/([0-9a-f]{32})/files/(.+)", path)
            if match:
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                if length < 0:
                    raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
                return state.receive_upload_file(match.group(1), match.group(2),
                                                 self.rfile, length)

        if self.command == "DELETE":
            if path == "/session":
                body = self._body()
                return state.deduplicated(body.get("command_id"), state.delete)
            if path == "/session/ephemeral-inputs":
                body = self._body()
                return state.deduplicated(body.get("command_id"),
                                          lambda: state.remove_input(body))
            match = re.fullmatch(r"/session/inputs/([0-9a-f]{32})", path)
            if match:
                return state.delete_upload(match.group(1))

        if self.command == "POST":
            match = re.fullmatch(r"/session/inputs/([0-9a-f]{32})/finalize", path)
            if match:
                self._body()
                return state.finalize_upload(match.group(1))
            body = self._body()
            command_id = body.get("command_id")
            if path == "/dataset/resolve":
                return state.resolve(body.get("dataset_root", ""))
            if path == "/service/restart":
                return state.deduplicated(command_id, self.server.request_restart)
            if path == "/session/inputs":
                return state.begin_upload(body)
            if path == "/session/load":
                return state.deduplicated(command_id, lambda: state.load(body))
            if path == "/session/run/plan":
                return state.plan_run(body)
            if path == "/session/run":
                return state.deduplicated(command_id, lambda: state.run(body))
            if path == "/session/stop":
                return state.deduplicated(command_id, state.stop)
            if path == "/session/save-checkpoint":
                return state.deduplicated(command_id, lambda: state.save_checkpoint(body))
            if path == "/session/download-checkpoint":
                return state.deduplicated(command_id, state.download_checkpoint)
            if path == "/session/commit-inputs":
                return state.deduplicated(command_id, state.commit_inputs)
            if path == "/session/export-full":
                raise ApiError(HTTPStatus.NOT_IMPLEMENTED, "Full diagnostic export is not implemented by the interactive service")
        raise ApiError(HTTPStatus.NOT_FOUND, "Unknown endpoint")

    def _handle(self):
        try:
            response = self._dispatch()
            if response is not None:
                self._send(HTTPStatus.OK, response)
        except ApiError as exc:
            payload = self.server.state._base()
            payload.update({"error": exc.message, "details": exc.details})
            # The request body may not have been fully consumed; do not reuse
            # the connection after an error.
            self._send(exc.status, payload, close=True)
        except Exception as exc:
            payload = self.server.state._base()
            payload.update({"error": f"{type(exc).__name__}: {exc}"})
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, payload, close=True)

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle


def _install_parent_watch(parent_pid, shutdown):
    if not parent_pid:
        return
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            libc.prctl(1, signal.SIGTERM)
        except Exception:
            pass

    def watch():
        while not shutdown.is_set():
            try:
                os.kill(parent_pid, 0)
            except OSError:
                shutdown.set()
                return
            shutdown.wait(2.0)
    threading.Thread(target=watch, name="spiral-parent-watch", daemon=True).start()


def default_api_key_path():
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return config_home / "vc3d" / "spiral_api_key"


def load_or_create_api_key(path):
    """Load the API key file, generating a strong key with mode 0600 on first use."""
    path = Path(path).expanduser()
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key, False
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                 stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, (key + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return key, True


def _is_loopback(bind):
    if bind in ("localhost",):
        return True
    try:
        import ipaddress
        return ipaddress.ip_address(bind).is_loopback
    except ValueError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1",
                        help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=0,
                        help="Port (0 selects a free port; recommended only for "
                             "a VC3D-owned local process)")
    parser.add_argument("--api-key-file", default=None,
                        help="File holding the bearer API key; auto-generated at "
                             f"{default_api_key_path()} when omitted")
    parser.add_argument("--nonce", default=None,
                        help="Ephemeral credential for a VC3D-owned local process")
    parser.add_argument("--parent-pid", type=int, default=0)
    parser.add_argument("--dataset", default=None,
                        help="Dataset root owned by this service; required for a "
                             "non-loopback bind. Clients cannot repoint base inputs.")
    parser.add_argument("--service-name", default=None)
    parser.add_argument(
        "--session-name", type=parse_session_name, default=None, metavar="NAME",
        help="Stable output namespace under <dataset>/spiral_output; requires --dataset")
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=(0,), metavar="DEVICE[,DEVICE...]",
        help="Physical CUDA device indices to use (default: 0; example: 0,1,2,3)")
    args = parser.parse_args(argv)

    # fit_spiral and Torch are imported lazily when a session is loaded. Narrow
    # visibility now so even the single-process path consistently uses the
    # operator-selected physical device as its local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.gpus)

    loopback = _is_loopback(args.bind)
    if not loopback and not args.dataset:
        parser.error("--dataset is required for a non-loopback bind: remote "
                     "clients never supply host paths")
    if not loopback and args.nonce:
        parser.error("--nonce is only for VC3D-owned loopback processes; use the "
                     "API key file for network binds")
    if args.session_name and not args.dataset:
        parser.error("--session-name requires --dataset")

    credentials = []
    if args.nonce:
        credentials.append(args.nonce)
    else:
        key_path = Path(args.api_key_file).expanduser() if args.api_key_file \
            else default_api_key_path()
        key, created = load_or_create_api_key(key_path)
        credentials.append(key)
        print(f"SPIRAL_SERVICE_KEY_FILE {key_path}", flush=True)
        print(f"Spiral API key ({'generated' if created else 'reused'}; copy "
              f"into VC3D): {key}", flush=True)

    dataset_resolution = None
    session_lease = None
    if args.dataset:
        dataset_resolution = resolve_dataset_root(
            args.dataset, session_name=args.session_name or "")
        if not dataset_resolution.ok:
            print("Refusing to start: the launch dataset is incomplete.",
                  file=sys.stderr, flush=True)
            for key in dataset_resolution.missing_required:
                print(f"  missing required: {key}", file=sys.stderr, flush=True)
            for key, options in dataset_resolution.ambiguities.items():
                print(f"  ambiguous {key}: {', '.join(options)}",
                      file=sys.stderr, flush=True)
            return 2
        for warning in dataset_resolution.warnings:
            print(f"  dataset warning: {warning}", file=sys.stderr, flush=True)
        if args.session_name:
            output_directory = Path(
                dataset_resolution.resolved["output_directory"])
            try:
                output_directory.mkdir(parents=True, exist_ok=True)
                session_lease = ExclusiveFileLock(
                    output_directory / ".spiral-service.lock")
                session_lease.acquire()
            except FileLockUnavailable:
                print(
                    f"Refusing to start: Spiral session {args.session_name!r} "
                    "is already owned by another service process.",
                    file=sys.stderr, flush=True)
                return 2
            except OSError as exc:
                print(
                    f"Refusing to start: cannot create or lock named session "
                    f"output {output_directory}: {exc}",
                    file=sys.stderr, flush=True)
                return 2

    logs = ServiceLogBuffer()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, logs, "stdout")
    sys.stderr = _TeeStream(original_stderr, logs, "stderr")
    state = ServiceState(dataset_root=args.dataset,
                         dataset_resolution=dataset_resolution,
                         service_name=args.service_name,
                         session_name=args.session_name or "",
                         logs=logs,
                         gpu_ids=args.gpus)
    # A stable, operator-chosen port must survive TIME_WAIT restarts; an
    # ephemeral port must not reuse an address it did not own.
    SpiralServer.allow_reuse_address = args.port != 0
    try:
        server = SpiralServer((args.bind, args.port), credentials, state)
    except BaseException:
        if session_lease is not None:
            session_lease.release()
        raise
    shutdown = threading.Event()
    _install_parent_watch(args.parent_pid, shutdown)

    def gc_loop():
        while not shutdown.is_set():
            shutdown.wait(60.0)
            try:
                state.gc_uploads()
            except Exception:
                pass
    threading.Thread(target=gc_loop, name="spiral-upload-gc", daemon=True).start()

    def request_shutdown(_signum=None, _frame=None):
        shutdown.set()
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    # The ready line intentionally carries only the port. Clients learn the API
    # version from the authenticated /health handshake so local launch and
    # remote attach validate compatibility through one code path.
    print(f"Spiral CUDA devices: {','.join(str(gpu_id) for gpu_id in args.gpus)}",
          flush=True)
    if args.session_name:
        print(f"Spiral session name: {args.session_name}", flush=True)
    print(f"SPIRAL_SERVICE_READY port={server.server_port}", flush=True)
    server.timeout = 0.5
    try:
        while not shutdown.is_set():
            if server.restart_requested.is_set():
                break
            server.handle_request()
    finally:
        server.server_close()
        try:
            state.close()
        finally:
            if session_lease is not None:
                session_lease.release()
            sys.stdout, sys.stderr = original_stdout, original_stderr
    if server.restart_requested.is_set():
        restart_args = list(sys.argv[1:] if argv is None else argv)
        os.execv(sys.executable,
                 [sys.executable, str(Path(__file__).resolve()), *restart_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
