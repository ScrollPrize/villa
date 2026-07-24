#!/usr/bin/env python3
"""HTTP service for a persistent interactive Spiral fit.

The service binds to loopback by default. Non-loopback binds are explicit and
always carry bearer authentication; every client — including VC3D talking to a
process it launched itself — uses the same authenticated HTTP protocol.

Generated display data (previews, geometry snapshots, downloadable
checkpoints) is published as immutable, opaque artifacts and transferred
through ``/artifacts/...`` instead of host filesystem paths. Session inputs
(patches, fibers, PCL documents) can be uploaded into a session-scoped
ephemeral folder and later committed into the dataset.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
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
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from fit_session import (API_VERSION, PclRole, RUN_MUTABLE_BOOLEAN_KEYS,
                         RUN_MUTABLE_SAMPLING_KEYS,
                         parse_session_request,
                         resolve_dataset_root, validate_checkpoint_container,
                         validate_session_request)


SERVICE_VERSION = "5.1.0"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256
TRANSFER_CHUNK_BYTES = 1024 * 1024
PREVIEW_ARTIFACTS_KEPT = 3
CHECKPOINT_ARTIFACTS_KEPT = 2
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
MAX_LOG_ENTRIES = 2000
MAX_LOG_READ_ENTRIES = 1000
MAX_LOG_ENTRY_CHARS = 8192

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@ -]{0,127}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

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


def _validate_run_influence_config(value):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "influence_config must be a JSON object")
    allowed = {
        "interactive_influence_enabled",
        "interactive_influence_z",
        "interactive_influence_windings",
        "interactive_influence_theta_frac",
        "interactive_influence_disable_dt_frac",
        "interactive_influence_sigma",
        "interactive_influence_footprint_points",
        "interactive_influence_anchor_lattice_points",
        "interactive_influence_anchor_geometry_points",
        "interactive_influence_anchor_samples_per_step",
        "interactive_influence_anchor_ramp_power",
        "loss_weight_anchor",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       f"Unknown influence configuration keys: {unknown}")
    result = {}
    if "interactive_influence_enabled" in value:
        enabled = value["interactive_influence_enabled"]
        if not isinstance(enabled, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "interactive_influence_enabled must be boolean")
        result["interactive_influence_enabled"] = enabled
    ranges = {
        "interactive_influence_z": (1.0, 1_000_000.0),
        "interactive_influence_windings": (0.1, 100.0),
        "interactive_influence_theta_frac": (0.01, 1.0),
        "interactive_influence_disable_dt_frac": (0.0, 1.0),
        "interactive_influence_sigma": (0.000001, 10.0),
        "interactive_influence_footprint_points": (1.0, 1_000_000.0),
        "interactive_influence_anchor_lattice_points": (1.0, 1_000_000.0),
        "interactive_influence_anchor_geometry_points": (1.0, 100_000.0),
        "interactive_influence_anchor_samples_per_step": (1.0, 1_000_000.0),
        "interactive_influence_anchor_ramp_power": (0.000001, 100.0),
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
        "interactive_influence_footprint_points",
        "interactive_influence_anchor_lattice_points",
        "interactive_influence_anchor_geometry_points",
        "interactive_influence_anchor_samples_per_step",
    }
    for key in integer_keys & result.keys():
        if not result[key].is_integer():
            raise ApiError(HTTPStatus.BAD_REQUEST, f"{key} must be an integer")
        result[key] = int(result[key])
    return result


def _validate_run_config(value, current, limits=None):
    """Validate settings which the resident fitter can change between Runs."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "run_config must be a JSON object")
    current = current if isinstance(current, dict) else {}
    limits = limits if isinstance(limits, dict) else {}
    unknown = sorted(set(value) - set(current))
    if unknown:
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       f"Unknown or non-mutable Run configuration keys: {unknown}")

    result = {}
    for key, item in value.items():
        if key == "track_length_bin_weights":
            if item is None:
                result[key] = None
                continue
            if (not isinstance(item, list) or len(item) != 3
                    or any(isinstance(weight, bool)
                           or not isinstance(weight, (int, float))
                           or not math.isfinite(float(weight))
                           or float(weight) < 0 for weight in item)
                    or sum(float(weight) for weight in item) <= 0):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"{key} must be null or three finite non-negative weights "
                    "with a positive sum")
            result[key] = [float(weight) for weight in item]
            continue
        if key == "max_track_crossing_per_step":
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or not float(item).is_integer() or int(item) < 0):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               f"{key} must be a non-negative integer")
            maximum = limits.get(key)
            if (not isinstance(maximum, bool)
                    and isinstance(maximum, (int, float))
                    and int(item) > int(maximum)):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"{key} cannot exceed this session's prepared limit ({int(maximum)})")
            result[key] = int(item)
            continue
        if key in ("track_min_sample_spacing", "track_max_sample_spacing"):
            if (isinstance(item, bool) or not isinstance(item, (int, float))
                    or not math.isfinite(float(item)) or float(item) <= 0):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               f"{key} must be a finite positive number")
            result[key] = float(item)
            continue
        if key in RUN_MUTABLE_BOOLEAN_KEYS:
            if not isinstance(item, bool):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               f"{key} must be boolean")
            result[key] = item
            continue
        if key.startswith("loss_start_") and item is None:
            if key == "loss_start_patch_dt":
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "loss_start_patch_dt cannot be null")
            result[key] = None
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"{key} must be numeric")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"{key} must be a finite non-negative number")
        if key in RUN_MUTABLE_SAMPLING_KEYS or key.startswith("loss_start_"):
            if not number.is_integer():
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               f"{key} must be an integer")
            number = int(number)
            if key in RUN_MUTABLE_SAMPLING_KEYS and number < 1:
                # Disabled optional inputs have their loss weights and sample
                # counts forced to zero when the session is loaded.  VC3D
                # round-trips those advertised active values on every Run.
                # Permit that unchanged disabled value, while still rejecting
                # attempts to turn an active sampler off by setting its count
                # to zero at a Run boundary.
                current_value = current.get(key)
                current_is_zero = (
                    not isinstance(current_value, bool)
                    and isinstance(current_value, (int, float))
                    and float(current_value) == 0.0
                )
                if number != 0 or not current_is_zero:
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   f"{key} must be at least 1")
        result[key] = number
    effective = dict(current)
    effective.update(result)
    minimum = effective.get("track_min_sample_spacing")
    maximum = effective.get("track_max_sample_spacing")
    if (not isinstance(minimum, bool) and isinstance(minimum, (int, float))
            and not isinstance(maximum, bool) and isinstance(maximum, (int, float))
            and float(minimum) > float(maximum)):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "track_min_sample_spacing must be <= track_max_sample_spacing")
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
                           entry_point, *, delete_root_on_prune=False):
        root = Path(root).resolve(strict=True)
        files = {}
        for directory, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames.sort()
            for filename in sorted(filenames):
                path = Path(directory) / filename
                if path.is_symlink() or not path.is_file():
                    continue
                relative = path.relative_to(root).as_posix()
                files[relative] = {"size": path.stat().st_size,
                                   "sha256": _sha256_file(path)}
                if len(files) > MAX_ARTIFACT_FILES:
                    raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                                   "Artifact has too many files to register")
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


class ServiceState:
    def __init__(self, dataset_root=None, dataset_resolution=None,
                 service_name=None, logs=None, gpu_ids=(0,)):
        self.lock = threading.RLock()
        self.session = None
        self.session_id = None
        self.session_paths = None
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
        self.logs = logs if logs is not None else ServiceLogBuffer()
        self.gpu_ids = tuple(gpu_ids)
        self.artifacts = ArtifactRegistry()
        self.uploads = {}
        self.ephemeral_records = []
        self._registered_preview_generation = 0
        self._preview_artifact = None
        self._geometry_artifact = None
        self._registered_geometry_manifest = None

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    def _base(self):
        return {
            "api_version": API_VERSION,
            "service_version": SERVICE_VERSION,
            "service_name": self.service_name,
            "session_id": self.session_id,
            "service_generation": self.service_generation,
            "session_generation": self.session_generation,
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
            })
            response["preview_artifact"] = self._preview_artifact
            response["geometry_artifact"] = self._geometry_artifact
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
        return {**self._base(), **resolve_dataset_root(root_value).to_dict()}

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

        # A dataset-owned service resolves conventional paths itself, but the
        # client still controls which optional sources belong to this session.
        # Clear disabled paths so the manifest and worker agree that they were
        # not loaded.
        config = (request.get("run") or {}).get("config") or {}
        selected_paths = {
            "use_verified_patches": ("verified_patches",),
            "use_unverified_patches": ("unverified_patches",),
            "use_normals": ("normal_x", "normal_y"),
            "use_surf_sdt": ("surf_sdt",),
            "use_tracks": ("tracks_dbm",),
            "use_gradient_magnitude": ("gradient_magnitude",),
            "use_fibers": ("fibers",),
        }
        for flag, field_names in selected_paths.items():
            if not bool(config.get(flag, True)):
                for field_name in field_names:
                    paths[field_name] = ""

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
                self._reset_session_scope()
                self.session = create_session(
                    paths, run, preview, self._status_changed,
                    gpu_ids=self.gpu_ids)
                self.status_generation += 1
                response = self.status()
                response["accepted"] = True
                return response
        finally:
            with self.lock:
                self.replacing = False

    def _reset_session_scope(self):
        self.ephemeral_records = []
        self.uploads = {}
        self._registered_preview_generation = 0
        self._preview_artifact = None
        self._geometry_artifact = None
        self._registered_geometry_manifest = None

    def _status_changed(self, status):
        # Runs on the fitter thread inside the pause/export window, so artifact
        # digests are computed while training is stopped.
        try:
            self._maybe_register_artifacts(status)
        except Exception as exc:
            print(f"SPIRAL_ARTIFACT_ERROR {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        with self.lock:
            self.status_generation += 1

    def _maybe_register_artifacts(self, status):
        with self.lock:
            session_id = self.session_id
            preview_generation = int(status.get("preview_generation") or 0)
            preview_manifest = status.get("preview_manifest_path")
            geometry_manifest = status.get("geometry_snapshot_manifest_path")
            register_preview = (preview_manifest
                                and preview_generation > self._registered_preview_generation)
            register_geometry = (geometry_manifest
                                 and geometry_manifest != self._registered_geometry_manifest)
        if register_preview:
            manifest_path = Path(preview_manifest)
            ref = self.artifacts.register_directory(
                "spiral-preview", session_id, preview_generation,
                manifest_path.parent, manifest_path.name,
                delete_root_on_prune=True)
            with self.lock:
                if self.session_id == session_id:
                    self._preview_artifact = ref
                    self._registered_preview_generation = preview_generation
            self.artifacts.prune("spiral-preview", session_id, PREVIEW_ARTIFACTS_KEPT)
        if register_geometry:
            manifest_path = Path(geometry_manifest)
            ref = self.artifacts.register_directory(
                "spiral-geometry", session_id, 1,
                manifest_path.parent, manifest_path.name)
            with self.lock:
                if self.session_id == session_id:
                    self._geometry_artifact = ref
                    self._registered_geometry_manifest = geometry_manifest

    def run(self, request):
        session = self._require_session()
        status = session.status()
        influence_config = _validate_run_influence_config(
            request.get("influence_config"))
        run_config = _validate_run_config(
            request.get("run_config"), status.get("run_config"),
            status.get("run_config_limits"))
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

        target = session.run(int(request.get("iterations", 0)),
                             pending_inputs=pending,
                             mark_incorporated=mark_incorporated,
                             influence_config=influence_config,
                             run_config=run_config)
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True, "target_iteration": target}

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
        with self.lock:
            if kind == "checkpoint":
                # Resume checkpoints are needed before a session exists, so
                # they are service-scoped: allowed whenever an output
                # directory is known (a --dataset launch or a live session).
                if self._output_root() is None:
                    raise ApiError(HTTPStatus.CONFLICT,
                                   "Checkpoint uploads need a --dataset service "
                                   "or an active session")
                if len(manifest) != 1:
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "A checkpoint upload must declare exactly one file")
                if declared > MAX_CHECKPOINT_UPLOAD_BYTES:
                    raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                                   "The checkpoint exceeds the upload size limit")
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
        destination = root / upload.input_id
        if destination.exists():
            # Never overwrite: an earlier upload with the same name may be the
            # one a resident session is resuming from.
            destination = root / f"{destination.stem}-{secrets.token_hex(4)}{destination.suffix}"
        os.replace(source, destination)
        shutil.rmtree(upload.staging_dir, ignore_errors=True)
        self._prune_uploaded_checkpoints(destination)
        return {
            "id": upload.input_id,
            "kind": "checkpoint",
            "role": None,
            "path": str(destination),
            "bytes": upload.declared_bytes(),
            "state": "uploaded",
            "upload_id": upload.upload_id,
        }

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
            records = [record for record in self.ephemeral_records
                       if record["state"] in ("pending", "incorporated")
                       and not record.get("committed")]
            paths = self.session_paths
        dataset_root = Path(paths.dataset_root)
        patches_dir = Path(paths.verified_patches) if paths.verified_patches \
            else dataset_root / "verified_patches"
        fibers_dir = Path(paths.fibers) if paths.fibers else dataset_root / "fibers"

        # Validate everything before moving anything: a patch-id collision is a
        # commit error, not an overwrite.
        for record in records:
            if record["kind"] == "patch" and (patches_dir / record["id"]).exists():
                raise ApiError(HTTPStatus.CONFLICT,
                               f"A patch named {record['id']!r} already exists in the dataset")
            if record["kind"] == "fiber" and (fibers_dir / f"{record['id']}.json").exists():
                raise ApiError(HTTPStatus.CONFLICT,
                               f"A fiber named {record['id']!r} already exists in the dataset")

        committed = []
        for record in records:
            source = Path(record["path"])
            # A still-pending record keeps its staged copy: it remains the
            # incorporation source for the next run, so committing never
            # removes an input from the live session's queue and never races
            # a concurrent run over the record's path.
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
                temp = target.with_name(f".{target.name}.incoming-{secrets.token_hex(4)}")
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
            # Committed records that already joined the resident fit are done;
            # committed-but-pending ones stay queued so they still join the
            # fit on the next run.
            self.ephemeral_records = [record for record in self.ephemeral_records
                                      if not (record.get("committed")
                                              and record["state"] == "incorporated")]
            if self.dataset_resolution is not None:
                self.dataset_resolution = resolve_dataset_root(self.dataset_root)
            self.status_generation += 1
        return {**self.status(), "committed": committed, "accepted": True}

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
    if args.dataset:
        dataset_resolution = resolve_dataset_root(args.dataset)
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

    logs = ServiceLogBuffer()
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, logs, "stdout")
    sys.stderr = _TeeStream(original_stderr, logs, "stderr")
    state = ServiceState(dataset_root=args.dataset,
                         dataset_resolution=dataset_resolution,
                         service_name=args.service_name,
                         logs=logs,
                         gpu_ids=args.gpus)
    # A stable, operator-chosen port must survive TIME_WAIT restarts; an
    # ephemeral port must not reuse an address it did not own.
    SpiralServer.allow_reuse_address = args.port != 0
    server = SpiralServer((args.bind, args.port), credentials, state)
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
    print(f"SPIRAL_SERVICE_READY port={server.server_port}", flush=True)
    server.timeout = 0.5
    try:
        while not shutdown.is_set():
            if server.restart_requested.is_set():
                break
            server.handle_request()
    finally:
        server.server_close()
        state.close()
        sys.stdout, sys.stderr = original_stdout, original_stderr
    if server.restart_requested.is_set():
        restart_args = list(sys.argv[1:] if argv is None else argv)
        os.execv(sys.executable,
                 [sys.executable, str(Path(__file__).resolve()), *restart_args])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
