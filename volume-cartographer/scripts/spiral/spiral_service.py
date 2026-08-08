#!/usr/bin/env python3
"""HTTP service for a persistent interactive Spiral fit.

The service binds to loopback by default. Non-loopback binds are explicit and
always carry bearer authentication; every client — including VC3D talking to a
process it launched itself — uses the same authenticated HTTP protocol.

Every service is bound to one dataset at startup: ``--dataset`` (inputs,
resolved once and advertised through ``/dataset``) and ``--output`` (all
generated state) are required; ``--cache`` defaults to the documented user
cache (``$XDG_CACHE_HOME/vc3d/spiral``). Both --output and --cache must
resolve outside the dataset root — the dataset holds inputs only.

Generated display data (previews, downloadable
checkpoints) is published as immutable, opaque artifacts and transferred
through ``/artifacts/...`` instead of host filesystem paths. Session inputs
(patches, fibers, PCL documents) can be uploaded into a session-scoped
ephemeral folder and later committed into the dataset.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
import errno
import json
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

from fit_session import (API_VERSION, FIT_INPUT_CATALOG, ScrollSpecError,
                         SpiralInputPaths, default_user_cache_dir, fit_input,
                         input_change_impact, load_scroll_spec,
                         parse_session_request, resolve_dataset_root,
                         validate_session_request)
from config import Config
from service_http import (ApiError, TRANSFER_CHUNK_BYTES,
                          is_safe_relative_name)
from service_artifacts import ArtifactRegistry
from service_uploads import (PCL_ROLE_FILES, UPLOADED_CHECKPOINTS_KEPT,
                             UPLOAD_GC_SECONDS, UploadEnvironment,
                             UploadManager, _copy_publish,
                             _merge_pcl_documents, _utc_stamp)
from lasagna_publish import LasagnaPublisher, stop_process_group
# Re-exported for the service's own test surface, which addresses the preview
# mapping helpers through this module.
from lasagna_publish import (_load_flatten_correspondence,  # noqa: F401
                             _mapped_winding_ids,
                             _prepare_cleaned_lasagna_surface,
                             _raw_run_diff_rgba, _sample_rgba_through_map,
                             _validate_tifxyz_output_step)


SERVICE_VERSION = "6.1.0"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256
PREVIEW_ARTIFACTS_KEPT = 3
CHECKPOINT_ARTIFACTS_KEPT = 2
EPHEMERAL_QUOTA_BYTES = int(os.environ.get("SPIRAL_EPHEMERAL_QUOTA_BYTES",
                                           4 * 1024 * 1024 * 1024))
# This buffer is also the reconnect/late-attach history for a remote VC3D
# client.  tqdm produces one entry for each carriage-return redraw, so leave
# enough room for the loading bars and a substantial portion of a long fit.
MAX_LOG_ENTRIES = 20000
MAX_LOG_READ_ENTRIES = 1000
MAX_LOG_ENTRY_CHARS = 8192
# Structured event ring served through /events. Sized like the log relay so
# a reconnecting client can recover a comparable window of history.
MAX_EVENT_ENTRIES = 20000
MAX_EVENT_READ_ENTRIES = 1000
# High-frequency event kinds (per-iteration metrics, progress redraws)
# coalesce to at most ~one record per key per interval. The interval matches
# the ProgressReporter publish interval, so the event stream carries the same
# cadence a status poller already observes.
EVENT_COALESCE_SECONDS = 1.0
DATASET_COMMIT_LOCK_TIMEOUT_SECONDS = 20.0

_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Base input paths are owned by the service (every launch carries --dataset);
# a load request may only choose among service-advertised values for these
# keys.
_DATASET_CLIENT_SELECTABLE = ("checkpoint", "tracks_dbm")


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


def bind_service_paths(resolution, output_directory, cache_directory):
    """Attach the startup-resolved output/cache roots to the advertisement.

    Dataset resolution describes inputs only; where generated state lives
    (--output) and where derived host caches live (--cache) are service
    startup decisions. /dataset advertises the bound result so clients see
    one immutable set of paths.
    """
    resolution.resolved["output_directory"] = str(output_directory)
    resolution.resolved["cache_directory"] = str(cache_directory)
    return resolution


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


# Console lines whose information is already published as structured
# /events records: ProgressReporter console snapshots and the fitter's
# periodic step-metric prints. They stay on the terminal and in the /logs
# compatibility relay, but the event stream must not double-report them as
# log records next to the structured progress/metric records.
_STRUCTURED_CONSOLE_LINE = re.compile(r"^(?:PROGRESS |step \d+: loss = )")


class ServiceEventBuffer:
    """Bounded ring of structured service events served through ``/events``.

    Every record carries a monotonically increasing ``sequence``;
    ``GET /events?cursor=N`` returns records with ``sequence > N`` plus
    ``next_cursor`` for the following read. Cursor semantics:

    * A cursor newer than the newest record (a cursor kept across a service
      restart) answers ``cursor_reset`` true and the read restarts from the
      beginning of the retained ring.
    * A cursor older than the ring start answers ``overrun`` true with
      ``dropped``/``dropped_from`` describing the gap. The event stream is
      bounded history, not reconnect state: an overrun client refreshes its
      durable view from ``/session/status`` and continues from
      ``next_cursor``.

    Reconnect protocol: read the ``/session/status`` snapshot first, then
    subscribe from the cursor position the first ``/events`` read reports.

    Records submitted with a ``coalesce_key`` are rate limited: while a
    record with the same key was emitted less than ``coalesce_seconds`` ago,
    the newest record is parked in a per-key pending slot (replacing any
    older pending record) and flushed on the next append or read once the
    interval has elapsed. The ring therefore stores at most ~one record per
    key per interval while the latest values still reach clients.
    """

    def __init__(self, max_entries=MAX_EVENT_ENTRIES,
                 coalesce_seconds=EVENT_COALESCE_SECONDS,
                 clock=time.monotonic):
        self._lock = threading.Lock()
        self._entries = deque(maxlen=max_entries)
        self._next_sequence = 1
        self._coalesce_seconds = float(coalesce_seconds)
        self._clock = clock
        self._pending = {}
        self._last_emit = {}
        # Stamps records that do not carry an explicit session generation.
        # Must never take another lock: it is called under this buffer's own.
        self.session_generation_provider = None

    def append(self, kind, text="", *, severity="info", source="service",
               rank=None, session_generation=None, operation=None,
               payload=None, coalesce_key=None, force=False):
        now = self._clock()
        with self._lock:
            if session_generation is None \
                    and self.session_generation_provider is not None:
                try:
                    session_generation = self.session_generation_provider()
                except Exception:
                    session_generation = None
            record = {
                "timestamp": time.time(),
                "severity": str(severity),
                "kind": str(kind),
                "source": str(source),
                "rank": rank,
                "session_generation": session_generation,
                "operation": operation,
                "text": str(text or ""),
                "payload": payload,
            }
            self._flush_due(now)
            if coalesce_key is None:
                self._append(record)
                return
            last = self._last_emit.get(coalesce_key)
            if force or last is None or now - last >= self._coalesce_seconds:
                self._pending.pop(coalesce_key, None)
                self._last_emit[coalesce_key] = now
                self._append(record)
            else:
                self._pending[coalesce_key] = record

    def _append(self, record):
        record["sequence"] = self._next_sequence
        self._next_sequence += 1
        self._entries.append(record)

    def _flush_due(self, now):
        for key in list(self._pending):
            if now - self._last_emit.get(key, float("-inf")) \
                    >= self._coalesce_seconds:
                self._last_emit[key] = now
                self._append(self._pending.pop(key))

    def read_after(self, cursor, limit=MAX_EVENT_READ_ENTRIES):
        limit = max(1, min(int(limit), MAX_EVENT_READ_ENTRIES))
        cursor = int(cursor)
        with self._lock:
            self._flush_due(self._clock())
            latest = self._next_sequence - 1
            cursor_reset = cursor > latest
            if cursor_reset:
                cursor = 0
            oldest = (self._entries[0]["sequence"] if self._entries
                      else self._next_sequence)
            dropped = max(0, oldest - max(0, cursor + 1))
            events = [dict(record) for record in self._entries
                      if record["sequence"] > cursor][:limit]
            next_cursor = events[-1]["sequence"] if events \
                else min(cursor, latest)
        return {
            "events": events,
            "next_cursor": next_cursor,
            "latest_sequence": latest,
            "dropped": dropped,
            "dropped_from": (cursor + 1) if dropped else None,
            "overrun": dropped > 0,
            "cursor_reset": cursor_reset,
        }


class ServiceLogBuffer:
    """Bounded, incremental copy of the service's stdout and stderr lines.

    When an event buffer is attached, every complete non-structured console
    line is also published as a ``log``-kind event record; lines already
    covered by structured progress/metric events are kept out of the event
    stream so the same information is never double-reported.
    """

    def __init__(self, max_entries=MAX_LOG_ENTRIES, events=None):
        self._lock = threading.Lock()
        self._entries = deque(maxlen=max_entries)
        self._pending = {"stdout": "", "stderr": ""}
        self._next_sequence = 1
        self._events = events

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
                if len(line) > MAX_LOG_ENTRY_CHARS:
                    line = line[:MAX_LOG_ENTRY_CHARS] + " … [truncated]"
                self._entries.append({
                    "sequence": self._next_sequence,
                    "stream": stream,
                    "text": line,
                })
                self._next_sequence += 1
                if self._events is not None \
                        and not _STRUCTURED_CONSOLE_LINE.match(line):
                    self._events.append("log", line, source=stream)

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


class ServiceState:
    def __init__(self, dataset_root=None, dataset_resolution=None,
                 service_name=None, session_name="", logs=None, events=None,
                 gpu_ids=(0,)):
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
        self.events = events if events is not None else ServiceEventBuffer()
        # Log-kind records produced by the console tee carry the current
        # session generation. Reading the attribute is lock-free by design;
        # the provider runs under the event buffer's own lock.
        self.events.session_generation_provider = \
            lambda: self.session_generation
        # Per-rank change trackers so repeated status snapshots do not
        # re-emit identical structured events.
        self._event_progress_signatures = {}
        self._event_metric_iterations = {}
        self._event_errors = {}
        self.gpu_ids = tuple(gpu_ids)
        self.artifacts = ArtifactRegistry()
        self.uploads_manager = UploadManager(self._upload_environment())
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
            # The status snapshot carries raw progress facts only. ETA is a
            # presentation value clients derive from step/total/elapsed.
            if isinstance(response.get("progress"), dict):
                response["progress"] = {
                    key: value
                    for key, value in response["progress"].items()
                    if key != "eta_seconds"
                }
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
                    response["progress"] = {
                        "operation": "publishing_preview",
                        "stage_name": stage_name,
                        "detail": None,
                        "step": int(step) if step is not None else None,
                        "total_steps": (
                            int(total) if total is not None else None),
                        "unit": "steps",
                        "elapsed_seconds": elapsed,
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
        for key in (*(spec.key for spec in FIT_INPUT_CATALOG
                      if spec.kind != "pcl-set"),
                    "output_directory", "cache_directory"):
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
        # The scroll specification is resolved from the dataset root; it
        # carries the physical scroll facts (including the outward sense,
        # which is not part of the load request).
        scroll = None
        try:
            scroll = load_scroll_spec(paths.dataset_root)
        except ScrollSpecError as exc:
            errors.append({"field": "scroll_spec", "message": str(exc)})
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
                        paths, run, preview, scroll, self._status_changed,
                        gpu_ids=self.gpu_ids,
                        event_callback=self._session_event)
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
        self._event_progress_signatures = {}
        self._event_metric_iterations = {}
        self._event_errors = {}
        self.ephemeral_records = []
        self.uploads_manager.reset()
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

    def _session_event(self, rank, status):
        """Derive structured event records from one rank's status snapshot.

        The single-GPU runtime reports as rank 0; child ranks of a
        distributed session publish their snapshots through the parent
        queue and arrive here tagged with their originating rank, so every
        record names the process that produced it.
        """
        if not isinstance(status, dict):
            return
        generation = self.session_generation
        progress = status.get("progress")
        if isinstance(progress, dict):
            # Elapsed time changes on every snapshot; only a change in the
            # underlying stage/step content is a new progress event.
            signature = {key: value for key, value in progress.items()
                         if key not in ("elapsed_seconds", "eta_seconds")}
            with self.lock:
                changed = self._event_progress_signatures.get(rank) != signature
                if changed:
                    self._event_progress_signatures[rank] = signature
            if changed:
                step = progress.get("step")
                total = progress.get("total_steps")
                finished = (isinstance(step, int) and isinstance(total, int)
                            and total > 0 and step >= total)
                self.events.append(
                    "progress", str(progress.get("stage_name") or ""),
                    source="fitter", rank=rank,
                    session_generation=generation,
                    operation=progress.get("operation"),
                    payload={key: value for key, value in progress.items()
                             if key != "eta_seconds"},
                    coalesce_key=("progress", rank), force=finished)
        metrics = status.get("latest_metrics")
        iteration = status.get("current_iteration")
        if metrics and isinstance(iteration, int):
            with self.lock:
                emit = iteration > self._event_metric_iterations.get(rank, -1)
                if emit:
                    self._event_metric_iterations[rank] = iteration
            if emit:
                self.events.append(
                    "metric", f"iteration {iteration}",
                    source="fitter", rank=rank,
                    session_generation=generation,
                    operation="optimizing",
                    payload={"iteration": iteration, **dict(metrics)},
                    coalesce_key=("metric", rank))
        error = status.get("error")
        if status.get("state") == "Error" and error:
            with self.lock:
                emit = self._event_errors.get(rank) != error
                if emit:
                    self._event_errors[rank] = error
            if emit:
                self.events.append(
                    "error", str(error), severity="error", source="fitter",
                    rank=rank, session_generation=generation)

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
            self.events.append(
                "error", f"Preview publication failed: {error}",
                severity="error", source="service",
                operation="publishing_preview")
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
            snapshot = dict(current)
        self.events.append(
            "progress", str(snapshot.get("stage_name") or ""),
            source="service", operation="publishing_preview",
            payload=snapshot, coalesce_key=("preview-publish",))

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
        input_changes = []
        for key, value in sorted(path_changes.items()):
            impact, path_dependencies = input_change_impact(key)
            impacts.add(impact)
            dependencies.extend(path_dependencies)
            input_changes.append({
                "key": key,
                "before": current_manifest.get(key),
                "after": value,
                "runtime_impact": impact,
            })
        # Path changes a resident session takes live (without a session
        # reload) are validated eagerly against their catalog kind; changes
        # that force a reload are validated when the session reloads.
        for key in sorted(path_changes):
            spec = fit_input(key)
            if spec is None or spec.runtime_impact == "prepared_input_rebuild":
                continue
            value = str(path_changes[key] or "").strip()
            if spec.kind in ("directory", "zarr-group") and (
                    not value or not Path(value).is_dir()):
                label = spec.key.replace("_", " ").capitalize()
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    f"{label} path is not a readable directory",
                    [{"field": spec.key,
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
        """Run one Lasagna preview publication for this session.

        The publisher owns the whole operation; this method only binds it to
        the current session: one progress path into
        ``_update_preview_publish``, the subprocess handle the service kills
        on shutdown, the session-validity check, and the previous raw
        generation the run-difference overlay is built against.
        """
        with self.lock:
            output_directory = self.session_paths.output_directory
            voxel_size_um = (
                (self.session_request or {}).get("run") or {}
            ).get("voxel_size_um")

        def attach_process(process):
            with self.lock:
                if (self.session_id == session_id
                        and self._publishing_preview_generation == generation):
                    self._preview_process = process

        def detach_process(process):
            with self.lock:
                if self._preview_process is process:
                    self._preview_process = None
                self.status_generation += 1

        def session_valid():
            with self.lock:
                return self.session_id == session_id

        def previous_raw_manifest():
            with self.lock:
                return self._previous_raw_preview_manifest

        def adopt_raw_manifest(path):
            with self.lock:
                old_raw = self._previous_raw_preview_manifest
                self._previous_raw_preview_manifest = path
            return old_raw

        publisher = LasagnaPublisher(
            progress=lambda **values: self._update_preview_publish(
                generation, **values),
            attach_process=attach_process,
            detach_process=detach_process,
            session_valid=session_valid,
            previous_raw_manifest=previous_raw_manifest,
            adopt_raw_manifest=adopt_raw_manifest)
        return publisher.publish(
            preview_manifest_path, session_id=session_id,
            generation=generation, output_directory=output_directory,
            voxel_size_um=voxel_size_um)

    # ------------------------------------------------------------------
    # Session input uploads
    # ------------------------------------------------------------------

    @property
    def uploads(self):
        """Uploads in flight, keyed by upload ID (owned by the manager)."""
        return self.uploads_manager.uploads

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
        return self.uploads_manager.staging_root()

    def _checkpoint_upload_root(self):
        return self.uploads_manager.checkpoint_root()

    def _upload_environment(self):
        """The whole of what the upload manager may ask this service."""
        return UploadEnvironment(
            lock=self.lock,
            output_root=self._output_root,
            session_id=lambda: self.session_id,
            ephemeral_dir=self._session_ephemeral_dir,
            require_session=self._require_session,
            active_checkpoint=self._active_checkpoint,
            reserve_ephemeral=self._reserve_ephemeral)

    def _active_checkpoint(self):
        with self.lock:
            return self.session_paths.checkpoint if self.session_paths else ""

    def _reserve_ephemeral(self, kind, input_id, declared):
        """Admit one new ephemeral input, or refuse it.

        Duplicate identities and the ephemeral quota are ledger questions,
        not transfer questions, so the upload manager delegates them here.
        """
        with self.lock:
            if any(record["id"] == input_id and record["kind"] == kind
                   for record in self.ephemeral_records):
                raise ApiError(HTTPStatus.CONFLICT,
                               f"An ephemeral {kind} named {input_id!r} already exists")
            if self._ephemeral_bytes_in_use() + declared > EPHEMERAL_QUOTA_BYTES:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               "The ephemeral input quota is exhausted")

    def _ephemeral_bytes_in_use(self):
        total = sum(record["bytes"] for record in self.ephemeral_records)
        return total + self.uploads_manager.staged_ephemeral_bytes()

    def begin_upload(self, request):
        return {**self._base(), **self.uploads_manager.begin(request)}

    def receive_upload_file(self, upload_id, relative_name, stream, length):
        received = self.uploads_manager.receive(
            upload_id, relative_name, stream, length)
        return {**self._base(), "received": received, "accepted": True}

    def finalize_upload(self, upload_id):
        finalized = self.uploads_manager.finalize(upload_id)
        if not finalized.replayed:
            with self.lock:
                if finalized.kind != "checkpoint":
                    self.ephemeral_records.append(finalized.record)
                self.status_generation += 1
        return {**self.status(), "input": dict(finalized.record),
                "accepted": True}

    def delete_upload(self, upload_id):
        self.uploads_manager.delete(upload_id)
        return {**self._base(), "deleted": True}

    def gc_uploads(self):
        self.uploads_manager.collect_garbage()

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
                    target = dataset_root / PCL_ROLE_FILES[record["role"]]
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
                    # Re-advertise the dataset with the committed inputs, but
                    # keep the startup-bound output/cache roots: deployment
                    # paths never change after launch.
                    previous = self.dataset_resolution.resolved
                    self.dataset_resolution = bind_service_paths(
                        resolve_dataset_root(self.dataset_root),
                        previous.get("output_directory", ""),
                        previous.get("cache_directory", ""))
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
        stop_process_group(process)
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

    def log_request(self, code="-", size="-"):
        """Suppress successful polling requests at the source.

        Status, log, and event reads arrive several times a second from
        every connected client; logging them would drown the terminal and
        the relay buffers in access lines. Failed polls still log.
        """
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if self.command == "GET" and 200 <= status < 400:
            path = urlparse(self.path).path.rstrip("/")
            if path in ("/session/status", "/logs", "/events"):
                return
        super().log_request(code, size)

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
            if path == "/events":
                query = parse_qs(parsed_url.query)
                try:
                    cursor = int(query.get("cursor", ["0"])[-1])
                    limit = int(query.get(
                        "limit", [str(MAX_EVENT_READ_ENTRIES)])[-1])
                except (TypeError, ValueError):
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "The event cursor and limit must be integers")
                if cursor < 0 or limit < 1:
                    raise ApiError(HTTPStatus.BAD_REQUEST,
                                   "The event cursor must not be negative and "
                                   "the limit must be at least 1")
                return state.events.read_after(cursor, limit)
            if path == "/dataset":
                return state.dataset()
            match = re.fullmatch(r"/artifacts/([A-Za-z0-9._-]+)/manifest", path)
            if match:
                return state.artifacts.manifest(match.group(1))
            match = re.fullmatch(r"/artifacts/([A-Za-z0-9._-]+)/files/(.+)", path)
            if match:
                if not is_safe_relative_name(match.group(2)):
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
    parser.add_argument("--dataset", required=True,
                        help="Dataset root owned by this service (inputs only; "
                             "resolved once at startup and advertised through "
                             "/dataset). Clients cannot repoint base inputs.")
    parser.add_argument("--output", required=True,
                        help="Root for all generated state (run directories, "
                             "autosaves, previews, ephemeral inputs, upload "
                             "staging, uploaded checkpoints). Must resolve "
                             "outside the dataset root.")
    parser.add_argument("--cache", default=None,
                        help="Directory for derived host caches; must resolve "
                             "outside the dataset root (default: "
                             "$XDG_CACHE_HOME/vc3d/spiral, i.e. "
                             "~/.cache/vc3d/spiral)")
    parser.add_argument("--service-name", default=None)
    parser.add_argument(
        "--session-name", type=parse_session_name, default=None, metavar="NAME",
        help="Stable output namespace: generated state moves to "
             "<output>/NAME, held under an exclusive lease")
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=(0,), metavar="DEVICE[,DEVICE...]",
        help="Physical CUDA device indices to use (default: 0; example: 0,1,2,3)")
    args = parser.parse_args(argv)

    # fit_spiral and Torch are imported lazily when a session is loaded. Narrow
    # visibility now so even the single-process path consistently uses the
    # operator-selected physical device as its local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.gpus)

    loopback = _is_loopback(args.bind)
    if not loopback and args.nonce:
        parser.error("--nonce is only for VC3D-owned loopback processes; use the "
                     "API key file for network binds")

    # Deployment roots are bound once at startup. --output owns every piece
    # of generated state; the dataset root holds inputs only, so both the
    # output and cache roots must resolve (realpath) outside it.
    dataset_root = Path(args.dataset).expanduser().resolve(strict=False)
    output_root = Path(args.output).expanduser().resolve(strict=False)
    if output_root == dataset_root or output_root.is_relative_to(dataset_root):
        parser.error(f"--output must resolve outside the dataset root: "
                     f"{output_root} is inside {dataset_root}")
    if args.session_name:
        output_root = output_root / args.session_name
    cache_root = Path(args.cache).expanduser().resolve(strict=False) \
        if args.cache else Path(default_user_cache_dir())
    if cache_root == dataset_root or cache_root.is_relative_to(dataset_root):
        parser.error(f"--cache must resolve outside the dataset root: "
                     f"{cache_root} is inside {dataset_root}")

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

    session_lease = None
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
    bind_service_paths(dataset_resolution, output_root, cache_root)
    if args.session_name:
        # The named-session exclusive lease lives under the corresponding
        # output namespace (<output>/<session-name>).
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            session_lease = ExclusiveFileLock(
                output_root / ".spiral-service.lock")
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
                f"output {output_root}: {exc}",
                file=sys.stderr, flush=True)
            return 2

    events = ServiceEventBuffer()
    logs = ServiceLogBuffer(events=events)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _TeeStream(original_stdout, logs, "stdout")
    sys.stderr = _TeeStream(original_stderr, logs, "stderr")
    state = ServiceState(dataset_root=str(dataset_root),
                         dataset_resolution=dataset_resolution,
                         service_name=args.service_name,
                         session_name=args.session_name or "",
                         logs=logs,
                         events=events,
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
    print(f"Spiral dataset root: {dataset_root}", flush=True)
    print(f"Spiral output root: {output_root}", flush=True)
    print(f"Spiral cache root: {cache_root}", flush=True)
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
