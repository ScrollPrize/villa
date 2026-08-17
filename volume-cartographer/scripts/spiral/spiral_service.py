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

The session is eager and always loaded: once the dataset and its scroll
specification validate, the service builds its runtime asynchronously and
reports ``Loading`` (then ``Idle``, or ``Error`` with the cause) without any
client request. There is no state in which no session exists, and no verb
that deletes one; ``POST /session/rebuild`` replaces the resident session and
is the only path that may change the model domain or structural
configuration.

Generated display data (previews, downloadable
checkpoints) is published as immutable, opaque artifacts and transferred
through ``/artifacts/...`` instead of host filesystem paths. Session inputs
(patches, fibers, PCL documents) can be uploaded into a session-scoped
ephemeral folder and later committed into the dataset.

Host filesystem paths are the service's business. A client never invents
one: a saved checkpoint is a name the service places under the session
output directory, uploads land in service-chosen staging, and everything
read back is an artifact ID. The one path a client does send — the
checkpoint to load into the resident fit — has to be one this service
advertised or wrote itself.

Long operations accept and return. A preview export costs minutes, so
``POST /session/export-preview`` starts one and answers immediately; the
client follows it through ``/session/status``, which it already polls. The
only verbs that hold a request open are the ones that are genuinely quick.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from collections.abc import Mapping
import copy
import dataclasses
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

from fit_session import (API_VERSION, FIT_INPUT_CATALOG, SESSION_BUSY_STATES,
                         SCROLL_SPEC_FILENAME, SCROLL_SPEC_OWNED_RUN_KEYS,
                         AutosaveError, ScrollSpecError, SessionState,
                         SpiralInputPaths, default_user_cache_dir,
                         load_scroll_spec,
                         parse_session_request, resolve_dataset_root,
                         select_startup_autosave, validate_autosave,
                         validate_session_request)
from config import (CHECKPOINT_MODEL_SHAPE_KEYS, Config, durable_config,
                    rebuild_stage)
from service_http import (ApiError, TRANSFER_CHUNK_BYTES,
                          is_safe_relative_name)
from service_artifacts import ArtifactRegistry
from service_uploads import (EphemeralLedger, PCL_ROLE_FILES,
                             UPLOADED_CHECKPOINTS_DIRNAME,
                             UPLOADED_CHECKPOINTS_KEPT,
                             UPLOAD_GC_SECONDS, UploadEnvironment,
                             UploadManager, _copy_publish,
                             _merge_pcl_documents, _utc_stamp)
from lasagna_publish import (LasagnaPublisher, PreviewPublication,
                             stop_process_group)
# Re-exported for the service's own test surface, which addresses the preview
# mapping helpers through this module.
from lasagna_publish import (_load_flatten_correspondence,  # noqa: F401
                             _mapped_winding_ids,
                             _prepare_cleaned_lasagna_surface,
                             _raw_run_diff_rgba, _sample_rgba_through_map,
                             _validate_tifxyz_output_step)


SERVICE_VERSION = "9.0.0"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256
PREVIEW_ARTIFACTS_KEPT = 3
CHECKPOINT_ARTIFACTS_KEPT = 2
# Upper bound on the checkpoint listing /dataset advertises. A client offers
# this as a choice, so it is a menu, not an inventory.
SESSION_CHECKPOINTS_LISTED = 200
EPHEMERAL_QUOTA_BYTES = int(os.environ.get("SPIRAL_EPHEMERAL_QUOTA_BYTES",
                                           4 * 1024 * 1024 * 1024))
MAX_LOG_ENTRY_CHARS = 8192
# Structured event ring served through /events. This is the whole of what a
# reconnecting client can recover, so it is sized to hold the loading bars
# plus a substantial portion of a long fit.
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


def _cause(exc):
    """One human-readable line for a failure that has no client to raise to."""
    if isinstance(exc, ApiError):
        details = "; ".join(
            f"{detail.get('field')}: {detail.get('message')}"
            for detail in (exc.details or []))
        return f"{exc.message}{f' ({details})' if details else ''}"
    if isinstance(exc, AutosaveError):
        return f"Startup autosave cannot be loaded: {exc}"
    return f"{type(exc).__name__}: {exc}"


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
                           "influence_enabled must be boolean")
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
# periodic step-metric prints. They stay on the terminal, but the event
# stream must not double-report them as log records next to the structured
# progress/metric records.
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
    """Splits the service's stdout and stderr into whole console lines.

    Every complete non-structured line is published to the event buffer as a
    ``log``-kind record; lines already covered by structured
    progress/metric events are kept out of the event stream so the same
    information is never double-reported.

    This used to also retain its own ring for a ``GET /logs`` relay. Nothing
    read it: ``/events`` carries these same lines, with a cursor, a
    session generation and an overrun signal that the log cursor never had,
    and retaining every line twice was the single largest thing this process
    held for the benefit of no client.
    """

    def __init__(self, events=None):
        self._lock = threading.Lock()
        self._pending = {"stdout": "", "stderr": ""}
        self._events = events

    def write(self, stream, text):
        if not text or self._events is None:
            return
        # Carriage-return progress displays should still give remote clients
        # useful snapshots even though they overwrite one terminal line.
        text = str(text).replace("\r", "\n")
        # Splitting and publishing stay under one lock so concurrently
        # written streams cannot interleave their lines in the event ring.
        with self._lock:
            parts = (self._pending.get(stream, "") + text).split("\n")
            self._pending[stream] = parts.pop()
            for line in parts:
                if not line or _STRUCTURED_CONSOLE_LINE.match(line):
                    continue
                if len(line) > MAX_LOG_ENTRY_CHARS:
                    line = line[:MAX_LOG_ENTRY_CHARS] + " … [truncated]"
                self._events.append("log", line, source=stream)


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
    """HTTP-facing state of the service process.

    The session is eager and always loaded: the service starts building its
    runtime as soon as it is up, and there is no state in which no session
    exists. While the runtime is being constructed (or after a construction
    failure) there is no session *object* to ask, so the service reports the
    lifecycle state it is driving itself — ``Loading``, or ``Error`` with the
    cause. Once the object exists the runtime owns the state again: every
    decision reads ``session.status()["state"]`` and the service never keeps
    or advances a copy of it.

    What the service does own is service-scoped bookkeeping — session and
    command generations, artifacts, and uploads.
    """

    def __init__(self, dataset_root=None, dataset_resolution=None,
                 service_name=None, session_name="", logs=None, events=None,
                 gpu_ids=(0,), startup_run=None):
        self.lock = threading.RLock()
        self.session = None
        self.session_id = None
        self.session_paths = None
        self.session_request = None
        self.session_generation = 0
        self.status_generation = 0
        self.commands = OrderedDict()
        self.inflight_commands = set()
        self.command_condition = threading.Condition(self.lock)
        # Lifecycle the service drives while there is no session object to
        # ask: Loading until the runtime is built, Error (with the cause) if
        # building it failed. Never Idle/Running — those belong to the
        # runtime, which is authoritative the moment it exists.
        self._session_state = SessionState.Loading
        self._session_phase = "Starting the fit session"
        self._session_error = None
        self._autosave_selection = None
        self._building = False
        self.startup_run = dict(startup_run or {})
        self.dataset_root = str(dataset_root) if dataset_root else None
        self.dataset_resolution = dataset_resolution
        self.service_name = service_name or socket.gethostname()
        self.session_name = str(session_name or "")
        self.events = events if events is not None else ServiceEventBuffer()
        self.logs = logs if logs is not None else ServiceLogBuffer(self.events)
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
        self.ephemeral_records = EphemeralLedger(self.lock)
        # One record for the whole of preview publication (see
        # LasagnaPublisher's PreviewPublication), guarded by self.lock.
        self._preview = PreviewPublication()
        # A preview export runs off the HTTP thread (it costs minutes); this
        # is what makes the verb single-flight and what /session/status
        # reports so a client reconnecting mid-export can see one is running.
        self._preview_export_active = False
        self.config_catalog = Config.catalog()
        self.session_revision = 0

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    def _base(self):
        """The counters every response carries, and nothing else.

        Three survive, because each answers a question no other one can:

        ``session_generation``
            Which resident session this is. It advances on every rebuild,
            stamps log and fitter event records, and is what a client uses
            to notice that the session it adopted has been replaced.
        ``session_revision``
            Which configuration/input revision the session is at. Mutations
            carrying an older revision are refused.
        ``generation`` (the status revision)
            Ordering for status snapshots, so a client can drop a reply that
            overtook a newer one.

        ``service_generation`` and ``command_generation`` used to be here.
        The first was the constant 1 and identified nothing (process
        identity is ``process_id`` on /health, and clients reset their
        cursors per connection); the second counted replayed commands while
        the replay cache is keyed by (operation, command ID). Nothing read
        either.
        """
        return {
            "api_version": API_VERSION,
            "service_version": SERVICE_VERSION,
            "service_name": self.service_name,
            "session_name": self.session_name,
            "session_id": self.session_id,
            "session_generation": self.session_generation,
            "session_revision": self.session_revision,
            "generation": self.status_generation,
            "gpus": list(self.gpu_ids),
        }

    def _commit_availability(self):
        if self.session is None or self.session_paths is None:
            return False, "No fit session is loaded"
        if not self.ephemeral_records:
            return False, "No ephemeral inputs have been added"
        if not self.ephemeral_records.uncommitted():
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
                # No session object yet (or no longer): the service is
                # building one, or building it failed. Both are real
                # lifecycle states, so there is nothing like "Empty" to
                # report.
                "state": self._session_state, "phase": self._session_phase,
                "current_iteration": 0,
                "target_iteration": 0, "latest_metrics": {}, "warnings": [],
                "error": self._session_error, "preview_manifest_path": None,
                "preview_generation": 0,
                "progress": None,
            })
            response["autosave_selection"] = self._autosave_selection
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
            response["preview_artifact"] = self._preview.artifact
            # Published separately from, and after, the surface: a client that
            # never opens an overlay never waits for one.
            response["preview_diagnostics_artifact"] = (
                self._preview.diagnostics_artifact)
            response["preview_publish"] = (
                dict(self._preview.progress)
                if self._preview.progress else None)
            response["preview_publish_error"] = self._preview.error
            publishing = self._preview.status_progress()
            if publishing is not None:
                response["phase"] = publishing["stage_name"]
                response["progress"] = publishing
            response["ephemeral_inputs"] = self.ephemeral_records.status_entries()
            # Persistence and incorporation are independent: an input can be
            # in the user's dataset while the resident fit has not taken it
            # yet. Name that set explicitly instead of leaving clients to
            # rediscover it from the pair of fields above.
            response["committed_not_incorporated"] = [
                {"id": record.id, "kind": record.kind, "role": record.role}
                for record in self.ephemeral_records.committed_not_incorporated()
            ]
            available, reason = self._commit_availability()
            response["commit_available"] = available
            response["commit_unavailable_reason"] = reason
            response["preview_exporting"] = self._preview_export_active
            return response

    def session_state(self):
        """The authoritative lifecycle state, session object or not."""
        with self.lock:
            if self.session is None:
                return self._session_state
            return self.session.status()["state"]

    def health(self):
        # Answered from service-owned facts only, so it keeps answering while
        # CUDA and the model are being constructed and after a construction
        # failure.
        state = self.session_state()
        response = self._base()
        response.update({
            "ready": True,
            "process_id": os.getpid(),
            "dataset_root": self.dataset_root,
            "session_state": state,
            "cuda_ready": None if state == SessionState.Loading
            else state != SessionState.Error,
        })
        return response

    def configuration_catalog(self):
        return {**self._base(), **self.config_catalog}

    def dataset(self):
        return {**self._base(), **self.dataset_resolution.to_dict(),
                "session_checkpoints": self.session_checkpoints()}

    @property
    def scroll_spec(self):
        """The parsed spiral-scroll.json manifest, or None if it is invalid."""
        if self.dataset_resolution is None:
            return None
        return self.dataset_resolution.scroll_spec

    def session_checkpoints(self):
        """Checkpoints under the session output directory, newest first.

        Between this and ``detected_checkpoints`` a client has the whole set
        of checkpoints it may name: the dataset root holds the ones that came
        with the dataset, and the output directory holds everything this
        service wrote or received (saves, autosaves, uploads). Advertising
        both is what lets a client offer a choice instead of asking the user
        to type a path on a host it may never have seen.
        """
        root = self._output_root()
        if root is None or not root.is_dir():
            return []
        found = []
        for path in root.glob("**/*.ckpt"):
            relative = path.relative_to(root)
            # Artifact staging is transfer plumbing, and the upload store
            # holds digest-named copies a client already has a handle on
            # (``uploaded_checkpoint``). Neither is something a user picks
            # from, which also keeps the two load sources disjoint.
            if any(part.startswith(".") for part in relative.parts) \
                    or relative.parts[0] == UPLOADED_CHECKPOINTS_DIRNAME:
                continue
            if path.is_file():
                found.append((path.stat().st_mtime, str(path)))
        found.sort(key=lambda entry: (-entry[0], entry[1]))
        return [path for _, path in found[:SESSION_CHECKPOINTS_LISTED]]

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

    def _prepare_session_request(self, request):
        """Validate one session request into the arguments a build needs."""
        # The scroll specification in the dataset root owns these. A request
        # that names one is refused rather than quietly overruled by the file.
        scroll_owned = sorted(
            key for key in SCROLL_SPEC_OWNED_RUN_KEYS
            if key in (request.get("run") or {}))
        if scroll_owned:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                f"{SCROLL_SPEC_FILENAME} in the dataset root owns these "
                "values; the request must not carry them",
                [{"field": f"run.{key}",
                  "message": (f"Owned by {SCROLL_SPEC_FILENAME} as "
                              f"{SCROLL_SPEC_OWNED_RUN_KEYS[key]!r}")}
                 for key in scroll_owned])
        if self.dataset_resolution is not None:
            request = self._dataset_session_request(request)
        try:
            paths, run, preview = parse_session_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            # An unparseable request field (an unknown PCL role, say) is the
            # caller's error, not a service fault.
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Malformed session request: {exc}") from exc
        errors = validate_session_request(paths, run)
        # The scroll specification is resolved from the dataset root; it
        # carries the physical scroll facts (including the outward sense,
        # which is not part of the session request).
        scroll = None
        try:
            scroll = load_scroll_spec(paths.dataset_root)
        except ScrollSpecError as exc:
            errors.append({"field": "scroll_spec", "message": str(exc)})
        if errors:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Session validation failed", errors)
        return paths, run, preview, scroll

    def startup_session_request(self, *, resume=True):
        """The request this service builds its own session from.

        With ``resume``, the durable autosave for this service's output
        namespace is selected from metadata (see
        ``fit_session.select_startup_autosave``) and resumed. A selected
        autosave that fails container or identity validation raises
        ``AutosaveError``: the service says so rather than silently starting
        from scratch or from an older state.
        """
        request = {"run": dict(self.startup_run)}
        output_root = self._output_root()
        if not resume or output_root is None or self.dataset_root is None:
            with self.lock:
                self._autosave_selection = None
            return request
        selection = select_startup_autosave(
            output_root, session_namespace=output_root,
            dataset_root=self.dataset_root)
        with self.lock:
            self._autosave_selection = selection.manifest()
        if selection.selected is not None:
            validate_autosave(selection.selected)
            request["paths"] = {"checkpoint": selection.selected.checkpoint}
        return request

    def start_initial_session(self):
        """Build the startup session asynchronously.

        Returns as soon as the work is handed to a thread: the HTTP surface
        is already listening, and /health, /dataset, /configuration, /events
        and status must keep answering while CUDA and the model come up.
        """
        def bootstrap():
            try:
                request = self.startup_session_request()
                prepared = self._prepare_session_request(request)
            except BaseException as exc:
                self._fail_session(None, _cause(exc))
                return
            self._begin_build(*prepared)
        threading.Thread(target=bootstrap, name="spiral-session-bootstrap",
                         daemon=True).start()

    def rebuild(self, request):
        """Rebuild the resident session, from the model stage or from nothing.

        This is the only verb that may replace the model domain or the
        structural configuration: teardown is visible as ``Loading`` instead
        of hidden inside a load. ``{"defaults": true}`` rebuilds from the
        launch defaults and ignores every autosave, which is how a service
        stuck in ``Error`` recovers.

        A request that changes nothing but model configuration keeps the
        loaded host inputs and the brick pools and replaces the model stage
        alone (see ``_rebuild_stage_locked``); everything else is the full
        teardown and reconstruction it has always been.
        """
        request = dict(request or {})
        request.pop("command_id", None)
        defaults = request.pop("defaults", False)
        if not isinstance(defaults, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "defaults must be true or false")
        if defaults:
            if set(request):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "A defaults rebuild takes no other request fields")
            request = self.startup_session_request(resume=False)
        paths, run, preview, scroll = self._prepare_session_request(request)
        if paths.checkpoint and run.config:
            self._reject_overrides_the_checkpoint_contradicts(
                paths.checkpoint, run.config)
        with self.lock:
            # Idle|Error -> Loading. A resident session that is mid-operation
            # has to settle first, and a build already in flight is its own
            # conflict (there is nothing to tear down twice).
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            state = self.session.status()["state"] if self.session else None
            if state in SESSION_BUSY_STATES:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"A rebuild requires an idle or failed session (state is "
                    f"{SessionState(state).name})")
            stage = ("all" if defaults
                     else self._rebuild_stage_locked(paths, run, preview, state))
        if stage == "model":
            self._begin_model_rebuild(paths, run, preview)
        else:
            self._begin_build(paths, run, preview, scroll)
        return {**self.status(), "accepted": True, "rebuilding": True,
                "stage": stage}

    def _rebuild_stage_locked(self, paths, run, preview, state):
        """How much of the resident session this request has to replace.

        Everything outside ``run.config`` is ``all``. Paths name host inputs
        whose contents another process may have changed, so retaining a stage
        across one would need a content fingerprint nothing computes; the
        preview block, the run tag, the z window and the storage backend are
        read before or outside the model. Within ``run.config`` the answer is
        ``config.rebuild_stage`` over the keys whose requested value differs
        from the live session's, which is "model" only for the audited
        allowlist and "all" for everything else.

        Call with the lock held.
        """
        current = self.session_request
        if (self.session is None or current is None
                or state != SessionState.Idle):
            # Nothing to keep: there is no resident session, or it has no
            # model to rebuild around (Error), or it is not quiescent.
            return "all"
        if (paths.manifest() != current.get("paths")
                or preview.manifest() != current.get("preview")):
            return "all"
        live_run = dict(current.get("run") or {})
        new_run = run.manifest()
        live_config = dict(live_run.pop("config", None) or {})
        new_config = dict(new_run.pop("config", None) or {})
        if live_run != new_run:
            return "all"
        changed = {
            key for key in set(live_config) | set(new_config)
            if live_config.get(key) != new_config.get(key)
        }
        return rebuild_stage(changed)

    def _begin_model_rebuild(self, paths, run, preview):
        """Publish the new request and rebuild the model off the HTTP thread.

        The session object, its generation and its whole session scope
        survive: the host inputs the ephemeral uploads were incorporated into
        are retained, so neither the ephemeral ledger nor the uploaded files
        behind it are reset here, and the session reports its own ``Loading``
        while the fitter thread works.
        """
        with self.lock:
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            self._building = True
            self.session_paths = paths
            self.session_request = {
                "paths": paths.manifest(),
                "run": run.manifest(),
                "preview": preview.manifest(),
            }
            self.session_revision += 1
            self.status_generation += 1
            session_id = self.session_id
            session = self.session
        threading.Thread(
            target=self._rebuild_model,
            args=(session_id, session, paths, run),
            name="spiral-model-rebuild", daemon=True).start()

    def _rebuild_model(self, session_id, session, paths, run):
        """Ask the resident session to replace its model stage."""
        try:
            session.rebuild_model(paths, run)
        except BaseException as exc:
            self._fail_session(session_id, _cause(exc))
            return
        with self.lock:
            self._building = False
            self.status_generation += 1

    def _begin_build(self, paths, run, preview, scroll):
        """Publish ``Loading`` and construct the runtime off the HTTP thread."""
        with self.lock:
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            previous = self.session
            previous_ephemeral = self._session_ephemeral_dir()
            self.session = None
            self._building = True
            self.session_generation += 1
            self.session_id = f"spiral-{self.session_generation}-{secrets.token_hex(5)}"
            self.session_paths = paths
            self.session_request = {
                "paths": paths.manifest(),
                "run": run.manifest(),
                "preview": preview.manifest(),
            }
            self.session_revision += 1
            self._reset_session_scope()
            self._session_state = SessionState.Loading
            self._session_phase = "Building the fit session"
            self._session_error = None
            self.status_generation += 1
            session_id = self.session_id
        threading.Thread(
            target=self._build,
            args=(session_id, previous, previous_ephemeral, paths, run,
                  preview, scroll),
            name="spiral-session-build", daemon=True).start()

    def _build(self, session_id, previous, previous_ephemeral, paths, run,
               preview, scroll):
        """Close the old resident session, then construct the new one.

        A rebuild is an all-rank teardown and reconstruction: the previous
        context is released through the session's own ``close()`` (on its
        fitter thread) and the replacement is a fresh session object with a
        fresh fitter thread.
        """
        session = None
        try:
            if previous is not None:
                previous.close()
            if previous_ephemeral:
                shutil.rmtree(previous_ephemeral, ignore_errors=True)
            from spiral_runtime import create_session
            session = create_session(
                paths, run, preview, scroll, self._status_changed,
                gpu_ids=self.gpu_ids, event_callback=self._session_event)
        except BaseException as exc:
            self._fail_session(session_id, _cause(exc))
            return
        superseded = None
        with self.lock:
            self._building = False
            if self.session_id == session_id:
                self.session = session
            else:
                superseded = session
            self.status_generation += 1
        if superseded is not None:
            superseded.close()

    def _fail_session(self, session_id, cause):
        """Report a session that could not be built, and why."""
        with self.lock:
            self._building = False
            if session_id is not None and self.session_id != session_id:
                return
            self._session_state = SessionState.Error
            self._session_phase = "Error"
            self._session_error = cause
            self.status_generation += 1
        print(f"SPIRAL_SESSION_ERROR {cause}", file=sys.stderr, flush=True)
        self.events.append("error", cause, severity="error", source="service",
                           operation="building_session")

    def _reset_session_scope(self):
        self._preview_export_active = False
        self._event_progress_signatures = {}
        self._event_metric_iterations = {}
        self._event_errors = {}
        self.ephemeral_records.clear()
        self.uploads_manager.reset()
        previous_raw = self._preview.reset_session_scope()
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
        if status.get("state") == SessionState.Error and error:
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
            publish_preview = bool(preview_manifest) and self._preview.claim(
                session_id, preview_generation)
        if not publish_preview:
            return

        try:
            publisher, published = self._publish_flattened_preview(
                session_id, preview_generation, Path(preview_manifest))

            def index(kind, root, entry_point, label):
                def indexing_progress(current, total, relative):
                    self._update_preview_publish(
                        preview_generation, state="indexing",
                        stage_name=(
                            f"{label} ({current}/{total}): {relative}"),
                        step=current, total_steps=total,
                        overall_progress=(
                            float(current) / float(total) if total else 1.0))

                started = time.perf_counter()
                self._update_preview_publish(
                    preview_generation, state="indexing", stage_name=label,
                    step=0, total_steps=0, overall_progress=0.0)
                ref = self.artifacts.register_directory(
                    kind, session_id, preview_generation, root, entry_point,
                    delete_root_on_prune=True, progress=indexing_progress,
                    hash_workers=4)
                print(
                    "SPIRAL_PREVIEW_TIMING "
                    f"generation={preview_generation} stage={label!r} "
                    f"seconds={time.perf_counter() - started:.6f}",
                    flush=True)
                return ref

            # The surface is complete and immutable here, so it is indexed and
            # announced now; a client starts transferring it while the
            # overlays below are still being mapped.
            ref = index("spiral-preview", published.manifest_path.parent,
                        published.manifest_path.name,
                        "Indexing preview files")
            with self.lock:
                if self.session_id == session_id:
                    self._preview.artifact = ref
                    self._preview.error = None
                self.status_generation += 1
            self.artifacts.prune(
                "spiral-preview", session_id, PREVIEW_ARTIFACTS_KEPT)

            # The overlays are a second, optional wave. Their failure is
            # reported as a warning, not as a failed preview: the surface is
            # published, announced, and very likely already downloading.
            if bool(status.get("preview_diagnostics")):
                try:
                    diagnostics_manifest = publisher.publish_diagnostics(
                        published)
                    diagnostics_ref = index(
                        "spiral-preview-diagnostics",
                        diagnostics_manifest.parent,
                        diagnostics_manifest.name,
                        "Indexing preview diagnostics")
                    with self.lock:
                        if self.session_id == session_id:
                            self._preview.diagnostics_artifact = diagnostics_ref
                    self.artifacts.prune("spiral-preview-diagnostics",
                                         session_id, PREVIEW_ARTIFACTS_KEPT)
                except Exception as exc:
                    self.events.append(
                        "log",
                        f"Preview loss overlays could not be published: "
                        f"{type(exc).__name__}: {exc}",
                        severity="warning", source="service",
                        operation="publishing_preview")
            published.release()
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
                retained_raw = self._preview.previous_raw_manifest
            if (not retained_raw
                    or failed_raw != Path(retained_raw).parent):
                shutil.rmtree(failed_raw, ignore_errors=True)
            with self.lock:
                if self.session_id == session_id:
                    self._preview.error = error
        finally:
            with self.lock:
                if self.session_id == session_id:
                    self._preview.finish(preview_generation)
                    self.status_generation += 1

    def _update_preview_publish(self, generation, **values):
        with self.lock:
            snapshot = self._preview.record_progress(generation, values)
            if snapshot is None:
                return
            self.status_generation += 1
        self.events.append(
            "progress", str(snapshot.get("stage_name") or ""),
            source="service", operation="publishing_preview",
            payload=snapshot, coalesce_key=("preview-publish",))

    def run(self, request):
        autosave_on_pause = request.get("autosave_on_pause", True)
        if not isinstance(autosave_on_pause, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "autosave_on_pause must be true or false")
        session = self._require_session()
        if session.status().get("state") != SessionState.Idle:
            raise ApiError(HTTPStatus.CONFLICT,
                           "Running requires an idle session")
        expected = request.get("expected_session_revision")
        if expected != self.session_revision:
            raise ApiError(HTTPStatus.CONFLICT, "Session revision is stale")
        configuration = request.get("configuration")
        if not isinstance(configuration, dict) or \
                set(configuration) != set(self.config_catalog["defaults"]):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Running requires a complete configuration")
        try:
            configuration = Config(configuration).as_dict()
            iterations = int(request.get("iterations", 0))
        except (TypeError, ValueError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        if iterations < 1:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "iterations must be at least 1")
        current = session.status().get("applied_config")
        if current is None:
            current = Config(
                (self.session_request.get("run") or {}).get("config") or {}
            ).as_dict()
        changes = {key: value for key, value in configuration.items()
                   if current.get(key) != value}
        fields = self.config_catalog["schema"]["fields"]
        forbidden = {
            key: fields[key]["runtime_impact"] for key in changes
            if fields[key]["runtime_impact"] != "run_boundary"
        }
        if forbidden:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "The requested configuration requires rebuilding the fit",
                [{"field": f"configuration.{key}",
                  "message": f"Runtime impact is {impact}"}
                 for key, impact in sorted(forbidden.items())])
        current_manifest = self.session_paths.manifest()
        input_manifest = request.get("inputs")
        if input_manifest is not None and input_manifest != current_manifest:
            raise ApiError(
                HTTPStatus.CONFLICT,
                "Static dataset inputs cannot be changed by a run")
        influence_config = _validate_run_influence_config(
            request.get("influence") or {})
        run_config = changes
        with self.lock:
            # The fitter (and, under DDP, its child ranks) receives plain
            # records; the ledger maps them back to its own entries when the
            # incorporation outcome arrives.
            pending = [record.payload()
                       for record in self.ephemeral_records.pending()]

            def mark_incorporated(records, error=None):
                with self.lock:
                    self.ephemeral_records.mark_incorporated(
                        self.ephemeral_records.resolve(records), error=error)
                    self.status_generation += 1

        run_arguments = {
                "pending_inputs": pending,
                "mark_incorporated": mark_incorporated,
                "influence_config": influence_config,
                "run_config": run_config,
                # Whether this run's pause writes the durable autosave. It
                # belongs to the run request, not to the plan: it changes
                # nothing about the model, so it needs no planning round.
                "autosave_on_pause": autosave_on_pause,
        }
        target = session.run(iterations, **run_arguments)
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True, "target_iteration": target}

    def stop(self):
        self._require_session().stop()
        with self.lock:
            self.status_generation += 1
        return {**self.status(), "accepted": True}

    def save_checkpoint(self, request):
        """Write a named checkpoint into this session's checkpoint folder.

        The client names the file; the service decides where it lives. A
        checkpoint has only ever been allowed under the session output
        directory, so asking the client for an absolute path on a host it may
        never have seen only ever meant "type the prefix I am about to check
        for". A name says the same thing without the path policing, and it is
        the same name ``/session/status`` reports back as
        ``checkpoint_path``.
        """
        session = self._require_session()
        name = self._checkpoint_file_name(request.get("name"))
        with self.lock:
            root = Path(self.session_paths.output_directory) / "checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        saved = session.save_checkpoint(str(root / name))
        return {**self.status(), "checkpoint_path": saved}

    @staticmethod
    def _checkpoint_file_name(value):
        """One safe file name for a client-named checkpoint."""
        name = str(value or "").strip()
        if not name:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Checkpoint name is required")
        if "/" in name or not is_safe_relative_name(name):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "Checkpoint name must be a single file name",
                [{"field": "name",
                  "message": "Not a valid checkpoint file name"}])
        return name if name.endswith(".ckpt") else f"{name}.ckpt"

    def export_preview(self, request=None):
        """Start one preview generation; do not wait for it.

        Previews are not a side effect of pausing or of resuming from a
        checkpoint any more: they cost minutes, and a client that wants one
        asks for one. Because they cost minutes, this verb accepts the work
        and returns. Holding the request open for the whole export and its
        Lasagna publication meant every real preview outlived the client's
        transfer timeout, and each retry then queued behind the original on
        the command-replay condition and timed out in turn — so a preview
        that in fact succeeded was reported as a failure.

        What the client watches instead is the status it already polls:
        ``preview_exporting`` while this is running, ``preview_publish`` for
        publication progress, then ``preview_artifact`` for the result or
        ``preview_publish_error`` for the cause.

        ``diagnostics`` asks for the loss overlays as well. They roughly
        double the cost of a preview - a second evaluation of every enabled
        loss in the fitter, then a per-overlay remap through the flatten - and
        they arrive as their own artifact after the surface, so a client that
        is not displaying them should not ask for them.
        """
        diagnostics = (request or {}).get("diagnostics", False)
        if not isinstance(diagnostics, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "diagnostics must be true or false")
        session = self._require_session()
        with self.lock:
            if self._preview_export_active:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A preview export is already in progress")
            state = session.status().get("state")
            if state != SessionState.Idle:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"Exporting a preview requires an idle session (state is "
                    f"{SessionState(state).name})")
            self._preview_export_active = True
            self.status_generation += 1
            session_id = self.session_id
        threading.Thread(
            target=self._export_preview,
            args=(session, session_id, diagnostics),
            name="spiral-preview-export", daemon=True).start()
        return {**self.status(), "accepted": True}

    def _export_preview(self, session, session_id, diagnostics=False):
        """Run one export off the HTTP thread and report it through status.

        Publication follows the session's preview status on the fitter
        thread, so this returns only once the whole generation is published;
        nothing but this thread is waiting on it.
        """
        try:
            session.export_preview(diagnostics=diagnostics)
        except BaseException as exc:
            error = _cause(exc)
            print(f"SPIRAL_PREVIEW_ERROR {error}", file=sys.stderr, flush=True)
            self.events.append(
                "error", f"Preview export failed: {error}", severity="error",
                source="service", operation="exporting_preview")
            with self.lock:
                if self.session_id == session_id:
                    self._preview.error = error
        finally:
            with self.lock:
                self._preview_export_active = False
                self.status_generation += 1

    def _resolve_load_source(self, request):
        """The single checkpoint a load request names, resolved by its host.

        A load names exactly one of two things, and both are strings this
        service handed out: ``host_checkpoint`` is one of the checkpoints
        ``/dataset`` advertises, and ``uploaded_checkpoint`` is the path a
        checkpoint upload returned. Neither asks the client to reason about a
        filesystem it may never have seen — that is why they are separate
        fields rather than one free path: they are checked against different
        sets, and the client knows which it has without inspecting the string.
        """
        host = str(request.get("host_checkpoint") or "").strip()
        uploaded = str(request.get("uploaded_checkpoint") or "").strip()
        if bool(host) == bool(uploaded):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "A load names exactly one of host_checkpoint or "
                "uploaded_checkpoint")
        if host:
            advertised = set(
                self.dataset_resolution.to_dict()["detected_checkpoints"])
            advertised.update(self.session_checkpoints())
            if host not in advertised:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "host_checkpoint must be one of the checkpoints /dataset "
                    "advertises",
                    [{"field": "host_checkpoint",
                      "message": "Not a service-advertised checkpoint"}])
            return host
        root = self._output_root()
        store = None if root is None \
            else (root / UPLOADED_CHECKPOINTS_DIRNAME).resolve(strict=False)
        resolved = Path(uploaded).expanduser().resolve(strict=False)
        if store is None or not resolved.is_relative_to(store) \
                or not resolved.is_file():
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "uploaded_checkpoint must be a path a checkpoint upload "
                "returned",
                [{"field": "uploaded_checkpoint",
                  "message": "Not an uploaded checkpoint"}])
        return str(resolved)

    def load_checkpoint(self, request):
        """Load a checkpoint into the resident fit; rebuild only on request.

        One verb, three outcomes. Without ``allow_rebuild`` this is the strict
        in-place load it has always been: the session keeps its model, its
        inputs and its identity, this replaces only
        model/optimiser/scheduler/RNG state, and a checkpoint that does not
        match the live model exactly is refused rather than rebuilt behind the
        client's back. The refusal carries the preflight's own reasons and,
        when a rebuild could accept the checkpoint, the stage that rebuild
        would need; when nothing a rebuild can do would help it says
        ``refused`` instead, and offers nothing.

        With ``allow_rebuild`` the service performs that rebuild itself, from
        the live session request with this checkpoint set and the advanced
        overrides dropped — see ``_rebuild_onto_checkpoint``.

        The preflight therefore runs twice on the escalation path: once to
        refuse, once inside the rebuild. That is a real cost, it is only paid
        on a refusal the client chose to escalate, and it buys a single
        client-side code path.
        """
        request = dict(request or {})
        allow_rebuild = request.pop("allow_rebuild", False)
        if not isinstance(allow_rebuild, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "allow_rebuild must be true or false")
        session = self._require_session()
        path = self._resolve_load_source(request)
        state = session.status().get("state")
        if state != SessionState.Idle:
            raise ApiError(
                HTTPStatus.CONFLICT,
                f"Loading a checkpoint requires an idle session (state is "
                f"{SessionState(state).name})")
        if allow_rebuild:
            return self._rebuild_onto_checkpoint(path)
        try:
            result = session.load_checkpoint(path)
        except ApiError:
            raise
        except BaseException as exc:
            if session.status().get("state") == SessionState.Error:
                # The failure happened while the checkpoint was being applied.
                # The session is gone, not merely unchanged; say so.
                raise ApiError(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    f"Checkpoint load failed after preflight: {exc}") from exc
            raise self._checkpoint_refusal(path, exc) from exc
        with self.lock:
            if self.session_paths is not None:
                self.session_paths = dataclasses.replace(
                    self.session_paths, checkpoint=path)
            if self.session_request:
                paths = dict(self.session_request.get("paths") or {})
                paths["checkpoint"] = path
                self.session_request = {**self.session_request, "paths": paths}
            # Loading replaces the model state represented by this revision.
            self.session_revision += 1
            self.status_generation += 1
        return {**self.status(), "loaded": True, "checkpoint_path": path,
                "restored_iteration": result.get("completed_iterations"),
                "config_revision": result.get("config_revision")}

    def _checkpoint_durable_cfg(self, path):
        """The durable configuration and dataset a checkpoint records.

        CPU-only and read afresh: the same bytes an escalated rebuild would
        apply, so nothing here can go stale between the refusal and the
        rebuild the client may ask for next. A file that will not load at all
        reports no configuration, which every caller treats as "no rebuild
        can help".
        """
        from checkpoint_io import load_checkpoint_cpu
        try:
            payload = load_checkpoint_cpu(path)
        except Exception:
            return None, ""
        try:
            if not isinstance(payload, dict):
                return None, ""
            cfg = payload.get("cfg")
            manifest = payload.get("input_manifest") or {}
            return (dict(cfg) if isinstance(cfg, Mapping) else None,
                    str(manifest.get("dataset_root") or ""))
        finally:
            # A refusal must not leave a whole model + optimiser archive
            # mapped for the lifetime of the service.
            del payload

    def _checkpoint_refusal(self, path, cause):
        """Turn a preflight refusal into the 409 a client can act on.

        The stage comes from the *whole* cfg diff, not from the invariants the
        preflight named. Two reasons. Some model-shaping keys the preflight
        reports (model_flow_bounds_z_margin) are read during host preparation,
        so "only shape keys mismatched" would not imply a model-stage rebuild.
        And a checkpoint's stored cfg overrides host-affecting keys the
        preflight never checks, so a checkpoint differing in, say, a track_*
        setting needs the whole build even though it reported no
        incompatibility there. A model z-domain mismatch reaches "all" through
        z_begin/z_end on this same path rather than through a special case.
        """
        reasons = [line for line in str(cause).splitlines() if line.strip()]
        checkpoint_cfg, checkpoint_dataset = self._checkpoint_durable_cfg(path)
        with self.lock:
            status = self.session.status() if self.session else {}
            dataset_root = str(
                getattr(self.session_paths, "dataset_root", "") or "")
        live = durable_config(status.get("applied_config") or {})
        # What no rebuild can fix: a checkpoint from another dataset, or one
        # whose configuration is not this schema's at all.
        if checkpoint_cfg is None or (
                set(checkpoint_cfg) - set(live)
                or set(live) - set(checkpoint_cfg) - {"z_begin", "z_end"}):
            return ApiError(
                HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}",
                payload={"reasons": reasons, "refused": True})
        if checkpoint_dataset and dataset_root \
                and checkpoint_dataset != dataset_root:
            return ApiError(
                HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}",
                payload={"reasons": reasons, "refused": True})
        changed = {key for key, value in checkpoint_cfg.items()
                   if live.get(key) != value}
        return ApiError(
            HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}",
            payload={"reasons": reasons, "stage": rebuild_stage(changed)})

    def _reject_overrides_the_checkpoint_contradicts(self, path, overrides):
        """Refuse a rebuild whose overrides fight the checkpoint it resumes.

        The runtime applies ``run.config`` on top of the checkpoint's stored
        cfg, so an override of a model-shaping key wins for the session's
        configuration while the model it is resuming is still the
        checkpoint's. The build's own preflight then refuses it from inside a
        session build, where the only possible outcome is a failed session.
        Say so as a request error instead.

        Only the model-shaping keys are restricted. Every other override —
        loss weights, sample counts, schedules — is a legitimate change to
        make while resuming, and the fit is built to take them.
        """
        checkpoint_cfg, _ = self._checkpoint_durable_cfg(path)
        if not checkpoint_cfg:
            return
        conflicts = sorted(
            key for key in CHECKPOINT_MODEL_SHAPE_KEYS
            if key in overrides and key in checkpoint_cfg
            and checkpoint_cfg[key] != overrides[key])
        if conflicts:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "The checkpoint this rebuild resumes disagrees with the "
                "advanced configuration it carries; drop these overrides, or "
                "rebuild without the checkpoint",
                [{"field": f"run.config.{key}",
                  "message": (f"The checkpoint was written with "
                              f"{checkpoint_cfg[key]!r}")}
                 for key in conflicts])

    def _rebuild_onto_checkpoint(self, path):
        """Rebuild the session with this checkpoint as its resume path.

        The escalated request is the live one with the checkpoint set and
        ``run.config`` emptied. Emptying it is not a simplification:
        spiral_runtime applies run.config on top of the checkpoint's stored
        cfg, so resending the advanced profile that just failed the preflight
        would re-impose exactly the mismatching keys, and the rebuild would
        fail the same preflight from inside a session build.
        """
        with self.lock:
            current = copy.deepcopy(self.session_request or {})
        if not current:
            raise ApiError(HTTPStatus.CONFLICT,
                           "There is no session request to rebuild from")
        paths = dict(current.get("paths") or {})
        paths["checkpoint"] = path
        run = dict(current.get("run") or {})
        run["config"] = {}
        return self.rebuild({**current, "paths": paths, "run": run})

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

    def _require_session(self):
        """The resident session, or why there is nothing to operate on.

        The service is always trying to hold a session, so the only reasons
        the object is absent are that it is still being built or that
        building it failed. Both are reported as the lifecycle state the
        client is already polling, with the cause when there is one.
        """
        with self.lock:
            if self.session is None:
                if self._session_state == SessionState.Error:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"The fit session failed to build: "
                        f"{self._session_error}. Rebuild with defaults to "
                        f"recover.")
                raise ApiError(HTTPStatus.CONFLICT,
                               "The fit session is still loading")
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

        Returns ``(publisher, published_surface)``: the surface is finished,
        and the publisher is still holding what a diagnostics wave would need.
        """
        with self.lock:
            output_directory = self.session_paths.output_directory
        # The physical resolution of the preview is the scroll's own, read from
        # the specification the dataset root carries.
        voxel_size_um = (self.scroll_spec or {}).get("voxel_size_um")

        def attach_process(process):
            with self.lock:
                if (self.session_id == session_id
                        and self._preview.owns(generation)):
                    self._preview.process = process

        def detach_process(process):
            with self.lock:
                if self._preview.process is process:
                    self._preview.process = None
                self.status_generation += 1

        def session_valid():
            with self.lock:
                return self.session_id == session_id

        def previous_raw_manifest():
            with self.lock:
                return self._preview.previous_raw_manifest

        def adopt_raw_manifest(path):
            with self.lock:
                old_raw = self._preview.previous_raw_manifest
                self._preview.previous_raw_manifest = path
            return old_raw

        publisher = LasagnaPublisher(
            progress=lambda **values: self._update_preview_publish(
                generation, **values),
            attach_process=attach_process,
            detach_process=detach_process,
            session_valid=session_valid,
            previous_raw_manifest=previous_raw_manifest,
            adopt_raw_manifest=adopt_raw_manifest)
        return publisher, publisher.publish(
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
            if self.ephemeral_records.contains(kind, input_id):
                raise ApiError(HTTPStatus.CONFLICT,
                               f"An ephemeral {kind} named {input_id!r} already exists")
            if self._ephemeral_bytes_in_use() + declared > EPHEMERAL_QUOTA_BYTES:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                               "The ephemeral input quota is exhausted")

    def _ephemeral_bytes_in_use(self):
        return (self.ephemeral_records.bytes_in_use()
                + self.uploads_manager.staged_ephemeral_bytes())

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
                    self.ephemeral_records.add(finalized.record)
                self.status_generation += 1
        return {**self.status(), "input": dict(finalized.record),
                "accepted": True}

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
                records = self.ephemeral_records.uncommitted()
                paths = self.session_paths
            patches_dir = Path(paths.verified_patches) if paths.verified_patches \
                else dataset_root / "verified_patches"
            fibers_dir = Path(paths.fibers) if paths.fibers else dataset_root / "fibers"

            # Validation happens entirely under the dataset lock, before any
            # record is published: collision checks cannot race a cooperating
            # service process, and a record whose staged copy went missing
            # fails the whole commit instead of leaving it half applied.
            for record in records:
                if not Path(record.path).exists():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"The staged copy of {record.kind} {record.id!r} is gone; "
                        "it can no longer be committed")
                if record.kind == "patch" and (patches_dir / record.id).exists():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"A patch named {record.id!r} already exists in the dataset")
                if record.kind == "fiber" and \
                        (fibers_dir / f"{record.id}.json").exists():
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"A fiber named {record.id!r} already exists in the dataset")

            committed = []
            for record in records:
                source = Path(record.path)
                # A still-pending record keeps its staged copy: it remains the
                # incorporation source for the next run, so committing never
                # removes an input from the live session's queue.
                keep_source = not record.incorporated
                if record.kind == "patch":
                    _copy_publish(source, patches_dir / record.id, keep_source)
                elif record.kind == "fiber":
                    _copy_publish(source, fibers_dir / f"{record.id}.json", keep_source)
                else:
                    target = dataset_root / PCL_ROLE_FILES[record.role]
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
                committed.append(record.id)
            with self.lock:
                # Committed records that already joined the resident fit are
                # done; the rest stay queued for the next run.
                self.ephemeral_records.mark_committed(records)
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

    def remove_input(self, kind, input_id):
        with self.lock:
            self._require_session()
            record = self.ephemeral_records.find(kind, input_id)
            if record is None:
                raise ApiError(HTTPStatus.NOT_FOUND,
                               f"No ephemeral {kind or 'input'} named {input_id!r} exists")
            if record.incorporated:
                raise ApiError(HTTPStatus.CONFLICT,
                               "This input already joined the resident fit; removing it "
                               "requires reloading the session")
            self.ephemeral_records.remove(record)
            self.status_generation += 1
        # The staged copy is only deleted when the dataset holds no committed
        # copy; a committed record's file is the user's data now.
        if not record.committed:
            path = Path(record.path)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
        return {**self.status(), "removed": input_id, "accepted": True}

    # ------------------------------------------------------------------
    # Command-ID replay
    # ------------------------------------------------------------------

    def replay_command(self, operation_name, command_id, operation):
        """Run a logical mutation at most once per (operation, command ID).

        The replay cache is namespaced by operation: a client that reuses one
        command ID for two different operations gets both operations, not the
        first one's response twice. A concurrent duplicate waits for the
        in-flight original and receives its response.
        """
        if not isinstance(command_id, str) or not command_id.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "A non-empty command_id is required")
        key = (operation_name, command_id)
        with self.lock:
            while key in self.inflight_commands:
                self.command_condition.wait()
            if key in self.commands:
                cached = self.commands[key]
                self.commands.move_to_end(key)
                return cached
            self.inflight_commands.add(key)
        try:
            response = operation()
            with self.lock:
                self.commands[key] = response
                while len(self.commands) > MAX_DEDUPLICATED_COMMANDS:
                    self.commands.popitem(last=False)
            return response
        finally:
            with self.lock:
                self.inflight_commands.discard(key)
                self.command_condition.notify_all()

    def close(self):
        with self.lock:
            session = self.session
            self.session = None
            process = self._preview.process
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


class Idempotency:
    """How a route survives being retried.

    A single ``needs_dedup`` flag cannot describe this surface: the three
    mutating families are safe for different reasons.

    ``NONE``
        Reads, and allocations whose result is a fresh identifier. Retrying
        is either free (reads) or deliberately produces a new resource.
    ``COMMAND_ID``
        Logical mutations. The client stamps the request with a command ID
        and a repeat of that (operation, command ID) replays the first
        response instead of acting twice.
    ``CONTENT``
        Upload PUTs. There is no command ID: the declared offset (the whole
        file), size and SHA-256 in the upload manifest decide the outcome, so
        any number of retries converges on exactly the declared bytes.
    ``UPLOAD_ID``
        Finalize. Naturally idempotent per upload ID — the first call records
        the published input on the upload and every later call returns it.
    """

    NONE = "none"
    COMMAND_ID = "command_id"
    CONTENT = "content"
    UPLOAD_ID = "upload_id"


class RouteContext:
    """Everything a route handler is allowed to look at."""

    __slots__ = ("handler", "state", "args", "query", "body")

    def __init__(self, handler, state, args, query, body):
        self.handler = handler
        self.state = state
        #: Captured path groups, in pattern order.
        self.args = args
        #: Parsed query string, as returned by ``parse_qs``.
        self.query = query
        #: Decoded JSON request body, or None for methods that do not read one.
        self.body = body


class Route:
    """One method/path pair, its handler, and its retry semantics."""

    __slots__ = ("method", "path", "pattern", "operation", "handler",
                 "idempotency", "reads_body")

    def __init__(self, method, path, operation, handler, idempotency,
                 reads_body=False):
        self.method = method
        #: Literal path, or None when this route matches by pattern.
        self.path = path if not hasattr(path, "fullmatch") else None
        #: Compiled pattern, or None for a literal route.
        self.pattern = path if hasattr(path, "fullmatch") else None
        #: Stable operation name; also the command-ID replay namespace.
        self.operation = operation
        self.handler = handler
        self.idempotency = idempotency
        self.reads_body = reads_body


def _route_events(ctx):
    try:
        cursor = int(ctx.query.get("cursor", ["0"])[-1])
        limit = int(ctx.query.get(
            "limit", [str(MAX_EVENT_READ_ENTRIES)])[-1])
    except (TypeError, ValueError):
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "The event cursor and limit must be integers")
    if cursor < 0 or limit < 1:
        raise ApiError(HTTPStatus.BAD_REQUEST,
                       "The event cursor must not be negative and "
                       "the limit must be at least 1")
    return ctx.state.events.read_after(cursor, limit)


def _route_artifact_file(ctx):
    if not is_safe_relative_name(ctx.args[1]):
        raise ApiError(HTTPStatus.FORBIDDEN, "Unsafe artifact file name")
    ctx.handler._send_artifact_file(ctx.args[0], ctx.args[1])
    return None


def _route_upload_file(ctx):
    handler = ctx.handler
    try:
        length = int(handler.headers.get("Content-Length", "-1"))
    except ValueError:
        raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
    if length < 0:
        raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
    return ctx.state.receive_upload_file(
        ctx.args[0], ctx.args[1], handler.rfile, length)


_UPLOAD_ID = r"[0-9a-f]{32}"

# The whole HTTP surface, declared once. Dispatch walks this table; there is
# no hand-written if-ladder, so a route's method, path, handler and retry
# semantics are visible in one place.
ROUTES = (
    Route("GET", "/health", "health",
          lambda ctx: ctx.state.health(), Idempotency.NONE),
    Route("GET", "/configuration", "configuration",
          lambda ctx: ctx.state.configuration_catalog(), Idempotency.NONE),
    Route("GET", "/session/status", "session_status",
          lambda ctx: ctx.state.status(), Idempotency.NONE),
    Route("GET", "/events", "events", _route_events, Idempotency.NONE),
    Route("GET", "/dataset", "dataset",
          lambda ctx: ctx.state.dataset(), Idempotency.NONE),
    Route("GET", re.compile(r"/artifacts/([A-Za-z0-9._-]+)/manifest"),
          "artifact_manifest",
          lambda ctx: ctx.state.artifacts.manifest(ctx.args[0]),
          Idempotency.NONE),
    Route("GET", re.compile(r"/artifacts/([A-Za-z0-9._-]+)/files/(.+)"),
          "artifact_file", _route_artifact_file, Idempotency.NONE),

    Route("PUT", re.compile(rf"/session/inputs/({_UPLOAD_ID})/files/(.+)"),
          "upload_file", _route_upload_file, Idempotency.CONTENT),

    # There is deliberately no DELETE /session: the service always holds a
    # session, and replacing one is POST /session/rebuild.
    #
    # A removal names its target in the path, so it needs no body: the
    # operation is already idempotent (a second DELETE finds nothing to
    # remove), and clients do not retry it.
    Route("DELETE",
          re.compile(r"/session/ephemeral-inputs/([a-z]+)/([A-Za-z0-9._-]+)"),
          "ephemeral_input_remove",
          lambda ctx: ctx.state.remove_input(ctx.args[0], ctx.args[1]),
          Idempotency.NONE),

    Route("POST", re.compile(rf"/session/inputs/({_UPLOAD_ID})/finalize"),
          "upload_finalize",
          lambda ctx: ctx.state.finalize_upload(ctx.args[0]),
          Idempotency.UPLOAD_ID, reads_body=True),
    Route("POST", "/session/inputs", "upload_begin",
          lambda ctx: ctx.state.begin_upload(ctx.body), Idempotency.NONE,
          reads_body=True),
    Route("POST", "/session/rebuild", "session_rebuild",
          lambda ctx: ctx.state.rebuild(ctx.body), Idempotency.COMMAND_ID,
          reads_body=True),
    Route("POST", "/session/run", "session_run",
          lambda ctx: ctx.state.run(ctx.body), Idempotency.COMMAND_ID,
          reads_body=True),
    Route("POST", "/session/stop", "session_stop",
          lambda ctx: ctx.state.stop(), Idempotency.COMMAND_ID,
          reads_body=True),
    Route("POST", "/session/save-checkpoint", "save_checkpoint",
          lambda ctx: ctx.state.save_checkpoint(ctx.body),
          Idempotency.COMMAND_ID, reads_body=True),
    Route("POST", "/session/export-preview", "export_preview",
          lambda ctx: ctx.state.export_preview(ctx.body),
          Idempotency.COMMAND_ID, reads_body=True),
    Route("POST", "/session/load-checkpoint", "load_checkpoint",
          lambda ctx: ctx.state.load_checkpoint(ctx.body),
          Idempotency.COMMAND_ID, reads_body=True),
    Route("POST", "/session/download-checkpoint", "download_checkpoint",
          lambda ctx: ctx.state.download_checkpoint(),
          Idempotency.COMMAND_ID, reads_body=True),
    Route("POST", "/session/commit-inputs", "commit_inputs",
          lambda ctx: ctx.state.commit_inputs(), Idempotency.COMMAND_ID,
          reads_body=True),
)

# Methods whose body is read before the route is resolved, so a malformed or
# oversized body is reported as such even on an unknown path.
_BODY_BEFORE_MATCH = frozenset({"POST"})

_LITERAL_ROUTES = {(route.method, route.path): route for route in ROUTES
                   if route.path is not None}
_PATTERN_ROUTES = tuple(route for route in ROUTES if route.pattern is not None)


def resolve_route(method, path):
    """Return ``(route, captured groups)`` or ``(None, ())``."""
    route = _LITERAL_ROUTES.get((method, path))
    if route is not None:
        return route, ()
    for route in _PATTERN_ROUTES:
        if route.method != method:
            continue
        match = route.pattern.fullmatch(path)
        if match:
            return route, match.groups()
    return None, ()


class SpiralHandler(BaseHTTPRequestHandler):
    server_version = "VC3D-Spiral/2"
    # HTTP/1.1 keeps connections alive so multi-file artifact transfers and
    # uploads do not pay a fresh TCP (or tunnel) setup per file.
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("SPIRAL_HTTP " + (fmt % args), file=sys.stderr, flush=True)

    def log_request(self, code="-", size="-"):
        """Suppress successful polling requests at the source.

        Status and event reads arrive several times a second from every
        connected client; logging them would drown the terminal and the
        event ring in access lines. Failed polls still log.
        """
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if self.command == "GET" and 200 <= status < 400:
            path = urlparse(self.path).path.rstrip("/")
            if path in ("/session/status", "/events"):
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
        """Authorise, resolve one route, and apply its retry semantics."""
        self._authorise()
        parsed_url = urlparse(self.path)
        path = unquote(parsed_url.path).rstrip("/") or "/"
        if "\\" in path or "\x00" in path or "/../" in path + "/":
            raise ApiError(HTTPStatus.FORBIDDEN, "Malformed request path")
        state = self.server.state

        body = self._body() if self.command in _BODY_BEFORE_MATCH else None
        route, args = resolve_route(self.command, path)
        if route is None:
            raise ApiError(HTTPStatus.NOT_FOUND, "Unknown endpoint")
        if body is None and route.reads_body:
            body = self._body()
        context = RouteContext(self, state, args,
                               parse_qs(parsed_url.query), body)
        if route.idempotency == Idempotency.COMMAND_ID:
            return state.replay_command(
                route.operation, (body or {}).get("command_id"),
                lambda: route.handler(context))
        # CONTENT and UPLOAD_ID routes carry their own retry semantics
        # (declared digest, published upload record); NONE routes have none.
        return route.handler(context)

    def _handle(self):
        try:
            response = self._dispatch()
            if response is not None:
                self._send(HTTPStatus.OK, response)
        except ApiError as exc:
            payload = self.server.state._base()
            payload.update({"error": exc.message, "details": exc.details,
                            **exc.payload})
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
    # The session is eager, so the z-domain it is built with has to be
    # expressible at launch. Both are startup defaults only: changing them
    # afterwards is a rebuild, which is the one verb allowed to replace the
    # model domain.
    parser.add_argument("--z-begin", type=int, default=Config().z_begin,
                        help="First z slice of the startup session "
                             f"(default: {Config().z_begin})")
    parser.add_argument("--z-end", type=int, default=Config().z_end,
                        help="Last z slice (exclusive) of the startup session "
                             f"(default: {Config().z_end})")
    parser.add_argument("--config", default=None, metavar="JSON",
                        help="Advanced configuration overrides for the startup "
                             "session, as a JSON object. These are the "
                             "'defaults' a rebuild-with-defaults returns to.")
    parser.add_argument(
        "--gpus", type=parse_gpu_ids, default=(0,), metavar="DEVICE[,DEVICE...]",
        help="Physical CUDA device indices to use (default: 0; example: 0,1,2,3)")
    args = parser.parse_args(argv)

    # fit_spiral and Torch are imported lazily when a session is loaded. Narrow
    # visibility now so even the single-process path consistently uses the
    # operator-selected physical device as its local cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.gpus)

    if args.z_begin >= args.z_end:
        parser.error("--z-begin must be less than --z-end")
    startup_config = {}
    if args.config:
        try:
            startup_config = json.loads(args.config)
        except json.JSONDecodeError as exc:
            parser.error(f"--config must be a JSON object: {exc}")
        if not isinstance(startup_config, dict):
            parser.error("--config must be a JSON object")
        try:
            Config(startup_config)
        except (ValueError, AttributeError, TypeError) as exc:
            parser.error(f"--config is not a valid configuration: {exc}")

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
                         gpu_ids=args.gpus,
                         startup_run={"z_begin": args.z_begin,
                                      "z_end": args.z_end,
                                      "config": startup_config})
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
    print(f"Spiral z-range: [{args.z_begin}, {args.z_end})", flush=True)
    print(f"SPIRAL_SERVICE_READY port={server.server_port}", flush=True)
    # The session is eager: startup dataset and spec validation has passed,
    # so the runtime is built now, asynchronously, and the service reports
    # Loading while CUDA and the model come up. No client request creates a
    # session.
    state.start_initial_session()
    server.timeout = 0.5
    try:
        while not shutdown.is_set():
            server.handle_request()
    finally:
        server.server_close()
        try:
            state.close()
        finally:
            if session_lease is not None:
                session_lease.release()
            sys.stdout, sys.stderr = original_stdout, original_stderr
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
