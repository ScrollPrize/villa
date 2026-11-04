#!/usr/bin/env python3
"""HTTP service for a persistent interactive Spiral fit.

The service binds to loopback by default. Non-loopback binds are explicit and
always carry bearer authentication; every client — including VC3D talking to a
process it launched itself — uses the same authenticated HTTP protocol.

Every service binds its host paths once at startup and advertises them
through ``/bindings``: ``--output`` (all generated state) is required;
``--cache`` defaults to the documented user cache
(``$XDG_CACHE_HOME/vc3d/spiral``); the optional ``--winding-inference`` is
the winding-inference artifact winding-model supervision reads. There is no
dataset: the scroll's facts, umbilicus, Lasagna dataset and fiber-inference
dataset are the project's that each session request carries.

The service starts ``Uninitialized``. Binding and checkpoint discovery remain
available without importing the fitting runtime; ``POST /session/initialize``
creates the first resident session, and ``POST /session/rebuild`` replaces an
existing one.

Patch, PCL and fiber inputs come from the object store bound at launch
(``--object-store``), never from an upload. A fit reads the inputs of a
linked, pinned position of it, and ``POST /session/update-inputs`` replaces
a resident fit's inputs with a newer position's in place, keeping its model
and optimiser. With ``--object-store-hub`` the store is the service's
replica of a hub and positions are the hub's; otherwise the store is shared
with the client's processes on this machine and records positions of its
own. Without a position (a rebuild of an old checkpoint's inputs) the
inputs are frozen for a resident session, and
changing them is an edit in the store followed by a full-input rebuild. One
client owns the mutation surface through the session lease
(``POST /session/lease/claim``).

Generated display data (previews, downloadable
checkpoints) is published as immutable, opaque artifacts and transferred
through ``/artifacts/...`` instead of host filesystem paths.

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
from contextlib import contextmanager
from service_mutations import MutationCoordinator
from session_lease import SessionLease
from workspace_storage import reclaim_workspaces
from collections import OrderedDict, deque
from collections.abc import Mapping
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import logging
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
from types import SimpleNamespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

from fit_session import (API_VERSION, FIT_INPUT_CATALOG,
                         SESSION_BUSY_STATES,
                         SCROLL_SPEC_OWNED_RUN_KEYS,
                         PROJECT_LASAGNA_INPUTS, VOLUME_INPUT_KEYS,
                         ScrollSpecError, SessionScroll, SessionState,
                         SpiralProject, default_user_cache_dir,
                         fiber_directions_required, input_source_enabled,
                         lasagna_enabled,
                         winding_inference_enabled, winding_inference_error,
                         parse_session_request,
                         validate_session_request, write_project_umbilicus)
from lasagna_source import load_lasagna_source, write_lasagna_source
from config import (CHECKPOINT_MODEL_SHAPE_KEYS, Config, describe_catalog,
                    filter_known_config_keys, rebuild_stage)
from service_http import (ApiError, TRANSFER_CHUNK_BYTES,
                          is_safe_relative_name)
from service_artifacts import ArtifactRegistry
from preview_index import PublishedPreviewIndex
from service_files import ExclusiveFileLock, FileLockUnavailable
from service_uploads import (UPLOADED_CHECKPOINTS_DIRNAME,
                             UploadEnvironment, UploadManager)
from lasagna_publish import (LasagnaPublisher, PreviewPublication,
                             stop_process_group)
# Re-exported for the service's own test surface, which addresses the preview
# mapping helpers through this module.
from lasagna_publish import (_load_flatten_correspondence,  # noqa: F401
                             _mapped_winding_ids,
                             _prepare_cleaned_lasagna_surface,
                             _raw_run_diff_rgba, _sample_rgba_through_map,
                             _validate_tifxyz_output_step)


SERVICE_VERSION = "10.0.0"
# The current session's object-store input manifest, as one immutable
# artifact per input digest: too large for every status poll, and what a
# client resolves the fitted inputs' revision pins from.
INPUT_MANIFEST_ARTIFACT_KIND = "spiral-input-manifest"
INPUT_MANIFEST_FILE_NAME = "object_store_inputs.json"
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_DEDUPLICATED_COMMANDS = 256
PREVIEW_ARTIFACTS_KEPT = 3
CHECKPOINT_ARTIFACTS_KEPT = 2
# Parallel chunk reads when extracting missing fiber directions.
FIBER_DIRECTION_WORKERS = 64
# Parallel object fetches when copying remote Lasagna arrays to pack them.
FETCH_WORKERS = 64
# Upper bound on the checkpoint listing /bindings advertises. A client offers
# this as a choice, so it is a menu, not an inventory.
SESSION_CHECKPOINTS_LISTED = 200
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

_SAFE_SESSION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# The stable code of a refused input upload: patches, PCLs and fibers come
# from the object store, so an upload is never how inputs change.
INPUTS_IN_OBJECT_STORE = "inputs_in_object_store"
# Stable codes of refused input updates (POST /session/update-inputs), and
# of verbs refused while one is in flight.
INPUT_UPDATE_UNAVAILABLE = "input_update_unavailable"  # no hub, or no session
INPUT_UPDATE_BUSY = "input_update_busy"                # not an idle session
INPUT_GENERATION_STALE = "input_generation_stale"      # expected_input_generation
INPUT_UPDATE_IN_PROGRESS = "input_update_in_progress"  # another verb waits for one
LEASE_TOKEN_HEADER = "X-Spiral-Session-Token"


def input_upload_error():
    return ApiError(
        HTTPStatus.BAD_REQUEST,
        "Fit inputs are not uploaded: patches, PCLs and fibers come from the "
        "service's object store. Change them there, then update the fit's "
        "inputs (POST /session/update-inputs) or rebuild it with inputs "
        "reloaded.",
        payload={"code": INPUTS_IN_OBJECT_STORE})

# Input paths are the service's (its bindings) and the project's; a session
# request may name only these, each among service-advertised values.
_CLIENT_SELECTABLE_PATHS = ("checkpoint",)
# Request paths of the dataset service, refused with what replaced them.
_RETIRED_REQUEST_PATHS = {
    "dataset_root": "the service has no dataset; the project block carries "
                    "the scroll's inputs",
    "winding_inference": "winding inference is the service's --winding-inference "
                         "binding; restart the service to change it",
}


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
    return f"{type(exc).__name__}: {exc}"


def parse_session_name(value):
    """Validate a host-owned name which is also used as one path component."""
    name = str(value).strip()
    if not _SAFE_SESSION_NAME.fullmatch(name) or name in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "--session-name must be 1-64 characters, start with a letter or "
            "digit, and contain only letters, digits, '.', '_', or '-'")
    return name


class ObjectStoreBindingError(RuntimeError):
    """The launch object store is unusable, and why."""


@dataclasses.dataclass(frozen=True)
class ObjectStoreBinding:
    """The object store this service reads fit inputs from.

    Bound once at launch (``--object-store``) and never chosen by a client:
    patch, PCL and fiber inputs, and the links the service's refresh writes,
    live in this one store. ``path`` is meaningful on the service host only;
    clients identify the store by ``uuid``, which every replica of it shares.
    ``author`` names the service in the store when it refreshes links.

    A hosted binding (``hub`` set) is the service's own replica of a hub:
    it writes only detector links, pushes them only with a conditional
    push, and fits hub positions (object_store_inputs.prepare_position).
    ``hub_identity`` names the hub positions belong to. Otherwise the store
    is shared in place (VC3D on this machine edits the same file) and
    records positions of its own, linked in place. ``positions`` names
    whose positions the service fits: the hub's, or the store's own.
    """

    path: str
    uuid: str
    author: str
    hub: str = None
    hub_identity: str = None
    positions: str = None

    @property
    def hosted(self):
        return self.hub is not None

    @property
    def positioned(self):
        return self.positions is not None

    def config(self):
        """The service-owned configuration a session request receives."""
        return {"object_store_path": self.path,
                "object_store_author": self.author,
                "object_store_input_position": None}

    def hub_position(self):
        """``(position, error)``: the hub position this replica last pulled,
        read now (not the fitted one, which is status ``input_position``),
        or why it could not be read."""
        from vc_object_store import ObjectStore
        try:
            with ObjectStore.open(self.path, read_only=True) as store:
                return (store.hub_status() or {}).get("position"), None
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            return None, f"{type(exc).__name__}: {exc}"

    def store_position(self):
        """``(position, error)``: the head of a shared store, read now."""
        from vc_object_store import ObjectStore
        try:
            with ObjectStore.open(self.path, read_only=True) as store:
                return (store.store_positions() or {}).get("position"), None
        except Exception as exc:  # noqa: BLE001 - reported, not fatal
            return None, f"{type(exc).__name__}: {exc}"

    def advertisement(self):
        result = {"uuid": self.uuid, "path": self.path, "author": self.author,
                  "hub": None, "positions": None}
        if self.hosted:
            position, error = self.hub_position()
            result["hub"] = {"address": self.hub, "identity": self.hub_identity,
                             "position": position, "position_error": error}
        elif self.positioned:
            position, error = self.store_position()
            result["positions"] = {"identity": self.positions, "position": position,
                                   "position_error": error}
        return result


# Configuration keys the launch binding owns. A session request or --config
# naming one is refused, as the scroll-specification keys are, rather than
# quietly overruled: a client cannot repoint the inputs a service reads, or
# choose the position a build pins (it asks with ``inputs_position``).
SERVICE_OWNED_CONFIG_KEYS = ("object_store_path", "object_store_author",
                             "object_store_input_position")


# What /bindings tells a client this service supports, so it can enable or
# explain controls before any session exists.
SERVICE_CAPABILITIES = {
    # Patches, PCLs and fibers are read from the bound object store.
    "input_source": "object_store",
    # A resident session's inputs never change (a service without
    # positions); one with positions updates them in place instead.
    "inputs_frozen": True,
    # POST /session/rebuild {"inputs": "reload"} rereads the store.
    "input_reload_rebuild": True,
    # Initialize and full-input rebuilds link and pin a position, including
    # at least the request's ``inputs_position`` (bindings with positions).
    "input_positions": False,
    # POST /session/update-inputs replaces a resident fit's inputs with a
    # position's, keeping its model and optimiser (bindings with positions).
    "input_updates": False,
}


def _stat_or_none(path):
    try:
        return path.stat()
    except OSError:
        return None


def default_object_store_author():
    return f"spiral-service@{socket.gethostname()}"


def _canonical_hub(vos, address):
    """A hub address as the store records it (vos.hub_address: a remote URL
    in its canonical sftp:// form, a path made absolute), local paths
    resolved first so a symlinked directory names one hub."""
    if "://" not in str(address):
        address = Path(address).expanduser().resolve(strict=False)
    return vos.hub_address(str(address))


_REMOTE_HUB_HINT = (
    "a remote hub is reached with ssh in batch mode: the service account needs "
    "a key (or agent) with access to the hub file and its directory, and a "
    "known_hosts entry for the host")


def bind_object_store(path, author=None, hub=None):
    """Resolve and open the launch object store once; return its binding.

    With ``hub`` the store is this service's replica of that hub: cloned
    when absent, otherwise checked to replicate that hub. A store that is
    already a replica binds its own hub. A hosted binding pulls once now, so
    an unreachable hub (or, for an sftp:// hub, missing ssh credentials or
    host key) fails startup instead of the first update.

    Raises ObjectStoreBindingError when the vc_object_store module is not
    importable or the store cannot be opened, so startup fails with the
    cause instead of the first fit failing later.
    """
    resolved = Path(path).expanduser().resolve(strict=False)
    author = author or default_object_store_author()
    try:
        import vc_object_store as vos
        from vc_object_store import ObjectStore
    except ImportError as exc:
        raise ObjectStoreBindingError(
            f"the vc_object_store Python module is unavailable ({exc}); put "
            "its build's python directory on PYTHONPATH") from exc
    remote = hub is not None and "://" in str(hub)
    if hub is not None and not resolved.exists():
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            vos.hub_clone(_canonical_hub(vos, hub), str(resolved), author)
        except Exception as exc:
            raise ObjectStoreBindingError(
                f"cannot clone the hub {hub} to {resolved}: {exc}"
                + (f"; {_REMOTE_HUB_HINT}" if remote else "")) from exc
    if not resolved.is_file():
        raise ObjectStoreBindingError(f"no object store at {resolved}")
    try:
        with ObjectStore.open(str(resolved), read_only=True) as store:
            uuid = store.uuid
            status = store.hub_status()
    except Exception as exc:
        raise ObjectStoreBindingError(
            f"cannot open the object store {resolved}: {exc}") from exc
    if status is None:
        if hub is not None:
            raise ObjectStoreBindingError(
                f"{resolved} exists and is not a replica of the hub {hub}; "
                "name a new replica path, or omit --object-store-hub")
        # Shared in place: every write, VC3D's included, records a position
        # from now on, so the service links and pins positions of the store
        # itself (no hub, no replicas).
        try:
            with ObjectStore.open(str(resolved), author=author) as store:
                store.enable_store_positions()
                positions = store.store_positions()["store"]
        except Exception as exc:
            raise ObjectStoreBindingError(
                f"cannot record positions of the object store {resolved}: {exc}") from exc
        return ObjectStoreBinding(str(resolved), str(uuid), author, positions=positions)
    if hub is not None and _canonical_hub(vos, hub) != status["hub"]:
        raise ObjectStoreBindingError(
            f"{resolved} replicates the hub {status['hub']}, not {hub}")
    remote = "://" in status["hub"]
    # Checked before anything is changed: a replica someone else writes
    # (a VC3D user's, named by mistake) holds their unpushed edits.
    try:
        with ObjectStore.open(str(resolved), read_only=True) as store:
            others = [writer for writer in store.hub_writers() if writer != author]
    except Exception as exc:
        raise ObjectStoreBindingError(
            f"cannot open the object store {resolved}: {exc}") from exc
    if others:
        raise ObjectStoreBindingError(
            f"{resolved} is not this service's replica: {', '.join(others)} "
            f"wrote to it, not only {author}; name a new replica path (the "
            "service clones one), or start the service as its author")
    try:
        with ObjectStore.open(str(resolved), author=author) as store:
            # The service replica writes only links, and every update links
            # afresh: unpublished ones (an interrupted update's) are dropped.
            store.hub_revert()
            pulled = store.hub_sync(pull=True, push=False)
            conflicts = pulled["conflicts"]
            pending = store.hub_status()["pending"]
    except Exception as exc:
        raise ObjectStoreBindingError(
            f"cannot reach the hub {status['hub']} of {resolved}: {exc}"
            + (f"; {_REMOTE_HUB_HINT}" if remote else "")) from exc
    if conflicts or pending:
        raise ObjectStoreBindingError(
            f"{resolved} holds unpublished edits the service did not make "
            f"({len(pending)} pending, {len(conflicts)} in conflict with the hub); "
            "a service replica must hold none: resolve them, or name a new replica path")
    return ObjectStoreBinding(str(resolved), str(uuid), author,
                              status["hub"], status["hub_replica"], status["hub_replica"])


class ServiceBindingError(ValueError):
    """A launch path binding is unusable, and why."""


@dataclasses.dataclass(frozen=True)
class ServiceBindings:
    """The host paths a service binds once at launch; no client can repoint
    them. ``output_directory`` holds all generated state, ``cache_directory``
    the derived host caches (prepared Lasagna packs and fiber directions),
    and ``winding_inference`` ('' when unbound) the winding-inference
    artifact winding-model supervision reads directly: a supervision input,
    not a cache and not a project asset. /bindings advertises them."""

    output_directory: str
    cache_directory: str
    winding_inference: str = ""

    def winding_inference_status(self):
        """``{bound, path, available, error}``: whether the bound artifact is
        usable now (it is read when a fit needs it, so it is checked again)."""
        if not self.winding_inference:
            return {"bound": False, "path": None, "available": False,
                    "error": "the service was launched without --winding-inference"}
        error = winding_inference_error(self.winding_inference)
        return {"bound": True, "path": self.winding_inference,
                "available": not error, "error": error or None}

    def advertisement(self):
        return {"output_directory": self.output_directory,
                "cache_directory": self.cache_directory,
                "winding_inference": self.winding_inference_status()}

    def provenance(self, project):
        """Where a session's inputs come from: the client's project and
        these bindings."""
        result = {"project": project.name,
                  "output_directory": self.output_directory,
                  "cache_directory": self.cache_directory}
        if self.winding_inference:
            result["winding_inference"] = self.winding_inference
        return result


def bind_service_inputs(output_directory, cache_directory, winding_inference=None):
    """Resolve the launch path bindings once. A named winding-inference
    artifact must exist; one that is unusable for another reason is
    advertised as unavailable (``ServiceBindings.winding_inference_status``)
    and fails only the fits that need it."""
    winding = ""
    if winding_inference:
        winding = str(Path(winding_inference).expanduser().resolve(strict=False))
        if not Path(winding).is_dir():
            raise ServiceBindingError(
                f"--winding-inference {winding} is not a directory: name the "
                "winding-inference artifact (the directory holding its manifest.json)")
    return ServiceBindings(str(output_directory), str(cache_directory), winding)


def _validate_dt_loss_schedule(value):
    """Validate the required, transient DT schedule on a Run request."""
    if not isinstance(value, dict):
        raise ApiError(
            HTTPStatus.BAD_REQUEST, "dt_loss_schedule must be a JSON object")
    expected = {"enabled", "last_fraction"}
    if set(value) != expected:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "dt_loss_schedule must contain exactly enabled and last_fraction")
    enabled = value["enabled"]
    if not isinstance(enabled, bool):
        raise ApiError(
            HTTPStatus.BAD_REQUEST, "dt_loss_schedule.enabled must be boolean")
    fraction = value["last_fraction"]
    if (isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))):
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "dt_loss_schedule.last_fraction must be numeric")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "dt_loss_schedule.last_fraction must be finite and between 0 and 1")
    return {"enabled": enabled, "last_fraction": fraction}


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

    The service starts uninitialized. Dataset discovery and the HTTP control
    surface are available immediately, but constructing the fit runtime is an
    explicit client action. While a runtime is being constructed (or after a
    construction failure) there is no session *object* to ask, so the service
    reports the lifecycle state it is driving itself.

    What the service does own is service-scoped bookkeeping — session and
    command generations, artifacts, and uploads.
    """

    def __init__(self, bindings=None,
                 service_name=None, session_name="", logs=None, events=None,
                 gpu_ids=(0,), startup_run=None, object_store=None,
                 build_missing_lasagna_packs=False):
        self.build_missing_lasagna_packs = bool(build_missing_lasagna_packs)
        self.lock = threading.RLock()
        self._service_condition = threading.Condition(self.lock)
        self._closing = False
        self._service_closed = False
        self._service_users = 0
        self._background_jobs = set()
        self._retiring_sessions = []
        self._teardown_lock = threading.Lock()
        self._release_receipts = MutationCoordinator()
        self.session = None
        # The SessionLease of the client that owns this service, if any.
        self.session_lease = None
        self.session_id = None
        self.session_paths = None
        self.session_request = None
        self.session_generation = 0
        self.status_generation = 0
        self.commands = OrderedDict()
        self.inflight_commands = set()
        self.command_condition = threading.Condition(self.lock)
        # Lifecycle the service drives while there is no session object.
        # Idle/Running belong to the runtime once one has been initialized.
        self._session_state = SessionState.Uninitialized
        self._session_phase = "Waiting for fit initialization"
        self._session_error = None
        self._building = False
        self._build_serial = 0
        # Hub positions (hosted bindings): the one being linked and pinned
        # before a build, the last preparation that failed (the resident fit
        # is then unchanged), and the position the session reads.
        self._input_preparation = None
        self._input_preparation_error = None
        self._input_position = None
        self._session_scroll = None
        # The project the session was built from (fit_session.SpiralProject):
        # what a rebuild the service starts itself sends again.
        self._session_project = None
        # An input update of the resident session in flight (its build ID).
        self._input_update = None
        self.startup_run = dict(startup_run or {})
        # ServiceBindings of the launch; None only in tests that never build
        # a fit.
        self.bindings = bindings
        # ObjectStoreBinding of the launch --object-store; None only in
        # tests that never build a fit.
        self.object_store = object_store
        self._startup_inputs = None
        self._startup_inputs_lock = threading.Lock()
        # The store key a background read of startup_inputs is reading, and
        # the lock that lets one read run at a time.
        self._startup_inputs_reading = None
        self._startup_inputs_read_lock = threading.Lock()
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
        self._event_input_lines = {}
        self.gpu_ids = tuple(gpu_ids)
        self.artifacts = ArtifactRegistry()
        self.checkpoint_uploads = UploadManager(self._upload_environment())
        # One record for the whole of preview publication (see
        # LasagnaPublisher's PreviewPublication), guarded by self.lock.
        self._preview = PreviewPublication()
        # The published input manifest of this session: (digest, artifact).
        self._input_manifest = None
        # A preview export runs off the HTTP thread (it costs minutes); this
        # is what makes the verb single-flight and what /session/status
        # reports so a client reconnecting mid-export can see one is running.
        self._preview_export_active = False
        self._preview_schedule = None
        self._next_preview_iteration = None
        self._automatic_previews_disabled = False
        self._automatic_preview_generations = set()
        # The checkpoint the fitter last reported its resident model equal
        # to (see InteractiveFitSession.status()["checkpoint_state"]); a
        # change here pins that checkpoint's published preview and re-shows
        # it when one exists.
        self._checkpoint_state = None
        self.config_catalog = Config.catalog()
        self.session_revision = 0
        if self._output_root() is not None:
            reclaim_workspaces(self._output_root())

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    @contextmanager
    def service_use(self):
        """Count one request or job that teardown must wait for."""
        with self._service_condition:
            if self._closing or self._service_closed:
                raise ApiError(410, "The session lease is being released")
            self._service_users += 1
        try:
            yield
        finally:
            with self._service_condition:
                self._service_users -= 1
                self._service_condition.notify_all()

    def _start_background(self, *, target, args=(), name, daemon=True):
        def run():
            try:
                target(*args)
            finally:
                with self._service_condition:
                    self._background_jobs.discard(threading.current_thread())
                    self._service_condition.notify_all()
        with self._service_condition:
            if self._closing or self._service_closed:
                raise ApiError(410, "The session lease is being released")
            thread = threading.Thread(target=run, name=name, daemon=daemon)
            self._background_jobs.add(thread)
            thread.start()

    def release_lease(self, token, command_id):
        """Give up ownership and tear the resident session down."""
        def release(_):
            with self._teardown_lock:
                lease = self.session_lease
                if lease is None:
                    raise ApiError(403, "This client does not own the session lease")
                lease.require(token)
                self._teardown()
                return {"released": True}
        return self._release_receipts.execute(command_id, 'release',
            {'token_digest': hashlib.sha256(str(token).encode()).hexdigest()},
            release, recoverable=lambda exc: isinstance(exc, (TimeoutError, OSError)))

    def _teardown(self):
        try:
            self._drain_and_close()
        except Exception:
            logging.exception("Spiral teardown failed; the lease is retained")
            raise

    def _drain_and_close(self):
        # Do not drop any references or locks until every user has stopped.
        with self._service_condition:
            self._closing = True
            self._service_condition.notify_all()
            deadline = time.monotonic() + 15
            while self._service_users or self._background_jobs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Service users did not stop; the lease is retained")
                self._service_condition.wait(remaining)
            lease, session = self.session_lease, self.session
            process = self._preview.process
        stop_process_group(process)
        if process is not None and process.poll() is None:
            raise TimeoutError("Preview process did not stop; the lease is retained")
        for retiring in tuple(self._retiring_sessions):
            retiring.close()
            self._retiring_sessions.remove(retiring)
        if session is not None:
            session.close()
        if lease is not None:
            lease.close()
        with self.lock:
            self.session = None
            self.session_id = None
            self.session_lease = None
            self._session_state = SessionState.Uninitialized
            self._session_phase = "Waiting for fit initialization"
            self._session_error = None
            self._closing = False

    def lease(self, *, create=True):
        """The session lease; it survives resident generations and disconnects."""
        with self.lock:
            if self._closing or self._service_closed:
                raise ApiError(410, "The session lease is being released")
            if self.session_lease is None:
                if not create:
                    raise ApiError(409, "Claim the session lease first")
                if self._output_root() is None:
                    raise ApiError(409, "A service output root is required")
                self.session_lease = SessionLease(self._output_root())
            return self.session_lease

    def lease_lifecycle(self, token, operation, request, callback):
        """Run one lifecycle command for the lease owner, at most once.

        Initialize and rebuild hold the request until the resident build
        settles, so a construction failure is the command's answer.
        """
        lease = self.lease()
        lease.require(token)
        def perform(captured):
            lease.require(token)
            rebuilding = operation in {"session_initialize", "session_rebuild",
                                       "session_update_inputs"}
            command_id = captured.get("command_id")
            if rebuilding and command_id in lease.lifecycle_started:
                response = lease.lifecycle_started[command_id]
            else:
                response = callback(captured)
                if rebuilding:
                    lease.lifecycle_started[command_id] = response
            if rebuilding:
                while True:
                    with self.lock:
                        building, session = self._building, self.session
                        updating = self._input_update is not None
                        state = self._session_state if session is None else session.status()["state"]
                    if not building and not updating and state != SessionState.Loading:
                        break
                    time.sleep(0.05)
                if state == SessionState.Error:
                    raise ApiError(409, "Resident construction failed", payload=self.status())
                with self.lock:
                    failed = self._input_preparation_error
                if failed and failed.get("build_id") == response.get("build_id"):
                    raise ApiError(409, "The inputs could not be prepared; the "
                                   "resident fit is unchanged", payload=self.status())
                response = {**self.status(), "accepted": True,
                            "build_id": response.get("build_id")}
            return response
        return lease.coordinator.execute(request.get("command_id"), operation,
            request, perform, recoverable=lambda exc: isinstance(exc, TimeoutError))

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
            response.setdefault("progress", None)
            # The status snapshot carries raw progress facts only. ETA is a
            # presentation value clients derive from step/total/elapsed.
            if isinstance(response.get("progress"), dict):
                response["progress"] = {
                    key: value
                    for key, value in response["progress"].items()
                    if key != "eta_seconds"
                }
            response["lease_id"] = (self.session_lease.id
                                    if self.session_lease else None)
            response["session_request"] = self.session_request
            response["preview_artifact"] = self._preview.artifact
            # Published separately from, and after, the surface: a client that
            # never opens an overlay never waits for one.
            response["preview_diagnostics_artifact"] = (
                self._preview.diagnostics_artifact)
            response["input_manifest_artifact"] = (
                self._input_manifest[1] if self._input_manifest else None)
            response["input_position"] = copy.deepcopy(self._input_position)
            response["input_preparation"] = self._input_preparation_status()
            response["input_preparation_error"] = copy.deepcopy(
                self._input_preparation_error)
            response["input_update"] = copy.deepcopy(self._input_update)
            response["preview_publish"] = (
                dict(self._preview.progress)
                if self._preview.progress else None)
            response["preview_publish_error"] = self._preview.error
            response["preview_active"] = bool(self._preview.generation)
            response["preview_pending"] = bool(
                self._preview.pending_generation)
            response["preview_source_iteration"] = (
                self._preview.source_fit_iteration)
            response["preview_input_position"] = copy.deepcopy(
                self._preview.input_position)
            response["preview_active_source_iteration"] = (
                self._preview.active_source_fit_iteration)
            response["preview_pending_source_iteration"] = (
                self._preview.pending_source_fit_iteration)
            response["preview_schedule"] = copy.deepcopy(
                response.get("preview_schedule") or self._preview_schedule)
            response["next_preview_iteration"] = response.get(
                "next_preview_iteration", self._next_preview_iteration)
            response["automatic_previews_disabled"] = (
                self._automatic_previews_disabled
                or bool(response.get("automatic_previews_disabled")))
            publishing = self._preview.status_progress()
            if publishing is not None:
                response["phase"] = publishing["stage_name"]
                response["progress"] = publishing
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
            "session_state": state,
            "cuda_ready": (None if state == SessionState.Loading else
                           state not in {SessionState.Uninitialized,
                                         SessionState.Error}),
        })
        return response

    def configuration_catalog(self):
        # Descriptions are re-read per request, so edited text reaches a
        # client when it reconnects, without restarting the service.
        catalog = {**self._base(), **describe_catalog(self.config_catalog)}
        if self.object_store is not None:
            # The launch binding owns these: advertise its values, and mark
            # them so a client shows them without sending them back.
            fields = dict(catalog["schema"]["fields"])
            for key in SERVICE_OWNED_CONFIG_KEYS:
                fields[key] = {**fields[key], "service_owned": True}
            catalog["schema"] = {**catalog["schema"], "fields": fields}
            catalog["defaults"] = {**catalog["defaults"],
                                   **self.object_store.config()}
        if self.bindings is None \
                or not self.bindings.winding_inference_status()["available"]:
            # Winding-model supervision cannot run here, so a client starting
            # from these defaults leaves the source out rather than having
            # preparation refuse it.
            catalog["defaults"] = {**catalog["defaults"],
                                   "input_use_winding_inference": False}
        return catalog

    def binding_advertisement(self):
        """GET /bindings: the launch host bindings, what the service can do
        with them, the checkpoints it can load and the bound store's
        selection."""
        capabilities = dict(SERVICE_CAPABILITIES)
        positioned = self._positioned()
        capabilities["build_missing_lasagna_packs"] = self.build_missing_lasagna_packs
        capabilities["input_positions"] = positioned
        # A service with positions updates a resident fit's inputs in place.
        capabilities["inputs_frozen"] = not positioned
        capabilities["input_updates"] = positioned
        bindings = (self.bindings.advertisement() if self.bindings is not None
                    else {"output_directory": None, "cache_directory": None,
                          "winding_inference": {"bound": False, "path": None,
                                                "available": False, "error": None}})
        # Whether winding-model supervision can run here.
        capabilities["winding_inference"] = bindings["winding_inference"]["available"]
        response = {**self._base(), "bindings": bindings,
                    "session_checkpoints": self.session_checkpoints(),
                    "capabilities": capabilities}
        if self.object_store is not None:
            response["object_store"] = self.object_store.advertisement()
            response["inputs"] = self.startup_inputs(wait=False)
        return response

    def startup_inputs(self, *, wait=True):
        """What the launch configuration selects from the bound store.

        Patches, PCLs and fibers with their revision pins, labels, frames
        and metadata PCL roles, plus counts, consumed relationship families,
        exclusions and the fit volume (object_store_inputs.snapshot_summary).
        Read-only: no link refresh, no materialization and no Torch. The
        result is reused until the store's files change.

        Reading a large store takes minutes. Without ``wait`` (GET
        /bindings) the caller never reads it: a summary not yet read for the
        store as it stands is read in the background, and meanwhile
        ``{"pending": True}`` is returned; ask again for the result.
        """
        key = self._startup_inputs_key()
        with self._startup_inputs_lock:
            cached = self._startup_inputs
            if cached is not None and cached[0] == key:
                return cached[1]
            if not wait:
                if self._startup_inputs_reading != key:
                    self._startup_inputs_reading = key
                    threading.Thread(target=self._read_startup_inputs_logged,
                                     args=(key,), name="spiral-startup-inputs",
                                     daemon=True).start()
                return {"pending": True}
        return self._read_startup_inputs(key)

    def _startup_inputs_key(self):
        path = Path(self.object_store.path)
        # An empty WAL (a read-only open after the last writer closed
        # creates one) holds no writes: it keys as an absent one.
        return tuple((info.st_size, info.st_mtime_ns) if info and info.st_size else None
                     for info in (_stat_or_none(path),
                                  _stat_or_none(Path(f"{path}-wal"))))

    def _read_startup_inputs(self, key):
        """Read the summary for the store as ``key`` stated it, once: a
        concurrent reader of the same key waits for it instead."""
        with self._startup_inputs_read_lock:
            with self._startup_inputs_lock:
                cached = self._startup_inputs
                if cached is not None and cached[0] == key:
                    return cached[1]
            from object_store_inputs import describe_selection
            config = Config(self.startup_run.get("config") or {}).as_dict()
            config.update(self.object_store.config())
            summary = describe_selection(config)
            with self._startup_inputs_lock:
                self._startup_inputs = (key, summary)
            return summary

    def _read_startup_inputs_logged(self, key):
        try:
            self._read_startup_inputs(key)
        except Exception as exc:
            # Not cached: the next GET /bindings tries again.
            self.events.append(
                "log", f"Could not read the store's selection: "
                f"{type(exc).__name__}: {exc}", severity="warning",
                source="service")
        finally:
            with self._startup_inputs_lock:
                if self._startup_inputs_reading == key:
                    self._startup_inputs_reading = None

    def session_checkpoints(self):
        """Checkpoints under the session output directory, newest first.

        These are the host checkpoints a client may name: everything this
        service wrote or received (saves, autosaves, uploads) lives under the
        output directory. Advertising them is what lets a client offer a
        choice instead of asking the user to type a path on a host it may
        never have seen.
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

    def _bound_session_request(self, request, project, *, include_input_config=False):
        """Resolve a request's inputs from its project and this service's
        bindings (the service adapter of the shared input contract; the
        project's umbilicus and fiber directions join once the build knows
        the fit volume, _resolve_session_scroll).

        A checkpoint-backed request intentionally carries no UI profile
        overrides: the checkpoint owns its durable configuration. Input
        selection must therefore consult that checkpoint configuration before
        clearing mode-specific paths. Otherwise a winding-model checkpoint is
        first treated like the default phase mode, ``winding_inference`` is
        erased, and only then does the fitter discover that the checkpoint
        requires it.
        """
        bindings = self.bindings
        requested_paths = request.get("paths") or {}
        offending = sorted(
            key for key, value in requested_paths.items()
            if key not in _CLIENT_SELECTABLE_PATHS
            and (value or (isinstance(value, list) and value))
        )
        if offending:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "This service owns its input paths; the request may name only "
                "a checkpoint",
                [{"field": f"paths.{key}",
                  "message": _RETIRED_REQUEST_PATHS.get(
                      key, "Input paths are the service's bindings and the project's")}
                 for key in offending])
        paths = {"output_directory": bindings.output_directory,
                 "cache_directory": bindings.cache_directory,
                 "winding_inference": bindings.winding_inference,
                 "provenance": bindings.provenance(project)}
        # The Lasagna dataset is the project's selected one, written as the
        # source document the fit reads (its packs are this service's cache's).
        if project.lasagna is not None and not project.lasagna.error:
            paths["lasagna"] = write_lasagna_source(
                Path(paths["output_directory"]) / "project-inputs",
                project.lasagna.source())
        checkpoint = str(requested_paths.get("checkpoint") or "").strip()
        if checkpoint:
            resolved_checkpoint = str(Path(checkpoint).resolve(strict=False))
            output_root = Path(paths["output_directory"]).resolve(strict=False)
            if not Path(resolved_checkpoint).is_relative_to(output_root):
                raise ApiError(HTTPStatus.BAD_REQUEST,
                               "Checkpoint must be one under the session output "
                               "directory",
                               [{"field": "checkpoint", "message": "Not a service-advertised checkpoint"}])
            paths["checkpoint"] = resolved_checkpoint

        requested_config = dict(
            (request.get("run") or {}).get("config") or {})
        config = requested_config
        if checkpoint:
            checkpoint_config, *_ = self._checkpoint_durable_cfg(
                paths["checkpoint"])
            if checkpoint_config is not None:
                config = {**checkpoint_config, **requested_config}
        if ("input_use_winding_inference" not in requested_config
                and winding_inference_enabled(config)
                and not bindings.winding_inference_status()["available"]):
            # Winding-model supervision cannot run here. A checkpoint's
            # configuration or the Python default enables it without the
            # request asking, so leave the source out rather than refuse the
            # fit; an explicit request is still refused, saying why.
            enabled_by = ("the checkpoint it resumes was fitted with it"
                          if checkpoint and config is not requested_config
                          else "the default configuration enables it")
            config = {**config, "input_use_winding_inference": False}
            request = {**request, "run": {
                **(request.get("run") or {}),
                "config": {**requested_config,
                           "input_use_winding_inference": False}}}
            self.events.append(
                "log", f"Warning: this fit leaves winding-model supervision "
                f"out, though {enabled_by}: the service has no usable winding "
                f"inference ({bindings.winding_inference_status()['error']}). "
                f"Restart it with --winding-inference PATH to include it.",
                severity="warning", source="service")

        # The session config owns which optional sources participate. Clear
        # disabled paths before validation so the manifest, validator, and
        # fitter all describe the same source set.
        if not lasagna_enabled(config):
            paths.pop("lasagna", None)
        if not winding_inference_enabled(config):
            paths["winding_inference"] = ""

        resolved_request = {**request, "paths": paths}
        if include_input_config:
            return resolved_request, config
        return resolved_request

    def _prepare_session_request(self, request, *, restore_checkpoint_z=False):
        """Validate one session request into the arguments a build needs."""
        input_config = None
        try:
            # The project (its fit volume and Lasagna dataset) owns these. A
            # request that names one is refused rather than quietly overruled.
            scroll_owned = sorted(
                key for key in SCROLL_SPEC_OWNED_RUN_KEYS
                if key in (request.get("run") or {}))
            if scroll_owned:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "The project owns these values; the request must not carry them",
                    [{"field": f"run.{key}",
                      "message": (f"Owned by the project's scroll as "
                                  f"{SCROLL_SPEC_OWNED_RUN_KEYS[key]!r}")}
                     for key in scroll_owned])
            request = self._bind_service_owned_config(request)
            project = self._request_project(request)
            if self.bindings is not None:
                request, input_config = self._bound_session_request(
                    request, project, include_input_config=True)
            paths, run, preview = parse_session_request(request)
        except (KeyError, TypeError, ValueError) as exc:
            # An unparseable request field (an unknown PCL role, say) is the
            # caller's error, not a service fault.
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Malformed session request: {exc}") from exc
        if restore_checkpoint_z and paths.checkpoint:
            _, _, checkpoint_z_range, _ = self._checkpoint_durable_cfg(
                paths.checkpoint)
            if checkpoint_z_range is not None:
                run = dataclasses.replace(
                    run, z_begin=checkpoint_z_range[0],
                    z_end=checkpoint_z_range[1])
        misplaced_z = sorted(
            key for key in ("z_begin", "z_end") if key in run.config)
        if misplaced_z:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "The z range belongs to the run block, not advanced config",
                [{"field": f"run.config.{key}",
                  "message": f"Set run.{key} instead"}
                 for key in misplaced_z])
        validation_run = (
            dataclasses.replace(run, config=input_config)
            if input_config is not None else run)
        # The project supplies the umbilicus once the fit volume is known.
        errors = validate_session_request(
            paths, validation_run, supplied=VOLUME_INPUT_KEYS,
            build_lasagna_packs=self.build_missing_lasagna_packs,
            missing={"winding_inference": (
                "winding-model supervision reads the service's winding-inference "
                "binding, and this service has none: restart it with "
                "--winding-inference PATH and prepare a new fit")})
        # Lasagna stores the configuration needs and the project cannot give.
        if lasagna_error := project.lasagna_error():
            needed = [spec.key for spec in FIT_INPUT_CATALOG
                      if spec.key in PROJECT_LASAGNA_INPUTS
                      and spec.required(validation_run.config)]
            errors = [error for error in errors if error["field"] not in needed]
            errors.extend({"field": key, "message": lasagna_error} for key in needed)
        # The project's scroll, resolved for the fit volume when the build
        # knows it (_resolve_session_scroll).
        scroll = SessionScroll(project)
        errors.extend(self._early_volume_problems(
            scroll, paths, validation_run.config))
        if errors:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Session validation failed", errors)
        return paths, run, preview, scroll

    def _early_volume_problems(self, scroll, paths, config):
        """What the project lacks for a fit volume known before the inputs
        are read: the configured fit_volume_uuid, else the volume a resumed
        checkpoint recorded. When the store's selection will name the volume
        instead, a project with no volumes, or with no umbilicus for any of
        them, cannot serve it either. Reading the store can take minutes on
        a large one, so these are refused on the request, not after it
        (_resolve_session_scroll repeats the check once the volume is
        known)."""
        project = scroll.project
        volume = config.get("fit_volume_uuid")
        if not volume and paths.checkpoint:
            *_, manifest = self._checkpoint_durable_cfg(paths.checkpoint)
            volume = ((manifest or {}).get("object_store") or {}).get("volume_uuid")
        try:
            if volume:
                scroll.umbilicus(str(volume))
            elif not project.volumes:
                raise ScrollSpecError(
                    f"the project {project.name!r} has no volumes: attach the "
                    "volume the fit reads")
            elif not any(entry.umbilicus is not None
                         for entry in project.volumes.values()):
                raise ScrollSpecError(
                    f"the project {project.name!r} has no umbilicus for any of "
                    f"its volumes ({', '.join(sorted(project.volumes))}): import "
                    "one for the fit volume (File > Import Umbilicus)")
        except ScrollSpecError as exc:
            return [{"field": "project", "message": str(exc)}]
        return []

    @staticmethod
    def _request_project(request):
        """The request's ``project`` block (fit_session.SpiralProject): the
        VC3D project whose inputs the session fits. Every session request
        carries it; the service cannot read the project file itself."""
        value = request.get("project")
        if value is None:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "The request carries no project: this service fits the inputs "
                "of the VC3D project that asks for the session",
                [{"field": "project", "message": "Required"}])
        try:
            return SpiralProject.from_mapping(value)
        except ScrollSpecError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid project: {exc}",
                           [{"field": "project", "message": str(exc)}]) from exc

    def _effective_config(self, paths, run):
        """The configuration a build applies: a resumed checkpoint's stored
        one under the request's overrides."""
        from spiral_runtime import merge_session_config
        config, _, overrides = merge_session_config(
            paths.checkpoint, run, warn=lambda warning: None)
        config.update(overrides)
        return config

    def _resolve_session_scroll(self, scroll, volume, paths, config):
        """Complete a build's SessionScroll for its fit volume: the scroll it
        fits (with ``config``'s spiral_outward_sense, and the Lasagna scale
        when it reads the stores), its umbilicus (the project's, written
        under the output directory as the file the fit reads) and, when its
        losses need them, the fiber directions the cache holds for the
        project's fiber-inference dataset at the volume's scale. The fit infers
        the same volume from the same inputs, and an input update that would
        read another is refused (InputReplacementRefused). Returns ``(paths,
        ScrollSpec)``; raises ScrollSpecError."""
        spec = scroll.resolve(volume, config.get("spiral_outward_sense"),
                              lasagna=bool(paths.lasagna))
        umbilicus = write_project_umbilicus(
            Path(paths.output_directory) / "project-inputs", volume,
            scroll.umbilicus(volume))
        # Loaded whenever enabled, as the CLI loads its manifest's, so a later
        # run with the loss on finds them resident; required with the loss.
        fiber_directions = ""
        level = config.get("fit_volume_level", 0)
        if fiber_directions_required(config):
            fiber_directions = scroll.fiber_directions(volume, paths.cache_directory, level)
        elif input_source_enabled(config, "fiber_directions"):
            try:
                fiber_directions = scroll.fiber_directions(
                    volume, paths.cache_directory, level)
            except ScrollSpecError:
                pass
        return dataclasses.replace(paths, volume_uuid=volume, umbilicus=umbilicus,
                                   fiber_directions=fiber_directions), spec

    def _bind_service_owned_config(self, request):
        """Refuse client values for SERVICE_OWNED_CONFIG_KEYS, then set them
        from the launch binding.

        A client may leave them out, send them unset (a complete default
        configuration) or echo the binding's own values back; any other
        value is refused. Request overrides are applied after a resumed
        checkpoint's stored cfg (spiral_runtime._resolve_session_config), so
        the binding also wins over whatever store that checkpoint was
        written against.
        """
        run = dict(request.get("run") or {})
        config = dict(run.get("config") or {})
        bound = self.object_store.config() if self.object_store else {}
        owned = sorted(key for key in SERVICE_OWNED_CONFIG_KEYS
                       if config.get(key) is not None
                       and config[key] != bound.get(key))
        if owned:
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "This service is bound to its object store at launch; the "
                "request must not carry these",
                [{"field": f"run.config.{key}",
                  "message": "Owned by the service's --object-store binding"}
                 for key in owned])
        if self.object_store is None:
            return request
        config.update(bound)
        return {**request, "run": {**run, "config": config}}

    def startup_session_request(self):
        """The launch defaults used by explicit recovery rebuilds."""
        return {"run": dict(self.startup_run)}

    def initialize(self, request):
        """Create the first resident fit session on explicit client request."""
        request = dict(request or {})
        request.pop("command_id", None)
        minimum = self._inputs_position(request.pop("inputs_position", None))
        with self.lock:
            if self._building or self.session_id is not None \
                    or self._session_state != SessionState.Uninitialized:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "The fit session has already been initialized")
        paths, run, preview, scroll = self._prepare_session_request(
            request, restore_checkpoint_z=True)
        if paths.checkpoint and run.config:
            self._reject_overrides_the_checkpoint_contradicts(
                paths.checkpoint, run.config)
        build_id = self._begin_build(paths, run, preview, scroll, minimum=minimum)
        return {**self.status(), "accepted": True, "initializing": True,
                "build_id": build_id}

    def _refuse_during_input_update(self, verb):
        with self.lock:
            if self._input_update is not None:
                raise ApiError(HTTPStatus.CONFLICT,
                               f"{verb} waits for the input update in progress",
                               payload={"code": INPUT_UPDATE_IN_PROGRESS})

    def _positioned(self):
        return self.object_store is not None and self.object_store.positioned

    def _inputs_position(self, value):
        """A request's ``inputs_position``: the position (M) the build's
        inputs must include, e.g. where a client's saved (and, with a hub,
        pushed) edits reached."""
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "inputs_position must be a non-negative integer")
        if not self._positioned():
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "This service's object store records no positions")
        return value

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
        teardown and reconstruction it has always been, which rereads the
        object store. ``{"inputs": "reload"}`` asks for that full rebuild
        whatever the configuration change: it picks up new store revisions
        under an unchanged configuration. The response's ``inputs`` says
        whether the rebuild rereads the inputs ("reload") or keeps the frozen
        ones ("keep").
        """
        request = dict(request or {})
        request.pop("command_id", None)
        with self.lock:
            if self.session_id is None:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    "There is no fit session to rebuild; initialize it first")
        self._refuse_during_input_update("A rebuild")
        inputs = request.pop("inputs", "keep")
        if inputs not in ("keep", "reload"):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           'inputs must be "keep" or "reload"')
        minimum = self._inputs_position(request.pop("inputs_position", None))
        if minimum is not None:
            # A position to include is a reread of the inputs.
            inputs = "reload"
        defaults = request.pop("defaults", False)
        if not isinstance(defaults, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "defaults must be true or false")
        if defaults:
            # The project is the client's to send, as with any session request.
            project = request.pop("project", None)
            if set(request):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "A defaults rebuild takes no other request fields than the project")
            request = {**self.startup_session_request(), "project": project}
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
            if self._input_update is not None:  # began after the check above
                raise ApiError(HTTPStatus.CONFLICT,
                               "A rebuild waits for the input update in progress",
                               payload={"code": INPUT_UPDATE_IN_PROGRESS})
            state = self.session.status()["state"] if self.session else None
            if state in SESSION_BUSY_STATES:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"A rebuild requires an idle or failed session (state is "
                    f"{SessionState(state).name})")
            # Rereading the inputs is the whole build: a model-only rebuild
            # keeps the frozen inputs it loaded.
            stage = ("all" if defaults or inputs == "reload"
                     else self._rebuild_stage_locked(paths, run, preview, scroll, state))
            live_request = self.session_request or {}
            live_config = (live_request.get("run") or {}).get("config", {})
            live_paths = self.session_paths
        build_id = None
        if stage == "model":
            # The model stage keeps the frozen inputs, at their position, and
            # the project's umbilicus.
            run = dataclasses.replace(run, config={
                **run.config,
                "object_store_input_position": live_config.get("object_store_input_position")})
            paths = dataclasses.replace(paths, umbilicus=live_paths.umbilicus)
            self._begin_model_rebuild(paths, run, preview)
        else:
            build_id = self._begin_build(paths, run, preview, scroll,
                                         minimum=minimum,
                                         reload_inputs=inputs == "reload")
        return {**self.status(), "accepted": True, "rebuilding": True,
                "stage": stage, "inputs": "reload" if stage == "all" else "keep",
                "build_id": build_id}

    def update_inputs(self, request):
        """Update the resident fit's inputs to a position, keeping its
        model and optimiser state (stage 2b).

        ``inputs_position`` (M) is the position the inputs must include,
        e.g. where a client's pushed edits reached; the service links and
        pins N at the hub head (at least M), then every rank replaces its
        inputs with N's at an iteration boundary. ``expected_input_generation``
        refuses an update computed against another generation than the
        session's. The response returns at once; the work runs off the HTTP
        thread, and a lease-holding client's command is answered when it
        settles. A refusal leaves the fit as it was (input_preparation_error);
        a failure past the release fails the session, which then needs a
        rebuild or a checkpoint load.
        """
        request = dict(request or {})
        request.pop("command_id", None)
        minimum = self._inputs_position(request.pop("inputs_position", None))
        expected = request.pop("expected_input_generation", None)
        if request:
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           f"Unknown input update fields: {', '.join(sorted(request))}")
        if not self._positioned():
            raise ApiError(HTTPStatus.CONFLICT,
                           "This service's object store records no positions; rebuild "
                           "the fit with current inputs instead",
                           payload={"code": INPUT_UPDATE_UNAVAILABLE})
        with self.lock:
            session = self.session
            if session is None:
                raise ApiError(HTTPStatus.CONFLICT,
                               "There is no fit session to update; initialize it first",
                               payload={"code": INPUT_UPDATE_UNAVAILABLE})
            if self._building or self._input_update is not None:
                raise ApiError(HTTPStatus.CONFLICT,
                               "The fit session is being built, or its inputs updated",
                               payload={"code": INPUT_UPDATE_BUSY})
            status = session.status()
            if status["state"] != SessionState.Idle:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"An input update requires an idle session (state is "
                    f"{SessionState(status['state']).name})",
                    payload={"code": INPUT_UPDATE_BUSY})
            generation = int(status.get("input_generation") or 0)
            if expected is not None and expected != generation:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"The update was computed against input generation {expected}; "
                    f"the session is at {generation}",
                    payload={"code": INPUT_GENERATION_STALE})
            config = dict(status.get("applied_config") or {})
            paths, scroll = self.session_paths, self._session_scroll
            self._build_serial += 1
            update_id = self._build_serial
            self._input_update = {"build_id": update_id, "requested_position": minimum,
                                  "from_generation": generation}
            self._input_preparation = {"build_id": update_id, "requested_position": minimum}
            self._input_preparation_error = None
            self.status_generation += 1
            session_id = self.session_id
        try:
            self._start_background(
                target=self._update_inputs_job,
                args=(update_id, session_id, session, config, paths, scroll, minimum),
                name="spiral-input-update", daemon=True)
        except BaseException:
            with self.lock:
                self._input_update = self._input_preparation = None
            raise
        return {**self.status(), "accepted": True, "updating_inputs": True,
                "build_id": update_id}

    def _update_inputs_job(self, update_id, session_id, session, config, paths, scroll,
                           minimum):
        from spiral_runtime import InputUpdateFailed, InputUpdateRefused

        def refused(exc, phase):
            cause = _cause(exc)
            with self.lock:
                self._input_update = None
                self._input_preparation = None
                self._input_preparation_error = {
                    "build_id": update_id, "requested_position": minimum,
                    "phase": phase, "recovery_required": False, "message": cause,
                    "problems": list(getattr(exc, "problems", None) or [cause])}
                self.status_generation += 1
            self.events.append(
                "error", f"The inputs could not be updated; the fit is unchanged: {cause}",
                severity="error", source="service", operation="updating_inputs")
        try:
            prepared = self._prepare_position(config, paths, scroll, minimum,
                                              incremental=True)
        except BaseException as exc:
            refused(exc, "linking")
            return
        for line in prepared.lines:
            self.events.append("log", line, source="service", operation="updating_inputs")
        with self.lock:
            self._input_preparation = None
            self._input_update = {**self._input_update, "position": prepared.position.sequence}
            self.status_generation += 1
        try:
            result = session.update_inputs(prepared.position.sequence)
        except InputUpdateRefused as exc:
            refused(exc, "preparing")
            return
        except BaseException as exc:
            if not isinstance(exc, InputUpdateFailed) \
                    and session.status()["state"] != SessionState.Error:
                # Not past the release (InputUpdateFailed is), and the fit
                # did not fail: it still holds its inputs.
                refused(exc, "preparing")
                return
            cause = _cause(exc)
            with self.lock:
                self._input_update = None
                self._input_preparation_error = {
                    "build_id": update_id, "requested_position": minimum,
                    "phase": "installing", "recovery_required": True, "message": cause,
                    "problems": [cause]}
                self.status_generation += 1
            self.events.append(
                "error", f"The input update failed after the fit released its inputs; "
                f"rebuild the fit or load a checkpoint: {cause}", severity="error",
                source="service", operation="updating_inputs")
            return
        with self.lock:
            if self.session_id == session_id:
                # The live request names the inputs now resident: a later
                # model-only rebuild (and its checkpoints) keeps this position.
                if self.session_request:
                    run = dict(self.session_request.get("run") or {})
                    run["config"] = {**(run.get("config") or {}),
                                     "object_store_input_position": prepared.position.sequence}
                    self.session_request = {**self.session_request, "run": run}
                self._input_position = {
                    **prepared.position.provenance(), "requested": minimum,
                    "head": prepared.head, "attempts": prepared.attempts,
                    "source": "update",
                    "input_generation": result.get("input_generation")}
            self._input_update = None
            self.status_generation += 1
        self.events.append(
            "log", f"Inputs updated to position {prepared.position.sequence} "
            f"(input generation {result.get('input_generation')})",
            source="service", operation="updating_inputs")

    def _rebuild_stage_locked(self, paths, run, preview, scroll, state):
        """How much of the resident session this request has to replace.

        Everything outside ``run.config`` is ``all``. Paths name host inputs
        whose contents another process may have changed, so retaining a stage
        across one would need a content fingerprint nothing computes; the
        preview block, the run tag, the z window and the storage backend are
        read before or outside the model. Within ``run.config`` the answer is
        ``config.rebuild_stage`` over the keys whose requested value differs
        from the live session's, which is "model" only for the audited
        allowlist and "all" for everything else. The shell atlas keys widen
        to "all" when the session loaded both a tracks store and the outer
        shell, because it filtered the tracks against the shell then.

        Call with the lock held.
        """
        current = self.session_request
        if (self.session is None or current is None
                or state != SessionState.Idle):
            # Nothing to keep: there is no resident session, or it has no
            # model to rebuild around (Error), or it is not quiescent.
            return "all"
        if paths.checkpoint and int(self.session.status().get("input_generation") or 0):
            # The model stage resumes from the checkpoint, whose inputs an
            # update has since replaced: its manifest no longer matches the
            # resident inputs, and the refusal would come after the model was
            # released. The full rebuild reads the checkpoint's position.
            return "all"
        # The project supplies VOLUME_INPUT_KEYS (and the build the fit
        # volume itself) once the build knows the fit volume; they and the
        # scroll change with the project's facts of that volume.
        live_paths = dict(current.get("paths") or {})
        live_paths.update(dict.fromkeys(VOLUME_INPUT_KEYS, ""), volume_uuid="")
        if (paths.manifest() != live_paths
                or preview.manifest() != current.get("preview")):
            return "all"
        live_project = current.get("project") or {}
        volume = live_project.get("volume")
        if (volume not in scroll.project.volumes
                or scroll.project.record(volume) != live_project):
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
        # The service pins the position; a request never carries the live one.
        changed.discard("object_store_input_position")
        # A session that read tracks and the store's outer shell filtered
        # its tracks against the shell at load: the fitter refuses the shell
        # atlas settings there, and only the full rebuild can apply them.
        summary = self.session.status().get("input_summary") or {}
        return rebuild_stage(
            changed,
            shell_filtered_tracks=bool(
                (summary.get("selection") or {}).get("track")
                and summary.get("outer_shell")))

    def _begin_model_rebuild(self, paths, run, preview):
        """Publish the new request and rebuild the model off the HTTP thread.

        The session object, its generation, and its resident inputs survive;
        the session reports ``Loading`` while the fitter thread works.
        """
        with self.lock:
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            self._building = True
            self.session_paths = paths
            self.session_request = {
                **self.session_request,
                "paths": paths.manifest(),
                "run": run.manifest(),
                "preview": preview.manifest(),
            }
            self.session_revision += 1
            self.status_generation += 1
            session_id = self.session_id
            session = self.session
        self._start_background(
            target=self._rebuild_model,
            args=(session_id, session, paths, run),
            name="spiral-model-rebuild", daemon=True)

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

    def _begin_build(self, paths, run, preview, scroll, *, minimum=None,
                     reload_inputs=False):
        """Publish ``Loading`` and construct the runtime off the HTTP thread.

        On a binding with positions the build reads a position. A checkpoint
        resume reads the position the checkpoint recorded of this store (or hub),
        unless the request reloads the inputs or names a position to
        include; otherwise the inputs are linked and pinned first
        (``_prepare_then_build``), while the resident session, if any, is
        left alone. Returns the build's ID.
        """
        record = None
        if self._positioned():
            position = None
            if paths.checkpoint and minimum is None and not reload_inputs:
                position = self._checkpoint_position(paths.checkpoint)
            if position is None:
                return self._begin_prepared_build(paths, run, preview, scroll,
                                                  minimum)
            if self.build_missing_lasagna_packs:
                return self._begin_prepared_build(
                    paths, run, preview, scroll, minimum, checkpoint_position=position)
            run = self._pin_run(run, position)
            record = {"hub": self.object_store.positions,
                      "sequence": position, "requested": None, "head": None,
                      "attempts": 0, "source": "checkpoint"}
        # The checkpoint's inputs name the fit volume.
        project = scroll.project
        try:
            volume = self._unprepared_volume(paths, scroll)
            paths, scroll = self._resolve_session_scroll(
                scroll, volume, paths, self._effective_config(paths, run))
        except ScrollSpecError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, f"Session validation failed: {exc}",
                           [{"field": "project", "message": str(exc)}]) from exc
        with self.lock:
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            self._building = True
            build_id, session_id, previous = self._publish_build_locked(
                paths, run, preview, record, scroll, project, volume)
        self._start_background(
            target=self._build,
            args=(session_id, previous, paths, run, preview, scroll),
            name="spiral-session-build", daemon=True)
        return build_id

    @staticmethod
    def _pin_run(run, position):
        return dataclasses.replace(run, config={
            **run.config, "object_store_input_position": int(position)})

    def _unprepared_volume(self, paths, scroll):
        """The fit volume of a build that prepares no position: the one its
        checkpoint's inputs were read in. A build without one (on a service
        without an object store, which only tests construct) fits the
        project's only volume."""
        if paths.checkpoint:
            *_, manifest = self._checkpoint_durable_cfg(paths.checkpoint)
            volume = ((manifest or {}).get("object_store") or {}).get("volume_uuid")
            if volume:
                return str(volume)
        volumes = sorted(scroll.project.volumes)
        if len(volumes) != 1:
            raise ScrollSpecError(
                "the build reads no fit volume from its inputs, so it must be the "
                f"project's only volume (it has {len(volumes)})")
        return volumes[0]

    def _checkpoint_position(self, path):
        """The position on this service's hub a checkpoint's inputs were
        read at, or None."""
        *_, manifest = self._checkpoint_durable_cfg(path)
        position = ((manifest or {}).get("object_store") or {}).get("position")
        if not position or position.get("hub") != self.object_store.positions:
            return None
        return int(position["sequence"])

    def _begin_prepared_build(self, paths, run, preview, scroll, minimum,
                              checkpoint_position=None):
        """Prepare packs and a position off the HTTP thread, then build.

        A checkpoint keeps its recorded position without refreshing links.
        """
        with self.lock:
            if self._building:
                raise ApiError(HTTPStatus.CONFLICT,
                               "A session build is already in progress")
            self._building = True
            self._build_serial += 1
            build_id = self._build_serial
            self._input_preparation = {"build_id": build_id,
                                       "requested_position": minimum}
            self._input_preparation_error = None
            self.status_generation += 1
        try:
            self._start_background(
                target=self._prepare_then_build,
                args=(build_id, paths, run, preview, scroll, minimum, checkpoint_position),
                name="spiral-input-position", daemon=True)
        except BaseException:
            with self.lock:
                self._building = False
                self._input_preparation = None
            raise
        return build_id

    def _prepare_inputs(self, paths, run, scroll, minimum):
        """Link and pin the build's position with the configuration it will
        apply: the fit volume its selection names now resolves the scroll
        (the link refresh reads normals at its Lasagna scale), then
        object_store_inputs.prepare_position, which must leave that volume
        the fit volume. Returns ``(prepared, volume, paths, ScrollSpec)``."""
        from vc_object_store import ObjectStore
        from object_store_inputs import read_snapshot, reading_stage, request_from_config
        config = self._effective_config(paths, run)
        progress = self._input_progress()
        try:
            with ObjectStore.open(self.object_store.path, read_only=True) as store, \
                    reading_stage(progress, "Reading the selected inputs", "fit volume"):
                volume = read_snapshot(store, request_from_config(config)).volume_uuid
        finally:
            progress.close()
        self._prepare_fiber_directions(scroll, volume, paths, config)
        paths, spec = self._resolve_session_scroll(scroll, volume, paths, config)
        prepared, pinned = self._prepare_position(config, paths, spec, minimum, volume=True)
        if pinned != volume:
            raise ScrollSpecError(
                f"the inputs' fit volume changed from {volume} to {pinned} while "
                "they were linked; retry")
        return prepared, volume, paths, spec

    def _input_progress(self):
        """A ProgressReporter publishing to the input preparation's status
        (``input_preparation.progress``) while one is active."""
        from spiral_progress import ProgressReporter

        def publish(snapshot):
            with self.lock:
                if self._input_preparation is not None:
                    self._input_preparation["progress"] = snapshot
                    # A stage without steps (a store read) publishes once:
                    # its elapsed time is recomputed for every status.
                    self._input_preparation_progress_started = (
                        None if snapshot is None
                        else time.monotonic() - snapshot["elapsed_seconds"])
                    self.status_generation += 1
        return ProgressReporter(publish, stream=sys.stderr)

    def _input_preparation_status(self):
        """``input_preparation`` for /session/status (call with the lock
        held), its progress's elapsed time current."""
        preparation = copy.deepcopy(self._input_preparation)
        started = getattr(self, "_input_preparation_progress_started", None)
        if preparation and preparation.get("progress") and started is not None:
            preparation["progress"]["elapsed_seconds"] = time.monotonic() - started
        return preparation

    def _prepare_position(self, config, paths, scroll, minimum, *, volume=False,
                          incremental=False):
        """Link and pin a position for ``config`` on the bound store; with
        ``volume``, also the fit volume pinned there (``(prepared,
        volume)``). ``incremental`` (an input update) links only objects
        selected since the store's last refresh, when nothing else changed."""
        from vc_object_store import ObjectStore
        from fit_session import lasagna_normal_inputs
        from object_store_inputs import (fit_volume_at, prepare_position,
                                         request_from_config)
        self._prepare_lasagna_packs(paths, config)
        lasagna = lasagna_normal_inputs(
            load_lasagna_source(paths.lasagna) if paths.lasagna else None,
            config, scroll.lasagna_scale, paths.cache_directory)
        cache = (os.path.join(paths.cache_directory, "object_store_strips")
                 if paths.cache_directory else None)

        def log(line):
            self.events.append("log", line, source="service",
                               operation="preparing_inputs")
        request = request_from_config(config)
        progress = self._input_progress()
        try:
            with ObjectStore.open(self.object_store.path,
                                  author=self.object_store.author) as store:
                prepared = prepare_position(
                    store, config, request, lasagna=lasagna,
                    minimum=minimum, cache_directory=cache, log=log,
                    incremental=incremental, progress=progress)
                if not volume:
                    return prepared
                return prepared, fit_volume_at(store, request, prepared, progress)
        finally:
            progress.close()

    def _prepare_fiber_directions(self, scroll, volume, paths, config):
        """Unless it requires prepared inputs, a host extracts the fiber
        directions a fit with the fiber-direction loss reads, when the cache
        lacks them, from the project's fiber-inference dataset (whose own
        problems the scroll's resolution reports)."""
        if not self.build_missing_lasagna_packs or not fiber_directions_required(config):
            return
        try:
            level = config.get("fit_volume_level", 0)
            path = scroll.fiber_directions_cache_path(volume, paths.cache_directory, level)
        except ScrollSpecError:
            return
        if os.path.isfile(path):
            return
        from fiber_direction_samples import extract

        def log(line):
            self.events.append("log", f"Fiber directions: {line}", source="service",
                               operation="preparing_inputs")
        dataset = scroll.project.fiber_inference
        location = dataset.extraction_location()
        log(f"extracting from {location} into {path}")
        extract(location, None, path, identity=dataset.identity,
                output_scale=scroll.fiber_scale(volume, level),
                workers=FIBER_DIRECTION_WORKERS, log=log)

    def required_lasagna_packs(self, request):
        """``{"packs": [...]}``: the Lasagna pack kinds a build of this
        session request (as /session/initialize or /session/rebuild takes
        it) reads, so a client caches only the remote arrays they are built
        from."""
        request = dict(request or {})
        request.pop("command_id", None)
        paths, run, _, _ = self._prepare_session_request(request, restore_checkpoint_z=True)
        if not paths.lasagna:
            return {"packs": []}
        return {"packs": sorted(_lasagna_pack_kinds(
            self._effective_config(paths, run), load_lasagna_source(paths.lasagna)))}

    def _prepare_lasagna_packs(self, paths, config):
        """Unless it requires prepared packs, a host builds missing pools
        before linking: from local arrays, or from HTTP(S) ones it first
        fetches whole into the cache (beside the pack, removed once it is
        built; an interrupted fetch resumes)."""
        if not self.build_missing_lasagna_packs or not paths.lasagna:
            return
        from lasagna_source import is_remote_location
        from remote_zarr import is_http
        source = load_lasagna_source(paths.lasagna)
        kinds = _lasagna_pack_kinds(config, source)
        if not kinds:
            return
        for kind in sorted(kinds):
            if os.path.isfile(os.path.join(
                    source.pack_directory(paths.cache_directory, kind), "meta.json")):
                continue
            for array in source.arrays(kind):
                if is_remote_location(array) and not is_http(array):
                    raise ValueError(
                        f"Cannot build missing Lasagna {kind} pack: {array} is "
                        "neither local nor HTTP(S). Prepare the pack where the "
                        "arrays are and copy it into the service cache.")
                if not is_remote_location(array) and not os.path.isdir(array):
                    raise ValueError(
                        f"Cannot build missing Lasagna {kind} pack: {array} is not "
                        "an accessible local array. Prepare the pack where the "
                        "arrays are and copy it into the service cache.")
        from lasagna_data import ensure_fit_sparse_stores, _require_sidecar

        progress = self._input_progress()
        try:
            ensure_fit_sparse_stores(
                source=source, cache_directory=paths.cache_directory,
                use_normals="normals" in kinds, use_spacing="grad_mag" in kinds,
                progress=progress, local_arrays=_fetched_lasagna_arrays)
            for kind in sorted(kinds):
                _require_sidecar(source, paths.cache_directory, kind)
        finally:
            progress.close()

    def _prepare_then_build(self, build_id, paths, run, preview, scroll, minimum,
                            checkpoint_position=None):
        try:
            project = scroll.project
            if checkpoint_position is None:
                prepared, volume, paths, scroll = self._prepare_inputs(
                    paths, run, scroll, minimum)
                for line in prepared.lines:
                    self.events.append("log", line, source="service",
                                       operation="preparing_inputs")
                record = {**prepared.position.provenance(), "requested": minimum,
                          "head": prepared.head, "attempts": prepared.attempts,
                          "source": "prepared"}
                run = self._pin_run(run, prepared.position.sequence)
            else:
                run = self._pin_run(run, checkpoint_position)
                volume = self._unprepared_volume(paths, scroll)
                config = self._effective_config(paths, run)
                self._prepare_fiber_directions(scroll, volume, paths, config)
                paths, scroll = self._resolve_session_scroll(scroll, volume, paths, config)
                self._prepare_lasagna_packs(paths, config)
                record = {"hub": self.object_store.positions,
                          "sequence": checkpoint_position, "requested": None,
                          "head": None, "attempts": 0, "source": "checkpoint"}
        except BaseException as exc:
            cause = _cause(exc)
            with self.lock:
                self._building = False
                self._input_preparation = None
                self._input_preparation_error = {
                    "build_id": build_id, "requested_position": minimum,
                    "message": cause,
                    "problems": list(getattr(exc, "problems", None) or [cause])}
                self.status_generation += 1
            self.events.append(
                "error", f"The inputs could not be prepared; the resident fit "
                f"is unchanged: {cause}", severity="error", source="service",
                operation="preparing_inputs")
            return
        with self.lock:
            self._input_preparation = None
            _, session_id, previous = self._publish_build_locked(
                paths, run, preview, record, scroll, project, volume,
                build_id=build_id)
        self._build(session_id, previous, paths, run, preview, scroll)

    def _publish_build_locked(self, paths, run, preview, input_position, scroll,
                              project, volume, *, build_id=None):
        """Retire the resident session and publish ``Loading`` for a new one,
        fitting ``scroll`` (_resolve_session_scroll's ScrollSpec) of the
        ``project`` (a SpiralProject) in the fit volume ``volume``. Call with
        the lock held and ``_building`` claimed."""
        if build_id is None:
            self._build_serial += 1
            build_id = self._build_serial
        previous = self.session
        if previous is not None:
            self._retiring_sessions.append(previous)
        self.session = None
        self.session_generation += 1
        self.session_id = f"spiral-{self.session_generation}-{secrets.token_hex(5)}"
        self.session_paths = paths
        self.session_request = {
            "paths": paths.manifest(),
            "run": run.manifest(),
            "preview": preview.manifest(),
            "project": project.record(volume),
            "scroll": scroll.manifest(),
        }
        self._session_project = project
        self.session_revision += 1
        self._reset_session_scope()
        self._input_position = input_position
        self._session_scroll = scroll
        self._session_state = SessionState.Loading
        self._session_phase = "Building the fit session"
        self._session_error = None
        self.status_generation += 1
        return build_id, self.session_id, previous

    def _build(self, session_id, previous, paths, run, preview, scroll):
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
                with self.lock:
                    self._retiring_sessions.remove(previous)
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
        self._preview_schedule = None
        self._next_preview_iteration = None
        self._automatic_previews_disabled = False
        self._automatic_preview_generations.clear()
        self._checkpoint_state = None
        self._event_progress_signatures = {}
        self._event_metric_iterations = {}
        self._event_errors = {}
        self._event_input_lines = {}
        self._input_manifest = None
        stale_raw = self._preview.reset_session_scope()
        for manifest in stale_raw:
            shutil.rmtree(
                Path(manifest).parent, ignore_errors=True)

    def _status_changed(self, status):
        # Runs on the fitter thread. It may only claim immutable raw work;
        # publication itself is background work and must never hold the fit.
        try:
            self._maybe_register_artifacts(status)
        except Exception as exc:
            print(f"SPIRAL_ARTIFACT_ERROR {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        try:
            self._note_checkpoint_state(status)
        except Exception as exc:
            print(f"SPIRAL_ARTIFACT_ERROR {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        try:
            self._note_input_manifest(status)
        except Exception as exc:
            print(f"SPIRAL_ARTIFACT_ERROR {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
        with self.lock:
            self.status_generation += 1

    def _note_input_manifest(self, status):
        """Publish the session's input manifest once its inputs are frozen.

        Runs on the fitter thread: it claims the inputs (their digest and
        input generation: an update to another position with identical
        inputs still names that position) and leaves copying and hashing to
        a background thread.
        """
        digest = status.get("input_digest")
        path = status.get("input_manifest_path")
        if not digest or not path:
            return
        key = (digest, int(status.get("input_generation") or 0))
        with self.lock:
            if self.session_id is None or (
                    self._input_manifest and self._input_manifest[0] == key):
                return
            self._input_manifest = (key, None)
            session_id = self.session_id
            generation = self.session_generation
            output = self._output_root()
        threading.Thread(
            target=self._publish_input_manifest,
            args=(session_id, generation, key, Path(path), output),
            name="spiral-input-manifest-publish", daemon=True).start()

    def _publish_input_manifest(self, session_id, generation, key, source,
                                output):
        digest = key[0]
        root = (Path(output) / ".spiral-artifacts" /
                f"inputs-{digest[:12]}-{secrets.token_hex(6)}")
        try:
            root.mkdir(parents=True)
            shutil.copy2(source, root / INPUT_MANIFEST_FILE_NAME)
            ref = self.artifacts.register_directory(
                INPUT_MANIFEST_ARTIFACT_KIND, session_id, generation, root,
                INPUT_MANIFEST_FILE_NAME, delete_root_on_prune=True)
            ref["input_digest"] = digest
        except Exception as exc:
            shutil.rmtree(root, ignore_errors=True)
            self.events.append(
                "log", f"The input manifest could not be published: "
                f"{type(exc).__name__}: {exc}", severity="warning",
                source="service", operation="loading_inputs")
            return
        with self.lock:
            current = (self.session_id == session_id and self._input_manifest
                       and self._input_manifest[0] == key)
            if current:
                self._input_manifest = (key, ref)
                self.status_generation += 1
        if not current:
            # The session was replaced meanwhile: withdraw only this copy.
            self.artifacts.prune(
                INPUT_MANIFEST_ARTIFACT_KIND, session_id, 0,
                retain=lambda artifact: Path(artifact.root) != root)
            return
        self.artifacts.prune(INPUT_MANIFEST_ARTIFACT_KIND, session_id, 1)

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
        report = status.get("input_report")
        if isinstance(report, list):
            # The object-store adapter's report while it loads the inputs:
            # selection, refresh outcomes, constraint counts and warnings.
            with self.lock:
                published = self._event_input_lines.get(rank, 0)
                self._event_input_lines[rank] = max(published, len(report))
            for line in report[published:]:
                self.events.append(
                    "inputs", str(line),
                    severity=("warning" if str(line).startswith("WARNING")
                              else "info"),
                    source="fitter", rank=rank, session_generation=generation,
                    operation="loading_inputs")
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
                problems = status.get("input_problems")
                self.events.append(
                    "error", str(error), severity="error", source="fitter",
                    rank=rank, session_generation=generation,
                    payload={"input_problems": problems} if problems else None)

    def _maybe_register_artifacts(self, status):
        """Offer a raw snapshot to the bounded background publisher.

        This callback runs on the fitter thread.  It must only claim or
        coalesce the immutable raw generation; all Lasagna, mapping, hashing,
        and indexing work belongs to the background publication thread.
        """
        with self.lock:
            session_id = self.session_id
            preview_generation = int(status.get("preview_generation") or 0)
            preview_manifest = status.get("preview_manifest_path")
            if preview_generation and status.get("preview_schedule"):
                self._automatic_preview_generations.add(preview_generation)
            old_pending = self._preview.pending_manifest
            publish_preview = bool(preview_manifest) and self._preview.claim(
                session_id, preview_generation, manifest=preview_manifest,
                source_fit_iteration=status.get("preview_source_iteration",
                                                status.get("current_iteration")),
                diagnostics=bool(status.get("preview_diagnostics")))
            replaced_pending = (
                old_pending
                and old_pending != self._preview.pending_manifest
                and old_pending != self._preview.previous_raw_manifest)
        if replaced_pending:
            shutil.rmtree(Path(old_pending).parent, ignore_errors=True)
        if not publish_preview:
            return

        snapshot = dict(status)
        threading.Thread(
            target=self._publish_preview_artifact,
            args=(snapshot,), name="spiral-preview-publish",
            daemon=True).start()

    def _publish_preview_artifact(self, status):
        with self.lock:
            session_id = self.session_id
            preview_generation = int(status.get("preview_generation") or 0)
            preview_manifest = status.get("preview_manifest_path")

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
            model_state = (published.raw_manifest or {}).get(
                "model_state_sha256")
            input_digest = (published.raw_manifest or {}).get("input_digest")
            input_position = (published.raw_manifest or {}).get("input_position")
            # An input update may have landed while this was publishing: a
            # surface fitted to replaced inputs stays indexed (viewable as
            # old) but never displaces the active inputs' display.
            current_digest = self._current_input_digest()
            superseded = bool(input_digest and current_digest
                              and input_digest != current_digest)
            with self.lock:
                if self.session_id == session_id and not superseded:
                    self._preview.artifact = ref
                    self._preview.error = None
                    self._preview.source_fit_iteration = (
                        published.source_fit_iteration)
                    self._preview.model_state_sha256 = model_state
                    self._preview.input_position = copy.deepcopy(input_position)
                self.status_generation += 1
            if superseded:
                self.events.append(
                    "log", f"The preview of iteration {published.source_fit_iteration} "
                    f"was fitted to inputs the fit has since replaced; it is kept "
                    f"as an old preview, not shown as current",
                    severity="warning", source="service",
                    operation="publishing_preview")
            # Remember which model state this surface belongs to, so a
            # checkpoint load that lands on it can re-show it (see
            # _note_checkpoint_state) instead of flattening it again.
            preview_index = self._published_preview_index()
            if preview_index is not None and model_state:
                try:
                    preview_index.record(
                        model_state,
                        manifest_path=published.manifest_path,
                        session_id=session_id, generation=preview_generation,
                        source_fit_iteration=published.source_fit_iteration,
                        input_digest=input_digest, input_position=input_position)
                except Exception as exc:
                    self.events.append(
                        "log", "Could not index the published preview by "
                        f"model state: {type(exc).__name__}: {exc}",
                        severity="warning", source="service",
                        operation="publishing_preview")
            self.artifacts.prune(
                "spiral-preview", session_id, PREVIEW_ARTIFACTS_KEPT,
                retain=self._retain_pinned_previews())

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
                        if self.session_id == session_id and not superseded:
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
                    if preview_generation in self._automatic_preview_generations:
                        self._automatic_previews_disabled = True
                        disable_automatic = self.session
                    else:
                        disable_automatic = None
                else:
                    disable_automatic = None
            if disable_automatic is not None:
                try:
                    disable_automatic.disable_automatic_previews(error)
                except Exception as disable_exc:
                    self.events.append(
                        "log", "Could not disable automatic previews on the "
                        f"fit workers: {type(disable_exc).__name__}: "
                        f"{disable_exc}", severity="warning", source="service")
        finally:
            next_status = None
            with self.lock:
                if self.session_id == session_id:
                    self._preview.finish(preview_generation)
                    next_status = self._preview.take_pending()
                    if next_status is not None:
                        self._preview.claim(
                            session_id, next_status["preview_generation"],
                            manifest=next_status["preview_manifest_path"],
                            source_fit_iteration=next_status.get(
                                "current_iteration"),
                            diagnostics=next_status.get(
                                "preview_diagnostics", False))
                    self.status_generation += 1
            if next_status is not None:
                threading.Thread(
                    target=self._publish_preview_artifact,
                    args=(next_status,), name="spiral-preview-publish",
                    daemon=True).start()

    def _current_input_digest(self):
        """The resident fit's input digest now (None without a session)."""
        with self.lock:
            session = self.session
        if session is None:
            return None
        try:
            return session.status().get("input_digest")
        except Exception:  # noqa: BLE001 - a closing session has no inputs
            return None

    def _published_preview_index(self):
        """The on-disk model-state index for this session's output root."""
        with self.lock:
            paths = self.session_paths
        if paths is None or not paths.output_directory:
            return None
        return PublishedPreviewIndex(paths.output_directory)

    def _retain_pinned_previews(self):
        """A prune exemption for surfaces a saved checkpoint still names."""
        preview_index = self._published_preview_index()
        if preview_index is None:
            return None
        try:
            pinned = preview_index.pinned_roots()
        except Exception:
            return None
        if not pinned:
            return None
        return lambda artifact: Path(artifact.root).resolve(
            strict=False) in pinned

    def _note_checkpoint_state(self, status):
        """React to the fitter naming the checkpoint its model now equals.

        Runs on the fitter thread, so it only records the change; the index
        write and any artifact registration happen on a background thread.
        A save pins that checkpoint's published surface against retention.
        A load or resume also re-shows that surface, when the service has
        one, so the client sees the restored model at once rather than after
        a fresh export and flatten.
        """
        state = status.get("checkpoint_state")
        if not isinstance(state, dict):
            return
        path = str(state.get("path") or "")
        digest = str(state.get("model_state_sha256") or "")
        if not path or not digest:
            return
        with self.lock:
            key = (path, digest)
            if self._checkpoint_state == key or self.session_id is None:
                return
            self._checkpoint_state = key
            session_id = self.session_id
            shown = self._preview.model_state_sha256
        threading.Thread(
            target=self._restore_published_preview,
            args=(session_id, path, digest, shown,
                  state.get("completed_iterations"),
                  status.get("input_digest")),
            name="spiral-preview-restore", daemon=True).start()

    def _restore_published_preview(self, session_id, checkpoint_path, digest,
                                   shown_digest, completed_iterations,
                                   input_digest=None):
        preview_index = self._published_preview_index()
        if preview_index is None:
            return
        try:
            preview_index.pin(checkpoint_path, digest)
            if shown_digest == digest:
                # The surface on display already is this model state.
                return
            # Only a surface published from these same inputs: one fitted to
            # other inputs is not this session's preview, however equal the
            # model state.
            entry = preview_index.lookup(digest, input_digest=input_digest)
            if entry is None:
                return
            manifest_path = Path(str(entry["manifest_path"]))
            root = manifest_path.parent
            # One directory, one artifact: re-use a registration this
            # session already holds rather than letting two prunes race
            # over the same files. A directory another session registered
            # is shared read-only; that owner never prunes it again.
            ref = self.artifacts.find("spiral-preview", root, session_id)
            if ref is None:
                unclaimed = self.artifacts.find("spiral-preview", root) is None
                with self.lock:
                    generation = self._preview.completed_generation
                ref = self.artifacts.register_directory(
                    "spiral-preview", session_id, generation, root,
                    manifest_path.name,
                    delete_root_on_prune=unclaimed, hash_workers=4)
            with self.lock:
                if self.session_id != session_id:
                    return
                self._preview.artifact = ref
                self._preview.diagnostics_artifact = None
                self._preview.error = None
                self._preview.source_fit_iteration = entry.get(
                    "source_fit_iteration")
                self._preview.model_state_sha256 = digest
                self._preview.input_position = copy.deepcopy(entry.get("input_position"))
                self.status_generation += 1
            self.events.append(
                "log",
                f"Preview restored from the surface published for "
                f"{Path(checkpoint_path).name}"
                + (f" (iteration {completed_iterations})"
                   if completed_iterations is not None else ""),
                source="service", operation="publishing_preview")
        except Exception as exc:
            print(f"SPIRAL_PREVIEW_ERROR restore: {type(exc).__name__}: "
                  f"{exc}", file=sys.stderr, flush=True)
            self.events.append(
                "log", "Could not restore the published preview for "
                f"{Path(checkpoint_path).name}: {type(exc).__name__}: {exc}",
                severity="warning", source="service",
                operation="publishing_preview")

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

    def _warn_ignored_config(self, warning):
        print(warning)
        self.events.append(
            "log", warning, severity="warning", source="service",
            operation="run")

    def run(self, request):
        autosave_on_pause = request.get("autosave_on_pause", True)
        if not isinstance(autosave_on_pause, bool):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "autosave_on_pause must be true or false")
        session = self._require_session()
        self._refuse_during_input_update("Running")
        if session.status().get("state") != SessionState.Idle:
            raise ApiError(HTTPStatus.CONFLICT,
                           "Running requires an idle session")
        expected = request.get("expected_session_revision")
        if expected != self.session_revision:
            raise ApiError(HTTPStatus.CONFLICT, "Session revision is stale")
        configuration = request.get("configuration")
        if not isinstance(configuration, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST,
                           "Running requires a complete configuration")
        configuration = filter_known_config_keys(
            configuration, self.config_catalog["defaults"],
            label="run configuration", warn=self._warn_ignored_config)
        if set(configuration) != set(self.config_catalog["defaults"]):
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
        dt_loss_schedule = _validate_dt_loss_schedule(
            request.get("dt_loss_schedule"))
        schedule = request.get("preview_schedule")
        if schedule is not None:
            if not isinstance(schedule, dict):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "preview_schedule must be an object when enabled")
            try:
                cadence = int(schedule.get("cadence_iterations"))
            except (TypeError, ValueError):
                cadence = 0
            diagnostics = schedule.get("diagnostics", False)
            if cadence < 1 or not isinstance(diagnostics, bool):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "preview_schedule requires cadence_iterations >= 1 and "
                    "a boolean diagnostics value")
            schedule = {"cadence_iterations": cadence,
                        "diagnostics": diagnostics}
        current = session.status().get("applied_config")
        request_run = self.session_request.get("run") or {}
        if current is None:
            current = Config(
                request_run.get("config") or {}
            ).as_dict()
            for key in ("z_begin", "z_end"):
                if key in request_run:
                    current[key] = request_run[key]
        # A Run carries a complete advanced configuration but never owns the
        # z window. Preserve the resident run values that Config() supplied as
        # ordinary defaults while validating the advanced fields above.
        for key in ("z_begin", "z_end"):
            configuration[key] = current.get(
                key, request_run.get(key, configuration[key]))
        # Nor does it own the launch binding: the resident values stand.
        for key in SERVICE_OWNED_CONFIG_KEYS:
            configuration[key] = current.get(key, configuration[key])
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
                "Static fit inputs cannot be changed by a run")
        run_config = changes
        with self.lock:
            current_iteration = int(
                session.status().get("current_iteration") or 0)
            self._preview_schedule = copy.deepcopy(schedule)
            self._next_preview_iteration = (
                current_iteration + schedule["cadence_iterations"]
                if schedule else None)
            self._automatic_previews_disabled = False

        run_arguments = {
            "run_config": run_config,
            "autosave_on_pause": autosave_on_pause,
            "dt_loss_schedule": dt_loss_schedule,
        }
        if schedule is not None:
            run_arguments["preview_schedule"] = copy.deepcopy(schedule)
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
            if state not in {SessionState.Idle, SessionState.Running}:
                raise ApiError(
                    HTTPStatus.CONFLICT,
                    f"Exporting a preview requires an idle or running session (state is "
                    f"{SessionState(state).name})")
            self._preview_export_active = True
            self.status_generation += 1
            session_id = self.session_id
        self._start_background(
            target=self._export_preview,
            args=(session, session_id, diagnostics),
            name="spiral-preview-export", daemon=True)
        return {**self.status(), "accepted": True}

    def _export_preview(self, session, session_id, diagnostics=False):
        """Capture one raw generation off the HTTP thread.

        The session callback hands the immutable raw snapshot to the bounded
        publication coordinator. This worker returns as soon as capture is
        complete; Lasagna, mapping, indexing, and transfer remain background
        work and cannot hold either fitting or a later Run.
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
        ``/bindings`` advertises, and ``uploaded_checkpoint`` is the path a
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
            if host not in set(self.session_checkpoints()):
                raise ApiError(
                    HTTPStatus.BAD_REQUEST,
                    "host_checkpoint must be one of the checkpoints /bindings "
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
        This is also the explicit recovery path for an errored session, where
        applying any checkpoint in place is no longer safe or even possible.

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
        self._refuse_during_input_update("Loading a checkpoint")
        path = self._resolve_load_source(request)
        with self.lock:
            session = self.session
            lifecycle_state = self._session_state
        state = (session.status().get("state")
                 if session is not None else lifecycle_state)
        if allow_rebuild and state == SessionState.Error:
            return self._rebuild_onto_checkpoint(path)
        session = self._require_session()
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
        """The durable configuration, fit volume, z window and input manifest
        a checkpoint records: ``(cfg, volume_uuid, z_range, input_manifest)``
        (``volume_uuid`` '' when its manifest does not record one).

        CPU-only and read afresh: the same bytes an escalated rebuild would
        apply, so nothing here can go stale between the refusal and the
        rebuild the client may ask for next. A file that will not load at all
        reports no configuration, which every caller treats as "no rebuild
        can help".
        """
        # Both import Torch: the service does not until it reads a checkpoint.
        from checkpoint_io import load_checkpoint_cpu
        from checkpoint_migrations import tolerate_config
        try:
            payload = load_checkpoint_cpu(path)
        except Exception:
            return None, "", None, {}
        try:
            if not isinstance(payload, dict):
                return None, "", None, {}
            cfg = payload.get("cfg")
            if isinstance(cfg, Mapping):
                # The same normalisation the preflight applies, so the
                # refusal analysis diffs the configuration a rebuild would
                # actually resume with. An invalid stored value is what no
                # rebuild can fix.
                try:
                    cfg, _ = tolerate_config(cfg, defaults=Config().as_dict())
                except ValueError:
                    cfg = None
            manifest = payload.get("input_manifest") or {}
            # z_begin/z_end are run-block settings in the service API.  A
            # checkpoint load is the one other source allowed to choose them:
            # prefer the optimisation window in its durable cfg, falling back
            # to the top-level model domain for checkpoints from before the
            # fields joined the configuration schema.
            z_source = cfg if isinstance(cfg, Mapping) \
                and "z_begin" in cfg and "z_end" in cfg else payload
            z_range = None
            if "z_begin" in z_source and "z_end" in z_source:
                try:
                    candidate = (int(z_source["z_begin"]),
                                 int(z_source["z_end"]))
                    if candidate[0] < candidate[1]:
                        z_range = candidate
                except (TypeError, ValueError):
                    pass
            return (dict(cfg) if isinstance(cfg, Mapping) else None,
                    str(manifest.get("volume_uuid") or ""), z_range,
                    dict(manifest) if isinstance(manifest, Mapping) else {})
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
        checkpoint_cfg, checkpoint_volume, _, checkpoint_inputs = \
            self._checkpoint_durable_cfg(path)
        with self.lock:
            status = self.session.status() if self.session else {}
            volume_uuid = str(
                getattr(self.session_paths, "volume_uuid", "") or "")
        live = dict(status.get("applied_config") or {})
        inputs = (None if live.get("object_store_allow_changed_inputs")
                  else self._checkpoint_input_refusal(checkpoint_inputs, status))
        # What no rebuild can fix: a checkpoint fitted in another volume, or one
        # whose configuration is not this schema's at all.
        if checkpoint_cfg is None or set(checkpoint_cfg) != set(live):
            return ApiError(
                HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}",
                payload={"reasons": reasons, "refused": True})
        if checkpoint_volume and volume_uuid \
                and checkpoint_volume != volume_uuid:
            return ApiError(
                HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}",
                payload={"reasons": reasons, "refused": True})
        if inputs is not None and inputs["legacy"]:
            # Written before inputs came from the object store: no rebuild
            # can recreate what it fitted, only the explicit opt-in resumes it.
            return ApiError(
                HTTPStatus.CONFLICT,
                "Checkpoint refused: it records no object-store inputs (written "
                "before the service read them from a store); only a rebuild with "
                "object_store_allow_changed_inputs can resume it",
                payload={"reasons": reasons, "refused": True, "inputs": inputs})
        # The binding is the service's, whatever store the checkpoint was
        # written against; it is not a difference a rebuild could apply.
        changed = {key for key, value in checkpoint_cfg.items()
                   if live.get(key) != value
                   and key not in SERVICE_OWNED_CONFIG_KEYS}
        payload = {"reasons": reasons, "stage": rebuild_stage(changed)}
        if inputs is not None:
            # A full-input rebuild rereads the store, which is what resumes
            # it if the store now holds the checkpoint's inputs; otherwise
            # only object_store_allow_changed_inputs does.
            payload.update(stage="all", inputs=inputs)
        return ApiError(
            HTTPStatus.CONFLICT, f"Checkpoint refused: {cause}", payload=payload)

    def _checkpoint_input_refusal(self, checkpoint_inputs, status):
        """How a checkpoint's object-store inputs differ from the session's.

        None when they agree (or the session's manifest is not readable, in
        which case the preflight's reasons stand alone); otherwise
        ``{"legacy", "problems", "rebuild", "allow_changed_inputs"}``: whether
        the checkpoint predates object-store inputs, what differs, the
        full-input rebuild that rereads the store, and the configuration key
        that resumes against changed inputs anyway.
        """
        remedies = {"rebuild": {"inputs": "reload"},
                    "allow_changed_inputs": "object_store_allow_changed_inputs"}
        if checkpoint_inputs.get("object_store") is None:
            return {"legacy": True, "problems": [
                "the checkpoint records no object-store inputs"], **remedies}
        path = status.get("input_manifest_path")
        try:
            with open(path, encoding="utf-8") as stream:
                current = json.load(stream)["manifest"]
        except (OSError, TypeError, ValueError, KeyError):
            return None
        from object_store_inputs import resume_problems
        problems = resume_problems(
            checkpoint_inputs, SimpleNamespace(manifest=current),
            log=lambda line: None)
        if not problems:
            return None
        return {"legacy": False, "problems": problems, **remedies}

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
        checkpoint_cfg, *_ = self._checkpoint_durable_cfg(path)
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
            project = self._session_project
        if not current or project is None:
            raise ApiError(HTTPStatus.CONFLICT,
                           "There is no session request to rebuild from")
        # The live request records the project; the rebuild sends it whole.
        current.pop("scroll", None)
        current["project"] = project.to_mapping()
        paths = dict(current.get("paths") or {})
        if self.bindings is not None:
            # ``session_request`` is the canonical manifest the service built,
            # so it contains every resolved input path.  ``rebuild``
            # deliberately accepts the narrower client request shape and
            # rejects those same service-owned paths.  Turn the canonical
            # manifest back into that shape before re-entering request
            # validation; the bindings and the project restore the inputs.
            # This matters especially after a failed checkpoint build, when
            # this is the recovery path for trying another one.
            paths = {
                key: value for key, value in paths.items()
                if key in _CLIENT_SELECTABLE_PATHS and value
            }
        paths["checkpoint"] = path
        run = dict(current.get("run") or {})
        _, _, checkpoint_z_range, _ = self._checkpoint_durable_cfg(path)
        if checkpoint_z_range is not None:
            run["z_begin"], run["z_end"] = checkpoint_z_range
        run["config"] = {}
        response = self.rebuild({**current, "paths": paths, "run": run})
        return {**response, "checkpoint_path": path}

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

        The object is absent before initialization, while it is being built,
        or after building failed. Report the lifecycle state the client is
        already polling.
        """
        with self.lock:
            if self.session is None:
                if self._session_state == SessionState.Uninitialized:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        "The fit session has not been initialized")
                if self._session_state == SessionState.Error:
                    raise ApiError(
                        HTTPStatus.CONFLICT,
                        f"The fit session failed to build: "
                        f"{self._session_error}. Rebuild with defaults or "
                        f"from a checkpoint to recover.")
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
            # The physical resolution of the preview is the session's scroll's.
            voxel_size_um = (self._session_scroll.voxel_size_um
                             if self._session_scroll is not None else None)

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
        return self.checkpoint_uploads.uploads

    def _output_root(self):
        """Output directory known before any session: the bound one."""
        if self.session_paths is not None and self.session_paths.output_directory:
            return Path(self.session_paths.output_directory)
        if self.bindings is not None and self.bindings.output_directory:
            return Path(self.bindings.output_directory)
        return None

    def _staging_root(self):
        return self.checkpoint_uploads.staging_root()

    def _checkpoint_upload_root(self):
        return self.checkpoint_uploads.checkpoint_root()

    def _upload_environment(self):
        """The whole of what the upload manager may ask this service."""
        return UploadEnvironment(
            lock=self.lock,
            output_root=self._output_root,
            session_id=lambda: self.session_id,
            active_checkpoint=self._active_checkpoint,
            allowed_kinds=("checkpoint",))

    @staticmethod
    def _file_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _active_checkpoint(self):
        with self.lock:
            return self.session_paths.checkpoint if self.session_paths else ""

    def input_upload_manager(self, upload_id):
        if upload_id not in self.checkpoint_uploads.uploads:
            raise ApiError(HTTPStatus.NOT_FOUND, "Unknown upload")
        return self.checkpoint_uploads

    def begin_upload(self, request):
        """Begin a checkpoint upload; fit inputs are never uploaded."""
        if request.get("kind") in ("patch", "fiber", "pcl"):
            raise input_upload_error()
        return {**self._base(), **self.checkpoint_uploads.begin(request)}

    def receive_upload_file(self, upload_id, relative_name, stream, length, *, offset=None):
        received = self.input_upload_manager(upload_id).receive(
            upload_id, relative_name, stream, length, offset=offset)
        return {**self._base(), "received": received, "accepted": True}

    def finalize_upload(self, upload_id):
        finalized = self.input_upload_manager(upload_id).finalize(upload_id)
        return {**self._base(), "accepted": True, "input": finalized.record}

    def gc_uploads(self):
        try:
            with self.service_use():
                self.checkpoint_uploads.collect_garbage()
        except ApiError as exc:
            if exc.status != 410:
                raise


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
        with self._teardown_lock:
            with self.lock:
                self._service_closed = True
            self._teardown()


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
    offset = None
    if "offset" in ctx.query:
        try:
            offset = int(ctx.query["offset"][-1])
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "Invalid upload offset")
    return ctx.state.receive_upload_file(
        ctx.args[0], ctx.args[1], handler.rfile, length, offset=offset)


@contextlib.contextmanager
def _fetched_lasagna_arrays(arrays, sidecar, progress):
    """``arrays`` made local for building ``sidecar``: each HTTP(S) one
    fetched whole into ``<sidecar>.arrays/<i>``, removed once the pack is
    built and kept, to resume, when building fails."""
    from lasagna_source import is_remote_location
    from remote_zarr import mirror_array

    root = Path(f"{sidecar}.arrays")
    local = []
    for index, array in enumerate(arrays):
        if not is_remote_location(array):
            local.append(array)
            continue
        if progress is not None:
            progress.begin("fetching", "Fetching remote Lasagna arrays", step=0,
                           total_steps=0, unit="chunks", detail=array)
        local.append(mirror_array(
            array, root / str(index), workers=FETCH_WORKERS,
            progress=(None if progress is None else
                      lambda done, total: progress.update(done, total_steps=total))))
    yield local
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def _lasagna_pack_kinds(config, source):
    """The Lasagna pack kinds a build of ``config`` reads from ``source``:
    the fit's (lasagna_packs_required), and normals for the link refresh
    even when their dense loss is disabled."""
    from fit_session import lasagna_packs_required, input_source_enabled
    kinds = set(lasagna_packs_required(config))
    if (input_source_enabled(config, "normals")
            and source.stores.get("normal_x") and source.stores.get("normal_y")):
        kinds.add("normals")
    return kinds


_UPLOAD_ID = r"[0-9a-f]{32}"

_LEASE_OPERATIONS = frozenset({"lease_claim", "lease_release"})

# The whole HTTP surface, declared once. Dispatch walks this table; there is
# no hand-written if-ladder, so a route's method, path, handler and retry
# semantics are visible in one place.
ROUTES = (
    Route("POST", "/session/lease/claim", "lease_claim",
          lambda ctx: ctx.state.lease().claim(ctx.handler.headers.get(LEASE_TOKEN_HEADER),
                                              ctx.body.get("command_id")),
          Idempotency.NONE, reads_body=True),
    Route("POST", "/session/lease/release", "lease_release",
          lambda ctx: ctx.state.release_lease(ctx.handler.headers.get(LEASE_TOKEN_HEADER),
                                              ctx.body.get("command_id")),
          Idempotency.NONE, reads_body=True),
    Route("GET", "/health", "health",
          lambda ctx: ctx.state.health(), Idempotency.NONE),
    Route("GET", "/configuration", "configuration",
          lambda ctx: ctx.state.configuration_catalog(), Idempotency.NONE),
    Route("GET", "/session/status", "session_status",
          lambda ctx: ctx.state.status(), Idempotency.NONE),
    Route("GET", "/events", "events", _route_events, Idempotency.NONE),
    Route("GET", "/bindings", "bindings",
          lambda ctx: ctx.state.binding_advertisement(), Idempotency.NONE),
    Route("GET", re.compile(r"/artifacts/([A-Za-z0-9._-]+)/manifest"),
          "artifact_manifest",
          lambda ctx: ctx.state.artifacts.manifest(ctx.args[0]),
          Idempotency.NONE),
    Route("GET", re.compile(r"/artifacts/([A-Za-z0-9._-]+)/files/(.+)"),
          "artifact_file", _route_artifact_file, Idempotency.NONE),

    Route("PUT", re.compile(rf"/session/inputs/({_UPLOAD_ID})/files/(.+)"),
          "upload_file", _route_upload_file, Idempotency.CONTENT),
    Route("GET", re.compile(rf"/session/inputs/({_UPLOAD_ID})"),
          "upload_status",
          lambda ctx: ctx.state.input_upload_manager(ctx.args[0]).status(ctx.args[0]),
          Idempotency.NONE),
    Route("DELETE", re.compile(rf"/session/inputs/({_UPLOAD_ID})"),
          "upload_cancel",
          lambda ctx: ctx.state.input_upload_manager(ctx.args[0]).cancel(ctx.args[0]),
          Idempotency.NONE),

    Route("POST", re.compile(rf"/session/inputs/({_UPLOAD_ID})/finalize"),
          "upload_finalize",
          lambda ctx: ctx.state.finalize_upload(ctx.args[0]),
          Idempotency.UPLOAD_ID, reads_body=True),
    Route("POST", "/session/inputs", "upload_begin",
          lambda ctx: ctx.state.begin_upload(ctx.body), Idempotency.NONE,
          reads_body=True),
    # A query, not a mutation: POST only to carry a session request.
    Route("POST", "/session/lasagna-packs", "lasagna_packs",
          lambda ctx: ctx.state.required_lasagna_packs(ctx.body), Idempotency.NONE,
          reads_body=True),
    Route("POST", "/session/initialize", "session_initialize",
          lambda ctx: ctx.state.initialize(ctx.body), Idempotency.COMMAND_ID,
          reads_body=True),
    Route("POST", "/session/rebuild", "session_rebuild",
          lambda ctx: ctx.state.rebuild(ctx.body), Idempotency.COMMAND_ID,
          reads_body=True),
    Route("POST", "/session/update-inputs", "session_update_inputs",
          lambda ctx: ctx.state.update_inputs(ctx.body), Idempotency.COMMAND_ID,
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
        self._authorise()
        path = unquote(urlparse(self.path).path).rstrip("/")
        if self.command == "POST" and path == "/session/lease/release":
            return self._dispatch_active()
        with self.server.state.service_use():
            return self._dispatch_active()

    def _dispatch_active(self):
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
        lease = state.session_lease
        if (state.bindings is not None and lease is None
                and self.command != "GET"
                and route.operation not in _LEASE_OPERATIONS):
            raise ApiError(403, "Claim the session lease first")
        if lease is not None and self.command != "GET" and route.operation not in _LEASE_OPERATIONS:
            token = self.headers.get(LEASE_TOKEN_HEADER)
            lease.require(token)
            if route.idempotency == Idempotency.COMMAND_ID:
                return state.lease_lifecycle(token, route.operation, body or {},
                                             lambda captured: route.handler(RouteContext(
                                                 self, state, args, context.query, captured)))
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
            payload.update({"error": exc.message, "http_status": int(exc.status), "details": exc.details,
                            **exc.payload})
            # The request body may not have been fully consumed; do not reuse
            # the connection after an error.
            self._send(exc.status, payload, close=True)
        except Exception as exc:
            payload = self.server.state._base()
            payload.update({"error": f"{type(exc).__name__}: {exc}", "http_status": 500})
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


class _RetiredDatasetFlag(argparse.Action):
    """``--dataset``, refused with what replaced it."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            "--dataset is retired: the service has no dataset. The scroll's "
            "facts, umbilicus, Lasagna and fiber-inference datasets come from "
            "the VC3D project; bind winding inference with --winding-inference "
            "PATH, and prepared Lasagna packs and fiber directions live in --cache")


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
    parser.add_argument("--dataset", nargs="?", action=_RetiredDatasetFlag,
                        help=argparse.SUPPRESS)
    parser.add_argument("--output", required=True,
                        help="Root for all generated state (run directories, "
                             "autosaves, previews, input revisions, upload "
                             "staging, uploaded checkpoints)")
    parser.add_argument("--winding-inference", default=None, metavar="PATH",
                        help="Winding-inference artifact (a directory holding "
                             "its manifest.json) that winding-model supervision "
                             "reads directly. Bound at launch: clients cannot "
                             "repoint it, and changing it needs a restart and a "
                             "new fit. Without it, fits needing it are refused.")
    parser.add_argument("--object-store", required=True, metavar="STORE",
                        help="Object store (.vcobj) this service reads patch, "
                             "PCL and fiber inputs from, and writes startup "
                             "link refreshes to. Opened once at startup; "
                             "clients cannot repoint it.")
    parser.add_argument("--object-store-hub", default=None, metavar="HUB",
                        help="Hub (a path, or an sftp://[user@]host/path "
                             "address) whose replica --object-store is: cloned "
                             "there when absent. Updates then link and pin hub "
                             "positions. A store that already is a replica binds "
                             "its own hub without this flag.")
    parser.add_argument("--object-store-author", default=None, metavar="NAME",
                        help="Author of the links this service's refresh "
                             "writes (default: spiral-service@<host>)")
    parser.add_argument("--require-prepared-lasagna-packs", action="store_true",
                        help="Require Lasagna packs and fiber directions prepared in "
                             "--cache (for a shared host whose cache is curated). By "
                             "default missing packs are built from accessible local "
                             "arrays, and missing fiber directions extracted from the "
                             "project's fiber-inference dataset, during input "
                             "preparation")
    # The former opt-in, now the default; older VC3D builds still pass it.
    parser.add_argument("--build-missing-lasagna-packs", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--cache", default=None,
                        help="Directory for derived host caches: prepared "
                             "Lasagna packs and fiber directions (default: "
                             "$XDG_CACHE_HOME/vc3d/spiral, i.e. "
                             "~/.cache/vc3d/spiral)")
    parser.add_argument("--service-name", default=None)
    parser.add_argument(
        "--session-name", type=parse_session_name, default=None, metavar="NAME",
        help="Stable output namespace: generated state moves to "
             "<output>/NAME, held under an exclusive lease")
    # These launch defaults are used only by explicit recovery rebuilds.
    parser.add_argument("--z-begin", type=int, default=Config().z_begin,
                        help="Default first z slice for initialization/recovery "
                             f"(default: {Config().z_begin})")
    parser.add_argument("--z-end", type=int, default=Config().z_end,
                        help="Default last z slice for initialization/recovery "
                             f"(default: {Config().z_end})")
    parser.add_argument("--config", default=None, metavar="JSON",
                        help="Advanced configuration defaults for initialization "
                             "and recovery, as a JSON object. These are the "
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
        owned = sorted(key for key in SERVICE_OWNED_CONFIG_KEYS
                       if key in startup_config)
        if owned:
            parser.error(f"--config must not set {', '.join(owned)}: "
                         "--object-store binds the service's inputs")
        try:
            Config(startup_config)
        except (ValueError, AttributeError, TypeError) as exc:
            parser.error(f"--config is not a valid configuration: {exc}")

    loopback = _is_loopback(args.bind)
    if not loopback and args.nonce:
        parser.error("--nonce is only for VC3D-owned loopback processes; use the "
                     "API key file for network binds")

    # Host paths are bound once at startup. --output owns every piece of
    # generated state.
    output_root = Path(args.output).expanduser().resolve(strict=False)
    if args.session_name:
        output_root = output_root / args.session_name
    cache_root = Path(args.cache).expanduser().resolve(strict=False) \
        if args.cache else Path(default_user_cache_dir())
    try:
        bindings = bind_service_inputs(output_root, cache_root,
                                       args.winding_inference)
    except ServiceBindingError as exc:
        parser.error(str(exc))

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

    try:
        object_store = bind_object_store(args.object_store,
                                         args.object_store_author,
                                         args.object_store_hub)
    except ObjectStoreBindingError as exc:
        print(f"Refusing to start: {exc}", file=sys.stderr, flush=True)
        return 2

    session_lease = None
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
    state = ServiceState(bindings=bindings,
                         service_name=args.service_name,
                         session_name=args.session_name or "",
                         logs=logs,
                         events=events,
                         gpu_ids=args.gpus,
                         object_store=object_store,
                         build_missing_lasagna_packs=not args.require_prepared_lasagna_packs,
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
    print(f"Spiral object store: {object_store.path} ({object_store.uuid})",
          flush=True)
    if object_store.hosted:
        print(f"Spiral object-store hub: {object_store.hub} "
              f"({object_store.hub_identity})", flush=True)
    elif object_store.positioned:
        print(f"Spiral object store shared in place; positions of "
              f"{object_store.positions}", flush=True)
    print(f"Spiral output root: {output_root}", flush=True)
    print(f"Spiral cache root: {cache_root}", flush=True)
    winding = bindings.winding_inference_status()
    print(f"Spiral winding inference: "
          + (f"{winding['path']}" + ("" if winding["available"]
                                     else f" (unavailable: {winding['error']})")
             if winding["bound"] else "not bound"), flush=True)
    if args.session_name:
        print(f"Spiral session name: {args.session_name}", flush=True)
    print(f"Spiral z-range: [{args.z_begin}, {args.z_end})", flush=True)
    print(f"SPIRAL_SERVICE_READY port={server.server_port}", flush=True)
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
