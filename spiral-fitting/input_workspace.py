"""Revisioned editing contract, independent of transport and resident fitting.

Content holds immutable JSON (either a document or a manifest referring to
immutable uploaded bytes). The catalog never owns a mutable editor path.
Revision numbers are per logical UUID; PCL collection numbers belong to the
serialization source and are allocated separately. Mutating callers must use
one MutationCoordinator for acceptance, application, publication and rebuild.

This is the contract for the revisioned protocol. The legacy API's ephemeral
ledger is not an adapter for this model: it cannot express these guarantees.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import secrets
import threading
from uuid import UUID

from service_files import ExclusiveFileLock, FileLockUnavailable
from service_http import ApiError


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Content:
    encoded: str

    @classmethod
    def from_json(cls, value):
        return cls(_canonical(value))

    def json(self):
        return json.loads(self.encoded)

    @property
    def digest(self):
        return hashlib.sha256(self.encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InputIdentity:
    id: str
    kind: str
    source: str
    role: str | None = None
    collection_id: int | None = None

    def __post_init__(self):
        if str(UUID(self.id)) != self.id:
            raise ValueError("Logical input ids must be canonical UUIDs")
        if self.kind not in {"pcl", "patch", "fiber"} or not self.source:
            raise ValueError("An input needs a kind and persistence source")
        if self.collection_id is not None and (
                self.kind != "pcl" or type(self.collection_id) is not int
                or self.collection_id < 0):
            raise ValueError("Only PCLs have nonnegative numeric collection ids")


@dataclass(frozen=True)
class Revision:
    id: str
    number: int
    content: Content | None


@dataclass(frozen=True)
class Change:
    identity: InputIdentity
    expected: int
    content: Content | None  # None is a staged deletion, never missing content.


@dataclass(frozen=True)
class Entry:
    identity: InputIdentity
    base: Revision | None
    revisions: tuple[Revision, ...]
    applied: int = 0
    persisted: int = 0
    applied_history: frozenset[int] = frozenset()
    errors: tuple[tuple[int, str, str], ...] = ()

    @property
    def accepted(self):
        return len(self.revisions)

    @property
    def current(self):
        return self.revisions[-1]

    @property
    def deleted(self):
        return self.current.content is None

    def status(self):
        return {
            "id": self.identity.id, "kind": self.identity.kind,
            "source": self.identity.source, "role": self.identity.role,
            "collection_id": self.identity.collection_id,
            "accepted_revision": self.accepted, "applied_revision": self.applied,
            "persisted_revision": self.persisted, "deleted": self.deleted,
            "content": self.current.content.json() if self.current.content else None,
            "errors": [{"revision": number, "stage": stage, "message": message}
                       for number, stage, message in self.errors],
            "can_restore": (self.deleted and self.accepted != self.persisted
                            and any(revision.content for revision in self.revisions)),
        }


class Catalog:
    def __init__(self):
        self._entries = {}
        self._next_collection_ids = {}
        self._lock = threading.RLock()
        self._accepting = False

    def register_base(self, identity, content):
        """Seed all dataset entries before accepting any workspace edits."""
        with self._lock:
            if self._accepting:
                raise ValueError("Baseline registration must precede edits")
            if identity.id in self._entries or content is None:
                raise ValueError("Duplicate or empty baseline input")
            if identity.kind == "pcl":
                if identity.collection_id is None:
                    raise ValueError("Baseline PCLs must retain their numeric ids")
                if any(entry.identity.source == identity.source
                       and entry.identity.collection_id == identity.collection_id
                       for entry in self._entries.values()):
                    raise ValueError("Duplicate collection id in one source")
                self._next_collection_ids[identity.source] = max(
                    self._next_collection_ids.get(identity.source, 0),
                    identity.collection_id + 1)
            revision = Revision(identity.id, 1, content)
            self._entries[identity.id] = Entry(
                identity, revision, (revision,), 1, 1, frozenset({1}))

    def reserve_collection_ids(self, source, numbers):
        """External additions also advance the never-reused source counter."""
        with self._lock:
            self._next_collection_ids[source] = max(
                self._next_collection_ids.get(source, 0), max(numbers, default=-1) + 1)

    def entry(self, input_id):
        with self._lock:
            return self._entries[input_id]

    def entries(self):
        with self._lock:
            return tuple(self._entries.values())

    def status(self):
        with self._lock:
            return [entry.status() for entry in self._entries.values()]

    def accept(self, changes):
        """Validate the entire batch, then publish all accepted revisions.

        Run domain validation before calling this method. A conflict neither
        consumes a revision/collection id nor accepts an unrelated batch row.
        """
        changes = tuple(changes)
        if not changes or len({c.identity.id for c in changes}) != len(changes):
            raise ApiError(400, "A batch needs distinct logical input ids")
        with self._lock:
            entries = dict(self._entries)
            counters = dict(self._next_collection_ids)
            conflicts, accepted = [], []
            for change in changes:
                item = change.identity
                previous = entries.get(item.id)
                expected = previous.accepted if previous else 0
                if type(change.expected) is not int or change.expected != expected:
                    conflicts.append({"id": item.id, "accepted_revision": expected})
                    continue
                if previous and item != previous.identity:
                    raise ApiError(409, "An input's persistence identity cannot change")
                if previous is None:
                    if change.content is None:
                        raise ApiError(400, "Cannot delete an unknown input")
                    if item.kind == "pcl":
                        if item.collection_id is not None:
                            raise ApiError(400, "New collection ids are assigned by the workspace")
                        next_id = counters.get(item.source, 0)
                        item = replace(item, collection_id=next_id)
                        counters[item.source] = next_id + 1
                    previous = Entry(item, None, ())
                revision = Revision(item.id, expected + 1, change.content)
                entries[item.id] = replace(
                    previous, revisions=previous.revisions + (revision,))
                accepted.append(revision)
            if conflicts:
                raise ApiError(409, "Accepted revisions changed",
                               payload={"conflicts": conflicts})
            self._entries = entries
            self._next_collection_ids = counters
            self._accepting = True
            return tuple(accepted)

    def _validate_selection(self, revisions):
        if len({r.id for r in revisions}) != len(revisions):
            raise ApiError(400, "Select one revision per logical input")
        for revision in revisions:
            entry = self._entries.get(revision.id)
            if (entry is None or not 0 < revision.number <= entry.accepted
                    or entry.revisions[revision.number - 1] != revision):
                raise ApiError(409, "Unknown input revision")

    def mark_applied(self, revisions):
        revisions = tuple(revisions)
        with self._lock:
            self._validate_selection(revisions)
            for revision in revisions:
                entry = self._entries[revision.id]
                self._entries[revision.id] = replace(
                    entry, applied=max(entry.applied, revision.number),
                    applied_history=entry.applied_history | {revision.number},
                    errors=tuple(error for error in entry.errors
                                 if error[:2] != (revision.number, "apply")))

    def mark_persisted(self, revisions):
        revisions = tuple(revisions)
        with self._lock:
            self._validate_selection(revisions)
            for revision in revisions:
                if revision.number not in self._entries[revision.id].applied_history:
                    raise ApiError(409, "Only successfully applied revisions can be committed")
            for revision in revisions:
                entry = self._entries[revision.id]
                self._entries[revision.id] = replace(
                    entry, persisted=max(entry.persisted, revision.number),
                    errors=tuple(error for error in entry.errors
                                 if error[:2] != (revision.number, "commit")))

    def record_error(self, revisions, stage, message):
        if stage not in {"apply", "commit"}:
            raise ValueError("Expected an apply or commit stage")
        revisions = tuple(revisions)
        with self._lock:
            self._validate_selection(revisions)
            for revision in revisions:
                entry = self._entries[revision.id]
                errors = tuple(error for error in entry.errors
                               if error[:2] != (revision.number, stage))
                self._entries[revision.id] = replace(
                    entry, errors=errors + ((revision.number, stage, str(message)),))

    def reset_applied(self):
        """A new resident fit generation does not reset the editing workspace."""
        with self._lock:
            self._entries = {key: replace(entry, applied=0,
                                         applied_history=frozenset())
                             for key, entry in self._entries.items()}

    def desired(self, enabled=lambda entry: True):
        """Desired content in stable catalog order, independent of role toggles."""
        with self._lock:
            return tuple(entry.current for entry in self._entries.values()
                         if enabled(entry))


class MutationCoordinator:
    """FIFO mutations with lifetime receipts and resumable publication gating.

    The callable receives a detached payload. Read-only outcome requests stay
    responsive during long operations. A recoverable callable MUST resume its
    retained transaction when invoked again: it must not start publication
    from scratch. Only the recovery command can run until it succeeds.
    """

    def __init__(self):
        self._condition = threading.Condition(threading.RLock())
        self._receipts = {}
        self._queue = []
        self._running = None
        self._recovery = None
        self._closed = False

    def outcome(self, command_id):
        with self._condition:
            receipt = self._receipts.get(command_id)
            if receipt is None:
                raise ApiError(404, "Unknown command")
            return copy.deepcopy({key: value for key, value in receipt.items()
                                  if key != "signature"})

    def execute(self, command_id, operation, payload, callback, *, recoverable=False):
        if not isinstance(command_id, str) or not command_id.strip():
            raise ApiError(400, "A nonempty command id is required")
        signature = _canonical([operation, payload, bool(recoverable)])
        captured = json.loads(_canonical(payload))
        with self._condition:
            if self._closed:
                raise ApiError(410, "The editing workspace is closed")
            receipt = self._receipts.get(command_id)
            if receipt and receipt["signature"] != signature:
                raise ApiError(409, "Command id was reused with different content")
            if self._recovery not in (None, command_id):
                raise ApiError(409, "A dataset transaction needs recovery",
                               payload={"command_id": self._recovery})
            if receipt is None:
                receipt = {"command_id": command_id, "operation": operation,
                           "signature": signature, "state": "queued"}
                self._receipts[command_id] = receipt
                self._queue.append(command_id)
            while True:
                if self._closed:
                    raise ApiError(410, "The editing workspace is closed")
                state = receipt["state"]
                if state == "completed":
                    return copy.deepcopy(receipt["result"])
                if state in {"rejected", "failed"}:
                    if "error_status" in receipt:
                        raise ApiError(receipt["error_status"], receipt["error"],
                                       copy.deepcopy(receipt["error_details"]),
                                       copy.deepcopy(receipt["error_payload"]))
                    raise RuntimeError(receipt["error"])
                if self._recovery not in (None, command_id):
                    # Remain queued; a client may retry after recovery.
                    raise ApiError(409, "A dataset transaction needs recovery",
                                   payload={"command_id": self._recovery})
                if self._running is None and (
                        self._recovery == command_id
                        or self._queue and self._queue[0] == command_id):
                    self._running = command_id
                    receipt["state"] = "running"
                    break
                self._condition.wait()
        try:
            result = callback(captured)
        except BaseException as exc:
            with self._condition:
                # Recoverable transactions include scoped external conflicts
                # after partial publication; those must keep the gate closed.
                retry = isinstance(exc, Exception) and (
                    recoverable(exc) if callable(recoverable) else recoverable)
                receipt.update(state=("recovery_required" if retry else
                                      "rejected" if isinstance(exc, ApiError) else "failed"),
                               error=str(exc))
                # Do not retain exception tracebacks: they can keep entire
                # prepared geometry/device allocations alive for the lifetime
                # of a command receipt.
                if isinstance(exc, ApiError):
                    receipt.update(error_status=exc.status,
                                   error_details=copy.deepcopy(exc.details),
                                   error_payload=copy.deepcopy(exc.payload))
                if retry:
                    self._recovery = command_id
                self._finish(command_id)
            raise
        with self._condition:
            receipt.update(state="completed", result=copy.deepcopy(result))
            for key in ("error", "error_status", "error_details", "error_payload"):
                receipt.pop(key, None)
            if self._recovery == command_id:
                self._recovery = None
            self._finish(command_id)
            return copy.deepcopy(result)

    def shutdown(self, callback):
        """Stop accepting mutations and drain the active mutation before closing."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._running is not None:
                self._condition.wait()
        callback()

    def _finish(self, command_id):
        self._running = None
        if command_id in self._queue:
            self._queue.remove(command_id)
        self._condition.notify_all()


class WorkspaceLease:
    """One service holds the file lock; one client token owns its workspace."""

    def __init__(self, dataset_root):
        self._file_lock = ExclusiveFileLock(Path(dataset_root) / ".spiral-edit.lock")
        self._lock = threading.RLock()
        self._owner = None

    def claim(self, token):
        if not isinstance(token, str) or not token or not token.isascii():
            raise ApiError(400, "An ASCII editing token is required")
        with self._lock:
            if self._owner is not None:
                self.require(token)
                return
            try:
                self._file_lock.acquire()
            except FileLockUnavailable as exc:
                raise ApiError(409, "Another service owns dataset editing") from exc
            self._owner = token

    def require(self, token):
        with self._lock:
            if (not isinstance(token, str) or not token.isascii()
                    or self._owner is None
                    or not secrets.compare_digest(token, self._owner)):
                raise ApiError(403, "This client does not own the editing workspace")

    def release(self, token):
        with self._lock:
            self.require(token)
            self.close()

    def close(self):
        with self._lock:
            self._file_lock.release()
            self._owner = None
