"""Ownership and ordered, replayable mutations of a Spiral service.

MutationCoordinator runs one mutation at a time in arrival order, replays a
repeated command ID instead of acting twice, and keeps a failed recoverable
mutation's gate closed until that command is retried. ClientLease gives one
client token, and one service process (an exclusive file lock), the right to
mutate. The session lease (session_lease.SessionLease) is built from both.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import secrets
import time
import threading

from service_files import ExclusiveFileLock, FileLockUnavailable
from service_http import ApiError


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
                raise ApiError(410, "The lease is being released")
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
                    raise ApiError(410, "The lease is being released")
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
                    # This waiter is returning, so it must not retain a FIFO
                    # slot. Keep a terminal receipt for duplicate requests.
                    message = "A dataset transaction needs recovery"
                    payload = {"command_id": self._recovery}
                    receipt.update(state="rejected", error=message,
                                   error_status=409, error_details=None,
                                   error_payload=payload)
                    self._queue.remove(command_id)
                    self._condition.notify_all()
                    raise ApiError(409, message, payload=payload)
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

    def shutdown(self, callback, timeout=15):
        """Stop accepting mutations and drain the active mutation before closing."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            deadline = time.monotonic() + timeout
            while self._running is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("A mutation did not stop; the lease is retained")
                self._condition.wait(remaining)
        callback()

    def _finish(self, command_id):
        self._running = None
        if command_id in self._queue:
            self._queue.remove(command_id)
        self._condition.notify_all()


class ClientLease:
    """One service holds the file lock under ``root``; one client token owns
    the right to mutate through it."""

    def __init__(self, root, lock_name=".spiral-lease.lock"):
        self._file_lock = ExclusiveFileLock(Path(root) / lock_name)
        self._lock = threading.RLock()
        self._owner = None

    def claim(self, token):
        if not isinstance(token, str) or not token or not token.isascii():
            raise ApiError(400, "An ASCII client token is required")
        with self._lock:
            if self._owner is not None:
                self.require(token)
                return
            try:
                self._file_lock.acquire()
            except FileLockUnavailable as exc:
                raise ApiError(409, "Another service owns this lease") from exc
            self._owner = token

    def require(self, token):
        with self._lock:
            if (not isinstance(token, str) or not token.isascii()
                    or self._owner is None
                    or not secrets.compare_digest(token, self._owner)):
                raise ApiError(403, "This client does not own the lease")

    def release(self, token):
        with self._lock:
            self.require(token)
            self.close()

    def close(self):
        with self._lock:
            self._file_lock.release()
            self._owner = None
