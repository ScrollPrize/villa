"""Local tracker for `job.progress` notifications (SPEC.md section 3.18).

The bridge itself is the source of truth for job state (via the `job.status`
RPC, SPEC.md 3.17). This tracker does NOT back or replace `job.status`; it
exists only so the MCP server can:

  * implement the `wait: true` convenience param on the long-running tools
    (block until a `job.progress` notification with `phase:"finished"` arrives
    for that job, instead of polling),
  * forward `phase:"output"` lines to the MCP client as best-effort progress
    while a `wait: true` call is blocked (see ``core._wait_for_job``), and
  * keep a rolling console tail / terminal record locally as a fallback if the
    authoritative ``job.status`` RPC races with process shutdown.

Progress plumbing (best-effort, per the stabilization decision): each record
carries a bounded, ordered output buffer and a per-record wake event so a
waiter can (a) replay the retained buffered tail, then (b) stream new updates
in order, with no lost wakeup between reading state and parking. A bridge
disconnect wakes every waiter with an error so a `wait: true` call fails
promptly instead of blocking to the 30-minute cap.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

CONSOLE_TAIL_MAX = 50
# Retained tail of `phase:"output"` lines available for replay to a waiter that
# registers after some output has already arrived. Bounded for memory.
OUTPUT_LOG_MAX = 200
# Upper bound on tracked job records; terminal records are pruned first.
MAX_RECORDS = 256


@dataclass
class JobRecord:
    job_id: str
    kind: str | None = None
    state: str = "running"  # "running" | "succeeded" | "failed"
    message: str | None = None
    success: bool | None = None
    output_path: str | None = None
    console_tail: list[str] = field(default_factory=list)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)
    # --- progress plumbing (best-effort) ---
    # Ordered (seq, line) buffer of phase:"output" lines; bounded tail for replay.
    outputs: "deque[tuple[int, str]]" = field(
        default_factory=lambda: deque(maxlen=OUTPUT_LOG_MAX)
    )
    output_seq: int = 0
    # Wake event, replaced+set on every update so a parked waiter re-checks state
    # without losing a wakeup (see JobTracker._notify).
    wake: asyncio.Event = field(default_factory=asyncio.Event)

    def as_dict(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "kind": self.kind,
            "state": self.state,
            "message": self.message,
            "success": self.success,
            "outputPath": self.output_path,
            "consoleTail": list(self.console_tail),
        }

    def outputs_after(self, after_seq: int) -> list[tuple[int, str]]:
        """Retained output lines with a sequence number greater than `after_seq`."""
        return [(seq, line) for (seq, line) in self.outputs if seq > after_seq]


class JobTracker:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        # Set by fail_all() on bridge disconnect so parked/late waiters raise
        # promptly instead of blocking to the wait cap. Cleared on reconnect.
        self._error: Exception | None = None

    @property
    def error(self) -> Exception | None:
        return self._error

    def register(self, job_id: str) -> JobRecord:
        """Get (or create) the record a waiter will park on."""
        return self._get_or_create(job_id)

    def _get_or_create(self, job_id: str) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            record = JobRecord(job_id=job_id)
            self._jobs[job_id] = record
            self._prune()
        return record

    @staticmethod
    def _notify(record: JobRecord) -> None:
        # Swap in a fresh wake event and set the old one. A waiter captures
        # record.wake BEFORE reading state, so any update that lands between its
        # state read and its park sets the very event it is about to await --
        # no lost wakeup.
        old = record.wake
        record.wake = asyncio.Event()
        old.set()

    def _prune(self) -> None:
        if len(self._jobs) <= MAX_RECORDS:
            return
        # Drop the oldest terminal records first (insertion order preserved).
        for job_id, record in list(self._jobs.items()):
            if record.finished_event.is_set():
                del self._jobs[job_id]
                if len(self._jobs) <= MAX_RECORDS:
                    break

    def on_progress(self, params: dict[str, Any]) -> None:
        job_id = params.get("jobId")
        if not job_id:
            return
        record = self._get_or_create(job_id)

        kind = params.get("kind")
        if kind:
            record.kind = kind

        phase = params.get("phase")
        message = params.get("message")

        if phase == "started":
            record.state = "running"
            if message:
                record.message = message
        elif phase == "output":
            if message:
                record.message = message
                for line in str(message).splitlines() or [str(message)]:
                    record.console_tail.append(line)
                    record.output_seq += 1
                    record.outputs.append((record.output_seq, line))
                del record.console_tail[:-CONSOLE_TAIL_MAX]
        elif phase == "finished":
            success = params.get("success")
            record.success = success
            record.state = "succeeded" if success else "failed"
            if message:
                record.message = message
            if "outputPath" in params:
                record.output_path = params.get("outputPath")
            record.finished_event.set()

        self._notify(record)

    def fail_all(self, exc: Exception) -> None:
        """Bridge disconnect: record the error and wake every parked waiter.

        New waiters registered while `_error` is set observe it immediately.
        """
        self._error = exc
        for record in self._jobs.values():
            self._notify(record)

    def reset_error(self) -> None:
        """Clear a prior disconnect error (called on a fresh connection)."""
        self._error = None

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    async def wait_finished(self, job_id: str, timeout: float) -> JobRecord | None:
        """Block until `job_id` reaches phase:"finished", or `timeout` elapses.

        Returns the record on completion, or None on timeout. Raises the
        tracker's disconnect error if the bridge drops while waiting. If no
        notification for `job_id` has been seen yet, a record is created so the
        wait has something to park on.
        """
        record = self._get_or_create(job_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            wake = record.wake  # capture BEFORE reading state (no lost wakeup)
            if self._error is not None:
                raise self._error
            if record.finished_event.is_set():
                return record
            remaining = deadline - loop.time()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(wake.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
