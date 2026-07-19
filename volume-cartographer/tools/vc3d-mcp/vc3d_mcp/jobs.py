"""Local tracker for `job.progress` notifications (SPEC.md section 3.18).

The bridge itself is the source of truth for job state (via the `job.status`
RPC, SPEC.md 3.17) -- this tracker exists only so the MCP server can:

  * implement the `wait: true` convenience param on `vc3d_grow_segment` /
    `vc3d_grow_patch_from_seed` (block until a `job.progress` notification
    with `phase:"finished"` arrives for that job, instead of polling), and
  * keep a rolling console tail locally in case a caller wants it without
    a round trip, mirroring (not replacing) the `consoleTail` the bridge
    already returns from `job.status`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

CONSOLE_TAIL_MAX = 50


@dataclass
class JobRecord:
    job_id: str
    kind: str | None = None
    state: str = "running"  # "running" | "succeeded" | "failed"
    message: str | None = None
    success: bool | None = None
    output_path: str | None = None
    console_tail: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    finished_event: asyncio.Event = field(default_factory=asyncio.Event)

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


class JobTracker:
    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self.latest_job_id: str | None = None

    def _get_or_create(self, job_id: str) -> JobRecord:
        record = self._jobs.get(job_id)
        if record is None:
            record = JobRecord(job_id=job_id)
            self._jobs[job_id] = record
        return record

    def on_progress(self, params: dict[str, Any]) -> None:
        job_id = params.get("jobId")
        if not job_id:
            return
        record = self._get_or_create(job_id)
        self.latest_job_id = job_id

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

        record.updated_at = time.time()

    def get(self, job_id: str) -> JobRecord | None:
        return self._jobs.get(job_id)

    def latest(self) -> JobRecord | None:
        if self.latest_job_id is None:
            return None
        return self._jobs.get(self.latest_job_id)

    async def wait_finished(self, job_id: str, timeout: float) -> JobRecord | None:
        """Block until `job_id` reaches phase:"finished", or `timeout` elapses.

        Returns the record on completion, or None on timeout. If no
        notification for `job_id` has ever been seen yet (e.g. it hasn't
        arrived before this call), a record is created so the wait has
        something to park on.
        """
        record = self._get_or_create(job_id)
        try:
            await asyncio.wait_for(record.finished_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return record
