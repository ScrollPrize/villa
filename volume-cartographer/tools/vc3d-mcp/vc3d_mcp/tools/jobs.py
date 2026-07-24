"""Job inspection, waiting, and cancellation tools."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import Context

from ..core import _call, _strip_none, _wait_for_job, mcp


@mcp.tool()
async def vc3d_job_status(
    job_id: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Poll a job by id (or the latest job): state, message, console tail.

    source: optionally filter the latest job to tool, growth, lasagna, atlas,
    catalog, volume, flatten, seeding, or autosave when job_id is omitted.
    """
    return await _call(
        "job.status",
        _strip_none({"jobId": job_id, "source": source}),
    )


@mcp.tool()
async def vc3d_wait_job(
    job_id: str,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Block until an already-running job reaches a terminal state, then return
    its final job.status.

    Use this to park on a job you launched with wait=false, or one whose
    wait=true launch came back with waitTimedOut, instead of polling
    vc3d_job_status every turn. It blocks up to the same 30-minute cap as an
    inline wait, forwarding job progress where the MCP client supports it.

    Returns the terminal job.status fields (state "succeeded"/"failed",
    message, outputPath, consoleTail, ...) merged over {"jobId": ...}.
    On the 30-minute cap it returns {"jobId": ..., "waitTimedOut": true}; call
    again to keep waiting. A bridge disconnect fails the call promptly.
    """
    return await _wait_for_job(job_id, True, {"jobId": job_id}, ctx)


@mcp.tool()
async def vc3d_cancel_job(
    job_id: Optional[str] = None,
    source: Optional[str] = None,
) -> dict[str, Any]:
    """Request cancellation of a running job, dispatching to whichever subsystem
    owns it.

    Use this to stop a long-running job you launched (or found via
    vc3d_job_status / vc3d_get_state) instead of waiting for it to finish.
    Cancellation is best-effort: it asks the owning authority to stop, so poll
    vc3d_job_status afterward to confirm the job reaches a terminal state.

    Cancellable sources: "tool" (tracer.run_trace / render.tifxyz child
    processes), "atlas" (fiber-intersection search), "seeding" (seeding batch),
    "lasagna" (optimization). NOT cancellable -- these return -32010 and must
    be waited out: "growth" (segmentation.grow runs as a QtConcurrent future
    with no cancel API) and flatten jobs (self-owned, no cancel handle).

    Identify the job by either argument (at least one is required):
    job_id: the job's id (preferred; from vc3d_job_status / a launch result).
    source: the job's source ("growth" | "tool" | "atlas" | "seeding" |
    "lasagna" | ...); used to resolve the active job for that source when
    job_id is omitted.

    Returns {"cancelRequested": true, "jobId", "source", "kind"}. Errors:
    -32602 (neither job_id nor source given), -32007 (no such / not-running
    job), -32010 (the job's source has no cancel authority -- not cancellable).
    """
    return await _call(
        "job.cancel",
        _strip_none({"jobId": job_id, "source": source}),
    )
