"""General app-state / navigation tools (ping, state, screenshot, job status, workspace switch)."""

from __future__ import annotations

import base64
from typing import Any, Literal, Optional

from mcp.server.fastmcp import Context, Image

from ..core import mcp, _call, _strip_none, _wait_for_job

INLINE_SCREENSHOT_MAX_DIM = 2048


@mcp.tool()
async def vc3d_ping() -> dict[str, Any]:
    """Check the VC3D bridge is alive; returns pid and app version."""
    return await _call("ping")


@mcp.tool()
async def vc3d_get_state() -> dict[str, Any]:
    """Snapshot of VC3D: open volume package, current volume, active segment,
    viewers (ids/names), editing mode, running job. Call this first."""
    return await _call("state.get")


@mcp.tool()
async def vc3d_screenshot(
    target: str = "window",
    file_path: Optional[str] = None,
    max_dim: Optional[int] = None,
) -> Any:
    """Capture a PNG of the whole VC3D window or one viewer pane.

    Return contract depends on file_path:
    - file_path omitted: the PNG is returned inline as an MCP image content
      object (a FastMCP Image) so you can SEE the screenshot directly, rather
      than a wall of base64 text.
    - file_path set: the PNG is written to that path on disk and this returns
      the raw result dict (path/metadata; its base64 field is null). No image
      is embedded in the reply.

    target: "window" for the whole app, or a viewer ref (a "vN" registry id
    or a surface-slot name like "segmentation"/"xy plane"/"seg xz"/"seg yz").
    file_path: absolute path; when set, the PNG is written to disk and the
    result dict is returned. Omit to get the PNG back as an inline image.
    max_dim: optional downscale, longest side in pixels, aspect preserved.
    Inline captures default to 2048 pixels so their base64 response stays
    within the bridge transport budget. File captures default to full size.

    Fails -32009 if the target widget isn't currently visible (e.g. it's on a
    non-frontmost tab, such as a fiber/lasagna workspace pane while a
    different tab is active) or its captured size is degenerate (<8px on a
    side) -- rather than silently returning a meaningless near-zero-size
    image. Switch to the right tab/workspace first if you hit this.
    """
    effective_max_dim = max_dim
    if effective_max_dim is None and file_path is None:
        effective_max_dim = INLINE_SCREENSHOT_MAX_DIM

    result = await _call(
        "screenshot.capture",
        _strip_none(
            {"target": target, "filePath": file_path, "maxDim": effective_max_dim}
        ),
    )
    if file_path is not None:
        return result
    data = result.get("base64")
    if not data:
        # Bridge unexpectedly returned no image bytes for an inline capture;
        # hand back the raw dict so the caller still gets something diagnostic.
        return result
    return Image(data=base64.b64decode(data), format="png")


@mcp.tool()
async def vc3d_job_status(job_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    """Poll a job by id (or the latest job): state, message, console tail.

    source: optionally filter "the latest job" to one source ("tool" | "growth"
    | "lasagna" | "atlas") when job_id is omitted (SPEC §8.3).
    """
    return await _call("job.status", _strip_none({"jobId": job_id, "source": source}))


@mcp.tool()
async def vc3d_wait_job(job_id: str, ctx: Optional[Context] = None) -> dict[str, Any]:
    """Block until an already-running job reaches a terminal state, then return
    its final job.status.

    Use this to park on a job you launched with wait=false, or one whose
    wait=true launch came back with waitTimedOut, instead of polling
    vc3d_job_status every turn. It blocks up to the same 30-minute cap as an
    inline wait, forwarding new console output as MCP progress along the way.

    Returns the terminal job.status fields (state "succeeded"/"failed",
    message, success, outputPath, consoleTail, ...) merged over {"jobId": ...}.
    On the 30-minute cap it returns {"jobId": ..., "waitTimedOut": true}; call
    again to keep waiting. A bridge disconnect fails the call promptly.
    """
    return await _wait_for_job(job_id, True, {"jobId": job_id}, ctx)


@mcp.tool()
async def vc3d_cancel_job(
    job_id: Optional[str] = None, source: Optional[str] = None
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
    return await _call("job.cancel", _strip_none({"jobId": job_id, "source": source}))


@mcp.tool()
async def vc3d_switch_workspace(
    name: Literal["main", "lasagna", "fiber_slice"]
) -> dict[str, Any]:
    """Switch VC3D's active workspace tab. Requires a volume package to be open.

    name: "main" (the default segmentation/navigation workspace -- v1-v4's
    viewers live here), "lasagna" (the Lasagna optimization workspace), or
    "fiber_slice" (the fiber-slice workspace). Any viewers the workspace
    creates register with the ViewerManager and become targetable by
    vc3d_click / vc3d_drag / vc3d_screenshot etc. (they appear in
    vc3d_get_state's viewers list).

    "main" is the only documented way back from "lasagna"/"fiber_slice" --
    there's no automatic return. This tab is real Qt UI state that persists
    across app restarts: a freshly launched VC3D can silently open on a
    leftover "lasagna"/"fiber_slice" tab from a prior session, in which case
    vc3d_screenshot on a main-tab viewer fails -32009 (not visible) until you
    call this with name="main" first. Check vc3d_get_state's viewers, or just
    call this proactively if a main-tab capture unexpectedly fails.

    Returns {"workspace": str}. Errors: -32602 (unknown name), -32000 (no volume
    package loaded).
    """
    return await _call("workspace.switch", {"name": name})
