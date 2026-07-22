"""General app-state / navigation tools (ping, state, screenshot, job status, workspace switch).

Split out of the original monolithic ``server.py``; each tool registers on
the single shared ``mcp`` instance from ``vc3d_mcp.core``.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from ..core import mcp, _call, _strip_none

__all__ = [
    "vc3d_ping",
    "vc3d_get_state",
    "vc3d_screenshot",
    "vc3d_job_status",
    "vc3d_switch_workspace",
]


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
) -> dict[str, Any]:
    """Capture a PNG of the whole VC3D window or one viewer pane. Returns
    base64 or writes to file_path.

    target: "window" for the whole app, or a viewer ref (a "vN" registry id
    or a surface-slot name like "segmentation"/"xy plane"/"seg xz"/"seg yz").
    file_path: absolute path; when set, the PNG is written to disk and the
    result's base64 field is null. Omit to get the PNG back as base64.
    max_dim: optional downscale, longest side in pixels, aspect preserved.

    Fails -32009 if the target widget isn't currently visible (e.g. it's on a
    non-frontmost tab, such as a fiber/lasagna workspace pane while a
    different tab is active) or its captured size is degenerate (<8px on a
    side) -- rather than silently returning a meaningless near-zero-size
    image. Switch to the right tab/workspace first if you hit this.
    """
    return await _call(
        "screenshot.capture",
        _strip_none({"target": target, "filePath": file_path, "maxDim": max_dim}),
    )


@mcp.tool()
async def vc3d_job_status(job_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    """Poll a job by id (or the latest job): state, message, console tail.

    source: optionally filter "the latest job" to one source ("tool" | "growth"
    | "lasagna" | "atlas") when job_id is omitted (SPEC §8.3).
    """
    return await _call("job.status", _strip_none({"jobId": job_id, "source": source}))


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
