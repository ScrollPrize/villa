"""Seeding widget tools (rays, run/expand batches, path analysis)."""

from __future__ import annotations

from typing import Any, Optional

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _wait_for_job


@mcp.tool()
async def vc3d_seeding_set_winding_annotation_mode(active: bool) -> dict[str, Any]:
    """Toggle the Seeding widget's relative-winding annotation mode.
    Returns {"active"}."""
    return await _call("seeding.set_winding_annotation_mode", {"active": active})


@mcp.tool()
async def vc3d_seeding_preview_rays() -> dict[str, Any]:
    """Seeding: preview the radial rays from the current focus point (requires a
    focus POI). Fire-and-forget; returns {"requested": true}."""
    return await _call("seeding.preview_rays", {})


@mcp.tool()
async def vc3d_seeding_cast_rays() -> dict[str, Any]:
    """Seeding: cast rays and detect intensity peaks (runs asynchronously in a
    background thread; requires a focus POI in point mode). Returns
    {"requested": true}."""
    return await _call("seeding.cast_rays", {})


@mcp.tool()
async def vc3d_seeding_reset_points() -> dict[str, Any]:
    """Seeding: clear collected peaks/seeds and reset the widget. Returns
    {"reset": true}."""
    return await _call("seeding.reset_points", {})


@mcp.tool()
async def vc3d_seeding_run(
    wait: bool = False, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Seeding: "seed the volume" -- grow one segment per point in the Seeding
    widget's currently selected source collection by spawning a
    vc_grow_seg_from_seed child process per point (bounded concurrency by the
    widget's "Processes" setting). Async: a source:"seeding" job.

    Preconditions the bridge validates: a volume package + current volume are
    loaded; the Seeding widget's source-collection combo has a non-empty
    selection with points (populate it first, e.g. vc3d_commit_points into a
    collection, or the widget's Cast Rays flow); seed.json and a segmentation
    paths directory exist in the package; vc_grow_seg_from_seed is on disk next
    to VC3D.

    Returns {"jobId", "kind": "seeding.run", "source": "seeding", "points",
    "total"}. Poll vc3d_job_status (source="seeding") or pass wait=true; job
    progress is "run <completed>/<total>" as each child finishes. Cancel with
    vc3d_seeding_cancel.

    Errors: -32000 (no vpkg), -32001 (no volume), -32004 (a seeding batch is
    already running), -32005 (no source collection / no points / launch failure,
    data.detail), -32006 (vc_grow_seg_from_seed not found), -32007
    (data.kind:"file" -- seed.json / paths directory missing), -32010 (seeding
    widget unavailable).

    wait: if true (MCP-server-side only), block until the job finishes (30-minute
    cap) and return the terminal job.status inline.
    """
    result = await _call("seeding.run", {})
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_seeding_expand(
    wait: bool = False, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Seeding: run the iterative seed-expansion pass -- spawn vc_grow_seg_from_seed
    with expand.json for N iterations (the widget's "Expansion iterations"
    setting), bounded concurrency by the "Processes" setting. Async: a
    source:"seeding" job. Mutually exclusive with vc3d_seeding_run (both are the
    single "seeding" source).

    Preconditions: a volume package + current volume are loaded; expand.json and
    a segmentation paths directory exist in the package; vc_grow_seg_from_seed is
    on disk. (No source point collection is needed -- expansion is
    iteration-count-driven, not point-driven.)

    Returns {"jobId", "kind": "seeding.expand", "source": "seeding", "iterations",
    "total"}. Poll vc3d_job_status (source="seeding") or pass wait=true; progress
    is "expand <completed>/<total>". Cancel with vc3d_seeding_cancel.

    Errors: -32000, -32001, -32004 (a seeding batch is already running), -32005
    (launch failure), -32006 (executable not found), -32007 (data.kind:"file" --
    expand.json / paths directory missing), -32010.

    wait: if true (MCP-server-side only), block until the job finishes (30-minute
    cap) and return the terminal job.status inline.
    """
    result = await _call("seeding.expand", {})
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_seeding_cancel() -> dict[str, Any]:
    """Seeding: cancel the running run/expand batch (the Cancel button). Bounded
    synchronous teardown -- each running child is terminated (SIGTERM, then
    SIGKILL after ~1s). The source:"seeding" job is resolved (finished,
    success=false) by the time this returns. Returns {"cancelRequested": true}.

    Errors: -32007 (data.kind:"job" -- no seeding batch running), -32010."""
    return await _call("seeding.cancel", {})


@mcp.tool()
async def vc3d_seeding_analyze_paths() -> dict[str, Any]:
    """Seeding: analyze intensity peaks along the paths drawn in the widget's
    Draw mode, collecting seed candidates into the "seeding_peaks" collection.
    Synchronous in-process compute (NOT a job) -- returns once analysis
    completes. Requires paths to have been drawn first (Draw mode); with no
    paths this fails -32007 (data.kind:"path"). Returns {"analyzed": true,
    "paths": <count>, "peaks": <count>}.

    Errors: -32000, -32001, -32007 (data.kind:"path" -- no drawn paths), -32010."""
    return await _call("seeding.analyze_paths", {})
