"""Segment listing/activation, editing, growth, tags, push/pull, and run-trace tools."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _wait_for_job, _strip_none


class Point3D(TypedDict):
    x: float
    y: float
    z: float


@mcp.tool()
async def vc3d_list_segments() -> dict[str, Any]:
    """List the surfaces of the open volume in the store surfaces are kept in
    (the project's, or the store chosen for surfaces), Open Data catalog
    surfaces of a sample's project among them.

    Each entry has "id" (the surface's store UUID, used by every other segment
    tool), "name", record "tags", "catalog" (an Open Data catalog surface),
    "bound" (bound to the open volume) and "active". "loading" is true while the
    listing is still being read; retry for the complete list."""
    return await _call("segments.list", {})


@mcp.tool()
async def vc3d_activate_segment(segment_id: str) -> dict[str, Any]:
    """Make a surface the active editing target (the programmatic equivalent of
    clicking it in the surface list). Required before vc3d_enable_editing /
    vc3d_grow_segment and after any segment switch.

    segment_id: a surface UUID as returned by vc3d_list_segments. A catalog
    surface whose geometry is still remote is downloaded first (the call blocks
    meanwhile)."""
    return await _call("segments.activate", {"segmentId": segment_id})


@mcp.tool()
async def vc3d_delete_segment(segment_id: str, confirm: bool = False) -> dict[str, Any]:
    """Retire a surface in its store: it leaves the surface list (the
    store's history keeps it).

    You MUST pass confirm=True; the default confirm=False is rejected -32602
    ({param:"confirm"}) as a safety guard. Deleting the active surface is allowed
    (the active slot is cleared first), but deletion is refused while
    segmentation editing mode is enabled -- turn it off with
    vc3d_enable_editing(False) first -- and in a read-only store.

    segment_id: a surface UUID as returned by vc3d_list_segments.
    confirm: must be True to proceed.

    Returns {"deleted": [segmentId]}. Errors: -32602 (confirm not True),
    -32007 (unknown surface), -32009 (read-only store), -32004/-32010 (cannot
    delete while editing)."""
    return await _call("segments.delete", {"segmentId": segment_id, "confirm": confirm})


@mcp.tool()
async def vc3d_rename_segment(segment_id: str, new_name: str) -> dict[str, Any]:
    """Rename a surface in its store, the headless twin of the surface
    panel's rename action. The UUID (its identity) stays.

    segment_id: a surface UUID as returned by vc3d_list_segments.
    new_name: the new name; it must differ from the current one.

    Returns {"id", "name"}. Errors: -32602 (blank or unchanged new_name), -32007
    (unknown surface), -32009 (read-only store), -32010 (store refused the edit)."""
    return await _call("segments.rename", {"segmentId": segment_id, "newName": new_name})


@mcp.tool()
async def vc3d_enable_editing(enabled: bool) -> dict[str, Any]:
    """Turn segmentation editing mode on/off for the active segment.

    A catalog surface is edited in place like any other. Fails -32009 when the
    surface cannot be edited (a read-only store, or a past revision is being
    viewed)."""
    return await _call("segmentation.enable_editing", {"enabled": enabled})


@mcp.tool()
async def vc3d_save_segment(
    wait: bool = True, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Write the active segment's pending edits to the store now.

    Segment edits are normally flushed by a periodic autosave; this forces that
    flush immediately (the programmatic equivalent of not waiting for the timer).

    If nothing is dirty and no save is already in flight, this is a no-op and
    returns immediately with jobId=null and state="idle". Otherwise the flush runs
    as an "autosave"-source job: wait defaults to true (block until the save job
    finishes and return the terminal job.status inline); pass wait=false to get the
    jobId back immediately and poll vc3d_job_status yourself.
    """
    result = await _call("segmentation.save", {})
    job_id = result.get("jobId") if isinstance(result, dict) else None
    if not job_id:
        return result  # idle: nothing to flush, nothing to wait on
    return await _wait_for_job(job_id, wait, result, ctx)


@mcp.tool()
async def vc3d_grow_segment(
    steps: int,
    method: Literal["tracer", "corrections"] = "tracer",
    direction: Literal["all", "up", "down", "left", "right", "fill"] = "all",
    inpaint_only: bool = False,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Grow the active segmentation surface. Async: returns a jobId.

    method: "tracer" | "corrections". ("manual_add" is not a
    growth method -- it is an interactive editing mode; passing it is rejected
    with -32009. Use the segmentation.manual_add.* RPCs instead.)
    direction: "all" | "up" | "down" | "left" | "right" | "fill".
    steps: number of growth steps, >= 1.
    wait: if true (MCP-server-side only, not part of the underlying RPC),
    block until the job finishes (30-minute cap) and return the terminal
    job.status inline instead of just the jobId.
    """
    result = await _call(
        "segmentation.grow",
        {
            "method": method,
            "direction": direction,
            "steps": steps,
            "inpaintOnly": inpaint_only,
        },
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_grow_surface_from_seed(
    seed: Point3D,
    volume_id: Optional[str] = None,
    iterations: int = 200,
    min_area_cm: float = 0.002,
    output_tag: str = "paths",
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Create a new surface by growing it from a 3D seed point
    (headless GrowPatch). Async: returns a jobId; the new surface's UUID is the
    finished job's output.

    seed: volume-space {"x","y","z"} seed point.
    volume_id: vpkg volume id; default is the current volume.
    iterations: 1..100000, default 200 ("generations" in the underlying tool).
    min_area_cm: minimum surface area in cm^2, >= 0.
    output_tag: tag of the new surface (default "paths").
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.
    """
    result = await _call(
        "segmentation.grow_surface_from_seed",
        _strip_none(
            {
                "seed": seed,
                "volumeId": volume_id,
                "iterations": iterations,
                "minAreaCm": min_area_cm,
                "outputTag": output_tag,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)


@mcp.tool()
async def vc3d_set_segment_tag(
    segment_id: str,
    tag: Literal["approved", "defective", "reviewed", "inspect"],
    enabled: bool,
) -> dict[str, Any]:
    """Set (or clear) a review tag on a surface: a record tag of the surface in
    its store (-32009 in a read-only store). tag is one of
    "approved" | "defective" | "reviewed" | "inspect" (there is no "revisit"
    tag). Returns {"segmentId", "tag", "enabled"}."""
    return await _call(
        "tags.set", {"segmentId": segment_id, "tag": tag, "enabled": enabled}
    )


@mcp.tool()
async def vc3d_push_pull_set_config(
    start: Optional[float] = None,
    stop: Optional[float] = None,
    step: Optional[float] = None,
    low: Optional[float] = None,
    high: Optional[float] = None,
    blur_radius: Optional[int] = None,
    compute_scale: Optional[int] = None,
    per_vertex_limit: Optional[float] = None,
    per_vertex: Optional[bool] = None,
) -> dict[str, Any]:
    """Read-modify-write the Gaussian push/pull (alpha) config. Any omitted
    field keeps its current value; the result is the full sanitized effective
    config {start, stop, step, low, high, blurRadius, computeScale,
    perVertexLimit, perVertex}."""
    return await _call(
        "segmentation.push_pull.set_config",
        _strip_none(
            {
                "start": start,
                "stop": stop,
                "step": step,
                "low": low,
                "high": high,
                "blurRadius": blur_radius,
                "computeScale": compute_scale,
                "perVertexLimit": per_vertex_limit,
                "perVertex": per_vertex,
            }
        ),
    )


@mcp.tool()
async def vc3d_push_pull_start(
    direction: Literal["push", "pull"], alpha: Optional[bool] = None
) -> dict[str, Any]:
    """Start Gaussian push (direction="push") or pull (direction="pull") at the
    module's last recorded pointer position. Position the cursor first with a
    buttonless hover drag (vc3d_drag with button="none") ending on the target
    vertex, then start, wait, and stop. Requires editing enabled + an active
    edit session. Returns {"active"} (false when there is no valid hover
    target)."""
    return await _call(
        "segmentation.push_pull.start",
        _strip_none({"direction": direction, "alpha": alpha}),
    )


@mcp.tool()
async def vc3d_push_pull_stop() -> dict[str, Any]:
    """Stop all active push/pull operations. Returns {"stopped": true}."""
    return await _call("segmentation.push_pull.stop", {})


@mcp.tool()
async def vc3d_run_trace(
    segment_id: str,
    param_overrides: Optional[dict[str, Any]] = None,
    omp_threads: Optional[int] = None,
    source_tag: str = "paths",
    output_tag: str = "traces",
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Run the patch-stitching tracer (vc_grow_seg_from_segments) on a segment,
    the headless twin of the "Run Trace" context-menu action. Asynchronous
    (a source:"tool" job -- poll vc3d_job_status). param_overrides is merged
    over <project directory>/trace_params.json. The trace grows along the surfaces tagged
    source_tag; its surfaces are tagged output_tag and their UUIDs are the
    finished job's output. Rejects remote volumes. Returns {"jobId", "kind":
    "tracer.run_trace", "source": "tool"}.

    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline."""
    result = await _call(
        "tracer.run_trace",
        _strip_none(
            {
                "segmentId": segment_id,
                "paramOverrides": param_overrides,
                "ompThreads": omp_threads,
                "sourceTag": source_tag,
                "outputTag": output_tag,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)
