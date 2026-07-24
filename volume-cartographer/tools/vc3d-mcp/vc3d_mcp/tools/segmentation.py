"""Segment listing/activation, editing, growth, tags, push/pull, and run-trace tools."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from mcp.server.fastmcp import Context

from ..core import mcp, _call, _wait_for_job, _strip_none, _is_placeholder_error
from ..bridge_client import BridgeError


class Point3D(TypedDict):
    x: float
    y: float
    z: float


@mcp.tool()
async def vc3d_list_segments(only_loaded: bool = False) -> dict[str, Any]:
    """List segments in the open volume package with loaded/active flags."""
    return await _call("segments.list", {"onlyLoaded": only_loaded})


async def _fetch_segment_impl(
    segment_id: str, wait: bool, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Fetch implementation shared by the vc3d_fetch_segment tool and the
    auto-fetch path in vc3d_activate_segment. A plain (non-tool) coroutine so
    the latter can forward progress through _wait_for_job without calling a
    decorated tool."""
    result = await _call("segments.fetch", {"segmentId": segment_id})
    job_id = result.get("jobId") if isinstance(result, dict) else None
    if not job_id:
        return result  # synchronous: already materialized, nothing to wait on
    return await _wait_for_job(job_id, wait, result, ctx)


@mcp.tool()
async def vc3d_fetch_segment(
    segment_id: str, wait: bool = True, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Download ("materialize") an open-data placeholder segment so it can be
    activated and edited.

    Segments attached from the Open Data catalog are lazy placeholders that hold
    only metadata until fetched -- vc3d_activate_segment on one fails with
    "open-data placeholder; fetch it first". This runs the fetch, the
    programmatic equivalent of the GUI's fetch-on-click.

    If the segment is already materialized, returns immediately with
    alreadyMaterialized=true. Otherwise the fetch runs as a "catalog"-source job:
    wait defaults to true (block until the download finishes and the segment is
    activatable); pass wait=false to get the jobId back immediately and poll
    vc3d_job_status yourself.

    segment_id: a segment id as returned by vc3d_list_segments.
    """
    return await _fetch_segment_impl(segment_id, wait, ctx)


@mcp.tool()
async def vc3d_activate_segment(
    segment_id: str, auto_fetch: bool = True, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Make a segment the active editing target (the programmatic equivalent of
    clicking it in the segment list). Required before vc3d_enable_editing /
    vc3d_grow_segment and after any segment switch.

    segment_id: a segment id as returned by vc3d_list_segments (including
    folder-qualified display ids like "paths/foo").
    auto_fetch: if true (default), when segment_id is an unfetched open-data
    placeholder, transparently fetch it first (blocking, via vc3d_fetch_segment)
    and then activate -- mirroring a GUI double-click, which fetches on click.
    The result then carries fetched=true. Set auto_fetch=false to instead get
    the raw "open-data placeholder; fetch it first" error without downloading.
    """
    try:
        return await _call("segments.activate", {"segmentId": segment_id})
    except BridgeError as exc:
        if not (auto_fetch and _is_placeholder_error(exc)):
            raise

    # Placeholder + auto_fetch: fetch (blocking) then retry activation once.
    fetch = await _fetch_segment_impl(segment_id, wait=True, ctx=ctx)
    if fetch.get("waitTimedOut"):
        return {"activated": False, "fetched": False, "fetch": fetch,
                "detail": "segment fetch did not finish within the wait cap"}
    state = fetch.get("state")
    if state is not None and state != "succeeded":
        return {"activated": False, "fetched": False, "fetch": fetch,
                "detail": f"segment fetch did not succeed (state={state})"}

    result = await _call("segments.activate", {"segmentId": segment_id})
    result["fetched"] = True
    return result


@mcp.tool()
async def vc3d_delete_segment(segment_id: str, confirm: bool = False) -> dict[str, Any]:
    """Permanently delete a segment from the open volume package, removing it
    from the surface panel AND deleting its files on disk.

    IRREVERSIBLE: this deletes the segment's on-disk data. You MUST pass
    confirm=True to actually delete; the default confirm=False is rejected
    -32602 ({param:"confirm"}) as a safety guard. Deleting the active segment is
    allowed (the active slot is cleared first), but deletion is refused while
    segmentation editing mode is enabled -- turn it off with
    vc3d_enable_editing(False) first.

    segment_id: a segment id as returned by vc3d_list_segments.
    confirm: must be True to proceed (guards against accidental data loss).

    Returns {"deleted": [segmentId]}. Errors: -32602 (confirm not True),
    -32007 (unknown segment), -32004/-32010 (cannot delete while editing)."""
    return await _call("segments.delete", {"segmentId": segment_id, "confirm": confirm})


@mcp.tool()
async def vc3d_rename_segment(segment_id: str, new_name: str) -> dict[str, Any]:
    """Rename a segment (its id and on-disk directory), the headless twin of the
    surface panel's rename action.

    new_name must match ^[a-zA-Z0-9_-]+$ (letters, digits, underscore, hyphen)
    and must not collide with an existing segment. Renaming is refused while
    segmentation editing mode is enabled -- turn it off with
    vc3d_enable_editing(False) first. On any failure the on-disk state is rolled
    back so the segment is left intact.

    segment_id: a segment id as returned by vc3d_list_segments.
    new_name: the desired new id/name.

    Returns {"oldId", "newId"}. Errors: -32602 (invalid new_name), -32007
    (unknown segment), -32010 (target name already exists / cannot rename while
    editing)."""
    return await _call("segments.rename", {"segmentId": segment_id, "newName": new_name})


@mcp.tool()
async def vc3d_enable_editing(enabled: bool) -> dict[str, Any]:
    """Turn segmentation editing mode on/off for the active segment."""
    return await _call("segmentation.enable_editing", {"enabled": enabled})


@mcp.tool()
async def vc3d_save_segment(
    wait: bool = True, ctx: Optional[Context] = None
) -> dict[str, Any]:
    """Force the active segment's pending autosave to disk now.

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
    method: Literal["tracer", "corrections", "patch_tracer"] = "tracer",
    direction: Literal["all", "up", "down", "left", "right", "fill"] = "all",
    inpaint_only: bool = False,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Grow the active segmentation surface. Async: returns a jobId.

    method: "tracer" | "corrections" | "patch_tracer". ("manual_add" is not a
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
async def vc3d_grow_patch_from_seed(
    seed: Point3D,
    volume_id: Optional[str] = None,
    iterations: int = 200,
    min_area_cm: float = 0.002,
    output_dir: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Create a brand-new segment by growing a patch from a 3D seed point
    (headless GrowPatch). Async: returns a jobId and outputDir.

    seed: volume-space {"x","y","z"} seed point.
    volume_id: vpkg volume id; default is the current volume.
    iterations: 1..100000, default 200 ("generations" in the underlying tool).
    min_area_cm: minimum patch area in cm^2, >= 0.
    output_dir: absolute path, or relative to the volpkg root.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.
    """
    result = await _call(
        "segmentation.grow_patch_from_seed",
        _strip_none(
            {
                "seed": seed,
                "volumeId": volume_id,
                "iterations": iterations,
                "minAreaCm": min_area_cm,
                "outputDir": output_dir,
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
    """Set (or clear) a review tag on a segment. tag is one of
    "approved" | "defective" | "reviewed" | "inspect" (there is no "revisit"
    tag). As a documented side effect this selects segment_id in the surface
    panel. Returns {"segmentId", "tag", "enabled"}."""
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
    output_dir: Optional[str] = None,
    wait: bool = False,
    ctx: Optional[Context] = None,
) -> dict[str, Any]:
    """Run the patch-stitching tracer (vc_grow_seg_from_segments) on a segment,
    the headless twin of the "Run Trace" context-menu action. Asynchronous
    (a source:"tool" job -- poll vc3d_job_status). param_overrides is merged
    over <volpkg>/trace_params.json. output_dir defaults to <volpkg>/traces.
    Rejects remote volumes. Returns {"jobId", "kind": "tracer.run_trace",
    "source": "tool", "outputDir"}.

    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline."""
    result = await _call(
        "tracer.run_trace",
        _strip_none(
            {
                "segmentId": segment_id,
                "paramOverrides": param_overrides,
                "ompThreads": omp_threads,
                "outputDir": output_dir,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result, ctx)
