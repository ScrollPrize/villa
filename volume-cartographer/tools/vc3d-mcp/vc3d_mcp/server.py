"""MCP server exposing the VC3D Agent Bridge RPCs as MCP tools.

One MCP tool per JSON-RPC method in apps/VC3D/agent_bridge/SPEC.md section 3,
per the tool surface table in SPEC.md section 5. Uses the official MCP Python
SDK (`mcp` package, `FastMCP`) over stdio -- see README.md "Implementation
notes" for why (the SDK installs cleanly in this environment, so we didn't
need to hand-roll the protocol).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import subprocess
import sys
import time
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from .bridge_client import (
    BridgeClient,
    BridgeClientConfig,
    BridgeConnectionError,
    BridgeError,
    discover_registry_socket,
)

DEFAULT_WAIT_TIMEOUT_S = 30 * 60  # SPEC.md section 5: "the MCP server enforces a 30-minute cap"

mcp = FastMCP(
    name="vc3d-agent-bridge",
    instructions=(
        "Tools that drive a running VC3D instance through its Agent Bridge. "
        "Call vc3d_get_state first to see what's loaded (volume package, "
        "current volume, active segment, viewers, running job) before doing "
        "anything else."
    ),
)

# Set by main()/configure_client() before the stdio loop starts. A module
# global (rather than a class) because FastMCP tool functions are registered
# as free functions via the `@mcp.tool()` decorator.
_client: BridgeClient | None = None


def configure_client(socket: str, request_timeout: float = 30.0) -> BridgeClient:
    global _client
    _client = BridgeClient(BridgeClientConfig(socket=socket, request_timeout=request_timeout))
    return _client


def _get_client() -> BridgeClient:
    if _client is None:
        raise RuntimeError(
            "vc3d-mcp server was not configured with a bridge socket "
            "(configure_client() was never called before serving requests)"
        )
    return _client


async def _call(method: str, params: dict[str, Any] | None = None) -> Any:
    """Call a bridge RPC, translating transport failures into a clear message.

    JSON-RPC error replies (BridgeError) are left to propagate as-is -- its
    `__str__` is the `{"code", "message", "data"}` JSON object per SPEC.md
    section 2.5, which is what ends up in the MCP tool's error text, per the
    "RPC errors surface as MCP tool errors with code/message/data preserved"
    requirement in SPEC.md section 5.
    """
    client = _get_client()
    try:
        return await client.call(method, params)
    except BridgeConnectionError as exc:
        raise RuntimeError(str(exc)) from exc


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


async def _wait_for_job(job_id: str, wait: bool, initial_result: dict[str, Any]) -> dict[str, Any]:
    """Shared `wait: true` handling for the async grow.* tools.

    SPEC.md section 5: "forwards phase:'output' messages as tool progress
    ... when the caller opted into 'wait': true ... blocks the tool call
    until the job's finished notification and returns the terminal status
    inline. wait defaults to false; when true, the MCP server enforces a
    30-minute cap and returns the still-running jobId on timeout."
    """
    if not wait:
        return initial_result

    client = _get_client()
    record = await client.jobs.wait_finished(job_id, timeout=DEFAULT_WAIT_TIMEOUT_S)
    if record is None:
        return {**initial_result, "waitTimedOut": True}

    # job.status is the bridge's authoritative terminal record (SPEC.md 3.17);
    # prefer it over our locally-tracked notification data, falling back to
    # the local record if the RPC itself races with process shutdown.
    try:
        status = await _call("job.status", {"jobId": job_id})
    except Exception:
        status = record.as_dict()
    return {**initial_result, **status}


# ---------------------------------------------------------------------------
# Tools -- one per SPEC.md section 5 table row, same order.
# ---------------------------------------------------------------------------


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
async def vc3d_list_segments(only_loaded: bool = False) -> dict[str, Any]:
    """List segments in the open volume package with loaded/active flags."""
    return await _call("segments.list", {"onlyLoaded": only_loaded})


@mcp.tool()
async def vc3d_activate_segment(segment_id: str) -> dict[str, Any]:
    """Make a segment the active editing target (the programmatic equivalent of
    clicking it in the segment list). Required before vc3d_enable_editing /
    vc3d_grow_segment and after any segment switch.

    segment_id: a segment id as returned by vc3d_list_segments (including
    folder-qualified display ids like "paths/foo").
    """
    return await _call("segments.activate", {"segmentId": segment_id})


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
    """
    return await _call(
        "screenshot.capture",
        _strip_none({"target": target, "filePath": file_path, "maxDim": max_dim}),
    )


@mcp.tool()
async def vc3d_get_cursor_point(
    viewer: Optional[str] = None, scene: Optional[dict[str, float]] = None
) -> dict[str, Any]:
    """Resolve a viewer scene position (or the current cursor) to a 3D volume
    point + surface normal.

    viewer: viewer id ("v1") or surface-slot name; default "segmentation".
    scene: {"x", "y"} scene-space position; omit to use the viewer's last
    known cursor position.
    """
    return await _call(
        "canvas.get_cursor_volume_point", _strip_none({"viewer": viewer, "scene": scene})
    )


@mcp.tool()
async def vc3d_click(
    position: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    button: str = "left",
    modifiers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Synthesize a mouse click in a viewer at a volume-space (or scene-space)
    position, with button and modifiers (e.g. modifiers=["shift"] to place a
    point / set focus).

    position: {"x","y","z"} in volume space (default), or {"x","y"} in scene
    space when space="scene".
    button: "left" | "right" | "middle".
    modifiers: any of "shift", "ctrl", "alt", "meta", "keypad".
    """
    return await _call(
        "canvas.click",
        _strip_none(
            {
                "viewer": viewer,
                "position": position,
                "space": space,
                "button": button,
                "modifiers": modifiers if modifiers is not None else [],
            }
        ),
    )


@mcp.tool()
async def vc3d_shift_click(
    position: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    button: str = "left",
    modifiers: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Shift+click convenience: the canonical place-point / set-focus gesture.
    Identical to vc3d_click with "shift" unioned into modifiers."""
    return await _call(
        "canvas.shift_click",
        _strip_none(
            {
                "viewer": viewer,
                "position": position,
                "space": space,
                "button": button,
                "modifiers": modifiers if modifiers is not None else [],
            }
        ),
    )


@mcp.tool()
async def vc3d_center_viewer(
    point: dict[str, float], viewer: Optional[str] = None, force_render: bool = True
) -> dict[str, Any]:
    """Center a viewer pane on a 3D volume point."""
    return await _call(
        "viewer.center_on_point",
        _strip_none({"viewer": viewer, "point": point, "forceRender": force_render}),
    )


@mcp.tool()
async def vc3d_zoom_viewer(factor: float, viewer: Optional[str] = None) -> dict[str, Any]:
    """Multiply a viewer's zoom by a factor (>1 zooms in). Returns the new
    scale."""
    return await _call("viewer.zoom", _strip_none({"viewer": viewer, "factor": factor}))


@mcp.tool()
async def vc3d_enable_editing(enabled: bool) -> dict[str, Any]:
    """Turn segmentation editing mode on/off for the active segment."""
    return await _call("segmentation.enable_editing", {"enabled": enabled})


@mcp.tool()
async def vc3d_grow_segment(
    steps: int,
    method: str = "tracer",
    direction: str = "all",
    inpaint_only: bool = False,
    wait: bool = False,
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
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_grow_patch_from_seed(
    seed: dict[str, float],
    volume_id: Optional[str] = None,
    iterations: int = 200,
    min_area_cm: float = 0.002,
    output_dir: Optional[str] = None,
    wait: bool = False,
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
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_commit_points(
    collection: str, points: list[dict[str, float]], winding: Optional[float] = None
) -> dict[str, Any]:
    """Add annotation points (volume space) to a named collection, optionally
    with a winding annotation. The collection is created if absent."""
    return await _call(
        "points.commit", _strip_none({"collection": collection, "points": points, "winding": winding})
    )


@mcp.tool()
async def vc3d_list_points(collection: Optional[str] = None) -> dict[str, Any]:
    """List point collections and their points. Omit collection to list all
    collections."""
    return await _call("points.list", _strip_none({"collection": collection}))


@mcp.tool()
async def vc3d_open_volume(path: str, volume_id: Optional[str] = None) -> dict[str, Any]:
    """Open a volume package (.volpkg / .volpkg.json / zarr project) and
    optionally select a volume id."""
    return await _call("volume.open", _strip_none({"path": path, "volumeId": volume_id}))


@mcp.tool()
async def vc3d_open_catalog_sample(
    sample_id: str,
    resources: Optional[dict[str, Any]] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Open an Open Data catalog sample by its manifest sample id. Async: a
    remote open is a multi-second-to-multi-minute network operation, so this
    returns a jobId immediately (SPEC §18.4).

    resources: optional resource-selection filter to attach only a subset
    (SPEC §10.3). Omit to attach everything (original behavior). Shape:
    {"volumeIds": [str],            # subset of the sample's volume ids
     "representationRefs": [str],   # "vi:ai" refs from vc3d_describe_catalog_sample
     "kinds": [str]}                # subset of "normal_grids"|"lasagna"|"prediction"
    An absent sub-field means no filter on that axis. A raw source volume is
    attached iff volumeIds is absent or lists its id; a derived representation
    must pass all provided axes.

    Returns {jobId, kind, source:"catalog", sampleId}. Poll job.status (or pass
    wait=true) for the terminal record; its "result" carries the opened project
    plus an "attached" block (volumes/segments/normalGrids/lasagnaDatasets),
    "vpkgPath", "volumeIds", and "messages".

    wait: if true (MCP-server-side only), block until the job finishes (30-minute
    cap) and return the terminal job.status inline.
    """
    result = await _call(
        "catalog.open_sample", _strip_none({"sampleId": sample_id, "resources": resources})
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_list_catalog_samples(refresh: bool = False) -> dict[str, Any]:
    """List Open Data catalog samples from the manifest (id, type, description,
    volume/segment/scan counts).

    refresh: force a fresh manifest fetch (up to 30 s) instead of serving the
    cached copy; also fetches automatically when nothing is cached yet.
    """
    return await _call("catalog.list_samples", {"refresh": refresh})


@mcp.tool()
async def vc3d_describe_catalog_sample(
    sample_id: str, refresh: bool = False
) -> dict[str, Any]:
    """Describe one Open Data catalog sample: its volumes (id, scanId, shape,
    pixel size, data format) and derived representations categorized by kind
    (normal_grids / lasagna / prediction), each with a stable "ref" ("vi:ai")
    usable in vc3d_open_catalog_sample's resources.representationRefs.

    refresh: force a fresh manifest fetch (up to 30 s) before describing.
    """
    return await _call(
        "catalog.describe_sample", {"sampleId": sample_id, "refresh": refresh}
    )


@mcp.tool()
async def vc3d_select_volume(volume_id: str) -> dict[str, Any]:
    """Switch the current volume among the already-attached volumes of the open
    package (the programmatic equivalent of picking one in the volume combo).
    Selecting the already-current volume is a no-op success. Returns
    {"volumeId", "previousVolumeId"}."""
    return await _call("volume.select", {"volumeId": volume_id})


@mcp.tool()
async def vc3d_job_status(job_id: Optional[str] = None, source: Optional[str] = None) -> dict[str, Any]:
    """Poll a job by id (or the latest job): state, message, console tail.

    source: optionally filter "the latest job" to one source ("tool" | "growth"
    | "lasagna" | "atlas") when job_id is omitted (SPEC §8.3).
    """
    return await _call("job.status", _strip_none({"jobId": job_id, "source": source}))


# ---------------------------------------------------------------------------
# Manual-add (hole-fill) + corrections point authoring (SPEC 9.2-9.7)
# ---------------------------------------------------------------------------


@mcp.tool()
async def vc3d_manual_add_begin() -> dict[str, Any]:
    """Enter manual-add (hole-fill) mode on the active editing session.

    Requires segmentation editing enabled with an active edit session and no
    growth running. Returns {"active": true}. Idempotent. Once active, place
    plane constraints with vc3d_shift_click (shift+left adds/replaces, shift+
    right removes the nearest) on a plane viewer, then call
    vc3d_manual_add_finish to commit or discard.
    """
    return await _call("segmentation.manual_add.begin", {})


@mcp.tool()
async def vc3d_manual_add_finish(apply: bool = True) -> dict[str, Any]:
    """Leave manual-add mode. apply=true commits the fill preview (which may
    trigger an in-process growth run, observable as a source:"growth" job);
    apply=false discards it. Returns {"applied": bool}."""
    return await _call("segmentation.manual_add.finish", {"apply": apply})


@mcp.tool()
async def vc3d_manual_add_set_line_mode(mode: str) -> dict[str, Any]:
    """Set the manual-add line-preview mode. mode: "vertical" | "horizontal" |
    "cross" | "cross_fill". Callable whether or not manual-add mode is active
    (the config persists). Returns the effective {"mode": str}."""
    return await _call("segmentation.manual_add.set_line_mode", {"mode": mode})


@mcp.tool()
async def vc3d_manual_add_set_interpolation(mode: str) -> dict[str, Any]:
    """Set the manual-add interpolation (fill) method. mode:
    "thin_plate_spline" | "tracer_restricted_to_fill". Callable whether or not
    manual-add mode is active. Returns the effective {"mode": str}."""
    return await _call("segmentation.manual_add.set_interpolation", {"mode": mode})


@mcp.tool()
async def vc3d_manual_add_undo_constraint() -> dict[str, Any]:
    """Remove the most recently placed user plane constraint in manual-add mode.
    Returns {"undone": bool} (false when there was none to remove; not an
    error). Requires manual-add mode to be active."""
    return await _call("segmentation.manual_add.undo_constraint", {})


@mcp.tool()
async def vc3d_corrections_set_point_mode(active: bool) -> dict[str, Any]:
    """Enable/disable correction-point authoring mode (the G-key mode) without a
    keypress. Requires editing enabled, an active edit session, and no growth
    running. Returns {"active": bool}.

    While active, a plain vc3d_click (zero-length) commits a single un-anchored
    correction point (no solver run); a vc3d_drag longer than 1.0 voxel commits
    an anchored correction and auto-triggers the corrections solver (a
    source:"growth" job -- poll vc3d_job_status before further editing RPCs).
    Unlike the physical key, the mode is NOT auto-cleared on mouse release --
    switch it off with active=false when done.
    """
    return await _call(
        "segmentation.corrections.set_point_mode", {"active": active}
    )


# ---------------------------------------------------------------------------
# Atlas RPCs (SPEC 12)
# ---------------------------------------------------------------------------


@mcp.tool()
async def vc3d_atlas_open(atlas_dir: str) -> dict[str, Any]:
    """Open (load and display) a fiber atlas from a directory.

    atlas_dir: absolute path, or relative to the open volume package's root.
    Never shows a dialog or the interactive rebuild prompt. Returns
    {"opened": true, "atlasDir", "atlasName"}.
    """
    return await _call("atlas.open", {"atlasDir": atlas_dir})


@mcp.tool()
async def vc3d_atlas_status() -> dict[str, Any]:
    """Report the current atlas (dir/name, null when none is open) and the
    fiber-intersection search state: {"search": {"running", "phase",
    "phaseCount": 5, "completed", "total", "cancelRequested", "resultCount"}}.
    """
    return await _call("atlas.status", {})


@mcp.tool()
async def vc3d_atlas_search_start(
    mode: str = "atlas_to_non_atlas",
    required_tags: Optional[list[str]] = None,
    excluded_tags: Optional[list[str]] = None,
    max_distance: Optional[float] = None,
) -> dict[str, Any]:
    """Start an asynchronous atlas fiber-intersection search (a source:"atlas"
    job -- poll vc3d_job_status, then page results with vc3d_atlas_search_results).

    mode: "atlas_to_non_atlas" (requires an open atlas with fiber mappings) or
    "non_atlas_only". required_tags/excluded_tags filter the candidate fibers.
    max_distance: broad-phase segment radius in voxels; omit to keep the
    current spin-box value. Returns {"jobId", "kind": "atlas.fiber_search",
    "source": "atlas"}.
    """
    return await _call(
        "atlas.search_start",
        _strip_none(
            {
                "mode": mode,
                "requiredTags": required_tags,
                "excludedTags": excluded_tags,
                "maxDistance": max_distance,
            }
        ),
    )


@mcp.tool()
async def vc3d_atlas_search_cancel() -> dict[str, Any]:
    """Request cancellation of the running atlas fiber-intersection search.
    The job still terminates through its finished notification
    (success: false). Returns {"cancelRequested": true}."""
    return await _call("atlas.search_cancel", {})


@mcp.tool()
async def vc3d_atlas_search_results(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Page through the completed atlas search results (vector order).

    limit is clamped to [1, 1000]. Each row carries "index" (the id
    vc3d_atlas_open_result takes), source/target fiber ids (as strings),
    candidateDistance, refinedScore, windingDistance (null when infinite),
    signedWinding (null when absent), source/target points and arclengths,
    converged and message. Returns {"total", "offset", "results": [...]}.
    """
    return await _call("atlas.search_results", {"offset": offset, "limit": limit})


@mcp.tool()
async def vc3d_atlas_open_result(index: int) -> dict[str, Any]:
    """Open one atlas search result in the intersections inspection workspace.

    index: a result "index" as returned by vc3d_atlas_search_results (vector
    order). Returns {"opened": true, "index"}.
    """
    return await _call("atlas.open_result", {"index": index})


@mcp.tool()
async def vc3d_atlas_remap() -> dict[str, Any]:
    """Rebuild the current atlas's fiber mappings from its source fiber JSON
    (asynchronous; the atlas is redisplayed on completion and progress is
    visible in app status messages). Returns {"remapped": true} once the remap
    worker is launched."""
    return await _call("atlas.remap", {})


@mcp.tool()
async def vc3d_atlas_optimize_snap_candidates() -> dict[str, Any]:
    """Queue Laplace snap-candidate ranking for the current atlas through the
    lasagna fit service (requires the service to be available). Completion is
    observable via app status messages; not modeled as a job. Returns
    {"requested": true}."""
    return await _call("atlas.optimize_snap_candidates", {})


# ---------------------------------------------------------------------------
# Line annotation / fiber RPCs (SPEC 13)
# ---------------------------------------------------------------------------


@mcp.tool()
async def vc3d_fiber_launch(
    position: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    replace_owning: bool = True,
) -> dict[str, Any]:
    """Open the line-annotation (fiber tracing) workspace seeded at a position
    (the twin of the interactive launch gesture). The workspace's panes appear
    in vc3d_get_state's viewers and are drivable with vc3d_click / vc3d_drag.

    position: {"x","y","z"} in volume space (default) or {"x","y"} scene-space
    when space="scene". The point must lie on the target viewer's current
    view (same round-trip rule as vc3d_click). Requires a Lasagna dataset to
    be resolvable for the active volume; otherwise fails -32005 with detail.
    """
    return await _call(
        "fiber.launch",
        _strip_none(
            {
                "viewer": viewer,
                "position": position,
                "space": space,
                "replaceOwning": replace_owning,
            }
        ),
    )


@mcp.tool()
async def vc3d_fiber_list() -> dict[str, Any]:
    """List saved fibers of the open volume package: fiberId (string), name,
    control/line point counts, length, automatic/manual HV tags, tags, and
    per-span summaries; plus knownTags (every tag in use)."""
    return await _call("fiber.list", {})


@mcp.tool()
async def vc3d_fiber_open(
    fiber_id: str,
    control_point_index: Optional[int] = None,
    line_point_index: Optional[int] = None,
    span: Optional[list[int]] = None,
) -> dict[str, Any]:
    """Open a saved fiber in the line-annotation workspace, optionally focused
    at a control point, a line-point index, or a [first, second] control-index
    span. Pass at most one selector. Requires a Lasagna dataset to be
    resolvable for the active volume; otherwise fails -32005 with detail."""
    return await _call(
        "fiber.open",
        _strip_none(
            {
                "fiberId": fiber_id,
                "controlPointIndex": control_point_index,
                "linePointIndex": line_point_index,
                "span": span,
            }
        ),
    )


@mcp.tool()
async def vc3d_fiber_set_follow(enabled: bool) -> dict[str, Any]:
    """Toggle "current cut follows strip mouse" on the most recently opened
    line-annotation workspace. Fails -32007 kind:"fiber_workspace" when no
    workspace is open. Returns {"enabled": bool}."""
    return await _call("fiber.set_follow", {"enabled": enabled})


@mcp.tool()
async def vc3d_fiber_save() -> dict[str, Any]:
    """Save every open line-annotation workspace's fiber to disk. Saves are
    scheduled and complete asynchronously (headless twin of Save Open Fibers).
    Returns {"saved": true}."""
    return await _call("fiber.save", {})


@mcp.tool()
async def vc3d_fiber_delete(fiber_ids: list[str]) -> dict[str, Any]:
    """Delete saved fibers by id (>= 1 id; all ids validated first,
    all-or-nothing). Returns {"deleted": [ids]}."""
    return await _call("fiber.delete", {"fiberIds": fiber_ids})


@mcp.tool()
async def vc3d_fiber_set_tag(fiber_id: str, tag: str, enabled: bool) -> dict[str, Any]:
    """Add (enabled=true) or remove (enabled=false) a free-form tag on a saved
    fiber. vc3d_fiber_list's knownTags enumerates tags in use."""
    return await _call(
        "fiber.set_tag", {"fiberId": fiber_id, "tag": tag, "enabled": enabled}
    )


@mcp.tool()
async def vc3d_fiber_create_atlas(fiber_id: str) -> dict[str, Any]:
    """Create a single-fiber atlas from a saved fiber and display it
    (dialog-free). SYNCHRONOUS and potentially slow (heavy geometry work on
    the app thread) -- allow a generous client timeout. Requires a Lasagna
    dataset resolvable for the active volume. Returns {"atlasDir",
    "displayed"} (plus displayDetail when display failed but creation
    succeeded)."""
    return await _call("fiber.create_atlas", {"fiberId": fiber_id})


@mcp.tool()
async def vc3d_fiber_export(path: str, scale: float = 1.0) -> dict[str, Any]:
    """Export ALL saved fibers to one vc3d_fiber_collection JSON bundle at
    `path` (headless; no dialog). scale multiplies coordinates on export.
    Returns {"exported": count, "path"}."""
    return await _call("fiber.export", {"path": path, "scale": scale})


@mcp.tool()
async def vc3d_fiber_import(path: str, scale: float = 1.0) -> dict[str, Any]:
    """Import fibers from `path`: a single vc3d_fiber JSON, a
    vc3d_fiber_collection bundle, or a directory of fiber JSONs (headless; no
    dialog). Imported fibers are saved into the package's fibers directory
    and the fiber list reloads. Returns {"imported", "skipped"}."""
    return await _call("fiber.import", {"path": path, "scale": scale})


# ---------------------------------------------------------------------------
# Stage 6 backlog surface (SPEC §15): tags, seeding, push/pull, run-trace
# ---------------------------------------------------------------------------


@mcp.tool()
async def vc3d_set_segment_tag(segment_id: str, tag: str, enabled: bool) -> dict[str, Any]:
    """Set (or clear) a review tag on a segment. tag is one of
    "approved" | "defective" | "reviewed" | "inspect" (there is no "revisit"
    tag). As a documented side effect this selects segment_id in the surface
    panel. Returns {"segmentId", "tag", "enabled"}."""
    return await _call(
        "tags.set", {"segmentId": segment_id, "tag": tag, "enabled": enabled}
    )


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
    {"reset": true}.

    NOTE: seeding.run / seeding.expand / seeding.analyze_paths are intentionally
    NOT exposed -- those actions spin a nested event loop until child processes
    finish, which is unsafe for the bridge (SPEC §15.2 amendment)."""
    return await _call("seeding.reset_points", {})


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
    direction: str, alpha: Optional[bool] = None
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
) -> dict[str, Any]:
    """Run the patch-stitching tracer (vc_grow_seg_from_segments) on a segment,
    the headless twin of the "Run Trace" context-menu action. Asynchronous
    (a source:"tool" job -- poll vc3d_job_status). param_overrides is merged
    over <volpkg>/trace_params.json. output_dir defaults to <volpkg>/traces.
    Rejects remote volumes. Returns {"jobId", "kind": "tracer.run_trace",
    "source": "tool", "outputDir"}."""
    return await _call(
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


@mcp.tool()
async def vc3d_render_tifxyz(
    segment_id: str,
    output_format: str,
    volume_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    scale: float = 1.0,
    group_idx: int = 0,
    num_slices: int = 1,
    voxel_size: Optional[float] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Render a segment's flattened surface with vc_render_tifxyz, the headless
    twin of the "Render" context-menu action. Asynchronous (a source:"tool" job
    -- poll vc3d_job_status).

    segment_id: id of the segment to render.
    output_format: "zarr" (OME-Zarr store) or "tif_stack" (per-slice TIFFs).
    This choice is the headline capability over the GUI, which only produces a
    TIFF stack.
    volume_id: vpkg volume id; default is the current volume.
    output_dir: absolute path, or relative to the volpkg root. Default is the
    segment folder (output lands in <segment>/layers or <segment>/surface.zarr).
    scale: pixels per level-g voxel, > 0 (default 1.0).
    group_idx: OME-Zarr group index, >= 0 (default 0).
    num_slices: number of slices to render along the surface normal, >= 1
    (default 1).
    voxel_size: physical voxel size override in micrometers; omit to derive it
    from the volume metadata.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "render.tifxyz", "source": "tool", "outputDir",
    "outputFormat", "volumeId"}. Advanced render options (crop/affine/rotate/
    flip/in-render flatten/composite/alpha) are GUI-only and not exposed here.
    """
    result = await _call(
        "render.tifxyz",
        _strip_none(
            {
                "segmentId": segment_id,
                "outputFormat": output_format,
                "volumeId": volume_id,
                "outputDir": output_dir,
                "scale": scale,
                "groupIdx": group_idx,
                "numSlices": num_slices,
                "voxelSize": voxel_size,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_flatten_slim(
    segment_id: str,
    iterations: Optional[int] = None,
    tolerance: Optional[float] = None,
    energy_type: Optional[str] = None,
    keep_percent: Optional[float] = None,
    inpaint_holes: Optional[bool] = None,
    output_dir: Optional[str] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Flatten a segment with SLIM/flatboi -- the production-recommended
    flattening method, the headless twin of the "SLIM flatten" context-menu
    action. Asynchronous (a source:"flatten" job -- poll vc3d_job_status).

    Runs the flatboi pipeline (vc_tifxyz2obj -> flatboi -> vc_obj2tifxyz, with a
    vc_obj_uv_lift step only when decimating). No dialog is ever shown.

    segment_id: id of the segment to flatten.
    iterations: flatboi iterations, >= 1 (default 50).
    tolerance: convergence tolerance; 0 (default) runs all iterations.
    energy_type: "symmetric_dirichlet" (default) or "conformal".
    keep_percent: percentage of source grid points to keep, in (0, 100].
    Default 100 = full-resolution SLIM (no decimation), the production-
    recommended path. Below 100 decimates then lifts UVs back to full res.
    inpaint_holes: fill holes before flattening (default false).
    output_dir: absolute path, or relative to the volpkg root. Default is
    <segment>_flatboi.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "flatten.slim", "source": "flatten", "outputDir"}.
    """
    result = await _call(
        "flatten.slim",
        _strip_none(
            {
                "segmentId": segment_id,
                "iterations": iterations,
                "tolerance": tolerance,
                "energyType": energy_type,
                "keepPercent": keep_percent,
                "inpaintHoles": inpaint_holes,
                "outputDir": output_dir,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_flatten_abf(
    segment_id: str,
    iterations: Optional[int] = None,
    downsample_factor: Optional[int] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Flatten a segment with ABF++ -- the headless twin of the "ABF++ flatten"
    context-menu action. Runs in-process (no external tool), asynchronous (a
    source:"flatten" job -- poll vc3d_job_status). No dialog is ever shown.

    segment_id: id of the segment to flatten.
    iterations: ABF++ iterations, >= 1 (default 10).
    downsample_factor: mesh downsample factor, >= 1 (default 1).
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "flatten.abf", "source": "flatten", "outputDir"}
    -- output lands in <segment>_abf.
    """
    result = await _call(
        "flatten.abf",
        _strip_none(
            {
                "segmentId": segment_id,
                "iterations": iterations,
                "downsampleFactor": downsample_factor,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result)


@mcp.tool()
async def vc3d_flatten_straighten(
    segment_id: str,
    unbend: Optional[bool] = None,
    unbend_smooth_cols: Optional[float] = None,
    overlap_passes: Optional[int] = None,
    orthogonalize: Optional[bool] = None,
    trim: Optional[bool] = None,
    trim_max_edge: Optional[float] = None,
    output_dir: Optional[str] = None,
    wait: bool = False,
) -> dict[str, Any]:
    """Straighten a segment with vc_straighten -- the headless twin of the
    "Straighten" context-menu action. Asynchronous (a source:"flatten" job --
    poll vc3d_job_status). No dialog is ever shown.

    segment_id: id of the segment to straighten.
    unbend: run the unbend (spine-straightening) stage (default true).
    unbend_smooth_cols: spine Gaussian sigma in columns (default 300); only
    used when unbend is true.
    overlap_passes: number of --overlap-pairs passes (default 2).
    orthogonalize: run the orthogonalize stage (default true).
    trim: run the trim stage (default true).
    trim_max_edge: max edge length for the trim stage (default 100); only used
    when trim is true.
    output_dir: absolute path, or relative to the volpkg root. Default is
    <segment>_straightened. vc_straighten refuses to overwrite an existing dir.
    wait: if true (MCP-server-side only), block until the job finishes
    (30-minute cap) and return the terminal job.status inline.

    Returns {"jobId", "kind": "flatten.straighten", "source": "flatten",
    "outputDir"}.
    """
    result = await _call(
        "flatten.straighten",
        _strip_none(
            {
                "segmentId": segment_id,
                "unbend": unbend,
                "unbendSmoothCols": unbend_smooth_cols,
                "overlapPasses": overlap_passes,
                "orthogonalize": orthogonalize,
                "trim": trim,
                "trimMaxEdge": trim_max_edge,
                "outputDir": output_dir,
            }
        ),
    )
    return await _wait_for_job(result["jobId"], wait, result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# Handshake wait budget for the auto-launch path: VC3D can take a while to
# construct CWindow before AgentBridgeServer::listen() prints the handshake.
LAUNCH_HANDSHAKE_TIMEOUT_S = 30.0

# Kept alive for the process lifetime so a bridge we launched isn't reaped as a
# child zombie, and can be terminated on our own exit.
_launched_process: subprocess.Popen | None = None


def default_vc3d_binary() -> str:
    """The fallback VC3D binary path, resolved relative to the repo root.

    ``server.py`` lives at ``<repo>/tools/vc3d-mcp/vc3d_mcp/server.py``, so the
    repo root is three directories up.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))
    return os.path.join(repo_root, "build-macos", "bin", "VC3D")


def resolve_launch_binary(explicit: str | None) -> str | None:
    """Pick the VC3D binary to auto-launch, or None if none is usable.

    Priority: explicit ``--launch`` > ``VC3D_BINARY`` env var > the repo-root
    default. Returns the path only if it names a real, executable file.
    """
    candidate = explicit or os.environ.get("VC3D_BINARY") or default_vc3d_binary()
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def launch_vc3d(
    binary: str,
    volpkg: str | None = None,
    timeout: float = LAUNCH_HANDSHAKE_TIMEOUT_S,
) -> str:
    """Spawn VC3D with the agent bridge enabled and return its socket path.

    Reads the child's stdout until the ``VC3D-AGENT-BRIDGE: listening ...``
    handshake line appears (parsed by ``BridgeClient.socket_path_from_handshake``),
    then returns the authoritative ``path=`` value. Raises
    ``BridgeConnectionError`` if VC3D exits or the handshake never arrives
    within ``timeout`` seconds. The process is retained (module global) so it
    keeps running for the MCP session and is terminated on our exit.
    """
    global _launched_process

    args = [binary, "--agent-bridge"]
    if volpkg:
        # SPEC: forwarded to VC3D as --load-first so the agent doesn't have to
        # open a volume as its first action.
        args += ["--load-first", volpkg]

    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,  # line-buffered
    )
    _launched_process = proc
    atexit.register(_terminate_launched_process)

    deadline = time.monotonic() + timeout
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise BridgeConnectionError(
                f"VC3D ({binary!r}) exited with code {proc.returncode} "
                "before printing the agent-bridge handshake line"
            )
        line = proc.stdout.readline()
        if not line:
            # EOF without exit yet: brief pause, then re-check poll/deadline.
            time.sleep(0.05)
            continue
        parsed = BridgeClient.socket_path_from_handshake(line)
        if parsed is not None:
            _name, path = parsed
            return path

    _terminate_launched_process()
    raise BridgeConnectionError(
        f"timed out after {timeout:g}s waiting for VC3D ({binary!r}) to print "
        "its agent-bridge handshake line"
    )


def _terminate_launched_process() -> None:
    global _launched_process
    proc = _launched_process
    if proc is None:
        return
    _launched_process = None
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        pass


def resolve_connection(args: argparse.Namespace) -> tuple[str | None, str]:
    """Determine the bridge socket to use, per the documented priority order.

    Returns ``(socket_path_or_None, how)`` where ``how`` describes the source
    for diagnostics:

    (a) explicit ``--socket`` / ``VC3D_AGENT_BRIDGE_SOCKET`` (highest priority);
    (b) an already-running bridge found in the discovery registry;
    (c) auto-launch a new VC3D when ``--launch``/``VC3D_BINARY``/the default
        binary resolves to a real executable;
    (d) otherwise ``(None, ...)`` -- the caller prints the usage error.
    """
    if args.socket:
        return args.socket, "explicit --socket/env"

    discovered = discover_registry_socket()
    if discovered:
        return discovered, "discovery registry (already-running VC3D)"

    binary = resolve_launch_binary(args.launch)
    if binary:
        print(
            f"vc3d-mcp: no running bridge found; launching VC3D at {binary!r}...",
            file=sys.stderr,
        )
        path = launch_vc3d(binary, volpkg=args.volpkg)
        return path, f"auto-launched VC3D ({binary})"

    return None, "no socket, no running bridge, no launchable binary"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vc3d-mcp",
        description=(
            "MCP server (stdio) that drives a running VC3D instance via its "
            "Agent Bridge local socket. In the common case no socket needs to "
            "be passed: an already-running VC3D is found via the discovery "
            "registry (~/.vc3d/agent_bridge), and if none is running VC3D is "
            "auto-launched."
        ),
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("VC3D_AGENT_BRIDGE_SOCKET"),
        help=(
            "VC3D agent bridge socket: an explicit path (e.g. the 'path=' field "
            "from the VC3D-AGENT-BRIDGE stdout handshake line), or a bare "
            "QLocalServer name matching --agent-bridge-name. Highest priority "
            "when set. Defaults to the VC3D_AGENT_BRIDGE_SOCKET env var. When "
            "unset, an already-running bridge is auto-discovered, else VC3D is "
            "auto-launched (see --launch)."
        ),
    )
    parser.add_argument(
        "--launch",
        default=None,
        help=(
            "Path to a VC3D binary to auto-launch (with --agent-bridge) when no "
            "socket is given and no running bridge is discovered. Falls back to "
            "the VC3D_BINARY env var, then to the repo-root build "
            "(build-macos/bin/VC3D). Ignored when --socket or a running bridge "
            "is available."
        ),
    )
    parser.add_argument(
        "--volpkg",
        default=None,
        help=(
            "Optional volume-package path forwarded to an auto-launched VC3D as "
            "--load-first, so the agent's first action need not be opening a "
            "volume. Only used on the auto-launch path."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a bridge RPC response before failing the tool call (default: 30).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        socket, how = resolve_connection(args)
    except BridgeConnectionError as exc:
        print(f"vc3d-mcp: auto-launch failed: {exc}", file=sys.stderr)
        return 2

    if not socket:
        print(
            "vc3d-mcp: could not connect to a VC3D agent bridge. Options, in "
            "priority order:\n"
            "  1. Pass --socket <name-or-path> (or set VC3D_AGENT_BRIDGE_SOCKET) "
            "to attach to a known bridge.\n"
            "  2. Start VC3D with --agent-bridge and it will be auto-discovered "
            "via ~/.vc3d/agent_bridge.\n"
            "  3. Pass --launch <path-to-VC3D> (or set VC3D_BINARY, or build "
            "build-macos/bin/VC3D) to have this server launch VC3D itself.",
            file=sys.stderr,
        )
        return 2

    print(f"vc3d-mcp: using bridge socket {socket!r} via {how}.", file=sys.stderr)
    configure_client(socket, request_timeout=args.request_timeout)
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
