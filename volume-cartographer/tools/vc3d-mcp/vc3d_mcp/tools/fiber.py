"""Line-annotation (fiber tracing) workspace tools."""

from __future__ import annotations

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from ..core import mcp, _call, _strip_none

FIBER_SAVE_TIMEOUT_S = 130.0


class _Position(TypedDict):
    x: float
    y: float
    z: NotRequired[float]


@mcp.tool()
async def vc3d_fiber_launch(
    position: _Position,
    viewer: Optional[str] = None,
    space: Literal["volume", "scene"] = "volume",
    replace_owning: bool = True,
) -> dict[str, Any]:
    """Open the line-annotation (fiber tracing) workspace seeded at a position
    (the twin of the interactive launch gesture). The workspace's panes appear
    in vc3d_get_state's viewers and are drivable with vc3d_click / vc3d_drag.
    Add control points with plain vc3d_click (NOT vc3d_shift_click -- that
    triggers a different "predicted snap point" gesture on these panes).

    position: {"x","y","z"} in volume space (default) or {"x","y"} scene-space
    when space="scene". The point must lie on the target viewer's current
    view (same round-trip rule as vc3d_click). This opens the workspace and
    returns {"launched": true} even with no Lasagna dataset -- but TRACING
    needs one: seeding and every control point optimize against a
    manifest-backed Lasagna dataset (a "lasagna"-kind catalog representation)
    resolvable for the CURRENTLY SELECTED volume. Without one the panes open
    but no optimized line is produced (over the bridge the interactive "load
    lasagna normals" picker is suppressed, so the dataset must already be
    attached and its volume selected). Not every catalog sample/volume has one,
    and it resolves per-volume: a "normal_grids"-kind store is a DIFFERENT
    resource and does NOT satisfy this. Check vc3d_describe_catalog_sample for a
    "lasagna"-kind representation and vc3d_select_volume to the volume that
    carries it before tracing. This is unrelated to the vc3d_lasagna_* fit
    service.
    replace_owning: defaults True, which DISCARDS the caller's currently
    open/in-progress fiber workspace (verified: launching a second fiber with
    the default silently drops the first fiber's unsaved control points).
    Pass False to keep multiple fiber workspaces open at once, e.g. tracing
    several fibers before one combined vc3d_fiber_save.

    The workspace's viewer ids are NOT stable across edits -- each control
    point placed rebuilds/reoptimizes the panes under new "v<N>" ids. Call
    vc3d_get_state again before targeting a pane after any edit; a stale id
    fails -32002.
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
    span. Pass at most one selector. Requires a manifest-backed Lasagna
    dataset (a "lasagna"-kind catalog representation, NOT a "normal_grids"
    store and NOT the vc3d_lasagna_* fit service) resolvable for the currently
    selected volume; otherwise fails -32005 with detail."""
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
    """Save every open line-annotation workspace's fiber to disk. Returns only
    after persistence completes, with {"saved": true}."""
    return await _call("fiber.save", {}, timeout=FIBER_SAVE_TIMEOUT_S)


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
    the app thread) -- allow a generous client timeout. Requires a
    manifest-backed Lasagna dataset resolvable for the currently selected
    volume (a "lasagna"-kind representation, NOT a "normal_grids" store or the
    vc3d_lasagna_* fit service), AND that dataset's manifest must provide
    init_shell_dir -- a dataset can resolve for tracing yet still fail atlas
    creation if that atlas-only field is absent. Returns {"atlasDir",
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
