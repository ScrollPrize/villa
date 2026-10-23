"""Manual-add (hole-fill) mode and correction-point authoring tools."""

from __future__ import annotations

from typing import Any, Literal

from ..core import mcp, _call


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
async def vc3d_manual_add_set_line_mode(
    mode: Literal["vertical", "horizontal", "cross", "cross_fill"]
) -> dict[str, Any]:
    """Set the manual-add line-preview mode. mode: "vertical" | "horizontal" |
    "cross" | "cross_fill". Callable whether or not manual-add mode is active
    (the config persists). Returns the effective {"mode": str}."""
    return await _call("segmentation.manual_add.set_line_mode", {"mode": mode})


@mcp.tool()
async def vc3d_manual_add_set_interpolation(
    mode: Literal["thin_plate_spline", "tracer_restricted_to_fill"]
) -> dict[str, Any]:
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
async def vc3d_manual_add_extend(
    side: Literal["up", "down", "left", "right"],
    cells: int = 16,
    first: int | None = None,
    last: int | None = None,
    applyPending: bool = False,
) -> dict[str, Any]:
    """Continue the active segment past one edge by `cells` vertices.

    Each row (side "left"/"right") or column ("up"/"down") ending at that
    edge is continued along its own direction at the edge; first/last limit
    it to those rows or columns. With the panel's "Extensions follow the
    sheet" setting on (the default), each line is then re-walked along the
    CT sheet in the background; "followingSheet" is true until that finishes
    (vc3d_get_state reports it as manualAddFollowingSheet), so wait for it
    before adding constraints or applying. Enters manual-add mode if needed and grows
    the grid when there is no room. Refine the preview with
    vc3d_manual_add_add_constraint or vc3d_shift_click on a plane viewer, then
    vc3d_manual_add_finish to apply (apply=false removes the strip and any
    grid growth). With applyPending=true and an extension of the same side
    pending, that one is applied and saved first and the next starts from the
    new edge, keeping its first/last; "queued" is true while it waits for the
    save. Returns {"active": bool, "side": str, "queued": bool,
    "followingSheet": bool}.
    """
    params: dict[str, Any] = {"side": side, "cells": cells}
    if first is not None:
        params["first"] = first
    if last is not None:
        params["last"] = last
    if applyPending:
        params["applyPending"] = True
    return await _call("segmentation.manual_add.extend", params)


@mcp.tool()
async def vc3d_manual_add_add_constraint(
    world: list[float], radius: float = 0.0
) -> dict[str, Any]:
    """Pin the pending manual-add geometry to a world point [x, y, z], like a
    Shift+click on the sheet in a plane viewer. The pending vertex nearest to
    the point within `radius` (the panel's constraint radius when 0) moves
    there and the rest of the pending geometry follows. Returns
    {"placed": true}. Requires manual-add mode with pending geometry."""
    return await _call(
        "segmentation.manual_add.add_constraint",
        {"world": world, "radius": radius},
    )


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
