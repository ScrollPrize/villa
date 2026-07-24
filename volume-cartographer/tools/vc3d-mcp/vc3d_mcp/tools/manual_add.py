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
