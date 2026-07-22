"""Viewer + canvas interaction tools (cursor point, click, drag, center, zoom, rotate).

Split out of the original monolithic ``server.py``; each tool registers on
the single shared ``mcp`` instance from ``vc3d_mcp.core``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core import mcp, _call, _strip_none

__all__ = [
    "vc3d_get_cursor_point",
    "vc3d_click",
    "vc3d_shift_click",
    "vc3d_center_viewer",
    "vc3d_zoom_viewer",
    "vc3d_rotate_viewer",
    "vc3d_set_axis_aligned_slices",
    "vc3d_drag",
]


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

    On a fiber-tracing (Line Annotation) workspace pane specifically: a PLAIN
    click (no "shift" modifier) is what adds a control point. Adding "shift"
    switches to a different gesture there (see vc3d_shift_click) -- it does
    NOT place a control point on these panes, unlike its usual place-point
    role elsewhere.
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
    Identical to vc3d_click with "shift" unioned into modifiers.

    Exception: on a fiber-tracing (Line Annotation) workspace pane, this
    triggers a "predicted snap point" gesture instead of adding a control
    point -- use plain vc3d_click (no shift) there to add control points."""
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
    """Multiply a viewer's zoom scale by `factor` (>1 zooms in, <1 zooms out;
    e.g. 10 = 10x closer, 0.5 = half). It is a true multiplier applied in one
    call -- no need to repeat small steps. Returns the new scale, which is
    clamped to the viewer's zoom limits, so a very large factor saturates at
    max zoom and the returned scale may reflect less than the full multiply.
    Compare the returned scale to gauge how much room is left."""
    return await _call("viewer.zoom", _strip_none({"viewer": viewer, "factor": factor}))


@mcp.tool()
async def vc3d_rotate_viewer(
    plane: str, degrees: float, relative: bool = True
) -> dict[str, Any]:
    """Rotate an axis-aligned slice plane -- the same rotation a human gets by
    middle-drag on the "seg xz" / "seg yz" panes. `plane` must be "seg xz" or
    "seg yz" (the "xz"/"yz" shorthands are accepted); only these two planes
    rotate -- the main xy/segment view is not rotatable. By default `degrees` is
    a relative delta added to the current angle (positive = same sense as
    dragging up); pass relative=False to set an absolute angle. Requires
    axis-aligned slice mode to be active. Returns the new and previous angles in
    degrees."""
    return await _call(
        "viewer.rotate", {"plane": plane, "degrees": degrees, "relative": relative}
    )


@mcp.tool()
async def vc3d_set_axis_aligned_slices(enabled: bool) -> dict[str, Any]:
    """Enable or disable axis-aligned slice mode -- the same checkbox (and its
    keyboard shortcut) a human toggles to make "seg xz"/"seg yz" the rotatable
    canonical slice planes. This is the prerequisite for vc3d_rotate_viewer,
    which errors when the mode is off. Toggling is idempotent and persists the
    setting exactly like the human path. Returns {"enabled": bool} with the
    resulting mode state. Read the current state via vc3d_get_state's
    "axisAlignedSlices" field."""
    return await _call("viewer.set_axis_aligned_slices", {"enabled": enabled})


@mcp.tool()
async def vc3d_drag(
    from_point: dict[str, float],
    to_point: dict[str, float],
    viewer: Optional[str] = None,
    space: str = "volume",
    button: str = "left",
    modifiers: Optional[list[str]] = None,
    steps: int = 8,
) -> dict[str, Any]:
    """Synthesize a full press-move-release drag in a viewer, from one point to
    another (the twin of a human click-and-drag). Dispatched through the
    viewer's real mouse slots, so every signal fires exactly as for a hand drag
    (no synthetic onVolumeClicked -- a drag is not a click).

    from_point / to_point: the drag endpoints. {"x","y","z"} in volume space
    (default), or {"x","y"} scene-space when space="scene". In volume space both
    endpoints are round-trip validated against the target viewer's current view
    (same rule as vc3d_click); an off-view endpoint fails -32003 with data.point
    naming the offender ("from" or "to").
    viewer: viewer id ("v1") or surface-slot name; default "segmentation". Must
    resolve to a chunked volume viewer, else -32009.
    space: "volume" | "scene"; applies to both endpoints.
    button: "left" | "right" | "middle", or "none" for a hover-only positioning
    drag -- press/release are skipped and only the interpolated move events fire
    with no button held. button="none" is the cursor-placement primitive
    vc3d_push_pull_start depends on (hover onto the target vertex, then start).
    modifiers: any of "shift", "ctrl", "alt", "meta", "keypad".
    steps: number of interpolated move events, silently clamped to [1, 256]
    (default 8); a non-integer or a value < 1 is rejected -32602.

    Returns {"dragged": true, "from"/"to": {"scene": {"x","y"}, "volumePoint":
    Vec3 | null (null when off-surface)}, "steps", "button", "modifiers"}. A
    drag longer than 1.0 voxel over the surface while correction-point mode is
    active (see vc3d_corrections_set_point_mode) commits an anchored correction
    and auto-triggers the corrections solver (a source:"growth" job -- poll
    vc3d_job_status before further editing RPCs).
    """
    return await _call(
        "canvas.drag",
        _strip_none(
            {
                "viewer": viewer,
                "from": from_point,
                "to": to_point,
                "space": space,
                "button": button,
                "modifiers": modifiers if modifiers is not None else [],
                "steps": steps,
            }
        ),
    )
