"""Viewer + canvas interaction tools (cursor point, click, drag, center, zoom, rotate)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from ..core import mcp, _call, _strip_none


class _Vec3(TypedDict):
    x: float
    y: float
    z: float


class _Point(TypedDict):
    x: float
    y: float
    z: NotRequired[float]


class _ScenePoint(TypedDict):
    x: float
    y: float


_Modifiers = list[Literal["shift", "ctrl", "alt", "meta", "keypad"]]


class _Window(TypedDict):
    low: float
    high: float


class _OverlayComposite(TypedDict, total=False):
    enabled: bool
    method: Literal["max", "mean", "min"]
    layersFront: int
    layersBehind: int


_OverlayColormap = Literal[
    "",
    "fire",
    "viridis",
    "magma",
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "glasbey_black0",
]


@mcp.tool()
async def vc3d_get_cursor_point(
    viewer: Optional[str] = None,
    scene: Optional[_ScenePoint] = None,
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
    position: _Point,
    viewer: Optional[str] = None,
    space: Literal["volume", "scene"] = "volume",
    button: Literal["left", "right", "middle"] = "left",
    modifiers: Optional[_Modifiers] = None,
) -> dict[str, Any]:
    """Synthesize a mouse click in a viewer at a volume-space (or scene-space)
    position, with button and modifiers (e.g. modifiers=["shift"] to place a
    point / set focus).

    position: {"x","y","z"} in volume space (default), or {"x","y"} in scene
    space when space="scene". A volume point must lie on the viewer's current
    view or the click fails -32003 ("point is not on this viewer's view"); if
    you lack a known on-view point, vc3d_center_viewer there first or use
    space="scene" with on-screen pixel coordinates.
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
    position: _Point,
    viewer: Optional[str] = None,
    space: Literal["volume", "scene"] = "volume",
    button: Literal["left", "right", "middle"] = "left",
    modifiers: Optional[_Modifiers] = None,
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
    point: _Vec3, viewer: Optional[str] = None, force_render: bool = True
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
    plane: Literal["seg xz", "seg yz", "xz", "yz"],
    degrees: float,
    relative: bool = True,
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
async def vc3d_get_render_settings() -> dict[str, Any]:
    """Read the global viewer render settings (surface-intersection overlays,
    opacities, normals/direction hints, highlighted surfaces).

    Use this to inspect the current rendering state before tweaking it with
    vc3d_set_render_settings, or to confirm a change took effect.

    Returns the full settings object: {"intersectionOpacity" (0..1),
    "intersectionThickness" (0..100), "overlayOpacity" (0..1),
    "intersectionMaxSurfaces" (int >=0), "volumeWindow" ({"low","high"}),
    "samplingStride" (int >=1), "zScrollSensitivity" (0.1..100),
    "segmentationCursorMirroring" (bool),
    "planeIntersectionLinesVisible" (bool), "showSurfaceNormals" (bool),
    "showDirectionHints" (bool), "surfaceOverlayEnabled" (bool),
    "normalArrowLengthScale" (float), "normalMaxArrows" (int),
    "highlightedSurfaceIds" ([str...])}."""
    return await _call("viewer.get_render_settings")


@mcp.tool()
async def vc3d_set_render_settings(
    intersection_opacity: Optional[float] = None,
    intersection_thickness: Optional[float] = None,
    overlay_opacity: Optional[float] = None,
    intersection_max_surfaces: Optional[int] = None,
    plane_intersection_lines_visible: Optional[bool] = None,
    show_surface_normals: Optional[bool] = None,
    show_direction_hints: Optional[bool] = None,
    surface_overlay_enabled: Optional[bool] = None,
    highlighted_surface_ids: Optional[list[str]] = None,
    volume_window: Optional[_Window] = None,
    normal_arrow_length_scale: Optional[float] = None,
    normal_max_arrows: Optional[int] = None,
    segmentation_cursor_mirroring: Optional[bool] = None,
    sampling_stride: Optional[int] = None,
    z_scroll_sensitivity: Optional[float] = None,
) -> dict[str, Any]:
    """Update the global viewer render settings; changes apply across all
    viewers. Every argument is optional -- omitted (None) fields are left
    unchanged; pass only what you want to change.

    intersection_opacity: surface-intersection overlay opacity, clamped 0..1.
    intersection_thickness: intersection line thickness, clamped 0..100.
    overlay_opacity: surface overlay opacity, clamped 0..1.
    intersection_max_surfaces: cap on intersecting surfaces drawn, int >=0.
    plane_intersection_lines_visible: show plane/surface intersection lines.
    show_surface_normals: draw surface normal indicators.
    show_direction_hints: draw growth direction hints.
    surface_overlay_enabled: master toggle for the surface overlay.
    highlighted_surface_ids: replace the set of highlighted surface ids.
    volume_window: {"low", "high"} base-volume display window, clamped 0..255.
    normal_arrow_length_scale: surface-normal arrow length multiplier,
    clamped 0.1..2.0 (the GUI slider's range) and persisted like the slider.
    normal_max_arrows: cap on surface-normal arrows drawn per axis,
    clamped 4..100 (the GUI slider's range) and persisted like the slider.
    segmentation_cursor_mirroring: mirror the cursor position onto the
    segmentation viewer from other panes.
    sampling_stride: surface-patch intersection sampling stride, floored to 1.
    z_scroll_sensitivity: scroll-to-move-through-z sensitivity, clamped
    0.1..100.

    The numeric/global fields (opacities, thickness, max-surfaces, volume
    window, sampling stride, z-scroll sensitivity, cursor mirroring,
    normal-arrow length/count) are persisted (QSettings/ViewerManager) even
    with no viewer pane open. highlighted_surface_ids is retained by the
    surface panel and applies when viewers become available. The four
    per-viewer toggles (plane-intersection lines, show-normals,
    show-direction-hints, surface-overlay) are no-ops with zero viewers; their
    returned echo reports defaults until a viewer is open.

    Returns the full resulting settings object (same shape as
    vc3d_get_render_settings) so you can confirm what was applied."""
    return await _call(
        "viewer.set_render_settings",
        _strip_none(
            {
                "intersectionOpacity": intersection_opacity,
                "intersectionThickness": intersection_thickness,
                "overlayOpacity": overlay_opacity,
                "intersectionMaxSurfaces": intersection_max_surfaces,
                "planeIntersectionLinesVisible": plane_intersection_lines_visible,
                "showSurfaceNormals": show_surface_normals,
                "showDirectionHints": show_direction_hints,
                "surfaceOverlayEnabled": surface_overlay_enabled,
                "highlightedSurfaceIds": highlighted_surface_ids,
                "volumeWindow": volume_window,
                "normalArrowLengthScale": normal_arrow_length_scale,
                "normalMaxArrows": normal_max_arrows,
                "segmentationCursorMirroring": segmentation_cursor_mirroring,
                "samplingStride": sampling_stride,
                "zScrollSensitivity": z_scroll_sensitivity,
            }
        ),
    )


@mcp.tool()
async def vc3d_get_overlay() -> dict[str, Any]:
    """Read the current overlay-volume settings (the semi-transparent second
    volume rendered on top of the base volume).

    Returns {"volumeId" (str, "" when no overlay is set), "colormap" (str, ""
    when unset), "opacity" (0..1), "threshold" (0..255), "windowLow"/
    "windowHigh" (0..255), "maxDisplayedResolution" (int 0..5),
    "composite": {"enabled" (bool), "method" ("max"|"mean"|"min"),
    "layersFront"/"layersBehind" (int 0..64)}}."""
    return await _call("viewer.get_overlay")


@mcp.tool()
async def vc3d_set_overlay(
    volume_id: Optional[str] = None,
    clear: Optional[bool] = None,
    colormap: Optional[_OverlayColormap] = None,
    opacity: Optional[float] = None,
    threshold: Optional[float] = None,
    window: Optional[_Window] = None,
    max_displayed_resolution: Optional[int] = None,
    composite: Optional[_OverlayComposite] = None,
) -> dict[str, Any]:
    """Update the overlay-volume settings. Every argument is optional --
    omitted (None) fields are left unchanged; pass only what you want to
    change.

    volume_id: id of the volume to overlay (see vc3d_list_overlay_volumes /
    vc3d_list_attached_volumes). An empty string clears the overlay, same as
    clear=True. Unknown id raises -32007.
    clear: True clears the overlay volume (equivalent to volume_id="").
    colormap: colormap id, one of "fire", "viridis", "magma", "red", "green",
    "blue", "cyan", "magenta", "glasbey_black0"; empty string clears it.
    An unrecognized id raises -32602 rather than being silently ignored.
    opacity: overlay opacity, clamped 0..1.
    threshold: overlay display threshold, clamped 0..255.
    window: {"low", "high"} overlay display window, clamped 0..255.
    max_displayed_resolution: overlay resolution cap, clamped 0..5.
    composite: {"enabled", "method" ("max"|"mean"|"min"), "layersFront",
    "layersBehind"} -- any subset; merged over the current composite settings.
    An unrecognized method raises -32602.

    Returns the full resulting overlay settings object (same shape as
    vc3d_get_overlay). Note: ViewerManager re-validates the overlay volume's
    coordinate space against the base volume and may silently reject a
    mismatched volume_id -- check the echoed "volumeId" to confirm it stuck.
    """
    return await _call(
        "viewer.set_overlay",
        _strip_none(
            {
                "volumeId": volume_id,
                "clear": clear,
                "colormap": colormap,
                "opacity": opacity,
                "threshold": threshold,
                "window": window,
                "maxDisplayedResolution": max_displayed_resolution,
                "composite": composite,
            }
        ),
    )


@mcp.tool()
async def vc3d_list_overlay_volumes() -> dict[str, Any]:
    """List every volume id in the open package, for picking an overlay
    volume via vc3d_set_overlay. Not filtered by coordinate-space
    compatibility with the base volume -- vc3d_set_overlay re-validates and
    may silently reject a mismatched pick.

    Returns {"volumes": [{"id", "current" (bool, true for the base volume)}],
    "overlayVolumeId" (str, "" when no overlay is set)}."""
    return await _call("viewer.list_overlay_volumes")


@mcp.tool()
async def vc3d_set_intersects(
    surface_ids: list[str], viewer: Optional[str] = None
) -> dict[str, Any]:
    """Set which surfaces' intersection lines a viewer draws -- the same
    per-viewer state SurfacePanelController's filters apply.

    surface_ids: surface/slot ids to draw intersections for (e.g. "seg xz",
    "seg yz", or a segment id). "segmentation" is always included even if
    omitted here.
    viewer: viewer id or surface-slot name to target; if omitted, applies to
    every base viewer except the one whose surface slot is "segmentation"
    (mirrors the GUI's default, no-filter behavior). Targeting the
    "segmentation" viewer itself raises -32009.

    Returns {"surfaceIds": [str...] (the resulting set, with "segmentation"
    unioned in), "appliedToViewers": [viewer ids the set was applied to]}."""
    return await _call(
        "viewer.set_intersects", _strip_none({"viewer": viewer, "surfaceIds": surface_ids})
    )


@mcp.tool()
async def vc3d_drag(
    from_point: _Point,
    to_point: _Point,
    viewer: Optional[str] = None,
    space: Literal["volume", "scene"] = "volume",
    button: Literal["left", "right", "middle", "none"] = "left",
    modifiers: Optional[_Modifiers] = None,
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
