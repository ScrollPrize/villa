"""Segmentation editing settings and approval-mask mode."""

from __future__ import annotations

from typing import Any, Literal, Optional

from typing_extensions import NotRequired, TypedDict

from ..core import mcp, _call, _strip_none


class _Brush(TypedDict, total=False):
    radius: float
    sigma: float


class _PushPullBrush(TypedDict, total=False):
    radius: float
    sigma: float
    step: float


class _ZRange(TypedDict):
    enabled: bool
    min: NotRequired[int]
    max: NotRequired[int]


class _DirectionField(TypedDict):
    path: str
    orientation: NotRequired[Literal["normal", "horizontal", "vertical"]]
    scale: NotRequired[int]
    weight: NotRequired[float]


class _ApprovalMask(TypedDict, total=False):
    show: bool
    brushRadius: float
    brushDepth: float
    opacity: float
    autoApprove: bool
    autoApproveRadius: float
    autoApproveThreshold: float
    autoApproveMaxDistance: float


@mcp.tool()
async def vc3d_get_segmentation_settings() -> dict[str, Any]:
    """Read the segmentation editing settings shown in the Segmentation panel:
    brushes, smoothing, growth options, patch tracer parameters, direction
    fields and the approval mask.

    Returns {"editingEnabled", "drag" / "line" ({"radius", "sigma"}),
    "pushPull" ({"radius", "sigma", "step"}), "editScale",
    "smoothingStrength", "smoothingIterations", "showHoverMarker",
    "growthMethod" ("tracer" | "corrections" | "patch_tracer", the method
    growth uses), "manualAddSelected" (the Manual Add method is selected; use
    vc3d_manual_add_begin / finish for it), "growthSteps", "growthScale",
    "growthDirections" (subset of up/down/left/right), "zRange"
    ({"enabled", "min", "max"}), "growthVolumeId" (or null),
    "patchTracer" (tracer parameter object), "directionFields"
    ([{"path", "orientation", "scale", "weight"}]), "approvalMask" ({"show",
    "mode": "off" | "approve" | "unapprove", "brushRadius", "brushDepth",
    "opacity" (0-1), "autoApprove", "autoApproveRadius",
    "autoApproveThreshold", "autoApproveMaxDistance"})}. Brush radii and
    sigmas are in grid steps, as in the panel. The reply can be passed back to
    vc3d_set_segmentation_settings unchanged.
    """
    return await _call("segmentation.get_settings")


@mcp.tool()
async def vc3d_set_segmentation_settings(
    drag: Optional[_Brush] = None,
    line: Optional[_Brush] = None,
    push_pull: Optional[_PushPullBrush] = None,
    edit_scale: Optional[float] = None,
    smoothing_strength: Optional[float] = None,
    smoothing_iterations: Optional[int] = None,
    show_hover_marker: Optional[bool] = None,
    growth_method: Optional[Literal["tracer", "corrections", "patch_tracer"]] = None,
    growth_steps: Optional[int] = None,
    growth_scale: Optional[int] = None,
    growth_directions: Optional[list[Literal["up", "down", "left", "right"]]] = None,
    z_range: Optional[_ZRange] = None,
    growth_volume_id: Optional[str] = None,
    patch_tracer: Optional[dict[str, Any]] = None,
    direction_fields: Optional[list[_DirectionField]] = None,
    approval_mask: Optional[_ApprovalMask] = None,
) -> dict[str, Any]:
    """Change segmentation editing settings, the headless twin of the
    Segmentation panel's controls. Any subset; omitted fields stay as they
    are. Values are clamped to the panel's ranges and the reply reports what
    was applied.

    drag / line / push_pull: brush {"radius", "sigma"} in grid steps;
    push_pull also takes "step".
    edit_scale, smoothing_strength, smoothing_iterations, show_hover_marker:
    the panel's editing options.
    growth_method, growth_steps, growth_scale (0-5, the tracer's working
    scale), growth_directions (an empty list allows all directions):
    defaults for the next growth (vc3d_grow_segment passes its own method,
    direction and steps).
    z_range: {"enabled", "min", "max"} limits growth to those slices; a bound
    left out keeps its current value, and min above max fails with -32602.
    growth_volume_id: the volume growth runs on; an id the project or the
    panel does not offer fails with -32007 before anything changes.
    patch_tracer: patch tracer parameters by their keys in
    vc3d_get_segmentation_settings; unknown keys or wrong types fail with
    -32602.
    direction_fields: replaces the direction field list
    ([{"path", "orientation", "scale", "weight"}]).
    approval_mask: {"show", "brushRadius", "brushDepth", "opacity" (0-1),
    "autoApprove", "autoApproveRadius", "autoApproveThreshold",
    "autoApproveMaxDistance"}; use vc3d_set_approval_mask_mode to paint.

    Settings are saved as preferences, like the panel's own edits. Everything
    that can fail is checked before any value changes.

    Returns the full settings, as vc3d_get_segmentation_settings does.
    """
    return await _call(
        "segmentation.set_settings",
        _strip_none(
            {
                "drag": drag,
                "line": line,
                "pushPull": push_pull,
                "editScale": edit_scale,
                "smoothingStrength": smoothing_strength,
                "smoothingIterations": smoothing_iterations,
                "showHoverMarker": show_hover_marker,
                "growthMethod": growth_method,
                "growthSteps": growth_steps,
                "growthScale": growth_scale,
                "growthDirections": growth_directions,
                "zRange": z_range,
                "growthVolumeId": growth_volume_id,
                "patchTracer": patch_tracer,
                "directionFields": direction_fields,
                "approvalMask": approval_mask,
            }
        ),
    )


@mcp.tool()
async def vc3d_set_approval_mask_mode(
    mode: Literal["off", "approve", "unapprove"],
    show: Optional[bool] = None,
) -> dict[str, Any]:
    """Choose what dragging paints on the active segment's approval mask, the
    headless twin of the B (approve) and N (unapprove) keys and the panel's
    paint toggles. Paint with vc3d_drag in a segment view afterward.

    mode: "approve", "unapprove", or "off" to stop painting.
    show: show (or hide) the approval mask first. A painting mode needs the
    mask shown; while it is hidden, pass show=true or the call fails with
    -32602 (param show).

    Requires an active segment for a painting mode (-32007). Returns
    {"mode", "show"} as applied.
    """
    return await _call(
        "segmentation.approval_mask.set_mode",
        _strip_none({"mode": mode, "show": show}),
    )


@mcp.tool()
async def vc3d_undo_approval_stroke() -> dict[str, Any]:
    """Undo the last approval-mask stroke, the headless twin of Ctrl+B while
    painting. Requires a painting mode (-32009 otherwise). Returns
    {"undone": bool}; false when there was no stroke to undo."""
    return await _call("segmentation.approval_mask.undo")
