"""Viewer display state outside the shared render settings: the ink detection
overlay and per-viewer view resets."""

from __future__ import annotations

from typing import Any, Literal, Optional

from ..core import mcp, _call, _strip_none


@mcp.tool()
async def vc3d_get_ink_overlay(refresh: bool = False) -> dict[str, Any]:
    """Read the ink detection overlay drawn over the active segment's flattened
    view, the state behind the Ink Detection panel.

    refresh: rescan the cached Open Data ink detections for the active segment
    before reporting.

    Returns {"available": [{"label", "path", "sampleId", "segmentId",
    "segmentLongId"}], "path" (selected detection or null), "loaded",
    "visible", "opacity" (0-1), "colormap" ("" is grayscale), "colormaps"
    (accepted colormap ids), "singleChannel", "flipHorizontal",
    "flipVertical"}. Only detections matching the active segment are
    available.
    """
    return await _call("viewer.get_ink_overlay", {"refresh": refresh})


@mcp.tool()
async def vc3d_set_ink_overlay(
    path: Optional[str] = None,
    opacity: Optional[float] = None,
    colormap: Optional[str] = None,
    flip_horizontal: Optional[bool] = None,
    flip_vertical: Optional[bool] = None,
) -> dict[str, Any]:
    """Change the ink detection overlay, the headless twin of the Ink Detection
    panel. Any subset; every field is validated before any is applied, and the
    panel follows the change.

    path: one of the "available" paths from vc3d_get_ink_overlay, or "" to
    clear the selection; another path fails with -32007 and data.available.
    opacity: 0-1 (the panel's slider shows percent); 0 hides the overlay.
    colormap: one of the reported "colormaps" ("" is grayscale); applies to
    single-channel detections.
    flip_horizontal / flip_vertical: mirror the overlay image.

    Returns the full overlay state, as vc3d_get_ink_overlay does.
    """
    return await _call(
        "viewer.set_ink_overlay",
        _strip_none(
            {
                "path": path,
                "opacity": opacity,
                "colormap": colormap,
                "flipHorizontal": flip_horizontal,
                "flipVertical": flip_vertical,
            }
        ),
    )


@mcp.tool()
async def vc3d_reset_viewer(
    viewer: Optional[str] = None,
    fit: Literal["surface", "content", "none"] = "surface",
    z_offset: Optional[float] = None,
) -> dict[str, Any]:
    """Reset one viewer's view, the headless twin of the Reset View shortcut
    (M) and, with fit="content", Zoom to Fit (Ctrl+0).

    viewer: viewer id or name (see vc3d_get_state); default the active one.
    fit: "surface" fits the surface in view (Reset View), "content" fits the
    current content (Zoom to Fit), "none" keeps the zoom.
    z_offset: normal offset to set after the reset, absolute, in voxels; the
    viewer clamps it to the volume's largest dimension. Omitted: 0.

    Pan and normal offset are always cleared first. Returns {"viewer", "fit",
    "zOffset" (the resulting offset), "scale"}.
    """
    return await _call(
        "viewer.reset",
        _strip_none({"viewer": viewer, "fit": fit, "zOffset": z_offset}),
    )
