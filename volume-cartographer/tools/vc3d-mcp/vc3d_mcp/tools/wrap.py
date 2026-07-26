"""Same-winding wrap annotation tools (the tutorial's shift+E workflow).

Workflow (mirrors the human interaction):

1. ``vc3d_set_wrap_annotation_mode(True)`` — enable "Same-wrap annotation mode".
2. ``vc3d_shift_click(...)`` one or more times on a **chunked** volume viewer
   pane to seed preview points (this is the shift-click that seeds the preview,
   not a point placement).
3. ``vc3d_commit_wrap_annotation()`` — commit the previewed points into the
   point collection (the tutorial's shift+E).

``vc3d_undo_wrap_annotation()`` is the Ctrl+Z equivalent (clears the preview /
undoes the last committed collection). Only chunked-viewer panes participate;
other panes are ignored.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core import mcp, _call, _strip_none


@mcp.tool()
async def vc3d_set_wrap_annotation_mode(enabled: bool) -> dict[str, Any]:
    """Enable or disable "Same-wrap annotation mode" (the checkbox in the Wrap
    Annotation panel).

    This is the prerequisite for the shift+E commit workflow: with the mode on,
    a shift-click on a chunked volume viewer pane (vc3d_shift_click) seeds
    same-wrap preview points, and vc3d_commit_wrap_annotation (shift+E) commits
    them into the point collection.

    enabled: True to turn the mode on, False to turn it off.

    Returns {"enabled": bool} — the effective mode state after the call.
    """
    return await _call("wrap_annotation.set_mode", {"enabled": enabled})


@mcp.tool()
async def vc3d_commit_wrap_annotation(viewer: Optional[str] = None) -> dict[str, Any]:
    """Commit the seeded same-wrap annotation preview into the point collection
    -- the tutorial's shift+E.

    Requires "Same-wrap annotation mode" to be enabled first
    (vc3d_set_wrap_annotation_mode) and a preview to have been seeded via
    vc3d_shift_click on a chunked volume viewer pane; without a preview the
    commit is a no-op and returns committed=false.

    viewer: optional viewer id ("v1") or surface-slot name to commit on. When
    omitted, every base viewer is scanned and the first chunked pane with a
    preview to commit wins (exactly like the shift+E key handler). Must resolve
    to a chunked volume viewer, else -32009.

    Returns {"committed": bool, "hadPreview": bool}. Errors: -32002 if same-wrap
    annotation mode is not enabled.
    """
    return await _call("wrap_annotation.commit", _strip_none({"viewer": viewer}))


@mcp.tool()
async def vc3d_undo_wrap_annotation(viewer: Optional[str] = None) -> dict[str, Any]:
    """Undo the same-wrap annotation -- the Ctrl+Z equivalent: clears an
    uncommitted preview, or undoes the last committed same-wrap collection.

    viewer: optional viewer id ("v1") or surface-slot name to act on. When
    omitted, every base viewer is scanned and the first chunked pane that
    reports an undo wins. Must resolve to a chunked volume viewer, else -32009.

    Returns {"undone": bool}.
    """
    return await _call("wrap_annotation.undo", _strip_none({"viewer": viewer}))
