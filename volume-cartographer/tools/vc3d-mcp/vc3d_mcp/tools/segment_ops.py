"""Segment file operations: grid transforms, exports, neighbor copy, merge and
patch, backup restore, and moving traces to paths."""

from __future__ import annotations

from typing import Any, Literal

from ..core import mcp, _call

# Synchronous file operations save a whole segment before the bridge replies.
SEGMENT_FILE_OP_TIMEOUT_S = 120.0


@mcp.tool()
async def vc3d_transform_segment(
    segment_id: str,
    op: Literal["flip_u", "flip_v", "rotate90"],
    confirm: bool = False,
) -> dict[str, Any]:
    """Flip or rotate a segment's surface grid in place, the headless twin of the
    segment context menu's "Flip Surface" and "Rotate Surface 90° CW" entries.
    Synchronous: the transformed surface is saved before the call returns.

    segment_id: id of the segment to transform.
    op: "flip_u" reverses the rows (the menu's "V Flip"); "flip_v" reverses
    the columns (the menu's "H Flip", which also reverses the surface normals,
    so it is the menu's "Normals (H Flip)" too); "rotate90" rotates the grid by
    90 degrees clockwise.
    confirm: required (true) for "rotate90", which resamples the grid, so four
    rotations do not give back the original. Flips are exact and their own
    inverse and need no confirm.

    Not idempotent: a second flip undoes the first. If a call times out, read
    the segment before retrying. Requires a loaded volume package and current
    volume. Refused (-32004) for the segment being edited while segmentation
    editing is on, while a mask render runs, and while a source:"tool" job
    runs. Returns {"transformed": true, "segmentId", "op"}.
    """
    return await _call(
        "segment.transform",
        {"segmentId": segment_id, "op": op, "confirm": confirm},
        timeout=SEGMENT_FILE_OP_TIMEOUT_S,
    )
