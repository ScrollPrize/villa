"""Annotation point collection tools.

Split out of the original monolithic ``server.py``; each tool registers on
the single shared ``mcp`` instance from ``vc3d_mcp.core``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core import mcp, _call, _strip_none

__all__ = [
    "vc3d_commit_points",
    "vc3d_list_points",
]


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
