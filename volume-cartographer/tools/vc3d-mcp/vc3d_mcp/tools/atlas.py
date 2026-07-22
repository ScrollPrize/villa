"""Fiber atlas open/search/remap tools.

Split out of the original monolithic ``server.py``; each tool registers on
the single shared ``mcp`` instance from ``vc3d_mcp.core``.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core import mcp, _call, _strip_none

__all__ = [
    "vc3d_atlas_open",
    "vc3d_atlas_status",
    "vc3d_atlas_search_start",
    "vc3d_atlas_search_cancel",
    "vc3d_atlas_search_results",
    "vc3d_atlas_open_result",
    "vc3d_atlas_remap",
    "vc3d_atlas_optimize_snap_candidates",
]


@mcp.tool()
async def vc3d_atlas_open(atlas_dir: str) -> dict[str, Any]:
    """Open (load and display) a fiber atlas from a directory.

    atlas_dir: absolute path, or relative to the open volume package's root.
    Never shows a dialog or the interactive rebuild prompt. Returns
    {"opened": true, "atlasDir", "atlasName"}.
    """
    return await _call("atlas.open", {"atlasDir": atlas_dir})


@mcp.tool()
async def vc3d_atlas_status() -> dict[str, Any]:
    """Report the current atlas (dir/name, null when none is open) and the
    fiber-intersection search state: {"search": {"running", "phase",
    "phaseCount": 5, "completed", "total", "cancelRequested", "resultCount"}}.
    """
    return await _call("atlas.status", {})


@mcp.tool()
async def vc3d_atlas_search_start(
    mode: str = "atlas_to_non_atlas",
    required_tags: Optional[list[str]] = None,
    excluded_tags: Optional[list[str]] = None,
    max_distance: Optional[float] = None,
) -> dict[str, Any]:
    """Start an asynchronous atlas fiber-intersection search (a source:"atlas"
    job -- poll vc3d_job_status, then page results with vc3d_atlas_search_results).

    mode: "atlas_to_non_atlas" (requires an open atlas with fiber mappings) or
    "non_atlas_only". required_tags/excluded_tags filter the candidate fibers.
    max_distance: broad-phase segment radius in voxels; omit to keep the
    current spin-box value. Returns {"jobId", "kind": "atlas.fiber_search",
    "source": "atlas"}.
    """
    return await _call(
        "atlas.search_start",
        _strip_none(
            {
                "mode": mode,
                "requiredTags": required_tags,
                "excludedTags": excluded_tags,
                "maxDistance": max_distance,
            }
        ),
    )


@mcp.tool()
async def vc3d_atlas_search_cancel() -> dict[str, Any]:
    """Request cancellation of the running atlas fiber-intersection search.
    The job still terminates through its finished notification
    (success: false). Returns {"cancelRequested": true}."""
    return await _call("atlas.search_cancel", {})


@mcp.tool()
async def vc3d_atlas_search_results(offset: int = 0, limit: int = 100) -> dict[str, Any]:
    """Page through the completed atlas search results (vector order).

    limit is clamped to [1, 1000]. Each row carries "index" (the id
    vc3d_atlas_open_result takes), source/target fiber ids (as strings),
    candidateDistance, refinedScore, windingDistance (null when infinite),
    signedWinding (null when absent), source/target points and arclengths,
    converged and message. Returns {"total", "offset", "results": [...]}.
    """
    return await _call("atlas.search_results", {"offset": offset, "limit": limit})


@mcp.tool()
async def vc3d_atlas_open_result(index: int) -> dict[str, Any]:
    """Open one atlas search result in the intersections inspection workspace.

    index: a result "index" as returned by vc3d_atlas_search_results (vector
    order). Returns {"opened": true, "index"}.
    """
    return await _call("atlas.open_result", {"index": index})


@mcp.tool()
async def vc3d_atlas_remap() -> dict[str, Any]:
    """Rebuild the current atlas's fiber mappings from its source fiber JSON
    (asynchronous; the atlas is redisplayed on completion and progress is
    visible in app status messages). Returns {"remapped": true} once the remap
    worker is launched."""
    return await _call("atlas.remap", {})


@mcp.tool()
async def vc3d_atlas_optimize_snap_candidates() -> dict[str, Any]:
    """Queue Laplace snap-candidate ranking for the current atlas through the
    lasagna fit service (requires the service to be available). Completion is
    observable via app status messages; not modeled as a job. Returns
    {"requested": true}."""
    return await _call("atlas.optimize_snap_candidates", {})
