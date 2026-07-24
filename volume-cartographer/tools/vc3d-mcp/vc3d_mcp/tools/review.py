"""Review-state-aware segment listing (segments.review)."""

from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict

from ..core import mcp, _call, _strip_none


class _ReviewFilter(TypedDict, total=False):
    unreviewed: bool
    approved: bool
    defective: bool
    hideDefective: bool
    reviewed: bool
    inspect: bool
    partialReview: bool


@mcp.tool()
async def vc3d_review_segments(
    only_loaded: bool = False,
    filter: Optional[_ReviewFilter] = None,
) -> dict[str, Any]:
    """List segments together with their review-tag state, with optional
    server-side filtering (the programmatic equivalent of the surface panel's
    review filter checkboxes).

    only_loaded: same meaning as vc3d_list_segments -- restrict to currently
    loaded surfaces.
    filter: an object of optional bools, ANDed together (an absent or false key
    contributes nothing, same as an unchecked checkbox):
      - unreviewed: keep only segments WITHOUT the "reviewed" tag.
      - approved: keep only segments WITH the "approved" tag.
      - defective: keep only segments WITH the "defective" tag.
      - hideDefective: keep only segments WITHOUT the "defective" tag.
      - reviewed: keep only segments WITH the "reviewed" tag.
      - inspect: keep only segments WITH the "inspect" tag.
      - partialReview: keep only segments WITH the "partial_review" tag.

    Returns {"segments": [{"id", "path", "loaded", "active", "tags", "reviewState"}],
    "total": int, "returned": int} -- "total" is the onlyLoaded-scoped candidate
    count, "returned" is the count after "filter" is applied.
    """
    return await _call(
        "segments.review", _strip_none({"onlyLoaded": only_loaded, "filter": filter})
    )
