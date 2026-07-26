"""Annotation point collection tools."""

from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

from ..core import mcp, _call, _strip_none


class Point3D(TypedDict):
    x: float
    y: float
    z: float


@mcp.tool()
async def vc3d_commit_points(
    collection: str, points: list[Point3D], winding: Optional[float] = None
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


@mcp.tool()
async def vc3d_add_point_collection(name: Optional[str] = None) -> dict[str, Any]:
    """Create an (empty) point collection. Omit name for an auto-generated one.
    Returns {collectionId, name}."""
    return await _call("points.add_collection", _strip_none({"name": name}))


@mcp.tool()
async def vc3d_update_point(
    point_id: int,
    position: Optional[Point3D] = None,
    winding: Optional[float] = None,
    clear_winding: bool = False,
) -> dict[str, Any]:
    """Update an existing point's volume-space position and/or winding
    annotation. Set clear_winding to remove an existing winding. A winding
    value and clear_winding cannot be supplied together. Returns the updated
    {id, position, winding}."""
    if winding is not None and clear_winding:
        raise ValueError("winding and clear_winding are mutually exclusive")

    params = _strip_none(
        {"pointId": point_id, "position": position, "winding": winding}
    )
    if clear_winding:
        params["winding"] = None
    return await _call(
        "points.update_point",
        params,
    )


@mcp.tool()
async def vc3d_remove_point(point_id: int) -> dict[str, Any]:
    """Remove a single point by id. Returns {removed: true}."""
    return await _call("points.remove_point", {"pointId": point_id})


@mcp.tool()
async def vc3d_clear_point_collection(
    collection: Optional[str] = None, collection_id: Optional[int] = None
) -> dict[str, Any]:
    """Delete a collection and all of its points (identify by name or id).
    Returns {cleared: true}."""
    return await _call(
        "points.clear_collection",
        _strip_none({"collection": collection, "collectionId": collection_id}),
    )


@mcp.tool()
async def vc3d_clear_all_points() -> dict[str, Any]:
    """Delete every point collection. Returns {cleared: true}."""
    return await _call("points.clear_all", {})


@mcp.tool()
async def vc3d_rename_point_collection(
    new_name: str,
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Rename a collection (identify by name or id). Returns {collectionId,
    name}."""
    return await _call(
        "points.rename_collection",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "newName": new_name,
            }
        ),
    )


@mcp.tool()
async def vc3d_set_point_collection_color(
    color: list[float],
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Set a collection's [r, g, b] float color (identify by name or id).
    Echoes {collectionId, color}."""
    return await _call(
        "points.set_collection_color",
        _strip_none(
            {"collection": collection, "collectionId": collection_id, "color": color}
        ),
    )


@mcp.tool()
async def vc3d_set_point_collection_metadata(
    absolute_winding_number: bool,
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Set a collection's absolute-winding-number metadata flag (identify by
    name or id). Echoes {collectionId, absoluteWindingNumber}."""
    return await _call(
        "points.set_collection_metadata",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "absoluteWindingNumber": absolute_winding_number,
            }
        ),
    )


@mcp.tool()
async def vc3d_set_point_collection_tag(
    key: str,
    value: str,
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Set a key/value tag on a collection (identify by name or id). Returns
    {ok: true}."""
    return await _call(
        "points.set_collection_tag",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "key": key,
                "value": value,
            }
        ),
    )


@mcp.tool()
async def vc3d_remove_point_collection_tag(
    key: str,
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Remove a tag by key from a collection (identify by name or id). Returns
    {ok: true}."""
    return await _call(
        "points.remove_collection_tag",
        _strip_none(
            {"collection": collection, "collectionId": collection_id, "key": key}
        ),
    )


@mcp.tool()
async def vc3d_set_point_windings_linked(
    linked_collection_ids: list[int],
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
) -> dict[str, Any]:
    """Set the collections whose windings are linked to this one (identify by
    name or id). Echoes {collectionId, linkedCollectionIds}."""
    return await _call(
        "points.set_windings_linked",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "linkedCollectionIds": linked_collection_ids,
            }
        ),
    )


@mcp.tool()
async def vc3d_auto_fill_windings(
    mode: Literal["none", "incremental", "decremental", "constant"],
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
    constant: Optional[float] = None,
) -> dict[str, Any]:
    """Auto-fill winding annotations across a collection (identify by name or
    id). mode: none|incremental|decremental|constant (constant uses `constant`).
    Returns {ok: true}."""
    return await _call(
        "points.auto_fill_windings",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "mode": mode,
                "constant": constant,
            }
        ),
    )


@mcp.tool()
async def vc3d_set_auto_fill_mode(
    mode: Literal["none", "incremental", "decremental", "constant"],
    collection: Optional[str] = None,
    collection_id: Optional[int] = None,
    constant: Optional[float] = None,
) -> dict[str, Any]:
    """Set (without applying) a collection's auto-fill mode for future points
    (identify by name or id). mode: none|incremental|decremental|constant.
    Returns {ok: true}."""
    return await _call(
        "points.set_auto_fill_mode",
        _strip_none(
            {
                "collection": collection,
                "collectionId": collection_id,
                "mode": mode,
                "constant": constant,
            }
        ),
    )


@mcp.tool()
async def vc3d_reset_windings() -> dict[str, Any]:
    """Reset winding numbers across all collections. Returns {ok: true}."""
    return await _call("points.reset_windings", {})


@mcp.tool()
async def vc3d_apply_anchor_offset(
    offset_x: float, offset_y: float
) -> dict[str, Any]:
    """Apply a grid offset to every collection's 2D anchor (surface-growth
    remapping). Returns {ok: true}."""
    return await _call(
        "points.apply_anchor_offset", {"offsetX": offset_x, "offsetY": offset_y}
    )


@mcp.tool()
async def vc3d_save_points_json(path: str) -> dict[str, Any]:
    """Save all point collections to a JSON file. Returns {saved: bool}."""
    return await _call("points.save_json", {"path": path})


@mcp.tool()
async def vc3d_load_points_json(path: str) -> dict[str, Any]:
    """Load point collections from a JSON file. Returns {loaded: bool}."""
    return await _call("points.load_json", {"path": path})


@mcp.tool()
async def vc3d_save_points_segment_path(segment_path: str) -> dict[str, Any]:
    """Save 2D-anchored correction points into a segment directory. Returns
    {saved: bool}."""
    return await _call("points.save_segment_path", {"segmentPath": segment_path})


@mcp.tool()
async def vc3d_load_points_segment_path(segment_path: str) -> dict[str, Any]:
    """Load 2D-anchored correction points from a segment directory. Returns
    {loaded: bool}."""
    return await _call("points.load_segment_path", {"segmentPath": segment_path})
