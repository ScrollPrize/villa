"""Geometry builders, diagnostics and caller decisions for object-store tests."""
import struct

from scripts.object_store_sqlite import sqlite_rows


def remove_spatial_schema(sql):
    """Restore the pre-v4 schema for migration fixtures, including its indexes."""
    sql.execute("DROP TABLE object_bounds")  # Also drops its maintenance triggers.
    sql.execute("DROP TABLE spatial_regions")
    sql.execute("DROP TABLE spatial_regions_rtree")
    sql.execute("DROP INDEX objects_kind_uuid")


def edge_choice(db, edge_uuid, action, **anchors):
    edge = db.edge(edge_uuid)
    revisions = [db.current_geometry(edge[side]) for side in ("a", "b")]
    return dict(edge_uuid=edge_uuid, expected_generation=edge["generation"], action=action,
                a_revision=revisions[0]["uuid"] if revisions[0] else None,
                b_revision=revisions[1]["uuid"] if revisions[1] else None, **anchors)


def canonical_edge(edge):
    edge = dict(edge)
    if edge["b"] < edge["a"]:
        edge["a"], edge["b"] = edge["b"], edge["a"]
        edge["a_anchor"], edge["b_anchor"] = edge["b_anchor"], edge["a_anchor"]
    return edge


ALL = [-1e101] * 3 + [1e101] * 3


def _geometry(kind="fiber", value=1.0):
    return {"kind": kind, "arrays": [
        {"name": "control_points", "dtype": "<f8", "shape": [0, 3], "data": b""},
        {"name": "line_points", "dtype": "<f8", "shape": [1, 3],
         "data": struct.pack("<ddd", value, 2.0, 3.0)}]}


def track(points):
    return dict(kind="track", arrays=[dict(name="points", dtype="<i4", shape=[len(points), 3],
                data=struct.pack(f"<{len(points)*3}i", *(v for p in points for v in p)))])


def patch(rows, cols, points=None):
    if points is None:
        points = [(float(x), float(y), 1.) for y in range(rows) for x in range(cols)]
    return dict(kind="patch", attributes={"scale": [1, 1]}, arrays=[dict(
        name="points", dtype="<f4", shape=[rows, cols, 3],
        data=struct.pack(f"<{len(points)*3}f", *(v for p in points for v in p)))])
