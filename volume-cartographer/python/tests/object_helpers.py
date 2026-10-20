"""Geometry builders, diagnostics and caller decisions for object-store tests."""
import json
import struct
import subprocess
import sys


def sqlite_rows(path, statement, parameters=()):
    # Python and vc.objects can link different SQLite builds. Closing one
    # build's WAL connection in-process can invalidate the other's POSIX locks.
    code = "import json,sqlite3,sys; c=sqlite3.connect(sys.argv[1]); r=c.execute(sys.argv[2],json.loads(sys.argv[3])).fetchall(); c.commit(); print(json.dumps(r)); c.close()"
    return json.loads(subprocess.check_output(
        [sys.executable, '-c', code, str(path), statement, json.dumps(parameters)], text=True))


def remove_spatial_schema(sql):
    """Restore the pre-v4 schema for migration fixtures, including its indexes."""
    sql.execute("DROP TABLE object_bounds")  # Also drops its maintenance triggers.
    sql.execute("DROP TABLE spatial_regions")
    sql.execute("DROP TABLE spatial_regions_rtree")
    sql.execute("DROP INDEX objects_kind_uuid")


def remove_history_schema(sql):
    """Make a real pre-v9 fixture before lowering PRAGMA user_version."""
    for name, in sql.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE '%_history_%'").fetchall():
        sql.execute(f'DROP TRIGGER "{name}"')
    for table in ("object_relationship_deltas", "object_relationship_heads",
                  "relationship_revisions", "object_head_records", "object_heads", "object_revisions",
                  "history_dirty_objects", "history_dirty_edges", "history_expected",
                  "history_checked", "history_validated_edges", "history_required"):
        sql.execute(f"DROP TABLE {table}")


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
