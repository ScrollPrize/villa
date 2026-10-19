"""Ordering, filtering and integrity contracts for optimized storage queries."""
import json
import sqlite3
import struct

import pytest
from vc.objects import ObjectStore


def test_selection_intersection_pagination_and_export(db):
    with db.write() as tx:
        ids = [tx.upsert_object(dict(kind="fiber", name=str(i), tags=["even"] if i % 2 == 0 else [],
                                    fiber={"family": "H", "width": i + 1}, meta={"i": i})) for i in range(12)]
        for i in range(1, 6):
            tx.add_edge(dict(kind="chain", a=ids[i - 1], b=ids[i]))
        tx.add_edge(dict(kind="other", a=ids[0], b=ids[2]))
    expected = sorted(ids[i] for i in (0, 2, 4))
    kwargs = dict(object_kind="fiber", tag="even", connected_to=ids[0], edge_kinds=["chain"])
    records = db.select_objects(**kwargs)
    assert [r["uuid"] for r in records] == expected
    assert records == [db.object(uid) for uid in expected]
    assert db.select_objects(**kwargs, limit=0) == []
    assert db.select_objects(**kwargs, limit=1, offset=1) == records[1:2]
    assert db.select_objects(**kwargs, offset=99) == []
    assert db.select_objects(**kwargs, object_uuids=ids[4:]) == [db.object(ids[4])]
    assert db.select_objects(object_uuids=[ids[0].upper(), ids[0]]) == [db.object(ids[0])]
    assert db.select_objects(object_uuids=["00000000-0000-4000-8000-000000000000"]) == []
    for args in (dict(limit=-1), dict(offset=-1)):
        with pytest.raises(ValueError):
            db.select_objects(**args)
    full = json.loads(db.export_json(object_uuids=ids[:3]))
    assert len(full["edges"]) == 3
    projected = json.loads(db.export_json(object_uuids=ids[:3], fields=["name"], edge_kinds=["chain"]))
    assert len(projected["edges"]) == 2
    assert all(set(r) == {"uuid", "kind", "name"} for r in projected["objects"])
    assert all(set(r) == {"uuid", "kind", "a", "b", "geometry_bindings"} for r in projected["edges"])
    assert db.select_objects(object_uuids=ids * 400) == db.select_objects()
    # Read-only connections support the same indexed selection and pagination.
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.select_objects(**kwargs, limit=2) == records[:2]


def test_neighbours_and_traversal_keep_edge_order_and_self_edges(db):
    a, b, c = [db.upsert_object(dict(kind=kind)) for kind in ("note", "track", "note")]
    db.add_edge(dict(kind="chain", a=c, b=b))
    db.add_edge(dict(kind="chain", a=b, b=a))
    db.add_edge(dict(kind="other", a=a, b=b))
    db.add_edge(dict(kind="chain", a=b, b=b,
                     a_anchor={"points": [{"array": "points", "index": [0]}]},
                     b_anchor={"points": [{"array": "points", "index": [1]}]}))
    assert db.neighbours(b) == [c, a, b]
    assert db.neighbours(b, edge_kind="chain", neighbour_kind="note") == [c, a]
    assert db.neighbours(b, edge_kind="other") == [a]
    assert db.component(b, edge_kinds=["chain", "other"]) == [b, c, a]
    assert db.reachable_within(b, 0) == [b]
    assert db.neighbours(b, edge_kind="absent") == []


def test_batched_geometry_ranges_preserve_integrity(db):
    uid = db.upsert_object(dict(kind="fiber"))
    payload = struct.pack("<3d", 1, 2, 3) * 100000
    revision = db.put_geometry(uid, dict(kind="fiber", arrays=[
        dict(name="control_points", dtype="<f8", shape=[0, 3], data=b""),
        dict(name="line_points", dtype="<f8", shape=[100000, 3], data=payload)]))
    ranges = [(1048570, 30), (0, 24), (24, 24), (len(payload), 0), (0, 24)]
    assert db.read_geometry_ranges(revision, "line_points", ranges) == [payload[o:o+n] for o, n in ranges]
    for ranges in ([(-1, 2)], [(0, -1)], [(len(payload), 1)]):
        with pytest.raises(IndexError):
            db.read_geometry_ranges(revision, "line_points", ranges)
    # A later operation must reverify bytes, even on the same pooled connection.
    with sqlite3.connect(db.path) as sql:
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    with pytest.raises(RuntimeError, match="corrupt payload"):
        db.read_geometry_ranges(revision, "line_points", [(0, 24)])
