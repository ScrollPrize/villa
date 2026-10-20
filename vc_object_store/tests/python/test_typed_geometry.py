import math
import pytest

from vc_object_store import Fiber, Patch, PointCollection, Track


def put(db, kind, geometry, **kwargs):
    uid = db.upsert_object({"kind": kind})
    db.put_geometry(uid, geometry, **kwargs)
    return uid


def test_owned_geometry_and_typed_write_transaction(db):
    collection = PointCollection([42, 7], [[-0.0, 2, 3], [4, 5, 6]])
    collection.validate()
    assert collection.point(7) == [4, 5, 6]
    assert collection.point(8) is None
    with db.write() as write:
        uid = write.upsert_object({"kind": "point_collection"})
        revision = write.put_geometry(uid, collection)
    with db.geometry_snapshot() as snapshot:
        view = snapshot.point_collection(uid)
        assert len(view) == 2
        assert view.revision["uuid"] == revision
        assert view.ids(0, 2) == [42, 7]
        assert view.points(1, 1) == [[4, 5, 6]]
        assert math.copysign(1, view.point(42)[0]) == -1
        assert view.load().ids == collection.ids
    with pytest.raises(RuntimeError, match="closed"):
        view.point(7)


def test_snapshot_edits_are_checked_and_keep_old_views(db):
    uid = put(db, "point_collection", PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]))
    with db.geometry_snapshot() as before:
        view = before.geometry(uid)
        edit = view.edit()
        edit.move_point(7, [7, 8, 9])
        revision = edit.commit()
        assert view.point(7) == [4, 5, 6]
        with db.geometry_snapshot() as after:
            assert after.point_collection(uid).point(7) == [7, 8, 9]
            assert after.point_collection(uid).revision["uuid"] == revision
        stale = view.edit()
        stale.move_point(42, [0, 0, 0])
        with pytest.raises(RuntimeError):
            stale.commit()
        with pytest.raises(RuntimeError, match="already committed"):
            edit.commit()
        with pytest.raises(ValueError, match="kind mismatch"):
            before.patch(uid)
        with pytest.raises(IndexError):
            view.points(-1, 1)
        with pytest.raises(IndexError):
            view.points(1, 2)


@pytest.mark.parametrize("tile_size", [0, 32, 128])
def test_patch_tiles_and_sparse_edits(db, tile_size):
    points = [[float(c), float(r), float(r + c + 1)] for r in range(35) for c in range(36)]
    points[0] = [-1, -1, -1]
    uid = put(db, "patch", Patch(35, 36, [0.5, 0.25], points), patch_tile_size=tile_size)
    with db.geometry_snapshot() as before:
        view = before.patch(uid)
        assert (view.rows, view.columns) == (35, 36)
        assert not view.valid(0, 0)
        tile = view.read_tile(30, 31, 4, 4)
        assert tile.scale == [0.5, 0.25]
        for r in range(4):
            for c in range(4):
                assert tile.vertex(r, c) == points[(r + 30) * 36 + c + 31]
        edit = view.edit()
        edit.move_vertex(32, 33, [10, 20, 30])
        edit.commit()
        assert view.vertex(32, 33) == points[32 * 36 + 33]
    with db.geometry_snapshot() as after:
        assert after.patch(uid).vertex(32, 33) == [10, 20, 30]


def test_fiber_track_owned_operations_and_views(db):
    fiber = Fiber([[0, 0, 0], [3, 4, 0]])
    assert fiber.length() == 5
    fiber.line_points = [[0, 0, 0], [0, 0, 2]]
    assert fiber.length() == 2
    fid = put(db, "fiber", fiber)
    track = Track([[-2**31, 0, 1], [2**31 - 1, 0, 1]])
    assert track.length() == 2**32 - 1
    tid = put(db, "track", track)
    with db.geometry_snapshot() as snapshot:
        f = snapshot.fiber(fid)
        assert f.control_point_count == f.line_point_count == 2
        assert f.load().length() == 2
        edit = f.edit()
        edit.move_control_point(1, [6, 8, 0])
        edit.move_line_point(1, [0, 0, 3])
        edit.commit()
        t = snapshot.track(tid)
        assert t.points(0, 2) == track.points
        edit = t.edit()
        edit.move_track_point(0, [0, 0, 1])
        edit.commit()
    with db.geometry_snapshot() as current:
        assert current.fiber(fid).control_points(1, 1) == [[6, 8, 0]]
        assert current.track(tid).points(0, 1) == [[0, 0, 1]]


def test_invalid_owned_geometry_does_not_publish(db):
    uid = db.upsert_object({"kind": "point_collection"})
    for value in [
        PointCollection([1, 1], [[0, 0, 0], [1, 1, 1]]),
        PointCollection([-1], [[0, 0, 0]]),
        PointCollection([1], []),
        PointCollection([1], [[float("nan"), 0, 0]]),
    ]:
        with pytest.raises(ValueError):
            db.put_geometry(uid, value)
        assert db.current_geometry(uid) is None
    with pytest.raises(ValueError):
        Patch(1, 1, [1, 1], [[1, 2, 0]]).validate()
    with pytest.raises(ValueError):
        Patch(1, 1, [0, 1], [[1, 2, 3]]).validate()


def test_views_keep_snapshot_alive_and_explicit_close_invalidates(db):
    uid = put(db, "track", Track([[1, 2, 3]]))
    view = db.geometry_snapshot().track(uid)
    assert view.points(0, 1) == [[1, 2, 3]]
    with pytest.raises(RuntimeError, match="active transaction"):
        db.close()
    del view
    snapshot = db.geometry_snapshot()
    view = snapshot.track(uid)
    edit = view.edit()
    snapshot.close()
    assert not snapshot.active
    with pytest.raises(RuntimeError, match="closed"):
        view.load()
    with pytest.raises(RuntimeError, match="closed"):
        edit.commit()
    with pytest.raises(RuntimeError, match="closed"):
        snapshot.__enter__()


def test_metadata_changes_also_conflict(db):
    uid = put(db, "track", Track([[1, 2, 3]]))
    with db.geometry_snapshot() as snapshot:
        edit = snapshot.track(uid).edit()
        edit.move_track_point(0, [4, 5, 6])
        db.set_meta(uid, "reviewed", True)
        with pytest.raises(RuntimeError):
            edit.commit()
    with db.geometry_snapshot() as snapshot:
        assert snapshot.track(uid).points(0, 1) == [[1, 2, 3]]


def test_relationship_decisions_are_required_and_atomic(db):
    a = put(db, "point_collection", PointCollection([42], [[1, 2, 3]]))
    b = put(db, "track", Track([[1, 2, 3]]))
    eid = db.add_edge({
        "kind": "contact", "a": a, "b": b,
        "a_anchor": {"points": [{"array": "point_ids", "index": [42]}]},
        "b_anchor": {"points": [{"array": "points", "index": [0]}]},
    })
    replacement = PointCollection([42], [[7, 8, 9]])
    assert [e["uuid"] for e in db.affected_geometry_edges(a, replacement)] == [eid]
    with db.write() as write:
        assert [e["uuid"] for e in write.affected_geometry_edges(a, replacement)] == [eid]
    with db.geometry_snapshot() as snapshot:
        edit = snapshot.point_collection(a).edit()
        edit.move_point(42, [7, 8, 9])
        assert [e["uuid"] for e in edit.affected_relationships()] == [eid]
        with pytest.raises(RuntimeError):
            edit.commit()
        edge = db.edge(eid)
        edit.commit([{
            "edge_uuid": eid, "expected_generation": edge["generation"], "action": "remove",
            "a_revision": db.current_geometry(edge["a"])["uuid"],
            "b_revision": db.current_geometry(edge["b"])["uuid"],
        }])
    assert db.edge(eid) is None
    with db.geometry_snapshot() as snapshot:
        assert snapshot.point_collection(a).point(42) == [7, 8, 9]
