"""Spatial selection, projection, and persistence across geometry mutations."""
import json
import math
import random
import sqlite3
import struct

import pytest
from vc.objects import ObjectStore
from object_helpers import remove_spatial_schema


@pytest.fixture()
def db(tmp_path):
    with ObjectStore.create(str(tmp_path / "spatial.vcobj"), author="test") as store:
        yield store


def track(points):
    return dict(kind="track", arrays=[dict(name="points", dtype="<i4", shape=[len(points), 3],
                data=struct.pack(f"<{len(points)*3}i", *(v for p in points for v in p)))])


def identities(db, bbox, **kwargs):
    return [r["uuid"] for r in db.select_objects(bbox=bbox, fields=[], **kwargs)]


def test_shared_index_filters_and_projections(db):
    ids = []
    for kind in ("fiber", "patch", "track"):
        ids.append(db.upsert_object(dict(kind=kind, name=kind, tags=["visible"],
            meta={"large": "x" * 10000}, **{kind: dict(bbox=[1, 2, 3, 4, 5, 6])})))
    outside = db.upsert_object(dict(kind="track", track=dict(bbox=[10, 10, 10, 20, 20, 20])))
    db.upsert_object(dict(kind="note"))
    db.add_edge(dict(kind="chain", a=ids[0], b=ids[2]))
    box = [0, 0, 0, 4, 5, 6]
    assert identities(db, box) == sorted(ids)
    assert identities(db, box, object_kind="track", tag="visible") == [ids[2]]
    assert identities(db, box, connected_to=ids[0]) == sorted([ids[0], ids[2]])
    assert identities(db, box, object_uuids=[outside, ids[1]]) == [ids[1]]
    assert identities(db, box, limit=1, offset=1) == sorted(ids)[1:2]
    assert identities(db, box, limit=0) == []
    full = db.select_objects(bbox=box)
    for fields in ([], ["name"], ["tags", "meta"], ["fiber", "patch", "track"],
                   ["generation", "approval_state", "approved_by", "data_checksum"], list(full[0])):
        expected = [{k: v for k, v in row.items() if k in set(fields) | {"uuid", "kind"}} for row in full]
        assert db.select_objects(bbox=box, fields=fields) == expected
    with pytest.raises(ValueError, match="unknown object projection"):
        db.select_objects(fields=["typo"])
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.select_objects(bbox=box, fields=["name"]) == db.select_objects(bbox=box, fields=["name"])
    db.delete_object(ids[1])
    assert identities(db, box) == sorted([ids[0], ids[2]])
    with sqlite3.connect(db.path) as sql:
        assert sql.execute("SELECT count(*) FROM spatial_regions_rtree").fetchone()[0] == 3


def test_geometry_overrides_annotation_and_updates_atomically(db, tmp_path):
    uid = db.upsert_object(dict(kind="track", track=dict(bbox=[-10, -10, -10, -1, -1, -1])))
    box = [1, 2, 3, 1, 2, 3]
    first = db.put_geometry(uid, track([(1, 2, 3)]))
    assert identities(db, box) == [uid]
    assert identities(db, [-20, -20, -20, -1, -1, -1]) == []
    record = db.object(uid)
    record["track"]["bbox"] = [100, 100, 100, 200, 200, 200]
    db.upsert_object(record)
    assert identities(db, box) == [uid]
    with pytest.raises(RuntimeError, match="abort"):
        with db.write() as tx:
            tx.put_geometry(uid, track([(10, 20, 30)]), expected_revision=first)
            raise RuntimeError("abort")
    assert identities(db, box) == [uid]
    second = db.put_geometry(uid, track([(10, 20, 30)]), expected_revision=first)
    assert identities(db, box) == []
    assert identities(db, [10, 20, 30, 10, 20, 30]) == [uid]
    with ObjectStore.create(str(tmp_path / "peer.vcobj"), author="peer") as peer:
        assert db.synchronize(peer)["applied"]
        assert identities(peer, [10, 20, 30, 10, 20, 30]) == [uid]
        db.put_geometry(uid, track([]), expected_revision=second)
        assert db.synchronize(peer)["applied"]
        assert identities(peer, [-1000]*3 + [1000]*3) == []
    assert identities(db, [-1000]*3 + [1000]*3) == []


def test_geometry_kinds_and_invalid_patch_points(db):
    patch = db.upsert_object(dict(kind="patch"))
    db.put_geometry(patch, dict(kind="patch", attributes={"scale": [1, 1]}, arrays=[
        dict(name="points", dtype="<f4", shape=[1, 2, 3], data=struct.pack("<6f", -1, -1, -1, 4, 5, 6))]))
    collection = db.upsert_object(dict(kind="point_collection"))
    db.put_geometry(collection, dict(kind="point_collection", arrays=[
        dict(name="points", dtype="<f8", shape=[1, 3], data=struct.pack("<3d", 4, 5, 6)),
        dict(name="point_ids", dtype="<i8", shape=[1], data=struct.pack("<q", 42))]))
    fiber = db.upsert_object(dict(kind="fiber"))
    db.put_geometry(fiber, dict(kind="fiber", arrays=[
        dict(name="line_points", dtype="<f8", shape=[1, 3], data=struct.pack("<3d", 4, 5, 6)),
        dict(name="control_points", dtype="<f8", shape=[1, 3], data=struct.pack("<3d", 7, 8, 9))]))
    assert identities(db, [4, 5, 6, 4, 5, 6]) == sorted([patch, collection, fiber])
    assert identities(db, [-1]*6) == []
    assert identities(db, [7, 8, 9, 7, 8, 9]) == [fiber]


def test_rtree_rounding_matches_double_precision_brute_force(db):
    rng = random.Random(7)
    boxes = {}
    with db.write() as tx:
        for i in range(200):
            low = [rng.uniform(-100, 100) for _ in range(3)]
            high = [v + rng.random() * 10 for v in low]
            box = low + high
            uid = tx.upsert_object(dict(kind="track", track=dict(bbox=box)))
            boxes[uid] = box
        for value in (1.00000001, 1e100, -1e100):
            box = [value]*6
            boxes[tx.upsert_object(dict(kind="track", track=dict(bbox=box)))] = box
    queries = list(boxes.values()) + [[1]*6, [1.00000002]*6, [-100]*3 + [100]*3]
    for query in queries:
        expected = sorted(uid for uid, box in boxes.items()
                          if all(box[a] <= query[a+3] and box[a+3] >= query[a] for a in range(3)))
        assert identities(db, query) == expected
    for box in ([1, 0, 0, 0, 1, 1], [math.nan]*6, [-math.inf]*3 + [math.inf]*3):
        with pytest.raises(ValueError, match="finite, ordered"):
            db.select_objects(bbox=box)
        with pytest.raises(ValueError, match="finite, ordered"):
            db.upsert_object(dict(kind="track", track=dict(bbox=box)))


def test_projection_does_not_decode_unrequested_metadata(db):
    uid = db.upsert_object(dict(kind="track", name="lightweight", tags=["t"], track=dict(bbox=[0]*6)))
    # A deliberately damaged metadata payload must not be read/decoded by an
    # identity/name-only query. The ordinary full-record API still detects it.
    with sqlite3.connect(db.path) as sql:
        sql.execute("PRAGMA ignore_check_constraints=ON")
        sql.execute("UPDATE objects SET meta='invalid JSON'")
    assert db.select_objects(bbox=[0]*6, fields=["name"]) == [dict(uuid=uid, kind="track", name="lightweight")]
    with pytest.raises(json.JSONDecodeError):
        db.select_objects()


def test_failed_geometry_publication_restores_spatial_index_even_if_caught(db):
    uid = db.upsert_object(dict(kind="track"))
    first = db.put_geometry(uid, track([(1, 2, 3)]))
    with sqlite3.connect(db.path) as sql:
        sql.execute("CREATE TRIGGER fail_after_bounds BEFORE UPDATE ON objects "
                    "BEGIN SELECT RAISE(ABORT,'after spatial update'); END")
    with db.write() as tx:
        with pytest.raises(RuntimeError, match="after spatial update"):
            tx.put_geometry(uid, track([(9, 9, 9)]), expected_revision=first)
    assert identities(db, [1, 2, 3, 1, 2, 3]) == [uid]
    assert identities(db, [9]*6) == []
    assert db.current_geometry(uid)["uuid"] == first
    old_regions = db.spatial_regions([1, 2, 3, 1, 2, 3])
    assert len(old_regions) == 1 and old_regions[0]["revision_uuid"] == first
    assert db.spatial_regions([9]*6) == []


def test_annotation_update_removal_and_sync(db, tmp_path):
    uid = db.upsert_object(dict(kind="fiber", fiber=dict(bbox=[0]*3 + [1]*3)))
    with ObjectStore.create(str(tmp_path / "annotations.vcobj"), author="peer") as peer:
        assert db.synchronize(peer)["applied"]
        record = db.object(uid)
        record["fiber"]["bbox"] = [10]*3 + [11]*3
        db.upsert_object(record)
        assert db.synchronize(peer)["applied"]
        assert identities(peer, [0]*3 + [1]*3) == []
        assert identities(peer, [10]*3 + [11]*3) == [uid]
        record["fiber"].pop("bbox")
        db.upsert_object(record)
        assert db.synchronize(peer)["applied"]
        assert identities(peer, [10]*3 + [11]*3) == []


@pytest.mark.parametrize("version", [2, 3])
def test_migration_backfills_actual_geometry_and_preserves_identity(tmp_path, version):
    path = str(tmp_path / "legacy.vcobj")
    with ObjectStore.create(path, author="test") as db:
        uid = db.upsert_object(dict(kind="track", track=dict(bbox=[-100]*3 + [-90]*3)))
        db.put_geometry(uid, track([(1, 2, 3)]))
        registered = db.upsert_object(dict(kind="fiber", fiber=dict(bbox=[10]*3 + [20]*3)))
        before = db.export_json(), db.journal_since(), db.uuid
    with sqlite3.connect(path) as sql:
        remove_spatial_schema(sql)
        if version == 2:
            sql.execute("DROP TABLE sync_baseline_records")
            sql.execute("DROP TABLE sync_baseline_cursors")
            sql.execute("DROP INDEX sync_baselines_peer_seq")
        sql.execute(f"PRAGMA user_version={version}")
    with ObjectStore.open(path, read_only=True) as db:
        assert db.select_objects(fields=[])
        with pytest.raises(RuntimeError, match="open the store writable"):
            db.select_objects(bbox=[0]*3 + [30]*3)
    with ObjectStore.open(path, author="test") as db:
        assert (db.export_json(), db.journal_since(), db.uuid) == before
        assert identities(db, [1, 2, 3, 1, 2, 3]) == [uid]
        assert identities(db, [10]*3 + [20]*3) == [registered]
        assert identities(db, [-100]*3 + [-90]*3) == []
        hits = db.spatial_regions([1, 2, 3, 1, 2, 3])
        assert len(hits) == 1 and hits[0]["object_uuid"] == uid
        assert hits[0]["array_name"] == "points" and hits[0]["shape"] == [1]
    with sqlite3.connect(path) as sql:
        assert sql.execute("PRAGMA user_version").fetchone()[0] == 5
        assert sql.execute("PRAGMA foreign_key_check").fetchall() == []


def test_failed_migration_leaves_legacy_store_unchanged(tmp_path):
    path = str(tmp_path / "corrupt-legacy.vcobj")
    with ObjectStore.create(path, author="test") as db:
        uid = db.upsert_object(dict(kind="track"))
        db.put_geometry(uid, track([(1, 2, 3)]))
    with sqlite3.connect(path) as sql:
        remove_spatial_schema(sql)
        sql.execute("PRAGMA user_version=3")
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    with pytest.raises(RuntimeError, match="corrupt payload"):
        ObjectStore.open(path, author="test")
    with sqlite3.connect(path) as sql:
        assert sql.execute("PRAGMA user_version").fetchone()[0] == 3
        assert sql.execute("SELECT name FROM sqlite_master WHERE name LIKE 'object_bounds%'").fetchall() == []
