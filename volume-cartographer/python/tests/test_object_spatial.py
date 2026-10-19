"""Spatial selection, projection, and persistence across geometry mutations."""
import concurrent.futures
import math
import random
import sqlite3
import struct
import uuid

import pytest
from object_helpers import ALL, patch, track
from vc.objects import ObjectStore


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


def test_spatial_typed_filters_run_before_region_limit(db):
    def uid(i):
        return str(uuid.UUID(int=i + 1))

    for i in range(8):
        db.upsert_object(dict(uuid=uid(i), kind='track', track={
            'family': 'H' if i % 2 else 'V', 'length': i,
            'bbox': [0, 0, 0, 10, 10, 10]}))
    kwargs = dict(bbox=[1, 1, 1, 2, 2, 2], object_kind='track', limit=1)
    # Prime the resident unfiltered spatial index, then ask for typed filters.
    assert db.spatial_regions(**kwargs)[0]['object_uuid'] == uid(0)
    assert db.spatial_regions(**kwargs, family='H', min_length=4)[0]['object_uuid'] == uid(5)
    assert db.select_objects(**kwargs, family='H', min_length=4)[0]['uuid'] == uid(5)


@pytest.mark.parametrize("budget", [0, 1, 4096, 16 * 1024 * 1024])
def test_resident_regions_match_sql_for_moving_queries_and_filters(db, budget):
    rng = random.Random(842)
    ids = []
    for n in range(12):
        uid = db.upsert_object(dict(kind="patch", tags=["visible"]))
        ids.append(uid)
        # Curved, overlapping surfaces, including an interior bulge.
        pts = [(x + n * 17, y - n * 9, 2 + (x*y) % 19) for y in range(65) for x in range(70)]
        pts[17*70+17] = (n * 100, -n * 40, 999)
        db.put_geometry(uid, patch(65, 70, pts))
    line = db.upsert_object(dict(kind="track"))
    db.put_geometry(line, track([(i, -i, 0) for i in range(600)]))
    ids.append(line)
    db.add_edge(dict(kind="chain", a=ids[0], b=line,
                     a_anchor=dict(whole_geometry=True), b_anchor=dict(whole_geometry=True)))
    for value in (1e100, -1e100, 1.00000001):
        db.upsert_object(dict(kind="fiber", fiber=dict(bbox=[value]*6)))
    with ObjectStore.open(db.path, read_only=True, spatial_cache_bytes=budget,
                          geometry_cache_bytes=0) as cached, \
         ObjectStore.open(db.path, read_only=True, spatial_cache_bytes=0) as sql:
        queries = [(ALL, {}), ([1.00000002]*6, {}), ([1e100]*6, {}), ([-1e100]*6, {}),
                   (ALL, dict(tag="visible")), (ALL, dict(connected_to=ids[0], edge_kinds=["chain"]))]
        for i in range(180):
            x, y, z = rng.randrange(-10, 260), rng.randrange(-120, 70), rng.randrange(25)
            box = [x, y, z, x+rng.randrange(65), y+rng.randrange(65), z+20]
            filters = {}
            if i % 3 == 0:
                filters["object_uuids"] = [ids[i % len(ids)].upper(), ids[0], ids[0]]
            if i % 5 == 0:
                filters["object_kind"] = "patch"
            if i % 4 == 0:
                filters.update(limit=i % 11, offset=i % 7)
            queries.append((box, filters))
        expected = [sql.spatial_regions(box, **filters) for box, filters in queries]
        def check(i):
            box, filters = queries[i]
            assert cached.spatial_regions(box, **filters) == expected[i]
        # Repeated and concurrent leases exercise all per-connection caches.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check, list(range(len(queries))) * 2))
        for filters in (dict(object_uuids=["bad"]), dict(offset=-1), dict(limit=-1)):
            with pytest.raises((ValueError, RuntimeError)):
                cached.spatial_regions(ALL, **filters)


def test_resident_regions_invalidate_on_external_commit_without_geometry_cache(db):
    uid = db.upsert_object(dict(kind="track", track=dict(bbox=[0]*6)))
    with ObjectStore.open(db.path, read_only=True, reader_pool_size=1,
                          geometry_cache_bytes=0) as cached:
        assert cached.spatial_regions([0]*6)[0]["object_uuid"] == uid
        with ObjectStore.open(db.path, author="other") as writer:
            record = writer.object(uid)
            record["track"]["bbox"] = [10, 0, 0]*2
            writer.upsert_object(record)
        assert cached.spatial_regions([0]*6) == []
        assert cached.spatial_regions([10, 0, 0]*2)[0]["object_uuid"] == uid
        db.delete_object(uid)
        assert cached.spatial_regions(ALL) == []
        # An empty resident index must be invalidated by later insertion too.
        new = db.upsert_object(dict(kind="patch", patch=dict(bbox=[0]*6)))
        assert cached.spatial_regions([0]*6)[0]["object_uuid"] == new
