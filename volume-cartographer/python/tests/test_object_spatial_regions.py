"""Fine spatial broad phase: coverage, bounded reads and revision consistency."""
import math
import random
import sqlite3
import struct

import pytest
from vc.objects import ObjectStore
from test_object_spatial import db, track


ALL = [-1e101] * 3 + [1e101] * 3


def patch(rows, cols, points=None):
    if points is None:
        points = [(float(x), float(y), 1.) for y in range(rows) for x in range(cols)]
    return dict(kind="patch", attributes={"scale": [1, 1]}, arrays=[dict(
        name="points", dtype="<f4", shape=[rows, cols, 3],
        data=struct.pack(f"<{len(points)*3}f", *(v for p in points for v in p)))])


def regions(db, uid, bbox=ALL, **filters):
    return db.spatial_regions(bbox, object_uuids=[uid], **filters)


def read_tile(db, region, cols):
    y, x = region["start"]
    h, w = region["shape"]
    return db.read_geometry_ranges(region["revision_uuid"], region["array_name"],
        [((row * cols + x) * 12, w * 12) for row in range(y, y+h)])


def test_surface_boundary_vertices_bulges_and_revision_bound_loading(db):
    uid = db.upsert_object(dict(kind="patch"))
    points = [(x, y, 1) for y in range(66) for x in range(70)]
    points[17*70+17] = (17, 17, 999)  # Interior extrema cannot be sampled away.
    first = db.put_geometry(uid, patch(66, 70, points))
    all_tiles = regions(db, uid)
    assert len(all_tiles) == 9
    assert all(r["revision_uuid"] == first for r in all_tiles)
    assert all_tiles[-1]["start"] == [64, 64]
    assert all_tiles[-1]["shape"] == [2, 6]
    assert {tuple(r["start"]) for r in regions(db, uid, [32, 32, 1]*2)} == {
        (0, 0), (0, 32), (32, 0), (32, 32)}
    bulge = regions(db, uid, [17, 17, 999]*2)
    assert len(bulge) == 1 and bulge[0]["start"] == [0, 0]
    old_rows = read_tile(db, bulge[0], 70)
    assert struct.unpack_from("<3f", old_rows[17], 17*12) == (17, 17, 999)
    second = db.put_geometry(uid, patch(66, 70), expected_revision=first)
    assert regions(db, uid, [17, 17, 999]*2) == []
    assert read_tile(db, bulge[0], 70) == old_rows  # Immutable revision, despite new head.
    assert all(r["revision_uuid"] == second for r in regions(db, uid))
    # Whole-object API keeps its documented aggregate-AABB semantics.
    assert db.select_objects(bbox=[17, 17, 1]*2, fields=[]) == [dict(uuid=uid, kind="patch")]


@pytest.mark.parametrize("rows,cols", [(0, 8), (8, 0), (1, 1), (1, 66), (66, 1), (32, 33), (33, 33)])
def test_surface_empty_degenerate_and_exact_tile_sizes(db, rows, cols):
    uid = db.upsert_object(dict(kind="patch"))
    db.put_geometry(uid, patch(rows, cols))
    rs = regions(db, uid)
    if not rows or not cols:
        assert rs == []
        return
    covered = {(y, x) for r in rs for y in range(r['start'][0], r['start'][0]+r['shape'][0])
               for x in range(r['start'][1], r['start'][1]+r['shape'][1])}
    assert covered == {(y, x) for y in range(rows) for x in range(cols)}
    assert len(rs) == math.ceil(max(1, rows-1)/32) * math.ceil(max(1, cols-1)/32)


def test_surface_random_triangle_coverage_with_holes_and_both_validity_rules(db):
    rng = random.Random(144)
    rows, cols = 67, 71
    pts = [(rng.randrange(-300, 300), rng.randrange(-300, 300), rng.randrange(-20, 20))
           for _ in range(rows * cols)]
    for i in range(0, len(pts), 41):
        pts[i] = (-1, -1, -1)
    for i in range(13, len(pts), 97):
        pts[i] = (math.nan, 1, 5)
    uid = db.upsert_object(dict(kind="patch"))
    db.put_geometry(uid, patch(rows, cols, pts))
    for _ in range(150):
        y, x = rng.randrange(rows-1), rng.randrange(cols-1)
        # Both diagonals are covered; this does not depend on VC3D tessellation.
        for offsets in ((0, 1, cols), (0, 1, cols+1), (0, cols, cols+1), (1, cols, cols+1)):
            triangle = [pts[y*cols+x+d] for d in offsets]
            finite = all(math.isfinite(v) for p in triangle for v in p)
            valid_raw = all(p[0] != -1 for p in triangle)
            valid_mapped = all(p[2] > 0 for p in triangle)
            if not finite or not (valid_raw or valid_mapped):
                continue
            center = [sum(p[a] for p in triangle)/3 for a in range(3)]
            candidates = regions(db, uid, center*2)
            assert any(r['start'][0] <= y < r['start'][0]+r['shape'][0]-1 and
                       r['start'][1] <= x < r['start'][1]+r['shape'][1]-1 for r in candidates)
    invalid = [(-1, -1, -1), (-1, 4, 0), (math.inf, 0, 4)]
    rev = db.current_geometry(uid)['uuid']
    db.put_geometry(uid, patch(1, 3, invalid), expected_revision=rev)
    assert regions(db, uid) == []


def test_polyline_blocks_cover_crossing_segments_and_share_only_endpoints(db):
    uid = db.upsert_object(dict(kind="track"))
    pts = [(0, 0, 0)] * 514
    pts[256] = (1000, 0, 0)
    rev = db.put_geometry(uid, track(pts))
    rs = regions(db, uid, [500, 0, 0]*2)
    assert [(r['start'], r['shape']) for r in rs] == [([0], [257]), ([256], [257])]
    assert [r['shape'] for r in regions(db, uid)] == [[257], [257], [2]]
    # Every segment belongs to exactly one block; shared endpoints avoid gaps.
    segments = [i for r in regions(db, uid) for i in range(r['start'][0], r['start'][0]+r['shape'][0]-1)]
    assert segments == list(range(513))
    db.put_geometry(uid, track([(2, 3, 4)]), expected_revision=rev)
    assert [(r['start'], r['shape']) for r in regions(db, uid)] == [([0], [1])]
    assert regions(db, uid, [500, 0, 0]*2) == []


def test_fiber_arrays_point_blocks_and_exact_double_bounds(db):
    fiber = db.upsert_object(dict(kind="fiber"))
    arrays = [dict(name=name, dtype='<f8', shape=[1, 3], data=struct.pack('<3d', value, value, value))
              for name, value in [('control_points', 1e100), ('line_points', 1.00000001)]]
    db.put_geometry(fiber, dict(kind='fiber', arrays=arrays))
    assert [r['array_name'] for r in regions(db, fiber, [1e100]*6)] == ['control_points']
    assert regions(db, fiber, [1.00000002]*6) == []
    assert regions(db, fiber, [1]*6) == []
    assert len(regions(db, fiber, [1.00000001]*6)) == 1
    uid = db.upsert_object(dict(kind='point_collection'))
    db.put_geometry(uid, dict(kind='point_collection', arrays=[
        dict(name='points', dtype='<f8', shape=[514, 3], data=struct.pack('<1542d', *range(1542))),
        dict(name='point_ids', dtype='<i8', shape=[514], data=struct.pack('<514q', *range(514)))]))
    rs = regions(db, uid)
    assert [(r['start'], r['shape']) for r in rs] == [([0], [256]), ([256], [256]), ([512], [2])]
    assert len(regions(db, uid, [768, 769, 770]*2)) == 1


def test_region_filters_paging_external_boxes_and_read_only(db):
    ids = [db.upsert_object(dict(kind=kind, tags=['visible'], **{kind: dict(bbox=[0]*3+[10]*3)}))
           for kind in ('fiber', 'patch', 'track')]
    edge = db.add_edge(dict(kind='chain', a=ids[0], b=ids[2]))
    result = db.spatial_regions(ALL)
    assert [r['object_uuid'] for r in result] == sorted(ids)
    assert all(r['revision_uuid'] is None and r['start'] == r['shape'] == [] and r['array_name'] == '' for r in result)
    assert len(db.spatial_regions(ALL, object_kind='fiber', tag='visible')) == 1
    assert db.spatial_regions(ALL, tag='hidden') == []
    assert [r['object_uuid'] for r in db.spatial_regions(ALL, connected_to=ids[0], edge_kinds=['chain'])] == sorted([ids[0], ids[2]])
    assert db.spatial_regions(ALL, limit=1, offset=1) == result[1:2]
    assert db.spatial_regions(ALL, limit=0) == []
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.spatial_regions(ALL) == result
    for kwargs in [dict(limit=-1), dict(offset=-1)]:
        with pytest.raises(ValueError, match='nonnegative'):
            db.spatial_regions(ALL, **kwargs)
    with pytest.raises(ValueError, match='finite, ordered'):
        db.spatial_regions([math.nan]*6)
    db.remove_edge(edge)
    db.delete_object(ids[0])
    assert len(db.spatial_regions(ALL)) == 2
    with sqlite3.connect(db.path) as sql:
        assert sql.execute('SELECT count(*) FROM spatial_regions_rtree').fetchone()[0] == 2


def test_region_edit_rollback_sync_and_unchanged_tiles(db, tmp_path):
    uid = db.upsert_object(dict(kind='patch'))
    pts = [(x, y, 1) for y in range(65) for x in range(65)]
    pts[50*65+50] = (50, 50, 1000)  # Keep the whole-object box unchanged.
    first = db.put_geometry(uid, patch(65, 65, pts))
    before = regions(db, uid)
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        assert regions(peer, uid) == before
        pts[17*65+17] = (17, 17, 500)
        with pytest.raises(RuntimeError, match='rollback'):
            with db.write() as tx:
                tx.put_geometry(uid, patch(65, 65, pts), expected_revision=first)
                raise RuntimeError('rollback')
        assert regions(db, uid) == before
        # Count R-tree maintenance to catch rewriting every unchanged tile on
        # a local edit; this is a critical write-amplification regression guard.
        with sqlite3.connect(db.path) as sql:
            sql.execute('CREATE TABLE region_updates(id INTEGER)')
            sql.execute('CREATE TRIGGER count_regions AFTER UPDATE ON spatial_regions '
                        'BEGIN INSERT INTO region_updates VALUES(new.id); END')
        second = db.put_geometry(uid, patch(65, 65, pts), expected_revision=first)
        with sqlite3.connect(db.path) as sql:
            assert sql.execute('SELECT count(*) FROM region_updates').fetchone()[0] == 1
        assert all(r['revision_uuid'] == second for r in regions(db, uid))
        db.synchronize(peer)
        assert regions(peer, uid) == regions(db, uid)
        db.put_geometry(uid, patch(0, 0), expected_revision=second)
        db.synchronize(peer)
        assert regions(peer, uid) == regions(db, uid) == []
        with sqlite3.connect(db.path) as sql:
            assert sql.execute('SELECT count(*) FROM spatial_regions_rtree').fetchone()[0] == 0


def downgrade_to_v4(path):
    """Recreate the previous shared whole-object index, preserving payloads."""
    with sqlite3.connect(path) as sql:
        for trigger in ('insert', 'update', 'delete'):
            sql.execute(f'DROP TRIGGER object_bounds_{trigger}')
        sql.execute('DROP TABLE spatial_regions')
        sql.execute('DROP TABLE spatial_regions_rtree')
        sql.execute('CREATE VIRTUAL TABLE object_bounds_rtree USING rtree(object_id,min_x,max_x,min_y,max_y,min_z,max_z)')
        sql.execute('INSERT INTO object_bounds_rtree SELECT object_id,min_x,max_x,min_y,max_y,min_z,max_z FROM object_bounds')
        # Migration only drops these triggers; fixture data is already populated.
        for trigger in ('insert', 'update', 'delete'):
            sql.execute(f'CREATE TRIGGER object_bounds_{trigger} AFTER {trigger} ON object_bounds BEGIN SELECT 1; END')
        sql.execute('PRAGMA user_version=4')


def test_v4_migration_preserves_read_only_object_queries_and_builds_regions(tmp_path):
    path = str(tmp_path / 'v4.vcobj')
    with ObjectStore.create(path, author='test') as db:
        uid = db.upsert_object(dict(kind='patch'))
        db.put_geometry(uid, patch(65, 65))
        before = db.uuid, db.journal_since(), db.export_json()
        expected = regions(db, uid)
    downgrade_to_v4(path)
    with ObjectStore.open(path, read_only=True) as ro:
        assert ro.select_objects(bbox=[0, 0, 1, 64, 64, 1], fields=[]) == [dict(uuid=uid, kind='patch')]
        with pytest.raises(RuntimeError, match='schema 5'):
            ro.spatial_regions(ALL)
    with ObjectStore.open(path, author='test') as db:
        assert (db.uuid, db.journal_since(), db.export_json()) == before
        assert regions(db, uid) == expected
    with sqlite3.connect(path) as sql:
        assert sql.execute('PRAGMA user_version').fetchone()[0] == 5
        assert sql.execute('PRAGMA foreign_key_check').fetchall() == []
        assert sql.execute("SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%rtree%'").fetchall() == [('spatial_regions_rtree',)]


def test_failed_v4_migration_rolls_back_old_index(tmp_path):
    path = str(tmp_path / 'corrupt-v4.vcobj')
    with ObjectStore.create(path, author='test') as db:
        uid = db.upsert_object(dict(kind='track'))
        db.put_geometry(uid, track([(2, 3, 4)]))
    downgrade_to_v4(path)
    with sqlite3.connect(path) as sql:
        sql.execute('DROP TRIGGER geometry_chunks_immutable_update')
        sql.execute('UPDATE geometry_chunks SET data=zeroblob(length(data))')
    with pytest.raises(RuntimeError, match='corrupt payload'):
        ObjectStore.open(path, author='test')
    with sqlite3.connect(path) as sql:
        assert sql.execute('PRAGMA user_version').fetchone()[0] == 4
        assert sql.execute('SELECT count(*) FROM object_bounds_rtree').fetchone()[0] == 1
        assert sql.execute("SELECT name FROM sqlite_master WHERE name LIKE 'spatial_regions%'").fetchall() == []
