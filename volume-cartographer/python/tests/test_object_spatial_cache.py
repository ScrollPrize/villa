"""Resident spatial index agrees with SQL across queries and snapshot changes."""
import concurrent.futures
import random
import sqlite3

import pytest
from vc.objects import ObjectStore
from object_helpers import track
from object_helpers import ALL, patch


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
        # No journal change: invalidation must observe any SQLite commit.
        with sqlite3.connect(db.path) as sql:
            sql.execute("UPDATE object_bounds SET min_x=10,max_x=10")
        assert cached.spatial_regions([0]*6) == []
        assert cached.spatial_regions([10, 0, 0]*2)[0]["object_uuid"] == uid
        db.delete_object(uid)
        assert cached.spatial_regions(ALL) == []
        # An empty resident index must be invalidated by later insertion too.
        new = db.upsert_object(dict(kind="patch", patch=dict(bbox=[0]*6)))
        assert cached.spatial_regions([0]*6)[0]["object_uuid"] == new
