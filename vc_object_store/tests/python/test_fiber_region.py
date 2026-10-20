"""Oriented fiber browsing: original segments, budgets and revision snapshots."""
import math
import struct
import uuid

import pytest
from vc_object_store import Fiber, FiberSlice, FiberRegionOptions, FiberRegionCancellation, ObjectStore


def options(**kwargs):
    result = FiberRegionOptions()
    for k, v in kwargs.items():
        setattr(result, k, v)
    return result


def fiber(db, n, points, family='H', length=10, controls=False):
    uid = db.upsert_object(dict(uuid=str(uuid.UUID(int=n)), kind='fiber',
                                fiber=dict(family=family, length=length)))
    revision = db.put_geometry(uid, Fiber(points if controls else [], [] if controls else points))
    return uid, revision


def segments(result):
    return [(r['object_uuid'], r['first_segment']+i, r['points'][i], r['points'][i+1])
            for r in result['runs'] for i in range(len(r['points'])-1)]


def test_crossing_original_runs_and_misses(db):
    points = [[-3, 0, 0], [0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 0, 0]]
    uid, revision = fiber(db, 1, points)
    fiber(db, 2, [[0.9, 3, 0], [3, 0.9, 0]])  # AABB overlaps, segment misses
    result = db.fiber_region(FiberSlice())
    assert result['complete'] and result['stop'] == 'exhausted'
    assert [(r['first_segment'], r['points']) for r in result['runs']] == [(0, points[:3]), (3, points[3:])]
    assert all(r['revision_uuid'] == revision for r in result['runs'])
    assert result['intersecting_fibers'] == 1
    assert result['stats']['output_bytes'] == 5*24
    assert result['stats']['segment_tests'] == 5
    assert result['stats']['processing_bytes'] > result['stats']['output_bytes']


def test_oblique_slice_and_both_endpoints_outside(db):
    points = [[-4, -4, 0], [4, 4, 0]]
    uid, _ = fiber(db, 1, points)
    fiber(db, 2, [[-1, 1, 0], [-0.8, 1.2, 0]])
    view = FiberSlice()
    c = math.sqrt(0.5)
    view.u = [c, c, 0]
    view.v = [-c, c, 0]
    view.local_bounds = [-1, -0.1, -0.1, 1, 0.1, 0.1]
    result = db.fiber_region(view)
    assert [(r['object_uuid'], r['points']) for r in result['runs']] == [(uid, points)]
    view.local_bounds = [-1, -0.1, 0, 1, 0.1, 0]  # zero thickness is supported
    assert segments(db.fiber_region(view)) == segments(result)


def test_stable_distinct_fiber_limits_priority_and_filters(db):
    miss, _ = fiber(db, 1, [[0.9, 3, 0], [3, 0.9, 0]])
    a, _ = fiber(db, 2, [[-2, 0, 0], [2, 0, 0]]*300)
    b, _ = fiber(db, 3, [[-2, 0, 0], [2, 0, 0]], family='V', length=20)
    for _ in range(2):
        result = db.fiber_region(FiberSlice(), options(maximum_fibers=1, selected=miss))
        assert result['stop'] == 'fiber_limit' and not result['complete']
        assert result['intersecting_fibers'] == 1
        assert {r['object_uuid'] for r in result['runs']} == {a}
        assert len(segments(result)) == 599
    result = db.fiber_region(FiberSlice(), options(maximum_fibers=1, selected=b.upper()))
    assert [r['object_uuid'] for r in result['runs']] == [b]
    result = db.fiber_region(FiberSlice(), options(selected=b))  # unlimited ignores selected
    assert [r['object_uuid'] for r in result['runs']] == [a, b]
    for o in (options(family='V'), options(min_length=15)):
        assert [r['object_uuid'] for r in db.fiber_region(FiberSlice(), o)['runs']] == [b]
    assert db.fiber_region(FiberSlice(), options(family='none'))['complete']
    assert not db.fiber_region(FiberSlice(), options(min_points=601))['runs']


@pytest.mark.parametrize('field,budget,reason', [
    ('output_byte_budget', 47, 'output_budget'),
    ('processing_byte_budget', 0, 'processing_budget'),
    ('candidate_work_budget', 0, 'candidate_budget'),
    ('sql_step_budget', 0, 'sql_budget'),
    ('max_unit_bytes', 10, 'processing_budget'),
])
def test_strict_budgets_and_selected_insufficient(db, field, budget, reason):
    a, _ = fiber(db, 1, [[-2, 0, 0], [2, 0, 0]])
    b, _ = fiber(db, 2, [[-3, 0, 0], [3, 0, 0]])
    result = db.fiber_region(FiberSlice(), options(maximum_fibers=1, selected=b, **{field: budget}))
    assert result['stop'] == reason and not result['complete']
    assert not result['runs']
    if field == 'output_byte_budget':
        assert result['stats']['payload_bytes'] > 0  # attempted selected, no substitution
    assert db.fiber_region(FiberSlice())['complete']  # cancelled statement/handler cleaned up


def test_partial_output_extends_runs_and_does_not_cache_coverage(db):
    uid, _ = fiber(db, 1, [[-2, 0, 0], [0, 0, 0], [2, 0, 0]])
    r = db.fiber_region(FiberSlice(), options(output_byte_budget=48))
    assert r['stop'] == 'output_budget' and len(segments(r)) == 1
    assert r['stats']['output_bytes'] == 48
    assert len(segments(db.fiber_region(FiberSlice()))) == 2
    token = FiberRegionCancellation()
    token.cancel()
    r = db.fiber_region(FiberSlice(), cancel=token)
    assert r['stop'] == 'cancelled' and not r['complete'] and not r['runs']


def test_legacy_chunk_boundary_and_warm_accounting(db):
    # A very small region crosses a physical 1 MiB chunk boundary inside a point.
    n = 44000
    points = [[float(i), 0, 0] for i in range(n)]
    uid, _ = fiber(db, 1, points)
    view = FiberSlice()
    view.origin = [43690.5, 0, 0]
    view.local_bounds = [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]
    for _ in range(2):
        result = db.fiber_region(view)
        assert len(segments(result)) == 1
        assert result['runs'][0]['first_segment'] == 43690
        assert result['runs'][0]['points'] == points[43690:43692]
        assert result['stats']['payload_bytes'] == n*24
        assert result['stats']['output_bytes'] == 48
        limited = db.fiber_region(view, options(processing_byte_budget=1024*1024))
        assert limited['stop'] == 'processing_budget'
        assert limited['stats']['processing_bytes'] <= 1024*1024


def test_sparse_and_structural_revisions_seams_reopen(db):
    points = [[float(i), 0, 0] for i in range(6000)]
    uid, old = fiber(db, 1, points, controls=True)
    current = db.edit_geometry(uid, [dict(array='control_points', element_offset=3,
        data=struct.pack('<3d', 1, 0.25, 0))], expected_revision=old)
    points[1] = [1., 0.25, 0.]
    view = FiberSlice()
    view.local_bounds = [-1, -1, -1, 7000, 1, 1]
    result = db.fiber_region(view)
    assert result['complete'] and result['runs'][0]['points'] == points
    # Establish the persistent sequence tree, then insert an off-line point at a seam.
    with db.write(expected_revisions={uid: db.current_object_revision(uid)['uuid']}) as tx:
        current = tx.edit_geometry_sequence(uid, dict(array='control_points', start=2730, erase_count=0,
            parts=[dict(count=1, ids=[6000], data=struct.pack('<3d', 2729.5, 0.5, 0))]), current)
    points.insert(2730, [2729.5, 0.5, 0.])
    for store in (db, ObjectStore.open(db.path, read_only=True)):
        try:
            result = store.fiber_region(view)
            assert result['complete']
            assert result['runs'][0]['points'] == points
            indices = [s[1] for s in segments(result)]
            assert indices == list(range(6000))
            assert result['stats']['tree_visits'] > 0
            assert result['stats']['cache_bytes'] > 0
            assert all(r['revision_uuid'] == current for r in result['runs'])
            # Only a structural seam crosses each narrow box; neither endpoint
            # is included, and the adjacent leaf bounds do not cover its middle.
            for x, index in ((2729.25, 2729), (2729.75, 2730)):
                narrow = FiberSlice()
                narrow.origin = [x, 0.25, 0]
                narrow.local_bounds = [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01]
                seam = store.fiber_region(narrow)
                assert seam['complete']
                assert [(s[1], s[2], s[3]) for s in segments(seam)] == [
                    (index, points[index], points[index+1])]
        finally:
            if store is not db:
                store.close()


def test_control_fallback_and_retirement(db):
    uid, _ = fiber(db, 1, [[-2, 0, 0], [2, 0, 0]], controls=True)
    r = db.fiber_region(FiberSlice())
    assert r['runs'][0]['array_name'] == 'control_points'
    with db.write(expected_revisions={uid: db.current_object_revision(uid)["uuid"]}) as tx:
        tx.retire_object(uid)
    assert db.fiber_region(FiberSlice())['runs'] == []


@pytest.mark.parametrize('attribute,value', [
    ('u', [2, 0, 0]), ('origin', [math.nan, 0, 0]),
    ('local_bounds', [1, 0, 0, -1, 0, 0]),
])
def test_invalid_slice(db, attribute, value):
    view = FiberSlice()
    setattr(view, attribute, value)
    with pytest.raises(ValueError):
        db.fiber_region(view)
