"""Consumer query semantics: bounded pages, consistent graph views and edits."""
import gc
import json
import random
import struct
import uuid

import numpy as np
import pytest


def uid(i):
    return str(uuid.UUID(int=i + 1))


def test_edge_projection_filters_and_deleted_cursor(db):
    nodes = [db.upsert_object(dict(uuid=uid(i), kind='note')) for i in range(4)]
    edges = []
    for i, (a, b, state) in enumerate([(0, 1, 'approved'), (1, 2, 'pending'),
                                      (2, 3, 'approved'), (0, 3, 'rejected')]):
        e = db.add_edge(dict(uuid=uid(100-i), kind='contact', a=nodes[a], b=nodes[b],
                            tags=['review'], meta={'large': 'x' * 20000},
                            a_anchor={'position': [i, 0, 0]}))
        db.set_approval(e, state)
        edges.append(e)
    expected = sorted((edges[0], edges[2]))
    query = dict(edge_kind='contact', approval_states=['approved'], tag='review', fields=['a_anchor'])
    rows = db.select_edges(**query)
    assert [r['uuid'] for r in rows] == expected
    assert all(set(r) == {'uuid', 'kind', 'a', 'b', 'a_anchor'} for r in rows)
    assert rows[0]['a_anchor'] == db.edge(rows[0]['uuid'])['a_anchor']
    first = db.select_edges(**query, limit=1)[0]
    db.remove_edge(first['uuid'])
    assert [r['uuid'] for r in db.select_edges(**query, after_uuid=first['uuid'])] == expected[1:]
    assert db.select_edges(endpoint_uuids=[], fields=[]) == []
    assert db.select_edges(limit=0) == []
    assert {r['uuid'] for r in db.select_edges(endpoint_uuids=[nodes[0]], fields=[])} == {
        e for e in (edges[0], edges[3]) if db.edge(e) is not None}
    assert db.select_edges(endpoint_uuids=[nodes[0]], both_endpoints=True) == []
    for kwargs in (dict(limit=-1), dict(approval_states=['invalid']), dict(fields=['unknown'])):
        with pytest.raises(ValueError):
            db.select_edges(**kwargs)


@pytest.mark.parametrize('kind', ['track', 'fiber'])
def test_catalog_cursor_matches_full_sort_with_ties_and_filters(db, kind):
    rng = random.Random(7)
    for i in range(97):
        detail = {'family': 'H' if i % 2 else 'V', 'length': rng.randrange(10)}
        db.upsert_object(dict(uuid=uid(97-i), kind=kind, **{kind: detail}))
    expected = sorted((r for r in db.objects_by_kind(kind)
                       if r[kind]['family'] == 'H' and r[kind]['length'] >= 3),
                      key=lambda r: (-r[kind]['length'], r['uuid']))
    query = dict(object_kind=kind, family='H', min_length=3, order_by='length_desc')
    assert db.select_objects(**query) == expected
    pages, cursor = [], {}
    while page := db.select_objects(**query, limit=7, **cursor):
        pages.extend(page)
        cursor = dict(after_uuid=page[-1]['uuid'], after_length=page[-1][kind]['length'])
    assert pages == expected
    first = expected[0]
    db.delete_object(first['uuid'])
    assert db.select_objects(**query, after_uuid=first['uuid'], after_length=first[kind]['length']) == expected[1:]
    for kwargs in (dict(family='H'), dict(order_by='bad'), dict(after_length=3),
                   dict(object_kind=kind, min_length=float('nan')),
                   dict(object_kind=kind, order_by='length_desc', after_uuid=uid(0))):
        with pytest.raises(ValueError):
            db.select_objects(**kwargs)


@pytest.mark.parametrize('edge', [False, True])
def test_unchanged_metadata_preserves_approval_and_journal(db, edge):
    a = db.upsert_object(dict(kind='note'))
    b = db.upsert_object(dict(kind='note'))
    record = db.add_edge(dict(kind='link', a=a, b=b)) if edge else a
    read = db.edge if edge else db.object
    db.set_meta(record, 'nested.key', {'a': 1, 'b': [None, False]})
    db.set_approval(record, 'approved')
    before = read(record), db.journal_since()
    db.set_meta(record, 'nested.key', {'b': [None, False], 'a': 1})
    db.remove_meta(record, 'missing')
    assert (read(record), db.journal_since()) == before
    db.set_meta(record, 'null', None)  # missing != null
    assert read(record)['generation'] == before[0]['generation'] + 1
    assert read(record)['approval_state'] == 'pending'
    db.set_meta(record, 'null', None)
    assert read(record)['generation'] == before[0]['generation'] + 1
    db.remove_meta(record, 'null')
    assert read(record)['generation'] == before[0]['generation'] + 2


def geometry(value):
    return dict(kind='fiber', arrays=[dict(name='line_points', dtype='<f8', shape=[2, 3],
        data=struct.pack('<6d', value, 2, 3, 4, 5, 6)), dict(name='control_points', dtype='<f8', shape=[0, 3], data=b'')])


def test_graph_view_pins_topology_metadata_and_geometry(db):
    a, b, c = [db.upsert_object(dict(uuid=uid(i), kind='fiber')) for i in range(3)]
    rev = db.put_geometry(a, geometry(1))
    approved = db.add_edge(dict(kind='link', a=b, b=c, meta={'index': 7}))
    db.set_approval(approved, 'approved')
    db.add_edge(dict(kind='link', a=a, b=b, a_anchor={'points': [{'array': 'line_points', 'index': [1]}]}))
    with db.graph_read(edge_kind='link') as view:
        assert view.object_uuids == [a, b, c]
        assert view.edge_uuids == [approved]
        np.testing.assert_array_equal(view.offsets, [0, 0, 1, 2])
        np.testing.assert_array_equal(view.partners, [2, 1])
        np.testing.assert_array_equal(view.edge_indices, [0, 0])
        arrays = view.partners
        assert not arrays.flags.writeable
        assert view.geometry_revisions == [rev, None, None]
        before = view.journal_sequence
        db.set_meta(approved, 'index', 99)
        assert db.put_geometry(a, geometry(9), expected_revision=rev) != rev
        db.set_meta(a, 'later', 1)
        assert view.edges(fields=['meta'])[0]['meta'] == {'index': 7}
        assert 'later' not in view.objects(fields=['meta'])[0]['meta']
        assert view.read_geometry_ranges(0, 'line_points', [[0, 24]]) == [struct.pack('<3d', 1, 2, 3)]
        assert view.journal_sequence == before
        with pytest.raises(RuntimeError, match='active transaction'):
            db.close()
    with pytest.raises(RuntimeError, match='closed'):
        view.neighbours(0)
    gc.collect()
    np.testing.assert_array_equal(arrays, [2, 1])  # array owner outlives the view
    with db.graph_read(edge_kind='link') as fresh:
        assert fresh.edge_uuids == []  # metadata mutation reset approval
        assert fresh.endpoints.shape == (0, 2)
        assert fresh.partners.shape == (0,)
        assert fresh.journal_sequence > before
    with db.graph_read(object_kind='absent') as empty:
        assert empty.objects() == [] and empty.edges() == []
        np.testing.assert_array_equal(empty.offsets, [0])


def test_range_planning_overlaps_duplicates_and_more_than_eight_chunks(db):
    payload = bytes(range(256)) * (10 * 1024 * 1024 // 256 + 3)
    payload = payload[:len(payload) // 24 * 24]
    # Valid finite doubles rather than arbitrary NaN payloads.
    payload = struct.pack('<3d', 1, 2, 3) * (len(payload) // 24)
    a = db.upsert_object(dict(kind='fiber'))
    rev = db.put_geometry(a, dict(kind='fiber', arrays=[
        dict(name='line_points', dtype='<f8', shape=[len(payload)//24, 3], data=payload),
        dict(name='control_points', dtype='<f8', shape=[0, 3], data=b'')]))
    ranges = [(i % 10 * 1048576 + 3, 29) for i in range(40)]
    ranges += [(1048570, 100), (0, 24), (0, 24), (len(payload), 0)]
    assert db.read_geometry_ranges(rev, 'line_points', ranges) == [payload[o:o+n] for o, n in ranges]
    with pytest.raises(IndexError):
        db.read_geometry_ranges(rev, 'line_points', ranges + [(len(payload), 1)])


def test_graph_parallel_self_links_and_projected_export(db):
    # Insertion order deliberately differs from canonical UUID order.
    a = db.upsert_object(dict(uuid=uid(5), kind='note'))
    b = db.upsert_object(dict(uuid=uid(0), kind='note'))
    def anchor(i):
        return {'points': [{'array': 'points', 'index': [i]}]}
    for i in (0, 1):
        db.add_edge(dict(kind='link', a=a, b=b, a_anchor=anchor(i), b_anchor=anchor(i+2)))
    db.add_edge(dict(kind='link', a=a, b=a, a_anchor=anchor(1), b_anchor=anchor(0)))
    full = json.loads(db.export_json())
    for fields in (['name'], ['a_anchor'], ['b_anchor']):
        projected = json.loads(db.export_json(fields=fields))
        for actual, expected in zip(projected['edges'], full['edges']):
            assert actual == {key: value for key, value in expected.items()
                              if key in fields or key in ('uuid', 'kind', 'a', 'b', 'geometry_bindings')}
    with db.graph_read(approval_states=[]) as view:
        assert len(view.edge_uuids) == 3
        np.testing.assert_array_equal(view.offsets, [0, 2, 6])
        assert all(np.count_nonzero(view.edge_indices == i) == 2 for i in range(3))
        assert sorted(view.neighbours(1)) == [0, 0, 1, 1]


def test_bulk_metadata_atomicity_and_null(db):
    a = db.upsert_object(dict(kind='note'))
    before = db.object(a), db.journal_since()
    with pytest.raises(RuntimeError, match='not found'):
        db.set_meta_many([(a, 'key', 1), (uid(999), 'key', 2)])
    assert (db.object(a), db.journal_since()) == before
    db.set_meta_many([(a, 'key', None), (a, 'other', {'x': 1})])
    assert db.object(a)['meta'] == {'key': None, 'other': {'x': 1}}
    with db.write() as tx:
        tx.set_meta(a, 'other', None)
    assert db.object(a)['meta']['other'] is None


def test_v6_query_index_migration_preserves_records(tmp_path):
    from vc.objects import ObjectStore
    from object_helpers import sqlite_rows
    path = str(tmp_path / 'v6.vcobj')
    with ObjectStore.create(path, author='test') as store:
        store.upsert_object(dict(kind='track', track={'family': 'H', 'length': 8}))
        before = store.uuid, store.export_json(), store.journal_since()
    for name in ('edges_kind_approval_uuid', 'edges_kind_uuid', 'edges_kind_id', 'fibers_family_length',
                 'fibers_length', 'tracks_family_length', 'tracks_length'):
        sqlite_rows(path, f'DROP INDEX {name}')
    sqlite_rows(path, 'PRAGMA user_version=6')
    with ObjectStore.open(path, read_only=True) as store:
        assert len(store.select_objects(object_kind='track', family='H', order_by='length_desc')) == 1
    with ObjectStore.open(path, author='test') as store:
        assert (store.uuid, store.export_json(), store.journal_since()) == before
    assert sqlite_rows(path, 'PRAGMA user_version') == [[7]]
    assert sqlite_rows(path, "SELECT name FROM sqlite_master WHERE name='edges_kind_approval_uuid'")


def test_spatial_typed_filters_run_before_region_limit(db):
    for i in range(8):
        db.upsert_object(dict(uuid=uid(i), kind='track', track={
            'family': 'H' if i % 2 else 'V', 'length': i,
            'bbox': [0, 0, 0, 10, 10, 10]}))
    kwargs = dict(bbox=[1, 1, 1, 2, 2, 2], object_kind='track', limit=1)
    # Prime the resident unfiltered spatial index, then ask for typed filters.
    assert db.spatial_regions(**kwargs)[0]['object_uuid'] == uid(0)
    assert db.spatial_regions(**kwargs, family='H', min_length=4)[0]['object_uuid'] == uid(5)
    assert db.select_objects(**kwargs, family='H', min_length=4)[0]['uuid'] == uid(5)


def test_graph_view_rejects_cross_thread_access(db):
    from concurrent.futures import ThreadPoolExecutor
    db.upsert_object(dict(kind='note'))
    with db.graph_read() as view, ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match='thread-confined'):
            pool.submit(lambda: view.objects()).result()
