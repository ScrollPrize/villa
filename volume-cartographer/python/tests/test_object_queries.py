"""Ordering, filtering and integrity contracts for optimized storage queries."""
import gc
import json
import random
import struct
import uuid

import numpy as np
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


def test_graph_view_rejects_cross_thread_access(db):
    from concurrent.futures import ThreadPoolExecutor
    db.upsert_object(dict(kind='note'))
    with db.graph_read() as view, ThreadPoolExecutor(1) as pool:
        with pytest.raises(RuntimeError, match='thread-confined'):
            pool.submit(lambda: view.objects()).result()


def test_graph_compact_query_matches_edge_selector(db):
    nodes = [db.upsert_object(dict(kind='track', track={'bbox': [i, 0, 0, i+1, 1, 1]})) for i in range(7)]
    for i in range(7):
        edge = db.add_edge(dict(kind='crossing' if i % 2 else 'other', a=nodes[i], b=nodes[(i+1) % 7], tags=['chosen']))
        db.set_approval(edge, 'approved' if i % 3 else 'pending')
    for states in ([], ['approved'], ['pending', 'approved']):
        with db.graph_read(object_kind='track', bbox=[0, 0, 0, 4, 1, 1],
                           edge_kind='crossing', edge_tag='chosen', approval_states=states) as graph:
            expected = db.select_edges(endpoint_uuids=graph.object_uuids, both_endpoints=True,
                edge_kind='crossing', tag='chosen', approval_states=states, fields=[])
            assert graph.edge_uuids == [e['uuid'] for e in expected]
            for endpoints, edge in zip(graph.endpoints, expected):
                assert [graph.object_uuids[i] for i in endpoints] == [edge['a'], edge['b']]
