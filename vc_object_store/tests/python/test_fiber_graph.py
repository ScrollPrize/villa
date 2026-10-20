"""Same-winding graphs exclude adjacent links before computing connectivity."""
import json
import struct
import uuid

import numpy as np
import pytest

from vc_object_store import ObjectStore
from object_helpers import edge_choice
from test_object_ingestion import fiber
from test_object_history import heads, linked as linked_tracks


def linked(db, metadata_keys=()):
    a, b, ga, _, edge = linked_tracks(db, metadata_keys, kind='fiber_branch', meta={'adjacent': False})
    return a, b, ga, edge


@pytest.mark.parametrize('adjacent', [False, True, None])
def test_fiber_filter_is_consistent_before_connectivity(db, adjacent):
    a, b, c, d, isolated = [db.upsert_object(dict(kind='fiber')) for _ in range(5)]
    ordinary = db.add_edge(dict(kind='fiber_branch', a=a, b=b, meta={'adjacent': False}))
    cross = db.add_edge(dict(kind='fiber_branch', a=b, b=c, meta={'adjacent': True}))
    legacy = db.add_edge(dict(kind='fiber_branch', a=c, b=d))
    self_edge = db.add_edge(dict(kind='fiber_branch', a=b, b=b,
        a_anchor={'points': [{'array': 'control_points', 'index': [0]}]},
        b_anchor={'points': [{'array': 'control_points', 'index': [1]}]}))
    other = db.add_edge(dict(kind='other', a=a, b=isolated, meta={'adjacent': False}))
    expected = {False: {ordinary, legacy, self_edge}, True: {cross}, None: {ordinary, cross, legacy, self_edge, other}}[adjacent]
    assert {e['uuid'] for e in db.select_edges(adjacent=adjacent)} == expected
    component = {False: {a, b}, True: {a}, None: {a, b, c, d, isolated}}[adjacent]
    depth_one = {False: {a, b}, True: {a}, None: {a, b, isolated}}[adjacent]
    for applicable in [False, True]:
        assert set(db.component(a, adjacent=adjacent, applicable_only=applicable)) == component
        assert set(db.reachable_within(a, 1, adjacent=adjacent, applicable_only=applicable)) == depth_one
        assert set(db.neighbours(a, adjacent=adjacent, applicable_only=applicable)) == depth_one - {a}
        assert {e['uuid'] for e in db.select_edges(adjacent=adjacent, applicable_only=applicable)} == expected
        with db.graph_read(adjacent=adjacent, applicable_only=applicable, approval_states=[]) as graph:
            assert set(graph.edge_uuids) == expected
            assert set(graph.component(a)) == component
            assert set(graph.reachable_within(a, 1)) == depth_one
    with db.graph_read(revisions={u: None for u in [a, b, c, d, isolated]}, adjacent=adjacent, approval_states=[]) as graph:
        assert set(graph.edge_uuids) == expected
        assert set(graph.component(a)) == component
        assert set(graph.reachable_within(a, 1)) == depth_one
    # No alternative route through an excluded object, even if the full graph connects it.
    with db.graph_read(object_uuids=[a, c], approval_states=[], adjacent=False) as graph:
        assert graph.component(a) == [a]


def test_same_winding_never_uses_adjacent_parallel_edge(db):
    a, b, c = [db.upsert_object(dict(kind='fiber')) for _ in range(3)]
    db.add_edge(dict(kind='fiber_branch', a=a, b=b, meta={'adjacent': True}))
    ordinary = db.add_edge(dict(kind='fiber_branch', a=a, b=b, meta={'adjacent': False},
                                a_anchor={'points': [{'array': 'control_points', 'index': [0]}]}))
    db.add_edge(dict(kind='fiber_branch', a=b, b=c, meta={'adjacent': True}))
    assert db.neighbours(a, adjacent=False) == [b]
    assert set(db.component(a, adjacent=False)) == {a, b}
    db.remove_edge(ordinary)
    assert db.component(a, adjacent=False, applicable_only=True) == [a]
    assert db.neighbours(a, adjacent=False, applicable_only=True) == []


def test_adjacency_changes_do_not_rewrite_selected_graph_semantics(db):
    a, b, _, edge = linked(db)
    original = heads(db, a, b)
    db.set_meta(edge, 'adjacent', True)
    assert db.component(a, adjacent=False, applicable_only=True) == [a]
    with db.graph_read(revisions={a: None, b: None}, adjacent=False, approval_states=[]) as graph:
        assert graph.component(a) == [a]
    with db.graph_read(revisions=original, adjacent=False, approval_states=[]) as graph:
        assert set(graph.component(a)) == {a, b}
        assert graph.edges()[0]['meta']['adjacent'] is False


@pytest.mark.parametrize('malformed', [None, 1, 'true'])
def test_malformed_adjacency_is_never_same_winding(db, malformed):
    a, b = [db.upsert_object(dict(kind='fiber')) for _ in range(2)]
    db.add_edge(dict(kind='fiber_branch', a=a, b=b, meta={'adjacent': malformed}))
    assert db.component(a, adjacent=False) == [a]
    assert db.select_edges(adjacent=False) == []
    with db.graph_read(revisions={a: None, b: None}, adjacent=False, approval_states=[]) as graph:
        assert graph.component(a) == [a]


def test_preserved_link_is_historical_not_current_and_views_pin_records(db, db_path):
    a, b, ga, e = linked(db)
    original = heads(db, a, b)
    old_edge = db.edge(e)
    old_objects = {u: db.object(u) for u in [a, b]}
    with db.write(expected_revisions=original) as tx:
        tx.edit_geometry(a, [dict(array='points', element_offset=0, data=struct.pack('<i', 9))],
                         expected_revision=ga, edge_edits=[edge_choice(db, e, 'keep_recorded')])
    assert b in db.component(a)
    assert db.component(a, adjacent=False, applicable_only=True) == [a]
    assert db.reachable_within(a, 2, applicable_only=True) == [a]
    assert db.neighbours(a, applicable_only=True) == []
    assert db.select_edges(applicable_only=True) == []
    with db.graph_read(approval_states=[], applicable_only=True) as graph:
        assert graph.component(a) == [a]
    with db.graph_read(revisions=original, adjacent=False, approval_states=[]) as graph:
        assert set(graph.component(a)) == {a, b}
        assert graph.edges() == [old_edge]
        assert {r['uuid']: r for r in graph.objects()} == old_objects
        assert dict(zip(graph.object_uuids, graph.object_revisions)) == original
        node = graph.object_uuids.index(a)
        assert graph.geometry_revisions[node] == ga
        assert graph.read_geometry_ranges(node, 'points', [[0, 12]]) == [struct.pack('<3i', 1, 2, 3)]
        arrays = graph.partners
        db.remove_edge(e)
        db.set_meta(a, 'winding', 99)
        db.set_meta(b, 'winding', 99)
        assert graph.edges(fields=['meta'])[0]['meta'] == {'adjacent': False}
        assert graph.objects(fields=['meta']) == [dict(uuid=u, kind='track', meta=old_objects[u]['meta']) for u in graph.object_uuids]
    assert len(arrays) == 2 and not arrays.flags.writeable
    with pytest.raises(RuntimeError, match='closed'):
        graph.component(a)
    with ObjectStore.open(db_path, read_only=True) as reopened:
        with reopened.graph_read(revisions=original, adjacent=False, approval_states=[]) as historical:
            assert historical.edges() == [old_edge]
            assert {r['uuid']: r for r in historical.objects()} == old_objects
            values = historical.read_geometry_arrays('points').values
            expected = [[1, 2, 3], [4, 5, 6], [1, 2, 3]] if historical.object_uuids[0] == a else [[1, 2, 3], [1, 2, 3], [4, 5, 6]]
            np.testing.assert_array_equal(values, expected)


def test_historical_graph_after_physical_deletion(db):
    a, b = [db.upsert_object(dict(kind='fiber', name=name)) for name in ['a', 'b']]
    edge = db.add_edge(dict(kind='fiber_branch', a=a, b=b))
    original = heads(db, a, b)
    db.remove_edge(edge)
    db.delete_object(a)
    db.delete_object(b)
    with db.graph_read(revisions=original, adjacent=False, approval_states=[]) as graph:
        assert graph.edge_uuids == [edge]
        assert set(graph.component(a)) == {a, b}
        assert {o['name'] for o in graph.objects()} == {'a', 'b'}


def test_dependency_checks_revalidation_and_membership(db):
    a, b, _, e = linked(db, metadata_keys=['winding'])
    original = heads(db, a, b)
    db.set_meta(a, 'unrelated', True)
    assert len(db.select_edges(applicable_only=True)) == 1
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.set_meta(a, 'winding', 2)
        tx.keep_recorded_relationship(e)
    assert db.select_edges(applicable_only=True) == []
    with db.graph_read(revisions={a: None, b: None}, approval_states=[]) as graph:
        assert graph.edge_uuids == []
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(e)
    assert len(db.select_edges(applicable_only=True)) == 1
    with db.graph_read(revisions={a: original[a], b: None}, approval_states=[]) as graph:
        assert graph.edge_uuids == []
    with db.graph_read(revisions={a: None}, approval_states=[]) as graph:
        assert graph.object_uuids == [a] and graph.edge_uuids == []
    with db.graph_read(revisions=original, approval_states=[]) as graph:
        assert graph.edge_uuids == [e]


def test_lifecycle_and_pagination_after_applicability(db):
    a, b, ga, stale = linked(db)
    db.set_meta(stale, 'order', 'stale')
    original = heads(db, a, b)
    with db.write(expected_revisions=original) as tx:
        tx.retire_object(b)
        tx.keep_recorded_relationship(stale)
    with db.graph_read(revisions={a: None, b: None}, approval_states=[]) as graph:
        assert set(graph.object_uuids) == {a, b} and graph.edge_uuids == []
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.restore_object(b)
        tx.keep_recorded_relationship(stale)
    assert db.select_edges(applicable_only=True, include_retired=True) == []
    with db.graph_read(revisions=original, approval_states=[]) as graph:
        assert graph.edge_uuids == [stale]
    # A valid parallel edge must survive an earlier inapplicable candidate.
    anchor_a = dict(points=[dict(array='points', index=[1])])
    anchor_b = dict(points=[dict(array='points', index=[0])])
    valid = db.add_edge(dict(uuid=str(uuid.UUID(int=(1 << 128) - 1)), kind='fiber_branch',
                             a=a, b=b, a_anchor=anchor_a, b_anchor=anchor_b, tags=['wanted']))
    query = dict(adjacent=False, applicable_only=True, include_retired=True)
    rows = db.select_edges(**query, limit=1, fields=['tags'])
    assert [r['uuid'] for r in rows] == [valid] and rows[0]['tags'] == ['wanted']
    assert db.select_edges(**query, after_uuid=valid, limit=1) == []
    assert db.select_edges(**query, limit=0) == []
    assert db.neighbours(a, applicable_only=True) == [b]
    assert set(db.component(a, applicable_only=True)) == {a, b}
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(stale)
    assert len(db.select_edges(**query)) == 2


def test_empty_and_invalid_selected_graphs(db):
    a, b, _, _ = linked(db)
    with db.graph_read(revisions={}, approval_states=[]) as graph:
        assert graph.object_uuids == [] and graph.edge_uuids == []
        np.testing.assert_array_equal(graph.offsets, [0])
        assert graph.objects() == [] and graph.edges() == []
        with pytest.raises(ValueError, match='not in graph'):
            graph.component(a)
    with pytest.raises(ValueError, match='does not belong'):
        db.graph_read(revisions={a: heads(db, b)[b]})
    with pytest.raises(ValueError, match='cannot combine'):
        db.graph_read(revisions={}, object_kind='fiber')
    for query in [dict(limit=-1), dict(fields=['unknown'])]:
        with pytest.raises(ValueError):
            db.select_edges(applicable_only=True, **query)
    with db.graph_read(revisions={a: None}, approval_states=[]) as graph:
        with pytest.raises(ValueError, match='nonnegative|>= 0'):
            graph.reachable_within(a, -1)
        for read in [graph.objects, graph.edges]:
            with pytest.raises(ValueError, match='unknown'):
                read(fields=['unknown'])


@pytest.mark.parametrize('first_adjacent,second_adjacent', [(False, True), (True, False), (False, False), (True, True)])
def test_import_classification_conflicts_are_atomic(db, tmp_path, first_adjacent, second_adjacent):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    source = tmp_path / 'fibers'
    source.mkdir()
    for name, uid, target, adjacent in [('a', a, b, first_adjacent), ('b', b, a, second_adjacent)]:
        data = fiber(uid)
        data['adjacent_branches' if adjacent else 'branches'] = [dict(
            control_point_index=0, branch_control_point_index=0, target_uuid=target)]
        (source / f'{name}.json').write_text(json.dumps(data))
    report = db.ingest_geometry_source('fiber', source)
    if first_adjacent != second_adjacent:
        assert report['ingested'] == 0 and 'conflicting ordinary/adjacent' in report['problems'][0]
        assert db.object(a) is None and db.object(b) is None
        assert db.object_history(a) == [] and db.object_history(b) == [] and db.journal_since() == []
    else:
        assert report == dict(ingested=2, unchanged=0, edges_imported=1, problems=[])
        assert db.edges_of(a)[0]['meta']['adjacent'] == first_adjacent


def test_import_conflicting_lists_in_one_fiber_roll_back(db, tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    first = fiber(a, [dict(control_point_index=0, branch_control_point_index=1, target_uuid=b)])
    first['adjacent_branches'] = first['branches']
    source = tmp_path / 'fibers'
    source.mkdir()
    (source / 'a.json').write_text(json.dumps(first))
    (source / 'b.json').write_text(json.dumps(fiber(b)))
    report = db.ingest_geometry_source('fiber', source)
    assert 'conflicting ordinary/adjacent' in report['problems'][0]
    assert report['ingested'] == 0 and db.journal_since() == []
