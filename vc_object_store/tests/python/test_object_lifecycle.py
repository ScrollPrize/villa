"""Revisioned lifecycle, visibility, declared references and atomic restoration."""
import json
import sqlite3
import struct
import uuid

import pytest
from vc_object_store import ObjectStore
from object_helpers import track, sqlite_rows, edge_choice
from test_object_history import heads, linked


def retire(db, uid, *endpoints, edges=()):
    with db.write(expected_revisions=heads(db, uid, *endpoints)) as tx:
        tx.retire_object(uid)
        for edge in edges:
            tx.keep_recorded_relationship(edge)


def restore(db, uid, *endpoints, edges=(), source=None):
    with db.write(expected_revisions=heads(db, uid, *endpoints)) as tx:
        tx.restore_object(uid, source_revision=source)
        for edge in edges:
            tx.keep_recorded_relationship(edge)


def payload_tables(db):
    return {name: sqlite_rows(db.path, 'SELECT ' + ','.join(
                f'quote({row[1]})' for row in sqlite_rows(db.path, f'PRAGMA table_info({name})')) +
                f' FROM {name}') for name in
            ('geometry_revisions', 'geometry_chunks', 'geometry_revision_chunks',
             'geometry_deltas', 'geometry_current_chunks', 'geometry_materializations')}


def test_managed_retire_restore_history_storage_and_reopen(db, db_path):
    uid = db.upsert_object(dict(kind='track', name='kept', meta={'note': 'original'}, tags=['saved']))
    geometry = db.put_geometry(uid, track([(1, 2, 3), (4, 5, 6)]))
    before = db.current_object_revision(uid)
    payloads = payload_tables(db)
    retire(db, uid)
    retired = db.current_object_revision(uid)
    assert retired['record']['retired'] and retired['geometry_revision'] == geometry
    assert retired['parent_uuid'] == before['uuid']
    assert db.object(uid)['retired']  # UUID lookup deliberately remains available
    assert db.current_geometry(uid)['uuid'] == geometry
    assert db.read_geometry_array(geometry, 'points') == struct.pack('<6i', 1, 2, 3, 4, 5, 6)
    assert uid in db.revision_selection_at(before['published_at_us'])
    assert uid not in db.revision_selection_at(retired['published_at_us'])
    assert db.revision_selection_at(retired['published_at_us'], include_retired=True)[uid] == retired['uuid']
    assert db.revision_read_view_at(retired['published_at_us'], include_retired=True)['objects'][0]['record']['retired']
    assert not db.object_revision(before['uuid'])['record']['retired']
    assert payload_tables(db) == payloads
    with ObjectStore.open(db_path, read_only=True) as reader:
        assert reader.object(uid)['retired'] and reader.object_count('track') == 0
    restore(db, uid)
    after = db.current_object_revision(uid)
    assert not after['record']['retired'] and after['parent_uuid'] == retired['uuid']
    assert after['record']['name'] == 'kept' and after['record']['meta']['note'] == 'original'
    assert after['record']['tags'] == ['saved']
    assert after['geometry_revision'] == geometry and payload_tables(db) == payloads
    assert db.revision_selection_at(after['published_at_us'])[uid] == after['uuid']
    with ObjectStore.open(db_path, read_only=True) as reader:
        assert not reader.object(uid)['retired'] and reader.object_count('track') == 1


@pytest.mark.parametrize('spatial_cache_bytes', [0, 1024 * 1024])
def test_all_current_queries_and_no_hidden_traversal(tmp_path, spatial_cache_bytes):
    with ObjectStore.create(str(tmp_path / 'visibility.vcobj'), author='test',
                            spatial_cache_bytes=spatial_cache_bytes) as db:
        ids = [db.upsert_object(dict(kind='track', tags=['saved'])) for _ in range(3)]
        a, middle, c = ids
        for uid in ids:
            db.put_geometry(uid, track([(1, 2, 3), (4, 5, 6)]))
        anchor = dict(whole_geometry=True)
        edges = [db.add_edge(dict(kind='connected', a=u, b=v, a_anchor=anchor, b_anchor=anchor))
                 for u, v in ((a, middle), (middle, c))]
        box = [0, 0, 0, 10, 10, 10]
        assert {r['object_uuid'] for r in db.spatial_regions(box)} == set(ids)  # warm cache
        retire(db, middle, a, c, edges=edges)
        for records in (db.objects_by_kind('track'), db.objects_by_tag('saved'), db.select_objects(),
                        db.select_objects(bbox=box), db.select_objects(fields=['uuid', 'retired'])):
            assert {r['uuid'] for r in records} == {a, c}
        assert db.object_count('track') == 2
        assert db.object_count('track', include_retired=True) == 3
        for records in (db.objects_by_kind('track', include_retired=True),
                        db.objects_by_tag('saved', include_retired=True),
                        db.select_objects(include_retired=True)):
            assert {r['uuid'] for r in records} == set(ids)
        assert {r['object_uuid'] for r in db.spatial_regions(box)} == {a, c}
        assert {r['object_uuid'] for r in db.spatial_regions(box, include_retired=True)} == set(ids)
        assert db.component(a) == [a] and db.component(middle) == []
        assert db.reachable_within(a, 2) == [a] and db.neighbours(a) == []
        assert db.neighbours(middle) == []
        assert {r['uuid'] for r in db.select_objects(connected_to=a)} == {a}
        assert set(db.component(a, include_retired=True)) == set(ids)
        assert set(db.reachable_within(a, 2, include_retired=True)) == set(ids)
        assert set(db.neighbours(middle, include_retired=True)) == {a, c}
        assert len(db.select_objects(connected_to=a, include_retired=True)) == 3
        assert db.edges_of(a) == db.edges_by_kind('connected') == db.select_edges() == []
        assert len(db.edges_of(middle, include_retired=True)) == 2
        assert len(db.edges_by_kind('connected', include_retired=True)) == 2
        assert len(db.select_edges(include_retired=True)) == 2
        for include, n, e in ((False, 2, 0), (True, 3, 2)):
            with db.graph_read(approval_states=[], include_retired=include) as graph:
                assert len(graph.object_uuids) == n and len(graph.edge_uuids) == e
                assert len(graph.objects()) == n and len(graph.edges()) == e
        assert len(json.loads(db.export_json())['objects']) == 2
        assert len(json.loads(db.export_json(include_retired=True))['objects']) == 3


def test_preserved_links_need_revalidation_after_restore(db):
    a, b, ga, gb, edge = linked(db)
    before = heads(db, a, b)
    recorded = db.recorded_relationships(before[a])
    retire(db, a, b, edges=[edge])
    assert heads(db, b)[b] == before[b]
    assert db.recorded_relationships(heads(db, a)[a]) == recorded
    assert not db.relationships_at({a: None, b: None})[0]['applicable']
    assert db.relationships_at(before)[0]['applicable']
    # Change the other endpoint while the source is retired. Inactive incident
    # relationships still participate in geometry validation.
    decision = edge_choice(db, edge, 'keep_recorded')
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.edit_geometry(b, [dict(array='points', element_offset=0, data=struct.pack('<i', 9))],
                         expected_revision=gb, edge_edits=[decision])
    restore(db, a, b, edges=[edge])
    assert not db.relationships_at({a: None, b: None})[0]['applicable']
    assert db.edges_of(a) == [] and db.component(a) == [a]
    assert db.recorded_relationships(heads(db, a)[a]) == recorded
    assert db.object_revision(recorded[0]['a_revision']) is not None
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(edge)
    assert db.relationships_at({a: None, b: None})[0]['applicable']
    assert len(db.edges_of(a)) == 1 and set(db.component(a)) == {a, b}
    assert db.relationships_at(before)[0]['applicable']


def test_restore_unchanged_geometry_does_not_reactivate_links(db):
    a, b, _, _, edge = linked(db)
    retire(db, a, b, edges=[edge])
    restore(db, a, b, edges=[edge])
    assert not db.relationships_at({a: None, b: None})[0]['applicable']
    assert db.edges_of(a) == []
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(edge)
    assert len(db.edges_of(a)) == 1


@pytest.mark.parametrize('failure', ['missing_decision', 'missing_endpoint', 'stale'])
def test_retirement_failure_rolls_back_whole_multi_object_action(db, failure):
    a, b, _, _, edge = linked(db)
    before = heads(db, a, b)
    journal = db.journal_since()
    output = str(uuid.uuid4())
    expected = dict(before, **{output: None})
    if failure == 'missing_endpoint': del expected[b]
    if failure == 'stale': expected[a] = db.object_history(a)[0]['uuid']
    with pytest.raises(RuntimeError, match='expected|explicit'):
        with db.write(expected_revisions=expected) as tx:
            tx.upsert_object(dict(uuid=output, kind='track', name='split output'))
            tx.retire_object(a)
            tx.set_meta(a, 'aborted', True)
            if failure != 'missing_decision': tx.keep_recorded_relationship(edge)
    assert heads(db, a, b) == before and db.journal_since() == journal
    assert db.object_count('track', include_retired=True) == 2


def test_explicit_remove_and_modified_relationship_decisions(db):
    a, b, _, _, edge = linked(db)
    before = heads(db, a, b)
    with db.write(expected_revisions=before) as tx:
        tx.retire_object(a)
        tx.remove_edge(edge)
    assert db.edge(edge) is None
    assert db.recorded_relationships(before[a])[0]['record']['uuid'] == edge
    restore(db, a)
    assert db.edges_of(a, include_retired=True) == []  # removal is not undone implicitly
    new = db.add_edge(dict(kind='declared_reference', a=a, b=b,
                          a_anchor=dict(whole_geometry=True), b_anchor=dict(whole_geometry=True)))
    changed = db.edge(new)
    changed['meta']['purpose'] = 'retained archival reference'
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retire_object(a)
        tx.update_edge(changed)
    assert db.edges_of(b) == []
    restore(db, a, b, edges=[new])
    assert db.edges_of(b) == []


def test_selected_historical_content_restore_and_subsequent_edits(db):
    uid = db.upsert_object(dict(kind='track', meta={'note': 'old'}))
    original = db.put_geometry(uid, track([(1, 2, 3)]))
    old = db.current_object_revision(uid)
    changed = db.edit_geometry(uid, [dict(array='points', element_offset=0, data=struct.pack('<i', 20))],
                               expected_revision=original)
    db.set_meta(uid, 'note', 'new')
    newer = db.current_object_revision(uid)
    retire(db, uid)
    restore(db, uid, source=old['uuid'])
    assert db.current_geometry(uid)['uuid'] == original
    assert db.object(uid)['meta']['note'] == 'old'
    assert db.read_geometry_array(changed, 'points') == struct.pack('<3i', 20, 2, 3)
    assert db.object_revision(newer['uuid'])['record']['meta']['note'] == 'new'
    assert db.spatial_regions([0, 0, 0, 3, 4, 5])
    assert not db.spatial_regions([19, 0, 0, 21, 4, 5])
    next_geometry = db.edit_geometry(uid, [dict(array='points', element_offset=0, data=struct.pack('<i', 30))],
                                     expected_revision=original)
    assert db.read_geometry_array(next_geometry, 'points') == struct.pack('<3i', 30, 2, 3)
    # Restore a delta revision too, reusing its immutable payload.
    retire(db, uid)
    restore(db, uid, source=newer['uuid'])
    assert db.current_geometry(uid)['uuid'] == changed
    assert db.read_geometry_array(changed, 'points') == struct.pack('<3i', 20, 2, 3)


def test_default_restore_last_active_content_and_invalid_sources(db):
    uid = db.upsert_object(dict(kind='note', name='active'))
    other = db.upsert_object(dict(kind='note'))
    retire(db, uid)
    retired = heads(db, uid)[uid]
    db.upsert_object(dict(uuid=uid, kind='note', name='archival edit'))
    assert db.object(uid)['retired']  # upsert cannot unretire
    for source in (retired, heads(db, other)[other]):
        with pytest.raises(ValueError, match='active revision'):
            restore(db, uid, source=source)
    restore(db, uid)
    assert db.object(uid)['name'] == 'active'
    before = heads(db, uid)
    restore(db, uid)
    assert heads(db, uid) == before  # already active is a no-op
    with pytest.raises(RuntimeError, match='expected_revisions'):
        with db.write() as tx:
            tx.retire_object(uid)


def test_declared_references_and_arbitrary_json_boundary(db):
    owner = db.upsert_object(dict(kind='note'))
    target = db.upsert_object(dict(kind='note'))
    anchor = dict(whole_geometry=True)
    ref = db.add_edge(dict(kind='declared_reference', a=owner, b=target,
                          a_anchor=anchor, b_anchor=anchor, meta={'field': 'input_uuid'}))
    historical = heads(db, owner, target)
    retire(db, target, owner, edges=[ref])
    assert db.recorded_relationships(historical[owner])[0]['record']['uuid'] == ref
    with db.write(expected_revisions=heads(db, owner, target)) as tx:
        tx.remove_edge(ref)
    # Live-row deletion is not purge: historical references still resolve.
    db.delete_object(target)
    record = db.recorded_relationships(historical[owner])[0]
    for pin in ('a_revision', 'b_revision'):
        assert db.object_revision(record[pin]) is not None
    untracked = db.upsert_object(dict(kind='note'))
    db.set_meta(owner, 'arbitrary_uuid', untracked)
    db.delete_object(untracked)
    assert db.object(owner)['meta']['arbitrary_uuid'] == untracked
    assert db.recorded_relationships(heads(db, owner)[owner]) == []


def test_schema_nine_migration_and_readonly_compatibility(tmp_path):
    path = str(tmp_path / 'old.vcobj')
    with ObjectStore.create(path, author='test') as db:
        uid = db.upsert_object(dict(kind='note'))
        before = heads(db, uid)[uid]
    with sqlite3.connect(path) as sql:
        sql.execute('DROP TABLE edge_lifecycle_bindings')
        sql.execute('DROP TABLE object_lifecycle')
        sql.execute('PRAGMA user_version=9')
    with ObjectStore.open(path, read_only=True) as db:
        assert not db.object(uid)['retired'] and db.object_count('note') == 1
    with ObjectStore.open(path, author='test') as db:
        assert heads(db, uid)[uid] == before
        retire(db, uid)
        assert db.object_count('note') == 0
        assert not db.object_revision(before)['record']['retired']
    with sqlite3.connect(path) as sql:
        assert sql.execute('PRAGMA user_version').fetchone()[0] == 12


def test_restore_decisions_and_validation_order_are_atomic(db):
    a, b, _, _, edge = linked(db)
    retire(db, a, b, edges=[edge])
    before = heads(db, a, b)
    for order in ('missing', 'before'):
        with pytest.raises(RuntimeError, match='explicit'):
            with db.write(expected_revisions=before) as tx:
                if order == 'before': tx.keep_recorded_relationship(edge)
                tx.restore_object(a)
                tx.set_meta(b, 'aborted', True)
        assert heads(db, a, b) == before and 'aborted' not in db.object(b)['meta']
    # An explicit validation after restoration can reactivate the shared edge
    # in this same atomic action; no intermediate lifecycle state is published.
    with db.write(expected_revisions=before) as tx:
        tx.restore_object(a)
        tx.retain_relationship(edge)
    assert db.relationships_at({a: None, b: None})[0]['applicable']
    assert len(db.edges_of(a)) == 1


def test_multi_object_retirement_and_new_geometry_publish_together(db):
    a, b, _, _, edge = linked(db)
    output = str(uuid.uuid4())
    expected = dict(heads(db, a, b), **{output: None})
    with db.write(expected_revisions=expected) as tx:
        tx.upsert_object(dict(uuid=output, kind='track', name='merged'))
        tx.put_geometry(output, track([(1, 2, 3), (4, 5, 6)]))
        tx.retire_object(a)
        tx.retire_object(b)
        tx.remove_edge(edge)
    revisions = [db.current_object_revision(obj) for obj in (a, b, output)]
    assert len({r['published_at_us'] for r in revisions}) == 1
    assert {r['uuid'] for r in db.select_objects()} == {output}
    assert set(db.revision_selection_at(revisions[0]['published_at_us'])) == {output}


def test_current_state_sync_preserves_retirement_and_dormant_links(db, tmp_path):
    a, b, _, _, edge = linked(db)
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        retire(db, a, b, edges=[edge])
        db.synchronize(peer)
        assert peer.object(a)['retired'] and peer.edges_of(b) == []
        restore(db, a, b, edges=[edge])
        db.synchronize(peer)
        assert not peer.object(a)['retired'] and peer.edges_of(a) == []
        assert not peer.relationships_at({a: None, b: None})[0]['applicable']
        # A fresh replica lacks the original active timeline. Import must
        # preserve the dormant pins via explicitly incomplete baselines.
        with ObjectStore.create(str(tmp_path / 'fresh.vcobj'), author='fresh') as fresh:
            db.synchronize(fresh)
            assert fresh.edges_of(a) == []
            assert not fresh.relationships_at({a: None, b: None})[0]['applicable']
            with db.write(expected_revisions=heads(db, a, b)) as tx:
                tx.retain_relationship(edge)
            db.synchronize(peer)
            db.synchronize(fresh)
            for replica in (db, peer, fresh):
                assert len(replica.edges_of(a)) == 1
                assert replica.relationships_at({a: None, b: None})[0]['applicable']


def test_metadata_export_does_not_present_dormant_edges_as_current(db):
    a, b, _, _, edge = linked(db)
    retire(db, a, b, edges=[edge])
    restore(db, a, b, edges=[edge])
    assert json.loads(db.export_json())['edges'] == []
    assert len(json.loads(db.export_json(include_retired=True))['edges']) == 1
    target = db.upsert_object(dict(kind='note'))
    retire(db, target)
    with pytest.raises(RuntimeError, match='lifecycle state'):
        db.import_json(db.export_json(object_kind='note', include_retired=True))


def test_fresh_replica_does_not_restore_an_untimed_reference_baseline(db, tmp_path):
    a, b, _, _, edge = linked(db)
    retire(db, a, b, edges=[edge])
    with ObjectStore.create(str(tmp_path / 'retired-peer.vcobj'), author='peer') as peer:
        db.synchronize(peer)
        assert peer.object(a)['retired']
        with pytest.raises(ValueError, match='active revision'):
            restore(peer, a, b, edges=[edge])
        restore(db, a, b, edges=[edge])
        db.synchronize(peer)
        assert not peer.object(a)['retired']
