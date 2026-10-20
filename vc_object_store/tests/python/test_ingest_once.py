"""One-time import and document-scoped PCL identity lookup."""
import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from vc_object_store import ObjectStore
from test_object_ingestion import fiber
from test_structural_geometry import sequence, point, heads


def write_collection(path, key, uid=None, points=None, **extra):
    uid = uid or str(uuid.uuid4())
    value = {'uuid': uid, 'points': points or {'1': {'p': [1, 2, 3]}}, **extra}
    path.write_text(json.dumps({'vc_pointcollections_json_version': '1', 'collections': {str(key): value}}))
    return uid


def test_repeat_modified_retired_and_mixed_batch_are_atomic(db, tmp_path):
    path = tmp_path / 'source.json'
    a = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    before = db.export_json()
    history = db.object_history(a)
    for modified in (False, True):
        write_collection(path, 10, a, {'1': {'p': [90 if modified else 1, 2, 3], 'wind_a': 42}})
        report = db.ingest_geometry_source('point_collection', path)
        assert report['ingested'] == report['unchanged'] == report['edges_imported'] == 0
        assert 're-import is unsupported' in report['problems'][0]
        assert db.export_json() == before and db.object_history(a) == history
    b = str(uuid.uuid4())
    document = json.loads(path.read_text())
    document['collections']['0'] = {'uuid': b, 'points': {'2': {'p': [4, 5, 6]}}}
    path.write_text(json.dumps(document))
    assert db.ingest_geometry_source('point_collection', path)['problems']
    assert db.object(b) is None and db.export_json() == before
    with db.write(expected_revisions=heads(db, a)) as tx:
        tx.retire_object(a)
    assert db.ingest_geometry_source('point_collection', path)['problems']
    assert db.object(a)['retired']


def test_duplicate_fiber_uuid_in_batch_rolls_back(db, tmp_path):
    folder = tmp_path / 'fibers'; folder.mkdir()
    uid = str(uuid.uuid4())
    for name in ('a.json', 'b.json'):
        (folder / name).write_text(json.dumps(fiber(uid)))
    report = db.ingest_geometry_source('fiber', folder)
    assert 'duplicate object UUID' in report['problems'][0]
    assert db.object(uid) is None


def test_registered_placeholder_loads_once_and_history_prevents_reload(db, tmp_path):
    path = tmp_path / 'fiber.json'; uid = str(uuid.uuid4())
    path.write_text(json.dumps(fiber(uid)))
    db.upsert_object({'uuid': uid, 'kind': 'fiber', 'storage_kind': 'file', 'storage_ref': str(path)})
    placeholder = db.current_object_revision(uid)['uuid']
    assert db.ingest_geometry_source('fiber', path)['problems']
    assert not db.ingest_registered_geometry(object_uuids=[uid])['problems']
    before = db.current_object_revision(uid)
    assert db.ingest_registered_geometry(object_uuids=[uid])['problems']
    assert db.current_object_revision(uid) == before
    with db.write(expected_revisions=heads(db, uid)) as tx:
        tx.restore_object(uid, source_revision=placeholder)
    assert db.current_geometry(uid) is None
    assert db.ingest_registered_geometry(object_uuids=[uid])['problems']


def test_concurrent_import_publishes_one_object(db, tmp_path):
    path = tmp_path / 'source.json'; uid = write_collection(path, 10)
    barrier = Barrier(2)
    def ingest():
        with ObjectStore.open(db.path, author='concurrent') as peer:
            barrier.wait()
            return peer.ingest_geometry_source('point_collection', path)
    with ThreadPoolExecutor(2) as workers:
        reports = list(workers.map(lambda _: ingest(), range(2)))
    assert sorted(r['ingested'] for r in reports) == [0, 1]
    assert sum(bool(r['problems']) for r in reports) == 1
    assert len(db.object_history(uid)) == 1


def test_document_scoping_links_duplicate_and_missing_targets(db, tmp_path):
    path = tmp_path / 'source.json'; other = tmp_path / 'other.json'
    a = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    unrelated = write_collection(other, 10)
    assert not db.ingest_geometry_source('point_collection', other)['problems']
    b = write_collection(path, 20, points={'2': {'p': [4, 5, 6], 'links': [1]}}, windings_linked=[10])
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    assert db.neighbours(b) == [a] and db.edges_of(unrelated) == []
    edge = next(e for e in db.edges_of(b) if e['kind'] == 'point_link')
    anchors = {edge['a']: edge['a_anchor'], edge['b']: edge['b_anchor']}
    assert anchors[a]['position'] == (1, 2, 3) and anchors[b]['position'] == (4, 5, 6)
    for key, points, error in [(30, {'1': {'p': [9, 9, 9]}}, 'duplicate point ID'),
                               (10, {'3': {'p': [9, 9, 9]}}, 'duplicate source collection ID'),
                               (30, {'3': {'p': [9, 9, 9], 'links': [999]}}, 'target not ingested')]:
        uid = write_collection(path, key, points=points)
        report = db.ingest_geometry_source('point_collection', path)
        assert error in report['problems'][0]
        assert db.object(uid) is None


def test_index_tracks_structural_changes_metadata_and_reopen(db, tmp_path):
    path = tmp_path / 'source.json'; a = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    sequence(db, a, 1, 0, [point(2, 20)])
    sequence(db, a, 0, 1)
    with ObjectStore.open(db.path, author='reopened') as reopened:
        b = write_collection(path, 20, points={'3': {'p': [3, 2, 3], 'links': [2]}})
        assert not reopened.ingest_geometry_source('point_collection', path)['problems']
        edge = reopened.edges_of(b)[0]
        assert (edge['a_anchor'] if edge['a'] == a else edge['b_anchor'])['position'] == (20, 2, 3)
        uid = write_collection(path, 30, points={'4': {'p': [4, 2, 3], 'links': [1]}})
        assert reopened.ingest_geometry_source('point_collection', path)['problems']
        assert reopened.object(uid) is None
    # Rename the source namespace; derived rows move transactionally.
    source = db.object(a)['meta']['source']; source['origin'] = str(tmp_path / 'renamed.json')
    db.set_meta(a, 'source', source)
    with sqlite3.connect(db.path) as sql:
        assert sql.execute('SELECT origin,point_id FROM source_points WHERE object_uuid=?', (uuid.UUID(a).bytes,)).fetchall() == [(source['origin'], 2)]
        assert sql.execute('SELECT count(*) FROM source_identity_dirty').fetchone()[0] == 0


def test_v11_migration_backfills_index_without_new_history(db, tmp_path):
    path = tmp_path / 'source.json'; uid = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    before = db.object_history(uid)
    with sqlite3.connect(db.path) as sql:
        for (name,) in sql.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'source_identity_%'").fetchall():
            sql.execute(f'DROP TRIGGER {name}')
        for name in ('source_points', 'source_collections', 'source_identity_dirty'):
            sql.execute(f'DROP TABLE {name}')
        sql.execute('PRAGMA user_version=11')
    with ObjectStore.open(db.path, read_only=True) as legacy:
        assert legacy.current_geometry(uid) is not None
    with ObjectStore.open(db.path, author='migration') as migrated:
        assert migrated.object_history(uid) == before
        new = write_collection(path, 20, points={'2': {'p': [4, 5, 6], 'links': [1]}})
        assert not migrated.ingest_geometry_source('point_collection', path)['problems']
        assert migrated.neighbours(new) == [uid]


def test_index_tracks_restore_sync_and_sparse_coordinates(db, tmp_path):
    import struct
    path = tmp_path / 'source.json'; a = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    original = db.current_object_revision(a)['uuid']
    sequence(db, a, 1, 0, [point(2, 20)])
    with db.write(expected_revisions=heads(db, a)) as tx:
        tx.restore_object(a, source_revision=original)
    old = db.current_geometry(a)['uuid']
    with db.write(expected_revisions=heads(db, a)) as tx:
        tx.edit_geometry(a, [{'array': 'points', 'element_offset': 0, 'data': struct.pack('<d', 99)}], old)
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        assert db.synchronize(peer)['applied']
        with sqlite3.connect(peer.path) as sql:
            assert sql.execute('SELECT point_id FROM source_points').fetchall() == [(1,)]
            assert sql.execute('SELECT count(*) FROM source_identity_dirty').fetchone()[0] == 0
        b = write_collection(path, 20, points={'3': {'p': [3, 2, 3], 'links': [1]}})
        assert not peer.ingest_geometry_source('point_collection', path)['problems']
        edge = peer.edges_of(b)[0]
        assert (edge['a_anchor'] if edge['a'] == a else edge['b_anchor'])['position'] == (99, 2, 3)


def test_retired_owner_does_not_steal_transferred_point(db, tmp_path):
    path = tmp_path / 'source.json'; a = write_collection(path, 10)
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    original = db.current_geometry(a)['uuid']
    b = write_collection(path, 20, points={'2': {'p': [2, 2, 3]}})
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.edit_geometry_sequence(b, {'array': 'points', 'start': 1, 'erase_count': 0,
            'parts': [{'revision': original, 'start': 0, 'count': 1}]}, db.current_geometry(b)['uuid'])
        tx.retire_object(a)
    c = write_collection(path, 30, points={'3': {'p': [3, 2, 3], 'links': [1]}})
    assert not db.ingest_geometry_source('point_collection', path)['problems']
    assert db.neighbours(c) == [b]
    assert db.current_geometry(a)['uuid'] == original


def test_deleted_uuid_cannot_be_reused_by_source_import(db, tmp_path):
    uid = db.upsert_object({'kind': 'note'})
    db.delete_object(uid)
    path = tmp_path / 'fiber.json'
    path.write_text(json.dumps(fiber(uid)))
    report = db.ingest_geometry_source('fiber', path)
    assert 're-import is unsupported' in report['problems'][0]
    assert db.object(uid) is None
