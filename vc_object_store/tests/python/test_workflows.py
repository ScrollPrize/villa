from pathlib import Path
import sqlite3
import struct
import subprocess
import sys

import numpy as np
import pytest
from vc_object_store import ObjectStore, PointCollection, Fiber, Track, Patch
from object_helpers import edge_choice


def heads(db, *ids):
    return {uid: db.current_object_revision(uid)['uuid'] for uid in ids}


def put(db, geometry, kind='track', **options):
    record = dict(kind=kind)
    if kind == 'track':
        record['track'] = {'external_key': str(db.object_count('track', include_retired=True)), 'family': 'vertical'}
    uid = db.upsert_object(record)
    return uid, db.put_geometry(uid, geometry, **options)


def array(name, values):
    return dict(name=name, dtype=values.dtype.str, shape=list(values.shape), data=values.tobytes())


@pytest.mark.parametrize('kind,geometry,array_name', [
    ('point_collection', PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]), 'points'),
    ('fiber', Fiber([[1, 2, 3], [4, 5, 6]], []), 'control_points'),
    ('track', Track([[1, 2, 3], [4, 5, 6]]), 'points'),
    ('patch', Patch(1, 2, [1, 1], [[1, 2, 3], [4, 5, 6]]), 'points'),
])
def test_owned_views_native_export_and_one_time_ingest(db, tmp_path, kind, geometry, array_name):
    uid, revision = put(db, geometry, kind)
    raw = db.read_geometry_array(revision, array_name)
    with db.geometry_snapshot() as snapshot:
        view = snapshot.geometry(uid)
        owned = view.load()
        clone, copied = put(db, owned, kind)
        assert db.read_geometry_array(copied, array_name) == raw
        with pytest.raises(RuntimeError, match='active'):
            db.close()
    with pytest.raises(RuntimeError, match='closed'):
        view.load()
    coordinates = {'volume_uuid': '20260411134726', 'scale_factor': 2**60 + 1}
    db.set_meta(uid, 'coordinates', coordinates)
    binding = {'active_volume_id': 'display-only', 'shape_zyx': [8, 9, 10]}
    db.set_meta(uid, 'coordinate_binding', binding)
    db.set_meta(uid, 'source', {'coordinates': {'scale_factor': 7}, 'volume_uuid': 'stale', 'coordinate_binding': {'active_volume_id': 'stale'}, 'retained_note': 'native'})
    output = tmp_path / 'native'
    assert db.export_geometry(uid, output) == revision
    with pytest.raises((RuntimeError, ValueError)):
        db.export_geometry(uid, output)
    with ObjectStore.create(str(tmp_path / 'import.vcobj'), author='import') as other:
        report = other.ingest_geometry_source(kind, output)
        assert not report['problems'] and report['ingested'] == 1
        imported = other.objects_by_kind(kind)[0]['uuid']
        assert other.object(imported)['meta']['coordinates'] == coordinates
        assert other.object(imported)['meta']['coordinate_binding'] == binding
        assert other.object(imported)['meta']['source']['retained_note'] == 'native'
        assert 'coordinates' not in other.object(imported)['meta']['source']
        current = other.current_geometry(imported)['uuid']
        expected = raw if kind != 'point_collection' else np.asarray(geometry.points, dtype='<f8')[np.argsort(geometry.ids)].tobytes()
        assert other.read_geometry_array(current, array_name) == expected
        before = other.journal_since()
        assert other.ingest_geometry_source(kind, output)['problems']
        assert other.journal_since() == before


@pytest.mark.parametrize('tile', [0, 32, 64, 128])
def test_surface_channels_sparse_reads_history_and_resize(db, tile):
    values = np.arange(35 * 37 * 3, dtype='<f4').reshape(35, 37, 3)
    values.view('<u4').flat[3:7] = [0x7fc12345, 0x80000000, 0x7f800000, 0xff800000]
    channel = np.arange(35 * 37, dtype='<u2').reshape(35, 37)
    geometry = dict(kind='patch', attributes={'scale': [1, 1]},
                    arrays=[array('points', values), array('review', channel)])
    uid, first = put(db, geometry, 'patch', patch_tile_size=tile)
    for name, original in [('points', values), ('review', channel)]:
        assert db.read_geometry_array(first, name) == original.tobytes()
        rect = db.read_geometry_rectangle(first, name, 31, 31, 3, 4)
        assert rect == original[31:34, 31:35].tobytes()
        assert db.read_geometry_rectangles(first, name, [[31, 31, 3, 4]] * 2) == [rect] * 2
        ranges = [[1, 7], [0, 0], [3, 11], [1, 7]]
        assert db.read_geometry_ranges(first, name, ranges) == [original.tobytes()[o:o+n] for o, n in ranges]
        packed = db.read_geometry_arrays([first, first], name)
        assert packed.values.tobytes() == original.tobytes() * 2
        assert packed.offsets.tolist() == [0, 35, 70]
    second = db.edit_geometry_rectangle(uid, 'review', 31, 31, 2, 3, b'\xff\xff' * 6, first)
    channel[31:33, 31:34] = 65535
    assert db.read_geometry_array(second, 'review') == channel.tobytes()
    assert db.read_geometry_array(first, 'review') != channel.tobytes()
    assert db.edit_geometry(uid, [], second) == second
    with pytest.raises(RuntimeError, match='revision'):
        db.edit_geometry(uid, [], first)
    third = db.update_surface_channels(uid, [array('extra', np.zeros((35, 37), 'u1'))], ['review'], second)
    fills = [array('points', np.full((1, 1, 3), -1, '<f4')), array('extra', np.ones((1, 1), 'u1'))]
    with db.write(expected_revisions=heads(db, uid)) as tx:
        grown = tx.resize_surface(uid, dict(row=-1, column=-1, rows=37, columns=39, fill=fills), third)
    assert db.read_geometry_rectangle(grown, 'points', 1, 1, 35, 37) == values.tobytes()
    assert db.read_geometry_rectangle(grown, 'extra', 0, 0, 1, 1) == b'\x01'
    assert db.geometry_revision(grown)['grid_origin'] == [-1, -1]
    extra = np.arange(37 * 39, dtype='<u2').reshape(37, 39)
    updated = db.update_surface_channels(uid, [array('extra', extra)], [], grown)
    assert db.read_geometry_array(updated, 'extra') == extra.tobytes()
    changed = db.edit_geometry_rectangle(uid, 'extra', 31, 31, 2, 3, b'\xff\xff' * 6, updated)
    extra[31:33, 31:34] = 65535
    assert db.read_geometry_array(changed, 'extra') == extra.tobytes()
    assert db.read_geometry_rectangle(grown, 'extra', 0, 0, 1, 1) == b'\x01'
    with ObjectStore.create(db.path + '.peer', author='peer') as peer:
        assert db.synchronize(peer)['applied']
        assert peer.read_geometry_array(changed, 'extra') == extra.tobytes()
        assert peer.read_geometry_array(first, 'points') == values.tobytes()
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert ro.read_geometry_array(second, 'review') == channel.tobytes()


def test_checked_relationships_lifecycle_graph_and_sync(db, tmp_path):
    a, first = put(db, Track([[1, 2, 3], [4, 5, 6]]))
    b, _ = put(db, Track([[1, 2, 3]]))
    anchor = dict(points=[dict(array='points', index=[0])], metadata_keys=['coordinates'])
    edge = db.add_edge(dict(a=a, b=b, kind='fiber_branch', a_anchor=anchor, b_anchor=anchor))
    db.set_approval(edge, 'approved')
    selected = heads(db, a, b)
    assert db.component(a, adjacent=False, applicable_only=True) == [a, b]
    assert db.component(a, adjacent=True) == [a]
    assert db.neighbours(a) == [b] and db.edges_by_kind('fiber_branch')[0]['uuid'] == edge
    before = db.journal_since()
    with pytest.raises(RuntimeError):
        with db.write(expected_revisions={a: selected[a]}) as tx:
            tx.set_meta(a, 'rollback', True)
            tx.remove_edge(edge)  # b is also modified, but has no expectation
    assert db.journal_since() == before and db.edge(edge)
    with db.graph_read(revisions=selected, approval_states=[]) as graph:
        edits = [dict(array='points', element_offset=0, data=struct.pack('<i', 9))]
        assert db.affected_geometry_edit_edges(a, edits, first)[0]['uuid'] == edge
        with pytest.raises(RuntimeError):
            db.edit_geometry(a, edits, first)
        second = db.edit_geometry(a, edits, first, edge_edits=[edge_choice(db, edge, 'keep_recorded')])
        assert graph.read_geometry_arrays('points').values.tolist() in (
            [[1, 2, 3], [4, 5, 6], [1, 2, 3]], [[1, 2, 3], [1, 2, 3], [4, 5, 6]])
        assert graph.component(a) == [a, b]
    assert db.component(a, applicable_only=True) == [a]
    assert db.relationships_at(selected)[0]['applicable']
    assert db.recorded_relationships(selected[a]) == db.recorded_relationships(selected[b])
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.retain_relationship(edge)
    assert db.component(a, applicable_only=True) == [a, b]
    for action in ('retire_object', 'restore_object'):
        with db.write(expected_revisions=heads(db, a, b)) as tx:
            getattr(tx, action)(a)
            tx.keep_recorded_relationship(edge)
        assert db.current_geometry(a)['uuid'] == second
        assert db.component(b, applicable_only=True) == [b]
    assert not db.object(a)['retired']
    assert db.object_revision(selected[a])['geometry_revision'] == first
    at = db.object_revision(selected[a])['published_at_us']
    assert db.revision_read_view_at(at)['selection'] == selected
    with ObjectStore.create(str(tmp_path / 'peer.vcobj'), author='peer') as peer:
        assert db.synchronize(peer)['applied']
        db.set_meta(a, 'note', 'local')
        peer.set_meta(a, 'note', 'remote')
        preview = db.synchronize(peer, dry_run=True)
        assert preview['conflicts'] and not preview['applied']
        assert db.synchronize(peer, resolutions={a: 'remote'}, expected_plan=preview['plan_token'])['applied']
        assert db.object(a)['meta']['note'] == 'remote'
        assert peer.read_geometry_array(first, 'points') == db.read_geometry_array(first, 'points')


def test_stable_ids_annotations_splice_and_coordinates(db):
    label = {'label': 'A', 'note': 'x' * 70000}
    uid = db.upsert_object(dict(kind='point_collection', meta={'source': {
        'point_attributes': [label, {'label': 'B'}]}}))
    first = db.put_geometry(uid, PointCollection([10, 20], [[1, 2, 3], [4, 5, 6]]))
    with db.write(expected_revisions=heads(db, uid)) as tx:
        second = tx.edit_geometry_sequence(uid, dict(array='points', start=1, erase_count=0,
            parts=[dict(count=1, ids=[30], data=struct.pack('<3d', 7, 8, 9), annotations=[{'label': 'C'}])]), first)
        tx.set_meta(uid, 'coordinates', {'scale_factor': 4})
    assert db.geometry_point_annotations(second, 'points') == [label, {'label': 'C'}, {'label': 'B'}]
    assert db.read_geometry_array(second, 'point_ids') == struct.pack('<3q', 10, 30, 20)
    assert db.object(uid)['meta']['coordinates'] == {'volume_uuid': None, 'scale_factor': 4}
    with db.geometry_snapshot() as snapshot:
        view = snapshot.point_collection(uid)
        assert view.point(30) == [7, 8, 9]
        with db.write(expected_revisions={uid: view.object_revision}) as tx:
            third = tx.edit_geometry(uid, [dict(array="points", element_offset=3,
                data=struct.pack("<3d", 10, 11, 12))], view.revision["uuid"])
        assert view.point(30) == [7, 8, 9]
    assert db.read_geometry_array(third, 'points')[24:48] == struct.pack('<3d', 10, 11, 12)
    with ObjectStore.create(db.path + '.peer', author='peer') as peer:
        assert db.synchronize(peer)['applied']
        assert peer.geometry_point_annotations(third, 'points') == [label, {'label': 'C'}, {'label': 'B'}]


@pytest.mark.parametrize('damage', ['payload', 'manifest', 'missing_chunk'])
def test_reads_detect_corruption_after_success(db, damage):
    _, revision = put(db, Track([[1, 2, 3]]))
    expected = db.read_geometry_array(revision, 'points')
    sql = {
        'payload': 'DROP TRIGGER geometry_chunks_immutable_update; UPDATE geometry_chunks SET data=zeroblob(length(data));',
        'manifest': "DROP TRIGGER geometry_revisions_immutable_update; UPDATE geometry_revisions SET manifest='{}';",
        'missing_chunk': 'DROP TRIGGER geometry_revision_chunks_immutable_delete; DELETE FROM geometry_revision_chunks;',
    }[damage]
    # Separate SQLite builds must not share POSIX WAL locks in this process.
    subprocess.run([sys.executable, '-c', 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.executescript(sys.argv[2]); c.close()', db.path, sql], check=True)
    for read in (lambda: db.read_geometry_array(revision, 'points'),
                 lambda: db.read_geometry_arrays([revision], 'points')):
        with pytest.raises(RuntimeError):
            read()
    assert expected == struct.pack('<3i', 1, 2, 3)


def test_reopen_and_metadata_roundtrip(db, tmp_path):
    uid, revision = put(db, Track([[1, 2, 3]]))
    db.set_meta(uid, 'note', {'value': None})
    db.set_approval(uid, 'approved')
    assert db.object(uid)['approved_by'] == 'pytest'
    db.remove_meta(uid, 'note')
    path = db.path
    db.close()
    with ObjectStore.open(path, author='reopened') as reopened:
        assert reopened.read_geometry_array(revision, 'points') == struct.pack('<3i', 1, 2, 3)
        assert reopened.select_objects(object_kind='track', fields=['name'])[0]['uuid'] == uid
        note = reopened.upsert_object(dict(kind='note', tags=['tag'], meta={'value': None}))
        assert reopened.objects_by_tag('tag')[0]['uuid'] == note
        exported = reopened.export_json(object_kind='note')
        with ObjectStore.create(str(tmp_path / 'metadata.vcobj'), author='import') as other:
            other.import_json(exported)
            assert other.object(note)['meta'] == {'value': None}
        reopened.delete_object(note)
        assert reopened.object(note) is None


@pytest.mark.parametrize('version', [0, 999])
@pytest.mark.parametrize('read_only', [False, True])
def test_unsupported_schema_is_rejected(db, version, read_only):
    path = db.path
    db.close()
    with sqlite3.connect(path) as sql:
        sql.execute(f'PRAGMA user_version={version}')
    sql.close()
    before = Path(path).read_bytes()
    with pytest.raises(RuntimeError, match='unsupported database schema'):
        ObjectStore.open(path, author='test', read_only=read_only)
    assert Path(path).read_bytes() == before


@pytest.mark.parametrize('kind,detail', [
    ('fiber', dict(family='V', manual_family=True, control_point_count=2, line_point_count=3,
                  length=4.5, width=2.5, width_gap_fraction=.3, optimization_mode='fit', bbox=[0., 1., 2., 3., 4., 5.])),
    ('patch', dict(grid_width=3, grid_height=2, scale=(.5, .25), bbox=None)),
    ('track', dict(dataset_ref='dataset', external_key='track', family='H', point_count=3, length=4.5, bbox=None)),
])
def test_record_fields_and_transaction_bindings(db, kind, detail):
    with db.write() as tx:
        uid = tx.upsert_object(dict(kind=kind, name='sample', format_version=2,
            storage_kind='file', storage_ref='source', data_generation=3, data_checksum='checksum',
            tags=iter(['review']), meta={'annotation': None}, **{kind: detail}))
        tx.set_meta(uid, 'remove', True)
        tx.remove_meta(uid, 'remove')
        tx.set_approval(uid, 'approved')
    record = db.object(uid)
    assert record[kind] == detail and record['tags'] == ['review']
    assert record['approved_by'] == 'pytest' and record['approved_generation'] == record['generation']
    fields = ['name', 'format_version', 'storage_kind', 'storage_ref', 'data_generation', 'data_checksum', kind]
    assert db.select_objects(fields=fields) == [{key: record[key] for key in ['uuid', 'kind', *fields]}]
    with ObjectStore.create(db.path + '.copy', author='copy') as other:
        other.import_json(db.export_json())
        assert other.object(uid) == record
    with db.write() as tx:
        peer = tx.upsert_object(dict(kind='note'))
        edge = tx.add_edge(dict(a=uid, b=peer, kind='related'))
        tx.remove_edge(edge)
    assert db.edge(edge) is None
    with pytest.raises(RuntimeError, match='finished'):
        tx.set_meta(uid, 'invalid', True)
    with pytest.raises((RuntimeError, ValueError)):
        with db.write() as tx:
            tx.upsert_object(dict(kind=kind, **{kind: {'bbox': [1, 2]}}))
    assert db.object_count(kind) == 1


def test_fiber_branch_transfer_preserves_identity_approval_and_stable_ids(db, tmp_path):
    import json
    import uuid
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    edge_uuid = str(uuid.uuid4())
    for uid in ids:
        db.upsert_object(dict(uuid=uid, kind='fiber', name='same name'))
        geometry = Fiber([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]])
        geometry.control_point_ids = [90, 10]
        db.put_geometry(uid, geometry)
    anchor = lambda ident, position: dict(points=[dict(array='control_point_ids', index=[ident])],
                                        position=position, direction=[1, 0, 0])
    db.add_edge(dict(uuid=edge_uuid, kind='fiber_branch', a=ids[0], b=ids[1],
                    a_anchor=anchor(10, [4, 5, 6]), b_anchor=anchor(90, [1, 2, 3])))
    db.set_approval(edge_uuid, 'approved')
    directory = tmp_path / 'fibers'
    directory.mkdir()
    for uid in ids:
        db.export_geometry(uid, directory / (uid + '.json'))
    first = json.loads((directory / (ids[0] + '.json')).read_text())
    branch = first['branches'][0]
    assert branch['edge_uuid'] == edge_uuid and branch['pending'] is False
    assert branch['control_point_index'] == 1 and branch['branch_control_point_index'] == 0
    assert branch['target_uuid'] == ids[1]
    assert 'optimization_mode' not in first
    with ObjectStore.create(str(tmp_path / 'imported.vcobj'), author='test') as other:
        report = other.ingest_geometry_source('fiber', directory)
        assert report == dict(ingested=2, edges_imported=1, problems=[])
        edge = other.edges_by_kind('fiber_branch')[0]
        assert edge['uuid'] == edge_uuid and edge['approval_state'] == 'approved'
    # Conflicting reciprocal declarations must roll back the whole batch.
    first['branches'][0]['pending'] = True
    (directory / (ids[0] + '.json')).write_text(json.dumps(first))
    with ObjectStore.create(str(tmp_path / 'conflict.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', directory)['problems']
        assert other.object_count('fiber') == 0


def test_fiber_collection_and_linked_file_import_use_uuid_endpoints(db, tmp_path):
    import json
    import uuid
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    def fiber(uid, target, filename):
        return dict(type='vc3d_fiber', version=4, uuid=uid, name='same label',
                    control_point_ids=[10, 20], control_points=[{'position': [0, 0, 0]}, {'position': [1, 0, 0]}],
                    line_points=[[0, 0, 0], [1, 0, 0]], branches=[dict(
                        target_uuid=target, branch_file=filename, control_point_index=0,
                        branch_control_point_index=0, control_point_position=[0, 0, 0],
                        branch_control_point_position=[0, 0, 0], control_point_direction=[1, 0, 0],
                        branch_control_point_direction=[1, 0, 0])])
    first, second = fiber(a, b, 'peer.json'), fiber(b, a, 'root.json')
    root = tmp_path / 'root.json'
    root.write_text(json.dumps(first))
    (tmp_path / 'peer.json').write_text(json.dumps(second))
    result = db.ingest_geometry_source('fiber', root)
    assert not result['problems'] and result['ingested'] == 2 and result['edges_imported'] == 1
    # Exported UUID-addressed documents can be bundled without their original files.
    documents = []
    for uid in (a, b):
        output = tmp_path / (uid + '.json')
        db.export_geometry(uid, output)
        documents.append(json.loads(output.read_text()))
    db.upsert_object(dict(kind="fiber", name="unrelated pending geometry"))
    historical_pin = db.current_object_revision(a)['uuid']
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.remove_edge(documents[0]['branches'][0]['edge_uuid'])
    historical = tmp_path / 'historical-graph.json'
    db.export_geometry(a, historical, object_revision_uuid=historical_pin)
    assert json.loads(historical.read_text()) == documents[0]
    current = tmp_path / 'current-graph.json'
    db.export_geometry(a, current)
    assert not json.loads(current.read_text())['branches']
    collection = tmp_path / 'collection.json'
    collection.write_text(json.dumps(dict(type='vc3d_fiber_collection', version=1, point_collections=documents)))
    with ObjectStore.create(str(tmp_path / 'roundtrip.vcobj'), author='test') as other:
        result = other.ingest_geometry_source('fiber', collection)
        assert not result['problems'] and result['ingested'] == 2 and result['edges_imported'] == 1
    # No filename-based endpoint repair, even when both files are available.
    del first['branches'][0]['target_uuid']
    root.write_text(json.dumps(first))
    with ObjectStore.create(str(tmp_path / 'invalid.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', root)['problems']
        assert other.object_count('fiber') == 0
    collection.write_text(json.dumps(documents))
    with ObjectStore.create(str(tmp_path / 'bare.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', collection)['problems']
        assert other.object_count('fiber') == 0


@pytest.mark.parametrize('kind,geometry', [
    ('fiber', Fiber([[1, 2, 3], [4, 5, 6]], [])),
    ('point_collection', PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]])),
])
def test_native_export_pins_object_metadata_and_geometry(db, tmp_path, kind, geometry):
    import json
    uid, first_geometry = put(db, geometry, kind)
    db.set_meta(uid, 'coordinates', {'scale_factor': 2})
    before = db.current_object_revision(uid)['uuid']
    db.set_meta(uid, 'coordinates', {'scale_factor': 4})
    output = tmp_path / 'historical.json'
    assert db.export_geometry(uid, output, object_revision_uuid=before) == first_geometry
    doc = json.loads(output.read_text())
    if kind == 'point_collection':
        doc = next(iter(doc['collections'].values()))
    assert doc['coordinates']['scale_factor'] == 2
    with pytest.raises(RuntimeError, match='object revision'):
        db.export_geometry(uid, tmp_path / 'wrong.json', object_revision_uuid=first_geometry)
