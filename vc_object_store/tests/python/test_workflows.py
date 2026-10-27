from pathlib import Path
import json
import struct
import subprocess
import sys
import uuid

import numpy as np
import pytest
from vc_object_store import ObjectStore, PointCollection, Fiber, Track, Surface, Umbilicus
from object_helpers import export, raw


def put(db, geometry, kind='track', **options):
    record = dict(kind=kind)
    if kind == 'track':
        record['track'] = {'external_key': str(len(db.objects_by_kind('track', include_retired=True))), 'family': 'vertical'}
    uid = db.upsert_object(record)
    return uid, db.put_geometry(uid, geometry, **options)


def array(name, values):
    return dict(name=name, dtype=values.dtype.str, shape=list(values.shape), data=values.tobytes())


@pytest.mark.parametrize('kind,geometry,array_name', [
    ('point_collection', PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]]), 'points'),
    ('fiber', Fiber([[1, 2, 3], [4, 5, 6]], []), 'control_points'),
    ('track', Track([[1, 2, 3], [4, 5, 6]]), 'points'),
    ('surface', Surface(1, 2, [1, 1], [[1, 2, 3], [4, 5, 6]]), 'points'),
    ('umbilicus', Umbilicus([[10.5, 20, 3], [11, 21, 400]]), 'points'),
])
def test_owned_views_native_export_and_one_time_ingest(db, tmp_path, kind, geometry, array_name):
    uid, revision = put(db, geometry, kind)
    stored = raw(db, revision, array_name)
    with db.geometry_snapshot() as snapshot:
        view = snapshot.geometry(uid)
        owned = view.load()
        clone, copied = put(db, owned, kind)
        assert raw(db, copied, array_name) == stored
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
    exported = export(db, uid, output)
    assert exported.returncode == 0 and revision in exported.stdout
    assert export(db, uid, output).returncode != 0  # an existing destination is protected
    with ObjectStore.create(str(tmp_path / 'import.vcobj'), author='import') as other:
        report = other.ingest_geometry_source(kind, output)
        assert not report['problems'] and report['ingested'] == 1
        imported = other.objects_by_kind(kind)[0]['uuid']
        assert other.object(imported)['meta']['coordinates'] == coordinates
        assert other.object(imported)['meta']['coordinate_binding'] == binding
        assert other.object(imported)['meta']['source']['retained_note'] == 'native'
        assert 'coordinates' not in other.object(imported)['meta']['source']
        current = other.current_geometry(imported)['uuid']
        expected = stored if kind != 'point_collection' else np.asarray(geometry.points, dtype='<f8')[np.argsort(geometry.ids)].tobytes()
        assert raw(other, current, array_name) == expected
        assert other.ingest_geometry_source(kind, output)['problems']  # ingestion is one-time
        assert len(other.objects_by_kind(kind)) == 1


@pytest.mark.parametrize('tile', [0, 32, 64, 128])
def test_surface_channels_read_back_from_any_tiling(db, tile):
    values = np.arange(35 * 37 * 3, dtype='<f4').reshape(35, 37, 3)
    values.view('<u4').flat[3:7] = [0x7fc12345, 0x80000000, 0x7f800000, 0xff800000]
    channel = np.arange(35 * 37, dtype='<u2').reshape(35, 37)
    geometry = dict(kind='surface', attributes={'scale': [1, 1]},
                    arrays=[array('points', values), array('review', channel)])
    uid, first = put(db, geometry, 'surface', surface_tile_size=tile)
    for name, original in [('points', values), ('review', channel)]:
        packed = db.read_geometry_arrays([first, first], name)
        assert packed.values.tobytes() == original.tobytes() * 2  # exact bits, NaN payloads included
        assert packed.offsets.tolist() == [0, 35, 70]
    with ObjectStore.open(db.path, read_only=True) as ro:
        assert raw(ro, first, 'review') == channel.tobytes()


@pytest.mark.parametrize('damage', ['payload', 'manifest', 'missing_chunk'])
def test_reads_detect_corruption_after_success(db, damage):
    _, revision = put(db, Track([[1, 2, 3]]))
    expected = raw(db, revision, 'points')
    sql = {
        'payload': 'DROP TRIGGER geometry_chunks_immutable_update; UPDATE geometry_chunks SET data=zeroblob(length(data));',
        'manifest': "DROP TRIGGER geometry_revisions_immutable_update; UPDATE geometry_revisions SET manifest='{}';",
        'missing_chunk': 'DROP TRIGGER geometry_revision_chunks_immutable_delete; DELETE FROM geometry_revision_chunks;',
    }[damage]
    # Separate SQLite builds must not share POSIX WAL locks in this process.
    subprocess.run([sys.executable, '-c', 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.executescript(sys.argv[2]); c.close()', db.path, sql], check=True)
    with pytest.raises(RuntimeError):
        db.read_geometry_arrays([revision], 'points')
    assert expected == struct.pack('<3i', 1, 2, 3)


@pytest.mark.parametrize('kind,detail', [
    ('fiber', dict(family='V', manual_family=True, control_point_count=2, line_point_count=3,
                  length=4.5, width=2.5, width_gap_fraction=.3, optimization_mode='fit', bbox=[0., 1., 2., 3., 4., 5.])),
    ('surface', dict(grid_width=3, grid_height=2, scale=(.5, .25), area=0., bbox=None)),
    ('track', dict(dataset_ref='dataset', external_key='track', family='H', point_count=3, length=4.5, bbox=None)),
])
def test_records_round_trip_through_transactions_and_reopening(db, kind, detail):
    with db.write() as tx:
        uid = tx.upsert_object(dict(kind=kind, name='sample', format_version=2,
            storage_kind='file', storage_ref='source', data_generation=3, data_checksum='checksum',
            tags=iter(['review']), meta={'annotation': None}, **{kind: detail}))
        tx.set_meta(uid, 'note', {'value': None})
    record = db.object(uid)
    assert record[kind] == detail and record['tags'] == ['review'] and record['meta']['note'] == {'value': None}
    fields = ['name', 'format_version', 'storage_kind', 'storage_ref', 'data_generation', 'data_checksum', kind]
    assert db.select_objects(fields=fields) == [{key: record[key] for key in ['uuid', 'kind', *fields]}]
    with pytest.raises(RuntimeError, match='finished'):
        tx.set_meta(uid, 'invalid', True)
    with pytest.raises((RuntimeError, ValueError)):
        with db.write() as tx:
            tx.upsert_object(dict(kind=kind, **{kind: {'bbox': [1, 2]}}))
    assert len(db.objects_by_kind(kind)) == 1
    path = db.path
    db.close()
    with ObjectStore.open(path, author='reopened') as reopened:
        assert reopened.object(uid) == record
        assert reopened.select_objects(object_kind=kind, fields=['name'])[0]['uuid'] == uid


def fiber_document(uid, ids, branches=()):
    return dict(type='vc3d_fiber', version=4, uuid=uid, name='same name', control_point_ids=ids,
                control_points=[{'position': [1, 2, 3]}, {'position': [4, 5, 6]}],
                line_points=[[1, 2, 3], [4, 5, 6]], branches=list(branches))


def test_fiber_branch_transfer_preserves_identity_and_stable_ids(db, tmp_path):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    edge_uuid = str(uuid.uuid4())
    # The branch joins control point ID 10 (index 1) of the first to ID 90 (index 0) of the second.
    branch = lambda target, own, other, position, direction: dict(
        edge_uuid=edge_uuid, target_uuid=target, control_point_index=own, branch_control_point_index=other,
        control_point_position=position, branch_control_point_position=position,
        control_point_direction=direction, branch_control_point_direction=direction)
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'first.json').write_text(json.dumps(fiber_document(ids[0], [90, 10], [branch(ids[1], 1, 0, [4, 5, 6], [1, 0, 0])])))
    (source / 'second.json').write_text(json.dumps(fiber_document(ids[1], [90, 10], [branch(ids[0], 0, 1, [1, 2, 3], [1, 0, 0])])))
    assert db.ingest_geometry_source('fiber', source) == dict(ingested=2, edges_imported=1, problems=[])
    directory = tmp_path / 'fibers'
    directory.mkdir()
    for uid in ids:
        assert export(db, uid, directory / (uid + '.json')).returncode == 0
    first = json.loads((directory / (ids[0] + '.json')).read_text())
    exported = first['branches'][0]
    assert exported['edge_uuid'] == edge_uuid and exported['target_uuid'] == ids[1]
    assert exported['control_point_index'] == 1 and exported['branch_control_point_index'] == 0
    assert first['control_point_ids'] == [90, 10]
    with ObjectStore.create(str(tmp_path / 'imported.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', directory) == dict(ingested=2, edges_imported=1, problems=[])
        assert other.neighbours(ids[0]) == [ids[1]]
    # Conflicting reciprocal declarations must roll back the whole batch.
    first['branches'][0]['branch_control_point_index'] = 1
    (directory / (ids[0] + '.json')).write_text(json.dumps(first))
    with ObjectStore.create(str(tmp_path / 'conflict.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', directory)['problems']
        assert other.objects_by_kind('fiber') == []


def test_fiber_collection_and_linked_file_import_use_uuid_endpoints(db, tmp_path):
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
        assert export(db, uid, output).returncode == 0
        documents.append(json.loads(output.read_text()))
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
        assert other.objects_by_kind('fiber') == []
    collection.write_text(json.dumps(documents))
    with ObjectStore.create(str(tmp_path / 'bare.vcobj'), author='test') as other:
        assert other.ingest_geometry_source('fiber', collection)['problems']
        assert other.objects_by_kind('fiber') == []


def test_version3_fiber_directory_import_resolves_links_by_file(tmp_path):
    ids = [str(uuid.uuid4()) for _ in range(3)]
    files = [f'anon_{i}.json' for i in range(3)]

    def branch(target, own, other, direction, branch_direction):
        return dict(branch_file=files[target], branch_fiber_id=target, control_point_index=own,
                    branch_control_point_index=other, control_point_position=[0, 0, 0],
                    branch_control_point_position=[0, 0, 0], control_point_direction=direction,
                    branch_control_point_direction=branch_direction)

    def write(directory, noise):
        directory.mkdir()
        documents = [dict(type='vc3d_fiber', version=3, uuid=uid, filename=name, sequence=i,
                          control_points=[{'position': [i, 0, 0]}, {'position': [i, 1, 0]}],
                          line_points=[[i, 0, 0], [i, 1, 0]], branches=[])
                     for i, (uid, name) in enumerate(zip(ids, files))]
        documents[0].update(vc_open_data_coordinate_space='Sample/20260411134726@L0',
                            vc_open_data_source_coordinate_scale_factor=1)
        # VC3D v3 wrote each link from both ends; directions differ by float noise.
        documents[0]['branches'].append(branch(1, 1, 0, [1, 0, 0], [0, 1, 0]))
        documents[1]['branches'].append(branch(0, 0, 1, [0, 1 + noise, 0], [1, 0, 0]))
        for i, document in enumerate(documents):
            (directory / files[i]).write_text(json.dumps(document))

    write(tmp_path / 'fibers', 1e-16)
    with ObjectStore.create(str(tmp_path / 'v3.vcobj'), author='test') as store:
        report = store.ingest_geometry_source(
            'fiber', tmp_path / 'fibers', coordinate_context=dict(volume_uuid='20260411134726', scale_factor=1))
        assert report == dict(ingested=3, edges_imported=1, problems=[])
        assert store.neighbours(ids[0]) == [ids[1]] and store.neighbours(ids[2]) == []
        for uid in ids:
            assert store.object(uid)['meta']['coordinates'] == dict(volume_uuid='20260411134726', scale_factor=1)
    # Reciprocal declarations that genuinely disagree still roll back the batch.
    write(tmp_path / 'conflict', 1e-3)
    with ObjectStore.create(str(tmp_path / 'conflict.vcobj'), author='test') as store:
        assert store.ingest_geometry_source('fiber', tmp_path / 'conflict')['problems']
        assert store.objects_by_kind('fiber') == []


def test_uuidless_fiber_files_are_named_from_their_file_names(tmp_path):
    files = ['kb_20260713T041915396_000154.json', 'lt_20260707T055926767_000404.json']
    (tmp_path / 'fibers').mkdir()
    for i, name in enumerate(files):
        document = dict(type='vc3d_fiber', version=3, filename=name, sequence=i,
                        control_points=[{'position': [i, 0, 0]}, {'position': [i, 1, 0]}],
                        line_points=[[i, 0, 0], [i, 1, 0]],
                        branches=[dict(branch_file=files[1 - i], branch_fiber_id=1 - i, control_point_index=i,
                                       branch_control_point_index=1 - i, control_point_position=[0, 0, 0],
                                       branch_control_point_position=[0, 0, 0],
                                       control_point_direction=[1, 0, 0] if i == 0 else [0, 1, 0],
                                       branch_control_point_direction=[0, 1, 0] if i == 0 else [1, 0, 0])])
        (tmp_path / 'fibers' / name).write_text(json.dumps(document))
    # Version 1 shares the folder: bare XYZ control points, no optimization_mode.
    version1 = 'kb_20260712T101010101_000003.json'
    (tmp_path / 'fibers' / version1).write_text(json.dumps(dict(
        type='vc3d_fiber', version=1, filename=version1, sequence=3, branches=[],
        control_points=[[5, 0, 0], [5, 1, 0]], line_points=[[5, 0, 0], [5, 1, 0]])))
    # RFC 4122 version 5 in the store's fixed namespace: every machine mints the same UUIDs.
    namespace = uuid.UUID('e14f6968-9145-494d-8b06-ab2f2fac7f3b')
    expected = [str(uuid.uuid5(namespace, 'vc3d_fiber/' + name)) for name in files]
    for store_name in ('a.vcobj', 'b.vcobj'):
        with ObjectStore.create(str(tmp_path / store_name), author='test') as store:
            assert store.ingest_geometry_source('fiber', tmp_path / 'fibers') == dict(
                ingested=3, edges_imported=1, problems=[])
            assert store.neighbours(expected[0]) == [expected[1]]
            assert store.object(str(uuid.uuid5(namespace, 'vc3d_fiber/' + version1)))['kind'] == 'fiber'
    # A single-file import resolves its UUID-less branch peer by file name too.
    with ObjectStore.create(str(tmp_path / 'single.vcobj'), author='test') as store:
        assert store.ingest_geometry_source('fiber', tmp_path / 'fibers' / files[0])['problems'] == []
        assert {store.object(uid)['uuid'] for uid in expected} == set(expected)


@pytest.mark.parametrize('kind,geometry', [
    ('fiber', Fiber([[1, 2, 3], [4, 5, 6]], [])),
    ('point_collection', PointCollection([42, 7], [[1, 2, 3], [4, 5, 6]])),
])
def test_native_export_pins_object_metadata_and_geometry(db, tmp_path, kind, geometry):
    uid, first_geometry = put(db, geometry, kind)
    db.set_meta(uid, 'coordinates', {'scale_factor': 2})
    before = db.current_object_revision(uid)['uuid']
    db.set_meta(uid, 'coordinates', {'scale_factor': 4})
    output = tmp_path / 'historical.json'
    exported = export(db, uid, output, revision=before)
    assert exported.returncode == 0 and first_geometry in exported.stdout
    doc = json.loads(output.read_text())
    if kind == 'point_collection':
        doc = next(iter(doc['collections'].values()))
    assert doc['coordinates']['scale_factor'] == 2
    wrong = export(db, uid, tmp_path / 'wrong.json', revision=first_geometry)
    assert wrong.returncode != 0 and 'object revision' in wrong.stderr


def test_service_creates_surfaces_in_a_hosted_replica_through_a_session(tmp_path):
    """Lasagna's service: attach a replica, keep artifacts beside the hub, create surfaces, sync."""
    import vc_object_store as vos
    from vc_object_store.surfaces import create_surface, read_surface
    hub = tmp_path / 'share' / 'hub.vcobj'
    hub.parent.mkdir()
    ObjectStore.create(str(hub), author='curator').close()
    vos.hub_init(str(hub), 'curator')
    for name in ('service', 'viewer'):
        vos.hub_clone(str(hub), str(tmp_path / f'{name}.vcobj'), name)
    session = vos.ObjectStoreSession()
    replica = tmp_path / 'service.vcobj'
    attachment = session.attach(str(replica), 'service', sync_interval_ms=100, push_delay_ms=20)
    xyz = np.stack([*np.mgrid[:4, :5][::-1], np.full((4, 5), 9)], axis=-1).astype('<f4')
    surface = create_surface(replica, xyz, name='fit', scale=1.0, volume='vol', tool='lasagna', author='service')
    model = Path(session.artifact_directory(attachment, surface))
    assert model == hub.parent / 'hub.vcobj.artifacts' / surface and model.is_dir()
    status = session.sync_hub_and_wait(attachment, timeout_s=20)
    assert not status['pending'] and not status['error'] and not status['conflicts']
    session.detach(attachment)
    with ObjectStore.open(str(tmp_path / 'viewer.vcobj'), author='viewer') as viewer:
        assert viewer.hub_sync()['pulled'] == 1
    np.testing.assert_array_equal(read_surface(tmp_path / 'viewer.vcobj', surface, volume='vol').points, xyz)


def test_numpy_geometry_and_coordinate_bindings(db):
    from vc_object_store import geometry_scale_to_volume
    points = np.array([[[1, 2, 3], [-1, -1, -1], [4, 5, 6]]], dtype='<f4')
    surface = Surface.from_arrays(points, [1, 1], np.array([[255, 255, 0]], dtype='u1'))
    assert surface.points_array().tolist() == points.tolist()
    assert surface.valid_mask().tolist() == [[True, False, False]]
    uid, _ = put(db, surface, 'surface')
    bound, _ = put(db, Umbilicus([[1, 2, 3], [4, 5, 6]]), 'umbilicus')
    db.set_meta(bound, 'coordinates', {'volume_uuid': 'vol', 'scale_factor': 4})
    fiber, _ = put(db, Fiber([[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]), 'fiber')
    with db.geometry_snapshot() as snapshot:
        read, valid = snapshot.surface(uid).read_arrays()
        assert read.dtype == np.float32 and read.tolist() == points.tolist()
        assert valid.dtype == bool and valid.tolist() == [[True, False, False]]
        assert snapshot.surface(uid).read_arrays(0, 1, 1, 2)[0].shape == (1, 2, 3)
        assert snapshot.umbilicus(bound).points_array().tolist() == [[1, 2, 3], [4, 5, 6]]
        assert snapshot.fiber(fiber).line_points_array(1).tolist() == [[0.5, 0, 0], [1, 0, 0]]
        assert snapshot.fiber(fiber).control_points_array().shape == (2, 3)
    assert geometry_scale_to_volume(db.object(bound)['meta'], 'vol') == 0.25
    assert geometry_scale_to_volume(db.object(bound)['meta'], 'other') is None
    assert db.unbound_object_count('umbilicus', 'vol') == 0
    assert db.unbound_object_count('surface', 'vol') == 1


def test_numpy_constructors_for_polylines(db):
    fiber = Fiber.from_arrays(np.array([[0, 0, 0], [2, 0, 0]], '<f8'), np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], '<f8'))
    uid, _ = put(db, fiber, 'fiber')
    with db.geometry_snapshot() as snapshot:
        view = snapshot.fiber(uid)
        assert view.control_point_ids(0, 2) == [0, 1] and view.line_points_array().tolist()[1] == [1, 0, 0]
    assert Umbilicus.from_arrays(np.array([[1, 2, 3]], '<f8')).points == [[1, 2, 3]]


def test_surface_export_keeps_validity_or_writes_valid_only(db, tmp_path):
    points = np.array([[[1, 2, 3], [4, 5, 6]]], dtype='<f4')
    valid = np.array([[255, 0]], dtype='u1')
    uid = db.upsert_object(dict(kind='surface'))
    db.put_geometry(uid, dict(kind='surface', attributes=dict(scale=[1, 1]), arrays=[
        array('points', points), array('valid', valid)]))
    assert export(db, uid, tmp_path / 'masked').returncode == 0
    assert export(db, uid, tmp_path / 'valid_only', valid_only=True).returncode == 0
    assert (tmp_path / 'masked' / 'mask.tif').exists() and not (tmp_path / 'valid_only' / 'mask.tif').exists()
    def reingest(name):
        other = ObjectStore.create(str(tmp_path / (name + '.vcobj')), author='t')
        assert not other.ingest_geometry_source('surface', tmp_path / name)['problems']
        return other, other.current_geometry(other.objects_by_kind('surface')[0]['uuid'])
    other, masked = reingest('masked')
    assert np.frombuffer(raw(other, masked['uuid'], 'valid'), 'u1').tolist() == [255, 0]
    other.close()
    other, only = reingest('valid_only')
    assert 'valid' not in {a['name'] for a in only['arrays']}
    stored = np.frombuffer(raw(other, only['uuid'], 'points'), '<f4').reshape(1, 2, 3)
    assert stored[0, 1].tolist() == [-1, -1, -1] and stored[0, 0].tolist() == [1, 2, 3]
    other.close()
