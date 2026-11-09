"""scripts/import_afv.py streams an Automated Fiber Volume into a new store."""
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest
from vc_object_store import ObjectStore
from object_helpers import raw

spec = importlib.util.spec_from_file_location('import_afv', Path(__file__).parents[2] / 'scripts/import_afv.py')
import_afv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(import_afv)


def fixture(path):
    points = np.arange(780, dtype='<f8').reshape(-1, 3) / 8
    points[0, 0] = -0.0
    metadata = dict(uuid='1482bcc8-f93e-4c3b-b380-2ddf369318f4', complete=True,
        fiber_count=2, point_count=262, frame={},
        root=dict(coordinate_order='XYZ', coordinate_units='voxel'), provenance={'test': True})
    with sqlite3.connect(path) as db:
        db.executescript('''PRAGMA application_id=1447249475; PRAGMA user_version=1;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,point_count INTEGER,length REAL,
                min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER,first_segment INTEGER,points BLOB,
                UNIQUE(fiber_id,first_segment));''')
        db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
        for fid, xyz in ((0, points), (9, points[-2:])):
            lo, hi = xyz.min(axis=0), xyz.max(axis=0)
            db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                (fid, 'same name', 'H' if fid == 0 else 'V', len(xyz), 7.25,
                 lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], json.dumps({'fiber_follow': {'links': [-1, 0]}})))
            for start in range(0, len(xyz) - 1, 256):
                db.execute('INSERT INTO blocks(fiber_id,first_segment,points) VALUES(?,?,?)',
                           (fid, start, xyz[start:start + 257].tobytes()))
    return metadata, points


def test_import_keeps_exact_bytes_metadata_and_refuses_existing_output(tmp_path):
    source = tmp_path / 'input.afv'
    metadata, points = fixture(source)
    original = source.read_bytes()
    output = tmp_path / 'objects.vcobj'
    result = import_afv.import_afv(source, output, volume_id='20260411134726', scale_factor=4,
                                   batch_size=1, progress=False)
    assert result['fibers'] == 2 and result['points'] == 262
    assert source.read_bytes() == original
    with ObjectStore.open(str(output), read_only=True) as store:
        objects = store.objects_by_kind('fiber')
        assert len(objects) == 2 and objects[0]['uuid'] != objects[1]['uuid']
        for record in objects:
            fid = record['meta']['source']['afv']['fiber_id']
            expected = points if fid == 0 else points[-2:]
            revision = store.current_geometry(record['uuid'])['uuid']
            assert raw(store, revision, 'line_points') == expected.tobytes()
            assert raw(store, revision, 'control_points') == b''
            assert raw(store, revision, 'control_point_ids') == b''
            assert record['fiber']['control_point_count'] == 0
            assert record['meta']['coordinates'] == dict(volume_uuid='20260411134726', scale_factor=4)
            assert record['fiber']['family'] == ('H' if fid == 0 else 'V')
            assert record['meta']['source']['afv']['annotation']['fiber_follow']['links'] == [-1, 0]
        collection = store.object(metadata['uuid'])
        assert collection['meta']['afv']['metadata'] == metadata
        assert collection['meta']['import'] == dict(fibers=2, points=262)
    with pytest.raises(FileExistsError):
        import_afv.import_afv(source, output, unbound=True, progress=False)


@pytest.mark.parametrize('mutation', [
    'PRAGMA user_version=2',
    "UPDATE metadata SET value='false' WHERE key='complete'",
    "UPDATE metadata SET value='263' WHERE key='point_count'",
    'UPDATE blocks SET first_segment=255 WHERE first_segment=256',
    'DELETE FROM blocks WHERE fiber_id=9',
    'UPDATE blocks SET points=zeroblob(25) WHERE id=1',
    'UPDATE blocks SET points=zeroblob(length(points)) WHERE first_segment=256',
    'UPDATE fibers SET point_count=261 WHERE id=0',
    'UPDATE fibers SET max_x=max_x+1 WHERE id=0',
    "UPDATE metadata SET value=json_set(value, '$.scale', 2) WHERE key='root'",
])
def test_malformed_input_publishes_nothing(tmp_path, mutation):
    source = tmp_path / 'input.afv'
    fixture(source)
    with sqlite3.connect(source) as db:
        db.execute(mutation)
    target = tmp_path / 'failed.vcobj'
    with pytest.raises(ValueError):
        import_afv.import_afv(source, target, unbound=True, batch_size=1, progress=False)
    assert not target.exists(), 'Failure must not publish even previously committed batches'
    assert not list(tmp_path.glob('.' + target.name + '-*')), 'Failure must remove staging files'


@pytest.mark.parametrize('shards', [1, 2])
@pytest.mark.parametrize('verify', [None, True])
def test_native_batch_command_readback_is_opt_in(tmp_path, shards, verify):
    build = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    tool = Path(build).resolve().parent / 'bin' / 'vc_object_import_afv' if build else None
    if not tool or not tool.exists():
        pytest.skip('needs the built vc_object_import_afv')
    source = tmp_path / 'input.afv'
    fixture(source)
    output = tmp_path / 'native.vcobj'
    report = tmp_path / 'report.json'
    config = dict(source=str(source), output=str(output), unbound=True,
                  shards=shards, report=str(report))
    if verify is not None:
        config['verify_readback'] = verify
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    completed = subprocess.run([str(tool), str(path)], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(report.read_text())
    assert result['fibers'] == 2 and result['points'] == 262
    assert result['readback_verified'] is bool(verify)
    assert (result['validation_seconds'] > 0) is bool(verify)
    assert output.exists()

    config.update(initial_batch=False, output=str(tmp_path / 'legacy'))
    path.write_text(json.dumps(config))
    rejected = subprocess.run([str(tool), str(path)], capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0
    assert 'initial_batch:false is removed' in rejected.stderr
    assert not Path(config['output']).exists()


def constraint_fixture(path, source_uuid):
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE constraints(id INTEGER PRIMARY KEY,fiber_a INTEGER,fiber_b INTEGER,
                arc_a REAL,arc_b REAL,ax REAL,ay REAL,az REAL,bx REAL,by REAL,bz REAL,
                parallel_score REAL,perpendicular_score REAL,winding_distance REAL,
                parallel_winding_distance REAL,evidence_weight REAL,normal_a TEXT,normal_b TEXT,measurement TEXT);''')
        metadata = dict(format='fiber-constraints', schema_version=1, complete=True,
                        provenance=dict(afv_uuid=source_uuid), statistics=dict(constraints=17))
        db.executemany('INSERT INTO metadata VALUES(?,?)', [(k, json.dumps(v)) for k, v in metadata.items()])
        for i in range(17):
            db.execute('INSERT INTO constraints VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                       (i * 3, 0, 9, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, .5, .5, 1, None, None, '{}'))


@pytest.mark.parametrize('packed', [False, True])
def test_native_uuid_dispatch_combined_and_linked(tmp_path, packed):
    build = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    tool = Path(build).resolve().parent / 'bin' / 'vc_object_import_afv' if build else None
    if not tool or not tool.exists():
        pytest.skip('needs native importer')
    source, constraints = tmp_path / 'source.afv', tmp_path / 'source.afc'
    metadata, _ = fixture(source)
    constraint_fixture(constraints, metadata['uuid'])
    config = dict(source=str(source), constraints=str(constraints), unbound=True,
                  shards=4, readers=2, batch_size=1, batch_bytes=256, queue_bytes=128,
                  packed=packed, verify_readback=True, memory_first=True, resident_bytes=128 * 1024**2)
    path = tmp_path / 'config.json'
    for mode in ('combined', 'geometry', 'linked', 'limited', 'failure'):
        output, report = tmp_path / mode, tmp_path / (mode + '.json')
        current = dict(config, output=str(output), report=str(report))
        if mode == 'geometry':
            del current['constraints']
        elif mode == 'linked':
            current['geometry_collection'] = str(tmp_path / 'geometry' if packed else tmp_path / 'geometry' / 'collection.vcobj')
        elif mode == 'limited':
            current['constraint_limit'] = 5
        elif mode == 'failure':
            with sqlite3.connect(source) as db:
                db.execute('UPDATE blocks SET points=zeroblob(25) WHERE fiber_id=9')
        path.write_text(json.dumps(current))
        result = subprocess.run([str(tool), str(path)], text=True, capture_output=True, timeout=30)
        if mode == 'failure':
            assert result.returncode != 0
            assert not output.exists()
            assert not list(tmp_path.glob('.failure-*'))
            continue
        assert result.returncode == 0, result.stderr
        data = json.loads(report.read_text())
        assert data['uuid_hash_routing'] and data['readback_verified']
        assert data['fibers'] == (0 if mode == 'linked' else 2)
        assert data['constraints'] == (0 if mode == 'geometry' else 5 if mode == 'limited' else 17)
        if mode == 'limited':
            assert data['estimated_full_import_seconds'] is None
            assert 'sampling fractions differ' in data['estimate_basis']
        else:
            assert data['estimated_full_import_seconds'] >= 0
        if not packed:
            seen = 0
            for shard in output.glob('part-*.vcobj'):
                with sqlite3.connect(shard) as db:
                    for (uuid,) in db.execute('SELECT uuid FROM objects'):
                        h = 14695981039346656037
                        for byte in uuid:
                            h = ((h ^ byte) * 1099511628211) & ((1 << 64) - 1)
                        assert h % 4 == int(shard.stem.split('-')[-1])
                        seen += 1
            assert seen == (1 if mode == 'linked' else 3)


@pytest.mark.parametrize('key,value', [
    ('queue_bytes', -1), ('queue_bytes', 0), ('queue_bytes', .5),
    ('queue_bytes', 2**64 - 1), ('shards', 2**32 + 1),
    ('readers', 2**32 + 1), ('constraint_limit', .5),
])
def test_native_queue_limits_reject_invalid_integers(tmp_path, key, value):
    build = os.environ.get('VC_OBJECT_STORE_BUILD_PYTHON_PATH')
    tool = Path(build).resolve().parent / 'bin' / 'vc_object_import_afv' if build else None
    if not tool or not tool.exists():
        pytest.skip('needs native importer')
    source, output = tmp_path / 'source.afv', tmp_path / 'output'
    fixture(source)
    config = dict(source=str(source), output=str(output), unbound=True)
    config[key] = value
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    result = subprocess.run([str(tool), str(path)], text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert key in result.stderr
    assert not output.exists()
