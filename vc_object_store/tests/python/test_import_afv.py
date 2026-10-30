"""scripts/import_afv.py streams an Automated Fiber Volume into a new store."""
import importlib.util
import json
from pathlib import Path
import sqlite3
import struct

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
            assert raw(store, revision, 'control_points') == expected[[0, -1]].tobytes()
            assert raw(store, revision, 'control_point_ids') == struct.pack('<2q', 0, 1)
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
