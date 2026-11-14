import json
import sqlite3
import uuid
import numpy as np
from vesuvius.neural_tracing.fiber_trace_3d.fiber_afv_transform import mapped_bounds, run
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_initial_fibers, read_polylines
from .fiber_graph_fixture import uid


def afv_source(path, fid, xyz):
    """Real legacy input, matching the native AfvSource fixture schema."""
    metadata = dict(uuid=uid(123), complete=True, fiber_count=1, point_count=len(xyz),
                    frame=dict(vc_open_data_coordinate_space='old'),
                    root=dict(coordinate_order='XYZ', coordinate_units='voxel'))
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    with sqlite3.connect(path) as db:
        db.executescript('''PRAGMA application_id=1447249475; PRAGMA user_version=1;
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE fibers(id INTEGER PRIMARY KEY,name TEXT,family TEXT,point_count INTEGER,length REAL,
                min_x REAL,max_x REAL,min_y REAL,max_y REAL,min_z REAL,max_z REAL,annotation TEXT);
            CREATE TABLE blocks(id INTEGER PRIMARY KEY,fiber_id INTEGER,first_segment INTEGER,points BLOB,
                UNIQUE(fiber_id,first_segment));''')
        db.executemany('INSERT INTO metadata VALUES(?,?)', [(k,json.dumps(v)) for k,v in metadata.items()])
        db.execute('INSERT INTO fibers VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (fid,'seven','H',len(xyz),float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
             lo[0],hi[0],lo[1],hi[1],lo[2],hi[2],'{}'))
        db.execute('INSERT INTO blocks(fiber_id,first_segment,points) VALUES(?,?,?)',
                   (fid,0,np.asarray(xyz,dtype='<f8').tobytes()))
    return metadata


def test_inverse_registration_keeps_whole_fibers(tmp_path):
    src=tmp_path/'source.afv'; target=tmp_path/'target.vcobj'
    source_id=2**53+7  # Preserve full SQLite integer IDs through bbox filtering.
    xyz=np.array([[1.,2.,3.],[1.,2.,9.]])
    metadata=afv_source(src,source_id,xyz)
    write_initial_fibers(target,[(dict(id=uid(3)),xyz)],volume='new',workers=1,packed=True)
    transform=dict(fixed_volume='old_volume',transformation_matrix=[[2,0,0,0],[0,2,0,0],[0,0,2,1]])
    path=tmp_path/'transform.json';path.write_text(json.dumps(transform))
    run(dict(afv=str(src),frame_template=str(target),transform=str(path),invert=True,
        fixed_volume='old_volume',source_coordinate_space='old',target_coordinate_space='new',
        output_root=str(tmp_path/'output'),z_range=[2,3],workers=2))
    rows=read_polylines(tmp_path/'output/fibers.vcobj')
    expected=str(uuid.uuid5(uuid.UUID(metadata['uuid']),'fiber:'+str(source_id)))
    assert len(rows)==1 and rows[0][0]['id']==expected
    assert rows[0][0]['family']=='H'
    np.testing.assert_array_equal(rows[0][1],[[.5,1,1],[.5,1,4]])


def test_bounds_reflection():
    m=np.array([[-2,1,0,4],[0,1,0,0],[0,0,3,1],[0,0,0,1.]])
    lo,hi=mapped_bounds(np.array([[0,0,0]]),np.array([[2,3,4]]),m)
    np.testing.assert_array_equal(lo,[[0,0,1]])
    np.testing.assert_array_equal(hi,[[7,3,13]])
