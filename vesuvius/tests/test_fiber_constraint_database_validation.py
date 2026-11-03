import json
import sqlite3
import numpy as np
import pytest
from vesuvius.neural_tracing.fiber_trace_3d.fiber_constraint_database_validation import validate
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import write_fibers


def test_validation_reads_store_geometry(tmp_path):
    ids=['00000000-0000-4000-8000-000000000001','00000000-0000-4000-8000-000000000002','00000000-0000-4000-8000-000000000003']
    curves=[np.array([[0.,0,k],[10.,0,k],[10.,10,k]]) for k in range(3)]
    store=tmp_path/'fibers.vcobj';write_fibers(store,[(dict(id=f),xyz) for f,xyz in zip(ids,curves)],volume='123')
    header=tmp_path/'constraint.hpp'
    header.write_text('struct FiberTraceConstraint {\n    bool localEvent=false;\n    bool hardContinuity=false;\n'
        '    std::optional<double> signedWindingDelta;\n    double arcABaseVoxels=0;\n    double arcBBaseVoxels=0;\n'
        '    cv::Vec3d pointABaseXYZ;\n    cv::Vec3d pointBBaseXYZ;\n    double parallelScore=0;\n'
        '    double perpendicularScore=0;\n    double windingDistance=0;\n    double parallelWindingDistance=0;\n'
        '    double evidenceWeight=0;\n};\n')
    path=tmp_path/'constraints.sqlite';db=sqlite3.connect(path)
    db.executescript('CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);'
        'CREATE TABLE fibers(id TEXT PRIMARY KEY,source_index INTEGER UNIQUE,length_base REAL);'
        'CREATE TABLE constraints(id INTEGER PRIMARY KEY,fiber_a TEXT REFERENCES fibers(id),fiber_b TEXT REFERENCES fibers(id),'
        'arc_a,arc_b,ax,ay,az,bx,by,bz,parallel_score,perpendicular_score,winding_distance,parallel_winding_distance,'
        'evidence_weight,normal_a,normal_b,measurement);')
    meta=dict(complete=True,format='fiber-constraints',statistics=dict(constraints=1,fibers=3,isolated_fibers=1),
              provenance=dict(config=dict(full_volume=True,coordinate_space='scroll/123')))
    db.executemany('INSERT INTO metadata VALUES(?,?)',[(k,json.dumps(v)) for k,v in meta.items()])
    db.executemany('INSERT INTO fibers VALUES(?,?,?)',[(f,k,20.) for k,f in enumerate(ids)])
    measurement=dict(localEvent=True,hardContinuity=False,signedWindingDelta=None,arcABaseVoxels=15.,arcBBaseVoxels=5.,
        pointABaseXYZ=[10.,5.,0.],pointBBaseXYZ=[5.,0.,1.],parallelScore=1.,perpendicularScore=0.,windingDistance=1.,
        parallelWindingDistance=1.,evidenceWeight=1.,supportAEnd=0,supportBEnd=0)
    normal=json.dumps(dict(valid=True,xyz=[0.,0.,1.]))
    db.execute('INSERT INTO constraints VALUES('+','.join('?'*19)+')',(0,ids[0],ids[1],15.,5.,10.,5.,0.,5.,0.,1.,1.,0.,1.,1.,1.,normal,normal,json.dumps(measurement)))
    db.commit();db.close()
    result=validate(path,store,header=header)
    assert result['fibers']==3 and result['constraints']==1 and result['geometry_endpoints_checked']==2
    db=sqlite3.connect(path);db.execute('UPDATE constraints SET bx=6');db.commit();db.close()
    with pytest.raises(AssertionError):validate(path,store,header=header)
