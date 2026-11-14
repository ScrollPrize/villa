import msgpack
import json
import uuid
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import pytest
from numcodecs import Zstd
from vesuvius.neural_tracing.fiber_trace_3d.fiber_stream_import import tile_rows
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines
from .fiber_graph_fixture import uid


def test_tile_preserves_whole_geometry_and_ids(tmp_path):
    xyz=[[1.,2.,0.],[1.,2.,40.]]
    piece=dict(id=0,points_base=xyz,links=[-1,-1],reasons=['x','y'],seed_base=xyz[0],seed_direction=[0,0,1],fiber_attributes={})
    tile=dict(version=2,attribute_schema_version=1,coordinate_units='base_voxels',pieces=[piece])
    path=tmp_path/'0_0_0.msgpack.zst';path.write_bytes(Zstd().encode(msgpack.packb(tile)))
    ids,rows=tile_rows((path,[10,20]));assert ids==[0] and rows[0][0]['id']==0
    assert rows[0][0]['family']=='unknown'
    np.testing.assert_array_equal(rows[0][1],xyz)
    assert tile_rows((path,[50,60]))==([0],[])
    assert tile_rows((path,None))[0]==[0]
    np.testing.assert_array_equal(tile_rows((path,None))[1][0][1],xyz)
    tile['coordinate_units']='trace_voxels';path.write_bytes(Zstd().encode(msgpack.packb(tile)))
    with pytest.raises(ValueError,match='Unsupported'):tile_rows((path,[10,20]))


def test_stream_import_stable_uuid_and_attributes_without_paths(tmp_path,monkeypatch):
    from vesuvius.neural_tracing.fiber_trace_3d import fiber_stream_import as workflow
    # A source-tree test build needs the same native-module bootstrap in spawned workers.
    # Installed packages need no initializer and keep the production default.
    import os
    if os.environ.get('VC_READER_BUILD_DIR'):
        from build_support import load_build
        monkeypatch.setattr(workflow,'ProcessPoolExecutor',partial(ProcessPoolExecutor,initializer=load_build))
    source=tmp_path/'tiles';source.mkdir()
    fid=2**53+7
    xyz=[[1.,2.,0.],[1.,2.,40.]]
    attributes=dict(width_mean=dict(values=105.,units='base_voxel'))
    piece=dict(id=fid,points_base=xyz,links=[-1,-1],reasons=['x','y'],seed_base=xyz[0],
               seed_direction=[0,0,1],fiber_attributes=attributes)
    (source/'0_0_0.msgpack.zst').write_bytes(Zstd().encode(msgpack.packb(dict(
        version=2,attribute_schema_version=1,coordinate_units='base_voxels',pieces=[piece]))))
    manifest=dict(identity='test-stream',output_schema='fiber-stream-attributes-v2',coordinates='base voxels',
                  config=dict(stream=dict(ct_locator='test:volume')))
    (tmp_path/'run.json').write_text(json.dumps(manifest))
    (tmp_path/'summary.json').write_text(json.dumps(dict(complete=True,finished_tiles=1,total_tiles=1,pieces=1)))
    monkeypatch.setattr(workflow,'open_data_volume',lambda locator:('sample',uid(999)))
    expected=str(uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_URL,'test-stream'),'fiber:'+str(fid)))
    for index,slab in enumerate(([10,20],None)):
        root=tmp_path/f'output_{index}'
        result=workflow.run(dict(tiles=str(source),output_root=str(root),volume=uid(999),z_range=slab,workers=2))
        assert result['fibers']==1 and result['source_fibers']==1 and result['whole_fibers']
        rows=read_polylines(root/'fibers.vcobj')
        assert rows[0][0]['id']==expected and rows[0][0]['family']=='unknown'
        assert rows[0][0]['meta']['fiber_attributes']==attributes
        assert str(tmp_path) not in json.dumps(rows[0][0]['meta'])
        np.testing.assert_array_equal(rows[0][1],xyz)
