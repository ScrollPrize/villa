import msgpack
import numpy as np
import pytest
from numcodecs import Zstd
from vesuvius.neural_tracing.fiber_trace_3d.fiber_stream_import import tile_rows


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
