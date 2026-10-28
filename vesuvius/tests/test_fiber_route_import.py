import msgpack
import numpy as np
import pytest
from numcodecs import Zstd
from vesuvius.neural_tracing.fiber_trace_3d.fiber_route_import import route_rows


def test_whole_route_filter_and_native_coordinates(tmp_path):
    p=tmp_path/'page'
    xyz=[[2.,3.,29000.],[2.,3.,31000.]]
    p.write_bytes(Zstd().encode(msgpack.packb(dict(generation='g',component=4,page=0,
        fibers=[dict(points_base=xyz,length_base=2000.)]))))
    rows,bounds=route_rows(p,'g',[30000,50000])
    assert len(rows)==1
    np.testing.assert_array_equal(rows[0][1],xyz)
    assert bounds==[(29000.,31000.)]
    assert route_rows(p,'g',[40000,50000])[0]==[]
    with pytest.raises(ValueError,match='generation'):
        route_rows(p,'bad',[30000,50000])
