import msgpack
import numpy as np
import pytest
import uuid
from numcodecs import Zstd
from vesuvius.neural_tracing.fiber_trace_3d.fiber_route_import import route_rows
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import read_polylines, fiber_volume
from .fiber_graph_fixture import uid


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


def test_route_import_preserves_native_geometry_and_crop_independent_uuid(tmp_path):
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_route_import import run
    source=tmp_path/'routes';source.mkdir()
    (source/'routes.manifest.msgpack.zst').write_bytes(Zstd().encode(msgpack.packb(
        dict(format='vc-fiber-routes-v4-reduced',generation='test-generation'))))
    xyz=[[2.,3.,29000.],[2.,3.,31000.]]
    (source/'component_4.routes.0.msgpack.zst').write_bytes(Zstd().encode(msgpack.packb(
        dict(generation='test-generation',component=4,page=0,fibers=[dict(points_base=xyz,length_base=2000.)]))))
    expected=str(uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_URL,'test-generation'),'component_4_page_0_route_0'))
    for index,slab in enumerate(([30000,50000],[29000,30000])):
        root=tmp_path/f'output_{index}'
        result=run(dict(routes=str(source),output_root=str(root),volume=uid(999),z_range=slab,workers=2))
        assert result['fibers']==1 and result['source_routes']==1
        rows=read_polylines(root/'fibers.vcobj')
        assert rows[0][0]['id']==expected and rows[0][0]['family']=='unknown'
        assert fiber_volume(root/'fibers.vcobj')==uid(999)
        np.testing.assert_array_equal(rows[0][1],xyz)
