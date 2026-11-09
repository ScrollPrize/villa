"""The workflow's storage boundary is identical for plain/sharded/packed stores."""
import uuid
import json
from pathlib import Path
import numpy as np
import pytest

from vc_object_store import CollectionReader, fiber_graph_solve_table, write_extension_store
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    fiber_catalog, fiber_volume, fibers_digest, read_polylines, write_initial_fibers,
)


@pytest.mark.parametrize('workers,packed', [(1, False), (4, False), (4, True)])
def test_initial_collection_geometry_and_identity(tmp_path, workers, packed):
    volume = str(uuid.UUID(int=900))
    curves = [(dict(id=str(uuid.UUID(int=i+1)), family='H'),
               np.array([[i, 0., 0.], [i+1, 2., 3.]])) for i in range(12)]
    result = write_initial_fibers(tmp_path/'fibers.vcobj', curves, volume=volume,
                                  workers=workers, packed=packed)
    path = result['path']
    assert fiber_volume(path) == volume
    assert len(fiber_catalog(path, volume)) == 12
    loaded = read_polylines(path)
    for (row, actual), (expected, xyz) in zip(loaded, curves):
        assert row['id'] == expected['id']
        np.testing.assert_array_equal(actual, xyz)
    assert fibers_digest(path) == fibers_digest(path, [r['id'] for r, _ in curves])
    with CollectionReader(path) as store:
        assert all(r['fiber']['control_point_count'] == 0 for r in store.objects('fiber'))
        source = store.source
    table = fiber_graph_solve_table('metrics')
    table['based_on'] = [source]
    run = uuid.UUID(int=700).bytes
    row = [run, 0, 'all', 'count', None, 12, 'fibers']
    derived = tmp_path/'result.vcobj'
    write_extension_store(derived, [path], [table], {table['name']: [row]})
    with CollectionReader(derived) as store:
        assert len(store.objects('fiber')) == 12
        assert store.extension_rows(table['name']) == [row]
    assert fibers_digest(derived) == fibers_digest(path)
    alias=tmp_path/'alias.vcobj';alias.symlink_to(Path(path).resolve())
    assert fibers_digest(alias)==fibers_digest(path)


def test_failed_result_write_is_not_published(tmp_path):
    path = tmp_path/'failed.vcobj'
    table = fiber_graph_solve_table('metrics')
    with pytest.raises((ValueError, RuntimeError)):
        write_extension_store(path, [], [table], {table['name']: [[1]]})
    assert list(tmp_path.iterdir()) == []


def test_typed_graph_cache_and_source_endpoint_order(tmp_path):
    from .fiber_graph_fixture import measured_store,uid
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    path=measured_store(tmp_path,[(8,3,2,False,.25),(2,1,2,True,1.5)])
    graph=FiberGraph(path)
    assert graph.event_ids.tolist()==[2,8]
    assert graph.fiber_a.tolist()==[uid(1),uid(3)]
    assert graph.fiber_b.tolist()==[uid(2),uid(2)]
    np.testing.assert_array_equal(graph.point_a,np.zeros((2,3)))
    np.testing.assert_array_equal(graph.point_b,[[1,0,0],[1,0,0]])
    cached=FiberGraph.cached(path,tmp_path/'cache')
    for key,value in vars(graph).items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(getattr(cached,key),value)
            assert not getattr(cached,key).flags.writeable
    assert list(graph.batches(2,first=8,last=9))==[slice(1,2)]
    manifest=tmp_path/'cache'/'manifest.json'
    info=json.loads(manifest.read_text());info['format_version']=-1
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='cache format'):
        FiberGraph.cached(path,tmp_path/'cache')


def test_measurement_frame_and_registered_geometry(tmp_path):
    from vc_object_store import ObjectStore, Fiber, import_initial_records, fiber_constraint_extension
    from .fiber_graph_fixture import uid
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    path=tmp_path/'source.vcobj'
    source,target=uid(900),uid(901)
    xyz=np.array([[0.,0.,0.],[6.,8.,0.]])
    with ObjectStore.create(str(path),author='test',history_enabled=False,spatial_index=False) as store:
        for i in (1,2):
            store.upsert_object(dict(uuid=uid(i),kind='fiber',meta=dict(
                coordinates=dict(volume_uuid=source,scale_factor=2))))
            store.put_geometry(uid(i),Fiber([],xyz.tolist()))
        store.register_volume_transform(source,target,[[2,.5,0,10],[0,3,0,-5],[0,0,1,2]])
        store.seal()
    expected=np.array([[10.,-5.,2.],[18.,7.,2.]])
    for _,points in read_polylines(path,[uid(1),uid(2)],volume=target):
        np.testing.assert_array_equal(points,expected)
    with CollectionReader(path) as store:
        table=fiber_constraint_extension('',coordinate_space='sample/'+target,based_on=[store.source])
    fields=dict(edge_uuid=uuid.UUID(uid(100)).bytes,source_id=5,source_fiber_a=1,source_fiber_b=2,
                source_reversed=0,arc_a=0.,arc_b=0.,parallel_score=0.,perpendicular_score=1.,
                las_winding=.5,parallel_las_winding=.5,evidence_weight=1.,normal_a_valid=0,normal_b_valid=0)
    for endpoint in ('a','b'):
        fields.update({f'point_{endpoint}_{axis}':v for axis,v in zip('xyz',expected[0])})
    edge=dict(record=dict(uuid=uid(100),a=uid(1),b=uid(2),kind='perpendicular',
                         a_anchor=dict(position=xyz[0].tolist()),b_anchor=dict(position=xyz[0].tolist())),
              extensions=[dict(table=table['name'],values=[fields.get(c['name']) for c in table['columns']])])
    output=import_initial_records(tmp_path/'graph.vcobj',[],edges=[edge],extensions=[table],
                                  dependencies=[path],workers=2,packed=True)['path']
    graph=FiberGraph(output)
    assert graph.volume==target
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import workflow_volume
    assert workflow_volume(dict(fibers=str(path),database=output))==target
    with pytest.raises(ValueError,match='measurement frame'):
        workflow_volume(dict(fibers=str(path),database=output,working_volume=source))
    np.testing.assert_allclose(graph.lengths,np.sqrt(8**2+12**2))
    np.testing.assert_array_equal(graph.point_a,expected[:1])
    np.testing.assert_array_equal(graph.point_b,expected[:1])
    with CollectionReader(output) as store:
        np.testing.assert_array_equal(store.edge(uid(100))['a_anchor']['position'],xyz[0])


def test_result_stages_share_geometry_and_preserve_runs(tmp_path):
    from .fiber_graph_fixture import measured_store,uid
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import write_results,write_winding_result
    path=measured_store(tmp_path,[(1,1,2,False,.5)])
    first=tmp_path/'classify.vcobj';second=tmp_path/'solve.vcobj'
    run1=write_results(first,[path],'classification','adam',{'fibers':[
        dict(fiber_uuid=uid(1),family=0,p_h=.8,p_v=.1,p_broken=.1,enabled=1)]})
    run2=write_winding_result(second,first,[uid(1),uid(2)],[3,3],method='irls',continuous=[3.1,2.9])
    with CollectionReader(second) as store:
        rows=store.extension_rows('ext_fiber_graph_solve_fibers')
        assert len(rows)==3
        assert {uuid.UUID(bytes=r[1]).hex for r in rows}=={uuid.UUID(run1).hex,uuid.UUID(run2).hex}
        assert len(store.fiber_constraints()['edge_uuid'])==1
    assert fibers_digest(path)==fibers_digest(second)
    with pytest.raises(ValueError,match='probability'):
        write_results(tmp_path/'invalid.vcobj',[path],'classification','adam',
                      {'fibers':[dict(fiber_uuid=uid(1),p_h=1.1)]})
    assert not (tmp_path/'invalid.vcobj').exists()
