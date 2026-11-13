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


def test_implicit_volume_catalog_loads_records_once_and_matches_explicit(monkeypatch):
    from vesuvius.neural_tracing.tracers.shared.annotations import fiber_store
    volume=str(uuid.UUID(int=900));calls=[]
    records=[dict(uuid=str(uuid.UUID(int=1)),name='fiber',
                  meta=dict(coordinates=dict(volume_uuid=volume,scale_factor=2.)),
                  fiber=dict(family='H',length=10.,line_point_count=3,bbox=[0.,0.,0.,4.,6.,8.]))]
    class Reader:
        def __init__(self,path):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def select_objects(self,**query):calls.append(query['object_kind']);return records
    monkeypatch.setattr(fiber_store,'open_store',Reader)
    monkeypatch.setattr(fiber_store,'_read_rows',lambda store,selected,volume,**kwargs:
                        iter([(dict(length=5.,point_count=3),np.zeros((3,3)))]))
    expected=fiber_store.fiber_catalog('source',volume)
    assert calls==['fiber'];calls.clear()
    assert fiber_store.fiber_catalog('source')==expected
    assert calls==['fiber'];calls.clear()
    assert fiber_store.fiber_volume('source')==volume
    assert calls==['fiber']
    assert expected[0]['length']==5. and expected[0]['point_count']==3


@pytest.mark.parametrize('resolver',[fiber_catalog,fiber_volume])
def test_implicit_volume_rejects_mixed_source_frames_once(monkeypatch,resolver):
    from vesuvius.neural_tracing.tracers.shared.annotations import fiber_store
    calls=[]
    class Reader:
        def __init__(self,path):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def select_objects(self,**query):
            calls.append(query['object_kind'])
            return [dict(meta=dict(coordinates=dict(volume_uuid=str(uuid.UUID(int=i))))) for i in (1,2)]
    monkeypatch.setattr(fiber_store,'open_store',Reader)
    with pytest.raises(ValueError,match='must be bound to one volume'):resolver('source')
    assert calls==['fiber']


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


@pytest.mark.parametrize('workers,packed', [(1,False),(2,False),(2,True)])
def test_source_position_resolves_public_collection_root(tmp_path,workers,packed):
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_source_version
    curves=[(dict(id=str(uuid.UUID(int=i+1))),np.array([[0.,0.,0.],[10.,0.,0.]])) for i in range(2)]
    source=write_initial_fibers(tmp_path/'source',curves,volume='123',workers=workers,packed=packed)['path']
    pin=fiber_source_version(source)
    with CollectionReader(source) as reader:
        assert Path(reader.resolve_store_position(pin))==Path(source).resolve()
        assert reader.resolve_store_position(reader.source)==reader.resolve_store_position(pin)
        for changed in (dict(pin,store=str(uuid.uuid4())),dict(pin,generation=str(uuid.uuid4())),
                        dict(pin,commit_sequence=pin['commit_sequence']+1),dict(pin,extra=True)):
            with pytest.raises((ValueError,RuntimeError)):
                reader.resolve_store_position(changed)
    with pytest.raises(RuntimeError,match='closed'):
        reader.resolve_store_position(pin)


def test_typed_graph_cache_and_source_endpoint_order(tmp_path):
    from .fiber_graph_fixture import measured_store,uid
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import FiberGraph
    path=measured_store(tmp_path,[(8,3,2,False,.25),(2,1,2,True,1.5)])
    graph=FiberGraph(path)
    assert graph.event_ids.tolist()==[0,1]
    assert graph.edge_uuid.tolist()==[uid(1002),uid(1008)]
    assert graph.fiber_a.tolist()==[uid(1),uid(2)]
    assert graph.fiber_b.tolist()==[uid(2),uid(3)]
    np.testing.assert_array_equal(graph.connector,[[1,0,0],[1,0,0]])
    for name in ('point_a','point_b'):
        with pytest.raises(ValueError,match='do not store endpoint XYZ'):
            getattr(graph,name)
    cached=FiberGraph.cached(path,tmp_path/'cache')
    for key,value in vars(graph).items():
        if isinstance(value,np.ndarray):
            np.testing.assert_array_equal(getattr(cached,key),value)
            assert not getattr(cached,key).flags.writeable
    assert list(graph.batches(2,first=1,last=2))==[slice(1,2)]
    manifest=tmp_path/'cache'/'manifest.json'
    info=json.loads(manifest.read_text());info['format_version']=-1
    manifest.write_text(json.dumps(info))
    with pytest.raises(ValueError,match='cache format'):
        FiberGraph.cached(path,tmp_path/'cache')


def test_measurement_frame_and_registered_geometry(tmp_path):
    from vc_object_store import (ObjectStore, Fiber, import_initial_records, fiber_constraint_extension,
                                 encode_fiber_arc,encode_fiber_direction,encode_fiber_winding)
    from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_source_version
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
    table['provisioning']['fiber_source']=fiber_source_version(path)
    fields=dict(edge_uuid=uuid.UUID(uid(100)).bytes,arc_a=encode_fiber_arc(0),arc_b=encode_fiber_arc(0),
                normal_a=encode_fiber_direction([0,0,0]),normal_b=encode_fiber_direction([0,0,0]),
                connector=encode_fiber_direction([0,0,0]),winding_distance=encode_fiber_winding(.5),extra=None)
    edge=dict(record=dict(uuid=uid(100),a=uid(1),b=uid(2),kind='perpendicular',
                         a_anchor={},b_anchor={}),
              extensions=[dict(table=table['name'],values=[fields.get(c['name']) for c in table['columns']])])
    output=import_initial_records(tmp_path/'graph.vcobj',[],edges=[edge],extensions=[table],
                                  dependencies=[path],workers=2,packed=True,
                                  edge_columns=[],edge_adjacency_index=False)['path']
    graph=FiberGraph(output)
    assert graph.volume==target
    from vesuvius.neural_tracing.fiber_trace_3d.fiber_graph_store import workflow_volume
    assert workflow_volume(dict(fibers=str(path),database=output))==target
    with pytest.raises(ValueError,match='measurement frame'):
        workflow_volume(dict(fibers=str(path),database=output,working_volume=source))
    np.testing.assert_allclose(graph.lengths,np.sqrt(8**2+12**2))
    for name in ('point_a','point_b'):
        with pytest.raises(ValueError,match='do not store endpoint XYZ'):
            getattr(graph,name)
    with CollectionReader(output) as store:
        assert store.edge(uid(100))['a_anchor'].get('position') is None


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
