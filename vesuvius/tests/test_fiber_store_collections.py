"""The workflow's storage boundary is identical for plain/sharded/packed stores."""
import uuid
import json
from pathlib import Path
import numpy as np
import pytest

from vc_object_store import CollectionReader, fiber_graph_solve_table, write_extension_store
from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    fiber_catalog, fiber_volume, fibers_digest, read_polylines, read_polyline_arrays, write_initial_fibers,
)


@pytest.mark.parametrize('workers,packed', [(1,False),(3,True)])
def test_regular_history_store_is_a_direct_bulk_import_dependency(tmp_path,workers,packed):
    from vc_object_store import ObjectStore,Fiber,import_initial_records
    source=tmp_path/'neural-fibers.vcobj'
    volume=str(uuid.UUID(int=900));key=str(uuid.UUID(int=1))
    xyz=np.array([[1.,2.,3.],[11.,2.,3.]],dtype='<f8')
    with ObjectStore.create(str(source),author='neural-tracer',spatial_index=False) as writer:
        writer.upsert_object(dict(uuid=key,kind='fiber',format_version=4,
            meta=dict(coordinates=dict(volume_uuid=volume,scale_factor=1.))))
        writer.put_geometry(key,Fiber([],xyz.tolist()))
        before=writer.provisioning()
        # Checked bulk reading of the regular source itself is also allowed;
        # every parallel worker must match the captured source publication.
        direct_ids,direct=read_polyline_arrays(source,[key],volume=volume,workers=3,trusted_immutable='auto')
        assert direct_ids.tolist()==[key]
        np.testing.assert_array_equal(direct.values,xyz)
        # The same native initial-construction API used by extraction accepts
        # this source directly: no conversion, source sealing, or geometry copy.
        result=import_initial_records(tmp_path/'constraints.vcobj',[],dependencies=[source],
                                      workers=workers,packed=packed)
        assert writer.provisioning()==before and not before['sealed']
        assert before['history_enabled']
        with CollectionReader(result['path']) as reader:
            assert not reader.supports_trusted_geometry_reads
            assert reader.object(key)['uuid']==key
        selected,geometry=read_polyline_arrays(result['path'],[key],volume=volume,
                                               workers=3,trusted_immutable='auto')
        assert selected.tolist()==[key]
        np.testing.assert_array_equal(geometry.values,xyz)
        derived=tmp_path/'classified.vcobj'
        write_extension_store(derived,[result['path']],[],{})
        with CollectionReader(derived) as reader:
            assert reader.object(key)['uuid']==key


def test_changed_history_dependency_reads_its_saved_publication(tmp_path):
    from vc_object_store import ObjectStore,StoreBusy
    source=tmp_path/'source.vcobj';derived=tmp_path/'result.vcobj'
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='note',meta=dict(label='original')))
        write_extension_store(derived,[source],[],{})
        with CollectionReader(derived) as loaded,CollectionReader(derived) as not_yet_read:
            assert loaded.object(key)['meta']['label']=='original'
            writer.set_meta(key,'label','changed')
            # An already acquired SQLite snapshot remains coherent. A newly
            # acquired dependency transaction must validate the stored pin.
            assert loaded.object(key)['meta']['label']=='original'
            assert not_yet_read.object(key)['meta']['label']=='original'
        with CollectionReader(derived) as reopened:
            assert reopened.object(key)['meta']['label']=='original'


@pytest.mark.parametrize('workers,packed',[(1,False),(3,True)])
@pytest.mark.parametrize('old_delta',[False,True])
def test_historical_store_reference_preserves_geometry_and_graph_membership(tmp_path,workers,packed,old_delta):
    from vc_object_store import ObjectStore,Fiber,import_initial_records
    source=tmp_path/'source.vcobj';volume=str(uuid.UUID(int=900))
    ids=[str(uuid.UUID(int=i+1)) for i in range(4)]
    xyz=np.array([[0.,1.,2.],[10.,1.,2.]],dtype='<f8')
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        for key in ids[:3]:
            writer.upsert_object(dict(uuid=key,kind='fiber',name='old',meta=dict(label='old',
                coordinates=dict(volume_uuid=volume,scale_factor=1.))))
            writer.put_geometry(key,Fiber([],xyz.tolist()))
        if old_delta:
            current=writer.current_object_revision(ids[2])
            writer.edit_geometry(ids[2],[dict(array='line_points',element_offset=0,data=(xyz+10).tobytes())],current['version'])
        edge=writer.add_edge(dict(kind='declared_reference',a=ids[0],b=ids[1],
                                  a_anchor=dict(whole_geometry=True),b_anchor=dict(whole_geometry=True)))
        with CollectionReader(source) as reader:pin=reader.source
        derived=import_initial_records(tmp_path/'constraints.vcobj',[],dependencies=[source],workers=workers,packed=packed)['path']
        writer.remove_edge(edge)
        writer.set_meta(ids[0],'label','new')
        current=writer.current_object_revision(ids[2])
        writer.edit_geometry(ids[2],[dict(array='line_points',element_offset=0,data=(xyz+30).tobytes())],current['version'])
        writer.delete_object(ids[1])
        writer.upsert_object(dict(uuid=ids[3],kind='fiber',meta=dict(coordinates=dict(volume_uuid=volume,scale_factor=1.))))
        writer.put_geometry(ids[3],Fiber([],xyz.tolist()))
        writer.add_edge(dict(kind='declared_reference',a=ids[0],b=ids[2],
                             a_anchor=dict(whole_geometry=True),b_anchor=dict(whole_geometry=True)))
        for _ in range(2):
            with CollectionReader(derived) as reader:
                assert sorted(row['uuid'] for row in reader.objects('fiber'))==ids[:3]
                assert reader.object(ids[0])['meta']['label']=='old'
                assert reader.object(ids[1]) is not None
                assert reader.object(ids[3]) is None
                assert [row['uuid'] for row in reader.edges('declared_reference')]==[edge]
            selected,geometry=read_polyline_arrays(derived,ids[:3],volume=volume,workers=3,trusted_immutable='auto')
            assert selected.tolist()==ids[:3]
            np.testing.assert_array_equal(geometry.values,np.concatenate([xyz,xyz,xyz+10 if old_delta else xyz]))
        # A resolved source locator must also be reopenable at its saved pin,
        # including the shared legacy catalog/interpolator consumer APIs.
        ordinary_pin=dict(store=pin['store_uuid'],generation=pin['generation'],commit_sequence=pin['commit_sequence'])
        with CollectionReader(source,position=pin) as reader:
            assert sorted(row['uuid'] for row in reader.objects('fiber'))==ids[:3]
        selected,geometry=read_polyline_arrays(source,ids[:3],volume=volume,source_version=ordinary_pin,
                                               workers=3,trusted_immutable='auto')
        np.testing.assert_array_equal(geometry.values,np.concatenate([xyz,xyz,xyz+10 if old_delta else xyz]))
        from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import fiber_point_counts,fiber_count
        assert fiber_count(source,source_version=ordinary_pin)==3
        assert fiber_point_counts(source,source_version=ordinary_pin)[1].tolist()==[2,2,2]
        restored=read_polylines(source,ids[:3],volume=volume,source_version=ordinary_pin)
        np.testing.assert_array_equal(restored[2][1],xyz+10 if old_delta else xyz)
        with ObjectStore.open(str(derived),read_only=True) as reader:
            assert len(reader.geometry_history(ids[2]))==(2 if old_delta else 1)
        # A further layer must retain the old dependency pin, not repin it to
        # the mutable source's newer head during construction.
        nested=tmp_path/'nested.vcobj';write_extension_store(nested,[derived],[],{})
        with CollectionReader(nested) as reader:
            assert sorted(row['uuid'] for row in reader.objects('fiber'))==ids[:3]
        older=import_initial_records(tmp_path/'explicit.vcobj',[],dependencies=[source],dependency_pins=[pin],
                                     workers=workers,packed=packed)['path']
        with CollectionReader(older) as reader:
            assert sorted(row['uuid'] for row in reader.objects('fiber'))==ids[:3]
            assert [row['uuid'] for row in reader.edges('declared_reference')]==[edge]


def test_history_removal_fails_instead_of_reading_current_source(tmp_path):
    from vc_object_store import ObjectStore,remove_history_and_disable
    source=tmp_path/'source.vcobj';derived=tmp_path/'result.vcobj'
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='note',meta=dict(value='old')))
        write_extension_store(derived,[source],[],{})
        writer.set_meta(key,'value','new')
    remove_history_and_disable(source)
    with pytest.raises(RuntimeError,match='history|retention'):
        CollectionReader(derived)
    from vc_object_store import enable_history
    enable_history(source)
    with pytest.raises(RuntimeError,match='history|lineage'):
        CollectionReader(derived)


def test_reference_to_history_enabled_baseline_survives_later_edits(tmp_path):
    from vc_object_store import ObjectStore,enable_history
    source=tmp_path/'source.vcobj';derived=tmp_path/'result.vcobj'
    with ObjectStore.create(str(source),author='test',history_enabled=False,spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='note',meta=dict(value='old')))
    enable_history(source)
    write_extension_store(derived,[source],[],{})
    with ObjectStore.open(str(source),author='test') as writer:writer.set_meta(key,'value','new')
    with CollectionReader(derived) as reader:assert reader.object(key)['meta']['value']=='old'


def test_explicit_version_reference_is_not_replaced_by_store_snapshot_default(tmp_path):
    from vc_object_store import ObjectStore,object_reference
    from vc_object_store.selected import SelectedObject
    source=tmp_path/'source.vcobj';derived=tmp_path/'result.vcobj'
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='note',meta=dict(value='old')))
        write_extension_store(derived,[source],[],{})
        writer.set_meta(key,'value','new')
        explicit=object_reference(writer.object(key))
        with ObjectStore.open(str(derived),read_only=True) as reader:
            for _ in range(6):
                assert reader.object(key)['meta']['value']=='old'
                chosen=SelectedObject(reader.objects_at_versions([explicit])[0])
                assert chosen.record['meta']['value']=='new' and chosen.reference==explicit


def test_conflicting_historical_source_pins_fail_without_mixing_versions(tmp_path):
    from vc_object_store import ObjectStore
    source=tmp_path/'source.vcobj';old=tmp_path/'old.vcobj';new=tmp_path/'new.vcobj'
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='note',meta=dict(value='old')))
        write_extension_store(old,[source],[],{})
        writer.set_meta(key,'value','new')
        write_extension_store(new,[source],[],{})
        with pytest.raises(RuntimeError,match='conflicting retained publication pins'):
            write_extension_store(tmp_path/'mixed.vcobj',[old,new],[],{})
    assert not (tmp_path/'mixed.vcobj').exists()


@pytest.mark.parametrize('versioned',[False,True])
def test_historical_extension_reads_use_retained_owner_rows_or_fail(tmp_path,versioned):
    from vc_object_store import ObjectStore,extension_uuid
    source=tmp_path/'source.vcobj';derived=tmp_path/'result.vcobj'
    table=dict(name='ext_snapshot_test',extension='test.snapshot',format_version=1,versioned=versioned,
               key_space='objects',provisioning={},based_on=[],
               columns=[dict(name='owner',type='blob'),dict(name='value',type='integer')],
               primary_key=['owner','value'])
    with ObjectStore.create(str(source),author='test',spatial_index=False) as writer:
        writer.create_extension_table(table)
        key=writer.upsert_object(dict(kind='note'))
        old=[[extension_uuid(key),1]];new=[[extension_uuid(key),2]]
        if versioned:
            with writer.write(expected_versions=writer.select_expected_versions({key:writer.current_version(key)})) as tx:
                tx.replace_extension_rows(table['name'],key,old)
        else:writer.append_extension_rows(table['name'],old)
        write_extension_store(derived,[source],[],{})
        if versioned:
            with writer.write(expected_versions=writer.select_expected_versions({key:writer.current_version(key)})) as tx:
                tx.replace_extension_rows(table['name'],key,new)
        else:writer.append_extension_rows(table['name'],new)
        with CollectionReader(derived) as reader:
            assert reader.object(key) is not None
            if versioned:assert reader.extension_rows(table['name'])==old
            else:
                with pytest.raises(RuntimeError,match='historical rows were not retained'):
                    reader.extension_rows(table['name'])


def test_mutable_history_off_dependency_still_rejected(tmp_path):
    from vc_object_store import ObjectStore
    source=tmp_path/'unretained.vcobj'
    with ObjectStore.create(str(source),author='test',history_enabled=False) as writer:
        writer.upsert_object(dict(kind='note'))
        with pytest.raises(RuntimeError,match='sealed or history enabled'):
            write_extension_store(tmp_path/'result.vcobj',[source],[],{})
    assert not (tmp_path/'result.vcobj').exists()


def test_neural_volume_export_can_feed_constraint_store_without_conversion(tmp_path,monkeypatch):
    from vc_object_store import import_initial_records,ObjectStore
    from vesuvius.neural_tracing.tracers.fibers.tracing import volume_store
    run=tmp_path/'tracing';tiles=run/'tiles';tiles.mkdir(parents=True)
    namespace=str(uuid.UUID(int=800));volume=str(uuid.UUID(int=900))
    (run/'run.json').write_text(json.dumps(dict(fiber_namespace=namespace,grid_scale=4.,ct_zarr='test:ct')))
    xyz=np.array([[1.,2.,3.],[2.,2.,3.],[3.,2.,3.]],dtype=np.float32)
    np.savez(tiles/'z00000_y00000_x00000.npz',points=xyz,offsets=[0,3],id=[7],
             seed=xyz[:1],plane=['x'],family=['H'],reasons=[['finished','finished']],links=[[-1,-1]])
    monkeypatch.setattr(volume_store,'open_data_volume',lambda locator:('sample',volume))
    exported=volume_store.export_store(run)
    source=Path(exported['path'])
    with ObjectStore.open(str(source),read_only=True) as reader:
        assert reader.provisioning()['history_enabled'] and not reader.provisioning()['sealed']
    result=import_initial_records(tmp_path/'constraints.vcobj',[],dependencies=[source],workers=2,packed=True)
    key=volume_store.fiber_uuid(namespace,7)
    ids,packed=read_polyline_arrays(result['path'],[key],volume=volume,workers=2,trusted_immutable='auto')
    assert ids.tolist()==[key]
    np.testing.assert_array_equal(packed.values,xyz.astype(np.float64)*4.)


def test_implicit_volume_catalog_loads_records_once_and_matches_explicit(monkeypatch):
    from vesuvius.neural_tracing.tracers.shared.annotations import fiber_store
    volume=str(uuid.UUID(int=900));calls=[]
    records=[dict(uuid=str(uuid.UUID(int=1)),name='fiber',
                  meta=dict(coordinates=dict(volume_uuid=volume,scale_factor=2.)),
                  fiber=dict(family='H',length=10.,line_point_count=3,bbox=[0.,0.,0.,4.,6.,8.]))]
    class Reader:
        def __init__(self,path,*,position=None):assert position is None
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
        def __init__(self,path,*,position=None):assert position is None
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
    packed_ids,packed=read_polyline_arrays(path,[r['id'] for r,_ in curves],volume=volume)
    assert packed_ids.tolist()==[r['id'] for r,_ in loaded]
    np.testing.assert_array_equal(packed.values,np.concatenate([xyz for _,xyz in loaded]))
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
    packed_ids,packed=read_polyline_arrays(path,[uid(1),uid(2)],volume=target)
    assert packed_ids.tolist()==[uid(1),uid(2)]
    np.testing.assert_array_equal(packed.values,np.concatenate([expected,expected]))
    empty_ids,empty=read_polyline_arrays(path,[],volume=target)
    assert not len(empty_ids) and empty.values.shape==(0,3)
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
