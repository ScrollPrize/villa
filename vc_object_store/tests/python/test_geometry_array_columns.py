"""Shared parallel packed reads preserve current snapshots and array ownership."""
import gc
import subprocess
import sys
import uuid

import numpy as np
import pytest

from vc_object_store import CollectionReader,Fiber,ObjectStore,StoreBusy,import_initial_records


def _keys(ids):
    return np.frombuffer(b''.join(uuid.UUID(value).bytes for value in ids),dtype=np.uint8).reshape(-1,16).copy()


def _immutable_fibers(tmp_path,count,*,packed):
    ids=[str(uuid.UUID(int=i+1)) for i in range(count)]
    rows=[];expected=[]
    for i,key in enumerate(ids):
        xyz=np.array([[float(i),-0.,2.],[float(i)+6.,8.,4.]],dtype='<f8')
        scale=2. if i%2 else 4.
        rows.append(dict(record=dict(uuid=key,kind='fiber',format_version=4,
            meta=dict(coordinates=dict(volume_uuid='source',scale_factor=scale))),
            geometry=dict(kind='fiber',arrays=[
                dict(name='control_points',dtype='<f8',shape=[0,3],data=b''),
                dict(name='control_point_ids',dtype='<i8',shape=[0],data=b''),
                dict(name='line_points',dtype='<f8',shape=[2,3],data=xyz.tobytes())])))
        expected.append(xyz*(1./scale))
    result=import_initial_records(tmp_path/'source.vcobj',rows,workers=4,packed=packed)
    return result['path'],ids,np.concatenate(expected)


@pytest.mark.parametrize('packed',[False,True])
@pytest.mark.parametrize('workers',[1,4])
@pytest.mark.parametrize('trusted',[False,True])
def test_parallel_geometry_columns_match_mapping_and_survive_closed_reader(tmp_path,packed,workers,trusted):
    # More than one bounded block is essential: small fixtures would only test
    # one connection and miss the worker snapshot/routing implementation.
    path,ids,expected=_immutable_fibers(tmp_path,4103,packed=packed)
    request=_keys(ids[::-1]+[ids[0]])
    with CollectionReader(path) as reader:
        source=reader.source
        position=source if workers>1 else dict(store=source['store_uuid'],generation=source['generation'],
                                               commit_sequence=source['commit_sequence'])
        result=reader.geometry_array_columns(request,'line_points',object_kind='fiber',volume='source',
                                             expected_position=position,workers=workers,profile=True,
                                             trusted_immutable=trusted)
    gc.collect()
    packed_result=result['geometry']
    actual_ids=result['uuid_ascii'].view('S36').ravel().astype('U36')
    assert actual_ids.tolist()==ids
    np.testing.assert_array_equal(result['object_uuid'],_keys(ids))
    assert result['source']==source
    assert packed_result.dtype=='<f8' and packed_result.shape==[8206,3]
    np.testing.assert_array_equal(packed_result.offsets,np.arange(4104,dtype=np.uint64)*2)
    np.testing.assert_array_equal(packed_result.values.view(np.uint64),expected.view(np.uint64))
    assert not result['object_uuid'].flags.writeable
    assert not packed_result.values.flags.writeable
    timings=result['timings']
    assert timings['objects']==4103 and timings['blocks']==2
    assert timings['workers']==min(workers,2)
    assert timings['selection_seconds_sum']>=0 and timings['payload_mapping_seconds_sum']>=0
    assert timings['total_seconds']>0


@pytest.mark.parametrize('history',[False,True])
def test_mutable_single_worker_keeps_snapshot_and_parallel_never_follows_newer_heads(tmp_path,history):
    path=tmp_path/'mutable.vcobj'
    before=np.array([[0.,-0.,2.],[6.,8.,4.]])
    with ObjectStore.create(str(path),author='test',history_enabled=history,spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='fiber',meta=dict(coordinates=dict(volume_uuid='source',scale_factor=1.))))
        writer.put_geometry(key,Fiber([],before.tolist()))
        with CollectionReader(str(path)) as reader:
            pin=reader.source
            current=writer.current_object_revision(key)
            writer.edit_geometry(key,[dict(array='line_points',element_offset=0,
                data=np.ascontiguousarray(before+10,dtype='<f8').tobytes())],current['version'])
            result=reader.geometry_array_columns(_keys([key]),'line_points',object_kind='fiber',volume='source',
                                                 expected_position=pin,workers=1)
            np.testing.assert_array_equal(result['geometry'].values.view(np.uint64),before.view(np.uint64))
            assert result['source']==pin
            error=StoreBusy if history else ValueError
            message='source position changed' if history else 'sealed store'
            with pytest.raises(error,match=message):
                reader.geometry_array_columns(_keys([key]),'line_points',workers=4)
            with pytest.raises(ValueError,match='sealed history-off'):
                reader.geometry_array_columns(_keys([key]),'line_points',workers=1,trusted_immutable=True)
        with CollectionReader(str(path)) as latest:
            with pytest.raises(StoreBusy,match='source position'):
                latest.geometry_array_columns(_keys([key]),'line_points',expected_position=pin,workers=1)


def test_trusted_geometry_rejects_sealed_history_enabled_scope(tmp_path):
    path=tmp_path/'history.vcobj'
    with ObjectStore.create(str(path),author='test',history_enabled=True,spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='fiber'))
        writer.put_geometry(key,Fiber([],[[1.,2.,3.]]))
        writer.seal()
    with CollectionReader(path) as reader:
        assert reader.geometry_array_columns(_keys([key]),'line_points',workers=2)['geometry'].shape==[1,3]
        with pytest.raises(ValueError,match='sealed history-off'):
            reader.geometry_array_columns(_keys([key]),'line_points',workers=2,trusted_immutable=True)


def test_trusted_geometry_skips_integrity_audit_but_keeps_payload_bounds(tmp_path):
    path=tmp_path/'integrity.vcobj'
    with ObjectStore.create(str(path),author='test',history_enabled=False,spatial_index=False) as writer:
        key=writer.upsert_object(dict(kind='fiber'))
        writer.put_geometry(key,Fiber([],[[1.,2.,3.]]))
        writer.seal()
    def damage(statement):
        # Separate SQLite libraries must not share POSIX WAL locks in-process.
        subprocess.run([sys.executable,'-c',
            'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.executescript(sys.argv[2]); c.close()',
            str(path),statement],check=True)
    damage('DROP TRIGGER IF EXISTS geometry_chunks_immutable_update; '
           'UPDATE geometry_chunks SET data=zeroblob(length(data)) WHERE length(data)>0;')
    with CollectionReader(path) as reader:
        with pytest.raises(RuntimeError):
            reader.geometry_array_columns(_keys([key]),'line_points',workers=1)
        trusted=reader.geometry_array_columns(_keys([key]),'line_points',workers=1,trusted_immutable=True)
        np.testing.assert_array_equal(trusted['geometry'].values,np.zeros((1,3)))
    damage('UPDATE geometry_chunks SET data=zeroblob(1) WHERE length(data)>0;')
    with CollectionReader(path) as reader:
        with pytest.raises(RuntimeError,match='byte length'):
            reader.geometry_array_columns(_keys([key]),'line_points',workers=1,trusted_immutable=True)


def test_parallel_geometry_columns_validate_inputs_and_missing_objects(tmp_path):
    path,ids,_=_immutable_fibers(tmp_path,2,packed=True)
    with CollectionReader(path) as reader:
        with pytest.raises(ValueError,match='16'):
            reader.geometry_array_columns(np.zeros((2,15),np.uint8),'line_points',workers=2)
        with pytest.raises(ValueError,match='positive'):
            reader.geometry_array_columns(_keys(ids),'line_points',workers=0)
        with pytest.raises(ValueError,match='requested object'):
            reader.geometry_array_columns(_keys([str(uuid.UUID(int=99))]),'line_points',workers=2)
        with pytest.raises(ValueError,match='array not found'):
            reader.geometry_array_columns(_keys(ids),'missing_array',workers=2)
        with pytest.raises(ValueError,match='requested object'):
            reader.geometry_array_columns(_keys(ids),'line_points',volume='unmapped',workers=2)
        empty=reader.geometry_array_columns(np.empty((0,16),np.uint8),'line_points',volume='source',workers=2)
        assert empty['geometry'].values.shape==(0,3)
        assert empty['object_uuid'].shape==(0,16)
