"""Current bulk geometry is one snapshot, not a historical revalidation loop."""
import numpy as np
import pytest

from vc_object_store import ObjectStore,Fiber,StoreBusy,object_reference


@pytest.mark.parametrize('history',[False,True])
def test_current_bulk_geometry_matches_exact_read_and_pins_metadata(tmp_path,history):
    with ObjectStore.create(str(tmp_path/'current.vcobj'),author='test',history_enabled=history,spatial_index=False) as store:
        ids=[]
        xyz=np.array([[0.,-0.,2.],[6.,8.,4.]])
        for scale in (2.,4.):
            uid=store.upsert_object(dict(kind='fiber',meta={'coordinates':{'volume_uuid':'source','scale_factor':scale}}))
            store.put_geometry(uid,Fiber([],xyz.tolist()));ids.append(uid)
        ids.sort()
        records=store.select_objects(object_kind='fiber')
        refs=[object_reference(r) for r in records]
        with store.graph_read(versions=refs,edge_limit=0) as exact:
            expected=exact.read_geometry_arrays('line_points',volume='source').values.copy()
        position={k:store.status()[k] for k in ('store','generation','commit_sequence')}
        with store.geometry_read(object_kind='fiber',object_uuids=ids,volume='source',expected_position=position) as view:
            assert view.object_uuids==ids and view.edge_uuids==[]
            assert view.partners.size==0
            packed=view.read_geometry_arrays('line_points',volume='source')
            np.testing.assert_array_equal(packed.values.view(np.uint64),expected.view(np.uint64))
            before=view.objects(['meta'])
            store.set_meta(ids[0],'coordinates',{'volume_uuid':'source','scale_factor':8})
            assert view.objects(['meta'])==before
            np.testing.assert_array_equal(view.read_geometry_arrays('line_points',volume='source').values,expected)
            assert view.status()['commit_sequence']==position['commit_sequence']
        assert packed.values.shape==(4,3)  # owned after closing its read
        with pytest.raises(StoreBusy,match='source position'):
            store.geometry_read(object_kind='fiber',expected_position=position)


def test_bulk_geometry_registered_mapping_and_volume_pagination(tmp_path):
    with ObjectStore.create(str(tmp_path/'mapping.vcobj'),author='test',history_enabled=False,spatial_index=False) as store:
        uid=store.upsert_object(dict(kind='fiber',meta={'coordinates':{'volume_uuid':'source','scale_factor':2}}))
        xyz=[[0.,0.,0.],[6.,8.,0.]]
        store.put_geometry(uid,Fiber([],xyz))
        other=store.upsert_object(dict(kind='fiber',meta={'coordinates':{'volume_uuid':'unmapped','scale_factor':1}}))
        store.put_geometry(other,Fiber([],xyz))
        store.register_volume_transform('source','target',[[2,.5,0,10],[0,3,0,-5],[0,0,1,2]])
        with store.geometry_snapshot() as snapshot:
            expected=snapshot.fiber(uid).view_transformed_at('target').points('line_points',0,2)
        with store.geometry_read(object_kind='fiber',volume='target',limit=1) as view:
            assert view.object_uuids==[uid]
            np.testing.assert_array_equal(view.read_geometry_arrays('line_points',volume='target').values,expected)
        with store.geometry_read(object_kind='fiber',volume='target',after_uuid=uid) as view:
            assert view.object_uuids==[]
        store.delete_object(uid)
        with store.geometry_read(object_kind='fiber',volume='target') as view:
            assert view.object_uuids==[]
