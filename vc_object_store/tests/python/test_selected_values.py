"""Selected records carry identity and metadata together, independent of payload reuse."""
import uuid

import pytest

from vc_object_store import ObjectStore, StoreBusy, Track, object_reference
from vc_object_store.selected import SelectedObject, SelectedRecord


@pytest.mark.parametrize('history', [False, True])
def test_geometry_array_counts_are_descriptor_only_and_owned(tmp_path, history):
    import numpy as np
    from vc_object_store import Fiber
    path=tmp_path/'counts.vcobj'
    with ObjectStore.create(str(path), author='test', history_enabled=history) as store:
        a=store.upsert_object(dict(kind='fiber'))
        b=store.upsert_object(dict(kind='fiber'))
        store.put_geometry(a, Fiber([], [[1.,2.,3.],[4.,5.,6.]]))
        columns=store.geometry_array_counts('fiber','line_points')
        assert columns['object_uuid'].shape==(2,16)
        ids=np.frombuffer(columns['uuid_ascii'],dtype='S36').astype('U36')
        assert dict(zip(ids,columns['count']))=={a:2,b:0}
        assert not columns['count'].flags.writeable
        # A metadata-only edit must retain the selected geometry descriptor.
        store.set_meta(a,'label','changed')
        assert list(store.geometry_array_counts('fiber','line_points')['count'])==list(columns['count'])
        store.delete_object(b)
        assert store.geometry_array_counts('fiber','line_points')['count'].tolist()==[2]
    assert sorted(columns['count'].tolist())==[0,2]  # owned past close


def test_selected_geometry_keeps_metadata_for_shared_payload(db):
    uid = db.upsert_object(dict(kind="track", meta={"coordinates": {"volume_uuid": "a", "scale_factor": 1}}))
    payload = db.put_geometry(uid, Track([[2, 4, 6]]))
    first = object_reference(db.object(uid))
    db.set_meta(uid, "coordinates", {"volume_uuid": "a", "scale_factor": 2})
    current = object_reference(db.object(uid))
    selected = SelectedObject(db.objects_at_versions([first])[0])
    assert selected.reference == first == SelectedRecord(selected.record).reference
    assert selected.geometry_revision == payload and selected.uuid == uid
    with db.geometry_snapshot() as snapshot:
        old = snapshot.track(selected.reference)
        new = snapshot.track(current)
        assert old.reference == first and new.reference == current
        assert old.view_transformed_at("a").points("points", 0, 1) == [[2, 4, 6]]
        assert new.view_transformed_at("a").points("points", 0, 1) == [[1, 2, 3]]
        assert snapshot.track(uid, payload).reference is None
        with pytest.raises(ValueError, match="cannot be combined"):
            snapshot.track(first, payload)
        with pytest.raises(RuntimeError):
            db.edit_transformed_coordinates(old.view_transformed_at("a"), "points", 0, [[3, 4, 5]])
    db.delete_object(uid)
    with db.geometry_snapshot() as snapshot:
        assert snapshot.track(first).reference == first
        assert snapshot.track(first).view_transformed_at("a").points("points", 0, 1) == [[2, 4, 6]]


def test_exact_graph_projection_uses_native_field_policy(db):
    uid = db.upsert_object(dict(kind="track", name="selected", meta={"original": True}))
    db.put_geometry(uid, Track([[2, 4, 6]]))
    reference = object_reference(db.object(uid))
    db.set_meta(uid, "original", False)
    with db.graph_read(versions=[reference]) as graph:
        row, = graph.objects(["name", "meta"])
        assert row["name"] == "selected" and row["meta"]["original"] is True
        for method in (graph.objects, graph.edges):
            with pytest.raises(ValueError, match="projection field"):
                method(["unknown_field"])
        assert graph.edges(["a_anchor"]) == []


def test_geometry_only_graph_read_preserves_selected_arrays_without_edges(tmp_path):
    import numpy as np
    from vc_object_store import import_initial_records
    path=tmp_path/'no-adjacency.vcobj'
    a,b=str(uuid.uuid4()),str(uuid.uuid4())
    points=np.array([[0.,1.,2.],[3.,4.,5.]],dtype='<f8')
    objects=[dict(record=dict(uuid=fid,kind='fiber'),geometry=dict(kind='fiber',format_version=1,
                 arrays=[dict(name='control_points',dtype='<f8',shape=[0,3],data=b''),
                         dict(name='control_point_ids',dtype='<i8',shape=[0],data=b''),
                         dict(name='line_points',dtype='<f8',shape=[2,3],data=points.tobytes())])) for fid in (a,b)]
    import_initial_records(path,objects,edges=[dict(record=dict(uuid=str(uuid.uuid4()),kind='parallel',a=a,b=b))],
                           workers=1,packed=False,edge_adjacency_index=False)
    with ObjectStore.open(str(path),read_only=True) as store:
        refs=[object_reference(store.object(fid)) for fid in (a,b)]
        with store.graph_read(versions=refs,approval_states=[]) as graph:
            assert len(graph.edge_uuids)==1
        with store.graph_read(versions=refs,edge_limit=0) as geometry:
            assert len(geometry.object_uuids)==2 and geometry.edge_uuids==[]
            assert len(geometry.read_geometry_arrays('line_points').values)==4
        with pytest.raises(ValueError,match='limit'):
            store.graph_read(versions=refs,edge_limit=-1)


@pytest.mark.parametrize("history", [False, True])
def test_selected_input_capture_validates_position(tmp_path, history):
    with ObjectStore.create(str(tmp_path / "inputs.vcobj"), author="test", history_enabled=history) as db:
        uid = db.upsert_object(dict(kind="note", name="selected"))
        reference = object_reference(db.object(uid))
        position = db.status()
        result = db.selected_inputs([reference], position)
        assert SelectedObject(result["objects"][0]).reference == reference
        assert result["relationships"] == []
        with pytest.raises(StoreBusy, match="position"):
            db.selected_inputs([reference], dict(position, store=str(uuid.uuid4())))
        for invalid in (True, 1.5, -1, 2**63, 2**64):
            with pytest.raises((ValueError, RuntimeError), match="commit_sequence"):
                db.selected_inputs([reference], dict(position, commit_sequence=invalid))
        db.set_meta(uid, "changed", True)
        if history:
            assert db.selected_inputs([reference], position)["objects"][0]["record"]["name"] == "selected"
        else:
            with pytest.raises(ValueError, match="not retained"):
                db.selected_inputs([reference], position)
