"""Selected records carry identity and metadata together, independent of payload reuse."""
import uuid

import pytest

from vc_object_store import ObjectStore, StoreBusy, Track, object_reference
from vc_object_store.selected import SelectedObject, SelectedRecord


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
