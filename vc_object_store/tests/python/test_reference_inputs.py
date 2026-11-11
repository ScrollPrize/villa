"""Legacy input decoding stops at the native boundary; internal values stay complete."""
import json
import pytest
from vc_object_store import ObjectStore, ObjectStoreSession, object_reference, object_version_from_json


@pytest.mark.parametrize("history", [False, True])
def test_external_selectors_decode_once_to_shared_reference(tmp_path, history):
    path = tmp_path / "input.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=history) as store:
        uuid = store.upsert_object(dict(kind="note"))
        ref = object_reference(store.object(uuid))
    session = ObjectStoreSession()
    attached = session.attach_location(str(path), "test", False)
    inputs = [
        ref, dict(type="note", reference=ref), dict(type="note", uuid=uuid),
        f"vcobj:{uuid}", f"vcobj:{uuid}@{ref['generation']}/01",
        f"vcobj:{uuid}@{ref['store']}/{ref['generation']}/1",
        dict(uuid=uuid, store=ref["store"], generation=ref["generation"], version=1),
    ]
    for value in inputs:
        assert session.decode_object_reference(attached, value) == ref
    assert object_version_from_json(json.dumps(ref)) == ref
    for bad in [f"vcobj:{uuid}@g/0", f"vcobj:{uuid}@g/1x",
                dict(uuid=uuid, revision="old"), dict(uuid=uuid, generation=ref["generation"]),
                dict(reference=ref, uuid="11111111-2222-4333-8444-555555555555")]:
        with pytest.raises((ValueError, RuntimeError)):
            session.decode_object_reference(attached, bad)
    session.detach(attached)


def test_exact_input_does_not_depend_on_current_head(tmp_path):
    path = tmp_path / "input.vcobj"
    with ObjectStore.create(str(path), author="test") as store:
        uuid = store.upsert_object(dict(kind="note"))
        ref = object_reference(store.object(uuid))
        store.delete_object(uuid)
    session = ObjectStoreSession()
    attached = session.attach_location(str(path), "test", False)
    assert session.decode_object_reference(attached, ref) == ref
    assert session.decode_object_reference(attached, dict(uuid=uuid, **{
        k: ref[k] for k in ("store", "generation", "version")})) == ref
    with pytest.raises(ValueError, match="unavailable"):
        session.decode_object_reference(attached, dict(uuid=uuid, generation=ref["generation"], version=1))
    session.detach(attached)
