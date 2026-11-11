"""Current discovery and complete exact inputs meet at one native boundary."""
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
        f"vcobj:{uuid}",
        f"vcobj:{uuid}@{ref['store']}/{ref['generation']}/1",
    ]
    for value in inputs:
        assert session.decode_object_reference(attached, value) == ref
    assert object_version_from_json(json.dumps(ref)) == ref
    for bad in [f"vcobj:{uuid}@{ref['generation']}/01", f"vcobj:{uuid}@g/0", f"vcobj:{uuid}@g/1x",
                dict(uuid=uuid, store=ref["store"], generation=ref["generation"], version=1),
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
    with pytest.raises(ValueError, match="complete reference"):
        session.decode_object_reference(attached, dict(uuid=uuid, generation=ref["generation"], version=1))
    session.detach(attached)


@pytest.mark.parametrize('value', [True, 1.5, '1', -1, 2**63])
def test_native_position_and_surface_selector_reject_coercion(db_path, value):
    from vc_object_store import Surface, store_position_from_json
    from vc_object_store.surfaces import read_surface
    with ObjectStore.create(db_path, author='test') as store:
        uuid = store.upsert_object(dict(kind='surface', meta={'coordinates': {'volume_uuid': 'vol', 'scale_factor': 1}}))
        store.put_geometry(uuid, Surface(1, 1, [1, 1], [[2, 4, 6]]))
        status = store.status()
    with pytest.raises((ValueError, RuntimeError), match='commit_sequence'):
        store_position_from_json(json.dumps(dict(status, commit_sequence=value)))
    with pytest.raises((ValueError, RuntimeError), match='version'):
        read_surface(db_path, uuid, volume='vol', version=value)
    session = ObjectStoreSession()
    attachment = session.attach(db_path, 'test', read_only=True)
    with pytest.raises((ValueError, RuntimeError), match='version'):
        session.read_surface_grid(attachment, uuid, 'vol', 1, value)
    session.detach(attachment)


def test_manifest_membership_uses_complete_native_references(db):
    from vc_object_store import manifest_references
    uuid = db.upsert_object(dict(kind='note'))
    reference = object_reference(db.object(uuid))
    entry = dict(uuid=uuid, reference=reference, version=reference['version'])
    assert manifest_references({'objects': [entry]})['objects'] == {uuid: reference}
    for invalid in ({'objects': [entry, entry]}, {'objects': [entry], 'outer_shell': entry},
                    {'objects': [dict(entry, reference=dict(reference, object='11111111-2222-4333-8444-555555555555'))]},
                    {'objects': [dict(entry, version=True)]}, {'objects': [dict(entry, version=1.5)]}):
        with pytest.raises((ValueError, RuntimeError)):
            manifest_references(invalid)


@pytest.mark.parametrize('value', [True, 1.5, -0.5, '1', -1, 2**63])
def test_selection_artifacts_share_strict_position_ingress(db, value):
    import struct
    from vc_object_store import encode_selection_record, decode_selection_record
    record = dict(db.status(), query={}, transforms=[], objects=[])
    encoded = encode_selection_record(record)
    with pytest.raises(ValueError, match='commit_sequence'):
        encode_selection_record(dict(record, commit_sequence=value))
    length, = struct.unpack('<I', encoded[8:12])
    header = json.loads(encoded[12:12 + length])
    for field in ('commit_sequence', 'count'):
        changed = json.dumps(dict(header, **{field: value})).encode()
        malformed = encoded[:8] + struct.pack('<I', len(changed)) + changed + encoded[12 + length:]
        with pytest.raises(ValueError, match='commit_sequence' if field == 'commit_sequence' else 'entry count'):
            decode_selection_record(malformed)
