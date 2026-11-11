"""Typed result tables and owner-versioned manual fields share native semantics."""
import uuid

import pytest

from vc_object_store import ObjectStore, extension_uuid, fiber_source_extension, fiber_source_row


def definition(versioned=False, key_space="objects"):
    return dict(name="ext_python_measurements", extension="test.python", format_version=1,
                key_space=key_space, versioned=versioned,
                provisioning={"units": "turns"}, based_on=[],
                columns=[dict(name="owner_uuid", type="blob"), dict(name="run", type="integer"),
                         dict(name="value", type="real"), dict(name="note", type="text"),
                         dict(name="payload", type="blob"), dict(name="optional", type="blob", nullable=True)],
                primary_key=["owner_uuid", "run"], indexes=[dict(columns=["value"], unique=False)])


def row(owner, run=1):
    return [extension_uuid(owner), run, 1.25, "UTF-8 \u03b1\x00 text", b"", None]


def assert_reference_validation(db, table, ref):
    malformed = [(dict(ref, unexpected=1), ValueError, "unknown key")]
    malformed += [(dict(ref, version=value), ValueError, "positive integer")
                  for value in (True, 1.5, "2", -1, 2**63)]
    malformed += [({key: value for key, value in ref.items() if key != "generation"},
                   RuntimeError, "generation")]
    for invalid, error, message in malformed:
        with pytest.raises(error, match=message):
            db.extension_rows_at_version(table, invalid)
        with db.write() as txn:
            with pytest.raises(error, match=message):
                txn.extension_rows_at_version(table, invalid)


def test_typed_rows_and_transactional_registration(db):
    spec = definition()
    assert db.status()["extensions_present"] is False
    owner = str(uuid.uuid4())
    expected = [row(owner), row(owner, 2)]
    expected[1][-1] = b""
    with db.write() as txn:
        txn.create_extension_table(spec)
        txn.append_extension_rows(spec["name"], expected)
        assert txn.extension_rows(spec["name"]) == expected
    info, = db.extension_tables()
    assert db.status()["extensions_present"] is True
    assert info["columns"] == [dict(nullable=False, **c) if "nullable" not in c else c for c in spec["columns"]]
    assert info["based_on"] == [] and info["last_modified_commit"] == db.status()["commit_sequence"]
    assert db.extension_rows(spec["name"], [extension_uuid(owner), 2]) == [expected[1]]
    assert isinstance(expected[0][0], bytes) and len(expected[0][0]) == 16
    assert extension_uuid(expected[0][0]) == owner
    visited = []
    assert db.scan_extension_rows(spec["name"], lambda item: visited.append(item) or False) == 1
    assert visited == expected[:1]
    before = db.status()["commit_sequence"]
    db.create_extension_table(spec)
    db.append_extension_rows(spec["name"], [])
    assert db.status()["commit_sequence"] == before
    with pytest.raises(RuntimeError):
        db.append_extension_rows(spec["name"], [row(owner, 3), row(owner)])
    assert db.extension_rows(spec["name"]) == expected
    with pytest.raises(ValueError, match="rollback"):
        with db.write() as txn:
            txn.append_extension_rows(spec["name"], [row(owner, 3)])
            raise ValueError("rollback")
    assert db.extension_rows(spec["name"]) == expected
    with pytest.raises(ValueError):
        db.create_extension_table(dict(spec, format_version=2))
    with pytest.raises(ValueError):
        db.append_extension_rows(spec["name"], [[b"not a uuid", 1, 1.25, "", b"", None]])
    with pytest.raises((RuntimeError, ValueError)):
        db.export_json()


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("owner_kind", ["fiber", "extension.manual_annotation"])
def test_manual_fields_follow_owner_versions(tmp_path, history, owner_kind):
    with ObjectStore.create(str(tmp_path / "manual.vcobj"), author="pytest", history_enabled=history) as db:
        spec = definition(versioned=True)
        db.create_extension_table(spec)
        owner = db.upsert_object(dict(kind=owner_kind))
        expected = [row(owner)]
        with db.write(expected_versions=db.select_expected_versions({owner: 1})) as txn:
            txn.replace_extension_rows(spec["name"], owner, expected)
        assert db.current_version(owner) == 2
        status = db.status()
        ref = dict(store=status["store"], generation=status["generation"], object=owner, version=2)
        assert db.extension_rows_at_version(spec["name"], ref) == expected
        assert_reference_validation(db, spec["name"], ref)
        with db.write(expected_versions=db.select_expected_versions({owner: 2})) as txn:
            txn.replace_extension_rows(spec["name"], owner, expected)
        assert db.current_version(owner) == 2 and db.status() == status
        newer = [row(owner, 2)]
        newer[0][-2] = b"\x00\xff" * 1024
        with db.write(expected_versions=db.select_expected_versions({owner: 2})) as txn:
            txn.replace_extension_rows(spec["name"], owner, newer)
        assert db.current_version(owner) == 3 and db.extension_rows(spec["name"]) == newer
        if history:
            assert db.extension_rows_at_version(spec["name"], ref) == expected
            with db.write(expected_versions=db.select_expected_versions({owner: 3})) as txn:
                txn.restore_object(owner, source_version=2)
            assert db.extension_rows(spec["name"]) == expected
        else:
            with pytest.raises((ValueError, RuntimeError)):
                db.extension_rows_at_version(spec["name"], ref)
        with pytest.raises(RuntimeError):
            with db.write(expected_versions=db.select_expected_versions({owner: 2})):
                pass
        with pytest.raises(ValueError):
            db.set_meta(owner, "vcobj_extensions", {})


def test_manual_relationship_reference(db):
    spec = definition(versioned=True, key_space="edges")
    db.create_extension_table(spec)
    a = db.upsert_object(dict(kind="fiber"))
    b = db.upsert_object(dict(kind="fiber"))
    edge = db.add_edge(dict(kind="test_link", a=a, b=b,
                            a_anchor=dict(whole_geometry=True), b_anchor=dict(whole_geometry=True)))
    first = [row(edge)]
    with db.write(expected_versions=db.select_expected_versions({a: db.current_version(a), b: db.current_version(b)}),
                  expected_relationships=db.select_expected_versions({edge: 1}, relationships=True)) as txn:
        txn.replace_extension_rows(spec["name"], edge, first)
    state = db.status()
    selected = db.relationships_at_versions([
        dict(store=state["store"], generation=state["generation"], object=uid, version=db.current_version(uid))
        for uid in (a, b)])[0]
    ref = dict(store=selected["store"], generation=selected["generation"], relationship=edge, version=selected["version"])
    assert ref["version"] == 2 and selected["record"]["version_store"] == ref["store"]
    assert db.extension_rows_at_version(spec["name"], ref) == first
    assert_reference_validation(db, spec["name"], ref)
    with db.write(expected_versions=db.select_expected_versions({a: db.current_version(a), b: db.current_version(b)}),
                  expected_relationships=db.select_expected_versions({edge: 2}, relationships=True)) as txn:
        txn.replace_extension_rows(spec["name"], edge, [])
    assert db.extension_rows(spec["name"]) == []
    assert db.extension_rows_at_version(spec["name"], ref) == first


def test_fiber_source_helpers_share_native_schema(db):
    source_uuid = str(uuid.uuid4())
    spec = fiber_source_extension(source_uuid, fiber_count=1, point_count=20)
    db.create_extension_table(spec)
    owner = str(uuid.uuid4())
    value = fiber_source_row(owner, 17, {"manual": {"confidence": 0.75}})
    assert value["table"] == spec["name"] == "ext_fiber_graph_solve_sources"
    db.append_extension_rows(value["table"], [value["values"]])
    assert db.extension_rows(spec["name"], [extension_uuid(owner)]) == [value["values"]]
    info, = db.extension_tables()
    assert info["provisioning"] == dict(source_uuid=source_uuid, fiber_count=1, point_count=20)
    assert info["based_on"] == []
