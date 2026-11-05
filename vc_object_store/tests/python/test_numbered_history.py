"""Numbered history, checked requests, artifacts and selection records from Python."""
import uuid

import pytest

import vc_object_store as vos


def create(db, name, status):
    obj = str(uuid.uuid4())
    request = dict(command_version=1, request=str(uuid.uuid4()), store=status["store"],
                   generation=status["generation"], author="py", expected_objects={obj: None},
                   expected_relationships={}, read_dependencies={}, expected_commit_sequence=None,
                   operations=[dict(op="create_object", record=dict(uuid=obj, kind="note", name=name))])
    return obj, request, db.submit(request)


def test_checked_request_versions_and_selection_record(db):
    status = db.status()
    a, request, receipt = create(db, "a", status)
    assert db.submit(request)["replayed"] and db.receipt(request["request"]) is not None
    b, _, _ = create(db, "b", status)
    assert db.current_version(a) == 1 and db.status()["commit_sequence"] == status["commit_sequence"] + 2

    stale = dict(request, request=str(uuid.uuid4()), expected_objects={a: None},
                 operations=[dict(op="set_meta", uuid=a, key="x", value=1)])
    with pytest.raises(RuntimeError):
        db.submit(stale)

    rows = {r["uuid"]: r for r in db.select_objects(object_kind="note")}
    assert rows[a]["version"] == 1 and rows[a]["version_generation"] == status["generation"]

    ref = dict(store=status["store"], generation=status["generation"], object=a, version=1)
    missing = dict(ref, version=9)
    found = db.objects_at_versions([ref, missing])
    assert found[0]["record"]["name"] == "a" and found[1] is None
    assert db.relationships_at_versions([ref]) == []
    changes = db.changes_since(dict(store=status["store"], generation=status["generation"], sequence=0))
    assert {c["uuid"] for commit in changes["commits"] for c in commit["changes"]} >= {a, b}

    record = db.record_selection(dict(selection=dict(kind="note")))
    assert record["commit_sequence"] == db.status()["commit_sequence"]
    assert sorted(o["object"] for o in record["objects"]) == sorted([a, b])
    data = vos.encode_selection_record(record)
    assert len(data) < 1024 and vos.decode_selection_record(data) == record
    with pytest.raises(ValueError):
        vos.decode_selection_record(data[:-1])


def test_artifacts_named_by_manifest(db, tmp_path):
    status = db.status()
    obj, _, _ = create(db, "model", status)
    source = tmp_path / "weights.bin"
    source.write_bytes(b"abc" * 1000)
    digest = db.put_artifact(source)
    assert db.artifact_length(digest) == 3000 and db.artifact_length("0" * 64) is None
    assert db.read_artifact(digest, 3, 3) == b"abc"
    manifest = vos.artifact_manifest_json([dict(name="w/weights.bin", sha256=digest, length=3000)])
    request = dict(command_version=1, request=str(uuid.uuid4()), store=status["store"],
                   generation=status["generation"], author="py", expected_objects={obj: 1},
                   expected_relationships={}, read_dependencies={}, expected_commit_sequence=None,
                   operations=[dict(op="set_meta", uuid=obj, key=vos.ARTIFACTS_META_KEY, value=__import__("json").loads(manifest))])
    db.submit(request)
    assert db.current_object_revision(obj)["record"]["meta"]["artifacts"][0]["sha256"] == digest
    unheld = dict(request, request=str(uuid.uuid4()), expected_objects={obj: 2},
                  operations=[dict(op="set_meta", uuid=obj, key="artifacts",
                                   value=[dict(name="x", sha256="1" * 64, length=1)])])
    with pytest.raises(Exception):
        db.submit(unheld)
    assert db.current_version(obj) == 2
