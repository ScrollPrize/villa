"""Tests for the ``vc.objects`` bindings (object records and relationships).

Run against the built module with
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python
.venv/bin/python -m pytest python/tests/test_object_records.py``.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

ObjectStore = pytest.importorskip("vc.objects").ObjectStore


def test_snapshot_preserves_provenance_and_conflicts_are_atomic(db, tmp_path):
    a = db.upsert_object({"kind": "patch", "data_generation": 1,
                          "patch": {"scale": [0.25, 0.5]}, "tags": ["review"]})
    b = db.upsert_object({"kind": "fiber"})
    edge = db.add_edge({"kind": "connected", "a": b, "b": a,
                        "a_anchor": {"points": [{"array": "control_points", "index": [3]}], "position": [1, 2, 3]}})
    db.set_meta(a, "source", "scan")
    db.set_meta(edge, "confidence", 0.8)
    db.set_approval(a, "approved")
    db.set_approval(edge, "rejected")
    snapshot = json.loads(db.export_json())
    assert snapshot["version"] == 2
    with ObjectStore.create(str(tmp_path / "restored.vcobj"), author="importer", hostname="elsewhere") as restored:
        # Different internal IDs must not affect endpoint/anchor comparison.
        swapped = dict(snapshot, objects=list(reversed(snapshot["objects"])))
        restored.import_json(json.dumps(swapped))
        for uid in (a, b):
            assert restored.object(uid) == db.object(uid)
        result = json.loads(restored.export_json())
        assert result["edges"] == snapshot["edges"]
        assert sorted(result["objects"], key=lambda o: o["uuid"]) == sorted(snapshot["objects"], key=lambda o: o["uuid"])
        journal = restored.journal_since()
        assert all(e["author"] == "importer" for e in journal)
        restored.import_json(json.dumps(snapshot))
        assert restored.journal_since() == journal
        # Fail after a new record was inserted: the entire import rolls back.
        conflicting = json.loads(json.dumps(snapshot))
        new_object = dict(conflicting["objects"][0], uuid=str(uuid.uuid4()))
        conflicting["objects"].insert(0, new_object)
        conflicting["edges"][0]["meta"]["confidence"] = 0.1
        with pytest.raises(RuntimeError, match="conflicts with existing edge"):
            restored.import_json(json.dumps(conflicting))
        assert restored.object(new_object["uuid"]) is None
        assert restored.journal_since() == journal


@pytest.mark.parametrize("kind", ["object", "edge"])
def test_approval_journal_keeps_decisions_and_stamps(db, kind):
    a = db.upsert_object({"kind": "fiber"})
    uid = a
    if kind == "edge":
        b = db.upsert_object({"kind": "patch"})
        uid = db.add_edge({"kind": "connected", "a": a, "b": b})
    for state in ("approved", "rejected", "pending", "approved"):
        db.set_approval(uid, state)
    db.set_meta(uid, "edited", True)
    entries = [e for e in db.journal_since() if e["record_uuid"] == uid]
    assert [e["approval_state"] for e in entries] == ["pending", "approved", "rejected", "pending", "approved", "pending"]
    for entry in (entries[1], entries[2], entries[4]):
        assert entry["approved_by"] == "pytest"
        assert entry["approved_host"] == "testhost"
        assert entry["approved_at"]
        assert entry["approved_generation"] == 1
    assert entries[-1]["generation"] == 2
    assert entries[-1]["approved_by"] is None



def test_object_and_edge_roundtrip(db) -> None:
    a = db.upsert_object({"kind": "fiber", "name": "a", "tags": ["h"],
                          "fiber": {"family": "H", "control_point_count": 3}})
    b = db.upsert_object({"kind": "patch", "name": "b",
                          "patch": {"grid_width": 10, "grid_height": 20}})
    edge = db.add_edge({"a": a, "b": b, "kind": "same-surface",
                        "meta": {"source": "pytest"},
                        "a_anchor": {"points": [{"array": "control_points", "index": [1]}], "position": [0.0, 0.0, 0.0]}})

    obj = db.object(a)
    assert obj["kind"] == "fiber"
    assert obj["generation"] == 1
    assert obj["created_by"] == "pytest"
    assert obj["created_host"] == "testhost"
    assert obj["approval_state"] == "pending"
    assert obj["fiber"]["family"] == "H"
    assert db.object("00000000-0000-4000-8000-000000000000") is None

    stored = db.edge(edge)
    assert stored["kind"] == "same-surface"
    assert stored["meta"] == {"source": "pytest"}
    anchor = stored["a_anchor"] if stored["a"] == a else stored["b_anchor"]
    assert anchor["points"][0]["index"] == [1]

    assert set(db.component(a)) == {a, b}
    assert db.neighbours(a) == [b]
    assert db.neighbours(a, neighbour_kind="patch") == [b]
    assert db.neighbours(a, neighbour_kind="fiber") == []
    assert len(db.edges_of(b)) == 1


@pytest.mark.parametrize("kind", ["fiber", "patch"])
def test_resolve_storage_path_by_uuid(db, tmp_path, kind) -> None:
    uuid = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    root = tmp_path / "dataset"
    root.mkdir()
    original = root / ("original" if kind == "patch" else "original.json")
    if kind == "patch":
        original.mkdir()
        metadata = original / "meta.json"
    else:
        metadata = original
    metadata.write_text(json.dumps({"uuid": uuid, "line_points": []}))
    db.upsert_object({"uuid": uuid, "kind": kind, "name": "original",
                      "storage_kind": "file", "storage_ref": original.name})
    assert db.resolve_storage_path(uuid, dataset_root=root) == original
    renamed = original.with_name("renamed" if kind == "patch" else "renamed.json")
    original.rename(renamed)
    assert db.resolve_storage_path(uuid, dataset_root=root) == renamed
    assert db.resolve_storage_path(uuid) == renamed
    assert db.object(uuid)["storage_ref"] == original.name
    assert db.resolve_storage_path("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb") is None


def test_kinds_are_data_not_api(db) -> None:
    # A kind the library has no detail table for uses the same verbs.
    odd = db.upsert_object({"kind": "widget", "name": "w"})
    assert db.object(odd)["kind"] == "widget"
    assert len(db.objects_by_kind("widget")) == 1
    with pytest.raises(RuntimeError):
        db.upsert_object({"name": "kindless"})


def test_approval_and_meta(db) -> None:
    uuid = db.upsert_object({"kind": "fiber", "name": "f"})
    db.set_approval(uuid, "approved")
    obj = db.object(uuid)
    assert obj["approval_state"] == "approved"
    assert obj["approved_by"] == "pytest"
    assert obj["approved_host"] == "testhost"
    assert obj["generation"] == 1

    db.set_meta(uuid, "review.note", {"text": "ok", "score": 3})
    obj = db.object(uuid)
    assert obj["approval_state"] == "pending"  # content change resets approval
    assert obj["generation"] == 2
    assert obj["meta"] == {"review.note": {"text": "ok", "score": 3}}
    db.remove_meta(uuid, "review.note")
    assert db.object(uuid)["meta"] == {}


def test_write_transaction_commit_and_rollback(db) -> None:
    with db.write() as w:
        kept = w.upsert_object({"kind": "fiber", "name": "kept"})
    assert db.object(kept) is not None

    with pytest.raises(RuntimeError):
        with db.write() as w:
            w.upsert_object({"kind": "fiber", "name": "dropped"})
            raise RuntimeError("boom")
    assert [o["name"] for o in db.objects_by_kind("fiber")] == ["kept"]


def test_reopen_read_only_and_export(db_path: str) -> None:
    with ObjectStore.create(db_path, author="pytest") as db:
        a = db.upsert_object({"kind": "fiber", "name": "a"})
        b = db.upsert_object({"kind": "patch", "name": "b"})
        db.add_edge({"a": a, "b": b, "kind": "same-surface"})
        exported = db.export_json()
    doc = json.loads(exported)
    assert doc["format"] == "vc_object_metadata"
    assert len(doc["objects"]) == 2 and len(doc["edges"]) == 1

    with ObjectStore.open(db_path, read_only=True) as db:
        assert len(db.objects_by_kind("fiber")) == 1
        with pytest.raises(RuntimeError):
            db.upsert_object({"kind": "fiber", "name": "nope"})

    fresh = str(Path(db_path).with_name("copy.vcobj"))
    with ObjectStore.create(fresh, author="pytest") as db:
        db.import_json(exported)
        assert len(db.objects_by_kind("patch")) == 1






def test_object_edge_uuids_share_namespace(db):
    a = db.upsert_object({"kind": "fiber", "uuid": "abcdefab-1111-4111-8111-111111111111"})
    b = db.upsert_object({"kind": "fiber"})
    before = db.journal_since()
    with pytest.raises(RuntimeError, match="belongs to an object"):
        db.add_edge({"uuid": a.upper(), "kind": "connected", "a": a, "b": b})
    assert db.journal_since() == before and db.edges_of(a) == []
    edge = db.add_edge({"kind": "connected", "a": a, "b": b})
    before = db.journal_since()
    with pytest.raises(RuntimeError, match="belongs to an edge"):
        db.upsert_object({"uuid": edge.upper(), "kind": "fiber"})
    assert db.journal_since() == before and db.object(edge) is None
    db.set_approval(edge.upper(), "approved")
    assert db.edge(edge)["approval_state"] == "approved"
    assert db.object(a)["approval_state"] == "pending"
    assert db.neighbours(a.upper()) == [b]





def test_caught_upsert_failure_preserves_record_details_approval_and_journal(db):
    uid = db.upsert_object(dict(kind="track", name="original", tags=["keep"],
        track=dict(dataset_ref="tracks", external_key="one", bbox=[1, 2, 3, 4, 5, 6])))
    other = db.upsert_object(dict(kind="track", track=dict(dataset_ref="tracks", external_key="two")))
    db.set_approval(uid, "approved")
    before = db.object(uid)
    journal = db.journal_since()
    with db.write() as tx:
        # The uniqueness failure occurs after the old detail row was deleted.
        with pytest.raises(RuntimeError):
            tx.upsert_object(dict(before, name="must roll back",
                track=dict(before["track"], external_key="two")))
        with pytest.raises(RuntimeError):
            tx.upsert_object(dict(kind="fiber", patch={"grid_width": 4}))
        tx.set_meta(other, "continued", True)
    assert db.object(uid) == before
    assert db.object_count("fiber") == 0
    assert db.object(other)["meta"] == {"continued": True}
    assert db.journal_since()[:-1] == journal
    assert [r["uuid"] for r in db.select_objects(bbox=[1, 2, 3, 4, 5, 6], fields=[])] == [uid]


@pytest.mark.parametrize("field", ["position", "direction"])
def test_anchor_vector_validation_matches_python_and_json_import(db, tmp_path, field):
    a = db.upsert_object(dict(kind="patch"))
    b = db.upsert_object(dict(kind="patch"))
    with pytest.raises(RuntimeError, match="3 numbers"):
        db.add_edge(dict(kind="contact", a=a, b=b, a_anchor={field: [1, 2]}))
    db.add_edge(dict(kind="contact", a=a, b=b))
    document = json.loads(db.export_json())
    document["edges"][0]["a_anchor"][field] = [1, 2]
    with ObjectStore.create(str(tmp_path / "import.vcobj"), author="test") as imported:
        with pytest.raises(ValueError, match="3 numbers"):
            imported.import_json(json.dumps(document))
        assert imported.object_count("patch") == 0
        assert imported.journal_since() == []
