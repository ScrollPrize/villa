"""Tests for the ``vc.objects`` bindings (object records and relationships).

Run against the built module with
``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 VC_BUILD_PYTHON=build/python
.venv/bin/python -m pytest python/tests/test_graph.py``.
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
                        "a_anchor": {"index": 3, "position": [1, 2, 3]}})
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


@pytest.mark.parametrize("scale, expected", [([0.25, 0.5], (0.25, 0.5)), (0.5, (0.5, 0.5))])
def test_patch_import_tifxyz_metadata(db, tmp_path, scale, expected):
    patches = tmp_path / "patches"
    patch = patches / "valid"
    patch.mkdir(parents=True)
    uid = str(uuid.uuid4())
    (patch / "meta.json").write_text(json.dumps({"uuid": uid, "type": "seg", "format": "tifxyz",
        "scale": scale, "bbox": [[10, 20, 30], [40, 50, 60]]}))
    bad = patches / "invalid"
    bad.mkdir()
    (bad / "meta.json").write_text(json.dumps({"uuid": str(uuid.uuid4()), "scale": [1, 2, 3]}))
    report = db.import_patches(str(patches))
    assert report["objects_imported"] == 1 and len(report["problems"]) == 1
    record = db.object(uid)
    assert record["patch"]["scale"] == expected
    assert record["patch"]["bbox"] == [10, 20, 30, 40, 50, 60]
    with ObjectStore.create(str(tmp_path / "copy.vcobj"), author="copy") as copy:
        copy.import_json(db.export_json())
        assert copy.object(uid) == record


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "graph.vcobj")


@pytest.fixture()
def db(db_path: str):
    with ObjectStore.create(db_path, author="pytest", hostname="testhost") as handle:
        yield handle


def test_object_and_edge_roundtrip(db) -> None:
    a = db.upsert_object({"kind": "fiber", "name": "a", "tags": ["h"],
                          "fiber": {"family": "H", "control_point_count": 3}})
    b = db.upsert_object({"kind": "patch", "name": "b",
                          "patch": {"grid_width": 10, "grid_height": 20}})
    edge = db.add_edge({"a": a, "b": b, "kind": "same-surface",
                        "meta": {"source": "pytest"},
                        "a_anchor": {"index": 1, "position": [0.0, 0.0, 0.0]}})

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
    assert anchor["index"] == 1

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
    assert doc["format"] == "vc_graph_export"
    assert len(doc["objects"]) == 2 and len(doc["edges"]) == 1

    with ObjectStore.open(db_path, read_only=True) as db:
        assert len(db.objects_by_kind("fiber")) == 1
        with pytest.raises(RuntimeError):
            db.upsert_object({"kind": "fiber", "name": "nope"})

    fresh = str(Path(db_path).with_name("copy.vcobj"))
    with ObjectStore.create(fresh, author="pytest") as db:
        db.import_json(exported)
        assert len(db.objects_by_kind("patch")) == 1


def test_track_list_registration(db, tmp_path: Path) -> None:
    track_file = tmp_path / "tracks.json"
    track_file.write_text(json.dumps([
        {"dataset_ref": "spiral://s1", "external_key": "t1",
         "family": "H", "point_count": 5},
        {"external_key": "no-dataset"},
    ]))
    report = db.import_track_list(str(track_file))
    assert report["objects_imported"] == 1
    assert len(report["problems"]) == 1
    first = db.objects_by_kind("track")
    assert first[0]["track"]["external_key"] == "t1"

    again = db.import_track_list(str(track_file))
    assert again["objects_imported"] == 0  # stable identity, no-op
    assert db.objects_by_kind("track")[0]["uuid"] == first[0]["uuid"]


@pytest.mark.parametrize("line, expected", [
    ([[0, 0, 0], [0, 3, 4], [0, 3, 4], [12, 3, 4]], 17),
    (None, 13), ([], 13), ([[3, 2, 1]], 0),
])
def test_fiber_import_lengths_and_legacy_backfill(db, tmp_path, line, expected):
    import uuid
    fibers = tmp_path / "fibers"
    fibers.mkdir()
    uid = str(uuid.uuid4())
    root = {"uuid": uid, "control_points": [{"position": [0, 0, 0]}, {"position": [12, 3, 4]}],
            "hv_classification": {"automatic_tag": "H", "manual_tag": "V"}}
    if line is not None:
        root["line_points"] = line
    (fibers / "fiber.json").write_text(json.dumps(root))
    report = db.import_fibers(str(fibers))
    assert report["problems"] == [] and report["objects_imported"] == 1
    obj = db.object(uid)
    assert obj["fiber"]["length"] == expected
    assert obj["fiber"]["family"] == "V" and obj["fiber"]["manual_family"]
    # Keep the matching file checksum to exercise the old importer no-op path.
    obj["fiber"]["length"] = 0
    db.upsert_object(obj)
    db.set_approval(uid, "approved")
    before = db.object(uid)
    assert db.import_fibers(str(fibers))["problems"] == []
    after = db.object(uid)
    assert after["fiber"] == dict(before["fiber"], length=expected)
    assert after["generation"] == before["generation"] + (expected != 0)
    assert after["approval_state"] == ("pending" if expected != 0 else "approved")
    if expected != 0:
        assert after["approved_by"] is None and after["approved_generation"] is None
    snapshot = db.export_json()
    journal = db.journal_since(0)
    assert db.import_fibers(str(fibers))["objects_imported"] == 0
    assert db.export_json() == snapshot and db.journal_since(0) == journal


def test_fiber_import_rejects_invalid_dense_geometry(db, tmp_path):
    import uuid
    fibers = tmp_path / "fibers"
    fibers.mkdir()
    (fibers / "bad.json").write_text(json.dumps({"uuid": str(uuid.uuid4()),
        "line_points": [[0, 0, 0], [1, 2]], "control_points": [[0, 0, 0], [1, 0, 0]]}))
    report = db.import_fibers(str(fibers))
    assert report["problems"] and db.objects_by_kind("fiber") == []


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


def test_legacy_fiber_refresh_preserves_annotations_and_reconciles_branches(db, tmp_path):
    source = tmp_path / "fibers"; source.mkdir()
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    def branch(target, xyz):
        return {"control_point_index": 0, "branch_file": target, "branch_control_point_index": 0,
                "control_point_position": xyz, "branch_control_point_position": [3, 4, 5], "pending": False}
    first = {"uuid": a, "control_points": [[1, 2, 3]], "line_points": [[1, 2, 3], [4, 5, 6]],
             "tags": ["source"], "branches": [branch("b.json", [1, 2, 3])]}
    second = {"uuid": b, "control_points": [[3, 4, 5]], "branches": []}
    (source / "a.json").write_text(json.dumps(first)); (source / "b.json").write_text(json.dumps(second))
    assert db.import_fibers(str(source))["edges_imported"] == 1
    edge = db.edges_of(a)[0]
    db.set_meta(a, "review_note", "keep")
    db.set_meta(edge["uuid"], "review_note", "edge note")
    annotated = db.object(a); annotated["tags"] = ["database"]
    db.upsert_object(annotated)
    db.set_approval(edge["uuid"], "approved")
    first["line_points"][1] = [8, 2, 3]
    first["branches"][0]["control_point_position"] = [2, 2, 3]
    first["tags"] = ["changed-source"]
    (source / "a.json").write_text(json.dumps(first))
    assert db.import_fibers(str(source))["objects_imported"] == 1
    assert db.object(a)["meta"] == {"review_note": "keep"} and db.object(a)["tags"] == ["database"]
    refreshed = db.edge(edge["uuid"])
    anchor = refreshed["a_anchor"] if refreshed["a"] == a else refreshed["b_anchor"]
    assert anchor["position"] == (2, 2, 3)
    assert refreshed["approval_state"] == "pending" and refreshed["meta"]["review_note"] == "edge note"
    # Omitted/unreadable target files cannot establish a removed branch.
    first["branches"] = []; (source / "a.json").write_text(json.dumps(first))
    (source / "b.json").write_text("{")
    assert db.import_fibers(str(source))["problems"]
    assert db.edge(edge["uuid"]) is not None
    (source / "b.json").write_text(json.dumps(second))
    db.import_fibers(str(source))
    assert db.edge(edge["uuid"]) is None
    before = db.journal_since()
    assert db.import_fibers(str(source))["objects_imported"] == 0
    assert db.journal_since() == before


def test_legacy_import_does_not_refresh_managed_geometry_from_files(db, tmp_path):
    source = tmp_path / "fibers"; source.mkdir()
    uid = str(uuid.uuid4())
    data = {"uuid": uid, "type": "vc3d_fiber", "control_points": [], "line_points": [[1, 2, 3]]}
    file = source / "fiber.json"; file.write_text(json.dumps(data))
    assert db.ingest_geometry_source("fiber", file)["ingested"] == 1
    db.set_meta(uid, "review_note", "keep")
    before, journal = db.object(uid), db.journal_since()
    data["line_points"] = [[9, 8, 7]]; file.write_text(json.dumps(data))
    report = db.import_fibers(str(source))
    assert "database-owned" in report["problems"][0]
    assert db.object(uid) == before and db.journal_since() == journal
