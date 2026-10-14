"""Tests for the ``vc.graph`` bindings (unified metadata graph DB).

Run with the vesuvius interpreter that has the editable
volume-cartographer install, e.g.
``VC_BUILD_PYTHON=build/python .venv/bin/python -m pytest python/tests/test_graph.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import vc
if os.environ.get("VC_BUILD_PYTHON"):
    vc.__path__.insert(0, str(Path(os.environ["VC_BUILD_PYTHON"]) / "vc"))
graph = pytest.importorskip("vc.graph")
GraphDb = graph.GraphDb


@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "graph.vcgr")


@pytest.fixture()
def db(db_path: str):
    with GraphDb.create(db_path, author="pytest", hostname="testhost") as handle:
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
    with GraphDb.create(db_path, author="pytest") as db:
        a = db.upsert_object({"kind": "fiber", "name": "a"})
        b = db.upsert_object({"kind": "patch", "name": "b"})
        db.add_edge({"a": a, "b": b, "kind": "same-surface"})
        exported = db.export_json()
    doc = json.loads(exported)
    assert doc["format"] == "vc_graph_export"
    assert len(doc["objects"]) == 2 and len(doc["edges"]) == 1

    with GraphDb.open(db_path, read_only=True) as db:
        assert len(db.objects_by_kind("fiber")) == 1
        with pytest.raises(RuntimeError):
            db.upsert_object({"kind": "fiber", "name": "nope"})

    fresh = str(Path(db_path).with_name("copy.vcgr"))
    with GraphDb.create(fresh, author="pytest") as db:
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
