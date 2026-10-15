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


def _packed_tracks(path, tracks, families, source_ids):
    """Small VCTRK01 fixtures; no dependency on Spiral or a source DBM."""
    import struct
    import numpy as np
    path.mkdir(exist_ok=True)
    points = sum(len(track) for track in tracks)
    (path / "header.bin").write_bytes(struct.pack(
        "<8sIIQQ4Q", b"VCTRK01\0", 1, 64, len(tracks), points, 0, 0, 0, 0))
    coords = np.concatenate(tracks) if points else np.empty((0, 3))
    coords.astype("<i4").tofile(path / "coordinates.i32")
    np.asarray([0, *np.cumsum([len(t) for t in tracks])], dtype="<i8").tofile(path / "offsets.i64")
    np.asarray(source_ids, dtype="<u8").tofile(path / "source_ids.u64")
    np.asarray(families, dtype="i1").tofile(path / "family_codes.i8")


def _register_tracks(db, dataset, ids, families):
    return [db.upsert_object({
        "kind": "track", "storage_kind": "dataset", "storage_ref": str(dataset),
        "track": {"dataset_ref": str(dataset), "external_key": str(id), "family": family},
    }) for id, family in zip(ids, families)]


def _crossing_lines():
    import numpy as np
    h = np.array([[7, 10, x] for x in range(31)], dtype=np.int32)
    v = np.array([[7, y, 15] for y in range(31)], dtype=np.int32)
    return h, v


def test_track_linking_exact_voxels_families_angles_and_identity(db, tmp_path):
    import numpy as np
    h, v = _crossing_lines()
    h = np.insert(h, 16, h[15], axis=0)  # repeated voxel uses its first local index
    near = v + np.array([1, 0, 0])
    ids = [11, 15, 20, 25, 30, 1 << 40]
    packed = tmp_path / "tiny.vctracks"
    _packed_tracks(packed, [h, v, near, h, v, v[10:11]], [0, 1, 1, 1, -1, 1], ids)
    uuids = _register_tracks(db, packed.name, ids, ["horizontal", "vertical", "vertical", "vertical", "", "vertical"])
    report = db.link_tracks_to_tracks(workers=3)
    assert report["problems"] == []
    assert report["objects_considered"] == 6
    assert report["edges_created"] == report["candidates_found"] == 1
    edge = db.edges_of(uuids[0])[0]
    assert {edge["a"], edge["b"]} == set(uuids[:2])
    assert edge["kind"] == "crossing" and edge["approval_state"] == "pending"
    anchors = {edge["a"]: edge["a_anchor"], edge["b"]: edge["b_anchor"]}
    assert anchors[uuids[0]]["index"] == 15
    assert anchors[uuids[1]]["index"] == 10
    assert anchors[uuids[0]]["position"] == (15, 10, 7)  # ZYX -> XYZ
    assert anchors[uuids[0]]["direction"] == (1, 0, 0)
    assert anchors[uuids[1]]["direction"] == (0, 1, 0)
    assert edge["meta"]["track_crossing"]["clearance_voxels"] == 10
    endpoints = edge["meta"]["track_crossing"]["endpoints"]
    assert endpoints[uuids[0]] == {"source_id": "11", "arclength_voxels": 15}
    assert endpoints[uuids[1]] == {"source_id": "15", "arclength_voxels": 10}
    assert edge["meta"]["detector"]["name"] == "track-track"
    repeat = db.link_tracks_to_tracks(workers=1)
    assert repeat["edges_created"] == repeat["stale_removed"] == 0
    assert repeat["edges_skipped_existing"] == 1
    assert db.edges_of(uuids[0])[0]["uuid"] == edge["uuid"]


def test_track_linking_subset_stale_edges_and_dataset_isolation(db, tmp_path):
    h, v = _crossing_lines()
    uuids = []
    for name in ("a.vctracks", "b.vctracks"):
        _packed_tracks(tmp_path / name, [h, v], [0, 1], [50, 60])
        uuids += _register_tracks(db, name, [50, 60], ["horizontal", "vertical"])
    assert db.link_tracks_to_tracks()["edges_created"] == 2
    assert db.link_tracks_to_tracks(track_uuids=[uuids[0]])["stale_removed"] == 0
    # Valid no-crossing rerun replaces only its own pending edge in this dataset.
    _packed_tracks(tmp_path / "a.vctracks", [h, v + [1, 0, 0]], [0, 1], [50, 60])
    report = db.link_tracks_to_tracks(track_uuids=uuids[:2])
    assert report["problems"] == [] and report["stale_removed"] == 1
    assert db.edges_of(uuids[0]) == []
    assert len(db.edges_of(uuids[2])) == 1
    # Reviewed edges survive valid empty reruns.
    second = db.edges_of(uuids[2])[0]
    db.set_approval(second["uuid"], "approved")
    _packed_tracks(tmp_path / "b.vctracks", [h, v + [1, 0, 0]], [0, 1], [50, 60])
    assert db.link_tracks_to_tracks()["stale_removed"] == 0
    assert db.edge(second["uuid"])["approval_state"] == "approved"


@pytest.mark.parametrize("damage", ["header", "offsets", "coordinates", "ids", "family", "missing_id", "negative_voxel", "large_voxel"])
def test_track_linking_bad_input_is_atomic(db, tmp_path, damage):
    import numpy as np
    h, v = _crossing_lines()
    packed = tmp_path / "tracks.vctracks"
    _packed_tracks(packed, [h, v], [0, 1], [50, 60])
    a, b = _register_tracks(db, packed.name, [50, 60], ["horizontal", "vertical"])
    assert db.link_tracks_to_tracks()["edges_created"] == 1
    original = db.edges_of(a)
    if damage == "header":
        (packed / "header.bin").write_bytes(b"bad")
    elif damage == "offsets":
        np.array([0, -1, 62], dtype="<i8").tofile(packed / "offsets.i64")
    elif damage == "coordinates":
        (packed / "coordinates.i32").write_bytes(b"bad")
    elif damage == "ids":
        np.array([50, 50], dtype="<u8").tofile(packed / "source_ids.u64")
    elif damage == "family":
        np.array([0, 0], dtype="i1").tofile(packed / "family_codes.i8")
    elif damage == "missing_id":
        np.array([50, 61], dtype="<u8").tofile(packed / "source_ids.u64")
    else:
        h[0, 0] = -1 if damage == "negative_voxel" else (1 << 20)
        _packed_tracks(packed, [h, v], [0, 1], [50, 60])
    report = db.link_tracks_to_tracks()
    assert report["problems"] and report["edges_created"] == report["stale_removed"] == 0
    assert db.edges_of(a) == original


def test_track_linking_empty_and_invalid_selection(db, tmp_path):
    import numpy as np
    assert db.link_tracks_to_tracks()["problems"] == []
    packed = tmp_path / "empty.vctracks"
    _packed_tracks(packed, [np.empty((0, 3))], [-1], [100])
    _register_tracks(db, packed.name, [100], [""])
    assert db.link_tracks_to_tracks()["problems"] == []
    assert db.link_tracks_to_tracks(track_uuids=["missing"])["problems"]
    assert db.link_tracks_to_tracks(dataset_ref="missing")["problems"]
    with pytest.raises((ValueError, RuntimeError)):
        db.link_tracks_to_tracks(min_angle_degrees=float("nan"))
    with pytest.raises((ValueError, RuntimeError)):
        db.link_tracks_to_tracks(workers=0)
    with pytest.raises(RuntimeError, match="unknown parameter"):
        db.link_tracks_to_tracks(misspelled=3)


def test_track_linking_angle_gate_and_atomic_multiple_datasets(db, tmp_path):
    import numpy as np
    h, _ = _crossing_lines()
    h[:, 1] = 20
    diagonal = np.array([[7, x + 5, x] for x in range(31)])
    packed = tmp_path / "angle.vctracks"
    _packed_tracks(packed, [h, diagonal], [0, 1], [5, 8])
    a, b = _register_tracks(db, packed.name, [5, 8], ["horizontal", "vertical"])
    assert db.link_tracks_to_tracks()["edges_created"] == 1
    stricter = db.link_tracks_to_tracks(min_angle_degrees=60, tangent_radius_voxels=1)
    assert stricter["candidates_found"] == 0 and stricter["stale_removed"] == 1
    _register_tracks(db, "z-missing.vctracks", [9], ["vertical"])
    # The first dataset yields a candidate, but a later load failure aborts all writes.
    report = db.link_tracks_to_tracks()
    assert report["problems"] and report["edges_created"] == 0
    assert db.edges_of(a) == []
