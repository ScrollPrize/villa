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
    assert db.object(uid) == dict(before, fiber=dict(before["fiber"], length=expected))
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


def _dbm_track_fixture(tmp_path):
    import dbm.dumb
    import pickle
    h, v = _crossing_lines()
    path = tmp_path / "tracks.dbm"
    with dbm.dumb.open(str(path), "n") as source:
        source[b"h:0"] = pickle.dumps([h])
        source[b"vy:0"] = pickle.dumps([v])
    packed = Path(str(path) + ".vctracks")
    _packed_tracks(packed, [h, v], [0, 1], [0, 1 << 32])
    signature = sorted([
        [f.name, f.stat().st_size, f.stat().st_mtime_ns]
        for suffix in ("", ".db", ".dat", ".dir", ".bak", ".pag")
        if (f := Path(str(path) + suffix)).is_file()
    ])
    (packed / "metadata.json").write_text(json.dumps({
        "version": 1, "source_db_signature": signature,
        "track_count": 2, "point_count": len(h) + len(v),
    }))
    return path, packed


@pytest.mark.parametrize("with_unreadable_crossings", [False, True])
def test_dbm_entry_registers_and_computes_without_crossings_npz(db, tmp_path, with_unreadable_crossings):
    path, packed = _dbm_track_fixture(tmp_path)
    sidecar = Path(str(path) + ".crossings.npz")
    if with_unreadable_crossings:
        sidecar.write_bytes(b"deliberately invalid NPZ: must never be read")
    report = db.link_tracks_to_tracks(str(path), workers=2)
    assert report["problems"] == []
    assert report["objects_considered"] == 2 and report["edges_created"] == 1
    objects = db.objects_by_kind("track")
    assert {o["track"]["external_key"] for o in objects} == {"0", str(1 << 32)}
    assert all(o["track"]["dataset_ref"] == str(packed.resolve()) for o in objects)
    assert all(o["track"]["point_count"] == 31 for o in objects)
    assert all(o["track"]["length"] == 30 for o in objects)
    assert {o["track"]["external_key"]: o["track"]["family"] for o in objects} == {
        "0": "horizontal", str(1 << 32): "vertical"}
    edge = db.edges_of(objects[0]["uuid"])[0]
    assert edge["a_anchor"]["position"] == edge["b_anchor"]["position"] == (15, 10, 7)
    before = db.export_json()
    # Packed path and DBM path resolve to the same graph identities.
    repeat = db.link_tracks_to_tracks(str(packed))
    assert repeat["edges_created"] == 0 and repeat["edges_skipped_existing"] == 1
    assert db.export_json() == before
    if with_unreadable_crossings:
        assert sidecar.read_bytes() == b"deliberately invalid NPZ: must never be read"
    else:
        assert not sidecar.exists()


def test_dbm_entry_subset_and_existing_relative_identity(db, tmp_path):
    path, packed = _dbm_track_fixture(tmp_path)
    a = _register_tracks(db, packed.name, [0], [""])[0]
    db.set_approval(a, "approved")
    before = db.object(a)
    first = db.link_tracks_to_tracks(str(path), source_ids=[0])
    assert first["problems"] == [] and first["candidates_found"] == 0
    assert len(db.objects_by_kind("track")) == 1
    assert db.object(a)["generation"] == 1
    expected = dict(before, track=dict(before["track"], length=30.0, family="horizontal"))
    assert db.object(a) == expected
    second = db.link_tracks_to_tracks(str(path))
    assert second["problems"] == [] and second["edges_created"] == 1
    assert len(db.objects_by_kind("track")) == 2
    assert db.object(a)["generation"] == 1
    assert all(o["track"]["dataset_ref"] == packed.name for o in db.objects_by_kind("track"))


@pytest.mark.parametrize("registered", [False, True])
def test_track_polyline_lengths_include_bends_repeats_and_degenerate_tracks(db, tmp_path, registered):
    import numpy as np
    packed = tmp_path / "lengths.vctracks"
    # Length 5 + 0 + 12 = 17, although the endpoints are only 13 apart.
    bent = np.array([[0, 0, 0], [0, 3, 4], [0, 3, 4], [12, 3, 4]])
    diagonal = np.array([[0, 0, 0], [1, 1, 1]])
    _packed_tracks(packed, [bent, diagonal, np.array([[3, 2, 1]]), np.empty((0, 3))],
                   [0, 0, 0, -1], [10, 20, 30, 40])
    if registered:
        ids = _register_tracks(db, packed.name, [10, 20, 30, 40], [""] * 4)
        db.set_approval(ids[0], "approved")
    report = db.link_tracks_to_tracks("" if registered else str(packed))
    assert report["problems"] == []
    objects = db.objects_by_kind("track")
    lengths = {o["track"]["external_key"]: o["track"]["length"] for o in objects}
    assert lengths == pytest.approx({"10": 17, "20": 3 ** 0.5, "30": 0, "40": 0})
    assert {o["track"]["external_key"]: o["track"]["family"] for o in objects} == {
        "10": "horizontal", "20": "horizontal", "30": "horizontal", "40": ""}
    if registered:
        assert db.object(ids[0])["approval_state"] == "approved"
        assert db.object(ids[0])["generation"] == 1
    before = db.export_json()
    assert db.link_tracks_to_tracks(str(packed))["problems"] == []
    assert db.export_json() == before


def test_dbm_entry_validation_does_not_register_partial_inputs(db, tmp_path):
    import numpy as np
    path, packed = _dbm_track_fixture(tmp_path)
    missing = db.link_tracks_to_tracks(str(path), source_ids=[0, 999999])
    assert missing["problems"] and db.objects_by_kind("track") == []
    coords = np.fromfile(packed / "coordinates.i32", dtype="<i4")
    coords[0] = -1
    coords.tofile(packed / "coordinates.i32")
    invalid = db.link_tracks_to_tracks(str(path))
    assert invalid["problems"] and db.objects_by_kind("track") == []
    # Mutating a backing file invalidates the packed geometry fingerprint.
    backing = Path(str(path) + ".dat")
    backing.write_bytes(backing.read_bytes() + b"changed")
    stale = db.link_tracks_to_tracks(str(path))
    assert any("stale" in p for p in stale["problems"])
    assert db.objects_by_kind("track") == []


def test_dbm_entry_missing_geometry_is_explicit(db, tmp_path):
    report = db.link_tracks_to_tracks(str(tmp_path / "missing.dbm"))
    assert any("convert_track_store.py" in p for p in report["problems"])
    assert db.objects_by_kind("track") == []


def test_track_patch_binding_validates_parameters_and_preserves_failed_inputs(db, tmp_path):
    path, _ = _dbm_track_fixture(tmp_path)
    patches = tmp_path / "patches"
    patches.mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        db.link_tracks_to_patches(str(path), str(patches), "missing-ct", track_scale=0)
    with pytest.raises((ValueError, RuntimeError)):
        db.link_tracks_to_patches(str(path), str(patches), "missing-ct", strip_weights=[1, 2])
    with pytest.raises(RuntimeError, match="unknown parameter"):
        db.link_tracks_to_patches(str(path), str(patches), "missing-ct", fiber_scale=1)
    report = db.link_tracks_to_patches(str(path), str(patches), "missing-ct", source_ids=[0])
    assert report["problems"]
    assert db.objects_by_kind("track") == []
    assert db.objects_by_kind("patch") == []
