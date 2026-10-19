"""Tests for the ``vc.graph`` bindings (unified metadata graph DB).

Run with the vesuvius interpreter that has the editable
volume-cartographer install, e.g.
``VC_BUILD_PYTHON=build/python .venv/bin/python -m pytest python/tests/test_graph.py``.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from pathlib import Path

import pytest

import vc
if os.environ.get("VC_BUILD_PYTHON"):
    vc.__path__.insert(0, str(Path(os.environ["VC_BUILD_PYTHON"]) / "vc"))
graph = pytest.importorskip("vc.graph")
GraphDb = graph.GraphDb


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
    with GraphDb.create(str(tmp_path / "restored.vcgr"), author="importer", hostname="elsewhere") as restored:
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


def test_schema_v1_migration_preserves_surviving_approval(tmp_path):
    path = tmp_path / "legacy.vcgr"
    uid = uuid.uuid4()
    with sqlite3.connect(path) as connection:
        connection.executescript((Path(__file__).parent / "data" / "graph_v1.sql").read_text())
        connection.execute("""INSERT INTO objects(uuid,kind,generation,created_by,created_at,created_host,
            changed_by,changed_at,changed_host,approval_state,approved_by,approved_at,approved_host,approved_generation)
            VALUES(?,'patch',7,'creator','created','original','editor','changed','workstation',
                   'approved','reviewer','reviewed','reviewhost',7)""", (uid.bytes,))
        connection.execute("INSERT INTO patches(object_id,scale) VALUES(1,0.25)")
        connection.execute("""INSERT INTO journal(record_uuid,generation,author,hostname,timestamp,change_kind)
            VALUES(?,7,'reviewer','reviewhost','reviewed','approval')""", (uid.bytes,))
    with pytest.raises(RuntimeError, match="open read-write once"):
        GraphDb.open(str(path), read_only=True)
    with GraphDb.open(str(path), author="migrator") as migrated:
        record = migrated.object(str(uid))
        assert record["patch"]["scale"] == (0.25, 0.25)
        assert record["generation"] == 7 and record["approved_by"] == "reviewer"
        history = migrated.journal_since()
        assert history[0]["approval_state"] is None  # no invented historical decision
        assert history[1]["change_kind"] == "migration_snapshot"
        assert history[1]["author"] == "migrator"
        assert history[1]["approval_state"] == "approved"
        assert history[1]["approved_generation"] == 7
        migrated.set_meta(str(uid), "edited", True)
        assert migrated.journal_since()[1] == history[1]
    with GraphDb.open(str(path), read_only=True) as reopened:
        assert reopened.object(str(uid))["generation"] == 8
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        connection.execute("PRAGMA user_version=99")
    for readonly in (True, False):
        with pytest.raises(RuntimeError, match="unsupported schema version"):
            GraphDb.open(str(path), author="tester", read_only=readonly)


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
    with GraphDb.create(str(tmp_path / "copy.vcgr"), author="copy") as copy:
        copy.import_json(db.export_json())
        assert copy.object(uid) == record


def test_export_version_and_legacy_scalar_import(db):
    uid = str(uuid.uuid4())
    legacy = {"format": "vc_graph_export", "version": 1, "objects": [
        {"uuid": uid, "kind": "patch", "data_generation": 5, "patch": {"scale": 0.5},
         "approval_state": "approved"}], "edges": []}
    db.import_json(json.dumps(legacy))
    record = db.object(uid)
    assert record["patch"]["scale"] == (0.5, 0.5)
    assert record["generation"] == record["data_generation"] == 1
    assert record["approved_by"] is None
    snapshot = json.loads(db.export_json())
    snapshot["version"] = 99
    with pytest.raises(RuntimeError, match="unsupported export version"):
        db.import_json(json.dumps(snapshot))


def _write_pcl_patch(path, rows, cols, z, scale=(1.0, 1.0)):
    import numpy as np
    import tifffile
    path.mkdir(parents=True)
    yy, xx = np.mgrid[:rows, :cols]
    uid = str(uuid.uuid4())
    (path / "meta.json").write_text(json.dumps({"uuid": uid, "type": "seg", "format": "tifxyz", "scale": scale}))
    for axis, values in zip("xyz", (xx + 100, yy + 100, np.full_like(xx, z))):
        tifffile.imwrite(path / f"{axis}.tif", values.astype(np.float32))
    return uid


def _write_pcl(path, collections):
    path.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": collections}))


def _pcl(name, points):
    return {"uuid": str(uuid.uuid4()), "name": name, "metadata": {},
            "points": {str(key): {"p": xyz, "wind_a": wind} for key, xyz, wind in points}}


def test_pcl_patch_selection_anchors_identity_and_rerun(db, tmp_path):
    patches = tmp_path / "patches"
    large = _write_pcl_patch(patches / "large", 12, 12, 100)
    small = _write_pcl_patch(patches / "small", 4, 4, 101)
    other = _write_pcl_patch(patches / "other__name", 5, 5, 103)
    general = _pcl("general", [(42, [202, 202, 202], 3.5)])
    between = _pcl("between_patches__small__other__name", [(19, [202, 202, 202], None)])
    fallback = _pcl("between_patches__small__missing", [(9, [202, 202, 202], 2)])
    file = tmp_path / "points.json"
    _write_pcl(file, {"8": general, "9": between, "10": fallback})
    original = file.read_bytes()
    report = db.link_point_collections_to_patches(str(file), str(patches), point_scale=0.5)
    assert report["edges_created"] == 3
    assert len(db.objects_by_kind("point_collection")) == 3
    for collection, expected in [(general, large), (between, small), (fallback, large)]:
        uid = collection["uuid"]
        edge, = db.edges_of(uid)
        assert {edge["a"], edge["b"]} == {uid, expected}
        point = edge["a_anchor"] if edge["a"] == uid else edge["b_anchor"]
        foot = edge["b_anchor"] if edge["a"] == uid else edge["a_anchor"]
        assert point["index"] == int(next(iter(collection["points"])))
        assert point["position"] == (202, 202, 202)
        assert foot["position"] == (101, 101, 101 if expected == small else 100)
        evidence = edge["meta"]["point_collection_patch"]
        assert evidence["ij"] == pytest.approx([1, 1])
        assert evidence["distance"] == pytest.approx(0 if expected == small else 1)
        assert evidence["winding_annotation"] == next(iter(collection["points"].values()))["wind_a"]
        assert edge["approval_state"] == "pending"
        assert db.resolve_storage_path(uid) == file
    before = db.export_json(), db.journal_since()
    assert db.link_point_collections_to_patches(str(file), str(patches), point_scale=0.5)["edges_created"] == 0
    assert (db.export_json(), db.journal_since()) == before
    assert file.read_bytes() == original
    renamed = file.with_name("renamed.json")
    file.rename(renamed)
    assert db.resolve_storage_path(general["uuid"]) == renamed


def test_pcl_support_window_and_absolute_role(db, tmp_path):
    patches = tmp_path / "patches"
    _write_pcl_patch(patches / "patch", 8, 8, 100)
    # Numeric ID order is 2,10,20,30. Last point has insufficient support;
    # the first two support each other in the rounded-up three-point window.
    collection = _pcl("chain", [(10, [102, 102, 100], 1), (2, [101, 101, 100], 1),
                                (20, [200, 200, 100], 1), (30, [103, 103, 100], 1)])
    collection["metadata"]["winding_is_absolute"] = True  # ignored by fit_spiral for regular filenames
    file = tmp_path / "points.json"
    _write_pcl(file, {"1": collection})
    report = db.link_point_collections_to_patches(str(file), str(patches), window_points=2, window_min_points=2)
    assert report["edges_created"] == 2
    ids = {e["a_anchor" if e["a"] == collection["uuid"] else "b_anchor"]["index"] for e in db.edges_of(collection["uuid"])}
    assert ids == {2, 10}
    absolute = _pcl("absolute", [(10, [102, 102, 100], 1), (20, [200, 200, 100], 2), (30, [103, 103, 100], 3)])
    absolute_file = tmp_path / "abs_winding.json"
    _write_pcl(absolute_file, {"1": absolute})
    report = db.link_point_collections_to_patches(str(absolute_file), str(patches), window_points=3, window_min_points=3)
    assert report["edges_created"] == 2
    overridden = db.link_point_collections_to_patches(str(absolute_file), str(patches), window_points=3,
                                                   window_min_points=3, absolute_winding=False)
    assert overridden["stale_removed"] == 2
    assert db.edges_of(collection["uuid"])  # rerunning another file never removes these links


def test_pcl_area_uses_both_grid_scales_and_preserves_source_coordinates(db, tmp_path):
    patches = tmp_path / "patches"
    _write_pcl_patch(patches / "large_grid", 12, 12, 100)
    # 4 / (.1 * .2) > 121 / (1 * 1): grid dimensions alone rank these backwards.
    winner = _write_pcl_patch(patches / "large_area", 3, 3, 101, scale=(0.1, 0.2))
    collection = _pcl("scaled", [(10000000000, [202, 202, 201], -2)])
    file = tmp_path / "points.json"
    _write_pcl(file, {"1": collection})
    report = db.link_point_collections_to_patches(str(file), str(patches), patch_scale=2, tolerance=1)
    assert report["edges_created"] == 1  # tolerance equality is included
    edge, = db.edges_of(collection["uuid"])
    assert winner in (edge["a"], edge["b"])
    anchor = edge["a_anchor"] if edge["a"] == winner else edge["b_anchor"]
    assert anchor["position"] == (101, 101, 101)
    evidence = edge["meta"]["point_collection_patch"]
    assert evidence["ij"] == pytest.approx([1, 1])
    assert evidence["distance"] == pytest.approx(1)
    assert db.object(winner)["patch"]["scale"] == pytest.approx((0.1, 0.2))


def test_pcl_stale_review_protection_and_invalid_input_atomicity(db, tmp_path):
    patches = tmp_path / "patches"
    _write_pcl_patch(patches / "patch", 8, 8, 100)
    collection = _pcl("reviewed", [(1, [101, 101, 100], 1), (2, [102, 102, 100], 1)])
    file = tmp_path / "points.json"
    _write_pcl(file, {"1": collection})
    db.link_point_collections_to_patches(str(file), str(patches))
    edge = db.edges_of(collection["uuid"])[0]
    db.set_approval(edge["uuid"], "approved")
    collection["points"] = {}
    _write_pcl(file, {"1": collection})
    report = db.link_point_collections_to_patches(str(file), str(patches))
    assert report["stale_removed"] == 1
    assert db.edges_of(collection["uuid"]) == [db.edge(edge["uuid"])]
    before = db.export_json(), db.journal_since()
    good = _pcl("new", [(1, [101, 101, 100], 1)])
    broken = _pcl("bad", [(1, [101, 101, 100], 1)])
    del broken["uuid"]
    _write_pcl(file, {"1": good, "2": broken})
    with pytest.raises(RuntimeError, match="source UUID"):
        db.link_point_collections_to_patches(str(file), str(patches))
    assert (db.export_json(), db.journal_since()) == before
    good["points"]["1"]["p"].append(999)
    _write_pcl(file, {"1": good})
    with pytest.raises(RuntimeError, match="three XYZ coordinates"):
        db.link_point_collections_to_patches(str(file), str(patches))
    assert (db.export_json(), db.journal_since()) == before
    for kwargs in ({"tolerance": 0}, {"point_scale": -1}, {"window_points": 1, "window_min_points": 2}):
        with pytest.raises(ValueError):
            db.link_point_collections_to_patches(str(file), str(patches), **kwargs)
    assert (db.export_json(), db.journal_since()) == before


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
    after = db.object(a)
    assert after["generation"] == before["generation"] + 1
    assert after["track"] == dict(before["track"], length=30.0, family="horizontal")
    assert after["approval_state"] == "pending"
    assert after["approved_generation"] is None
    second = db.link_tracks_to_tracks(str(path))
    assert second["problems"] == [] and second["edges_created"] == 1
    assert len(db.objects_by_kind("track")) == 2
    assert db.object(a) == after
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
        assert db.object(ids[0])["approval_state"] == "pending"
        assert db.object(ids[0])["generation"] == 2
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


def _geometry(kind="fiber", value=1.0):
    import struct
    return {"kind": kind, "arrays": [
        {"name": "control_points", "dtype": "<f8", "shape": [0, 3], "data": b""},
        {"name": "line_points", "dtype": "<f8", "shape": [1, 3],
         "data": struct.pack("<ddd", value, 2.0, 3.0)}]}


def test_geometry_revisions_approval_history_and_source_independence(db, db_path):
    a = db.upsert_object({"kind": "fiber", "tags": ["keep"]})
    b = db.upsert_object({"kind": "fiber"})
    edge = db.add_edge({"kind": "connected", "a": a, "b": b})
    db.set_approval(edge, "approved")
    first = db.put_geometry(a, _geometry())
    assert db.current_geometry(a)["uuid"] == first
    assert db.edge_geometry_state(edge)["stale"]
    assert db.edge(edge)["approval_state"] == "pending"
    with pytest.raises(RuntimeError, match="revalidate"):
        db.set_approval(edge, "approved")
    db.revalidate_edge_geometry(edge)
    db.set_approval(edge, "approved")
    db.set_approval(a, "approved")
    before = db.journal_since()
    assert db.put_geometry(a, _geometry(), expected_revision=first) == first
    assert db.journal_since() == before
    assert db.object(a)["approval_state"] == "approved"
    second = db.put_geometry(a, _geometry(value=9), expected_revision=first)
    assert db.geometry_revision(second)["parent_uuid"] == first
    assert db.object(a)["approval_state"] == db.edge(edge)["approval_state"] == "pending"
    assert db.edge_geometry_state(edge)["stale"]
    assert first in db.edge_geometry_state(edge).values()
    assert db.read_geometry_array(first, "line_points") == _geometry()["arrays"][1]["data"]
    assert db.read_geometry_array(second, "line_points") == _geometry(value=9)["arrays"][1]["data"]
    assert len(db.geometry_history(a)) == 2
    db.set_meta(a, "annotation", "changed")
    assert db.current_geometry(a)["uuid"] == second
    assert db.object(a)["tags"] == ["keep"]
    assert not db.sweep_external_data()
    with pytest.raises(RuntimeError, match="SQLite backup"):
        db.export_json()
    with GraphDb.open(db_path, read_only=True) as read:
        assert read.read_geometry_array(first, "line_points") == db.read_geometry_array(first, "line_points")
        with pytest.raises(RuntimeError, match="read-only"):
            read.put_geometry(a, _geometry(), expected_revision=second)


def test_geometry_chunk_ranges_dedup_and_immutable_sql(db, db_path):
    a = db.upsert_object({"kind": "fiber"})
    b = db.upsert_object({"kind": "fiber"})
    data = _geometry()
    # A separate opaque byte array exercises ranges across the 1 MiB boundary.
    payload = bytes(range(256)) * 10000
    data["arrays"].append({"name": "auxiliary", "dtype": "|u1", "shape": [len(payload)], "data": payload})
    first = db.put_geometry(a, data)
    db.put_geometry(b, data)
    assert db.read_geometry_array(first, "auxiliary", 1048570, 23) == payload[1048570:1048593]
    assert db.read_geometry_array(first, "auxiliary", len(payload), 0) == b""
    assert db.read_geometry_array(first, "control_points") == b""
    for offset, count in ((-1, 1), (0, -1), (len(payload), 1)):
        with pytest.raises(IndexError):
            db.read_geometry_array(first, "auxiliary", offset, count)
    with sqlite3.connect(db_path) as sql:
        # The repeated first two chunks deduplicate too, even within one array.
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 3
        for table, field, value in (("geometry_chunks", "data", "x'00'"),
                                    ("geometry_revisions", "created_by", "'changed'")):
            with pytest.raises(sqlite3.IntegrityError, match="immutable"):
                sql.execute(f"UPDATE {table} SET {field}={value}")
        with pytest.raises(sqlite3.IntegrityError, match="retained"):
            sql.execute("DELETE FROM geometry_revisions")


def test_geometry_conflicts_validation_and_atomic_rollback(db, db_path):
    a = db.upsert_object({"kind": "fiber"})
    first = db.put_geometry(a, _geometry())
    with pytest.raises(RuntimeError, match="current revision changed"):
        db.put_geometry(a, _geometry(value=2))
    with pytest.raises(RuntimeError, match="current revision changed"):
        db.put_geometry(a, _geometry(value=2), expected_revision=str(uuid.uuid4()))
    before = db.journal_since()
    with pytest.raises(ValueError, match="shape"):
        bad = _geometry(); bad["arrays"][1]["shape"] = [10, 3]
        db.put_geometry(a, bad, expected_revision=first)
    with pytest.raises(RuntimeError, match="abort transaction"):
        with db.write() as w:
            w.put_geometry(a, _geometry(value=5), expected_revision=first)
            raise RuntimeError("abort transaction")
    assert db.current_geometry(a)["uuid"] == first
    assert db.journal_since() == before
    # Force a failure after inserting the revision/chunks; catch it and commit
    # other work to prove the internal savepoint protects callers too.
    with sqlite3.connect(db_path) as sql:
        sql.execute("CREATE TRIGGER reject_geometry_head BEFORE UPDATE ON geometry_heads BEGIN SELECT RAISE(ABORT,'injected'); END")
    with db.write() as w:
        with pytest.raises(RuntimeError, match="injected"):
            w.put_geometry(a, _geometry(value=5), expected_revision=first)
        w.upsert_object({"kind": "note", "name": "survives"})
    assert len(db.geometry_history(a)) == 1
    with sqlite3.connect(db_path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 1
        sql.execute("DROP TRIGGER reject_geometry_head")


def test_geometry_ingest_patch_fiber_and_point_collections(db, tmp_path):
    import numpy as np
    patch_dir = tmp_path / "patch"
    patch_id = _write_pcl_patch(patch_dir, 3, 4, 50, scale=(0.25, 0.5))
    fiber_id, pcl_id = str(uuid.uuid4()), str(uuid.uuid4())
    fiber_file, pcl_file = tmp_path / "fiber.json", tmp_path / "pcl.json"
    fiber = {"type": "vc3d_fiber", "uuid": fiber_id,
             "control_points": [{"position": [1, 2, 3], "segment_to_next": {"mode": "trace"}}],
             "line_points": [[1, 2, 3], [4, 6, 3]]}
    fiber_file.write_text(json.dumps(fiber))
    _write_pcl(pcl_file, {"1": {"uuid": pcl_id, "name": "collection", "points": {
        "10": {"p": [7, 8, 9], "wind_a": 2}, "2": {"p": [1, 2, 3], "custom": True}}}})
    for kind, path in (("patch", patch_dir), ("fiber", fiber_file), ("point_collection", pcl_file)):
        assert db.ingest_geometry_source(kind, path) == {"ingested": 1, "unchanged": 0, "problems": []}
        assert db.ingest_geometry_source(kind, path) == {"ingested": 0, "unchanged": 1, "problems": []}
    pcl = db.current_geometry(pcl_id)
    assert np.frombuffer(db.read_geometry_array(pcl["uuid"], "point_ids"), dtype="<i8").tolist() == [2, 10]
    assert pcl["attributes"]["point_attributes"] == [{"custom": True}, {"wind_a": 2}]
    patch = db.current_geometry(patch_id)
    points = np.frombuffer(db.read_geometry_array(patch["uuid"], "points"), dtype="<f4").reshape(3, 4, 3)
    assert points[2, 3].tolist() == [103, 102, 50]
    assert db.object(patch_id)["patch"]["scale"] == (0.25, 0.5)
    first = db.current_geometry(fiber_id)
    assert first["attributes"]["control_points_attributes"][0]["segment_to_next"] == {"mode": "trace"}
    db.set_meta(fiber_id, "keep", True)
    fiber["line_points"][1] = [8, 2, 3]
    fiber_file.write_text(json.dumps(fiber))
    assert db.ingest_registered_geometry(object_uuids=[fiber_id])["ingested"] == 1
    assert db.object(fiber_id)["meta"] == {"keep": True}
    assert db.object(fiber_id)["fiber"]["length"] == 7
    second = db.current_geometry(fiber_id)
    # Malformed re-ingestion preserves the last good geometry and graph state.
    fiber["line_points"] = "invalid"; fiber_file.write_text(json.dumps(fiber))
    report = db.ingest_registered_geometry(object_uuids=[fiber_id])
    assert len(report["problems"]) == 1 and report["ingested"] == 0
    assert db.current_geometry(fiber_id) == second
    fiber_file.unlink(); pcl_file.unlink()
    assert db.read_geometry_array(first["uuid"], "line_points")
    assert db.read_geometry_array(pcl["uuid"], "points")
    assert not db.sweep_external_data()


def test_geometry_ingest_packed_tracks(db, tmp_path):
    import numpy as np
    import struct
    packed = tmp_path / "tracks.vctracks"; packed.mkdir()
    (packed / "header.bin").write_bytes(struct.pack("<8sIIQQ", b"VCTRK01\0", 1, 64, 2, 3).ljust(64, b"\0"))
    arrays = {"coordinates.i32": np.array([[1, 2, 3], [4, 2, 3], [7, 8, 9]], dtype="<i4"),
              "offsets.i64": np.array([0, 2, 3], dtype="<i8"),
              "source_ids.u64": np.array([42, 99], dtype="<u8"),
              "family_codes.i8": np.array([0, 1], dtype="i1")}
    for name, values in arrays.items(): values.tofile(packed / name)
    assert db.ingest_geometry_source("track", packed) == {"ingested": 2, "unchanged": 0, "problems": []}
    objects = db.objects_by_kind("track")
    by_id = {o["track"]["external_key"]: o for o in objects}
    track = by_id["42"]
    assert track["track"]["length"] == 3
    rev = db.current_geometry(track["uuid"])
    assert np.frombuffer(db.read_geometry_array(rev["uuid"], "points"), dtype="<i4").reshape(-1, 3).tolist() == [[3, 2, 1], [3, 2, 4]]
    assert db.ingest_geometry_source("track", packed)["unchanged"] == 2
    assert {o["uuid"] for o in db.objects_by_kind("track")} == {o["uuid"] for o in objects}
    assert db.ingest_registered_geometry()["unchanged"] == 2


def test_geometry_metadata_updates_cannot_detach_payloads(db, db_path):
    uid = db.upsert_object({"kind": "fiber", "storage_kind": "file", "storage_ref": "missing.json"})
    revision = db.put_geometry(uid, _geometry())
    stored = db.object(uid)
    db.upsert_object({"uuid": uid, "kind": "fiber", "name": "renamed", "data_checksum": "wrong", "data_generation": 99})
    updated = db.object(uid)
    assert updated["storage_kind"] == "sqlite"
    assert updated["storage_ref"] == stored["storage_ref"]
    assert updated["data_generation"] == stored["data_generation"]
    assert updated["data_checksum"] == stored["data_checksum"]
    assert db.current_geometry(uid)["uuid"] == revision
    with pytest.raises(RuntimeError, match="FOREIGN KEY"):
        db.delete_object(uid)
    with sqlite3.connect(db_path) as sql:
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    with pytest.raises(RuntimeError, match="corrupt payload"):
        db.read_geometry_array(revision, "line_points")


@pytest.mark.parametrize("mutation", ["dtype", "rank", "duplicate", "overflow", "kind", "attributes"])
def test_geometry_rejects_invalid_descriptors(db, mutation):
    uid = db.upsert_object({"kind": "fiber"})
    data = _geometry()
    if mutation == "dtype": data["arrays"][1]["dtype"] = ">f8"
    if mutation == "rank": data["arrays"][1]["shape"] = [3]
    if mutation == "duplicate": data["arrays"].append(data["arrays"][0])
    if mutation == "overflow": data["arrays"][1]["shape"] = [2**62, 3]
    if mutation == "kind": data["kind"] = "other"
    if mutation == "attributes": data["attributes"] = []
    with pytest.raises(ValueError): db.put_geometry(uid, data)
    assert db.current_geometry(uid) is None


def test_schema_v2_migration_preserves_metadata(tmp_path):
    path = tmp_path / "v2.vcgr"
    with GraphDb.create(str(path), author="tester") as db:
        uid = db.upsert_object({"kind": "fiber"})
        db.set_approval(uid, "approved")
        before = db.object(uid)
    with sqlite3.connect(path) as sql:
        # Empty geometry tables are the only schema difference from v2.
        for table in ("edge_geometry_bindings", "geometry_heads", "geometry_revision_chunks", "geometry_revisions", "geometry_chunks"):
            sql.execute(f"DROP TABLE {table}")
        sql.execute("PRAGMA user_version=2")
    with pytest.raises(RuntimeError, match="schema v2 requires migration"):
        GraphDb.open(str(path), read_only=True)
    with GraphDb.open(str(path), author="migrator") as db:
        assert db.object(uid) == before
        assert db.current_geometry(uid) is None
        db.put_geometry(uid, _geometry())
    with GraphDb.open(str(path), read_only=True) as db:
        assert db.current_geometry(uid) is not None


def test_new_edges_bind_revisions_after_endpoint_normalization(db):
    a = db.upsert_object({"kind": "fiber"})
    b = db.upsert_object({"kind": "fiber"})
    ra = db.put_geometry(a, _geometry())
    rb = db.put_geometry(b, _geometry(value=2))
    edge = db.add_edge({"kind": "connected", "a": b, "b": a})
    record, state = db.edge(edge), db.edge_geometry_state(edge)
    revisions = {a: ra, b: rb}
    assert state == {"a_revision": revisions[record["a"]], "b_revision": revisions[record["b"]], "stale": False}
    db.set_approval(edge, "approved")


def test_geometry_storage_kind_cannot_claim_missing_payload(db):
    with pytest.raises(ValueError, match="putGeometry"):
        db.upsert_object({"kind": "fiber", "storage_kind": "sqlite"})
    assert db.objects_by_kind("fiber") == []


def test_geometry_ingest_rejects_damaged_patch_and_incomplete_save(db, tmp_path):
    patch = tmp_path / "patch"
    uid = _write_pcl_patch(patch, 4, 4, 20)
    assert db.ingest_geometry_source("patch", patch)["ingested"] == 1
    before, journal = db.current_geometry(uid), db.journal_since()
    rollback = patch / ".vc-save-rollback"
    rollback.mkdir(); (rollback / "keep").write_text("do not recover during ingestion")
    assert "incomplete save" in db.ingest_geometry_source("patch", patch)["problems"][0]
    assert (rollback / "keep").read_text() == "do not recover during ingestion"
    (rollback / "keep").unlink(); rollback.rmdir()
    data = (patch / "x.tif").read_bytes()
    (patch / "x.tif").write_bytes(data[:-3])
    assert "unreadable TIFF" in db.ingest_geometry_source("patch", patch)["problems"][0]
    assert db.current_geometry(uid) == before and db.journal_since() == journal


def _stored_geometry(db, uid):
    revision = db.current_geometry(uid)
    return {"kind": revision["kind"], "attributes": revision["attributes"],
            "arrays": [{**{k: a[k] for k in ("name", "dtype", "shape")},
                        "data": db.read_geometry_array(revision["uuid"], a["name"])}
                       for a in revision["arrays"]]}


def test_geometry_export_revision_readonly_and_overwrite(db, db_path, tmp_path):
    uid = db.upsert_object({"kind": "fiber"})
    first = db.put_geometry(uid, _geometry(value=1))
    second = db.put_geometry(uid, _geometry(value=7), expected_revision=first)
    before = db.journal_since()
    destination = tmp_path / "export.json"
    with GraphDb.open(db_path, read_only=True) as reader:
        assert reader.export_geometry(uid, destination) == second
        assert json.loads(destination.read_text())["line_points"] == [[7, 2, 3]]
        with pytest.raises(RuntimeError, match="already exists"):
            reader.export_geometry(uid, destination, revision_uuid=first)
        assert json.loads(destination.read_text())["line_points"] == [[7, 2, 3]]
        assert reader.export_geometry(uid.upper(), destination, revision_uuid=first, overwrite=True) == first
        assert json.loads(destination.read_text())["line_points"] == [[1, 2, 3]]
        with pytest.raises(RuntimeError, match="database"):
            reader.export_geometry(uid, Path(db_path), overwrite=True)
        with pytest.raises(RuntimeError, match="database"):
            reader.export_geometry(uid, Path(db_path).parent, overwrite=True)
        other = db.upsert_object({"kind": "fiber"})
        with pytest.raises(RuntimeError, match="another object"):
            reader.export_geometry(other, tmp_path / "wrong.json", revision_uuid=first)
        with pytest.raises(RuntimeError, match="revision not found"):
            reader.export_geometry(uid, tmp_path / "wrong.json", revision_uuid=str(uuid.uuid4()))
    assert db.journal_since()[:len(before)] == before
    assert len(db.journal_since()) == len(before) + 1  # only creating 'other'
    assert db.current_geometry(uid)["uuid"] == second
    assert not (tmp_path / "wrong.json").exists()
    assert not list(tmp_path.glob(".vc-export-*"))
    with GraphDb.create(str(tmp_path / "roundtrip.vcgr"), author="test") as restored:
        assert restored.ingest_geometry_source("fiber", destination)["ingested"] == 1
        revision = restored.current_geometry(uid)
        assert restored.read_geometry_array(revision["uuid"], "line_points") == db.read_geometry_array(first, "line_points")


def test_geometry_export_modified_patch_and_annotations(db, tmp_path):
    import numpy as np
    import shutil
    patch = tmp_path / "source-patch"
    patch_id = _write_pcl_patch(patch, 3, 4, 50, scale=(0.25, 0.5))
    fiber_id, pcl_id = str(uuid.uuid4()), str(uuid.uuid4())
    fiber_file, pcl_file = tmp_path / "fiber.json", tmp_path / "pcl.json"
    fiber = {"type": "vc3d_fiber", "version": 4, "uuid": fiber_id,
             "control_points": [{"position": [1, 2, 3], "tags": ["kept"]}, {"position": [4, 5, 6]}],
             "line_points": [[1, 2, 3], [4, 5, 6]], "tags": ["source"]}
    fiber_file.write_text(json.dumps(fiber))
    pcl = {"uuid": pcl_id, "name": "collection", "color": [1, 2, 3], "points": {
        "10": {"p": [7, 8, 9], "wind_a": 2}, "2": {"p": [1, 2, 3], "creation_time": "kept"}}}
    _write_pcl(pcl_file, {"17": pcl})
    for kind, source in (("patch", patch), ("fiber", fiber_file), ("point_collection", pcl_file)):
        assert db.ingest_geometry_source(kind, source)["ingested"] == 1
    shutil.rmtree(patch); fiber_file.unlink(); pcl_file.unlink()
    data = _stored_geometry(db, patch_id)
    grid = np.array([[[1, 2, 3], [-1, -1, -1]], [[6, 7, 8], [10, 11, 12]]], dtype="<f4")
    data["arrays"][0].update(shape=list(grid.shape), data=grid.tobytes())
    db.put_geometry(patch_id, data, expected_revision=db.current_geometry(patch_id)["uuid"])
    db.set_meta(fiber_id, "not_source_metadata", True)
    before = db.journal_since()
    out = tmp_path / "exported-patch"
    out.mkdir(); (out / "unrelated-old-file").write_text("old")
    with pytest.raises(RuntimeError, match="already exists"):
        db.export_geometry(patch_id, out)
    assert (out / "unrelated-old-file").read_text() == "old"
    db.export_geometry(patch_id, out, overwrite=True)
    assert set(p.name for p in out.iterdir()) == {"meta.json", "x.tif", "y.tif", "z.tif"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["tiff_dimensions"] == [2, 2]
    assert meta["bbox"] == [[1, 2, 3], [10, 11, 12]]
    assert meta["scale"] == [0.25, 0.5]
    db.export_geometry(fiber_id, fiber_file)
    db.export_geometry(pcl_id, pcl_file)
    assert json.loads(fiber_file.read_text()) == fiber
    assert json.loads(pcl_file.read_text())["collections"] == {"17": pcl}
    assert db.journal_since() == before
    with GraphDb.create(str(tmp_path / "roundtrip.vcgr"), author="test") as restored:
        for uid, kind, source in ((patch_id, "patch", out), (fiber_id, "fiber", fiber_file), (pcl_id, "point_collection", pcl_file)):
            assert restored.ingest_geometry_source(kind, source) == {"ingested": 1, "unchanged": 0, "problems": []}
            original, roundtrip = _stored_geometry(db, uid), _stored_geometry(restored, uid)
            assert roundtrip["arrays"] == original["arrays"]
            if kind != "patch": assert roundtrip["attributes"] == original["attributes"]
    assert not list(tmp_path.glob(".vc-export-*"))


def test_geometry_export_v4_unannotated_controls(db, tmp_path):
    uid = str(uuid.uuid4())
    source = tmp_path / "fiber.json"
    root = {"uuid": uid, "type": "vc3d_fiber", "version": 4,
            "control_points": [{"position": [1, 2, 3]}], "line_points": [[1, 2, 3]]}
    source.write_text(json.dumps(root))
    assert db.ingest_geometry_source("fiber", source)["ingested"] == 1
    assert "control_points_attributes" not in db.current_geometry(uid)["attributes"]
    db.export_geometry(uid, source, overwrite=True)
    assert json.loads(source.read_text()) == root


@pytest.mark.parametrize("count", [0, 1, 3])
def test_geometry_export_track_roundtrip(db, tmp_path, count):
    import numpy as np
    import struct
    uid = db.upsert_object({"kind": "track"})
    xyz = np.array([[1, 2, 3], [1, 2, 6], [5, 2, 6]], dtype="<i4")[:count]
    revision = db.put_geometry(uid, {"kind": "track", "attributes": {"family": "vertical", "external_key": str(2**64-1)},
        "arrays": [{"name": "points", "dtype": "<i4", "shape": [count, 3], "data": xyz.tobytes()}]})
    destination = tmp_path / "export.vctracks"
    db.export_geometry(uid, destination)
    assert struct.unpack("<8sIIQQ4Q", (destination / "header.bin").read_bytes())[:5] == (b"VCTRK01\0", 1, 64, 1, count)
    np.testing.assert_array_equal(np.fromfile(destination / "coordinates.i32", dtype="<i4").reshape(-1, 3), xyz[:, ::-1])
    assert np.fromfile(destination / "arclengths.f64", dtype="<f8")[0] == (7 if count == 3 else 0)
    assert np.fromfile(destination / "tortuosities.f64", dtype="<f8")[0] == (1.4 if count == 3 else np.inf)
    assert json.loads((destination / "metadata.json").read_text())["graph_revision_uuid"] == revision
    assert db.ingest_geometry_source("track", destination)["ingested"] == 1
    restored = next(o for o in db.objects_by_kind("track") if o["uuid"] != uid)
    assert _stored_geometry(db, restored["uuid"]) == _stored_geometry(db, uid)
    db.export_geometry(uid, destination, overwrite=True)
    assert not list(tmp_path.glob(".vc-export-*"))


@pytest.mark.parametrize("failure", ["annotations", "nonfinite", "auxiliary", "symlink", "type"])
def test_geometry_export_failures_preserve_destination(db, tmp_path, failure):
    import struct
    uid = db.upsert_object({"kind": "fiber"})
    geometry = _geometry()
    if failure == "annotations": geometry["attributes"] = {"line_points_attributes": []}
    if failure == "nonfinite": geometry["arrays"][1]["data"] = struct.pack("<ddd", float("nan"), 2, 3)
    if failure == "auxiliary": geometry["arrays"].append({"name": "extra", "dtype": "|u1", "shape": [1], "data": b"x"})
    revision = db.put_geometry(uid, geometry)
    before = db.journal_since()
    dest = tmp_path / "destination"
    if failure == "type":
        dest.mkdir(); (dest / "keep").write_text("old")
    elif failure == "symlink":
        (tmp_path / "keep").write_text("old"); dest.symlink_to(tmp_path / "keep")
    else: dest.write_text("old")
    with pytest.raises(RuntimeError): db.export_geometry(uid, dest, overwrite=True)
    assert (dest / "keep" if failure == "type" else dest).read_text() == "old"
    assert db.current_geometry(uid)["uuid"] == revision and db.journal_since() == before
    assert not list(tmp_path.glob(".vc-export-*"))


@pytest.mark.parametrize("explicit_path", [False, True])
@pytest.mark.parametrize("duplicates", [False, True])
def test_track_uuid_selection_is_case_insensitive(db, tmp_path, explicit_path, duplicates):
    packed = tmp_path / "tracks.vctracks"
    _packed_tracks(packed, _crossing_lines(), [0, 1], [11, 22])
    ids = _register_tracks(db, packed, [11, 22], ["horizontal", "vertical"])
    selected = [u.upper() for u in ids]
    if duplicates: selected += ids
    report = db.link_tracks_to_tracks(str(packed) if explicit_path else "", track_uuids=selected)
    assert report["problems"] == []
    assert report["objects_considered"] == 2 and report["edges_created"] == 1
    assert {o["uuid"] for o in db.objects_by_kind("track")} == set(ids)
    assert db.object_count("track") == 2 and db.object_count("missing") == 0
    before = db.journal_since()
    missing = db.link_tracks_to_tracks(str(packed) if explicit_path else "", track_uuids=selected + [str(uuid.uuid4()).upper()])
    assert missing["problems"] and db.journal_since() == before


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
