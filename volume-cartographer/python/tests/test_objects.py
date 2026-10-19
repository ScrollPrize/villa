"""Standalone object storage: no graph database is needed."""
import json
import os
import sqlite3
import uuid
from pathlib import Path
import pytest
import vc
if os.environ.get("VC_BUILD_PYTHON"):
    vc.__path__.insert(0, str(Path(os.environ["VC_BUILD_PYTHON"]) / "vc"))
ObjectStore = pytest.importorskip("vc.objects").ObjectStore

@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "objects.vcobj")

@pytest.fixture()
def db(db_path):
    with ObjectStore.create(db_path, author="pytest", hostname="testhost") as store:
        yield store

def _geometry(kind="fiber", value=1.0):
    import struct
    return {"kind": kind, "arrays": [
        {"name": "control_points", "dtype": "<f8", "shape": [0, 3], "data": b""},
        {"name": "line_points", "dtype": "<f8", "shape": [1, 3],
         "data": struct.pack("<ddd", value, 2.0, 3.0)}]}


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


def test_geometry_revisions_approval_history_and_source_independence(db, db_path):
    a = db.upsert_object({"kind": "fiber", "tags": ["keep"]})
    first = db.put_geometry(a, _geometry())
    db.set_approval(a, "approved")
    before = db.journal_since()
    assert db.put_geometry(a, _geometry(), expected_revision=first) == first
    assert db.journal_since() == before
    second = db.put_geometry(a, _geometry(value=9), expected_revision=first)
    assert db.geometry_revision(second)["parent_uuid"] == first
    assert db.object(a)["approval_state"] == "pending"
    assert db.read_geometry_array(first, "line_points") == _geometry()["arrays"][1]["data"]
    assert len(db.geometry_history(a)) == 2
    db.set_meta(a, "annotation", "changed")
    assert db.current_geometry(a)["uuid"] == second
    assert db.object(a)["tags"] == ["keep"]
    assert not db.sweep_external_data()
    with ObjectStore.open(db_path, read_only=True) as read:
        assert read.uuid == db.uuid
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
    with ObjectStore.open(db_path, read_only=True) as reader:
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
    with ObjectStore.create(str(tmp_path / "roundtrip.vcgr"), author="test") as restored:
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
    with ObjectStore.create(str(tmp_path / "roundtrip.vcgr"), author="test") as restored:
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


