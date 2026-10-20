"""Native ingestion/export preserves geometry, annotations and relationships."""
import json
from pathlib import Path
import uuid

import pytest
from vc_object_store import ObjectStore
from object_helpers import edge_choice
from object_helpers import _geometry


def fiber(uid, branches=()):
    return {"type": "vc3d_fiber", "version": 4, "uuid": uid,
            "tags": ["seed"], "width": 3, "hv_classification": {"manual_tag": "H"},
            "control_points": [{"position": [1, 2, 3]}, {"position": [4, 5, 6]}],
            "line_points": [[1, 2, 3], [4, 5, 6]], "branches": list(branches)}


def test_fiber_directory_topology_metadata_and_export(db, tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    source = tmp_path / "fibers"; source.mkdir()
    first = fiber(a, [{"control_point_index": 1, "branch_control_point_index": 0, "branch_file": "second.json"}])
    second = fiber(b, [{"control_point_index": 0, "branch_control_point_index": 1, "branch_file": "first.json"}])
    (source / "first.json").write_text(json.dumps(first))
    (source / "second.json").write_text(json.dumps(second))
    report = db.ingest_geometry_source("fiber", source)
    assert report == {"ingested": 2, "unchanged": 0, "edges_imported": 1, "problems": []}
    assert db.neighbours(a) == [b]
    edge = db.edges_of(a)[0]
    assert edge["kind"] == "fiber_branch" and edge["approval_state"] == "pending"
    anchors = {edge["a"]: edge["a_anchor"], edge["b"]: edge["b_anchor"]}
    assert anchors[a]["points"][0]["index"] == [1] and anchors[a]["position"] == (4, 5, 6)
    assert anchors[b]["points"][0]["index"] == [0]
    assert not db.current_geometry(a)["attributes"]
    assert "branches" not in db.object(a)["meta"]["source"]
    assert db.select_objects(tag="seed")
    updated = db.object(a)
    updated["name"], updated["tags"] = "renamed", ["reviewed"]
    updated["fiber"]["width"] = 9
    db.upsert_object(updated)
    db.set_approval(edge["uuid"], "approved")
    assert db.ingest_geometry_source("fiber", source)["unchanged"] == 2
    assert db.edge(edge["uuid"])["approval_state"] == "approved"
    out = tmp_path / "out.json"
    db.export_geometry(a, out)
    exported = json.loads(out.read_text())
    assert exported["name"] == "renamed" and exported["tags"] == ["reviewed"]
    assert exported["width"] == 9 and exported["branches"][0]["target_uuid"] == b
    assert not exported["branches"][0]["pending"]
    # DB removals are authoritative; re-ingestion doesn't resurrect source topology.
    db.remove_edge(edge["uuid"])
    db.ingest_geometry_source("fiber", source)
    assert db.edges_of(a) == []


def test_bad_source_relationship_rolls_back_whole_batch(db, tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    source = tmp_path / "fibers"; source.mkdir()
    (source / "first.json").write_text(json.dumps(fiber(a, [{
        "control_point_index": 99, "branch_control_point_index": 0, "branch_file": "second.json"}])))
    (source / "second.json").write_text(json.dumps(fiber(b)))
    report = db.ingest_geometry_source("fiber", source)
    assert report["ingested"] == 0 and "out of range" in report["problems"][0]
    assert db.object(a) is None and db.object(b) is None and db.journal_since() == []


def test_point_links_self_edges_winding_and_native_export(db, tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    source = tmp_path / "collections.json"
    source.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": {
        "10": {"uuid": a, "name": "A", "tags": {"reviewed": "yes"}, "windings_linked": [20], "points": {
            "1": {"p": [1, 2, 3], "links": [2, 3], "wind_a": 2},
            "2": {"p": [4, 5, 6], "links": [1]}}},
        "20": {"uuid": b, "name": "B", "windings_linked": [10], "points": {
            "3": {"p": [7, 8, 9], "links": [1]}}}}}))
    report = db.ingest_geometry_source("point_collection", source)
    assert report == {"ingested": 2, "unchanged": 0, "edges_imported": 3, "problems": []}
    assert [o["uuid"] for o in db.select_objects(tag="reviewed")] == [a]
    edges = db.edges_of(a)
    assert sum(e["kind"] == "point_link" for e in edges) == 2
    same = next(e for e in edges if e["a"] == e["b"])
    assert same["a_anchor"]["points"][0]["index"] == [1] and same["b_anchor"]["points"][0]["index"] == [2]
    assert not db.current_geometry(a)["attributes"]
    meta = db.object(a)["meta"]["source"]
    assert "links" not in meta["point_attributes"][0] and "windings_linked" not in meta
    meta["point_attributes"][0]["wind_a"] = 4
    db.set_meta(a, "source", meta)
    out = tmp_path / "out.json"
    db.export_geometry(a, out)
    exported = json.loads(out.read_text())["collections"]["10"]
    assert exported["tags"] == {"reviewed": "yes"}
    assert exported["windings_linked"] == [20]
    assert exported["points"]["1"]["links"] == [2, 3]
    assert exported["points"]["2"]["links"] == [1]
    assert exported["points"]["1"]["wind_a"] == 4
    # Export preserves the generic self-edge through JSON as well.
    snapshot = json.loads(db.export_json())
    assert len(snapshot["edges"]) == 3


def test_selection_and_projected_metadata_export(db):
    a = db.upsert_object({"kind": "fiber", "tags": ["chosen"], "name": "a"})
    revision = db.put_geometry(a, _geometry())
    b = db.upsert_object({"kind": "note", "tags": ["chosen"], "name": "b"})
    c = db.upsert_object({"kind": "note", "name": "c"})
    ab = db.add_edge({"kind": "connection", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    db.add_edge({"kind": "other", "a": b, "b": c})
    selection = dict(connected_to=a, edge_kinds=["connection"], tag="chosen")
    records = db.select_objects(**selection)
    snapshot = json.loads(db.export_json(**selection, fields=["name", "tags"]))
    assert {o["uuid"] for o in records} == {a, b} == {o["uuid"] for o in snapshot["objects"]}
    assert snapshot["metadata_only"] and snapshot["projected"]
    assert [e["uuid"] for e in snapshot["edges"]] == [ab]
    record = next(o for o in snapshot["objects"] if o["uuid"] == a)
    assert record["geometry"]["revision_uuid"] == revision
    assert "meta" not in record and "arrays" not in record
    assert db.select_objects(object_uuids=[c])[0]["uuid"] == c
    with pytest.raises(RuntimeError, match="projected"):
        db.import_json(json.dumps(snapshot))
    with pytest.raises(RuntimeError, match="synchronize"):
        db.import_json(db.export_json())


def test_metadata_export_reports_explicitly_retained_edges(db):
    a = db.upsert_object({"kind": "fiber"})
    b = db.upsert_object({"kind": "note"})
    edge = db.add_edge({"kind": "connected", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    revision = db.put_geometry(a, _geometry(), edge_edits=[edge_choice(db, edge, "retain")])
    exported = json.loads(db.export_json())
    assert exported["edges"][0]["uuid"] == edge
    assert exported["edges"][0]["geometry_bindings"] == {"a": revision if exported["edges"][0]["a"] == a else None,
                                                              "b": revision if exported["edges"][0]["b"] == a else None, "stale": False}


def test_retained_fiber_relationship_exports_current_referenced_positions(db, tmp_path):
    import struct
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    source = tmp_path / "fibers"; source.mkdir()
    (source / "a.json").write_text(json.dumps(fiber(a, [{
        "control_point_index": 1, "branch_control_point_index": 0, "branch_file": "b.json"}])))
    (source / "b.json").write_text(json.dumps(fiber(b)))
    assert not db.ingest_geometry_source("fiber", source)["problems"]
    edge = db.edges_of(a)[0]["uuid"]
    revision = db.current_geometry(a)
    geometry = {"kind": "fiber", "arrays": []}
    for descriptor in revision["arrays"]:
        data = db.read_geometry_array(revision["uuid"], descriptor["name"])
        if descriptor["name"] == "control_points":
            data = struct.pack("<6d", 1, 2, 3, 40, 50, 60)
        geometry["arrays"].append({**descriptor, "data": data})
    db.put_geometry(a, geometry, expected_revision=revision["uuid"], edge_edits=[edge_choice(db, edge, "retain")])
    db.export_geometry(a, tmp_path / "export.json")
    exported = json.loads((tmp_path / "export.json").read_text())
    assert exported["branches"][0]["control_point_index"] == 1
    assert exported["branches"][0]["control_point_position"] == [40, 50, 60]
    assert exported["branches"][0]["branch_control_point_position"] == [1, 2, 3]


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
        assert db.ingest_geometry_source(kind, path) == {"ingested": 1, "unchanged": 0, "edges_imported": 0, "problems": []}
        assert db.ingest_geometry_source(kind, path) == {"ingested": 0, "unchanged": 1, "edges_imported": 0, "problems": []}
    pcl = db.current_geometry(pcl_id)
    assert np.frombuffer(db.read_geometry_array(pcl["uuid"], "point_ids"), dtype="<i8").tolist() == [2, 10]
    assert db.object(pcl_id)["meta"]["source"]["point_attributes"] == [{"custom": True}, {"wind_a": 2}]
    patch = db.current_geometry(patch_id)
    points = np.frombuffer(db.read_geometry_array(patch["uuid"], "points"), dtype="<f4").reshape(3, 4, 3)
    assert points[2, 3].tolist() == [103, 102, 50]
    assert db.object(patch_id)["patch"]["scale"] == (0.25, 0.5)
    first = db.current_geometry(fiber_id)
    assert db.object(fiber_id)["meta"]["source"]["control_points_attributes"][0]["segment_to_next"] == {"mode": "trace"}
    db.set_meta(fiber_id, "keep", True)
    fiber["line_points"][1] = [8, 2, 3]
    fiber_file.write_text(json.dumps(fiber))
    assert db.ingest_registered_geometry(object_uuids=[fiber_id])["ingested"] == 1
    assert db.object(fiber_id)["meta"]["keep"] is True
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
    assert db.ingest_geometry_source("track", packed) == {"ingested": 2, "unchanged": 0, "edges_imported": 0, "problems": []}
    objects = db.objects_by_kind("track")
    by_id = {o["track"]["external_key"]: o for o in objects}
    track = by_id["42"]
    assert track["track"]["length"] == 3
    rev = db.current_geometry(track["uuid"])
    assert np.frombuffer(db.read_geometry_array(rev["uuid"], "points"), dtype="<i4").reshape(-1, 3).tolist() == [[3, 2, 1], [3, 2, 4]]
    assert db.ingest_geometry_source("track", packed)["unchanged"] == 2
    assert {o["uuid"] for o in db.objects_by_kind("track")} == {o["uuid"] for o in objects}
    assert db.ingest_registered_geometry()["unchanged"] == 2


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
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as restored:
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
    exported = json.loads(fiber_file.read_text())
    for field, value in fiber.items(): assert exported[field] == value
    exported = json.loads(pcl_file.read_text())["collections"]["17"]
    for field, value in pcl.items():
        if field == "points":
            for pid, point in value.items():
                for key, expected in point.items(): assert exported[field][pid][key] == expected
        else: assert exported[field] == value
    assert db.journal_since() == before
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as restored:
        for uid, kind, source in ((patch_id, "patch", out), (fiber_id, "fiber", fiber_file), (pcl_id, "point_collection", pcl_file)):
            assert restored.ingest_geometry_source(kind, source) == {"ingested": 1, "unchanged": 0, "edges_imported": 0, "problems": []}
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
    exported = json.loads(source.read_text())
    for key, value in root.items(): assert exported[key] == value


@pytest.mark.parametrize("count", [0, 1, 3])
def test_geometry_export_track_roundtrip(db, tmp_path, count):
    import numpy as np
    import struct
    uid = db.upsert_object({"kind": "track", "track": {"external_key": str(2**64-1), "family": "vertical"}})
    xyz = np.array([[1, 2, 3], [1, 2, 6], [5, 2, 6]], dtype="<i4")[:count]
    revision = db.put_geometry(uid, {"kind": "track",
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
    assert _stored_geometry(db, restored["uuid"])["arrays"] == _stored_geometry(db, uid)["arrays"]
    assert restored["track"]["family"] == "vertical"
    db.export_geometry(uid, destination, overwrite=True)
    assert not list(tmp_path.glob(".vc-export-*"))


@pytest.mark.parametrize("failure", ["annotations", "nonfinite", "auxiliary", "symlink", "type"])
def test_geometry_export_failures_preserve_destination(db, tmp_path, failure):
    import struct
    uid = db.upsert_object({"kind": "fiber"})
    geometry = _geometry()
    if failure == "annotations": db.set_meta(uid, "source", {"line_points_attributes": []})
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
