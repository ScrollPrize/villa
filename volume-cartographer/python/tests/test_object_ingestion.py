"""Object-store ingestion seeds editable metadata and UUID topology atomically."""
import json
import sqlite3
import uuid

import pytest
from vc.objects import ObjectStore
from object_helpers import edge_choice
from test_objects import db, db_path, _geometry


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
