"""Discovery consumes stored revisions, never object source paths."""
import uuid

import numpy as np
import pytest

from object_helpers import edge_choice


def array(name, values, dtype):
    data = np.asarray(values, dtype=dtype)
    return {"name": name, "dtype": dtype, "shape": list(data.shape), "data": data.tobytes()}


def put(db, kind, arrays, attributes=None, **record):
    uid = db.upsert_object({"kind": kind, "storage_ref": "/missing/source", **record})
    geometry = {"kind": kind, "attributes": attributes or {}, "arrays": arrays}
    db.put_geometry(uid, geometry)
    return uid, geometry


def plane(db, offset=0, name="surface"):
    y, x = np.mgrid[:33, :33]
    points = np.stack([x + offset, y, np.full_like(x, 20)], axis=-1)
    return put(db, "surface", [array("points", points, "<f4")], {"scale": [1, 1]}, name=name)


def test_surface_discovery_uses_current_revisions_and_preserves_review(db):
    a, ga = plane(db)
    b, gb = plane(db)
    objects = [db.object(a), db.object(b)]
    report = db.link_surface_overlaps()
    assert report["edges_created"] == 1
    assert [db.object(a), db.object(b)] == objects
    edge = db.edges_of(a)[0]["uuid"]
    assert not db.edge_geometry_state(edge)["stale"]
    assert db.link_surface_overlaps()["edges_skipped_existing"] == 1
    # A new revision with identical coordinates still needs revalidation.
    gb["attributes"]["edit"] = True
    db.put_geometry(b, gb, expected_revision=db.current_geometry(b)["uuid"], edge_edits=[edge_choice(db, e["uuid"], "retain") for e in db.affected_geometry_edges(b, gb)])
    assert not db.edge_geometry_state(edge)["stale"]
    assert db.link_surface_overlaps()["edges_skipped_existing"] == 1
    assert db.edges_of(a)[0]["uuid"] == edge
    assert not db.edge_geometry_state(edge)["stale"]
    db.set_approval(edge, "approved")
    approved = db.edge(edge)
    assert db.link_surface_overlaps(min_inliers=10000)["stale_removed"] == 0
    assert db.edge(edge) == approved


def test_surface_selection_missing_geometry_and_stale_scope(db):
    a, _ = plane(db)
    b, _ = plane(db)
    c, _ = plane(db)
    assert db.link_surface_overlaps()["edges_created"] == 3
    assert db.link_surface_overlaps(surface_uuids=[a.upper(), a, b], min_inliers=10000)["stale_removed"] == 1
    assert len(db.edges_of(c)) == 2
    with pytest.raises((ValueError, RuntimeError), match="missing"):
        db.link_surface_overlaps(surface_uuids=[str(uuid.uuid4())])
    missing = db.upsert_object({"kind": "surface", "storage_ref": "/missing/source"})
    report = db.link_surface_overlaps(surface_uuids=[missing])
    assert any("no stored geometry" in p for p in report["problems"])
    assert len(db.edges_of(c)) == 2


def track(db, key, points, family, dataset="dataset-label"):
    return put(db, "track", [array("points", points, "<i4")], {"family": family},
               track={"dataset_ref": dataset, "external_key": str(key), "family": family})


def test_stored_track_crossings_selection_and_atomic_failure(db):
    a, _ = track(db, 11, [[x, 15, 20] for x in range(31)], "horizontal")
    b, gb = track(db, 99, [[15, y, 20] for y in range(31)], "vertical")
    track(db, 11, [[15, y, 20] for y in range(31)], "vertical", "other-dataset")
    report = db.link_tracks_to_tracks(workers=2)
    assert not report["problems"]
    assert report["edges_created"] == 1
    edge = db.edges_of(a)[0]
    assert edge["a_anchor"]["position"] == (15, 15, 20)
    assert edge["b_anchor"]["position"] == (15, 15, 20)
    assert db.link_tracks_to_tracks(track_uuids=[a.upper(), a])["stale_removed"] == 0
    gb["arrays"] = [array("points", [[50, y, 20] for y in range(31)], "<i4")]
    edits = [dict(array="points", element_offset=0, data=gb["arrays"][0]["data"])]
    db.edit_geometry(b, edits, db.current_geometry(b)["uuid"], edge_edits=[edge_choice(db, e["uuid"], "remove") for e in db.affected_geometry_edit_edges(b, edits, db.current_geometry(b)["uuid"])])
    assert db.link_tracks_to_tracks(track_uuids=[a, b])["stale_removed"] == 0
    bad = db.upsert_object({"kind": "track", "track": {"dataset_ref": "missing", "external_key": "0"}})
    before = db.journal_since()
    report = db.link_tracks_to_tracks()
    assert any("no stored geometry" in p for p in report["problems"])
    assert db.journal_since() == before
    assert db.link_tracks_to_tracks(track_uuids=[bad], dataset_ref="other")["problems"]


def test_stored_collection_ids_winding_scaling_and_rerun(db):
    surface, _ = plane(db)
    collection, geometry = put(db, "point_collection", [
        array("points", [[10, 10, 40], [20, 20, 40]], "<f8"),
        array("point_ids", [90, 7], "<i8"),
    ], meta={"source": {"point_attributes": [{"wind_a": 3}, {"wind_a": 4}], "winding_is_absolute": True}})
    report = db.link_point_collections_to_surfaces(point_scale=0.5, window_points=3, window_min_points=3)
    assert report["edges_created"] == 2
    edges = db.edges_of(collection)
    for edge in edges:
        anchor = edge["a_anchor"] if edge["a"] == collection else edge["b_anchor"]
        assert anchor["points"][0]["index"][0] in [7, 90]
        assert anchor["position"][2] == 40
        evidence = edge["meta"]["point_collection_surface"]
        assert evidence["winding_is_absolute"]
        assert evidence["winding_annotation"] in [3, 4]
    assert db.link_point_collections_to_surfaces(point_scale=0.5)["edges_updated"] == 2
    assert db.link_point_collections_to_surfaces(point_scale=0.5)["edges_skipped_existing"] == 2
    geometry["arrays"][0] = array("points", [[1000, 1000, 40], [2000, 2000, 40]], "<f8")
    edits = [dict(array="points", element_offset=0, data=geometry["arrays"][0]["data"])]
    db.edit_geometry(collection, edits, db.current_geometry(collection)["uuid"], edge_edits=[edge_choice(db, e["uuid"], "remove") for e in db.affected_geometry_edit_edges(collection, edits, db.current_geometry(collection)["uuid"])])
    assert db.link_point_collections_to_surfaces(collection_uuids=[collection], surface_uuids=[surface], point_scale=0.5)["stale_removed"] == 0
