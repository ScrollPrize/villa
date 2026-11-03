"""Incremental detector runs (new_uuids), detector versions, progress callbacks
and the store's link marker."""
import numpy as np
import pytest

from vc_object_store import Fiber, detector_versions
from test_coordinate_frames import LINE_OPTIONS, VOLUME, frame, patch as framed_patch
from test_object_links import array, plane, put


def named_edges(db, kind):
    """{(name, name): edge} of every edge of a kind, endpoints by object name."""
    result = {}
    for edge in db.edges_by_kind(kind):
        names = tuple(sorted(db.object(uid)["name"] for uid in (edge["a"], edge["b"])))
        result[names] = edge
    return result


def detector_edge(db, a, b, kind, detector):
    whole = dict(whole_geometry=True)
    return db.add_edge(dict(a=a, b=b, kind=kind, a_anchor=whole, b_anchor=whole,
                            meta={"detector": {"name": detector, "version": 1}}))


def test_detector_versions_name_every_detector():
    assert detector_versions() == {
        "surface-overlap": 1, "fiber-surface": 3, "track-track": 1, "track-surface": 1,
        "point-collection-surface": 1}


def test_overlap_increment_matches_a_full_run_for_new_pairs(db):
    a, _ = plane(db, 0, "a")
    b, _ = plane(db, 8, "b")
    c, _ = plane(db, 500, "c")
    assert db.link_surface_overlaps()["edges_created"] == 1
    kept = named_edges(db, "same-surface")[("a", "b")]["uuid"]
    stale_old = detector_edge(db, a, c, "same-surface", "surface-overlap")
    d, _ = plane(db, 4, "d")
    e, _ = plane(db, 1000, "e")
    stale_new = detector_edge(db, d, c, "same-surface", "surface-overlap")

    report = db.link_surface_overlaps(new_uuids=[d, e])
    assert report["committed"] and not report["failed"], report
    # c is no partner of d or e: it is never loaded.
    assert report["objects_considered"] == 4
    assert report["edges_created"] == 2 and report["stale_removed"] == 1
    edges = named_edges(db, "same-surface")
    assert set(edges) == {("a", "b"), ("a", "d"), ("b", "d"), ("a", "c")}
    assert edges[("a", "b")]["uuid"] == kept  # not a new object's pair: untouched
    assert stale_new not in {edge["uuid"] for edge in db.edges_by_kind("same-surface")}

    # A full run proposes exactly the incremental pairs (same poses: nothing
    # updated) and removes the stale edge between existing surfaces.
    full = db.link_surface_overlaps()
    assert full["edges_created"] == full["edges_updated"] == 0
    assert full["edges_skipped_existing"] == 3 and full["stale_removed"] == 1
    assert stale_old not in {edge["uuid"] for edge in db.edges_by_kind("same-surface")}


def test_overlap_increment_with_nothing_new_changes_nothing(db):
    a, _ = plane(db, 0, "a")
    c, _ = plane(db, 500, "c")
    stale = detector_edge(db, a, c, "same-surface", "surface-overlap")
    report = db.link_surface_overlaps(new_uuids=[])
    assert report["committed"] and report["objects_considered"] == 0
    assert report["edges_created"] == report["stale_removed"] == 0
    assert db.edge(stale)["uuid"] == stale


def test_new_uuids_must_be_selected(db):
    a, _ = plane(db, 0, "a")
    b, _ = plane(db, 8, "b")
    with pytest.raises(ValueError, match="not selected"):
        db.link_surface_overlaps(surface_uuids=[a], new_uuids=[b])
    collection = put(db, "point_collection", [array("points", [[10, 10, 20]], "<f8"),
                                              array("point_ids", [1], "<i8")])[0]
    with pytest.raises(ValueError, match="not selected"):
        db.link_point_collections_to_surfaces(new_uuids=[collection, "00000000-0000-0000-0000-000000000001"])


def big_plane(db, name):
    y, x = np.mgrid[:65, :65]
    points = np.stack([x - 16, y - 16, np.full_like(x, 20)], axis=-1)
    return put(db, "surface", [array("points", points, "<f4")], {"scale": [1, 1]}, name=name)[0]


def pcl(db, name, point):
    return put(db, "point_collection", [array("points", [point], "<f8"), array("point_ids", [1], "<i8")],
               name=name)[0]


def test_pcl_increment_links_new_objects_and_keeps_existing_links(db):
    plane(db, 0, "small")
    pcl(db, "p1", [10, 10, 20])
    plane(db, 900, "far")
    pcl(db, "far_points", [905, 10, 20])
    assert db.link_point_collections_to_surfaces()["edges_created"] == 2
    old = named_edges(db, "point-surface")[("p1", "small")]["uuid"]

    # A larger new surface: the existing collection gains a link to it and keeps
    # its old one (a full run would move it).
    big = big_plane(db, "big")
    report = db.link_point_collections_to_surfaces(new_uuids=[big])
    assert report["edges_created"] == 1 and report["stale_removed"] == 0, report
    # big and p1: an existing collection is linked to the new surfaces alone.
    assert report["objects_considered"] == 2
    edges = named_edges(db, "point-surface")
    assert set(edges) == {("p1", "small"), ("big", "p1"), ("far", "far_points")}
    assert edges[("p1", "small")]["uuid"] == old

    # A new collection is linked as a full run would: to the larger surface.
    p2 = pcl(db, "p2", [12, 12, 20])
    report = db.link_point_collections_to_surfaces(new_uuids=[p2])
    assert report["edges_created"] == 1, report
    assert ("big", "p2") in named_edges(db, "point-surface")

    full = db.link_point_collections_to_surfaces()
    assert full["edges_created"] == full["edges_updated"] == 0
    assert full["stale_removed"] == 1  # p1-small: big wins
    assert set(named_edges(db, "point-surface")) == {("big", "p1"), ("big", "p2"), ("far", "far_points")}


def test_pcl_increment_keeps_named_pair_policy_over_all_selected_surfaces(db):
    plane(db, 0, "a")
    big_plane(db, "b")
    named = pcl(db, "between_surfaces__a__b", [10, 10, 20])
    report = db.link_point_collections_to_surfaces(new_uuids=[named])
    assert report["edges_created"] == 1, report
    edge = next(iter(named_edges(db, "point-surface").values()))
    assert edge["meta"]["point_collection_surface"]["selection"] == "nearest_named_pair"


def framed_fiber(db, name, y=96):
    xyz = np.array([[x, y, 96] for x in range(72, 121)], dtype=float)
    uid = db.upsert_object({"kind": "fiber", "name": name, "meta": {"coordinates": frame(1)}})
    db.put_geometry(uid, Fiber(xyz[[0, -1]].tolist(), xyz.tolist()))
    return uid


def named(db, uid, name):
    record = db.object(uid)
    record["name"] = name
    db.upsert_object(record)
    return uid


def test_fiber_increment_links_new_fibers_and_surfaces(db):
    named(db, framed_patch(db, 1, 65, (64, 64, 96)), "a")
    far = named(db, framed_patch(db, 1, 65, (64, 64, 400)), "far")
    framed_fiber(db, "f1")
    assert db.link_fibers_to_surfaces(**LINE_OPTIONS)["edges_created"] == 1
    old = named_edges(db, "same-surface")[("a", "f1")]["uuid"]

    b = named(db, framed_patch(db, 1, 65, (64, 64, 96)), "b")
    f2 = framed_fiber(db, "f2", y=98)
    stale = detector_edge(db, f2, far, "same-surface", "fiber-surface")
    report = db.link_fibers_to_surfaces(**LINE_OPTIONS, new_uuids=[b, f2])
    assert report["committed"] and not report["failed"], report
    # Surfaces a, b and fibers f1, f2; the far surface is no partner.
    assert report["objects_considered"] == 4, report
    assert report["stale_removed"] == 1
    edges = named_edges(db, "same-surface")
    # f1 (existing) gains b; f2 (new) links to a and b, as in a full run.
    assert set(edges) == {("a", "f1"), ("b", "f1"), ("a", "f2"), ("b", "f2")}
    assert edges[("a", "f1")]["uuid"] == old

    full = db.link_fibers_to_surfaces(**LINE_OPTIONS)
    assert full["edges_created"] == full["stale_removed"] == 0, full


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_progress_reaches_python_and_its_errors_do_not_stop_a_detector(db):
    plane(db, 0, "a")
    plane(db, 8, "b")
    seen = []
    report = db.link_surface_overlaps(progress=lambda *update: seen.append(update))
    assert report["edges_created"] == 1
    assert {stage for stage, *_ in seen} >= {"selecting stored surfaces", "loading surfaces",
                                             "checking surface pairs"}
    assert all(isinstance(completed, int) and isinstance(detail, str) for _, completed, _, detail in seen)

    def broken(*update):
        raise RuntimeError("progress display failed")
    # Reported as unraisable (a warning under pytest); the detector finishes.
    assert db.link_surface_overlaps(progress=broken)["edges_skipped_existing"] == 1
    with pytest.raises(TypeError):
        db.link_surface_overlaps(progress=3)


def test_link_marker_is_not_journaled_and_stays_with_its_replica(db):
    assert db.link_marker() is None
    db.enable_store_positions()
    plane(db, 0, "a")
    head = db.store_positions()["position"]
    journal = db.journal_since()
    marker = {"position": head, "key": "abc", "exact": True, "selection": {"surface": ["x"]}}
    db.set_link_marker(marker)
    assert db.link_marker() == marker
    assert db.journal_since() == journal and db.store_positions()["position"] == head
    db.set_link_marker({"position": head, "exact": False})
    assert db.link_marker() == {"position": head, "exact": False}
    # A copy with its own replica identity reads none.
    db.fork_replica()
    assert db.link_marker() is None
