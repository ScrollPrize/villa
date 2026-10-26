"""Structured detector outcomes: failures are distinct from ordinary diagnostics."""
import pytest

from test_coordinate_frames import VOLUME, frame, patch as framed_patch, collection as framed_collection
from test_object_links import array, plane, put


def collection(db, point, winding=1):
    return put(db, "point_collection", [array("points", [point], "<f8"), array("point_ids", [1], "<i8")],
               meta={"source": {"point_attributes": [{"wind_a": winding}]}})[0]


def detector_edge(db, a, b):
    """A pending PCL-patch edge that a rerun will not propose again."""
    whole = dict(whole_geometry=True)
    return db.add_edge(dict(a=a, b=b, kind="point-patch", a_anchor=whole, b_anchor=whole,
                            meta={"detector": {"name": "point-collection-patch", "version": 1}}))


def failures(report):
    return [issue for issue in report["issues"] if issue["severity"] == "failure"]


def test_failed_collection_is_reported_skipped_and_keeps_its_edges(db):
    patch, _ = plane(db)
    good = collection(db, [10, 10, 20])
    bad = collection(db, [12, 12, 20], winding="not a number")
    unproposed, kept = detector_edge(db, good, patch), detector_edge(db, bad, patch)
    report = db.link_point_collections_to_patches()
    assert report["failed"] and report["committed"]
    assert failures(report) == [dict(severity="failure", stage="load", object_uuids=[bad],
                                     reason="wind_a must be numeric or null", count=1)]
    assert report["problems"] == [f"{bad}: wind_a must be numeric or null, skipped"]
    # Independent work commits; only objects that loaded are in stale-edge scope.
    assert report["edges_created"] == 1 and report["stale_removed"] == 1
    assert db.edge(kept)["uuid"] == kept
    assert unproposed not in {e["uuid"] for e in db.edges_of(good)}
    again = db.link_point_collections_to_patches()
    assert again["edges_skipped_existing"] == 1 and again["edges_created"] == again["stale_removed"] == 0


def test_missing_geometry_is_a_failure_for_each_detector(db):
    patch, _ = plane(db)
    empty_patch = db.upsert_object({"kind": "patch", "storage_ref": "/missing/source"})
    empty_collection = db.upsert_object({"kind": "point_collection", "storage_ref": "/missing/source"})
    report = db.link_point_collections_to_patches()
    assert report["committed"]
    assert sorted(i["object_uuids"][0] for i in failures(report)) == sorted([empty_patch, empty_collection])
    report = db.link_patch_overlaps()
    assert report["failed"] and [i["object_uuids"] for i in failures(report)] == [[empty_patch]]


def test_rejected_overlap_pairs_are_diagnostics(db):
    plane(db)
    plane(db, offset=16)  # overlaps half of the first patch
    report = db.link_patch_overlaps(min_overlap_fraction=0.9)
    assert report["committed"] and not report["failed"] and report["edges_created"] == 0
    diagnostics = report["issues"]
    assert diagnostics and all(i["severity"] == "diagnostic" and i["stage"] == "discover" for i in diagnostics)
    assert sum(i["count"] for i in diagnostics) == 1
    assert report["problems"] == ["patch-overlap: 1 pair(s) rejected: insufficient_overlap_fraction"]
    assert diagnostics[0]["reason"] == "insufficient_overlap_fraction" and diagnostics[0]["object_uuids"] == []


def test_clean_run_reports_no_issues(db):
    plane(db)
    plane(db)
    report = db.link_patch_overlaps()
    assert report["committed"] and not report["failed"]
    assert report["issues"] == [] and report["problems"] == []


@pytest.mark.parametrize("kind,detector", [
    ("point_collection", "link_point_collections_to_patches"),
    ("fiber", "link_fibers_to_patches"),
    ("track", "link_tracks_to_patches"),
    ("patch", "link_patch_overlaps"),
])
def test_frame_preflight_collects_all_failures_without_writes(db, kind, detector):
    good_patch = framed_patch(db)
    good_collection = framed_collection(db)
    kept = detector_edge(db, good_collection, good_patch)
    # No geometry: frame validation must finish before loading these objects.
    bad = [db.upsert_object(dict(kind=k, meta={"coordinates": coordinates}))
           for k, coordinates in ((kind, {}), (kind, frame(volume="other")),
                                  ("patch", frame(5e-324)))]
    before = db.journal_since()
    options = dict(volume_uuid=VOLUME)
    if kind in ("fiber", "track"):
        options["ct_volume"] = "/missing-ct"
    report = getattr(db, detector)(**options)
    assert report["failed"] and not report["committed"]
    issues = failures(report)
    assert sorted(i["object_uuids"][0] for i in issues) == sorted(bad)
    assert all(i["stage"] == "validate" and i["count"] == 1 for i in issues)
    assert {i["reason"] for i in issues} == {
        "unknown geometry coordinate frame",
        "geometry volume_uuid does not match target volume",
        "geometry coordinate conversion is not representable",
    }
    assert report["problems"] == [f'{i["object_uuids"][0]}: {i["reason"]}' for i in issues]
    assert report["edges_created"] == report["edges_updated"] == report["stale_removed"] == 0
    assert db.journal_since() == before
    assert db.edge(kept)["uuid"] == kept
