"""Detector outcomes: an object that cannot be loaded is reported, skipped and
keeps its edges; invalid coordinate frames refuse the run before any write."""
import pytest

from object_helpers import VOLUME, array, frame, framed_collection, framed_patch, plane, put


def collection(db, point, winding=1):
    return put(db, "point_collection", [array("points", [point], "<f8"), array("point_ids", [1], "<i8")],
               meta={"source": {"point_attributes": [{"wind_a": winding}]}})[0]


def detector_edge(db, a, b):
    """A pending PCL-patch edge that a rerun will not propose again."""
    whole = dict(whole_geometry=True)
    return db.add_edge(dict(a=a, b=b, kind="point-surface", a_anchor=whole, b_anchor=whole,
                            meta={"detector": {"name": "point-collection-surface", "version": 1}}))


def test_failed_collection_is_reported_skipped_and_keeps_its_edges(db):
    patch, _ = plane(db)
    good = collection(db, [10, 10, 20])
    bad = collection(db, [12, 12, 20], winding="not a number")
    unproposed, kept = detector_edge(db, good, patch), detector_edge(db, bad, patch)
    report = db.link_point_collections_to_surfaces()
    assert report["problems"] == [f"{bad}: wind_a must be numeric or null, skipped"]
    # Independent work commits; only objects that loaded are in stale-edge scope.
    assert report["edges_created"] == 1 and report["stale_removed"] == 1
    assert db.edge(kept)["uuid"] == kept
    assert unproposed not in {e["uuid"] for e in db.edges_of(good)}
    again = db.link_point_collections_to_surfaces()
    assert again["edges_skipped_existing"] == 1 and again["edges_created"] == again["stale_removed"] == 0


def test_missing_geometry_is_reported_for_each_detector(db):
    patch, _ = plane(db)
    empty_patch = db.upsert_object({"kind": "surface", "storage_ref": "/missing/source"})
    empty_collection = db.upsert_object({"kind": "point_collection", "storage_ref": "/missing/source"})
    report = db.link_point_collections_to_surfaces()
    assert sorted(problem.split(":")[0] for problem in report["problems"]) == sorted([empty_patch, empty_collection])
    report = db.link_surface_overlaps()
    assert [problem.split(":")[0] for problem in report["problems"]] == [empty_patch]


def test_rejected_overlap_pairs_are_reported(db):
    plane(db)
    plane(db, offset=16)  # overlaps half of the first patch
    report = db.link_surface_overlaps(min_overlap_fraction=0.9)
    assert report["edges_created"] == 0
    assert report["problems"] == ["surface-overlap: 1 pair(s) rejected: insufficient_overlap_fraction"]


@pytest.mark.parametrize("kind,detector", [
    ("point_collection", "link_point_collections_to_surfaces"),
    ("fiber", "link_fibers_to_surfaces"),
    ("track", "link_tracks_to_surfaces"),
    ("surface", "link_surface_overlaps"),
])
def test_frame_preflight_refuses_the_run_without_writes(db, kind, detector):
    good_patch = framed_patch(db)
    good_collection = framed_collection(db)
    kept = detector_edge(db, good_collection, good_patch)
    # No geometry: frame validation must finish before loading these objects.
    bad = [db.upsert_object(dict(kind=k, meta={"coordinates": coordinates}))
           for k, coordinates in ((kind, {}), (kind, frame(volume="other")),
                                  ("surface", frame(5e-324)))]
    before = db.journal_since()
    options = dict(volume_uuid=VOLUME)
    if kind in ("fiber", "track"):
        options["ct_volume"] = "/missing-ct"
    with pytest.raises(ValueError) as refused:
        getattr(db, detector)(**options)
    message = str(refused.value)
    assert all(uuid in message for uuid in bad), message
    for reason in ("unknown geometry coordinate frame", "geometry volume_uuid does not match target volume",
                   "geometry coordinate conversion is not representable"):
        assert reason in message, message
    assert db.journal_since() == before
    assert db.edge(kept)["uuid"] == kept
