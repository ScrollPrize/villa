"""link_objects from Python: one request runs the plan's detectors over a snapshot and
publishes their links with the link marker in one commit."""
import uuid

import numpy as np
import pytest

import vc_object_store as vos


def array(name, values, dtype):
    data = np.asarray(values, dtype=dtype)
    return {"name": name, "dtype": dtype, "shape": list(data.shape), "data": data.tobytes()}


def put(db, kind, arrays, attributes=None, **record):
    uid = db.upsert_object({"kind": kind, "storage_ref": "/missing/source", **record})
    db.put_geometry(uid, {"kind": kind, "attributes": attributes or {}, "arrays": arrays})
    return uid


def plane(db, offset=0):
    y, x = np.mgrid[:33, :33]
    points = np.stack([x + offset, y, np.full_like(x, 20)], axis=-1)
    return put(db, "surface", [array("points", points, "<f4")], {"scale": [1, 1]}, name="surface")


def collection(db):
    meta = {"source": {"point_attributes": [{"wind_a": 3}, {"wind_a": 4}], "winding_is_absolute": True}}
    return put(db, "point_collection", [array("points", [[10, 10, 40], [20, 20, 40]], "<f8"),
                                        array("point_ids", [90, 7], "<i8")], meta=meta)


def link_request(status, plan):
    return dict(command_version=1, request=str(uuid.uuid4()), store=status["store"], generation=status["generation"],
                author="py", expected_objects={}, expected_relationships={}, read_dependencies={},
                expected_commit_sequence=None, operations=[{"op": vos.LINK_OBJECTS_OPERATION, "plan": plan}])


PLAN = {
    "detectors": [
        {"detector": "point-collection-surface", "params": {"point_scale": 0.5, "window_points": 3, "window_min_points": 3}},
        {"detector": "surface-overlap", "params": {}},
    ],
    "marker": {"key": "fixture"},
}


def test_link_objects_publishes_links_and_marker(db):
    a, b, points = plane(db), plane(db), collection(db)
    stages = []
    request = link_request(db.status(), PLAN)
    receipt = db.submit(request, progress=lambda stage, completed, total, detail: stages.append(stage))
    assert receipt["commit_sequence"] == db.status()["commit_sequence"]
    assert any(stage.startswith("surface-overlap: ") for stage in stages)
    assert len(db.select_edges("point-surface")) == 2
    assert len(db.select_edges("same-surface")) == 1
    marker = db.object(vos.link_marker_uuid())
    assert marker["kind"] == vos.LINK_MARKER_KIND
    assert marker["meta"]["key"] == "fixture"
    assert marker["meta"]["commit_sequence"] == receipt["commit_sequence"]
    reports = [step["report"] for step in marker["meta"]["detectors"]]
    assert [r["edges_created"] for r in reports] == [2, 1]
    assert receipt["objects"][vos.link_marker_uuid()] == 1
    assert set(receipt["objects"]) >= {a, b, points}

    again = db.submit(request)
    assert again["replayed"] and again["commit_sequence"] == receipt["commit_sequence"]

    rerun = db.submit(link_request(db.status(), PLAN))
    assert rerun["commit_sequence"] == receipt["commit_sequence"] + 1
    assert len(db.select_edges("same-surface")) == 1


def test_link_objects_refuses_a_bad_plan(db):
    plane(db)
    bad = {"detectors": [{"detector": "surface-overlap", "params": {"tolerence": 1}}], "marker": {}}
    with pytest.raises(ValueError):
        db.submit(link_request(db.status(), bad))
    assert db.object(vos.link_marker_uuid()) is None


def test_submit_retains_store_through_reentrant_progress(db):
    plane(db), plane(db)
    closes = []

    def progress(*_):
        with pytest.raises(RuntimeError, match="cannot close"):
            db.close()
        closes.append(True)

    receipt = db.submit(link_request(db.status(), {
        "detectors": [{"detector": "surface-overlap", "params": {}}], "marker": {},
    }), progress=progress)
    assert closes
    assert receipt["commit_sequence"] == db.status()["commit_sequence"]


def test_link_objects_fails_when_a_detector_cannot_run(db):
    plane(db), plane(db), collection(db)
    put(db, "track", [array("points", [[1, 1, 20], [8, 8, 20], [16, 16, 20]], "<i4")], {"family": "h"},
        track={"dataset_ref": "dataset", "external_key": "0", "family": "h"})
    before = db.status()["commit_sequence"]
    failing = {"detectors": PLAN["detectors"] + [{"detector": "track-surface", "ct_volume": "/no/such/ct.zarr", "params": {}}],
               "marker": {}}
    with pytest.raises(vos.PreparationFailed, match="track-surface"):
        db.submit(link_request(db.status(), failing))
    assert db.status()["commit_sequence"] == before
    assert db.select_edges("point-surface") == []
    assert db.object(vos.link_marker_uuid()) is None
    # fiber-surface throws on a missing CT volume rather than reporting it;
    # the request fails the same way.
    throwing = {"detectors": PLAN["detectors"] + [{"detector": "fiber-surface", "ct_volume": "/no/such/ct.zarr", "params": {}}],
                "marker": {}}
    with pytest.raises(vos.PreparationFailed, match="fiber-surface"):
        db.submit(link_request(db.status(), throwing))
    assert db.status()["commit_sequence"] == before
    assert db.object(vos.link_marker_uuid()) is None
