import numpy as np
import pytest
from vc_object_store import ObjectStore, PointCollection, Fiber, Track, Surface, Umbilicus


@pytest.mark.parametrize("kind,geometry,array", [
    ("point_collection", PointCollection([7, 19], [[1, 2, 3], [4, 5, 6]]), "points"),
    ("fiber", Fiber([[1, 2, 3], [4, 5, 6]], []), "control_points"),
    ("track", Track([[1, 2, 3], [4, 5, 6]]), "points"),
    ("surface", Surface(1, 2, [1, 1], [[1, 2, 3], [4, 5, 6]]), "points"),
    ("umbilicus", Umbilicus([[1, 2, 3], [4, 5, 6]]), "points"),
])
def test_explicit_view_preserves_canonical_geometry(db, kind, geometry, array):
    uid = db.upsert_object(dict(kind=kind, meta={"coordinates": {"volume_uuid": "source", "scale_factor": 1}}))
    revision = db.put_geometry(uid, geometry)
    original = db.read_geometry_array(revision, array)
    db.register_volume_transform("source", "target", [[2, .5, 0, 10], [0, 1, 0, 0], [0, 0, 1, -4]])
    assert db.available_transforms(uid)[0]["volume"] == "target"
    with db.geometry_snapshot() as snapshot:
        view = getattr(snapshot, kind)(uid).view_transformed_at("target")
        if kind == "surface":
            tile = view.read_tile(0, 0, 1, 2)
            np.testing.assert_array_equal(tile["points"][0], [[13, 2, -1], [20.5, 5, 2]])
            assert tile["validity"].all()  # A transformed -1 component is not a missing vertex.
            assert db.edit_transformed_surface(view, 0, 0, 1, 2, tile["points"].reshape(-1, 3).tolist()) == revision
        else:
            assert view.points(array, 0, 2) == [[13, 2, -1], [20.5, 5, 2]]
            assert db.edit_transformed_coordinates(view, array, 0, view.points(array, 0, 2)) == revision
        with pytest.raises(RuntimeError, match="unambiguous"):
            snapshot.view_transformed_at(uid, "unreachable")
    assert db.read_geometry_array(revision, array) == original
    assert db.current_geometry(uid)["uuid"] == revision


def test_inverse_track_edit_and_stale_registration(db):
    uid = db.upsert_object(dict(kind="track", meta={"coordinates": {"volume_uuid": "a", "scale_factor": 1}}))
    db.put_geometry(uid, Track([[0, 0, 0], [3, 0, 0]]))
    db.register_volume_transform("a", "b", [[2, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
    with db.geometry_snapshot() as snapshot:
        view = snapshot.track(uid).view_transformed_at("b")
        db.edit_transformed_coordinates(view, "points", 1, [[17, 0, 0]])
    with db.geometry_snapshot() as snapshot:
        assert snapshot.track(uid).points(1, 1) == [[4, 0, 0]]
        view = snapshot.track(uid).view_transformed_at("b")
        db.register_volume_transform("a", "b", [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0]])
        with pytest.raises(RuntimeError, match="registration changed"):
            db.edit_transformed_coordinates(view, "points", 1, [[19, 0, 0]])
    available = db.available_transforms(uid)[0]
    assert available["ambiguous"] and not available["available"]
    db.choose_volume_transform("a", "b", available["candidates"][0]["path"])
    assert db.available_transforms(uid)[0]["available"]
    with db.geometry_snapshot() as snapshot:
        assert snapshot.track(uid).available_transforms[0]["available"]
        assert snapshot.track(uid).view_transformed_at("b").points("points", 0, 1)



def test_catalog_input_supplies_shared_native_registrations(tmp_path):
    catalog = {"samples": {"sample": {"sample": {"properties": {"volume_transforms": [{
        "from_volume_id": "a", "transforms": [{"to_volume_id": "b", "matrix":
        [[2, 0, 0, 10], [0, 2, 0, 0], [0, 0, 2, 0]]}]}]}}}}}
    with ObjectStore.create(str(tmp_path / "catalog.vcobj"), author="test", catalog=catalog) as db:
        uid = db.upsert_object(dict(kind="track", meta={"coordinates": {"volume_uuid": "a", "scale_factor": 2}}))
        db.put_geometry(uid, Track([[2, 4, 6]]))
        with db.geometry_snapshot() as snapshot:
            assert snapshot.track(uid).view_transformed_at("b").points("points", 0, 1) == [[12, 4, 6]]
