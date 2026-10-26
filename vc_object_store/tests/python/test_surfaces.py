import numpy as np
from vc_object_store import ObjectStore, Surface
from vc_object_store.surfaces import create_surface, read_surface


def test_created_surface_reads_back_through_registrations(db_path):
    ObjectStore.create(db_path, author="pytest").close()
    xyz = np.array([[[1, 2, 3], [4, 5, 6]], [[-1, -1, -1], [np.nan, 0, 0]]], dtype=np.float32)
    uid = create_surface(db_path, xyz, name="grown", scale=0.05, volume="a", tool="pytest",
                         parameters={"seed": 7}, sources=["src"], tags=["paths"],
                         meta={"format": "tifxyz", "uuid": "old", "area_vx2": 12.0},
                         channels={"approval": np.full((2, 2), 9, np.uint8)})
    with ObjectStore.open(db_path, author="pytest") as db:
        record = db.object(uid)
        assert record["tags"] == ["paths"]
        assert record["meta"]["coordinates"] == {"volume_uuid": "a", "scale_factor": 1}
        assert record["meta"]["provenance"] == {"tool": "pytest", "parameters": {"seed": 7}, "sources": ["src"]}
        assert record["meta"]["area_vx2"] == 12.0 and "format" not in record["meta"] and "uuid" not in record["meta"]
        db.register_volume_transform("a", "b", [[2, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
    points, valid, scale, _, revision, channels = read_surface(db_path, uid, volume="a")
    assert channels == {"approval"} and revision
    np.testing.assert_array_equal(valid, [[True, True], [False, False]])
    np.testing.assert_array_equal(points[0], xyz[0])
    assert (points[1] == -1).all() and scale == (0.05, 0.05)
    points, valid, *_ = read_surface(db_path, uid, volume="b")
    np.testing.assert_array_equal(points[0], [[12, 2, 3], [18, 5, 6]])
    assert (points[1] == -1).all()


def test_older_revisions_read_in_the_frame_they_were_stored_in(db_path):
    ObjectStore.create(db_path, author="pytest").close()
    xyz = np.array([[[2, 4, 6], [8, 10, 12]]], dtype=np.float32)
    uid = create_surface(db_path, xyz, name="s", scale=1, volume="a", tool="pytest")
    with ObjectStore.open(db_path, author="pytest") as db:
        first = db.current_geometry(uid)["uuid"]
        db.put_geometry(uid, Surface.from_arrays(np.ascontiguousarray(xyz * 2), [1.0, 1.0]), expected_revision=first)
        db.set_meta(uid, "coordinates", {"volume_uuid": "c", "scale_factor": 2})
    current = read_surface(db_path, uid, volume="c")
    older = read_surface(db_path, uid, volume="a", geometry_revision=first)
    np.testing.assert_array_equal(current.points, older.points)
    assert older.geometry_revision == first != current.geometry_revision
