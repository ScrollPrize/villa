import numpy as np
from vc_object_store import ObjectStore
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
    points, valid, scale, _ = read_surface(db_path, uid, volume="a")
    np.testing.assert_array_equal(valid, [[True, True], [False, False]])
    np.testing.assert_array_equal(points[0], xyz[0])
    assert (points[1] == -1).all() and scale == (0.05, 0.05)
    points, valid, scale, _ = read_surface(db_path, uid, volume="b")
    np.testing.assert_array_equal(points[0], [[12, 2, 3], [18, 5, 6]])
    assert (points[1] == -1).all()
