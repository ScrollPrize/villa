import pytest
import numpy as np
from vc_object_store import ObjectStore, Surface
from vc_object_store.surfaces import create_surface, read_surface


def test_exact_surface_reference_survives_deletion(db, db_path):
    from vc_object_store import Surface, object_reference
    uid = db.upsert_object(dict(kind='surface', meta={'coordinates': {'volume_uuid': 'a', 'scale_factor': 1}}))
    points = np.full((2, 2, 3), 4, dtype='<f4')
    db.put_geometry(uid, Surface.from_arrays(points, [1, 1]))
    reference = object_reference(db.object(uid))
    db.delete_object(uid)
    grid = read_surface(db_path, uid, volume='a', reference=reference)
    np.testing.assert_array_equal(grid.points, points)
    assert object_reference(grid.record) == reference
    with pytest.raises(ValueError, match='cannot be combined'):
        read_surface(db_path, uid, volume='a', version=reference['version'], reference=reference)


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
        # The store keeps the area itself (one valid row: no cell has four valid corners).
        assert "area_vx2" not in record["meta"] and "format" not in record["meta"] and "uuid" not in record["meta"]
        assert record["surface"]["area"] == 0 and record["surface"]["grid_width"] == 2
        db.register_volume_transform("a", "b", [[2, 0, 0, 10], [0, 1, 0, 0], [0, 0, 1, 0]])
    grid = read_surface(db_path, uid, volume="a")
    points, valid, scale = grid.points, grid.valid, grid.scale
    assert grid.channels == {"approval"} and grid.geometry_revision and grid.version == 1 and grid.generation
    np.testing.assert_array_equal(valid, [[True, True], [False, False]])
    np.testing.assert_array_equal(points[0], xyz[0])
    assert (points[1] == -1).all() and scale == (0.05, 0.05)
    grid = read_surface(db_path, uid, volume="b")
    points, valid = grid.points, grid.valid
    np.testing.assert_array_equal(points[0], [[12, 2, 3], [18, 5, 6]])
    assert (points[1] == -1).all()


def test_older_revisions_read_in_the_frame_they_were_stored_in(db_path):
    ObjectStore.create(db_path, author="pytest").close()
    xyz = np.array([[[2, 4, 6], [8, 10, 12]]], dtype=np.float32)
    uid = create_surface(db_path, xyz, name="s", scale=1, volume="a", tool="pytest")
    with ObjectStore.open(db_path, author="pytest") as db:
        first = db.current_geometry(uid)["uuid"]
        db.put_geometry(uid, Surface.from_arrays(np.ascontiguousarray(xyz * 2), [1.0, 1.0]),
                        expected_version=db.current_version(uid))
        db.set_meta(uid, "coordinates", {"volume_uuid": "c", "scale_factor": 2})
    current = read_surface(db_path, uid, volume="c")
    older = read_surface(db_path, uid, volume="a", version=1)
    np.testing.assert_array_equal(current.points, older.points)
    assert older.geometry_revision == first != current.geometry_revision
    assert (older.version, current.version) == (1, 3) and older.record["meta"]["coordinates"]["volume_uuid"] == "a"
    with pytest.raises(ValueError, match="version 9"):
        read_surface(db_path, uid, volume="a", version=9)
    assert read_surface(db_path, uid, volume="a", version=1, generation=older.generation).geometry_revision == first
    with pytest.raises(ValueError, match="version 1"):
        read_surface(db_path, uid, volume="a", version=1, generation="00000000-0000-4000-8000-000000000000")


def test_numbered_surface_uses_its_metadata_when_geometry_is_shared(db_path):
    ObjectStore.create(db_path, author="pytest").close()
    xyz = np.array([[[2, 4, 6], [8, 10, 12]]], dtype=np.float32)
    uid = create_surface(db_path, xyz, name="s", scale=1, volume="a", tool="pytest")
    with ObjectStore.open(db_path, author="pytest") as db:
        db.set_meta(uid, "coordinates", {"volume_uuid": "a", "scale_factor": 2})
    first = read_surface(db_path, uid, volume="a", version=1)
    current = read_surface(db_path, uid, volume="a")
    assert first.geometry_revision == current.geometry_revision
    np.testing.assert_array_equal(first.points, xyz)
    np.testing.assert_array_equal(current.points, xyz / 2)
    assert first.record["meta"]["coordinates"]["scale_factor"] == 1
