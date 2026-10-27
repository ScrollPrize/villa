from vc_object_store import ObjectStore, Track


def test_catalog_input_supplies_shared_native_registrations(tmp_path):
    catalog = {"samples": {"sample": {"sample": {"properties": {"volume_transforms": [{
        "from_volume_id": "a", "transforms": [{"to_volume_id": "b", "matrix":
        [[2, 0, 0, 10], [0, 2, 0, 0], [0, 0, 2, 0]]}]}]}}}}}
    with ObjectStore.create(str(tmp_path / "catalog.vcobj"), author="test", catalog=catalog) as db:
        uid = db.upsert_object(dict(kind="track", meta={"coordinates": {"volume_uuid": "a", "scale_factor": 2}}))
        db.put_geometry(uid, Track([[2, 4, 6]]))
        with db.geometry_snapshot() as snapshot:
            assert snapshot.track(uid).view_transformed_at("b").points("points", 0, 1) == [[12, 4, 6]]
