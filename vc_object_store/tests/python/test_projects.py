"""A store as a project: its record and the resources it references."""
import pytest
import vc_object_store as vos
from vc_object_store import ObjectStore


def test_project_record_and_resources(tmp_path):
    path = tmp_path / "project.vcobj"
    vos.create_project(str(path), "scroll", {"open_data": {"sample": "PHerc0139"}}, author="pytest")
    with ObjectStore.open(str(path), author="pytest") as store:
        record = store.project_record()
        assert record["name"] == "scroll" and record["meta"]["open_data"]["sample"] == "PHerc0139"
        uuid = store.add_project_resource("volume", "s3://bucket/scan.zarr/", ["vc-remote-auth:anonymous"])
        assert uuid == vos.project_resource_uuid("volume", "s3://bucket/scan.zarr/")
        store.add_project_resource("normal_grid", "/data/normals")
        [volume] = store.project_resources("volume")
        assert volume["location"] == "s3://bucket/scan.zarr/" and volume["tags"] == ["vc-remote-auth:anonymous"]
        assert [grid["location"] for grid in store.project_resources("normal_grid")] == ["/data/normals"]
        mine = store.add_project_resource("volume", "/home/me/scan.zarr", owner="me")
        assert mine == vos.project_resource_uuid("volume", "/home/me/scan.zarr", "me")
        assert [v["owner"] for v in store.project_resources("volume", viewer="me")] == ["", "me"]
        assert [v["owner"] for v in store.project_resources("volume", viewer="you")] == [""]
        with pytest.raises(Exception):
            store.add_project_resource("surface", "/not/a/resource")
    with pytest.raises(Exception):
        vos.create_project(str(path), "again", author="pytest")


def test_plain_store_has_no_project_record(db):
    assert db.project_record() is None
    assert db.project_resources("volume") == []
