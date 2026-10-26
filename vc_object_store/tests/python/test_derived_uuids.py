"""Ingest of sources without canonical UUIDs, using uuid5 identities from a namespace."""
import json
import uuid

import pytest
from vc_object_store import ObjectStore, Patch

NAMESPACE = "4d388e94-79e1-5047-aed1-a901c3dcb146"


def derived(name):
    return str(uuid.uuid5(uuid.UUID(NAMESPACE), name))


def write_v3_fibers(directory, kept_uuid=None):
    """Two linked VC3D version-3 fibers named only by file, as in older datasets."""
    directory.mkdir()
    names = ["anon_0", "anon_1"]
    for i, name in enumerate(names):
        other = names[1 - i]
        document = dict(type="vc3d_fiber", version=3, filename=name + ".json", sequence=i,
                        control_points=[{"position": [i, 0, 0]}, {"position": [i, 1, 0]}],
                        line_points=[[i, 0, 0], [i, 1, 0]],
                        branches=[dict(branch_file=other + ".json", branch_fiber_id=1 - i,
                                       control_point_index=i, branch_control_point_index=1 - i,
                                       control_point_position=[0, 0, 0], branch_control_point_position=[0, 0, 0],
                                       control_point_direction=[1, 0, 0] if i == 0 else [0, 1, 0],
                                       branch_control_point_direction=[0, 1, 0] if i == 0 else [1, 0, 0])])
        if i == 1 and kept_uuid:
            document["uuid"] = kept_uuid
        (directory / (name + ".json")).write_text(json.dumps(document))
    return names


def test_v3_fibers_without_uuids(tmp_path):
    names = write_v3_fibers(tmp_path / "fibers")
    with ObjectStore.create(str(tmp_path / "plain.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers")
        assert report["ingested"] == 0 and report["problems"], report
        assert store.object_count("fiber") == 0
    with ObjectStore.create(str(tmp_path / "derived.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE)
        assert report == dict(ingested=2, edges_imported=1, problems=[], notices=[])
        ids = [derived("fiber:" + name) for name in names]
        edge = store.edges_by_kind("fiber_branch")[0]
        assert {edge["a"], edge["b"]} == set(ids)
        source = store.object(ids[0])["meta"]["source"]
        assert source["uuid_derivation"] == dict(namespace=NAMESPACE, name="fiber:anon_0")
        assert store.object(ids[0])["name"] == "anon_0"
        # The same namespace maps the same files to the same identities, so a repeat is refused.
        again = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE)
        assert again["ingested"] == 0 and "already exists" in " ".join(again["problems"])


def test_canonical_fiber_uuids_are_kept_and_branches_resolve_to_them(tmp_path):
    kept = str(uuid.uuid4())
    write_v3_fibers(tmp_path / "fibers", kept_uuid=kept)
    with ObjectStore.create(str(tmp_path / "mixed.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE)
        assert report == dict(ingested=2, edges_imported=1, problems=[], notices=[])
        edge = store.edges_by_kind("fiber_branch")[0]
        assert {edge["a"], edge["b"]} == {derived("fiber:anon_0"), kept}
        assert "uuid_derivation" not in store.object(kept)["meta"]["source"]


def test_point_collections_without_uuids(tmp_path):
    source = tmp_path / "same_windings.json"
    source.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": {
        "3": {"name": "a", "points": {"10": {"p": [1.0, 2.0, 3.0]}}},
        "4": {"name": "b", "uuid": str(uuid.uuid4()), "points": {"11": {"p": [4.0, 5.0, 6.0]}}},
    }}))
    with ObjectStore.create(str(tmp_path / "pcl.vcobj"), author="test") as store:
        assert store.ingest_geometry_source("point_collection", source)["problems"]
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE)
        assert report["ingested"] == 2 and not report["problems"], report
        uid = derived("point_collection:same_windings:3")
        assert store.object(uid)["name"] == "a"
        assert store.object(uid)["meta"]["source"]["source_collection_id"] == "3"


def test_patch_named_by_directory(db, tmp_path):
    uid = db.upsert_object(dict(kind="patch"))
    db.put_geometry(uid, Patch(1, 2, [1, 1], [[1, 2, 3], [4, 5, 6]]))
    directory = tmp_path / "0000_low_band_final"
    db.export_geometry(uid, directory)
    meta = json.loads((directory / "meta.json").read_text())
    # VC3D stores the directory name as a tifxyz "uuid".
    meta["uuid"] = directory.name
    meta.pop("name", None)
    (directory / "meta.json").write_text(json.dumps(meta))
    with ObjectStore.create(str(tmp_path / "patch.vcobj"), author="test") as store:
        assert store.ingest_geometry_source("patch", directory)["problems"]
        report = store.ingest_geometry_source("patch", directory, uuid_namespace=NAMESPACE)
        assert report["ingested"] == 1 and not report["problems"], report
        assert store.object(derived("patch:0000_low_band_final"))["name"] == "0000_low_band_final"
    del meta["uuid"]
    (directory / "meta.json").write_text(json.dumps(meta))
    with ObjectStore.create(str(tmp_path / "unnamed.vcobj"), author="test") as store:
        assert store.ingest_geometry_source("patch", str(directory) + "/", uuid_namespace=NAMESPACE)["ingested"] == 1
        assert store.object(derived("patch:0000_low_band_final"))


@pytest.mark.parametrize("namespace", ["spiral", ""])
def test_namespace_must_be_a_uuid(tmp_path, namespace):
    write_v3_fibers(tmp_path / "fibers")
    with ObjectStore.create(str(tmp_path / "bad.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=namespace)
        assert report["ingested"] == 0 and "invalid UUID" in " ".join(report["problems"])


def edit_v3_fiber(directory, name, change):
    path = directory / (name + ".json")
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_v3_reciprocal_branches_are_reconciled_by_the_relationship_importer(tmp_path):
    directory = tmp_path / "fibers"
    names = write_v3_fibers(directory)

    def last_bit(document):
        # Each end computes the shared directions separately.
        document["branches"][0]["control_point_direction"] = [0, 1 + 2.0**-52, 0]
    edit_v3_fiber(directory, names[1], last_bit)
    with ObjectStore.create(str(tmp_path / "close.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", directory, uuid_namespace=NAMESPACE)
        assert report == dict(ingested=2, edges_imported=1, problems=[], notices=[])

    def adjacent(document):
        document["adjacent_branches"] = document.pop("branches")
    edit_v3_fiber(directory, names[1], adjacent)
    with ObjectStore.create(str(tmp_path / "kinds.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", directory, uuid_namespace=NAMESPACE)
        assert "conflicting ordinary/adjacent" in " ".join(report["problems"])
        assert store.object_count("fiber") == 0


@pytest.mark.parametrize("declared,ok", [
    ({"volume_uuid": "B", "scale_factor": 1}, True),
    ({"volume_uuid": "B"}, True),
    ({"volume_uuid": "A", "scale_factor": 2}, False),
    ({"scale_factor": 2}, False),
])
def test_v3_coordinate_space_merges_with_declared_coordinates(tmp_path, declared, ok):
    directory = tmp_path / "fibers"
    names = write_v3_fibers(directory)
    for name in names:
        edit_v3_fiber(directory, name, lambda d: d.update(
            coordinates=declared, vc_open_data_coordinate_space="PHerc/B@L0"))
    with ObjectStore.create(str(tmp_path / "frames.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", directory, uuid_namespace=NAMESPACE)
        if ok:
            assert report["ingested"] == 2 and not report["problems"], report
            coordinates = store.object(derived("fiber:" + names[0]))["meta"]["coordinates"]
            assert coordinates == {"volume_uuid": "B", "scale_factor": 1}
        else:
            assert "coordinate conflict" in " ".join(report["problems"]), report
            assert store.object_count("fiber") == 0
