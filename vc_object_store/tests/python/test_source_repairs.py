"""Opt-in ingest repairs for dataset defects: dangling fiber branches and colliding PCL point IDs."""
import json
import uuid

from test_derived_uuids import NAMESPACE, derived, write_v3_fibers
from vc_object_store import ObjectStore


def add_dangling_branch(directory, name, target="missing.json"):
    path = directory / (name + ".json")
    document = json.loads(path.read_text())
    document["branches"].append(dict(branch_file=target, branch_fiber_id=7, control_point_index=0,
                                     branch_control_point_index=0, control_point_position=[0, 0, 0],
                                     branch_control_point_position=[0, 0, 0]))
    path.write_text(json.dumps(document))


def test_dangling_branches_fail_unless_dropped(tmp_path):
    names = write_v3_fibers(tmp_path / "fibers")
    add_dangling_branch(tmp_path / "fibers", names[0])
    with ObjectStore.create(str(tmp_path / "strict.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE)
        assert report["ingested"] == 0 and "branch target file is absent" in " ".join(report["problems"]), report
        assert store.object_count("fiber") == 0
    with ObjectStore.create(str(tmp_path / "dropped.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE,
                                              drop_dangling_branches=True)
        assert report["ingested"] == 2 and report["edges_imported"] == 1 and not report["problems"], report
        owner = derived("fiber:" + names[0])
        assert report["notices"] == [f"fiber {owner} (anon_0): dropped 1 branch declarations to absent files: missing.json"]
        dropped = store.object(owner)["meta"]["source"]["dropped_branches"]
        assert [(d["branch_file"], d["declaration"]) for d in dropped] == [("missing.json", "branches")]
        assert "dropped_branches" not in store.object(derived("fiber:" + names[1]))["meta"]["source"]
        # The resolvable reciprocal link is unaffected.
        edge = store.edges_by_kind("fiber_branch")[0]
        assert {edge["a"], edge["b"]} == {derived("fiber:" + n) for n in names}


def test_dropping_keeps_resolvable_branches_and_needs_no_flag_without_defects(tmp_path):
    write_v3_fibers(tmp_path / "fibers")
    with ObjectStore.create(str(tmp_path / "clean.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", tmp_path / "fibers", uuid_namespace=NAMESPACE,
                                              drop_dangling_branches=True)
        assert report == dict(ingested=2, edges_imported=1, problems=[], notices=[])


def write_colliding_collections(path, links=False):
    points = lambda ids, z: {str(i): {"p": [float(i), 0.0, z], "wind_a": i} for i in ids}
    collections = {
        "1": {"name": "global", "points": points([5, 9], 1.0)},
        # These two number from zero, colliding with each other but not with "1".
        "2": {"name": "first", "points": points([1, 0], 2.0)},
        "10": {"name": "second", "points": points([0, 2], 3.0)},
    }
    if links:
        collections["1"]["points"]["5"]["links"] = [9]
    path.write_text(json.dumps({"vc_pointcollections_json_version": "1", "collections": collections}))


def test_colliding_point_ids_fail_unless_renumbered(tmp_path):
    source = tmp_path / "same_windings.json"
    write_colliding_collections(source)
    with ObjectStore.create(str(tmp_path / "strict.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE)
        assert report["ingested"] == 0, report
        assert "2 collections (10, 2) share point IDs" in " ".join(report["problems"])
    with ObjectStore.create(str(tmp_path / "renumbered.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE,
                                              renumber_point_ids=True)
        assert report["ingested"] == 3 and not report["problems"], report
        assert report["notices"] == [f"{source}: renumbered point IDs of 2 collections sharing IDs; "
                                     "file IDs kept as source_point_id"]
        uid = lambda cid: derived("point_collection:same_windings:" + cid)
        with store.geometry_snapshot() as snapshot:
            geometry = {cid: snapshot.point_collection(uid(cid)).load() for cid in ("1", "2", "10")}
        # The non-colliding collection keeps its IDs; the others get fresh IDs above the
        # document's largest (9), in collection then point-ID order, preserving point order.
        assert list(geometry["1"].ids) == [5, 9]
        assert list(geometry["10"].ids) == [10, 11]
        assert list(geometry["2"].ids) == [12, 13]
        assert [list(p) for p in geometry["10"].points] == [[0.0, 0.0, 3.0], [2.0, 0.0, 3.0]]
        source_meta = store.object(uid("10"))["meta"]["source"]
        assert source_meta["point_ids_renumbered"] is True
        annotations = lambda cid: store.geometry_point_annotations(store.current_geometry(uid(cid))["uuid"], "points")
        assert [(a["source_point_id"], a["wind_a"]) for a in annotations("10")] == [(0, 0), (2, 2)]
        assert "point_ids_renumbered" not in store.object(uid("1"))["meta"]["source"]
        assert all("source_point_id" not in a for a in annotations("1"))


def test_renumbering_refuses_documents_with_point_links(tmp_path):
    source = tmp_path / "linked.json"
    write_colliding_collections(source, links=True)
    with ObjectStore.create(str(tmp_path / "linked.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE,
                                              renumber_point_ids=True)
        assert report["ingested"] == 0 and "point links" in " ".join(report["problems"]), report
