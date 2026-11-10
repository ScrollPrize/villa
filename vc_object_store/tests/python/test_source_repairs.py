"""Opt-in ingest repairs for dataset defects: dangling fiber branches and colliding PCL point IDs."""
import json

from test_derived_uuids import NAMESPACE, derived, edit_v3_fiber, write_v3_fibers
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
        assert len(report["notices"]) == 1 and "missing.json" in report["notices"][0]
        dropped = store.object(derived("fiber:" + names[0]))["meta"]["source"]["dropped_branches"]
        assert [(d["branch_file"], d["declaration"]) for d in dropped] == [("missing.json", "branches")]
        assert "dropped_branches" not in store.object(derived("fiber:" + names[1]))["meta"]["source"]
        # The resolvable reciprocal link is unaffected.
        edge = store.edges_by_kind("fiber_branch")[0]
        assert {edge["a"], edge["b"]} == {derived("fiber:" + n) for n in names}


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
        assert "share point IDs" in " ".join(report["problems"])
    with ObjectStore.create(str(tmp_path / "renumbered.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE,
                                              renumber_point_ids=True)
        assert report["ingested"] == 3 and not report["problems"], report
        assert len(report["notices"]) == 1
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


def test_maximum_point_id_needs_no_successor_until_repair(tmp_path):
    source = tmp_path / "maximum.json"
    maximum = 2**63 - 1
    document = {"vc_pointcollections_json_version": "1", "collections": {
        "1": {"name": "maximum", "points": {str(maximum): {"p": [0, 0, 0]}}},
    }}
    source.write_text(json.dumps(document))
    with ObjectStore.create(str(tmp_path / "maximum.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE)
        assert report["ingested"] == 1 and not report["problems"], report
        with store.geometry_snapshot() as snapshot:
            geometry = snapshot.point_collection(derived("point_collection:maximum:1")).load()
            assert list(geometry.ids) == [maximum]
    document["collections"]["2"] = {"name": "collision", "points": {str(maximum): {"p": [1, 0, 0]}}}
    source.write_text(json.dumps(document))
    with ObjectStore.create(str(tmp_path / "exhausted.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE,
                                              renumber_point_ids=True)
        assert report["ingested"] == 0 and "ID range exhausted" in " ".join(report["problems"]), report
        assert store.object_count("point_collection") == 0
    # Two repaired points may consume the last two IDs exactly; no successor
    # is required after assigning the final representable ID.
    document["collections"] = {
        "1": {"name": "high", "points": {str(maximum - 2): {"p": [0, 0, 0]}}},
        "2": {"name": "first", "points": {"0": {"p": [1, 0, 0]}}},
        "3": {"name": "second", "points": {"0": {"p": [2, 0, 0]}}},
    }
    source.write_text(json.dumps(document))
    with ObjectStore.create(str(tmp_path / "last_ids.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("point_collection", source, uuid_namespace=NAMESPACE,
                                              renumber_point_ids=True)
        assert report["ingested"] == 3 and not report["problems"], report
        with store.geometry_snapshot() as snapshot:
            ids = [list(snapshot.point_collection(derived(f"point_collection:maximum:{cid}")).load().ids)
                   for cid in ("1", "2", "3")]
        assert ids == [[maximum - 2], [maximum - 1], [maximum]]


def test_v3_pending_auto_branches_survive_shared_reconciliation(tmp_path):
    directory = tmp_path / "fibers"
    names = write_v3_fibers(directory)
    for name in names:
        edit_v3_fiber(directory, name, lambda document: document["branches"][0].update(pending=True, auto=True))
    # The shared importer must tolerate separately computed directions without
    # dropping the reciprocal declaration before checking upstream's auto flag.
    edit_v3_fiber(directory, names[1], lambda document: document["branches"][0].update(
        control_point_direction=[0, 1 + 2.0**-52, 0]))
    with ObjectStore.create(str(tmp_path / "auto.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", directory, uuid_namespace=NAMESPACE)
        assert {k: report[k] for k in ("ingested", "edges_imported", "problems", "notices")} == dict(ingested=2, edges_imported=1, problems=[], notices=[])
        [edge] = store.edges_by_kind("fiber_branch")
        assert edge["approval_state"] == "pending" and edge["meta"]["auto"] is True
        exported = tmp_path / "exported"
        exported.mkdir()
        for name in names:
            store.export_geometry(derived("fiber:" + name), exported / (name + ".json"))
            document = json.loads((exported / (name + ".json")).read_text())
            assert document["branches"][0]["pending"] is True
            assert document["branches"][0]["auto"] is True
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as store:
        report = store.ingest_geometry_source("fiber", exported)
        assert {k: report[k] for k in ("ingested", "edges_imported", "problems", "notices")} == dict(ingested=2, edges_imported=1, problems=[], notices=[])
        [restored] = store.edges_by_kind("fiber_branch")
        assert restored["uuid"] == edge["uuid"] and restored["meta"]["auto"] is True
