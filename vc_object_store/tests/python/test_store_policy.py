"""The Python facade delegates schema-22 policy to the native store implementation."""
import sqlite3
import json
import uuid

import pytest

from vc_object_store import (
    Fiber, Surface, Track, Umbilicus, PointCollection, ObjectStore,
    enable_history, remove_history_and_disable,
)
from object_helpers import raw


def test_reduced_uuid_edge_layout_and_optional_adjacency(tmp_path):
    path = tmp_path / "slim-edges.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=False,
                            spatial_index=False, edge_adjacency_index=False,
                            edge_columns=[]) as db:
        a = db.upsert_object(dict(kind="note", meta=None))
        b = db.upsert_object(dict(kind="note", meta=""))
        edge = db.add_edge(dict(kind="parallel", a=a, b=b,
                               a_anchor=None, b_anchor="", meta=None))
        record = db.edge(edge)
        assert record["kind"] == "parallel" and record["meta"] == {}
        assert record["version"] == 1 and record["approval_state"] == "pending"
        with pytest.raises(RuntimeError, match="adjacency index is absent"):
            db.edges_of(a)
        with pytest.raises(RuntimeError, match="not supported.*meta"):
            db.set_meta(edge, "manual", True)
        with pytest.raises(RuntimeError, match="not supported"):
            db.set_approval(edge, "approved")
    with sqlite3.connect(path) as sql:
        assert {row[1] for row in sql.execute("PRAGMA table_info(edges)")} == {
            "uuid", "a_uuid", "b_uuid", "kind", "commit_sequence"}
        assert "WITHOUT ROWID" in sql.execute(
            "SELECT sql FROM sqlite_schema WHERE name='edges'").fetchone()[0]
        assert not sql.execute("SELECT 1 FROM key_allocators WHERE name='edges'").fetchall()
        assert not sql.execute("SELECT 1 FROM sqlite_schema WHERE name IN ('edges_a','edges_b')").fetchall()
        assert sql.execute("SELECT typeof(uuid),length(uuid),kind FROM edges").fetchone() == ("blob", 16, 0)
    with ObjectStore.open(str(path), author="test") as db:
        assert db.edge(edge)["meta"] == {}
        assert db.add_edge(dict(kind="perpendicular", a=a, b=b))


@pytest.mark.parametrize("meta", [None, "", {}])
def test_empty_json_forms_in_full_layout(tmp_path, meta):
    path = tmp_path / "empty.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=False) as db:
        a = db.upsert_object(dict(kind="note", meta=meta))
        b = db.upsert_object(dict(kind="note"))
        edge = db.add_edge(dict(kind="perpendicular", a=a, b=b, meta=meta))
        assert db.object(a)["meta"] == {} and db.edge(edge)["meta"] == {}
@pytest.mark.parametrize("history_enabled", [False, True])
def test_same_object_write_guard_does_not_hide_a_stale_read(tmp_path, history_enabled):
    from vc_object_store import TransactionConflict

    with ObjectStore.create(str(tmp_path / "conflict.vcobj"), author="test",
                            history_enabled=history_enabled) as store:
        obj = store.upsert_object(dict(kind="note"))
        state = store.status()
        request = dict(command_version=3, request=str(uuid.uuid4()), store=state["store"],
                       generation=state["generation"], author="test",
                       expected_objects={obj: dict(store=state['store'], generation=state['generation'], object=obj, version=1)},
                       read_dependencies={obj: dict(store=state['store'], generation=state['generation'], object=obj, version=2)},
                       expected_relationships={}, expected_commit_sequence=None,
                       operations=[dict(op="set_meta", uuid=obj, key="derived", value=True)])
        with pytest.raises(TransactionConflict):
            store.submit(request)
        assert store.status()["commit_sequence"] == state["commit_sequence"]
        assert "derived" not in store.object(obj)["meta"]
        request["read_dependencies"] = dict(request["expected_objects"])
        store.submit(request)
        assert store.object(obj)["meta"]["derived"] is True


def test_history_off_spatial_off_and_atomic_seal(tmp_path):
    path = tmp_path / "direct.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=False, spatial_index=False) as db:
        policy = db.provisioning()
        assert policy == dict(schema_version=22, generation=db.status()["generation"],
                              commit_sequence=db.status()["commit_sequence"], history_enabled=False,
                              spatial_index=False, sealed=False)
        identity = db.database_identity()
        assert identity["schema_version"] == 22
        tables = {table["name"] for table in identity["tables"]}
        assert {"objects", "geometry_chunks"} <= tables
        assert not {"geometry_heads", "current_geometry", "current_geometry_chunks", "current_shared_payloads"} & tables
        a = db.upsert_object(dict(kind="fiber"))
        geometry = db.put_geometry(a, Fiber([], [[1, 2, 3], [4, 5, 6]]))
        data = raw(db, geometry, "line_points")
        with pytest.raises(RuntimeError, match="history is disabled"):
            db.object_history(a)
        with pytest.raises(RuntimeError, match="spatial index is absent"):
            db.select_objects(bbox=[0, 0, 0, 10, 10, 10])
        before = db.provisioning()["commit_sequence"]
        with pytest.raises(ValueError, match="rollback"):
            with db.write() as txn:
                txn.seal()
                raise ValueError("rollback")
        assert not db.provisioning()["sealed"]
        assert db.provisioning()["commit_sequence"] == before
        db.seal()
        assert db.provisioning()["sealed"]
        assert db.provisioning()["commit_sequence"] == before + 1
        with pytest.raises(RuntimeError, match="sealed"):
            db.upsert_object(dict(kind="note"))
    with ObjectStore.open(str(path), read_only=True) as db:
        assert db.database_identity() == identity
        assert raw(db, db.current_geometry(a)["uuid"], "line_points") == data
        assert not db.provisioning()["history_enabled"]
    with sqlite3.connect(path) as sql:
        names = {row[0] for row in sql.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
    assert not names.intersection({"journal", "object_revisions", "relationship_revisions", "object_bounds"})
    with pytest.raises(RuntimeError, match="sealed"):
        enable_history(path)


def test_explicit_history_transitions_preserve_current_geometry(tmp_path):
    path = tmp_path / "policy.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=False) as db:
        a = db.upsert_object(dict(kind="fiber"))
        geometry = db.put_geometry(a, Fiber([], [[1, 2, 3], [4, 5, 6]]))
        data = raw(db, geometry, "line_points")
        sequence = db.provisioning()["commit_sequence"]
        generation = db.provisioning()["generation"]
        version = db.current_version(a)
    enable_history(path)
    with ObjectStore.open(str(path), author="test") as db:
        assert db.provisioning()["history_enabled"]
        assert db.provisioning()["commit_sequence"] == sequence + 1
        assert db.provisioning()["generation"] == generation
        assert db.current_version(a) == version
        assert db.object_history(a)
        assert raw(db, db.current_geometry(a)["uuid"], "line_points") == data
        db.set_meta(a, "review", True)
        sequence = db.provisioning()["commit_sequence"]
        version = db.current_version(a)
    remove_history_and_disable(path)
    with ObjectStore.open(str(path), author="test") as db:
        assert not db.provisioning()["history_enabled"]
        assert db.provisioning()["commit_sequence"] == sequence + 1
        assert db.provisioning()["generation"] == generation
        assert db.current_version(a) == version
        assert db.object(a)["meta"]["review"]
        assert raw(db, db.current_geometry(a)["uuid"], "line_points") == data
        with pytest.raises(RuntimeError, match="history is disabled"):
            db.geometry_history(a)


@pytest.mark.parametrize("kind,geometry,array_name", [
    ("fiber", Fiber([], [[1, 2, 3], [4, 5, 6]]), "line_points"),
    ("surface", Surface(1, 2, [1, 1], [[1, 2, 3], [4, 5, 6]]), "points"),
    ("track", Track([[1, 2, 3], [4, 5, 6]]), "points"),
    ("point_collection", PointCollection([7, 42], [[1, 2, 3], [4, 5, 6]]), "points"),
    ("umbilicus", Umbilicus([[1, 2, 3], [4, 5, 6]]), "points"),
])
def test_current_native_export_without_history(tmp_path, kind, geometry, array_name):
    with ObjectStore.create(str(tmp_path / "direct.vcobj"), author="test", history_enabled=False) as db:
        record = dict(kind=kind, name="current")
        if kind == "track":
            record["track"] = dict(external_key="1", family="vertical")
        uid = db.upsert_object(record)
        revision = db.put_geometry(uid, geometry)
        expected = raw(db, revision, array_name)
        output = tmp_path / ("native" if kind in ("surface", "track") else "native.json")
        exported = db.export_geometry(uid, output)
        assert exported["object"] == uid and exported["version"] == db.current_version(uid)
        assert output.exists()
        current_output = tmp_path / ("current-version" if kind in ("surface", "track") else "current-version.json")
        assert db.export_geometry(uid, current_output, version=db.current_version(uid)) == exported
        with pytest.raises(RuntimeError, match="active object version not found"):
            db.export_geometry(uid, tmp_path / "historical", version=1)
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as other:
        result = other.ingest_geometry_source(kind, output)
        assert not result["problems"] and result["ingested"] == 1
        imported = other.objects_by_kind(kind)[0]
        assert raw(other, other.current_geometry(imported["uuid"])["uuid"], array_name) == expected


def test_current_fiber_export_preserves_applicable_branches_without_history(tmp_path):
    with ObjectStore.create(str(tmp_path / "graph.vcobj"), author="test", history_enabled=False) as db:
        ids = [db.upsert_object(dict(kind="fiber")) for _ in range(2)]
        for uid in ids:
            db.put_geometry(uid, Fiber([[1, 2, 3], [4, 5, 6]], []))
        anchor = dict(points=[dict(array="control_point_ids", index=[0])], direction=[1, 0, 0])
        edge = db.add_edge(dict(kind="fiber_branch", a=ids[0], b=ids[1], a_anchor=anchor, b_anchor=anchor))
        db.set_approval(edge, "approved")
        output = tmp_path / "fiber.json"
        db.export_geometry(ids[0], output)
        branch = json.loads(output.read_text())["branches"][0]
        assert branch["target_uuid"] == ids[1] and branch["edge_uuid"] == edge
        assert not branch["pending"]
        db.remove_edge(edge)
        db.export_geometry(ids[0], output, overwrite=True)
        assert not json.loads(output.read_text())["branches"]


def test_current_collection_export_preserves_native_links_without_history(tmp_path):
    with ObjectStore.create(str(tmp_path / "points.vcobj"), author="test", history_enabled=False) as db:
        a = db.upsert_object(dict(kind="point_collection", meta=dict(source=dict(source_collection_id="10"))))
        b = db.upsert_object(dict(kind="point_collection", meta=dict(source=dict(source_collection_id="20"))))
        db.put_geometry(a, PointCollection([7, 42], [[1, 2, 3], [4, 5, 6]]))
        db.put_geometry(b, PointCollection([1], [[7, 8, 9]]))
        anchor = lambda ident: dict(points=[dict(array="point_ids", index=[ident])])
        db.add_edge(dict(kind="point_link", a=a, b=a, a_anchor=anchor(7), b_anchor=anchor(42)))
        whole = dict(whole_geometry=True)
        db.add_edge(dict(kind="collection_winding", a=a, b=b, a_anchor=whole, b_anchor=whole))
        output = tmp_path / "points.json"
        db.export_geometry(a, output)
        result = json.loads(output.read_text())["collections"]["10"]
        assert result["points"]["7"]["links"] == [42]
        assert result["points"]["42"]["links"] == [7]
        assert result["windings_linked"] == [20]
