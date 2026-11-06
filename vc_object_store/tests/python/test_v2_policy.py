"""The Python facade delegates v2 policy to the native store implementation."""
import sqlite3
import json

import pytest

from vc_object_store import (
    Fiber, Surface, Track, Umbilicus, PointCollection, ObjectStore,
    convert_v1_to_v2, enable_history, remove_history_and_disable,
)
from object_helpers import raw


def test_history_off_spatial_off_and_atomic_seal(tmp_path):
    path = tmp_path / "direct.vcobj"
    with ObjectStore.create(str(path), author="test", history_enabled=False, spatial_index=False) as db:
        assert db.provisioning() == dict(format_version=2, generation=1, history_enabled=False,
                                        spatial_index=False, sealed=False)
        identity = db.database_identity()
        assert identity["schema_version"] == 18
        assert any(table["name"] == "current_geometry" for table in identity["tables"])
        a = db.upsert_object(dict(kind="fiber"))
        geometry = db.put_geometry(a, Fiber([], [[1, 2, 3], [4, 5, 6]]))
        data = raw(db, geometry, "line_points")
        with pytest.raises(RuntimeError, match="history is disabled"):
            db.object_history(a)
        with pytest.raises(RuntimeError, match="spatial index is absent"):
            db.select_objects(bbox=[0, 0, 0, 10, 10, 10])
        before = db.provisioning()["generation"]
        with pytest.raises(ValueError, match="rollback"):
            with db.write() as txn:
                txn.seal()
                raise ValueError("rollback")
        assert not db.provisioning()["sealed"]
        assert db.provisioning()["generation"] == before
        db.seal()
        assert db.provisioning()["sealed"]
        assert db.provisioning()["generation"] == before + 1
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
        generation = db.provisioning()["generation"]
    enable_history(path)
    with ObjectStore.open(str(path), author="test") as db:
        assert db.provisioning()["history_enabled"]
        assert db.provisioning()["generation"] == generation + 1
        assert db.object_history(a)
        assert raw(db, db.current_geometry(a)["uuid"], "line_points") == data
        db.set_meta(a, "review", True)
        generation = db.provisioning()["generation"]
    remove_history_and_disable(path)
    with ObjectStore.open(str(path), author="test") as db:
        assert not db.provisioning()["history_enabled"]
        assert db.provisioning()["generation"] == generation + 1
        assert db.object(a)["meta"]["review"]
        assert raw(db, db.current_geometry(a)["uuid"], "line_points") == data
        with pytest.raises(RuntimeError, match="history is disabled"):
            db.geometry_history(a)
    # Conversion is separate and does not silently accept or rewrite v2.
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="unexpected vcobj schema"):
        convert_v1_to_v2(path)
    assert path.read_bytes() == before


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
        assert db.export_geometry(uid, output) == revision
        assert output.exists()
        with pytest.raises(RuntimeError, match="history is disabled"):
            db.export_geometry(uid, tmp_path / "historical", object_revision_uuid=revision)
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
