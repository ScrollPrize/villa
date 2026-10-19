"""Cross-store publication, graph pinning, and reference integrity."""
import os
import sqlite3
import struct
import uuid
from pathlib import Path

import pytest
import vc
if os.environ.get("VC_BUILD_PYTHON"):
    vc.__path__.insert(0, str(Path(os.environ["VC_BUILD_PYTHON"]) / "vc"))
ObjectStore = pytest.importorskip("vc.objects").ObjectStore
GraphDb = pytest.importorskip("vc.graph").GraphDb


def geometry(value=1):
    return {"kind": "fiber", "arrays": [
        {"name": "control_points", "dtype": "<f8", "shape": [0, 3], "data": b""},
        {"name": "line_points", "dtype": "<f8", "shape": [1, 3],
         "data": struct.pack("<ddd", value, 2, 3)}]}


@pytest.fixture()
def pair(tmp_path):
    with ObjectStore.create(str(tmp_path / "objects.vcobj"), author="author") as store:
        with GraphDb.create(str(tmp_path / "graph.vcgr"), author="reviewer") as graph:
            yield store, graph


def seed(store, graph):
    with store.write() as tx:
        a = tx.upsert_object({"kind": "fiber", "tags": ["source"]})
        ra = tx.put_geometry(a, geometry())
        b = tx.upsert_object({"kind": "fiber"})
        rb = tx.put_geometry(b, geometry(2))
    with graph.write() as tx:
        tx.include_object(store, a, revision=ra)
        tx.include_object(store, b, revision=rb)
        edge = tx.add_edge({"kind": "connected", "a": b, "b": a})
        tx.set_approval(edge, "approved")
    return a, b, ra, rb, edge


def test_storage_is_independent_and_schema_has_no_graph(pair):
    store, graph = pair
    a, b, ra, rb, edge = seed(store, graph)
    for database, absent in ((store, {"edges", "edge_tags", "edge_geometry_bindings"}),
                             (graph, {"geometry_chunks", "geometry_revisions", "geometry_revision_chunks", "store_identity"})):
        with sqlite3.connect(database.path) as sql:
            tables = {r[0] for r in sql.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            assert not tables & absent
            assert not list(sql.execute("PRAGMA foreign_key_check"))
    assert not hasattr(store, "add_edge")
    assert not hasattr(graph, "put_geometry")
    assert graph.selected_object(a) == {"object_uuid": a, "store_uuid": store.uuid, "revision_uuid": ra}
    record, state = graph.edge(edge), graph.edge_geometry_state(edge)
    revisions = {a: ra, b: rb}
    assert state == {"a_revision": revisions[record["a"]], "b_revision": revisions[record["b"]], "stale": False}


def test_publication_and_adoption_are_separate_and_graphs_pin_revisions(pair, tmp_path):
    store, graph = pair
    a, b, first, rb, edge = seed(store, graph)
    with GraphDb.create(str(tmp_path / "second.vcgr"), author="other") as other:
        other.include_object(store, a, revision=first)
        graph.set_meta(a, "local_note", "keep")
        before = graph.journal_since()
        second = store.put_geometry(a, geometry(9), expected_revision=first)
        assert graph.journal_since() == before
        assert graph.edge(edge)["approval_state"] == "approved"
        assert not graph.edge_geometry_state(edge)["stale"]
        assert graph.available_updates(store) == [{"object_uuid": a, "selected_revision": first, "current_revision": second}]
        with graph.write() as tx:
            tx.advance_object(store, a, revision=second, expected_revision=first)
        assert graph.selected_object(a)["revision_uuid"] == second
        assert other.selected_object(a)["revision_uuid"] == first
        assert graph.object(a)["meta"] == {"local_note": "keep"}
        assert graph.edge(edge)["approval_state"] == "pending"
        assert graph.edge_geometry_state(edge)["stale"]
        with pytest.raises(RuntimeError, match="revalidate"):
            graph.set_approval(edge, "approved")
        graph.revalidate_edge_geometry(edge)
        graph.set_approval(edge, "approved")
        assert not graph.available_updates(store)
        before = graph.journal_since()
        graph.advance_object(store, a, revision=second, expected_revision=second)
        assert graph.journal_since() == before
        with pytest.raises(RuntimeError, match="selected revision changed"):
            graph.advance_object(store, a, revision=second, expected_revision=first)


def test_reference_validation_and_failed_adoption_preserve_graph(pair, tmp_path):
    store, graph = pair
    a, b, first, rb, edge = seed(store, graph)
    before = graph.journal_since()
    for object_id, revision in ((a, rb), (a, str(uuid.uuid4())), (str(uuid.uuid4()), first)):
        with pytest.raises(ValueError, match="revision does not belong"):
            graph.include_object(store, object_id, revision=revision)
    with ObjectStore.create(str(tmp_path / "wrong.vcobj"), author="other") as wrong:
        wrong.upsert_object({"kind": "fiber", "uuid": a})
        wrong_revision = wrong.put_geometry(a, geometry())
        with pytest.raises(ValueError, match="another store"):
            graph.advance_object(wrong, a, revision=wrong_revision, expected_revision=first)
    assert graph.journal_since() == before
    second = store.put_geometry(a, geometry(5), expected_revision=first)
    with sqlite3.connect(graph.path) as sql:
        sql.execute("CREATE TRIGGER reject_edge BEFORE UPDATE ON edges BEGIN SELECT RAISE(ABORT,'injected'); END")
    with graph.write() as tx:
        with pytest.raises(RuntimeError, match="injected"):
            tx.advance_object(store, a, revision=second, expected_revision=first)
        tx.upsert_object({"kind": "note", "name": "survives"})
    assert graph.selected_object(a)["revision_uuid"] == first
    assert graph.edge(edge)["approval_state"] == "approved"
    assert store.current_geometry(a)["uuid"] == second


def test_graph_queries_work_without_store_and_store_can_move(pair, tmp_path):
    store, graph = pair
    a, b, first, rb, edge = seed(store, graph)
    store_id = store.uuid
    original = Path(store.path)
    store.close()
    moved = tmp_path / "moved.vcobj"
    original.rename(moved)
    with GraphDb.open(graph.path, read_only=True) as reader:
        assert reader.neighbours(a) == [b]
        assert reader.selected_object(a)["revision_uuid"] == first
    with ObjectStore.open(str(moved), read_only=True) as reader:
        assert reader.uuid == store_id
        assert not graph.available_updates(reader)
        assert reader.read_geometry_array(first, "line_points")


def test_uncommitted_revisions_cannot_be_referenced(pair):
    store, graph = pair
    with store.write() as tx:
        a = tx.upsert_object({"kind": "fiber"})
        revision = tx.put_geometry(a, geometry())
        with pytest.raises(ValueError, match="revision does not belong"):
            graph.include_object(store, a, revision=revision)
    graph.include_object(store, a, revision=revision)
    with pytest.raises(RuntimeError, match="retained|FOREIGN KEY"):
        store.delete_object(a)


def test_transactions_keep_handle_alive_and_reject_close(pair):
    store, graph = pair
    for db in (store, graph):
        with db.write() as tx:
            with pytest.raises(RuntimeError, match="active transaction"):
                db.close()
            tx.upsert_object({"kind": "note"})
        assert db.object_count("note") == 1


@pytest.mark.parametrize("readonly", [False, True])
def test_database_types_and_unsupported_schemas_are_rejected(pair, readonly):
    store, graph = pair
    options = {"read_only": readonly, "author": "tester"}
    with pytest.raises(RuntimeError):
        GraphDb.open(store.path, **options)
    with pytest.raises(RuntimeError):
        ObjectStore.open(graph.path, **options)
    with sqlite3.connect(graph.path) as sql:
        sql.execute("PRAGMA user_version=999")
    with pytest.raises(RuntimeError, match="unsupported schema"):
        GraphDb.open(graph.path, **options)


def test_object_module_does_not_import_graph_or_volume(tmp_path):
    import subprocess
    import sys
    import vc.objects
    package_dir = str(Path(vc.objects.__file__).parent)
    script = f"""
import sys
sys.meta_path[:] = [f for f in sys.meta_path if not type(f).__module__.startswith('_editable_skbc')]
sys.path.insert(0, {str(Path(package_dir).parent)!r})
from vc.objects import ObjectStore
assert 'vc.graph' not in sys.modules
assert 'vc.volume' not in sys.modules
with ObjectStore.create({str(tmp_path / 'independent.vcobj')!r}, author='test') as store:
    store.upsert_object({{'kind': 'note'}})
    assert store.object_count('note') == 1
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_membership_rollback_removal_and_updates_from_multiple_stores(pair, tmp_path):
    store, graph = pair
    a, b, first, rb, edge = seed(store, graph)
    with ObjectStore.create(str(tmp_path / "another.vcobj"), author="other") as other:
        c = other.upsert_object({"kind": "fiber"})
        rc = other.put_geometry(c, geometry())
        with pytest.raises(RuntimeError, match="abort"):
            with graph.write() as tx:
                tx.include_object(other, c, revision=rc)
                raise RuntimeError("abort")
        assert graph.object(c) is None and graph.selected_object(c) is None
        graph.include_object(other, c, revision=rc)
        rc2 = other.put_geometry(c, geometry(2), expected_revision=rc)
        assert not graph.available_updates(store)
        assert graph.available_updates(other) == [
            {"object_uuid": c, "selected_revision": rc, "current_revision": rc2}]
        graph.delete_object(c)
        assert graph.selected_object(c) is None
        assert other.read_geometry_array(rc, "line_points")
