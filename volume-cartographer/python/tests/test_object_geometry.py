"""Objects, geometry, and relationship metadata share one atomic database."""
import json
import sqlite3
import struct
from pathlib import Path

import pytest
import vc
from vc.objects import ObjectStore
from object_helpers import edge_choice


def geometry(value=1):
    return {"kind": "fiber", "arrays": [
        {"name": "control_points", "dtype": "<f8", "shape": [0, 3], "data": b""},
        {"name": "line_points", "dtype": "<f8", "shape": [1, 3],
         "data": struct.pack("<ddd", value, 2, 3)}]}


@pytest.fixture()
def store(tmp_path):
    with ObjectStore.create(str(tmp_path / "objects.vcobj"), author="author") as db:
        yield db


def seed(store):
    with store.write() as tx:
        a = tx.upsert_object({"kind": "fiber", "tags": ["source"]})
        ra = tx.put_geometry(a, geometry())
        b = tx.upsert_object({"kind": "fiber"})
        rb = tx.put_geometry(b, geometry(2))
        edge = tx.add_edge({"kind": "connected", "a": b, "b": a,
                            "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
        tx.set_approval(edge, "approved")
    return a, b, ra, rb, edge


def test_one_database_owns_objects_geometry_and_relationships(store):
    a, b, ra, rb, edge = seed(store)
    with sqlite3.connect(store.path) as sql:
        tables = {r[0] for r in sql.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
        assert tables >= {"objects", "edges", "edge_tags", "edge_geometry_bindings",
                          "geometry_chunks", "geometry_revisions", "geometry_revision_chunks", "store_identity"}
        assert {r[1] for r in sql.execute("PRAGMA table_info(geometry_heads)")} == {"object_uuid", "revision_uuid"}
        assert not list(sql.execute("PRAGMA foreign_key_check"))
        assert sql.execute("PRAGMA user_version").fetchone()[0] == 7
    assert store.neighbours(a) == [b]
    record, state = store.edge(edge), store.edge_geometry_state(edge)
    revisions = {a: ra, b: rb}
    assert state == {"a_revision": revisions[record["a"]], "b_revision": revisions[record["b"]], "stale": False}
    exported = json.loads(store.export_json())
    assert exported["metadata_only"]
    assert len(exported["objects"]) == 2 and len(exported["edges"]) == 1


def test_publication_requires_explicit_decisions_and_preserves_unrelated_edges(store):
    a, b, first, rb, edge = seed(store)
    c = store.upsert_object({"kind": "note"})
    unrelated = store.add_edge({"kind": "connected", "a": b, "b": c, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    store.set_approval(unrelated, "approved")
    unrelated_before = store.edge(unrelated)
    store.set_meta(a, "review_note", "keep")
    before_edge = store.edge(edge)
    before_state = store.edge_geometry_state(edge)
    before = store.journal_since()
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        store.put_geometry(a, geometry(9), expected_revision=first)
    assert store.journal_since() == before
    second = store.put_geometry(a, geometry(9), expected_revision=first, edge_edits=[edge_choice(store, edge, "retain")])
    assert store.object(a)["meta"] == {"review_note": "keep"}
    assert store.object(a)["tags"] == ["source"]
    assert store.edge(edge)["approval_state"] == "pending"
    assert store.edge(edge)["generation"] == before_edge["generation"] + 1
    assert store.edge(unrelated) == unrelated_before
    assert not store.edge_geometry_state(edge)["stale"]
    changes = store.journal_since(before[-1]["seq"])
    assert [(c["record_uuid"], c["change_kind"]) for c in changes] == [(a, "geometry"), (edge, "geometry_repaired")]
    assert store.read_geometry_array(first, "line_points") == geometry()["arrays"][1]["data"]
    assert store.read_geometry_array(second, "line_points") == geometry(9)["arrays"][1]["data"]
    store.set_approval(edge, "approved")
    before = store.journal_since()
    assert store.put_geometry(a, geometry(9), expected_revision=second) == second
    assert store.journal_since() == before
    with pytest.raises(RuntimeError, match="current revision changed"):
        store.put_geometry(a, geometry(9), expected_revision=first)
    assert store.journal_since() == before


def test_initial_geometry_requires_explicit_relationship_decisions(store):
    a = store.upsert_object({"kind": "fiber"})
    b = store.upsert_object({"kind": "note"})
    edge = store.add_edge({"kind": "connected", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    store.set_approval(edge, "approved")
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        store.put_geometry(a, geometry())
    store.put_geometry(a, geometry(), edge_edits=[edge_choice(store, edge, "retain")])
    assert store.edge(edge)["approval_state"] == "pending"
    assert not store.edge_geometry_state(edge)["stale"]


def test_relationship_failure_rolls_back_geometry_and_metadata_within_transaction(store):
    a, b, first, rb, edge = seed(store)
    c = store.upsert_object({"kind": "note"})
    second_edge = store.add_edge({"kind": "connected", "a": a, "b": c, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    store.set_approval(second_edge, "approved")
    before_second_edge = store.edge(second_edge)
    before_object, before_edge = store.object(a), store.edge(edge)
    before_journal = store.journal_since()
    with sqlite3.connect(store.path) as sql:
        counts = [sql.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("geometry_revisions", "geometry_chunks", "geometry_revision_chunks")]
        # Reject the second invalidation, after the first edge has been changed.
        sql.execute(f"CREATE TRIGGER reject_edge BEFORE UPDATE ON edges "
                    f"WHEN NEW.uuid=X'{second_edge.replace('-', '')}' "
                    "BEGIN SELECT RAISE(ABORT,'injected'); END")
    choices = [edge_choice(store, uid, "retain") for uid in (edge, second_edge)]
    with store.write() as tx:
        with pytest.raises(RuntimeError, match="injected"):
            tx.put_geometry(a, geometry(5), expected_revision=first, edge_edits=choices)
        tx.upsert_object({"kind": "note", "name": "survives"})
    assert store.current_geometry(a)["uuid"] == first
    assert store.object(a) == before_object
    assert store.edge(edge) == before_edge
    assert store.edge(second_edge) == before_second_edge
    assert not store.edge_geometry_state(edge)["stale"]
    assert store.journal_since()[:-1] == before_journal
    assert store.object_count("note") == 2
    with sqlite3.connect(store.path) as sql:
        assert counts == [sql.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                          for table in ("geometry_revisions", "geometry_chunks", "geometry_revision_chunks")]
        assert not list(sql.execute("PRAGMA foreign_key_check"))


def test_object_geometry_and_edge_creation_roll_back_together(store):
    with pytest.raises(RuntimeError, match="abort"):
        with store.write() as tx:
            a = tx.upsert_object({"kind": "fiber"})
            revision = tx.put_geometry(a, geometry())
            b = tx.upsert_object({"kind": "note"})
            edge = tx.add_edge({"kind": "connected", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
            raise RuntimeError("abort")
    assert store.object(a) is None and store.object(b) is None
    assert store.geometry_revision(revision) is None
    assert store.edge(edge) is None and not store.journal_since()
    with sqlite3.connect(store.path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 0


def test_checkpoint_and_move_keep_geometry_relationships_and_identity(store, tmp_path):
    a, b, first, rb, edge = seed(store)
    store_id = store.uuid
    original = Path(store.path)
    store.checkpoint()
    store.close()
    moved = tmp_path / "moved.vcobj"
    original.rename(moved)
    with ObjectStore.open(str(moved), read_only=True) as reader:
        assert reader.uuid == store_id
        assert reader.neighbours(a) == [b]
        assert reader.current_geometry(a)["uuid"] == first
        assert reader.read_geometry_array(first, "line_points")
        assert reader.edge(edge)["approval_state"] == "approved"


def test_transactions_keep_handle_alive_and_reject_close(store):
    with store.write() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            store.close()
        tx.upsert_object({"kind": "note"})
    assert store.object_count("note") == 1
    with pytest.raises(RuntimeError, match="finished"):
        tx.add_edge({"kind": "connected", "a": "unused", "b": "unused"})


@pytest.mark.parametrize("readonly", [False, True])
@pytest.mark.parametrize("application_id,version", [(0x12345678, 1), (0x56434F42, 999)])
def test_unrecognized_database_format_is_rejected_without_writes(tmp_path, readonly, application_id, version):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as sql:
        sql.execute(f"PRAGMA application_id={application_id}")
        sql.execute(f"PRAGMA user_version={version}")
        sql.execute("CREATE TABLE sentinel(value TEXT)")
        sql.execute("INSERT INTO sentinel VALUES('preserve')")
    with pytest.raises(RuntimeError, match="unsupported"):
        ObjectStore.open(str(path), read_only=readonly, author="tester")
    with sqlite3.connect(path) as sql:
        assert sql.execute("PRAGMA application_id").fetchone()[0] == application_id
        assert sql.execute("PRAGMA user_version").fetchone()[0] == version
        assert sql.execute("SELECT value FROM sentinel").fetchall() == [("preserve",)]


def test_object_module_does_not_import_volume(tmp_path):
    import subprocess
    import sys
    import vc.objects
    package_dir = str(Path(vc.objects.__file__).parent)
    script = f"""
import sys
sys.meta_path[:] = [f for f in sys.meta_path if not type(f).__module__.startswith('_editable_skbc')]
sys.path.insert(0, {str(Path(package_dir).parent)!r})
from vc.objects import ObjectStore
assert 'vc.volume' not in sys.modules
with ObjectStore.create({str(tmp_path / 'independent.vcobj')!r}, author='test') as store:
    a = store.upsert_object({{'kind': 'note'}})
    b = store.upsert_object({{'kind': 'note'}})
    store.add_edge({{'kind': 'connected', 'a': a, 'b': b}})
    assert store.neighbours(a) == [b]
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_real_segment_geometry_and_relationships_roundtrip(store, tmp_path):
    import json
    import shutil
    import uuid
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    fixtures = Path(__file__).resolve().parents[2] / "core/test/data/segments"
    ids = []
    for name in ("20241113070770", "20241113080880"):
        copied = tmp_path / name
        shutil.copytree(fixtures / name, copied)
        metadata = json.loads((copied / "meta.json").read_text())
        # These real segment fixtures predate canonical UUIDs. Only the copied
        # metadata identity changes; the measured coordinates remain untouched.
        uid = str(uuid.uuid5(uuid.NAMESPACE_URL, "vc-test-segment:" + name))
        metadata["uuid"] = uid
        (copied / "meta.json").write_text(json.dumps(metadata))
        assert store.ingest_geometry_source("patch", copied)["ingested"] == 1
        revision = store.current_geometry(uid)
        descriptor = revision["arrays"][0]
        actual = np.frombuffer(store.read_geometry_array(revision["uuid"], "points"), dtype="<f4").reshape(descriptor["shape"])
        expected = np.stack([tifffile.imread(copied / f"{axis}.tif") for axis in "xyz"], axis=-1)
        np.testing.assert_array_equal(actual, expected)
        ids.append(uid)
    edge = store.add_edge({"kind": "fixture-relationship", "a": ids[0], "b": ids[1], "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    store.set_approval(edge, "approved")
    store.checkpoint()
    with ObjectStore.open(store.path, read_only=True) as reader:
        assert reader.neighbours(ids[0]) == [ids[1]]
        assert reader.edge(edge)["approval_state"] == "approved"
        revision = reader.current_geometry(ids[0])
        points = reader.read_geometry_array(revision["uuid"], "points")
        reader.export_geometry(ids[0], tmp_path / "export")
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as restored:
        assert restored.ingest_geometry_source("patch", tmp_path / "export")["ingested"] == 1
        revision = restored.current_geometry(ids[0])
        assert restored.read_geometry_array(revision["uuid"], "points") == points
