import os
"""Objects, geometry, and relationship metadata share one atomic database."""
import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from vc_object_store import ObjectStore
from object_helpers import _geometry, edge_choice


def seed(db):
    with db.write() as tx:
        a = tx.upsert_object({"kind": "fiber", "tags": ["source"]})
        ra = tx.put_geometry(a, _geometry())
        b = tx.upsert_object({"kind": "fiber"})
        rb = tx.put_geometry(b, _geometry(value=2))
        edge = tx.add_edge({"kind": "connected", "a": b, "b": a,
                            "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
        tx.set_approval(edge, "approved")
    return a, b, ra, rb, edge


def test_publication_requires_explicit_decisions_and_preserves_unrelated_edges(db):
    a, b, first, rb, edge = seed(db)
    c = db.upsert_object({"kind": "note"})
    unrelated = db.add_edge({"kind": "connected", "a": b, "b": c, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    db.set_approval(unrelated, "approved")
    unrelated_before = db.edge(unrelated)
    db.set_meta(a, "review_note", "keep")
    before_edge = db.edge(edge)
    before_state = db.edge_geometry_state(edge)
    before = db.journal_since()
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        db.put_geometry(a, _geometry(value=9), expected_revision=first)
    assert db.journal_since() == before
    second = db.put_geometry(a, _geometry(value=9), expected_revision=first, edge_edits=[edge_choice(db, edge, "retain")])
    assert db.object(a)["meta"] == {"review_note": "keep"}
    assert db.object(a)["tags"] == ["source"]
    assert db.edge(edge)["approval_state"] == "pending"
    assert db.edge(edge)["generation"] == before_edge["generation"] + 1
    assert db.edge(unrelated) == unrelated_before
    assert not db.edge_geometry_state(edge)["stale"]
    changes = db.journal_since(before[-1]["seq"])
    assert [(c["record_uuid"], c["change_kind"]) for c in changes] == [(a, "geometry"), (edge, "geometry_repaired")]
    assert db.read_geometry_array(first, "line_points") == _geometry()["arrays"][1]["data"]
    assert db.read_geometry_array(second, "line_points") == _geometry(value=9)["arrays"][1]["data"]
    db.set_approval(edge, "approved")
    before = db.journal_since()
    assert db.put_geometry(a, _geometry(value=9), expected_revision=second) == second
    assert db.journal_since() == before
    with pytest.raises(RuntimeError, match="current revision changed"):
        db.put_geometry(a, _geometry(value=9), expected_revision=first)
    assert db.journal_since() == before


def test_initial_geometry_requires_explicit_relationship_decisions(db):
    a = db.upsert_object({"kind": "fiber"})
    b = db.upsert_object({"kind": "note"})
    edge = db.add_edge({"kind": "connected", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    db.set_approval(edge, "approved")
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        db.put_geometry(a, _geometry())
    db.put_geometry(a, _geometry(), edge_edits=[edge_choice(db, edge, "retain")])
    assert db.edge(edge)["approval_state"] == "pending"
    assert not db.edge_geometry_state(edge)["stale"]


def test_relationship_failure_rolls_back_geometry_and_metadata_within_transaction(db):
    a, b, first, rb, edge = seed(db)
    c = db.upsert_object({"kind": "note"})
    second_edge = db.add_edge({"kind": "connected", "a": a, "b": c, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    db.set_approval(second_edge, "approved")
    before_second_edge = db.edge(second_edge)
    before_object, before_edge = db.object(a), db.edge(edge)
    before_journal = db.journal_since()
    with sqlite3.connect(db.path) as sql:
        counts = [sql.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                  for table in ("geometry_revisions", "geometry_chunks", "geometry_revision_chunks")]
        # Reject the second invalidation, after the first edge has been changed.
        sql.execute(f"CREATE TRIGGER reject_edge BEFORE UPDATE ON edges "
                    f"WHEN NEW.uuid=X'{second_edge.replace('-', '')}' "
                    "BEGIN SELECT RAISE(ABORT,'injected'); END")
    choices = [edge_choice(db, uid, "retain") for uid in (edge, second_edge)]
    with db.write() as tx:
        with pytest.raises(RuntimeError, match="injected"):
            tx.put_geometry(a, _geometry(value=5), expected_revision=first, edge_edits=choices)
        tx.upsert_object({"kind": "note", "name": "survives"})
    assert db.current_geometry(a)["uuid"] == first
    assert db.object(a) == before_object
    assert db.edge(edge) == before_edge
    assert db.edge(second_edge) == before_second_edge
    assert not db.edge_geometry_state(edge)["stale"]
    assert db.journal_since()[:-1] == before_journal
    assert db.object_count("note") == 2
    with sqlite3.connect(db.path) as sql:
        assert counts == [sql.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                          for table in ("geometry_revisions", "geometry_chunks", "geometry_revision_chunks")]
        assert not list(sql.execute("PRAGMA foreign_key_check"))


def test_object_geometry_and_edge_creation_roll_back_together(db):
    with pytest.raises(RuntimeError, match="abort"):
        with db.write() as tx:
            a = tx.upsert_object({"kind": "fiber"})
            revision = tx.put_geometry(a, _geometry())
            b = tx.upsert_object({"kind": "note"})
            edge = tx.add_edge({"kind": "connected", "a": a, "b": b, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
            raise RuntimeError("abort")
    assert db.object(a) is None and db.object(b) is None
    assert db.geometry_revision(revision) is None
    assert db.edge(edge) is None and not db.journal_since()
    with sqlite3.connect(db.path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 0


def test_checkpoint_and_move_keep_geometry_relationships_and_identity(db, tmp_path):
    a, b, first, rb, edge = seed(db)
    store_id = db.uuid
    original = Path(db.path)
    db.checkpoint()
    db.close()
    moved = tmp_path / "moved.vcobj"
    original.rename(moved)
    with ObjectStore.open(str(moved), read_only=True) as reader:
        assert reader.uuid == store_id
        assert reader.neighbours(a) == [b]
        assert reader.current_geometry(a)["uuid"] == first
        assert reader.read_geometry_array(first, "line_points")
        assert reader.edge(edge)["approval_state"] == "approved"


def test_transactions_keep_handle_alive_and_reject_close(db):
    with db.write() as tx:
        with pytest.raises(RuntimeError, match="active transaction"):
            db.close()
        tx.upsert_object({"kind": "note"})
    assert db.object_count("note") == 1
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
    import vc_object_store
    package_dir = str(Path(vc_object_store.__file__).parent)
    script = f"""
import sys
sys.meta_path[:] = [f for f in sys.meta_path if not type(f).__module__.startswith('_editable_skbc')]
sys.path.insert(0, {str(Path(package_dir).parent)!r})
from vc_object_store import ObjectStore
assert 'vc.volume' not in sys.modules
with ObjectStore.create({str(tmp_path / 'independent.vcobj')!r}, author='test') as store:
    a = store.upsert_object({{'kind': 'note'}})
    b = store.upsert_object({{'kind': 'note'}})
    store.add_edge({{'kind': 'connected', 'a': a, 'b': b}})
    assert store.neighbours(a) == [b]
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_real_segment_geometry_and_relationships_roundtrip(db, tmp_path):
    import json
    import shutil
    import uuid
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    fixtures = Path(os.environ.get("VC_OBJECT_STORE_FIXTURES", Path(__file__).resolve().parents[3] / "volume-cartographer/core/test/data")) / "segments"
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
        assert db.ingest_geometry_source("patch", copied)["ingested"] == 1
        revision = db.current_geometry(uid)
        descriptor = revision["arrays"][0]
        actual = np.frombuffer(db.read_geometry_array(revision["uuid"], "points"), dtype="<f4").reshape(descriptor["shape"])
        expected = np.stack([tifffile.imread(copied / f"{axis}.tif") for axis in "xyz"], axis=-1)
        np.testing.assert_array_equal(actual, expected)
        ids.append(uid)
    edge = db.add_edge({"kind": "fixture-relationship", "a": ids[0], "b": ids[1], "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    db.set_approval(edge, "approved")
    db.checkpoint()
    with ObjectStore.open(db.path, read_only=True) as reader:
        assert reader.neighbours(ids[0]) == [ids[1]]
        assert reader.edge(edge)["approval_state"] == "approved"
        revision = reader.current_geometry(ids[0])
        points = reader.read_geometry_array(revision["uuid"], "points")
        reader.export_geometry(ids[0], tmp_path / "export")
    with ObjectStore.create(str(tmp_path / "roundtrip.vcobj"), author="test") as restored:
        assert restored.ingest_geometry_source("patch", tmp_path / "export")["ingested"] == 1
        revision = restored.current_geometry(ids[0])
        assert restored.read_geometry_array(revision["uuid"], "points") == points


def test_geometry_revisions_approval_history_and_source_independence(db, db_path):
    a = db.upsert_object({"kind": "fiber", "tags": ["keep"]})
    first = db.put_geometry(a, _geometry())
    db.set_approval(a, "approved")
    before = db.journal_since()
    assert db.put_geometry(a, _geometry(), expected_revision=first) == first
    assert db.journal_since() == before
    second = db.put_geometry(a, _geometry(value=9), expected_revision=first)
    assert db.geometry_revision(second)["parent_uuid"] == first
    assert db.object(a)["approval_state"] == "pending"
    assert db.read_geometry_array(first, "line_points") == _geometry()["arrays"][1]["data"]
    assert len(db.geometry_history(a)) == 2
    db.set_meta(a, "annotation", "changed")
    assert db.current_geometry(a)["uuid"] == second
    assert db.object(a)["tags"] == ["keep"]
    with ObjectStore.open(db_path, read_only=True) as read:
        assert read.uuid == db.uuid
        assert read.read_geometry_array(first, "line_points") == db.read_geometry_array(first, "line_points")
        with pytest.raises(RuntimeError, match="read-only"):
            read.put_geometry(a, _geometry(), expected_revision=second)


def test_geometry_conflicts_validation_and_atomic_rollback(db, db_path):
    a = db.upsert_object({"kind": "fiber"})
    first = db.put_geometry(a, _geometry())
    with pytest.raises(RuntimeError, match="current revision changed"):
        db.put_geometry(a, _geometry(value=2))
    with pytest.raises(RuntimeError, match="current revision changed"):
        db.put_geometry(a, _geometry(value=2), expected_revision=str(uuid.uuid4()))
    before = db.journal_since()
    with pytest.raises(ValueError, match="shape"):
        bad = _geometry(); bad["arrays"][1]["shape"] = [10, 3]
        db.put_geometry(a, bad, expected_revision=first)
    with pytest.raises(RuntimeError, match="abort transaction"):
        with db.write() as w:
            w.put_geometry(a, _geometry(value=5), expected_revision=first)
            raise RuntimeError("abort transaction")
    assert db.current_geometry(a)["uuid"] == first
    assert db.journal_since() == before
    # Force a failure after inserting the revision/chunks; catch it and commit
    # other work to prove the internal savepoint protects callers too.
    with sqlite3.connect(db_path) as sql:
        sql.execute("CREATE TRIGGER reject_geometry_head BEFORE UPDATE ON geometry_heads BEGIN SELECT RAISE(ABORT,'injected'); END")
    with db.write() as w:
        with pytest.raises(RuntimeError, match="injected"):
            w.put_geometry(a, _geometry(value=5), expected_revision=first)
        w.upsert_object({"kind": "note", "name": "survives"})
    assert len(db.geometry_history(a)) == 1
    with sqlite3.connect(db_path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 1
        sql.execute("DROP TRIGGER reject_geometry_head")


def test_geometry_metadata_updates_cannot_detach_payloads(db, db_path):
    uid = db.upsert_object({"kind": "fiber", "storage_kind": "file", "storage_ref": "missing.json"})
    revision = db.put_geometry(uid, _geometry())
    stored = db.object(uid)
    db.upsert_object({"uuid": uid, "kind": "fiber", "name": "renamed", "data_checksum": "wrong", "data_generation": 99})
    updated = db.object(uid)
    assert updated["storage_kind"] == "sqlite"
    assert updated["storage_ref"] == stored["storage_ref"]
    assert updated["data_generation"] == stored["data_generation"]
    assert updated["data_checksum"] == stored["data_checksum"]
    assert db.current_geometry(uid)["uuid"] == revision
    with pytest.raises(RuntimeError, match="FOREIGN KEY"):
        db.delete_object(uid)
    with sqlite3.connect(db_path) as sql:
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    with pytest.raises(RuntimeError, match="corrupt payload"):
        db.read_geometry_array(revision, "line_points")


@pytest.mark.parametrize("mutation", ["dtype", "rank", "duplicate", "overflow", "kind", "attributes"])
def test_geometry_rejects_invalid_descriptors(db, mutation):
    uid = db.upsert_object({"kind": "fiber"})
    data = _geometry()
    if mutation == "dtype": data["arrays"][1]["dtype"] = ">f8"
    if mutation == "rank": data["arrays"][1]["shape"] = [3]
    if mutation == "duplicate": data["arrays"].append(data["arrays"][0])
    if mutation == "overflow": data["arrays"][1]["shape"] = [2**62, 3]
    if mutation == "kind": data["kind"] = "other"
    if mutation == "attributes": data["attributes"] = []
    with pytest.raises(ValueError): db.put_geometry(uid, data)
    assert db.current_geometry(uid) is None


def test_geometry_storage_kind_cannot_claim_missing_payload(db):
    with pytest.raises(ValueError, match="putGeometry"):
        db.upsert_object({"kind": "fiber", "storage_kind": "sqlite"})
    assert db.objects_by_kind("fiber") == []
