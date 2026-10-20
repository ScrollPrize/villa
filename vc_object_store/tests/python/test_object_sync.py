"""Two-file merge, retained geometry, and transactional failure recovery."""
import json
import shutil
import sqlite3

import pytest
from vc_object_store import ObjectStore
from object_helpers import _geometry, edge_choice, track


@pytest.fixture()
def pair(tmp_path):
    with ObjectStore.create(str(tmp_path / "a.vcobj"), author="alice") as a, \
         ObjectStore.create(str(tmp_path / "b.vcobj"), author="bob") as b:
        yield a, b


def state(db):
    snapshot = json.loads(db.export_json())
    return {k: snapshot[k] for k in ("objects", "edges")}


def endpoint_bindings(db, edge_uuid):
    edge, binding = db.edge(edge_uuid), db.edge_geometry_state(edge_uuid)
    return ({edge["a"]: binding["a_revision"], edge["b"]: binding["b_revision"]}, binding["stale"])


def test_three_way_add_update_delete_and_noop(pair):
    a, b = pair
    first = a.upsert_object({"kind": "note", "name": "first"})
    second = b.upsert_object({"kind": "note", "name": "second"})
    before = state(a), state(b), a.journal_since(), b.journal_since()
    preview = a.synchronize(b, dry_run=True)
    assert not preview["applied"] and not preview["conflicts"]
    assert preview["local_changes"] == preview["remote_changes"] == 1
    assert before == (state(a), state(b), a.journal_since(), b.journal_since())
    assert a.synchronize(b)["applied"]
    assert state(a) == state(b)
    a.set_meta(first, "review", "alice")
    b.set_meta(second, "review", "bob")
    assert b.synchronize(a)["applied"]
    assert a.object(first)["meta"] == {"review": "alice"}
    assert a.object(second)["meta"] == {"review": "bob"}
    b.delete_object(second)
    assert a.synchronize(b)["applied"]
    assert a.object(second) is None
    journals = a.journal_since(), b.journal_since()
    report = a.synchronize(b)
    assert report["applied"] and report["local_changes"] == report["remote_changes"] == 0
    assert journals == (a.journal_since(), b.journal_since())


def test_conflict_resolution_requires_current_preview(pair):
    a, b = pair
    uid = a.upsert_object({"kind": "note", "name": "original"})
    a.synchronize(b)
    a.set_meta(uid, "note", "local")
    b.set_meta(uid, "note", "remote")
    before = state(a), state(b), a.journal_since(), b.journal_since()
    report = a.synchronize(b)
    assert not report["applied"] and len(report["conflicts"]) == 1
    c = report["conflicts"][0]
    assert c["uuid"] == uid and c["base"]["meta"] == {}
    assert c["local"]["meta"] == {"note": "local"}
    assert c["remote"]["meta"] == {"note": "remote"}
    assert before == (state(a), state(b), a.journal_since(), b.journal_since())
    with pytest.raises(RuntimeError, match="expected_plan"):
        a.synchronize(b, resolutions={uid: "local"})
    b.set_meta(uid, "later", True)
    with pytest.raises(RuntimeError, match="expected_plan"):
        a.synchronize(b, resolutions={uid: "local"}, expected_plan=report["plan_token"])
    report = a.synchronize(b, dry_run=True)
    assert a.synchronize(b, resolutions={uid: "remote"}, expected_plan=report["plan_token"])["applied"]
    assert state(a) == state(b)
    assert a.object(uid)["changed_by"] == "bob"


def test_delete_edit_and_delete_new_edge_conflicts(pair):
    a, b = pair
    x = a.upsert_object({"kind": "note"})
    y = a.upsert_object({"kind": "note"})
    a.synchronize(b)
    a.delete_object(x)
    b.set_meta(x, "edit", 1)
    preview = a.synchronize(b)
    assert preview["conflicts"][0]["local"] is None
    assert a.synchronize(b, resolutions={x: "remote"}, expected_plan=preview["plan_token"])["applied"]
    a.delete_object(x)
    edge = b.add_edge({"kind": "connected", "a": x, "b": y, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    before = state(a), state(b)
    report = a.synchronize(b)
    assert any(c["uuid"] == edge and "endpoint" in c["reason"] for c in report["conflicts"])
    assert before == (state(a), state(b))
    report = a.synchronize(b, resolutions={x: "remote"}, expected_plan=report["plan_token"])
    assert report["applied"] and a.edge(edge)


def test_geometry_history_bindings_and_explicit_retain(pair):
    a, b = pair
    x = a.upsert_object({"kind": "fiber"})
    first = a.put_geometry(x, _geometry())
    second = a.put_geometry(x, _geometry(value=2), expected_revision=first)
    y = a.upsert_object({"kind": "note"})
    edge = a.add_edge({"kind": "connected", "a": y, "b": x, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    a.set_approval(edge, "approved")
    assert a.synchronize(b)["applied"]
    assert state(a) == state(b)
    assert a.geometry_history(x) == b.geometry_history(x)
    assert b.read_geometry_array(first, "line_points") == a.read_geometry_array(first, "line_points")
    assert endpoint_bindings(a, edge) == endpoint_bindings(b, edge)
    before = a.journal_since()
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        a.put_geometry(x, _geometry(value=3), expected_revision=second)
    assert a.journal_since() == before
    third = a.put_geometry(x, _geometry(value=3), expected_revision=second, edge_edits=[edge_choice(a, edge, "retain")])
    assert a.synchronize(b)["applied"]
    assert b.current_geometry(x)["uuid"] == third
    assert endpoint_bindings(a, edge) == endpoint_bindings(b, edge)
    assert not b.edge_geometry_state(edge)["stale"]
    assert b.edge(edge)["approval_state"] == "pending"
    b.set_approval(edge, "approved")
    assert a.synchronize(b)["applied"]
    assert a.edge(edge)["approval_state"] == "approved"
    assert not a.edge_geometry_state(edge)["stale"]


def test_divergent_geometry_preserves_losing_history(pair):
    a, b = pair
    x = a.upsert_object({"kind": "fiber"})
    first = a.put_geometry(x, _geometry())
    a.synchronize(b)
    local = a.put_geometry(x, _geometry(value=2), expected_revision=first)
    remote = b.put_geometry(x, _geometry(value=3), expected_revision=first)
    report = a.synchronize(b)
    assert not report["applied"]
    assert a.synchronize(b, resolutions={x: "remote"}, expected_plan=report["plan_token"])["applied"]
    for db in (a, b):
        assert db.current_geometry(x)["uuid"] == remote
        assert {r["uuid"] for r in db.geometry_history(x)} == {first, local, remote}
        assert db.read_geometry_array(local, "line_points") == _geometry(value=2)["arrays"][1]["data"]


def test_failures_roll_back_both_stores(pair):
    a, b = pair
    x = a.upsert_object({"kind": "fiber"})
    a.put_geometry(x, _geometry())
    with sqlite3.connect(b.path) as sql:
        sql.execute("CREATE TRIGGER fail_sync BEFORE INSERT ON sync_baselines BEGIN SELECT RAISE(ABORT,'injected'); END")
    before = state(a), state(b), a.journal_since(), b.journal_since()
    with pytest.raises(RuntimeError, match="injected"):
        a.synchronize(b)
    assert before == (state(a), state(b), a.journal_since(), b.journal_since())
    with sqlite3.connect(b.path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_chunks").fetchone()[0] == 0
        sql.execute("DROP TRIGGER fail_sync")
    assert a.synchronize(b)["applied"]


def test_retry_uses_last_common_baseline_after_interruption(pair, tmp_path):
    a, b = pair
    x = a.upsert_object({"kind": "note"})
    a.synchronize(b)
    # Preserve the peer's last committed file, simulating loss of the second commit.
    saved = tmp_path / "peer-before.vcobj"
    with sqlite3.connect(b.path) as source, sqlite3.connect(saved) as target:
        source.backup(target)
    a.set_meta(x, "later", 1)
    y = b.upsert_object({"kind": "note", "name": "remote addition"})
    # The second store had this addition at the time sync started.
    with sqlite3.connect(b.path) as source, sqlite3.connect(saved) as target:
        source.backup(target)
    assert a.synchronize(b)["applied"]
    path = b.path
    b.close()
    shutil.copyfile(saved, path)
    with ObjectStore.open(path, author="bob") as recovered:
        a.set_meta(y, "after_interruption", True)
        report = a.synchronize(recovered)
        assert report["applied"] and not report["conflicts"]
        assert state(a) == state(recovered)
        assert recovered.object(x)["meta"] == {"later": 1}


def test_duplicate_edges_require_explicit_choice(pair):
    a, b = pair
    x = a.upsert_object({"kind": "note"})
    y = a.upsert_object({"kind": "note"})
    a.synchronize(b)
    local = a.add_edge({"kind": "connected", "a": x, "b": y, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    remote = b.add_edge({"kind": "connected", "a": y, "b": x, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    report = a.synchronize(b)
    assert not report["applied"] and "identity" in report["conflicts"][0]["reason"]
    assert a.synchronize(b, resolutions={remote: "local"}, expected_plan=report["plan_token"])["applied"]
    assert b.edge(local) and not b.edge(remote)


def test_copied_replica_identity_and_idle_handles(pair, tmp_path):
    a, b = pair
    a.upsert_object({"kind": "note"})
    copy = tmp_path / "copy.vcobj"
    with sqlite3.connect(a.path) as source, sqlite3.connect(copy) as target:
        source.backup(target)
    with ObjectStore.open(str(copy), author="copy") as c:
        assert c.replica_uuid == a.replica_uuid
        with pytest.raises(RuntimeError, match="fork_replica"):
            a.synchronize(c)
        assert c.fork_replica() != a.replica_uuid
        assert a.synchronize(c)["applied"]
        assert a.uuid == c.uuid  # logical identity is separate from replica identity
    with pytest.raises(ValueError, match="same database"):
        a.synchronize(a)
    with a.write():
        with pytest.raises(RuntimeError, match="idle stores"):
            a.synchronize(b)
    with ObjectStore.open(b.path, read_only=True) as read:
        with pytest.raises(ValueError, match="writable"):
            a.synchronize(read)


def test_corrupt_incoming_geometry_rolls_back_both_stores(pair):
    a, b = pair
    x = a.upsert_object({"kind": "fiber"})
    a.put_geometry(x, _geometry())
    with sqlite3.connect(a.path) as sql:
        sql.execute("DROP TRIGGER geometry_chunks_immutable_update")
        sql.execute("UPDATE geometry_chunks SET data=zeroblob(length(data))")
    before = state(a), state(b), a.journal_since(), b.journal_since()
    with pytest.raises(RuntimeError, match="corrupt geometry chunk"):
        a.synchronize(b)
    assert before == (state(a), state(b), a.journal_since(), b.journal_since())


def test_self_edge_sync_preserves_explicit_retain(pair):
    import struct
    a, b = pair
    x = a.upsert_object({"kind": "point_collection"})
    geometry = {"kind": "point_collection", "arrays": [
        {"name": "points", "dtype": "<f8", "shape": [2, 3], "data": struct.pack("<6d", 1, 2, 3, 4, 5, 6)},
        {"name": "point_ids", "dtype": "<i8", "shape": [2], "data": struct.pack("<2q", 1, 2)}]}
    first = a.put_geometry(x, geometry)
    edge = a.add_edge({"kind": "point_link", "a": x, "b": x,
                       "a_anchor": {"points": [{"array": "point_ids", "index": [2]}]},
                       "b_anchor": {"points": [{"array": "point_ids", "index": [1]}]}})
    assert a.synchronize(b)["applied"]
    assert state(a) == state(b)
    geometry["arrays"][0]["data"] = struct.pack("<6d", 9, 2, 3, 4, 5, 6)
    a.put_geometry(x, geometry, expected_revision=first, edge_edits=[edge_choice(a, edge, "retain")])
    before = b.edge(edge)["generation"]
    assert a.synchronize(b)["applied"]
    assert b.edge(edge)["generation"] > before and not b.edge_geometry_state(edge)["stale"]
    assert state(a) == state(b)
    assert any(j["change_kind"] == "sync" for j in b.journal_since())


def test_new_relationship_on_changed_geometry_requires_explicit_resolution(pair):
    a, b = pair
    x = a.upsert_object({"kind": "fiber"})
    first = a.put_geometry(x, _geometry())
    y = a.upsert_object({"kind": "note"})
    a.synchronize(b)
    a.put_geometry(x, _geometry(value=3), expected_revision=first)
    edge = b.add_edge({"kind": "connected", "a": x, "b": y, "a_anchor": {"whole_geometry": True}, "b_anchor": {"whole_geometry": True}})
    report = a.synchronize(b)
    assert any("explicit relationship" in c["reason"] for c in report["conflicts"])
    assert not a.synchronize(b)["applied"]
    b.remove_edge(edge)  # Caller explicitly chooses removal before retrying.
    assert a.synchronize(b)["applied"]
    assert a.edge(edge) is None


def test_kind_collision_is_reported_before_mutation(pair):
    a, b = pair
    uid = a.upsert_object({"kind": "fiber"})
    b.upsert_object({"uuid": uid, "kind": "patch"})
    report = a.synchronize(b)
    before = state(a), state(b)
    report = a.synchronize(b, resolutions={uid: "local"}, expected_plan=report["plan_token"])
    assert not report["applied"]
    assert any("kind cannot change" in c["reason"] for c in report["conflicts"])
    assert before == (state(a), state(b))


def test_sync_preserves_approval_only_changes_and_deletions(pair):
    a, b = pair
    with a.write() as tx:
        ids = [tx.upsert_object({"kind": "note"}) for _ in range(3)]
    a.synchronize(b)
    generation = a.object(ids[0])["generation"]
    a.set_approval(ids[0], "approved")  # Approval does not bump generation.
    assert a.object(ids[0])["generation"] == generation
    preview = a.synchronize(b, dry_run=True)
    assert preview["remote_changes"] == 1
    a.synchronize(b)
    assert b.object(ids[0])["approval_state"] == "approved"
    a.delete_object(ids[1])
    a.synchronize(b)
    assert b.object(ids[1]) is None
    # Tombstones remain usable by later merges.
    b.set_meta(ids[2], "later", True)
    a.synchronize(b)
    assert a.object(ids[1]) is None and a.object(ids[2])["meta"]["later"]


def test_incremental_failure_rolls_back_versions_cursors_and_geometry(pair):
    a, b = pair
    uid = a.upsert_object({"kind": "fiber"})
    first = a.put_geometry(uid, _geometry())
    a.synchronize(b)
    a.put_geometry(uid, _geometry(value=9), expected_revision=first)

    def counts(store):
        with sqlite3.connect(store.path) as sql:
            return [sql.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in (
                "sync_baselines", "sync_baseline_records", "sync_baseline_cursors", "geometry_chunks", "geometry_revisions")]

    before = state(a), state(b), a.journal_since(), b.journal_since(), counts(a), counts(b)
    with sqlite3.connect(b.path) as sql:
        sql.execute("CREATE TRIGGER fail_cursor BEFORE INSERT ON sync_baseline_cursors BEGIN SELECT RAISE(ABORT,'injected cursor failure'); END")
    with pytest.raises(RuntimeError, match="injected cursor failure"):
        a.synchronize(b)
    assert before == (state(a), state(b), a.journal_since(), b.journal_since(), counts(a), counts(b))
    with sqlite3.connect(b.path) as sql:
        sql.execute("DROP TRIGGER fail_cursor")
    assert a.synchronize(b)["applied"]
    assert state(a) == state(b)


def test_incremental_sync_forwards_history_without_head_changes(pair, tmp_path):
    a, b = pair
    uid = a.upsert_object({"kind": "fiber"})
    first = a.put_geometry(uid, _geometry())
    a.synchronize(b)
    with ObjectStore.create(str(tmp_path / "third.vcobj"), author="third") as c:
        a.synchronize(c)
        losing = c.put_geometry(uid, _geometry(value=7), expected_revision=first)
        current = a.put_geometry(uid, _geometry(value=8), expected_revision=first)
        a.synchronize(b)
        report = a.synchronize(c)
        a.synchronize(c, resolutions={uid: "local"}, expected_plan=report["plan_token"])
        assert a.current_geometry(uid)["uuid"] == b.current_geometry(uid)["uuid"] == current
        # a received losing history but its object record did not change.
        report = a.synchronize(b)
        assert report["applied"] and report["local_changes"] == report["remote_changes"] == 0
        assert b.read_geometry_array(losing, "line_points") == _geometry(value=7)["arrays"][1]["data"]


def test_oldest_schema_upgrade_preserves_geometry_queries_and_sync(pair):
    a, b = pair
    uid = a.upsert_object({"kind": "track", "track": {"family": "H", "length": 8}})
    revision = a.put_geometry(uid, track([(1, 2, 3)]), patch_tile_size=0)
    a.synchronize(b)
    legacy = {}
    for kind, records in state(a).items():
        for record in records:
            record["record_type"] = kind
            # A schema-v2 snapshot predates library-managed lifecycle state.
            record.pop("retired", None)
            record.pop("lifecycle_epoch", None)
            if "geometry_bindings" in record:
                record["geometry_bindings"].pop("stale", None)
            legacy[record["uuid"]] = record
    paths = a.path, b.path
    a.close(); b.close()
    for path in paths:
        with sqlite3.connect(path) as sql:
            from object_helpers import remove_spatial_schema, remove_history_schema
            remove_history_schema(sql)
            remove_spatial_schema(sql)
            sql.execute("DROP TABLE sync_baseline_records")
            sql.execute("DROP TABLE sync_baseline_cursors")
            sql.execute("DROP INDEX sync_baselines_peer_seq")
            sql.execute("UPDATE sync_baselines SET state=?", (json.dumps(legacy),))
            for table in ("geometry_deltas", "geometry_current_chunks", "geometry_materializations"):
                sql.execute(f"DROP TABLE {table}")
            sql.execute("PRAGMA user_version=2")
        with ObjectStore.open(path, read_only=True) as ro:
            assert ro.object(uid)
            assert ro.read_geometry_array(revision, "points") == track([(1, 2, 3)])["arrays"][0]["data"]
        with sqlite3.connect(path) as sql:
            assert sql.execute("PRAGMA user_version").fetchone()[0] == 2
    with ObjectStore.open(paths[0], author="alice") as a, ObjectStore.open(paths[1], author="bob") as b:
        for store in (a, b):
            assert store.select_objects(object_kind="track", family="H", order_by="length_desc")[0]["uuid"] == uid
            regions = store.spatial_regions([1, 2, 3, 1, 2, 3])
            assert len(regions) == 1 and regions[0]["revision_uuid"] == revision
        a.set_meta(uid, "migrated", True)
        assert a.synchronize(b)["applied"]
        assert state(a) == state(b)
        with sqlite3.connect(a.path) as sql:
            assert sql.execute("PRAGMA user_version").fetchone()[0] == 12
            assert sql.execute("SELECT count(*) FROM sync_baseline_records").fetchone()[0] == 1
        a.set_approval(uid, "approved")
        assert a.synchronize(b)["applied"]
        assert state(a) == state(b)
