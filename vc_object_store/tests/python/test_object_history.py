"""Unified revisions, pinned graph membership and checked atomic actions."""
import json
import sqlite3
import struct

import pytest
from vc_object_store import ObjectStore
from object_helpers import edge_choice, sqlite_rows, track


def heads(db, *objects):
    return {obj: db.current_object_revision(obj)["uuid"] for obj in objects}


def linked(db, metadata_keys=()):
    a = db.upsert_object(dict(kind="track", meta={"winding": 1}))
    b = db.upsert_object(dict(kind="track", meta={"winding": 1}))
    ga = db.put_geometry(a, track([(1, 2, 3), (4, 5, 6)]))
    gb = db.put_geometry(b, track([(1, 2, 3)]))
    anchor = dict(points=[dict(array="points", index=[0])],
                  metadata_keys=list(metadata_keys))
    edge = db.add_edge(dict(kind="connected", a=a, b=b,
                            a_anchor=anchor, b_anchor=anchor))
    return a, b, ga, gb, edge


def test_metadata_and_geometry_share_revision_and_reuse_storage(db):
    uid = db.upsert_object(dict(kind="track", meta={"large": "x" * 10000, "winding": 1}))
    g = db.put_geometry(uid, track([(1, 2, 3)]))
    before = db.current_object_revision(uid)
    geometry_rows = sqlite_rows(db.path, "SELECT count(*) FROM geometry_revisions")
    with db.write(expected_revisions={uid: before["uuid"]}) as tx:
        tx.set_meta(uid, "winding", 2)
        tx.set_meta(uid, "null", None)
    after = db.current_object_revision(uid)
    assert after["parent_uuid"] == before["uuid"]
    assert after["geometry_revision"] == g
    assert db.object_revision(before["uuid"])["record"]["meta"]["winding"] == 1
    assert after["record"]["meta"]["winding"] == 2
    assert sqlite_rows(db.path, "SELECT count(*) FROM geometry_revisions") == geometry_rows
    delta = sqlite_rows(db.path, "SELECT metadata_delta FROM object_revisions WHERE uuid=?", [after["uuid"]])[0][0]
    assert "large" not in delta and len(delta) < 2000
    with db.write(expected_revisions={uid: after["uuid"]}) as tx:
        tx.set_meta(uid, "winding", 3)
        tx.edit_geometry(uid, [dict(array="points", element_offset=0, data=struct.pack("<i", 9))],
                         expected_revision=g)
    latest = db.current_object_revision(uid)
    assert latest["parent_uuid"] == after["uuid"]
    assert latest["geometry_revision"] != g
    assert db.read_geometry_array(g, "points") == struct.pack("<3i", 1, 2, 3)
    assert len(db.object_history(uid)) == 4
    with db.write(expected_revisions={uid: latest["uuid"]}) as tx:
        tx.set_meta(uid, "winding", 3)
    assert db.current_object_revision(uid) == latest


def test_shared_membership_pins_and_selected_revision_queries(db):
    a, b, ga, gb, edge = linked(db)
    original = heads(db, a, b)
    recorded = db.recorded_relationships(original[a])
    assert recorded == db.recorded_relationships(original[b])
    assert len(recorded) == 1
    assert db.relationships_at(original)[0]["applicable"]
    view = db.revision_read_view({a: None, b: None})
    assert view["selection"] == original
    assert {obj["object_uuid"]: obj["geometry_revision"] for obj in view["objects"]} == {a: ga, b: gb}
    assert view["relationships"] == db.relationships_at(original)
    with db.write(expected_revisions=original) as tx:
        tx.set_meta(a, "unrelated", True)
    changed = heads(db, a, b)
    assert changed[a] != original[a] and changed[b] == original[b]
    assert db.recorded_relationships(changed[a]) == recorded
    assert db.relationships_at(changed)[0]["applicable"]
    # Explicitly preserve the historical reference, without retargeting it.
    decision = edge_choice(db, edge, "keep_recorded")
    with db.write(expected_revisions=changed) as tx:
        tx.edit_geometry(a, [dict(array="points", element_offset=0, data=struct.pack("<i", 9))],
                         expected_revision=ga, edge_edits=[decision])
    moved = heads(db, a, b)
    assert moved[b] == original[b]
    assert db.recorded_relationships(moved[a]) == recorded
    assert not db.relationships_at(moved)[0]["applicable"]
    assert db.relationships_at(original)[0]["applicable"]
    assert view["selection"] == original  # detached view retains the applied inputs
    assert db.object_revision(recorded[0]["b_revision"])["geometry_revision"] == gb
    # Revalidation is a separate, explicit shared edit.
    with db.write(expected_revisions=moved) as tx:
        tx.retain_relationship(edge)
    validated = heads(db, a, b)
    assert validated[a] != moved[a] and validated[b] != moved[b]
    assert db.relationships_at({a: None, b: None})[0]["applicable"]
    assert db.recorded_relationships(validated[a]) == db.recorded_relationships(validated[b])
    assert db.recorded_relationships(original[a]) == recorded
    # Conflicting memberships cannot masquerade as a selected connection.
    assert not any(e["applicable"] for e in db.relationships_at({a: original[a], b: validated[b]}))
    with db.write(expected_revisions=validated) as tx:
        tx.remove_edge(edge)
    deleted = heads(db, a, b)
    assert all(deleted[obj] != validated[obj] for obj in (a, b))
    assert db.relationships_at(deleted) == []
    assert db.recorded_relationships(original[a]) == recorded
    assert db.relationships_at(original)[0]["applicable"]


def test_checked_action_conflicts_and_missing_participants_roll_back_everything(db):
    a, b, ga, gb, edge = linked(db)
    before = heads(db, a, b)
    journal = db.journal_since()
    with pytest.raises(RuntimeError, match="expected"):
        with db.write(expected_revisions={a: before[a]}) as tx:
            tx.set_meta(a, "aborted", True)
            tx.set_meta(edge, "weight", 2)
    assert heads(db, a, b) == before and db.journal_since() == journal
    assert "aborted" not in db.object(a)["meta"]
    # Even an unchanged validation endpoint must be checked.
    with pytest.raises(RuntimeError, match="expected"):
        with db.write(expected_revisions={a: before[a]}) as tx:
            tx.edit_geometry(a, [dict(array="points", element_offset=0, data=struct.pack("<i", 8))],
                             expected_revision=ga, edge_edits=[edge_choice(db, edge, "keep_recorded")])
    assert heads(db, a, b) == before
    db.set_meta(b, "concurrent", 1)
    with pytest.raises(RuntimeError, match="conflict"):
        with db.write(expected_revisions=before):
            pytest.fail("stale action entered")
    # A caught geometry conflict still poisons the complete checked action.
    current = heads(db, a, b)
    with pytest.raises(RuntimeError, match="conflict"):
        with db.write(expected_revisions=current) as tx:
            tx.set_meta(a, "aborted", True)
            with pytest.raises(RuntimeError, match="revision changed"):
                tx.edit_geometry(a, [], expected_revision=gb)
    assert heads(db, a, b) == current and "aborted" not in db.object(a)["meta"]


def test_declared_metadata_dependencies_need_final_state_decisions(db):
    a, b, ga, gb, edge = linked(db, ["winding"])
    original = heads(db, a, b)
    with pytest.raises(RuntimeError, match="explicit"):
        with db.write(expected_revisions=original) as tx:
            tx.set_meta(a, "winding", 2)
    assert heads(db, a, b) == original
    with db.write(expected_revisions=original) as tx:
        tx.set_meta(a, "winding", 2)
        tx.keep_recorded_relationship(edge)
    preserved = heads(db, a, b)
    assert preserved[b] == original[b]
    assert not db.relationships_at(preserved)[0]["applicable"]
    assert db.relationships_at(original)[0]["applicable"]
    with db.write(expected_revisions=preserved) as tx:
        tx.retain_relationship(edge)
    current = heads(db, a, b)
    assert db.relationships_at(current)[0]["applicable"]
    assert current[b] != preserved[b]
    # Changing edge annotations must not silently validate an old metadata pin.
    with db.write(expected_revisions=current) as tx:
        tx.set_meta(a, "winding", 5)
        tx.keep_recorded_relationship(edge)
    stale = heads(db, a, b)
    with db.write(expected_revisions=stale) as tx:
        tx.set_meta(edge, "review_note", "still historical")
    assert not db.relationships_at({a: None, b: None})[0]["applicable"]
    current = heads(db, a, b)
    with pytest.raises(RuntimeError, match="explicit"):
        with db.write(expected_revisions=current) as tx:
            tx.set_meta(a, "winding", 3)
            tx.retain_relationship(edge)
            tx.set_meta(a, "winding", 4)
    assert heads(db, a, b) == current


def test_restoration_uses_the_same_endpoint_validation_and_history(db):
    a, b, ga, gb, edge = linked(db)
    old = heads(db, a, b)
    with db.write(expected_revisions=old) as tx:
        changed = tx.edit_geometry(a, [dict(array="points", element_offset=0, data=struct.pack("<i", 7))],
                                  expected_revision=ga, edge_edits=[edge_choice(db, edge, "retain")])
        tx.set_meta(a, "edited", True)
    current = heads(db, a, b)
    historical = db.object_revision(old[a])
    with pytest.raises(RuntimeError, match="explicit"):
        with db.write(expected_revisions=current) as tx:
            tx.upsert_object(historical["record"])
            tx.put_geometry(a, track([(1, 2, 3), (4, 5, 6)]), expected_revision=changed)
    assert heads(db, a, b) == current
    with db.write(expected_revisions=current) as tx:
        tx.upsert_object(historical["record"])
        restored = tx.put_geometry(a, track([(1, 2, 3), (4, 5, 6)]), expected_revision=changed,
                                   edge_edits=[edge_choice(db, edge, "retain")])
    assert restored != ga
    assert db.current_object_revision(a)["parent_uuid"] == current[a]
    assert db.read_geometry_array(restored, "points") == db.read_geometry_array(ga, "points")
    assert "edited" not in db.current_object_revision(a)["record"]["meta"]
    assert db.object_revision(current[a])["record"]["meta"]["edited"]
    assert db.relationships_at(old)[0]["applicable"]


def test_relationship_only_edits_do_not_copy_geometry_or_other_relationships(db):
    a, b, ga, gb, edge = linked(db)
    before = heads(db, a, b)
    counts = sqlite_rows(db.path, "SELECT (SELECT count(*) FROM geometry_revisions), (SELECT count(*) FROM geometry_chunks)")
    with db.write(expected_revisions=before) as tx:
        tx.set_meta(edge, "weight", 42)
    after = heads(db, a, b)
    assert all(after[obj] != before[obj] for obj in (a, b))
    assert db.recorded_relationships(before[a])[0]["record"]["meta"] == {}
    assert db.recorded_relationships(after[a])[0]["record"]["meta"] == {"weight": 42}
    assert sqlite_rows(db.path, "SELECT (SELECT count(*) FROM geometry_revisions), (SELECT count(*) FROM geometry_chunks)") == counts
    versions = sqlite_rows(db.path, "SELECT count(*) FROM relationship_revisions")[0][0]
    for i in range(5):
        with db.write(expected_revisions=heads(db, a)) as tx:
            tx.set_meta(a, "label", i)
    assert sqlite_rows(db.path, "SELECT count(*) FROM relationship_revisions")[0][0] == versions
    assert sqlite_rows(db.path, "SELECT count(*) FROM object_relationship_deltas")[0][0] == 4


def test_history_survives_reopen_and_cannot_be_mutated(db, db_path):
    a, b, ga, gb, edge = linked(db)
    before = heads(db, a, b)
    original = db.recorded_relationships(before[a])
    db.set_meta(a, "winding", 4)
    with ObjectStore.open(db_path, read_only=True) as reopened:
        assert reopened.object_revision(before[a])["record"]["meta"]["winding"] == 1
        assert reopened.recorded_relationships(before[a]) == original
    for table in ("object_revisions", "relationship_revisions", "object_relationship_deltas"):
        with pytest.raises(Exception):
            sqlite_rows(db_path, f"DELETE FROM {table}")


def test_time_selection_recovers_complete_graph_and_never_half_an_action(db):
    a, b, ga, gb, edge = linked(db)
    old = heads(db, a, b)
    at = db.current_object_revision(a)["published_at_us"]
    assert at == db.current_object_revision(b)["published_at_us"]
    saved = db.revision_read_view_at(at)
    assert saved["selection"] == old
    with db.write(expected_revisions=old) as tx:
        tx.set_meta(a, "label", "after")
        tx.set_meta(b, "label", "after")
        tx.set_meta(edge, "confidence", 0.9)
    a2, b2 = db.current_object_revision(a), db.current_object_revision(b)
    after = a2["published_at_us"]
    assert after == b2["published_at_us"] and after > at
    assert db.revision_selection_at(after - 1) == old
    assert db.revision_selection_at(after) == heads(db, a, b)
    assert db.revision_read_view_at(at) == saved
    assert saved["relationships"][0]["record"]["meta"] == {}
    assert saved["relationships"][0]["applicable"]
    assert {obj["object_uuid"]: obj["geometry_revision"] for obj in saved["objects"]} == {a: ga, b: gb}
    # The complete old graph survives removal of a link and later object creation.
    with db.write(expected_revisions=heads(db, a, b)) as tx:
        tx.remove_edge(edge)
    extra = db.upsert_object(dict(kind="note"))
    assert extra not in db.revision_selection_at(at)
    assert db.revision_read_view_at(at) == saved
    assert db.revision_selection_at(0) == {}
    with pytest.raises(ValueError):
        db.revision_selection_at(-1)


def test_time_selection_respects_physical_deletion_and_clock_rollback(db):
    a = db.upsert_object(dict(kind="note"))
    b = db.upsert_object(dict(kind="note"))
    edge = db.add_edge(dict(kind="reference", a=a, b=b))
    old = db.current_object_revision(a)
    at = old["published_at_us"]
    saved = db.revision_read_view_at(at)
    db.remove_edge(edge)
    db.delete_object(a)
    deletion = db.object_history(a)[-1]
    assert deletion["deleted"]
    assert a not in db.revision_selection_at(deletion["published_at_us"])
    assert db.recorded_relationships(deletion["uuid"]) == []
    assert db.revision_read_view_at(at) == saved
    # Simulate an earlier publication having been stamped by a faster clock.
    future = db.current_object_revision(b)["published_at_us"] + 10**12
    sqlite_rows(db.path, "DROP TRIGGER object_revisions_UPDATE")
    sqlite_rows(db.path, "UPDATE object_revisions SET published_at_us=? WHERE uuid=?",
                (future, db.current_object_revision(b)["uuid"]))
    db.set_meta(b, "later", True)
    assert db.current_object_revision(b)["published_at_us"] == future + 1


def test_v8_migration_with_stale_binding_marks_unrecoverable_metadata(tmp_path):
    from object_helpers import remove_history_schema
    path = str(tmp_path / "old.vcobj")
    with ObjectStore.create(path, author="test") as db:
        a, b, ga, gb, edge = linked(db)
        # Bypass the public edit policy to simulate a pre-v9 stale binding.
        db.remove_edge(edge)
        newer = db.put_geometry(a, track([(9, 2, 3)]), expected_revision=ga)
        edge = db.add_edge(dict(kind="connected", a=a, b=b,
                               a_anchor={"whole_geometry": True}, b_anchor={"whole_geometry": True}))
        order = db.edge(edge)
    with sqlite3.connect(path) as sql:
        remove_history_schema(sql)
        column = "a_revision" if order["a"] == a else "b_revision"
        sql.execute(f"UPDATE edge_geometry_bindings SET {column}=? WHERE edge_id=(SELECT id FROM edges WHERE uuid=?)",
                    (bytes.fromhex(ga.replace("-", "")), bytes.fromhex(edge.replace("-", ""))))
        sql.execute("PRAGMA user_version=8")
    with ObjectStore.open(path, author="test") as db:
        current = db.current_object_revision(a)
        assert current["geometry_revision"] == newer and not current["metadata_complete"]
        recorded = db.recorded_relationships(current["uuid"])[0]
        pinned = recorded["a_revision"] if order["a"] == a else recorded["b_revision"]
        old = db.object_revision(pinned)
        assert old["geometry_revision"] == ga and not old["metadata_complete"]
        assert old["published_at_us"] is None
        assert db.revision_selection_at(current["published_at_us"])[a] == current["uuid"]
        assert not db.relationships_at({a: None, b: None})[0]["applicable"]
        assert db.read_geometry_array(ga, "points") == track([(1, 2, 3), (4, 5, 6)])["arrays"][0]["data"]


def test_current_state_sync_preserves_older_geometry_bindings(db, tmp_path):
    a, b, ga, gb, edge = linked(db)
    # Editing an unrelated point leaves the link bound to ga, not the new head.
    db.edit_geometry(a, [dict(array="points", element_offset=3, data=struct.pack("<i", 99))],
                     expected_revision=ga)
    with ObjectStore.create(str(tmp_path / "peer.vcobj"), author="peer") as peer:
        assert db.synchronize(peer)["applied"]
        current = peer.current_object_revision(a)
        view = peer.revision_read_view_at(current["published_at_us"])
        assert view["relationships"][0]["applicable"]
        link = view["relationships"][0]
        side = "a" if link["record"]["a"] == a else "b"
        target = peer.object_revision(link[side + "_revision"])
        assert target["geometry_revision"] == ga
        assert not target["metadata_complete"] and target["published_at_us"] is None
        assert peer.read_geometry_array(ga, "points") == db.read_geometry_array(ga, "points")
