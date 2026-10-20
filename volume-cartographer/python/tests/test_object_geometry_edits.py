"""Point dependencies and explicit relationship decisions during geometry edits."""
import json
import sqlite3
import struct

import pytest
from vc.objects import ObjectStore
from object_helpers import edge_choice, canonical_edge


def array(name, shape, values, dtype="<f8"):
    scalar = {"<f8": "d", "<f4": "f", "<i8": "q"}[dtype]
    return dict(name=name, dtype=dtype, shape=shape, data=struct.pack("<" + scalar * len(values), *values))


def fiber(changed=5, value=5):
    points = [float(i) for i in range(18)]
    points[changed * 3] = value
    return dict(kind="fiber", arrays=[array("control_points", [6, 3], points), array("line_points", [0, 3], [])])


def anchor(name, *index):
    return {"points": [{"array": name, "index": list(index)}]}


@pytest.fixture
def linked(tmp_path):
    with ObjectStore.create(str(tmp_path / "edit.vcobj"), author="editor") as db:
        a = db.upsert_object({"kind": "fiber"})
        b = db.upsert_object({"kind": "patch"})
        first = db.put_geometry(a, fiber())
        patch = dict(kind="patch", attributes={"scale": [1, 1]}, arrays=[array("points", [2, 3, 3], list(range(18)), "<f4")])
        db.put_geometry(b, patch)
        edge = db.add_edge(dict(kind="contact", a=a, b=b, a_anchor=anchor("control_points", 5),
                                b_anchor=anchor("points", 1, 2), tags=["review"], meta={"note": "keep"}))
        db.set_approval(edge, "approved")
        yield db, a, b, first, edge


def test_referenced_point_requires_decision_and_unrelated_point_preserves_edge(linked):
    db, a, b, first, edge = linked
    before = db.export_json(), db.journal_since()
    assert [e["uuid"] for e in db.affected_geometry_edges(a, fiber(value=99))] == [edge]
    with pytest.raises(RuntimeError, match="explicit retain/update/remove"):
        db.put_geometry(a, fiber(value=99), expected_revision=first)
    assert (db.export_json(), db.journal_since()) == before
    untouched = db.edge(edge), db.edge_geometry_state(edge)
    unrelated = fiber(changed=0, value=99)
    # Restore the linked point's original value; only point 0 moves.
    values = list(struct.unpack("<18d", unrelated["arrays"][0]["data"]))
    values[15] = 5
    unrelated["arrays"][0] = array("control_points", [6, 3], values)
    assert db.affected_geometry_edges(a, unrelated) == []
    db.put_geometry(a, unrelated, expected_revision=first)
    assert (db.edge(edge), db.edge_geometry_state(edge)) == untouched
    assert not db.edge_geometry_state(edge)["stale"]


@pytest.mark.parametrize("action", ["retain", "update", "remove"])
def test_explicit_decisions_commit_with_geometry(linked, action):
    db, a, b, first, edge = linked
    before = db.edge(edge)
    choice = edge_choice(db, edge, action)
    if action == "update":
        choice.update(a_anchor=anchor("control_points", 4), b_anchor=anchor("points", 0, 1))
    second = db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[choice])
    assert second != first
    if action == "remove":
        assert db.edge(edge) is None
    else:
        after = db.edge(edge)
        assert after["generation"] == before["generation"] + 1
        assert after["approval_state"] == "pending"
        assert after["tags"] == before["tags"] and after["meta"] == before["meta"]
        assert after["a_anchor"] == (before["a_anchor"] if action == "retain" else choice["a_anchor"])
        assert not db.edge_geometry_state(edge)["stale"]
        assert db.journal_since()[-1]["change_kind"] == "geometry_repaired"


def test_patch_coordinates_and_interpolation_support_are_dependencies(linked):
    db, a, b, first, edge = linked
    patch = dict(kind="patch", attributes={"scale": [1, 1]}, arrays=[array("points", [2, 3, 3], list(range(18)), "<f4")])
    values = list(range(18)); values[15] = 99
    patch["arrays"][0] = array("points", [2, 3, 3], values, "<f4")
    assert [e["uuid"] for e in db.affected_geometry_edges(b, patch)] == [edge]
    with pytest.raises(RuntimeError, match="explicit"):
        db.put_geometry(b, patch, expected_revision=db.current_geometry(b)["uuid"])


@pytest.mark.parametrize("mutation", ["edge", "peer", "edited"])
def test_stale_decisions_reject_without_writes(linked, mutation):
    db, a, b, first, edge = linked
    choice = edge_choice(db, edge, "retain")
    if mutation == "edge":
        db.set_meta(edge, "new_review", True)
    elif mutation == "peer":
        choice["b_revision"] = first  # A real UUID belonging to the wrong endpoint.
    else:
        choice["a_revision"] = None
    before = db.export_json(), db.journal_since()
    with pytest.raises(RuntimeError, match="changed"):
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[choice])
    assert (db.export_json(), db.journal_since()) == before


@pytest.mark.parametrize("fault", ["binding", "journal"])
def test_failed_relationship_publication_rolls_back_geometry_even_if_caught(linked, fault):
    db, a, b, first, edge = linked
    choice = edge_choice(db, edge, "retain")
    before = db.export_json(), db.journal_since()
    with sqlite3.connect(db.path) as sql:
        if fault == "binding":
            sql.execute("CREATE TRIGGER fail BEFORE UPDATE ON edge_geometry_bindings BEGIN SELECT RAISE(ABORT,'injected'); END")
        else:
            sql.execute("CREATE TRIGGER fail BEFORE INSERT ON journal WHEN NEW.change_kind='geometry_repaired' BEGIN SELECT RAISE(ABORT,'injected'); END")
    with db.write() as tx:
        with pytest.raises(RuntimeError, match="injected"):
            tx.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[choice])
        tx.upsert_object({"kind": "note", "name": "survives"})
    assert db.current_geometry(a)["uuid"] == first
    assert len(db.geometry_history(a)) == 1
    assert db.journal_since()[:-1] == before[1]
    assert db.edge(edge)["approval_state"] == "approved"
    assert not db.edge_geometry_state(edge)["stale"]
    with sqlite3.connect(db.path) as sql:
        assert sql.execute("SELECT count(*) FROM geometry_revisions").fetchone()[0] == 2


def test_deleted_point_cannot_be_retained_and_update_requires_valid_references(linked):
    db, a, b, first, edge = linked
    shorter = fiber()
    shorter["arrays"][0] = array("control_points", [5, 3], list(range(15)))
    with pytest.raises(RuntimeError, match="missing or invalid point"):
        db.put_geometry(a, shorter, expected_revision=first, edge_edits=[edge_choice(db, edge, "retain")])
    choice = edge_choice(db, edge, "update", a_anchor=anchor("control_points", 4), b_anchor=anchor("points", 1, 2))
    db.put_geometry(a, shorter, expected_revision=first, edge_edits=[choice])
    assert db.edge(edge)["a_anchor"] == anchor("control_points", 4)


def test_multiple_edges_require_complete_nonduplicate_decisions(linked):
    db, a, b, first, edge = linked
    second = db.add_edge(dict(kind="other-contact", a=a, b=b, a_anchor=anchor("control_points", 5), b_anchor=anchor("points", 0, 0)))
    choices = [edge_choice(db, e, "retain") for e in (edge, second)]
    with pytest.raises(RuntimeError, match="explicit"):
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=choices[:1])
    with pytest.raises(ValueError, match="duplicate"):
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=choices + choices[:1])
    db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=choices)
    assert all(not db.edge_geometry_state(e)["stale"] for e in (edge, second))


def test_point_ids_survive_reordering_and_deletion_requires_choice(tmp_path):
    def collection(ids, values):
        return dict(kind="point_collection", arrays=[array("points", [len(ids), 3], values), array("point_ids", [len(ids)], ids, "<i8")])
    with ObjectStore.create(str(tmp_path / "ids.vcobj"), author="test") as db:
        a = db.upsert_object({"kind": "point_collection"})
        first = db.put_geometry(a, collection([20, 10], [1, 2, 3, 4, 5, 6]))
        left, right = anchor("point_ids", 20), anchor("point_ids", 10)
        edge = db.add_edge(dict(kind="point_link", a=a, b=a, a_anchor=left, b_anchor=right))
        before = db.edge(edge)
        second = db.put_geometry(a, collection([10, 20], [4, 5, 6, 1, 2, 3]), expected_revision=first)
        assert db.edge(edge) == before and not db.edge_geometry_state(edge)["stale"]
        with pytest.raises(RuntimeError, match="explicit"):
            db.put_geometry(a, collection([10], [4, 5, 6]), expected_revision=second)
        db.put_geometry(a, collection([10], [4, 5, 6]), expected_revision=second, edge_edits=[edge_choice(db, edge, "remove")])
        assert db.edge(edge) is None


def test_point_dependencies_persist_export_and_sync(linked, tmp_path):
    db, a, b, first, edge = linked
    with ObjectStore.create(str(tmp_path / "peer.vcobj"), author="peer") as peer:
        assert db.synchronize(peer)["applied"]
        record = peer.edge(edge)
        assert record["a_anchor" if record["a"] == a else "b_anchor"] == anchor("control_points", 5)
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[edge_choice(db, edge, "retain")])
        assert db.synchronize(peer)["applied"]
        assert canonical_edge(peer.edge(edge)) == canonical_edge(db.edge(edge))
        assert not peer.edge_geometry_state(edge)["stale"]
    with ObjectStore.open(db.path, read_only=True) as reopened:
        assert reopened.edge(edge)["b_anchor"] == anchor("points", 1, 2)
        exported = json.loads(reopened.export_json())["edges"][0]
        assert "points" in exported["a_anchor"] and "points" in exported["b_anchor"]


def test_geometry_endpoints_require_typed_dependencies_and_reject_invalid_points(linked):
    db, a, b, first, edge = linked
    with pytest.raises(RuntimeError, match="explicit point dependencies"):
        db.add_edge(dict(kind="untyped", a=a, b=b))
    with pytest.raises(RuntimeError, match="missing or invalid point"):
        db.add_edge(dict(kind="wrong-index", a=a, b=b, a_anchor=anchor("control_points", 99), b_anchor=anchor("points", 1, 2)))
    choice = edge_choice(db, edge, "update", a_anchor=anchor("control_points", 4), b_anchor=anchor("points", 20, 145))
    before = db.journal_since()
    with pytest.raises(RuntimeError, match="missing or invalid point"):
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[choice])
    assert db.journal_since() == before and db.current_geometry(a)["uuid"] == first


def test_standalone_relationship_update_checks_snapshot_and_collision(linked):
    db, a, b, first, edge = linked
    repair = edge_choice(db, edge, "update", a_anchor=anchor("control_points", 4), b_anchor=anchor("points", 0, 1))
    repair.pop("action")
    with ObjectStore.open(db.path, author="other") as other:
        other.set_meta(edge, "review", "concurrent")
    before = db.export_json(), db.journal_since()
    with pytest.raises(RuntimeError, match="generation changed"):
        db.repair_edge_geometry(repair)
    assert (db.export_json(), db.journal_since()) == before
    repair["expected_generation"] = db.edge(edge)["generation"]
    db.repair_edge_geometry(repair)
    assert db.edge(edge)["a_anchor"] == anchor("control_points", 4)
    # A second link to different points is a distinct relationship.
    other = db.add_edge(dict(kind="contact", a=a, b=b, a_anchor=anchor("control_points", 3), b_anchor=anchor("points", 0, 1)))
    repair["expected_generation"] = db.edge(edge)["generation"]
    repair["a_anchor"] = anchor("control_points", 3)
    before = db.edge(edge), db.journal_since()
    with pytest.raises(RuntimeError, match="UNIQUE"):
        db.repair_edge_geometry(repair)
    assert (db.edge(edge), db.journal_since()) == before
    assert db.edge(other)


def test_sync_of_unrelated_point_edit_preserves_approval(linked, tmp_path):
    db, a, b, first, edge = linked
    with ObjectStore.create(str(tmp_path / "unrelated.vcobj"), author="peer") as peer:
        db.synchronize(peer)
        geometry = fiber()
        values = list(struct.unpack("<18d", geometry["arrays"][0]["data"]))
        values[0] = 999
        geometry["arrays"][0] = array("control_points", [6, 3], values)
        before = db.edge(edge)
        db.put_geometry(a, geometry, expected_revision=first)
        assert db.synchronize(peer)["applied"]
        assert canonical_edge(peer.edge(edge)) == canonical_edge(before)
        assert not peer.edge_geometry_state(edge)["stale"]


@pytest.mark.parametrize("action", ["stale", "automatic", "ignore"])
def test_relationship_decisions_have_no_implicit_resolution_mode(linked, action):
    db, a, b, first, edge = linked
    before = db.export_json(), db.journal_since()
    with pytest.raises(ValueError, match="retain, update, remove, or keep_recorded"):
        db.put_geometry(a, fiber(value=99), expected_revision=first, edge_edits=[edge_choice(db, edge, action)])
    assert (db.export_json(), db.journal_since()) == before
