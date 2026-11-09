"""An edge's UUID identifies an independent observation."""
import pytest


def test_distinct_observations_use_their_own_keys(db):
    a = db.upsert_object(dict(kind="fiber", name="a"))
    b = db.upsert_object(dict(kind="fiber", name="b"))
    first = db.add_edge(dict(kind="perpendicular", a=a, b=b))
    second = db.add_edge(dict(kind="perpendicular", a=a, b=b))
    assert first != second
    assert db.edge(first)["uuid"] == first
    assert "relation_key" not in db.edge(first)
    with pytest.raises(Exception):
        db.add_edge(dict(uuid=first, kind="perpendicular", a=a, b=b))
    assert len(db.edges_of(a)) == 2


def test_record_versions_share_one_commit_and_lineage(db):
    with db.write() as tx:
        a = tx.upsert_object(dict(kind="fiber"))
        b = tx.upsert_object(dict(kind="fiber"))
        edge = tx.add_edge(dict(kind="perpendicular", a=a, b=b))
    revisions = {uid: db.current_object_revision(uid) for uid in (a, b)}
    status = db.status()
    generation = status["generation"]
    assert db.edge(edge)["version"] == 1
    assert all(revision["generation"] == generation for revision in revisions.values())
    assert all(revision["version"] == 1 and revision["commit_sequence"] == status["commit_sequence"]
               for revision in revisions.values())
    relationships = db.relationships_at_versions([
        dict(store=status["store"], generation=generation, object=uid, version=revision["version"])
        for uid, revision in revisions.items()
    ])
    assert len(relationships) == 1
    assert relationships[0]["generation"] == generation
