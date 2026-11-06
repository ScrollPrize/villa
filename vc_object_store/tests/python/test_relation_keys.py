"""Independent observations retain their native relationship identities."""
import pytest


def test_distinct_observations_roundtrip_and_keys_are_immutable(db):
    a = db.upsert_object(dict(kind="fiber", name="a"))
    b = db.upsert_object(dict(kind="fiber", name="b"))
    first = db.add_edge(dict(kind="perpendicular", a=a, b=b, relation_key="sample:1"))
    second = db.add_edge(dict(kind="perpendicular", a=a, b=b, relation_key="sample:2"))
    assert first != second
    assert db.edge(first)["relation_key"] == "sample:1"
    assert db.edge(second)["relation_key"] == "sample:2"
    with pytest.raises(RuntimeError, match="UNIQUE constraint"):
        db.add_edge(dict(kind="perpendicular", a=a, b=b, relation_key="sample:1"))
    with pytest.raises(Exception):
        db.add_edge(dict(uuid=first, kind="perpendicular", a=a, b=b, relation_key="changed"))
    assert db.edge(first)["relation_key"] == "sample:1"


def test_default_key_preserves_legacy_identity(db):
    a = db.upsert_object(dict(kind="fiber"))
    b = db.upsert_object(dict(kind="fiber"))
    edge = db.add_edge(dict(kind="parallel", a=a, b=b))
    assert db.edge(edge)["relation_key"] == ""
    with pytest.raises(RuntimeError, match="UNIQUE constraint"):
        db.add_edge(dict(kind="parallel", a=a, b=b))


def test_history_generations_are_the_shared_publication_generation(db):
    with db.write() as tx:
        a = tx.upsert_object(dict(kind="fiber"))
        b = tx.upsert_object(dict(kind="fiber"))
        edge = tx.add_edge(dict(kind="perpendicular", a=a, b=b, relation_key="observed"))
    revisions = {uid: db.current_object_revision(uid) for uid in (a, b)}
    generation = db.edge(edge)["generation"]
    assert generation == 2
    assert all(revision["generation"] == generation for revision in revisions.values())
    relationships = db.relationships_at({uid: revision["uuid"] for uid, revision in revisions.items()})
    assert len(relationships) == 1
    assert relationships[0]["generation"] == generation
