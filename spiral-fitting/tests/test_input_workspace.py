"""Revision/command contract; deliberately independent of the CUDA fitter."""

from concurrent.futures import ThreadPoolExecutor
import threading
from uuid import uuid4

import pytest

from input_workspace import (
    Catalog, Change, Content, InputIdentity, MutationCoordinator, WorkspaceLease,
)
from service_http import ApiError


def identity(kind="pcl", *, source="same_windings.json", collection_id=None):
    return InputIdentity(str(uuid4()), kind, source, "same_winding"
                         if kind == "pcl" else None, collection_id)


def content(value):
    return Content.from_json({"value": value})


def test_baseline_identity_survives_edit_delete_restore_and_commit():
    catalog = Catalog()
    item = identity(collection_id=8)
    catalog.register_base(item, content("base"))
    changed = catalog.accept([Change(item, 1, content("edit"))])
    catalog.mark_applied(changed)
    catalog.mark_persisted(changed)
    deletion = catalog.accept([Change(item, 2, None)])
    catalog.mark_applied(deletion)
    assert catalog.entry(item.id).deleted
    restored = catalog.accept([Change(item, 3, content("edit"))])
    catalog.mark_applied(restored)
    catalog.mark_persisted(restored)
    entry = catalog.entry(item.id)
    assert entry.identity == item
    assert entry.base.content == content("base")
    assert (entry.accepted, entry.applied, entry.persisted) == (4, 4, 4)
    assert not entry.deleted
    assert entry.revisions[2].content is None  # Tombstones retain history.


def test_failed_batch_does_not_accept_any_changes_or_consume_ids():
    catalog = Catalog()
    first, second = identity(collection_id=7), identity(collection_id=9)
    catalog.register_base(first, content("a"))
    catalog.register_base(second, content("b"))
    new = identity()
    with pytest.raises(ApiError) as error:
        catalog.accept([Change(new, 0, content("new")),
                        Change(first, 1, content("changed")),
                        Change(second, 0, None)])
    assert error.value.payload["conflicts"][0]["id"] == second.id
    assert catalog.entry(first.id).accepted == 1
    assert len(catalog.entries()) == 2
    catalog.accept([Change(new, 0, content("new"))])
    assert catalog.entry(new.id).identity.collection_id == 10


def test_highest_id_and_last_collection_deletion_never_reuses_ids():
    catalog = Catalog()
    base = identity(collection_id=42)
    catalog.register_base(base, content("base"))
    removed = catalog.accept([Change(base, 1, None)])
    catalog.mark_applied(removed)
    catalog.mark_persisted(removed)
    new = identity()
    added = catalog.accept([Change(new, 0, content("addition"))])
    assert catalog.entry(new.id).identity.collection_id == 43
    catalog.mark_applied(added)
    catalog.mark_persisted(added)
    removed = catalog.accept([Change(catalog.entry(new.id).identity, 1, None)])
    catalog.mark_applied(removed)
    catalog.mark_persisted(removed)
    other = identity()
    catalog.accept([Change(other, 0, content("another"))])
    assert catalog.entry(other.id).identity.collection_id == 44
    assert len(catalog.entries()) == 3


@pytest.mark.parametrize("kind", ["pcl", "patch", "fiber"])
def test_older_acknowledgements_preserve_newer_accepted_content(kind):
    catalog = Catalog()
    item = identity(kind)
    submitted = catalog.accept([Change(item, 0, content("first"))])
    current = catalog.entry(item.id).identity
    newer = catalog.accept([Change(current, 1, content("second"))])
    catalog.mark_applied(submitted)
    catalog.mark_persisted(submitted)
    entry = catalog.entry(item.id)
    assert (entry.accepted, entry.applied, entry.persisted) == (2, 1, 1)
    assert entry.current.content == content("second")
    assert submitted[0].content == content("first")
    catalog.mark_applied(newer)
    catalog.mark_applied(submitted)  # Delayed duplicate response.
    assert catalog.entry(item.id).applied == 2


def test_commit_requires_each_exact_revision_to_have_applied():
    catalog = Catalog()
    item = identity("fiber")
    first = catalog.accept([Change(item, 0, content("one"))])
    second = catalog.accept([Change(item, 1, content("two"))])
    catalog.mark_applied(second)
    with pytest.raises(ApiError):
        catalog.mark_persisted(first)
    assert catalog.entry(item.id).persisted == 0
    catalog.mark_persisted(second)


def test_errors_are_scoped_to_stage_and_revision_and_replay_keeps_newer_errors():
    catalog = Catalog()
    item = identity("patch")
    first = catalog.accept([Change(item, 0, content("first"))])
    second = catalog.accept([Change(item, 1, content("second"))])
    catalog.record_error(second, "apply", "invalid geometry")
    catalog.mark_applied(first)
    catalog.record_error(first, "commit", "source changed")
    status = catalog.status()[0]
    assert status["accepted_revision"] == 2
    assert status["applied_revision"] == 1
    assert status["persisted_revision"] == 0
    assert len(status["errors"]) == 2
    catalog.mark_persisted(first)
    assert catalog.status()[0]["errors"] == [
        {"revision": 2, "stage": "apply", "message": "invalid geometry"}]


def test_role_participation_and_fit_generation_do_not_erase_catalog():
    catalog = Catalog()
    item = identity("patch")
    batch = catalog.accept([Change(item, 0, content("geometry"))])
    catalog.mark_applied(batch)
    catalog.reset_applied()
    assert catalog.entry(item.id).accepted == 1
    assert catalog.entry(item.id).applied == 0
    assert catalog.desired(lambda entry: False) == ()
    assert catalog.desired(lambda entry: True) == batch


def test_content_snapshots_do_not_alias_caller_objects():
    original = {"points": [[1, 2, 3]]}
    reference = Content.from_json(original)
    original["points"][0][0] = 900
    exposed = reference.json()
    exposed["points"][0][1] = 800
    assert reference.json() == {"points": [[1, 2, 3]]}


def test_parallel_duplicate_commands_execute_once_and_reject_different_payload():
    coordinator = MutationCoordinator()
    started, finish = threading.Event(), threading.Event()
    calls = []

    def operation(payload):
        calls.append(payload)
        started.set()
        assert finish.wait(5)
        return {"revision": 1}

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(coordinator.execute, "one", "apply", {"v": 1}, operation)
        assert started.wait(5)
        assert coordinator.outcome("one")["state"] == "running"
        second = pool.submit(coordinator.execute, "one", "apply", {"v": 1}, operation)
        with pytest.raises(ApiError):
            coordinator.execute("one", "apply", {"v": 2}, operation)
        finish.set()
        assert first.result(5) == second.result(5) == {"revision": 1}
    assert calls == [{"v": 1}]
    for n in range(300):
        coordinator.execute(str(n), "apply", {}, lambda payload: {})
    assert coordinator.execute("one", "apply", {"v": 1}, operation) == {"revision": 1}
    assert len(calls) == 1  # No LRU expiration while the workspace lives.


def test_validation_failure_is_a_retained_command_outcome():
    coordinator = MutationCoordinator()
    calls = []

    def reject(payload):
        calls.append(payload)
        raise ApiError(409, "changed target", payload={"id": "input"})

    for _ in range(2):
        with pytest.raises(ApiError, match="changed target"):
            coordinator.execute("one", "apply", {}, reject)
    assert len(calls) == 1
    assert coordinator.outcome("one")["state"] == "rejected"


def test_failed_commit_blocks_later_mutations_until_same_command_recovers():
    coordinator = MutationCoordinator()
    calls = []

    def publish(payload):
        calls.append(payload)
        if len(calls) == 1:
            raise OSError("publication interrupted")
        return {"persisted": 2}

    with pytest.raises(OSError):
        coordinator.execute("commit", "commit", {}, publish, recoverable=True)
    with pytest.raises(ApiError, match="recovery"):
        coordinator.execute("apply", "apply", {}, lambda payload: {})
    assert coordinator.outcome("commit")["state"] == "recovery_required"
    assert coordinator.execute("commit", "commit", {}, publish, recoverable=True) == {"persisted": 2}
    assert coordinator.execute("apply", "apply", {}, lambda payload: {}) == {}


def test_lease_excludes_other_clients_and_services_and_retains_reconnect(tmp_path):
    first, second = WorkspaceLease(tmp_path), WorkspaceLease(tmp_path)
    try:
        first.claim("client-a")
        first.claim("client-a")  # Reconnect with the same token.
        with pytest.raises(ApiError):
            first.claim("client-b")
        with pytest.raises(ApiError):
            second.claim("client-b")
        with pytest.raises(ApiError):
            first.require("observer")
        first.release("client-a")
        second.claim("client-b")
        second.require("client-b")
    finally:
        first.close()
        second.close()


def test_external_discovery_is_atomic_and_preserves_reserved_targets():
    catalog = Catalog()
    local = identity()
    catalog.accept([Change(local, 0, content('local'))])
    before = catalog.entries()
    external = identity(collection_id=100)
    collision = identity(collection_id=catalog.entry(local.id).identity.collection_id)
    with pytest.raises(ValueError):
        catalog.register_external_bases([(external, content('external')),
                                         (collision, content('collision'))])
    assert catalog.entries() == before
    next_local = identity()
    catalog.accept([Change(next_local, 0, content('next'))])
    assert catalog.entry(next_local.id).identity.collection_id == 1
    revisions = catalog.register_external_bases([(external, content('external'))])
    imported = catalog.entry(external.id)
    assert (imported.accepted, imported.applied, imported.persisted) == (1, 0, 1)
    assert not imported.applied_history
    catalog.mark_applied(revisions)
    assert catalog.entry(external.id).applied == 1
    following = identity()
    catalog.accept([Change(following, 0, content('following'))])
    assert catalog.entry(following.id).identity.collection_id == 101
