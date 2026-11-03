"""Reads at a plain store's head use its current state, not its history.

A service pins the head it just linked and reads its inputs there; rebuilding
that state from history scans every revision in the store. These tests check
the current-state read agrees with the history read, and that the history
read still answers a position older than the head, including one overtaken
while it was being read."""
import dataclasses

import pytest

from vc_object_store import ObjectStore

import object_store_inputs as osi
from test_object_store_lazy_relink import run
from test_object_store_positions import (  # noqa: F401 - fixtures
    _names, _shared_edit, cpu, fit_config, shared)


class Recording:
    """A store whose history reads are counted; ``after_current`` runs after
    each current-state view (to write while a read is in flight)."""

    def __init__(self, store, after_current=None):
        self._store = store
        self.history_reads = 0
        self._after_current = after_current

    def hub_read_view_at(self, *args, **kwargs):
        self.history_reads += 1
        return self._store.hub_read_view_at(*args, **kwargs)

    def revision_read_view(self, *args, **kwargs):
        view = self._store.revision_read_view(*args, **kwargs)
        if self._after_current is not None:
            self._after_current()
        return view

    def __getattr__(self, name):
        return getattr(self._store, name)


def pinned(path):
    prepared, _ = run(path)
    config = fit_config(path)
    request = dataclasses.replace(osi.request_from_config(config), position=prepared.position)
    return prepared, request


def comparable(snapshot):
    return (snapshot.volume_uuid, snapshot.objects, snapshot.relationships,
            snapshot.exclusions, snapshot.outer_shell, snapshot.notes)


def test_a_read_at_the_head_matches_the_history_read(shared, monkeypatch):
    prepared, request = pinned(shared)
    with ObjectStore.open(shared, read_only=True) as db:
        assert db.store_positions()['position'] == prepared.position.sequence
        store = Recording(db)
        uuids = osi.resolve_selection(store, request)
        current = osi.read_snapshot(store, request, uuids)
        assert store.history_reads == 0
        assert uuids == prepared.uuids
        with monkeypatch.context() as patched:
            patched.setattr(osi, '_at_head', lambda store, position: False)
            assert osi.resolve_selection(store, request) == uuids
            historical = osi.read_snapshot(store, request, uuids)
        assert store.history_reads > 0
    assert current.relationships  # the links the refresh wrote are read
    assert comparable(current) == comparable(historical)


def test_a_position_before_the_head_is_read_from_history(shared):
    prepared, request = pinned(shared)
    near = _names(shared)['near']
    _shared_edit(shared, lambda db: db.set_meta(near, 'review_note', 'later'))
    with ObjectStore.open(shared, read_only=True) as db:
        store = Recording(db)
        snapshot = osi.read_snapshot(store, request)
        assert store.history_reads > 0
    assert 'review_note' not in snapshot.objects[near].meta


def test_a_write_during_a_current_read_falls_back_to_history(shared):
    prepared, request = pinned(shared)
    near = _names(shared)['near']
    with ObjectStore.open(shared, read_only=True) as db:
        writes = []

        def write_once():
            if not writes:
                writes.append(_shared_edit(
                    shared, lambda db: db.set_meta(near, 'review_note', 'meanwhile')))
        store = Recording(db, after_current=write_once)
        snapshot = osi.read_snapshot(store, request, prepared.uuids)
        assert writes and store.history_reads == 1
    assert 'review_note' not in snapshot.objects[near].meta
