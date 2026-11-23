"""Positions of object-store fit inputs: pinning the head (prepare_position)
of a store an editor writes as well, and pinned reads whatever the store's
head."""
import dataclasses

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
from object_store_fixture import fit_config, head, names, shared_edit  # noqa: E402

pytestmark = pytest.mark.usefixtures('cpu')


def prepare(path, config=None, **options):
    config = config or fit_config(path)
    with ObjectStore.open(path, read_only=True) as opened:
        return osi.prepare_position(opened, osi.request_from_config(config), **options)


def read(path, request):
    with ObjectStore.open(path, read_only=True) as db:
        return osi.read_snapshot(db, request)


def pinned(path, pins):
    return dataclasses.replace(osi.request_from_config(fit_config(path)), pins=pins)


def position_of(path, commit_sequence):
    with ObjectStore.open(path, read_only=True) as db:
        return dataclasses.replace(osi.InputPosition.of_status(db.status()),
                                   commit_sequence=commit_sequence)


def test_prepare_pins_the_head(shared):
    walk = names(shared)['walk']
    edited = shared_edit(shared, lambda db: db.set_meta(
        walk, 'source', {'spiral_patch_erode_cells': 2}))
    prepared = prepare(shared, minimum=position_of(shared, edited))
    assert prepared.head == edited == prepared.position.commit_sequence == head(shared)
    # The pins hold every selected object at its version there.
    assert set(prepared.pins.references) == set(names(shared).values())
    # Later commits do not change N: every reader reads the same inputs at it.
    request = pinned(shared, prepared.pins)
    before = read(shared, request)
    assert before.manifest()['position'] == prepared.position.provenance()
    shared_edit(shared, lambda db: db.set_meta(walk, 'source', {'spiral_patch_erode_cells': 3}))
    with ObjectStore.open(shared, read_only=True) as db:
        assert db.object(walk)['meta']['source'] == {'spiral_patch_erode_cells': 3}
    after = read(shared, request)
    assert after.digest() == before.digest()
    assert osi._compared(after.manifest())['objects'] == osi._compared(before.manifest())['objects']
    [entry] = [e for e in before.manifest()['objects'] if e['uuid'] == walk]
    assert entry['metadata']['spiral_patch_erode_cells'] == 2
    # A position alone (no versions) reads the same state from the history.
    by_position = read(shared, pinned(shared, osi.InputPins(prepared.position)))
    assert by_position.digest() == before.digest()
    assert by_position.request.pins.references == prepared.pins.references
    # The next preparation pins the new head.
    again = prepare(shared)
    assert again.position.commit_sequence == again.head == head(shared)


def test_a_commit_while_reading_the_selection_rereads_it(shared, monkeypatch):
    """The selection, the shell and the position are one capture: a commit
    between its queries does not label versions of commit N+1 as N."""
    walk = names(shared)['walk']
    original = osi.resolve_selector
    edits = []

    def racing(store, kind, selector, records=None):
        if not edits:  # after the head is read, before the objects are
            edits.append(shared_edit(shared, lambda db: db.set_meta(walk, 'review_note', 'mid-read')))
        return original(store, kind, selector, records)
    monkeypatch.setattr(osi, 'resolve_selector', racing)
    request = osi.request_from_config(fit_config(shared))
    with ObjectStore.open(shared, read_only=True) as db:
        selected = osi.resolve_selection(db, request)
        assert edits and selected.position.commit_sequence == edits[0] == head(shared)
        assert selected.references[walk]['version'] == osi.current_record(db, walk).version
        # The position alone reads the same versions from the history.
        by_position = osi.resolve_selection(
            db, dataclasses.replace(request, pins=osi.InputPins(selected.position)))
        assert by_position.references == selected.references
    # The pins are that capture, nothing later.
    prepared = prepare(shared)
    assert prepared.position.commit_sequence == head(shared)
    assert prepared.pins.references[walk] == selected.references[walk]


def test_positions_ahead_of_the_head_or_of_another_store_are_refused(shared):
    ahead = position_of(shared, head(shared) + 100)
    with pytest.raises(osi.ObjectStoreInputError, match='after this store'):
        prepare(shared, minimum=ahead)
    prepared = prepare(shared)
    other = dataclasses.replace(prepared.position, store='another-store')
    with pytest.raises(osi.ObjectStoreInputError, match='belongs to store another-store'):
        read(shared, pinned(shared, osi.InputPins(other)))
    generation = dataclasses.replace(prepared.position, generation='another-generation')
    with pytest.raises(osi.ObjectStoreInputError, match='generation another-generation'):
        read(shared, pinned(shared, osi.InputPins(generation)))


def test_selectors_resolve_at_the_pins(shared):
    far = names(shared)['far']
    tagged = shared_edit(shared, lambda db: db.upsert_object({**db.object(far), 'tags': ['reviewed']}))
    config = fit_config(shared, object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    request = osi.request_from_config(config)
    with ObjectStore.open(shared, read_only=True) as db:
        prepared = osi.prepare_position(db, request)
    assert prepared.selected['surface'] == [far] and prepared.head >= tagged
    shared_edit(shared, lambda db: db.upsert_object({**db.object(far), 'tags': []}))
    with ObjectStore.open(shared, read_only=True) as db:
        assert osi.resolve_selection(
            db, dataclasses.replace(request, pins=prepared.pins))['surface'] == [far]
        assert osi.resolve_selection(
            db, dataclasses.replace(request, pins=osi.InputPins(prepared.position)))['surface'] == [far]
        assert osi.resolve_selection(db, request)['surface'] == []



# ---------------------------------------------------------------------------
# Reads at pins use the numbered history of the pinned versions alone; a
# position without versions reads the current state when it is the head,
# else the versions live at the position's commit.
# ---------------------------------------------------------------------------

class Recording:
    """A store whose reads of the state at a past commit (versions_at) are
    counted; ``after_listing`` runs after each current-state listing (to
    write while a read is in flight)."""

    def __init__(self, store, after_listing=None):
        self._store, self._after_listing = store, after_listing
        self.history_reads = 0

    def versions_at(self, *args, **kwargs):
        self.history_reads += 1
        return self._store.versions_at(*args, **kwargs)

    def select_objects(self, *args, **kwargs):
        rows = self._store.select_objects(*args, **kwargs)
        if self._after_listing is not None:
            self._after_listing()
        return rows

    def __getattr__(self, name):
        return getattr(self._store, name)


@pytest.mark.parametrize('during_read', [False, True])
def test_a_position_overtaken_by_a_write_is_read_from_history(shared, during_read):
    # Written before the read, or while it lists the current state (which
    # then falls back to history).
    prepared = prepare(shared)
    request = pinned(shared, osi.InputPins(prepared.position))
    near = names(shared)['near']
    writes = []

    def write():
        if not writes:
            writes.append(shared_edit(shared, lambda db: db.set_meta(near, 'review_note', 'later')))
    if not during_read:
        write()
    with ObjectStore.open(shared, read_only=True) as db:
        store = Recording(db, after_listing=write if during_read else None)
        snapshot = osi.read_snapshot(store, request)
        assert writes and store.history_reads > 0
    assert 'review_note' not in snapshot.objects[near].meta
    assert snapshot.request.pins.references == prepared.pins.references
