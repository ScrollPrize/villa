"""Lazy link refresh: the store's link marker lets a refresh skip the
detectors when the links are current, and an input update link only the
objects added since. Reads at pins never touch the history; a position
alone reads the current state when it is the store's head, else the
history."""
import dataclasses

import pytest

pytest.importorskip('vc_object_store')
import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
from object_store_fixture import (  # noqa: E402
    fit_config as _fit_config, names, on_plane, plane, put_patch, put_pcl, shared_edit)

pytestmark = pytest.mark.usefixtures('cpu')


def fit_config(path, **overrides):
    return _fit_config(path, 'service', **overrides)


@pytest.fixture()
def detectors(monkeypatch):
    """Each link plan's ``new`` (None for a full run)."""
    calls = []
    original = osi.link_plan

    def recording(*args, **kwargs):
        calls.append(kwargs.get('new'))
        return original(*args, **kwargs)
    monkeypatch.setattr(osi, 'link_plan', recording)
    return calls


def run(path, config=None, **options):
    """prepare_position; returns (prepared, log)."""
    config = config or fit_config(path)
    log = []
    with ObjectStore.open(path, author='service') as store:
        prepared = osi.prepare_position(store, config, osi.request_from_config(config),
                                        log=log.append, **options)
    return prepared, log


def plan(log):
    [line] = [line for line in log if line.startswith('link refresh: ')
              and line.split(': ')[1] in ('skip', 'full', 'incremental')]
    return line.split(': ')[1]


def marker(path):
    with ObjectStore.open(path, read_only=True) as db:
        return osi.read_link_marker(db)


def edge_uuids(path, kind='point-surface'):
    with ObjectStore.open(path, read_only=True) as db:
        return {edge['uuid'] for edge in db.edges_by_kind(kind)}


def test_unchanged_inputs_skip_the_detectors(shared, detectors):
    first, log = run(shared)
    assert plan(log) == 'full' and detectors == [None]
    recorded = marker(shared)
    assert recorded['exact'] and recorded['commit_sequence'] == first.position.commit_sequence
    again, log = run(shared)
    assert plan(log) == 'skip' and len(detectors) == 1
    assert again.position == first.position and again.head == first.position.commit_sequence
    assert marker(shared) == recorded  # a skip records nothing


def test_a_write_outside_the_selection_still_skips(shared, detectors):
    uuids = names(shared)
    config = fit_config(shared, object_store_patch_uuids=[uuids['near']])
    run(shared, config)
    shared_edit(shared, lambda db: db.set_meta(uuids['far'], 'review_note', 'unselected'))
    prepared, log = run(shared, config)
    assert plan(log) == 'skip' and len(detectors) == 1
    assert prepared.position.commit_sequence == prepared.head


def _names_in(db):
    return {record['name']: record['uuid'] for kind in osi.KINDS
            for record in db.objects_by_kind(kind)}


def _edit_walk(db):
    db.set_meta(_names_in(db)['walk'], 'review_note', 'edited')


def _approve_link(db):
    db.set_approval(db.edges_by_kind('point-surface')[0]['uuid'], 'approved')


def _remove_link(db):
    db.remove_edge(db.edges_by_kind('point-surface')[0]['uuid'])


def _retire_far(db):
    # A selected object leaving the store: its links go first (the change the
    # marker sees), then the object.
    far = _names_in(db)['far']
    for edge in db.edges_of(far):
        db.remove_edge(edge['uuid'])
    with db.write(expected_versions=db.select_expected_versions({far: db.current_version(far)})) as write:
        write.retire_object(far)


@pytest.mark.parametrize('change', [_edit_walk, _approve_link, _remove_link, _retire_far])
def test_store_changes_force_a_full_refresh(shared, detectors, change):
    run(shared)
    shared_edit(shared, change)
    _, log = run(shared)
    assert plan(log) == 'full' and detectors == [None, None]


@pytest.mark.parametrize('overrides', [
    {'pcl_link_distance_tolerance': 3.0},
    {'object_store_full_link_refresh': True},
])
def test_configuration_forces_a_full_refresh(shared, detectors, overrides):
    run(shared)
    _, log = run(shared, fit_config(shared, **overrides))
    assert plan(log) == 'full' and detectors == [None, None]


def test_a_detector_version_bump_forces_a_full_refresh(shared, detectors, monkeypatch):
    run(shared)
    bumped = {**vos.detector_versions(), 'point-collection-surface': 99}
    monkeypatch.setattr(vos, 'detector_versions', lambda: bumped)
    _, log = run(shared)
    assert plan(log) == 'full' and detectors == [None, None]


def test_an_input_update_links_only_new_objects(shared, detectors):
    run(shared)
    before = edge_uuids(shared)

    def add(db):
        put_patch(db, plane(260.0), (0.25, 0.25), name='lone')
        put_pcl(db, on_plane(100.0, (20.0, 60.0)), name='extra')
    shared_edit(shared, add)
    uuids = names(shared)
    prepared, log = run(shared, incremental=True)
    assert plan(log) == 'incremental'
    assert detectors[-1] == {'surface': [uuids['lone']], 'point_collection': [uuids['extra']]}
    assert any('pcl_patch: 1 created' in line and 'incremental over 2 new' in line
               for line in prepared.lines)
    assert any('patch_overlap: 0 created' in line for line in prepared.lines)
    after = edge_uuids(shared)
    assert before < after  # nothing removed; the new collection is linked
    with ObjectStore.open(shared, read_only=True) as db:
        [added] = [db.edge(uuid) for uuid in after - before]
    assert {added['a'], added['b']} == {uuids['extra'], uuids['near']}
    assert not marker(shared)['exact']
    # Nothing new since: an update skips; a startup after an incremental
    # refresh runs in full, and agrees with it.
    _, log = run(shared, incremental=True)
    assert plan(log) == 'skip'
    prepared, log = run(shared)
    assert plan(log) == 'full'
    assert any(line.startswith('link refresh pcl_patch: 0 created, 0 updated, 4 unchanged, '
                               '0 stale removed') for line in prepared.lines), prepared.lines
    assert marker(shared)['exact']


def test_an_update_after_an_edit_of_an_existing_object_runs_in_full(shared, detectors):
    run(shared)

    def add_and_edit(db):
        put_pcl(db, on_plane(100.0, (20.0, 60.0)), name='extra')
        db.set_meta(_names_in(db)['near'], 'review_note', 'edited')
    shared_edit(shared, add_and_edit)
    _, log = run(shared, incremental=True)
    assert plan(log) == 'full' and detectors[-1] is None


def test_patch_overlaps_off_skip_their_detector_and_turning_them_on_relinks(shared, detectors):
    prepared, log = run(shared, fit_config(shared, input_use_patch_overlaps=False))
    assert 'link refresh patch_overlap: skipped (input_use_patch_overlaps is off)' in prepared.lines
    _, log = run(shared, fit_config(shared, input_use_patch_overlaps=False))
    assert plan(log) == 'skip'
    prepared, log = run(shared)
    assert plan(log) == 'full'
    assert any(line.startswith('link refresh patch_overlap: 0 created') for line in prepared.lines)


def test_the_store_reads_are_progress_stages(shared):
    from spiral_progress import ProgressReporter
    stages = []

    def publish(snapshot):
        if snapshot is not None and snapshot['stage_name'] not in stages:
            stages.append(snapshot['stage_name'])
    progress = ProgressReporter(publish)
    prepared, _ = run(shared, progress=progress)
    position = prepared.position.commit_sequence
    assert stages[0] == 'Reading the selected inputs'
    assert stages[-1] == f'Reading the inputs at commit {position}'
    assert 'link refresh pcl_patch' in stages
    assert progress.snapshot() is None  # every stage cleared


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


def _pinned(path):
    prepared, _ = run(path)
    return prepared, dataclasses.replace(osi.request_from_config(fit_config(path)),
                                         pins=prepared.pins)


def test_a_read_at_the_pins_matches_the_history_read(shared):
    def comparable(snapshot):
        return (snapshot.volume_uuid, snapshot.objects, snapshot.relationships,
                snapshot.exclusions, snapshot.outer_shell, snapshot.notes)
    prepared, request = _pinned(shared)
    near = names(shared)['near']
    shared_edit(shared, lambda db: db.set_meta(near, 'review_note', 'later'))
    with ObjectStore.open(shared, read_only=True) as db:
        store = Recording(db)
        pinned = osi.read_snapshot(store, request)
        assert store.history_reads == 0
        assert osi.resolve_selection(store, request).uuids == prepared.selected.uuids
        by_position = osi.read_snapshot(
            store, dataclasses.replace(request, pins=osi.InputPins(prepared.position)))
        assert store.history_reads > 0
    assert pinned.relationships  # the links the refresh wrote are read
    assert comparable(pinned) == comparable(by_position)
    assert 'review_note' not in pinned.objects[near].meta


@pytest.mark.parametrize('during_read', [False, True])
def test_a_position_overtaken_by_a_write_is_read_from_history(shared, during_read):
    # Written before the read, or while it lists the current state (which
    # then falls back to history).
    prepared, _ = _pinned(shared)
    request = dataclasses.replace(osi.request_from_config(fit_config(shared)),
                                  pins=osi.InputPins(prepared.position))
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
