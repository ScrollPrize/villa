"""Lazy link refresh: the store's link marker lets a refresh skip the
detectors when the links are current, and an input update link only the
objects added since (LAZY_RELINK_PLAN.md)."""
import pytest

import vc_object_store as vos
from vc_object_store import ObjectStore

import object_store_inputs as osi
from object_store_fixture import put_patch, put_pcl
from test_object_store_positions import (  # noqa: F401 - fixtures
    _edit, _grid, _lease_and, _names, _on, _shared_edit, cpu, fit_config, hub, shared)


@pytest.fixture()
def detectors(monkeypatch):
    """Each refresh_links call's ``new`` (None for a full run)."""
    calls = []
    original = osi.refresh_links

    def recording(*args, **kwargs):
        calls.append(kwargs.get('new'))
        return original(*args, **kwargs)
    monkeypatch.setattr(osi, 'refresh_links', recording)
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
        return db.link_marker()


def edge_uuids(path, kind='point-surface'):
    with ObjectStore.open(path, read_only=True) as db:
        return {edge['uuid'] for edge in db.edges_by_kind(kind)}


def test_unchanged_inputs_skip_the_detectors(shared, detectors):
    first, log = run(shared)
    assert plan(log) == 'full' and detectors == [None]
    recorded = marker(shared)
    assert recorded['exact'] and recorded['position'] == first.position.sequence
    again, log = run(shared)
    assert plan(log) == 'skip' and len(detectors) == 1
    assert again.position == first.position and again.head == first.position.sequence
    assert marker(shared) == recorded  # a skip records nothing


def test_a_write_outside_the_selection_still_skips(shared, detectors):
    names = _names(shared)
    config = fit_config(shared, object_store_patch_uuids=[names['near']])
    run(shared, config)
    _shared_edit(shared, lambda db: db.set_meta(names['far'], 'review_note', 'unselected'))
    prepared, log = run(shared, config)
    assert plan(log) == 'skip' and len(detectors) == 1
    assert prepared.position.sequence == prepared.head


def _edit_walk(db):
    db.set_meta(_names_in(db)['walk'], 'review_note', 'edited')


def _names_in(db):
    return {record['name']: record['uuid'] for kind in osi.KINDS
            for record in db.objects_by_kind(kind)}


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
    with db.write({far: db.current_object_revision(far)['uuid']}) as write:
        write.retire_object(far)


@pytest.mark.parametrize('change', [_edit_walk, _approve_link, _remove_link, _retire_far])
def test_store_changes_force_a_full_refresh(shared, detectors, change):
    run(shared)
    _shared_edit(shared, change)
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


def _edit_near(db):
    db.set_meta(_names_in(db)['near'], 'review_note', 'edited')


def test_an_input_update_links_only_new_objects(shared, detectors):
    run(shared)
    before = edge_uuids(shared)

    def add(db):
        put_patch(db, _grid(260.0), (0.25, 0.25), name='lone')
        put_pcl(db, _on(100.0, (20.0, 60.0)), name='extra')
    _shared_edit(shared, add)
    names = _names(shared)
    prepared, log = run(shared, incremental=True)
    assert plan(log) == 'incremental'
    assert detectors[-1] == {'surface': [names['lone']], 'point_collection': [names['extra']]}
    assert any('pcl_patch: 1 created' in line and 'incremental over 2 new' in line
               for line in prepared.lines)
    assert any('patch_overlap: 0 created' in line for line in prepared.lines)
    after = edge_uuids(shared)
    assert before < after  # nothing removed; the new collection is linked
    with ObjectStore.open(shared, read_only=True) as db:
        [added] = [db.edge(uuid) for uuid in after - before]
    assert {added['a'], added['b']} == {names['extra'], names['near']}
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
        put_pcl(db, _on(100.0, (20.0, 60.0)), name='extra')
        _edit_near(db)
    _shared_edit(shared, add_and_edit)
    _, log = run(shared, incremental=True)
    assert plan(log) == 'full' and detectors[-1] is None


def test_a_hub_replica_skips_and_records_a_marker_only_once_published(hub, detectors, monkeypatch):
    def unreachable(store, head):
        raise RuntimeError('hub unreachable')
    with monkeypatch.context() as patched:
        patched.setattr(osi, '_conditional_push', unreachable)
        with pytest.raises(osi.PositionError):
            run(hub['service'])
    assert marker(hub['service']) is None
    first, log = run(hub['service'])
    assert plan(log) == 'full'
    assert marker(hub['service'])['position'] == first.position.sequence
    again, log = run(hub['service'])
    assert plan(log) == 'skip' and len(detectors) == 2
    assert again.position == first.position
    # An editor's new PCL is linked incrementally and published.
    _edit(hub['editor'], 'editor', lambda db: put_pcl(db, _on(100.0, (20.0, 60.0)), name='extra'))
    updated, log = run(hub['service'], incremental=True)
    assert plan(log) == 'incremental'
    assert detectors[-1] == {'point_collection': [_names(hub['editor'])['extra']]}
    assert updated.position.sequence > again.position.sequence
    recorded = marker(hub['service'])
    assert recorded['position'] == updated.position.sequence and not recorded['exact']
    # A replica's marker is its own; the editor's replica has none.
    assert marker(hub['editor']) is None


def test_patch_overlaps_off_skip_their_detector_and_turning_them_on_relinks(shared, detectors):
    prepared, log = run(shared, fit_config(shared, input_use_patch_overlaps=False))
    assert 'link refresh patch_overlap: skipped (input_use_patch_overlaps is off)' in prepared.lines
    _, log = run(shared, fit_config(shared, input_use_patch_overlaps=False))
    assert plan(log) == 'skip'
    prepared, log = run(shared)
    assert plan(log) == 'full'
    assert any(line.startswith('link refresh patch_overlap: 0 created') for line in prepared.lines)


def test_a_full_refresh_names_the_changed_link_input(shared, detectors):
    run(shared)
    prepared, log = run(shared, fit_config(shared, pcl_link_distance_tolerance=3.0))
    [line] = [line for line in log if line.startswith('link refresh: full: ')]
    assert 'link_refresh.parameters.pcl_link_distance_tolerance: ' in line and '-> 3.0' in line
    # The report ends with the decision too.
    assert prepared.lines[-2].startswith('link refresh: full refresh (the link inputs changed: ')
    _, log = run(shared, fit_config(shared, pcl_link_distance_tolerance=3.0,
                                    input_use_patch_overlaps=False))
    assert any('patch_overlaps: True -> False' in line for line in log)


def test_a_marker_without_key_details_still_forces_a_full_refresh(shared, detectors):
    run(shared)
    with ObjectStore.open(shared, author='service') as db:
        old = db.link_marker()
        db.set_link_marker({key: value for key, value in old.items() if key != 'key_parts'}
                           | {'key': 'older'})
    _, log = run(shared)
    assert 'link refresh: full: the link inputs changed (the marker records no details)' in log


def test_a_skip_reports_its_reason_with_the_refresh_lines(shared):
    run(shared)
    prepared, _ = run(shared)
    assert prepared.lines[0].startswith('link refresh: skipped every detector (unchanged since position ')


def test_the_store_reads_are_progress_stages(shared):
    from spiral_progress import ProgressReporter
    stages = []

    def publish(snapshot):
        if snapshot is not None and snapshot['stage_name'] not in stages:
            stages.append(snapshot['stage_name'])
    progress = ProgressReporter(publish)
    prepared, _ = run(shared, progress=progress)
    position = prepared.position.sequence
    assert stages[0] == 'Reading the selected inputs'
    assert stages[-1] == f'Reading the inputs at position {position}'
    assert 'link refresh pcl_patch' in stages
    assert progress.snapshot() is None  # every stage cleared
