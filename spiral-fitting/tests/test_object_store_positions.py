"""Positions of object-store fit inputs: link-then-pin (prepare_position) on
a store an editor writes as well, pinned reads whatever the store's head,
and link staleness."""
import dataclasses

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
from object_store_fixture import fit_config, head, names, shared_edit  # noqa: E402

pytestmark = pytest.mark.usefixtures('cpu')


def prepare(path, config=None, **options):
    config = config or fit_config(path, 'service')
    with ObjectStore.open(path, author='service') as opened:
        return osi.prepare_position(opened, config, osi.request_from_config(config),
                                    log=lambda line: None, **options)


def read(path, request):
    with ObjectStore.open(path, read_only=True) as db:
        return osi.read_snapshot(db, request)


def pinned(path, pins):
    return dataclasses.replace(osi.request_from_config(fit_config(path, 'service')), pins=pins)


def position_of(path, commit_sequence):
    with ObjectStore.open(path, read_only=True) as db:
        return dataclasses.replace(osi.InputPosition.of_status(db.status()),
                                   commit_sequence=commit_sequence)


def test_prepare_links_commits_and_pins_a_position(shared):
    walk = names(shared)['walk']
    edited = shared_edit(shared, lambda db: db.set_meta(walk, 'sampling_group', 'reviewed'))
    prepared = prepare(shared, minimum=position_of(shared, edited))
    assert prepared.attempts == 1 and prepared.head == edited
    assert prepared.position.commit_sequence > edited  # the links were committed on top
    assert any('links committed at commit' in line for line in prepared.lines)
    assert prepared.position.commit_sequence == head(shared)
    # The pins hold every selected object at the version the links left it.
    assert set(prepared.pins.objects) == set(names(shared).values())
    # Later commits do not change N: every reader reads the same inputs at it.
    request = pinned(shared, prepared.pins)
    before = read(shared, request)
    assert len(before.relationships) == 3
    assert before.manifest()['position'] == prepared.position.provenance()
    shared_edit(shared, lambda db: db.set_meta(walk, 'sampling_group', 'later'))
    with ObjectStore.open(shared, read_only=True) as db:
        assert db.object(walk)['meta']['sampling_group'] == 'later'
    after = read(shared, request)
    assert after.digest() == before.digest()
    assert osi._compared(after.manifest())['objects'] == osi._compared(before.manifest())['objects']
    [entry] = [e for e in before.manifest()['objects'] if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'reviewed'
    # A position alone (no versions) reads the same state from the history.
    by_position = read(shared, pinned(shared, osi.InputPins(prepared.position)))
    assert by_position.digest() == before.digest()
    assert by_position.request.pins.objects == prepared.pins.objects
    # The edit of a selected object relinks: the marker's commit follows the
    # head. Then nothing is new: the links are current, N = H.
    again = prepare(shared)
    assert again.position.commit_sequence == again.head + 1
    third = prepare(shared)
    assert third.position == again.position and third.head == again.position.commit_sequence


def _racing(monkeypatch, edit, every=False):
    """Run ``edit`` once the first (or every) refresh has read its snapshot
    and planned its detectors, before its links commit."""
    original = osi.link_plan
    edits = []

    def racing(*args, **kwargs):
        planned = original(*args, **kwargs)
        if every or not edits:
            edits.append(edit(len(edits)))
        return planned
    monkeypatch.setattr(osi, 'link_plan', racing)
    return edits


def test_a_commit_while_linking_relinks(shared, monkeypatch):
    walk = names(shared)['walk']
    edits = _racing(monkeypatch, lambda n: shared_edit(
        shared, lambda db: db.set_meta(walk, 'review_note', 'saved')))
    prepared = prepare(shared)
    assert prepared.attempts == 2 and prepared.head >= edits[0]
    assert any('relinking' in line for line in prepared.lines)
    # Between the head and N only the service wrote.
    with ObjectStore.open(shared, read_only=True) as db:
        status = db.status()
        page = db.changes_since({'store': status['store'], 'generation': status['generation'],
                                 'sequence': prepared.head})
        assert {commit['author'] for commit in page['commits']
                if commit['sequence'] <= prepared.position.commit_sequence} <= {'service'}
    snapshot = read(shared, pinned(shared, prepared.pins))
    assert snapshot.objects[walk].meta['review_note'] == 'saved'


def test_continuous_edits_fail_the_update_cleanly(shared, monkeypatch):
    walk = names(shared)['walk']
    _racing(monkeypatch, lambda n: shared_edit(
        shared, lambda db: db.set_meta(walk, 'review_note', str(n))), every=True)
    with pytest.raises(osi.PositionError) as error:
        prepare(shared, attempts=3)
    text = '\n'.join(error.value.problems)
    assert 'inputs kept changing' in text
    # Nothing the service linked was written: a conflict commits nothing.
    with ObjectStore.open(shared, read_only=True) as db:
        assert db.edges_by_kind('point-surface') == []
        assert osi.read_link_marker(db) is None
    # Once the edits stop, the next update links afresh.
    monkeypatch.undo()
    prepared = prepare(shared)
    assert len(read(shared, pinned(shared, prepared.pins)).relationships) == 3


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
    config = fit_config(shared, 'service', object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    request = osi.request_from_config(config)
    with ObjectStore.open(shared, author='service') as db:
        prepared = osi.prepare_position(db, config, request, log=lambda line: None)
    assert prepared.selected['surface'] == [far] and prepared.head >= tagged
    shared_edit(shared, lambda db: db.upsert_object({**db.object(far), 'tags': []}))
    with ObjectStore.open(shared, read_only=True) as db:
        assert osi.resolve_selection(
            db, dataclasses.replace(request, pins=prepared.pins))['surface'] == [far]
        assert osi.resolve_selection(
            db, dataclasses.replace(request, pins=osi.InputPins(prepared.position)))['surface'] == [far]
        assert osi.resolve_selection(db, request)['surface'] == []


def test_changed_link_parameters_or_normals_mark_links_stale(shared):
    config = fit_config(shared, 'service')
    inputs = osi.load_fit_inputs(config, log=lambda line: None)
    none = osi.fiber_normal_source(config)
    assert osi.stale_link_reasons(inputs.manifest, config, none) == []
    changed = fit_config(shared, 'service', pcl_link_distance_tolerance=
                         config['pcl_link_distance_tolerance'] * 2)
    [reason] = osi.stale_link_reasons(inputs.manifest, changed, none)
    assert reason.startswith('pcl_link_distance_tolerance changed')
    lasagna = {'pool': '/cache/lasagna/run-1/normals_g4', 'scale': 4}
    assert osi.stale_link_reasons(inputs.manifest, config, osi.fiber_normal_source(
        config, lasagna)) == ['the fiber-patch normal source changed']
