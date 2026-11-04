"""Positions of object-store fit inputs: link-then-pin (prepare_position) on a
hub replica or a plain store shared in place, pinned reads on any replica,
and link staleness."""
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
from object_store_fixture import (  # noqa: E402
    fit_config, hub_edit, leased, names, shared_edit)

pytestmark = pytest.mark.usefixtures('cpu')


def prepare(path, config=None, store=None, **options):
    config = config or fit_config(path, 'service')
    with ObjectStore.open(path, author='service') as opened:
        return osi.prepare_position(store(opened) if store else opened, config,
                                    osi.request_from_config(config),
                                    log=lambda line: None, **options)


def read(path, request):
    with ObjectStore.open(path, read_only=True) as db:
        return osi.read_snapshot(db, request)


def pinned(path, position):
    return dataclasses.replace(osi.request_from_config(fit_config(path, 'service')),
                               position=position)


@pytest.fixture(params=['hub', 'shared'])
def store(request):
    """The service's store and how an editor changes it: a hub replica an
    editor's replica pushes to, or a plain store shared in place."""
    if request.param == 'hub':
        clones = request.getfixturevalue('replicas')
        return SimpleNamespace(
            kind='hub', path=clones['service'], editor=clones['editor'],
            edit=lambda uuid, change: hub_edit(clones['editor'], leased(uuid, change)),
            published='links published at hub position', raced='the hub moved from')
    path = request.getfixturevalue('shared')
    return SimpleNamespace(
        kind='shared', path=path, editor=None,
        edit=lambda uuid, change: shared_edit(path, change),
        published='links written at position', raced='editor wrote while linking')


def test_prepare_links_publishes_and_pins_a_position(store):
    walk = names(store.path)['walk']
    edited = store.edit(walk, lambda db: db.set_meta(walk, 'sampling_group', 'reviewed'))
    prepared = prepare(store.path, minimum=edited)
    assert prepared.attempts == 1 and prepared.head == edited
    assert prepared.position.sequence > edited  # the links were committed on top
    assert any(store.published in line for line in prepared.lines)
    with ObjectStore.open(store.path, read_only=True) as db:
        assert prepared.position.hub == osi.position_owner(db)
    if store.editor:
        # The editor sees the service's links after pulling to N.
        with ObjectStore.open(store.editor, author='editor') as db:
            assert db.hub_sync(pull=True, push=False)['pulled_position'] == \
                prepared.position.sequence
            assert len(db.edges_by_kind('point-surface')) == 3
    # Later commits do not change N: every replica reads the same inputs at it.
    request = pinned(store.path, prepared.position)
    # The service's store is at N once it has pinned it (a replica pulls its
    # own publication back), so the inputs at N are its current state: the
    # read never rebuilds N from the history, which for a replica is the
    # hub's (opened over the network for an sftp:// hub).
    with ObjectStore.open(store.path, read_only=True) as db:
        before = osi.read_snapshot(NoHistory(db), request)
    assert len(before.relationships) == 3
    assert before.manifest()['position'] == prepared.position.provenance()
    store.edit(walk, lambda db: db.set_meta(walk, 'sampling_group', 'later'))
    for path in filter(None, (store.path, store.editor)):
        if store.editor:
            with ObjectStore.open(path, author='reader') as db:
                db.hub_sync(pull=True, push=False)
        with ObjectStore.open(path, read_only=True) as db:
            assert db.object(walk)['meta']['sampling_group'] == 'later'
        after = read(path, request)
        assert after.digest() == before.digest()
        # Object revision UUIDs are each replica's own (the current state's
        # or the hub's history's): every shared identity agrees.
        assert osi._compared(after.manifest())['objects'] == \
            osi._compared(before.manifest())['objects']
    [entry] = [e for e in before.manifest()['objects'] if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'reviewed'
    # Nothing new to link: the conditional push validates the head, N = H.
    again = prepare(store.path)
    assert again.position.sequence == again.head


def _racing(monkeypatch, edit, every=False):
    """Run ``edit`` after the detectors of the first (or every) refresh."""
    original = osi.refresh_links
    edits = []

    def racing(*args, **kwargs):
        outcomes = original(*args, **kwargs)
        if every or not edits:
            edits.append(edit(len(edits)))
        return outcomes
    monkeypatch.setattr(osi, 'refresh_links', racing)
    return edits


def test_a_commit_while_linking_relinks(store, monkeypatch):
    walk = names(store.path)['walk']
    edits = _racing(monkeypatch, lambda n: store.edit(
        walk, lambda db: db.set_meta(walk, 'review_note', 'saved')))
    prepared = prepare(store.path)
    assert prepared.attempts == 2 and prepared.head >= edits[0]
    assert any(store.raced in line for line in prepared.lines)
    with ObjectStore.open(store.path, read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
        if store.kind == 'shared':
            # Between the head and N only the service wrote.
            assert {entry['author'] for entry in db.journal_since(prepared.head)
                    if entry['seq'] <= prepared.position.sequence} <= {'service'}
    [record] = [r['record'] for r in view['objects'] if r['object_uuid'] == walk]
    assert record['meta']['review_note'] == 'saved'


def test_continuous_edits_fail_the_update_cleanly(store, monkeypatch):
    walk = names(store.path)['walk']
    _racing(monkeypatch, lambda n: store.edit(
        walk, lambda db: db.set_meta(walk, 'review_note', str(n))), every=True)
    with pytest.raises(osi.PositionError) as error:
        prepare(store.path, attempts=3)
    text = '\n'.join(error.value.problems)
    assert 'inputs kept changing' in text
    if store.kind == 'shared':
        return
    # Nothing the service linked reached the hub, or stays pending on it.
    assert 'discarded 3 unpublished' in text
    with ObjectStore.open(store.editor, author='editor') as db:
        db.hub_sync(pull=True, push=False)
        assert db.edges_by_kind('point-surface') == []
    with ObjectStore.open(store.path, read_only=True) as db:
        assert db.hub_status()['pending'] == []
        assert db.edges_by_kind('point-surface') == []
    # Once the edits stop, the next update links afresh.
    monkeypatch.undo()
    prepared = prepare(store.path)
    with ObjectStore.open(store.path, read_only=True) as db:
        assert db.hub_status()['pending'] == []
        assert len(db.hub_read_view_at(prepared.position.sequence)['relationships']) == 3


# ---------------------------------------------------------------------------
# Hub replicas
# ---------------------------------------------------------------------------

def _tag(path, uuid, tags):
    return hub_edit(path, leased(uuid, lambda db: db.upsert_object({**db.object(uuid), 'tags': tags})))


def test_a_deselection_while_linking_discards_its_links(replicas, monkeypatch):
    # Links computed for a head where `far` was selected must not be
    # published at a later head where it is not: relinking only the current
    # selection would never replace them.
    uuids = names(replicas['editor'])
    far, near = uuids['far'], uuids['near']
    for uuid in (near, far):
        _tag(replicas['editor'], uuid, ['reviewed'])
    _racing(monkeypatch, lambda n: _tag(replicas['editor'], far, []))
    config = fit_config(replicas['service'], 'service', object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    with ObjectStore.open(replicas['service'], author='service') as db:
        prepared = osi.prepare_position(db, config, osi.request_from_config(config),
                                        log=lambda line: None)
    assert prepared.attempts == 2 and prepared.uuids['surface'] == [near]
    assert any('discarded 3 unpublished' in line for line in prepared.lines)
    with ObjectStore.open(replicas['service'], read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
        assert db.hub_status()['pending'] == []
    surfaces = {uuid for r in view['relationships'] for uuid in (r['record']['a'], r['record']['b'])}
    assert near in surfaces and far not in surfaces
    assert len(view['relationships']) == 2


def test_a_conflicting_approval_keeps_the_hub_side(replicas, monkeypatch):
    first = prepare(replicas['service'])
    with ObjectStore.open(replicas['editor'], author='editor') as db:
        db.hub_sync(pull=True, push=False)
        edge = db.edges_by_kind('point-surface')[0]['uuid']
    original = osi.refresh_links
    raced = []

    def conflicting(store, snapshot, *args, **kwargs):
        outcomes = original(store, snapshot, *args, **kwargs)
        if not raced:
            # The service rewrites the edge locally while a user approves it.
            store.set_meta(edge, 'link_note', 'service')
            raced.append(hub_edit(replicas['editor'], lambda db: db.set_approval(edge, 'approved')))
        return outcomes

    monkeypatch.setattr(osi, 'refresh_links', conflicting)
    prepared = prepare(replicas['service'], config=fit_config(
        replicas['service'], 'service', object_store_full_link_refresh=True))
    assert prepared.position.sequence > first.position.sequence and prepared.attempts >= 2
    with ObjectStore.open(replicas['service'], read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
        [record] = [r['record'] for r in view['relationships'] if r['record']['uuid'] == edge]
        assert record['approval_state'] == 'approved'
        assert db.hub_status()['conflicts'] == []


class NoHistory:
    """A store whose history reads fail: a read through it must use the
    current state."""

    def __init__(self, store):
        self.store = store

    def hub_read_view_at(self, *args, **kwargs):
        raise AssertionError('read the history instead of the current state')

    def __getattr__(self, name):
        return getattr(self.store, name)


class LostResponse:
    """A replica whose conditional pushes publish, then lose their response
    (the first only, or every one when ``always``)."""

    def __init__(self, store, always=False):
        self.store, self.always, self.lost = store, always, []

    def hub_sync(self, pull=True, push=True, expected_head=None):
        report = self.store.hub_sync(pull=pull, push=push, expected_head=expected_head)
        if expected_head is not None and (self.always or not self.lost):
            self.lost.append(report)
            raise RuntimeError('connection to the hub lost')
        return report

    def __getattr__(self, name):
        return getattr(self.store, name)


def test_a_lost_push_response_recovers_its_position(replicas):
    flaky = {}

    def lose_first(db):
        flaky['store'] = LostResponse(db)
        return flaky['store']
    prepared = prepare(replicas['service'], store=lose_first)
    [lost] = flaky['store'].lost
    assert lost['published_position'] == prepared.position.sequence
    assert prepared.attempts == 1
    # A push that is never confirmed is reported.
    with pytest.raises(osi.PositionError) as error:
        prepare(replicas['service'], config=fit_config(
            replicas['service'], 'service', object_store_full_link_refresh=True),
            store=lambda db: LostResponse(db, always=True))
    text = '\n'.join(error.value.problems)
    assert 'could not be confirmed as published' in text and 'connection to the hub lost' in text


def test_positions_ahead_of_the_hub_or_of_another_hub_are_refused(replicas):
    with ObjectStore.open(replicas['service'], read_only=True) as db:
        head = db.hub_status()['position']
    with pytest.raises(osi.PositionError, match='before the requested position'):
        prepare(replicas['service'], minimum=head + 100)
    prepared = prepare(replicas['service'])
    other = dataclasses.replace(prepared.position, hub='another-hub')
    with pytest.raises(osi.ObjectStoreInputError, match='belongs to another-hub'):
        read(replicas['service'], pinned(replicas['service'], other))


def test_selectors_resolve_at_the_position(replicas):
    far = names(replicas['editor'])['far']
    tagged = _tag(replicas['editor'], far, ['reviewed'])
    config = fit_config(replicas['service'], 'service', object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    request = osi.request_from_config(config)
    with ObjectStore.open(replicas['service'], author='service') as db:
        prepared = osi.prepare_position(db, config, request, log=lambda line: None)
    assert prepared.uuids['surface'] == [far] and prepared.head >= tagged
    _tag(replicas['editor'], far, [])
    with ObjectStore.open(replicas['service'], author='service') as db:
        db.hub_sync(pull=True, push=False)
    with ObjectStore.open(replicas['service'], read_only=True) as db:
        assert osi.resolve_selection(
            db, dataclasses.replace(request, position=prepared.position))['surface'] == [far]
        assert osi.resolve_selection(db, request)['surface'] == []


def test_changed_link_parameters_or_normals_mark_links_stale(replicas):
    config = fit_config(replicas['service'], 'service')
    inputs = osi.load_fit_inputs(config, log=lambda line: None)
    none = osi.fiber_normal_source(config)
    assert osi.stale_link_reasons(inputs.manifest, config, none) == []
    changed = fit_config(replicas['service'], 'service', pcl_link_distance_tolerance=
                         config['pcl_link_distance_tolerance'] * 2)
    [reason] = osi.stale_link_reasons(inputs.manifest, changed, none)
    assert reason.startswith('pcl_link_distance_tolerance changed')
    lasagna = {'pool': '/cache/lasagna/run-1/normals_g4', 'scale': 4}
    assert osi.stale_link_reasons(inputs.manifest, config, osi.fiber_normal_source(
        config, lasagna)) == ['the fiber-patch normal source changed']
