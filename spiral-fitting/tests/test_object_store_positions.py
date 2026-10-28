"""Hub positions of object-store fit inputs: link-then-pin (prepare_position),
pinned reads on any replica, and link staleness."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import put_patch, put_pcl  # noqa: E402


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def _grid(y=100.0, size=8):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([4.0 + rows * 4, np.full(rows.shape, y), 40.0 + cols * 4],
                    axis=-1)


def _on(y, *points):
    return {index: ([z, y, x], None) for index, (z, x) in enumerate(points)}


@pytest.fixture()
def hub(tmp_path):
    """A hub with two patches and a PCL touching each, and two replicas:
    the service's and an editor's."""
    path = str(tmp_path / 'hub.vcobj')
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        put_patch(db, _grid(), (0.25, 0.25), name='near')
        put_patch(db, _grid(140.0), (0.25, 0.25), name='far')
        put_pcl(db, {**_on(100.0, (12.0, 48.0), (16.0, 52.0)),
                     2: ([20.0, 140.0, 56.0], None)}, name='walk')
        db.checkpoint()
    vos.hub_init(path, 'ingest')
    replicas = {}
    for name in ('service', 'editor'):
        replicas[name] = str(tmp_path / f'{name}.vcobj')
        vos.hub_clone(path, replicas[name], name)
    return replicas


def fit_config(path, **overrides):
    config = reference.fit_config()
    config.update(object_store_path=path, object_store_author='service')
    config.update(overrides)
    return config


def _names(path):
    with ObjectStore.open(path, read_only=True) as db:
        return {record['name']: record['uuid'] for kind in osi.KINDS
                for record in db.objects_by_kind(kind)}


def _edit(path, author, change):
    """A checked edit on a replica, leased and pushed like an editor's."""
    with ObjectStore.open(path, author=author) as db:
        db.hub_sync(pull=True, push=False)
        change(db)
        report = db.hub_sync(pull=True, push=True)
    assert not report['conflicts']
    return report['published_position']


def _lease_and(uuid, edit):
    def change(db):
        db.acquire_leases([uuid])
        edit(db)
    return change


def prepare(path, **options):
    config = fit_config(path)
    request = osi.request_from_config(config)
    log = []
    with ObjectStore.open(path, author='service') as store:
        prepared = osi.prepare_position(store, config, request, log=log.append, **options)
    return prepared


def test_prepare_links_publishes_and_pins_a_position(hub):
    walk = _names(hub['editor'])['walk']
    edited = _edit(hub['editor'], 'editor', _lease_and(
        walk, lambda db: db.set_meta(walk, 'sampling_group', 'reviewed')))
    prepared = prepare(hub['service'], minimum=edited)
    assert prepared.attempts == 1 and prepared.head == edited
    assert prepared.position.sequence > edited  # the links were published on top
    assert any('links published at hub position' in line for line in prepared.lines)
    # The editor sees the service's links after pulling to N.
    with ObjectStore.open(hub['editor'], author='editor') as db:
        assert db.hub_sync(pull=True, push=False)['pulled_position'] == prepared.position.sequence
        assert len(db.edges_by_kind('point-surface')) == 3
    # Later commits do not change N: every replica reads the same inputs at it.
    request = dataclasses.replace(osi.request_from_config(fit_config(hub['service'])),
                                  position=prepared.position)
    with ObjectStore.open(hub['service'], read_only=True) as db:
        before = osi.read_snapshot(db, request)
    _edit(hub['editor'], 'editor', _lease_and(
        walk, lambda db: db.set_meta(walk, 'sampling_group', 'later')))
    for name in ('service', 'editor'):
        with ObjectStore.open(hub[name], author=name) as db:
            db.hub_sync(pull=True, push=False)
        with ObjectStore.open(hub[name], read_only=True) as db:
            after = osi.read_snapshot(db, request)
            assert db.object(walk)['meta']['sampling_group'] == 'later'
        assert after.digest() == before.digest()
        assert after.manifest()['objects'] == before.manifest()['objects']
    [entry] = [e for e in before.manifest()['objects'] if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'reviewed'
    assert before.manifest()['position'] == prepared.position.provenance()


def test_no_op_linking_pins_the_head(hub):
    first = prepare(hub['service'])
    again = prepare(hub['service'])
    # Nothing new to link: the conditional push validates the head, N=H.
    assert again.position == first.position and again.head == first.position.sequence


def test_a_foreign_commit_while_linking_relinks(hub, monkeypatch):
    walk = _names(hub['editor'])['walk']
    original = osi.refresh_links
    calls = []

    def racing(store, snapshot, *args, **kwargs):
        outcomes = original(store, snapshot, *args, **kwargs)
        if not calls:  # an unrelated metadata edit lands after H
            calls.append(_edit(hub['editor'], 'editor', _lease_and(
                walk, lambda db: db.set_meta(walk, 'review_note', 'unrelated'))))
        return outcomes

    monkeypatch.setattr(osi, 'refresh_links', racing)
    prepared = prepare(hub['service'])
    assert prepared.attempts == 2 and prepared.head == calls[0]
    assert any('the hub moved from' in line for line in prepared.lines)
    with ObjectStore.open(hub['service'], read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
    [walk_record] = [r['record'] for r in view['objects'] if r['object_uuid'] == walk]
    assert walk_record['meta']['review_note'] == 'unrelated'


def test_continuous_edits_fail_the_update_cleanly(hub, monkeypatch):
    walk = _names(hub['editor'])['walk']
    original = osi.refresh_links
    count = []

    def busy(store, snapshot, *args, **kwargs):
        outcomes = original(store, snapshot, *args, **kwargs)
        count.append(_edit(hub['editor'], 'editor', _lease_and(
            walk, lambda db: db.set_meta(walk, 'review_note', str(len(count))))))
        return outcomes

    monkeypatch.setattr(osi, 'refresh_links', busy)
    with pytest.raises(osi.PositionError) as error:
        prepare(hub['service'], attempts=3)
    text = '\n'.join(error.value.problems)
    assert 'inputs kept changing' in text and 'discarded 3 unpublished' in text
    # Nothing the service linked reached the hub, or stays pending on it.
    with ObjectStore.open(hub['editor'], author='editor') as db:
        db.hub_sync(pull=True, push=False)
        assert db.edges_by_kind('point-surface') == []
    with ObjectStore.open(hub['service'], read_only=True) as db:
        assert db.hub_status()['pending'] == []
        assert db.edges_by_kind('point-surface') == []
    # Once the edits stop, the next update links afresh.
    monkeypatch.setattr(osi, 'refresh_links', original)
    prepared = prepare(hub['service'])
    with ObjectStore.open(hub['service'], read_only=True) as db:
        assert db.hub_status()['pending'] == []
        assert len(db.hub_read_view_at(prepared.position.sequence)['relationships']) == 3


def test_a_deselection_while_linking_discards_its_links(hub, monkeypatch):
    # Links computed for a head where `far` was selected must not be
    # published at a later head where it is not: relinking only the current
    # selection would never replace them.
    names = _names(hub['editor'])
    far, near = names['far'], names['near']
    for uuid in (near, far):
        _edit(hub['editor'], 'editor', _lease_and(
            uuid, lambda db, uuid=uuid: db.upsert_object({**db.object(uuid), 'tags': ['reviewed']})))
    original = osi.refresh_links
    raced = []

    def racing(store, snapshot, *args, **kwargs):
        outcomes = original(store, snapshot, *args, **kwargs)
        if not raced:
            raced.append(_edit(hub['editor'], 'editor', _lease_and(
                far, lambda db: db.upsert_object({**db.object(far), 'tags': []}))))
        return outcomes

    monkeypatch.setattr(osi, 'refresh_links', racing)
    config = fit_config(hub['service'], object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    with ObjectStore.open(hub['service'], author='service') as store:
        prepared = osi.prepare_position(store, config, osi.request_from_config(config),
                                        log=lambda line: None)
    assert prepared.attempts == 2 and prepared.uuids['surface'] == [near]
    assert any('discarded 3 unpublished' in line for line in prepared.lines)
    with ObjectStore.open(hub['service'], read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
        assert db.hub_status()['pending'] == []
    surfaces = {uuid for r in view['relationships'] for uuid in (r['record']['a'], r['record']['b'])}
    assert near in surfaces and far not in surfaces
    assert len(view['relationships']) == 2


def test_a_conflicting_approval_keeps_the_hub_side(hub, monkeypatch):
    first = prepare(hub['service'])
    with ObjectStore.open(hub['editor'], author='editor') as db:
        db.hub_sync(pull=True, push=False)
        edge = db.edges_by_kind('point-surface')[0]['uuid']
    original = osi.refresh_links
    raced = []

    def conflicting(store, snapshot, *args, **kwargs):
        outcomes = original(store, snapshot, *args, **kwargs)
        if not raced:
            # The service rewrites the edge locally while a user approves it.
            store.set_meta(edge, 'link_note', 'service')
            raced.append(_edit(hub['editor'], 'editor', lambda db: db.set_approval(edge, 'approved')))
        return outcomes

    monkeypatch.setattr(osi, 'refresh_links', conflicting)
    prepared = prepare(hub['service'])
    assert prepared.position.sequence > first.position.sequence and prepared.attempts >= 2
    with ObjectStore.open(hub['service'], read_only=True) as db:
        view = db.hub_read_view_at(prepared.position.sequence)
        [record] = [r['record'] for r in view['relationships'] if r['record']['uuid'] == edge]
        assert record['approval_state'] == 'approved'
        assert db.hub_status()['conflicts'] == []


class LostResponse:
    """A replica whose first conditional push publishes, then loses its
    response."""

    def __init__(self, store):
        self.store, self.lost = store, []

    def hub_sync(self, pull=True, push=True, expected_head=None):
        report = self.store.hub_sync(pull=pull, push=push, expected_head=expected_head)
        if expected_head is not None and not self.lost:
            self.lost.append(report)
            raise RuntimeError('connection to the hub lost')
        return report

    def __getattr__(self, name):
        return getattr(self.store, name)


def test_a_lost_push_response_recovers_its_position(hub):
    config = fit_config(hub['service'])
    with ObjectStore.open(hub['service'], author='service') as store:
        flaky = LostResponse(store)
        prepared = osi.prepare_position(flaky, config, osi.request_from_config(config),
                                        log=lambda line: None)
    [lost] = flaky.lost
    assert lost['published_position'] == prepared.position.sequence
    assert prepared.attempts == 1


class Unreachable(LostResponse):
    """A replica whose conditional pushes always lose their response."""

    def hub_sync(self, pull=True, push=True, expected_head=None):
        report = self.store.hub_sync(pull=pull, push=push, expected_head=expected_head)
        if expected_head is not None:
            self.lost.append(report)
            raise RuntimeError('connection to the hub lost')
        return report


def test_an_unconfirmed_push_is_reported(hub):
    config = fit_config(hub['service'])
    with ObjectStore.open(hub['service'], author='service') as store:
        with pytest.raises(osi.PositionError) as error:
            osi.prepare_position(Unreachable(store), config, osi.request_from_config(config),
                                 log=lambda line: None)
    text = '\n'.join(error.value.problems)
    assert 'could not be confirmed as published' in text and 'connection to the hub lost' in text


def test_a_minimum_ahead_of_the_hub_is_refused(hub):
    with ObjectStore.open(hub['service'], read_only=True) as db:
        head = db.hub_status()['position']
    with pytest.raises(osi.PositionError, match='before the requested position'):
        prepare(hub['service'], minimum=head + 100)


def test_a_position_of_another_hub_is_refused(hub, tmp_path):
    prepared = prepare(hub['service'])
    other = dataclasses.replace(prepared.position, hub='another-hub')
    request = dataclasses.replace(osi.request_from_config(fit_config(hub['service'])),
                                  position=other)
    with ObjectStore.open(hub['service'], read_only=True) as db:
        with pytest.raises(osi.ObjectStoreInputError, match='belongs to hub another-hub'):
            osi.read_snapshot(db, request)


def test_selectors_resolve_at_the_position(hub):
    names = _names(hub['editor'])
    far = names['far']
    tagged = _edit(hub['editor'], 'editor', _lease_and(
        far, lambda db: db.upsert_object({**db.object(far), 'tags': ['reviewed']})))
    config = fit_config(hub['service'], object_store_patch_tags=['reviewed'],
                        object_store_patch_uuids=[])
    request = osi.request_from_config(config)
    with ObjectStore.open(hub['service'], author='service') as store:
        prepared = osi.prepare_position(store, config, request, log=lambda line: None)
    assert prepared.uuids['surface'] == [far] and prepared.head >= tagged
    _edit(hub['editor'], 'editor', _lease_and(
        far, lambda db: db.upsert_object({**db.object(far), 'tags': []})))
    with ObjectStore.open(hub['service'], author='service') as db:
        db.hub_sync(pull=True, push=False)
    pinned = dataclasses.replace(request, position=prepared.position)
    with ObjectStore.open(hub['service'], read_only=True) as db:
        assert osi.resolve_selection(db, pinned)['surface'] == [far]
        assert osi.resolve_selection(db, request)['surface'] == []


def test_load_fit_inputs_on_a_replica_matches_the_plain_store(hub, tmp_path):
    # The same content as an ordinary store gives the same digest: positions
    # and how links were made are provenance, not inputs.
    inputs = osi.load_fit_inputs(fit_config(hub['service']), log=lambda line: None)
    assert inputs.manifest['position']['sequence'] > 0
    assert inputs.manifest['link_refresh']['parameters'] == {
        key: fit_config(hub['service'])[key] for key in osi.LINK_PARAMETER_KEYS}
    import shutil
    plain = str(tmp_path / 'plain.vcobj')
    shutil.copyfile(hub['service'], plain)
    vos.hub_detach(plain, 'plain')
    copy = osi.load_fit_inputs(fit_config(plain, object_store_refresh_links=False),
                               log=lambda line: None)
    assert copy.manifest['position'] is None
    assert copy.digest == inputs.digest
    with pytest.raises(osi.ObjectStoreInputError, match='not a hosted replica'):
        osi.load_fit_inputs(fit_config(plain), log=lambda line: None, minimum_position=1)


def test_changed_link_parameters_or_normals_mark_links_stale(hub):
    inputs = osi.load_fit_inputs(fit_config(hub['service']), log=lambda line: None)
    config = fit_config(hub['service'])
    none = osi.fiber_normal_source(config)
    assert osi.stale_link_reasons(inputs.manifest, config, none) == []
    changed = fit_config(hub['service'], pcl_link_distance_tolerance=
                         config['pcl_link_distance_tolerance'] * 2)
    [reason] = osi.stale_link_reasons(inputs.manifest, changed, none)
    assert reason.startswith('pcl_link_distance_tolerance changed')
    lasagna = {'normal_x': '/x', 'normal_y': '/y', 'group': '4', 'scale': 4}
    assert osi.stale_link_reasons(inputs.manifest, config, osi.fiber_normal_source(
        config, lasagna)) == ['the fiber-patch normal source changed']
