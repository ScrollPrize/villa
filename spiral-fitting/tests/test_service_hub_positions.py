"""A service bound to a hub replica: the launch binding; initialize and
full-input rebuilds that link and pin a hub position; and input updates of a
resident fit at an iteration boundary.

An update replaces a session's inputs with those of a position in three
phases: every rank prepares them (touching nothing resident), installs them
(the destructive phase) and publishes them. A refusal leaves the fit as it
was; a failure after the release fails the session. The runtime is real on
the CPU with the device stages stubbed (the theta topology needs the model
they leave out, so updates stub it too).
"""
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import spiral_runtime  # noqa: E402
import spiral_service  # noqa: E402
from cpu_service import (  # noqa: E402
    REQUEST, RUN, request, service_state, settle, stub_device_stages, two_ranks,
    wait_until, write_inputs)
from object_store_fixture import hub_edit, leased, names, plane, put_patch  # noqa: E402
from service_project import with_project  # noqa: E402

pytestmark = pytest.mark.usefixtures('cpu')


@pytest.fixture()
def hub(store_path, tmp_path):
    """The startup fixture's store as a hub, with an editor's replica; the
    service's replica is cloned by its binding."""
    vos.hub_init(store_path, 'ingest')
    editor = str(tmp_path / 'editor.vcobj')
    vos.hub_clone(store_path, editor, 'editor')
    return {'hub': store_path, 'editor': editor,
            'service': str(tmp_path / 'replicas' / 'service.vcobj')}


@pytest.fixture()
def state(hub, tmp_path):
    state = service_state(
        spiral_service.bind_object_store(hub['service'], 'svc', hub['hub']), tmp_path)
    yield state
    state.close()


@pytest.fixture()
def no_theta_topology(monkeypatch):
    import fit_spiral
    monkeypatch.setattr(fit_spiral.FitContext, '_build_theta_crossing_map', lambda self: [])


def _edit(path, uuid, key, value):
    """An editor's saved metadata edit, pushed: the hub position it reached."""
    return hub_edit(path, leased(uuid, lambda db: db.set_meta(uuid, key, value)))


def _add_patch(path, y=120.0, name='added'):
    return lambda: hub_edit(path, lambda db: put_patch(db, plane(y), (0.25, 0.25), name=name))


def _pinned_at_new_position(state, change):
    """Make ``change``, then link and pin a position on the service replica
    as an update would. Returns it."""
    change()
    config = state.session._context.config
    with ObjectStore.open(state.object_store.path, author='svc') as store:
        prepared = osi.prepare_position(store, config, osi.request_from_config(config),
                                        log=lambda line: None)
    return prepared.position.sequence


def _settled_update(state, timeout=60):
    wait_until(lambda: state.status()['input_update'] is None, timeout,
               'the input update did not settle')
    return state.status()


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------

def test_the_binding_clones_and_checks_its_hub(hub, tmp_path, capsys):
    binding = spiral_service.bind_object_store(hub['service'], 'svc', hub['hub'])
    assert binding.hosted and Path(hub['service']).is_file()
    # The clone is announced before it starts and when it ends (the only
    # startup lines a client sees while a large hub copies); a reused
    # replica is not.
    printed = capsys.readouterr().out
    assert 'Cloning the object-store hub' in printed and 'Cloned the object-store hub' in printed
    assert binding.hub == str(Path(hub['hub']).resolve())
    with ObjectStore.open(hub['editor'], read_only=True) as db:
        assert binding.hub_identity == db.hub_status()['hub_replica']
    # An existing replica binds its own hub, with or without the flag.
    assert spiral_service.bind_object_store(hub['service'], 'svc') == binding
    assert 'Cloning' not in capsys.readouterr().out
    advertised = binding.advertisement()['hub']
    assert advertised['identity'] == binding.hub_identity and advertised['position'] > 0
    # A replica that cannot be read says why rather than reporting nothing.
    unreadable = dataclasses.replace(binding, path=str(tmp_path / 'gone.vcobj'))
    advertised = unreadable.advertisement()['hub']
    assert advertised['position'] is None and advertised['position_error']
    # A plain store, or another hub's replica, is refused for this hub.
    plain = str(tmp_path / 'plain.vcobj')
    ObjectStore.create(plain, author='x').close()
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='not a replica'):
        spiral_service.bind_object_store(plain, 'svc', hub['hub'])
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='replicates the hub'):
        spiral_service.bind_object_store(hub['service'], 'svc', plain)
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='cannot clone'):
        spiral_service.bind_object_store(str(tmp_path / 'new.vcobj'), 'svc',
                                         str(tmp_path / 'absent-hub.vcobj'))


def test_startup_drops_unpublished_links_and_refuses_other_edits(hub):
    binding = spiral_service.bind_object_store(hub['service'], 'svc', hub['hub'])
    uuids = names(hub['service'])
    whole = {'whole_geometry': True}
    # An interrupted update's links are dropped at the next start.
    with ObjectStore.open(hub['service'], author='svc') as db:
        db.add_edge({'kind': 'related', 'a': uuids['near'], 'b': uuids['far'],
                     'a_anchor': whole, 'b_anchor': whole})
        assert len(db.hub_status()['pending']) == 1
    assert spiral_service.bind_object_store(hub['service'], 'svc', hub['hub']) == binding
    with ObjectStore.open(hub['service'], read_only=True) as db:
        assert db.hub_status()['pending'] == []
        assert db.edges_by_kind('related') == []
    # Any other edit is not the service's: startup refuses it.
    with ObjectStore.open(hub['service'], author='svc') as db:
        db.acquire_leases([uuids['walk']])
        db.set_meta(uuids['walk'], 'sampling_group', 'not the service')
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='unpublished edits'):
        spiral_service.bind_object_store(hub['service'], 'svc', hub['hub'])
    # The editor's replica named by mistake: its unpushed edge survives.
    with ObjectStore.open(hub['editor'], author='editor') as db:
        db.hub_sync(pull=True, push=False)
        edge = db.add_edge({'kind': 'related', 'a': uuids['near'], 'b': uuids['far'],
                            'a_anchor': whole, 'b_anchor': whole})
    with pytest.raises(spiral_service.ObjectStoreBindingError,
                       match="not this service's replica: editor"):
        spiral_service.bind_object_store(hub['editor'], 'svc', hub['hub'])
    with ObjectStore.open(hub['editor'], read_only=True) as db:
        assert db.hub_status()['pending'] == [edge]


def test_hub_addresses_compare_as_the_store_records_them(tmp_path):
    # The store records a remote hub in canonical sftp:// form: an ssh://
    # address, or an unencoded path, must compare equal to it on restart.
    recorded = 'sftp://host/scrolls/a%20b/hub.vcobj'
    for given in ('ssh://host/scrolls/a b/hub.vcobj', 'sftp://host/scrolls/a b/hub.vcobj',
                  recorded):
        assert spiral_service._canonical_hub(vos, given) == recorded
    (tmp_path / 'real').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'real')
    assert spiral_service._canonical_hub(vos, tmp_path / 'link' / 'hub.vcobj') == \
        str((tmp_path / 'real' / 'hub.vcobj').resolve())


# ---------------------------------------------------------------------------
# Initialize and rebuilds link and pin a position
# ---------------------------------------------------------------------------

def test_initialize_links_and_pins_the_requested_position(state, hub, device_stubs):
    advertised = state.binding_advertisement()
    assert advertised['capabilities']['input_positions'] is True
    assert advertised['object_store']['hub']['identity'] == state.object_store.hub_identity
    walk = names(hub['editor'])['walk']
    edited = _edit(hub['editor'], walk, 'sampling_group', 'reviewed')
    assert state.initialize(with_project({**REQUEST, 'inputs_position': edited}))['build_id']
    status = settle(state)
    assert status['state'] == 'Idle', status['error']
    position = status['input_position']
    assert position['source'] == 'prepared' and position['requested'] == edited
    assert position['sequence'] > edited == position['head']  # the service's links
    context = state.session._context
    assert context.config['object_store_input_position'] == position['sequence']
    manifest = context.object_store_inputs.manifest
    assert manifest['position'] == {'hub': state.object_store.hub_identity,
                                    'sequence': position['sequence']}
    [entry] = [e for e in manifest['objects'] if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'reviewed'
    # The fit read the pinned position once; it did not link again.
    assert len(device_stubs) == 1
    # The editor reaches N by pulling, links included.
    with ObjectStore.open(hub['editor'], author='editor') as db:
        assert db.hub_sync(pull=True, push=False)['pulled_position'] == position['sequence']
        assert len(db.edges_by_kind('point-surface')) == 3
    # A later edit is not part of the fit; a model rebuild keeps the position.
    _edit(hub['editor'], walk, 'sampling_group', 'later')
    assert state.rebuild(with_project(request(model_num_flow_stages=3)))['stage'] == 'model'
    assert settle(state)['input_position'] == position
    assert state.session._context.config['object_store_input_position'] == position['sequence']
    # A reload rebuild links and pins a new position including it.
    assert state.rebuild(with_project({**REQUEST, 'inputs': 'reload'}))['inputs'] == 'reload'
    assert settle(state)['input_position']['sequence'] > position['sequence']
    [entry] = [e for e in state.session._context.object_store_inputs.manifest['objects']
               if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'later'


def test_a_failed_preparation_keeps_the_resident_fit(state, hub, device_stubs,
                                                      monkeypatch):
    state.initialize(with_project(REQUEST))
    first = settle(state)
    session_id, context = state.session_id, state.session._context

    def refuse(*args, **kwargs):
        raise osi.PositionError(['inputs kept changing: no hub position could be '
                                 'linked in 4 attempts; retry'])
    monkeypatch.setattr(osi, 'prepare_position', refuse)
    response = state.rebuild(with_project({**REQUEST, 'inputs': 'reload'}))
    wait_until(lambda: not state._building)
    status = state.status()
    assert status['state'] == 'Idle' and state.session_id == session_id
    assert state.session._context is context
    assert status['input_position'] == first['input_position']
    failure = status['input_preparation_error']
    assert failure['build_id'] == response['build_id']
    assert 'inputs kept changing' in failure['problems'][0]
    errors = [event for event in state.events.read_after(0)['events']
              if event['kind'] == 'error']
    assert 'the resident fit is unchanged' in errors[-1]['text']
    # The lease lifecycle answers the command with the failure.
    state.lease().claim('client-token', 'claim-1')
    with pytest.raises(spiral_service.ApiError, match='could not be prepared'):
        state.lease_lifecycle('client-token', 'session_rebuild',
                              with_project({**REQUEST, 'inputs': 'reload',
                                            'command_id': 'c-1'}),
                              state.rebuild)
    assert state.session_id == session_id


def test_a_checkpoint_resumes_at_its_recorded_position(state, tmp_path, monkeypatch):
    import torch
    built = []
    monkeypatch.setattr(state, '_build', lambda *args: built.append(args))
    monkeypatch.setattr(osi, 'prepare_position', lambda *args, **kwargs: pytest.fail(
        'a checkpoint resume does not link'))

    def checkpoint(name, hub_identity):
        torch.save({'cfg': {}, 'input_manifest': {'object_store': {'position': {
            'hub': hub_identity, 'sequence': 7}}}}, tmp_path / name)
        return str(tmp_path / name)
    path = checkpoint('fit.ckpt', state.object_store.hub_identity)
    assert state._checkpoint_position(path) == 7
    assert state._checkpoint_position(checkpoint('other.ckpt', 'another-hub')) is None
    paths, run, preview, scroll = state._prepare_session_request(with_project(REQUEST))
    state._begin_build(dataclasses.replace(paths, checkpoint=path), run, preview, scroll)
    wait_until(lambda: built)
    _, _, _, pinned_run, _, _ = built[0]
    assert pinned_run.config['object_store_input_position'] == 7
    assert state.status()['input_position']['source'] == 'checkpoint'


def test_a_shared_store_has_positions_a_client_cannot_pin(store_path, tmp_path):
    state = service_state(spiral_service.bind_object_store(store_path, 'svc'), tmp_path)
    try:
        assert state.binding_advertisement()['capabilities']['input_positions'] is True
        with pytest.raises(spiral_service.ApiError, match='non-negative integer'):
            state._inputs_position(-1)
        with pytest.raises(spiral_service.ApiError, match='must not carry'):
            state.initialize(with_project(request(object_store_input_position=3)))
    finally:
        state.close()


# ---------------------------------------------------------------------------
# A resident fit's inputs replaced at a new position: the context
# ---------------------------------------------------------------------------

def test_prepared_inputs_leave_the_resident_fit_untouched(state, hub, device_stubs,
                                                          no_theta_topology):
    state.initialize(with_project(REQUEST))
    settle(state)
    context = state.session._context
    before = {name: getattr(context, name) for name in context._INPUT_HOST_ATTRIBUTES}
    patches = context.num_verified_patches
    position = _pinned_at_new_position(state, _add_patch(hub['editor']))
    inputs = context.load_replacement_inputs(position)
    rng = np.random.get_state()
    candidate = context.prepare_input_replacement(inputs)
    # Nothing resident changed, and training draws continue as before.
    assert all(getattr(context, name) is value for name, value in before.items())
    assert np.array_equal(np.random.get_state()[1], rng[1])
    assert candidate.num_verified_patches == patches + 1
    assert candidate.patch_atlas is not context.patch_atlas
    # The device half.
    context.replace_inputs(candidate)
    assert context.num_verified_patches == patches + 1
    assert context.object_store_inputs is inputs
    assert context.config['object_store_input_position'] == position
    assert context.patch_atlas.zyxs_flat is not None  # materialized
    with open(Path(context.out_path) / 'object_store_inputs.json') as stream:
        assert json.load(stream)['digest'] == inputs.digest


def test_inputs_that_would_change_the_model_frame_are_refused(state, device_stubs):
    import copy
    import fit_spiral
    state.initialize(with_project(REQUEST))
    settle(state)
    context = state.session._context
    inputs = copy.copy(context.object_store_inputs)
    inputs.manifest = {**inputs.manifest, 'volume_uuid': 'another-volume'}
    with pytest.raises(fit_spiral.InputReplacementRefused, match='volume_uuid differs'):
        context.prepare_input_replacement(inputs)
    # Tracks and their crossings are prepared at build.
    inputs.manifest = {**context.object_store_inputs.manifest, 'relationships': [
        *context.object_store_inputs.manifest['relationships'],
        {'uuid': 'a-new-crossing', 'kind': 'crossing', 'family': 'track_crossing'}]}
    with pytest.raises(fit_spiral.InputReplacementRefused, match='tracks or their crossings'):
        context.prepare_input_replacement(inputs)
    context.using_tracks = True
    context.config.update({'track_exclusion_radius': 4.0})
    with pytest.raises(fit_spiral.InputReplacementRefused, match='track exclusion'):
        context.prepare_input_replacement(context.object_store_inputs)


# ---------------------------------------------------------------------------
# The session's update
# ---------------------------------------------------------------------------

def test_an_update_replaces_the_inputs_and_keeps_the_iteration(state, hub, device_stubs,
                                                                no_theta_topology):
    state.initialize(with_project(REQUEST))
    before = settle(state)
    session = state.session
    patches = session._context.num_verified_patches
    position = _pinned_at_new_position(state, _add_patch(hub['editor']))
    # A patch the old inputs' theta gate excluded is judged afresh.
    session._context.non_liftable_patch_ids.add('excluded-before')
    result = session.update_inputs(position)
    status = session.status()
    assert status['state'] == 'Idle'
    assert session._context.non_liftable_patch_ids == set()
    assert status['input_generation'] == 1 and result['input_generation'] == 1
    assert status['input_digest'] == result['digest'] != before['input_digest']
    assert status['current_iteration'] == before['current_iteration']
    assert status['config_revision'] == before['config_revision'] + 1
    assert status['applied_config']['object_store_input_position'] == position
    assert session._context.num_verified_patches == patches + 1
    with open(status['input_manifest_path']) as stream:
        assert json.load(stream)['manifest']['position']['sequence'] == position


def test_the_old_device_state_is_released_before_the_new_is_built(state, hub, device_stubs,
                                                                   no_theta_topology, monkeypatch):
    import weakref
    import fit_spiral
    state.initialize(with_project(REQUEST))
    settle(state)
    context = state.session._context
    old_atlas = context.patch_atlas
    # The theta topology's closures hold the old atlas, as the real one does.
    context.theta_crossing_map = [lambda atlas=old_atlas: atlas]
    released = weakref.ref(old_atlas)
    del old_atlas
    materialize = fit_spiral.PatchAtlas.materialize
    seen = []

    def checked(self, device=None):
        seen.append(released() is None)
        return materialize(self, device)
    monkeypatch.setattr(fit_spiral.PatchAtlas, 'materialize', checked)
    state.session.update_inputs(_pinned_at_new_position(state, _add_patch(hub['editor'])))
    assert seen == [True]


def test_a_failure_after_release_fails_the_session(state, hub, device_stubs, no_theta_topology,
                                                   monkeypatch):
    import fit_spiral
    state.initialize(with_project(REQUEST))
    settle(state)
    session = state.session
    position = _pinned_at_new_position(state, _add_patch(hub['editor']))

    def fail(self, device=None):
        raise MemoryError('CUDA out of memory')
    monkeypatch.setattr(fit_spiral.PatchAtlas, 'materialize', fail)
    with pytest.raises(spiral_runtime.InputUpdateFailed, match='rebuilt or loaded from a checkpoint'):
        session.update_inputs(position)
    wait_until(lambda: session.status()['state'] == 'Error')


def test_an_update_of_a_busy_session_is_refused_without_a_command(state, hub, device_stubs,
                                                                    no_theta_topology):
    # A run admitted first: the update refuses before admission, and queues
    # no discard (a stray command would break the running step's barrier).
    state.initialize(with_project(REQUEST))
    settle(state)
    session = state.session
    position = _pinned_at_new_position(state, _add_patch(hub['editor']))
    with session._condition:
        session._state = spiral_runtime.SessionState.Running
        epoch = session._command_epoch
    try:
        with pytest.raises(spiral_runtime.InputUpdateRefused, match='Running'):
            session.update_inputs(position)
        with session._condition:
            assert session._command_epoch == epoch and session._commands == []
    finally:
        with session._condition:
            session._state = spiral_runtime.SessionState.Idle
    assert session.update_inputs(position)['input_generation'] == 1


class _Coordinator(spiral_runtime.DistributedInteractiveFitSession):
    """The distributed coordinator's update logic over scripted rank answers."""

    def __init__(self, answers):
        self.answers, self.calls, self.failed = answers, [], None

    def status(self):
        return {'state': spiral_runtime.SessionState.Idle}

    def _call(self, name, arguments=None, ranks=None, timeout=0, collective=True,
              return_all=False):
        self.calls.append(name)
        answer = self.answers.get(name)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    @property
    def _failed_error(self):
        return self.failed

    def _fail_session(self, cause, *, rank=None, detail=None):
        self.failed = str(cause)
        return self.failed


def test_the_coordinator_installs_only_what_every_rank_prepared():
    prepared = {0: {'digest': 'a', 'position': 7}, 1: {'digest': 'a', 'position': 7}}
    coordinator = _Coordinator({'prepare_inputs': prepared, 'install_inputs': {},
                                'publish_inputs': {'input_generation': 1}})
    assert coordinator.update_inputs(7)['input_generation'] == 1
    assert coordinator.calls == ['prepare_inputs', 'install_inputs', 'publish_inputs']

    # Ranks that disagree, or a rank that refuses, install nothing.
    for answer in ({0: {'digest': 'a'}, 1: {'digest': 'b'}},
                   RuntimeError('rank 1: the inputs cannot replace this fit\'s')):
        coordinator = _Coordinator({'prepare_inputs': answer})
        with pytest.raises(spiral_runtime.InputUpdateRefused):
            coordinator.update_inputs(7)
        assert coordinator.calls == ['prepare_inputs', 'discard_inputs']
        assert coordinator.failed is None

    # A rank lost while preparing has failed the session: not a refusal.
    class Lost(_Coordinator):
        def _call(self, name, *args, **kwargs):
            if name == 'prepare_inputs':
                self.calls.append(name)
                self.failed = 'Timed out waiting for GPU worker ranks [1] to prepare_inputs'
                raise TimeoutError(self.failed)
            return super()._call(name, *args, **kwargs)
    coordinator = Lost({})
    with pytest.raises(spiral_runtime.InputUpdateFailed, match='Timed out'):
        coordinator.update_inputs(7)
    assert coordinator.calls == ['prepare_inputs']

    # A publish failure after every rank installed fails the session too.
    coordinator = _Coordinator({'prepare_inputs': prepared, 'install_inputs': {},
                                'publish_inputs': RuntimeError('rank 1: publish')})
    with pytest.raises(spiral_runtime.InputUpdateFailed, match='could not be published'):
        coordinator.update_inputs(7)
    assert 'rank 1: publish' in coordinator.failed

    # A busy session refuses before any rank is asked.
    coordinator = _Coordinator({})
    coordinator.status = lambda: {'state': spiral_runtime.SessionState.Running}
    with pytest.raises(spiral_runtime.InputUpdateRefused, match='Running'):
        coordinator.update_inputs(7)
    assert coordinator.calls == []

    # A failure past the release on any rank fails the whole session.
    coordinator = _Coordinator({'prepare_inputs': prepared,
                                'install_inputs': RuntimeError('rank 0: out of memory')})
    with pytest.raises(spiral_runtime.InputUpdateFailed, match='after resident inputs were released'):
        coordinator.update_inputs(7)
    assert coordinator.calls == ['prepare_inputs', 'install_inputs']
    assert 'out of memory' in coordinator.failed


def _rank_update(rank, world_size, rendezvous, root, config, position, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import fit_spiral
    from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig
    from spiral_runtime import InteractiveFitSession, RendezvousEndpoint
    import object_store_reference as reference
    stub_device_stages(fit_spiral)
    fit_spiral.FitContext._build_theta_crossing_map = lambda self: []
    session = InteractiveFitSession(
        SpiralInputPaths(umbilicus=f'{root}/umbilicus.json',
                         output_directory=f'{root}-out-{rank}'),
        SpiralRunConfig(0, 64, config=config), SpiralPreviewConfig(),
        reference.scroll_spec(), lambda status: None, publishes_outputs=rank == 0,
        rank=rank, world_size=world_size, local_rank=rank,
        rendezvous=RendezvousEndpoint(rendezvous))
    deadline = time.monotonic() + 120
    while session.status()['state'] not in ('Idle', 'Error') and time.monotonic() < deadline:
        time.sleep(0.05)
    before = session.status()
    outcome = {'before': before['input_digest']}
    try:
        result = session.update_inputs(position)
        outcome.update(after=result['digest'], patches=session._context.num_verified_patches,
                       iteration=session.status()['current_iteration'] == before['current_iteration'])
    except Exception as error:  # noqa: BLE001 - reported to the test
        outcome['error'] = f'{type(error).__name__}: {error}'
    with open(results[rank], 'w') as stream:
        json.dump(outcome, stream)
    session.close()


def test_two_ranks_install_the_same_inputs(state, hub, device_stubs, tmp_path):
    state.initialize(with_project(REQUEST))
    first = settle(state)['input_position']['sequence']
    position = _pinned_at_new_position(state, _add_patch(hub['editor']))
    # Each rank builds at the session's first position (read as pinned: the
    # service linked it), then updates to the new one.
    config = {**RUN, 'object_store_path': state.object_store.path, 'object_store_author': 'svc',
              'object_store_input_position': first}
    outcome = two_ranks(_rank_update, tmp_path, str(write_inputs(tmp_path / 'ranks-inputs')),
                        config, position)
    assert all('error' not in rank for rank in outcome), outcome
    assert outcome[0]['after'] == outcome[1]['after'] != outcome[0]['before']
    assert outcome[0]['patches'] == outcome[1]['patches']
    assert outcome[0]['iteration'] and outcome[1]['iteration']


# ---------------------------------------------------------------------------
# The service's update route
# ---------------------------------------------------------------------------

def test_the_service_updates_the_resident_fit_at_a_position(state, hub, device_stubs,
                                                             no_theta_topology):
    state.initialize(with_project(REQUEST))
    before = settle(state)
    capabilities = state.binding_advertisement()['capabilities']
    assert capabilities['input_updates'] is True and capabilities['inputs_frozen'] is False
    session_id = state.session_id
    reached = _add_patch(hub['editor'])()
    with pytest.raises(spiral_service.ApiError, match='input generation 3') as stale:
        state.update_inputs({'inputs_position': reached, 'expected_input_generation': 3})
    assert stale.value.payload['code'] == spiral_service.INPUT_GENERATION_STALE
    response = state.update_inputs({'inputs_position': reached, 'expected_input_generation': 0})
    assert response['updating_inputs'] and response['build_id']
    status = _settled_update(state)
    assert status['state'] == 'Idle' and state.session_id == session_id
    assert status['input_generation'] == 1
    assert status['input_digest'] != before['input_digest']
    assert status['current_iteration'] == before['current_iteration']
    position = status['input_position']
    assert position['source'] == 'update' and position['requested'] == reached
    assert position['sequence'] >= reached and position['input_generation'] == 1
    assert status['input_preparation_error'] is None
    # A model-only rebuild afterwards keeps the updated position.
    assert state.rebuild(with_project(request(model_num_flow_stages=3)))['stage'] == 'model'
    rebuilt = settle(state)
    assert rebuilt['input_position']['sequence'] == position['sequence']
    assert state.session._context.config['object_store_input_position'] == position['sequence']


def test_a_refused_service_update_leaves_the_fit(state, hub, device_stubs, no_theta_topology,
                                                 monkeypatch):
    import fit_spiral
    state.initialize(with_project(REQUEST))
    before = settle(state)
    inputs = state.session._context.object_store_inputs

    def refuse(self, inputs):
        raise fit_spiral.InputReplacementRefused('the inputs cannot replace this fit\'s')
    monkeypatch.setattr(fit_spiral.FitContext, 'prepare_input_replacement', refuse)
    response = state.update_inputs({})
    status = _settled_update(state)
    assert status['state'] == 'Idle' and status['input_digest'] == before['input_digest']
    assert status['input_position'] == before['input_position']
    assert state.session._context.object_store_inputs is inputs
    assert state.session._pending_inputs is None
    failure = status['input_preparation_error']
    assert failure['build_id'] == response['build_id'] and failure['phase'] == 'preparing'
    assert failure['recovery_required'] is False
    # The lease's lifecycle answers the command with the refusal, and a
    # repeat of the same command replays it instead of updating again.
    state.lease().claim('client-token', 'claim-1')
    for _ in range(2):
        with pytest.raises(spiral_service.ApiError, match='could not be prepared'):
            state.lease_lifecycle('client-token', 'session_update_inputs',
                                  {'command_id': 'update-1'}, state.update_inputs)
    assert state.status()['input_generation'] == 0


def test_verbs_that_change_the_fit_wait_for_an_input_update(state, hub, device_stubs,
                                                            no_theta_topology, monkeypatch):
    import threading
    state.initialize(with_project(REQUEST))
    settle(state)
    linking, release = threading.Event(), threading.Event()
    prepare = state._prepare_position

    def slow(*args, **kwargs):
        linking.set()
        assert release.wait(30)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(state, '_prepare_position', slow)
    state.update_inputs({})
    assert linking.wait(30)
    try:
        for verb in (lambda: state.run({'iterations': 1}),
                     lambda: state.rebuild(with_project({'inputs': 'reload'})),
                     lambda: state.load_checkpoint({'path': 'absent.ckpt'})):
            with pytest.raises(spiral_service.ApiError, match='input update in progress') as busy:
                verb()
            assert busy.value.payload['code'] == spiral_service.INPUT_UPDATE_IN_PROGRESS
        with pytest.raises(spiral_service.ApiError) as busy:
            state.update_inputs({})
        assert busy.value.payload['code'] == spiral_service.INPUT_UPDATE_BUSY
    finally:
        release.set()
    status = _settled_update(state)
    assert status['state'] == 'Idle' and status['input_preparation_error'] is None
    assert status['input_generation'] == 1
