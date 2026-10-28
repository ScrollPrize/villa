"""Input updates of a resident fit at an iteration boundary (stage 2b step 16).

A session replaces its inputs with those of a hub position in three phases:
every rank prepares them (touching nothing resident), installs them (the
destructive phase) and publishes them. A refusal leaves the fit as it was; a
failure after the release fails the session. The runtime is real on the CPU
with the device stages stubbed (the theta topology needs the model they
leave out, so it is stubbed too).
"""
import json
import os
import time

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import spiral_runtime  # noqa: E402
from object_store_fixture import put_patch  # noqa: E402
from test_runtime_object_store import (  # noqa: E402,F401
    RUN, cpu, device_stubs, settle, stub_device_stages, write_dataset)
from test_object_store_startup import store_path  # noqa: E402,F401
from test_service_hub_positions import (  # noqa: E402,F401
    REQUEST, _pinned_at_new_position, hub, state)


@pytest.fixture()
def no_theta_topology(monkeypatch):
    import fit_spiral
    monkeypatch.setattr(fit_spiral.FitContext, '_build_theta_crossing_map', lambda self: [])


def _add_patch(path, y=120.0, name='added'):
    def change():
        with ObjectStore.open(path, author='editor') as db:
            db.hub_sync(pull=True, push=False)
            rows, cols = np.mgrid[:8, :8]
            put_patch(db, np.stack([4.0 + rows * 4, np.full(rows.shape, y), 40.0 + cols * 4], axis=-1),
                      (0.25, 0.25), name=name)
            db.hub_sync(pull=True, push=True)
    return change


def test_an_update_replaces_the_inputs_and_keeps_the_iteration(state, hub, device_stubs,
                                                                no_theta_topology):
    state.initialize(REQUEST)
    before = settle(state)
    session = state.session
    patches = session._context.num_verified_patches
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))
    result = session.update_inputs(position)
    status = session.status()
    assert status['state'] == 'Idle'
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
    state.initialize(REQUEST)
    settle(state)
    session = state.session
    context = session._context
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
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))
    session.update_inputs(position)
    assert seen == [True]


def test_a_refused_update_leaves_the_fit(state, hub, device_stubs, no_theta_topology,
                                         monkeypatch):
    import fit_spiral
    state.initialize(REQUEST)
    before = settle(state)
    session = state.session
    inputs = session._context.object_store_inputs
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))

    def refuse(self, inputs):
        raise fit_spiral.InputReplacementRefused('the inputs cannot replace this fit\'s')
    monkeypatch.setattr(fit_spiral.FitContext, 'prepare_input_replacement', refuse)
    with pytest.raises(spiral_runtime.InputUpdateRefused, match='cannot replace'):
        session.update_inputs(position)
    status = session.status()
    assert status['state'] == 'Idle' and status['input_generation'] == 0
    assert status['input_digest'] == before['input_digest']
    assert session._context.object_store_inputs is inputs
    assert session._pending_inputs is None


def test_a_failure_after_release_fails_the_session(state, hub, device_stubs, no_theta_topology,
                                                   monkeypatch):
    import fit_spiral
    state.initialize(REQUEST)
    settle(state)
    session = state.session
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))

    def fail(self, device=None):
        raise MemoryError('CUDA out of memory')
    monkeypatch.setattr(fit_spiral.PatchAtlas, 'materialize', fail)
    with pytest.raises(spiral_runtime.InputUpdateFailed, match='rebuilt or loaded from a checkpoint'):
        session.update_inputs(position)
    deadline = time.monotonic() + 30
    while session.status()['state'] != 'Error':
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_an_update_of_a_busy_session_is_refused_without_a_command(state, hub, device_stubs,
                                                                    no_theta_topology):
    # A run admitted first: the update refuses before admission, and queues
    # no discard (a stray command would break the running step's barrier).
    state.initialize(REQUEST)
    settle(state)
    session = state.session
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))
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


# ---------------------------------------------------------------------------
# The distributed coordinator
# ---------------------------------------------------------------------------

class _Coordinator(spiral_runtime.DistributedInteractiveFitSession):
    """The coordinator's update logic over scripted rank answers."""

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


# ---------------------------------------------------------------------------
# Two ranks (gloo): the collective read agrees and both ranks install it
# ---------------------------------------------------------------------------

def _rank_update(rank, world_size, rendezvous, root, store, config, position, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import fit_spiral
    from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig
    from spiral_runtime import InteractiveFitSession, RendezvousEndpoint
    import object_store_reference as reference
    stub_device_stages(fit_spiral)
    fit_spiral.FitContext._build_theta_crossing_map = lambda self: []
    session = InteractiveFitSession(
        SpiralInputPaths(dataset_root=root, umbilicus=f'{root}/umbilicus.json',
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
    import torch.multiprocessing as mp
    state.initialize(REQUEST)
    first = settle(state)['input_position']['sequence']
    position = _pinned_at_new_position(state, hub, _add_patch(hub['editor']))
    # Each rank builds at the session's first position (read as pinned: the
    # service linked it), then updates to the new one.
    config = {**RUN, 'object_store_path': state.object_store.path, 'object_store_author': 'svc',
              'object_store_input_position': first}
    root = str(write_dataset(tmp_path / 'ranks-dataset'))
    results = [str(tmp_path / f'rank{rank}.json') for rank in range(2)]
    rendezvous = tmp_path / 'rendezvous'
    rendezvous.mkdir()
    mp.spawn(_rank_update, args=(2, str(rendezvous / 'store'), root, state.object_store.path,
                                 config, position, results), nprocs=2)
    outcome = [json.load(open(path)) for path in results]
    assert all('error' not in rank for rank in outcome), outcome
    assert outcome[0]['after'] == outcome[1]['after'] != outcome[0]['before']
    assert outcome[0]['patches'] == outcome[1]['patches']
    assert outcome[0]['iteration'] and outcome[1]['iteration']


# ---------------------------------------------------------------------------
# The service's update route (step 17)
# ---------------------------------------------------------------------------

def _settled_update(state, timeout=60):
    deadline = time.monotonic() + timeout
    while state.status()['input_update'] is not None:
        assert time.monotonic() < deadline, state.status()['input_update']
        time.sleep(0.05)
    return state.status()


def test_the_service_updates_the_resident_fit_at_a_position(state, hub, device_stubs,
                                                             no_theta_topology):
    import spiral_service
    state.initialize(REQUEST)
    before = settle(state)
    assert state.dataset()['capabilities']['input_updates'] is True
    assert state.dataset()['capabilities']['inputs_frozen'] is False
    session_id = state.session_id
    with ObjectStore.open(hub['editor'], author='editor') as db:
        db.hub_sync(pull=True, push=False)
    _add_patch(hub['editor'])()
    with ObjectStore.open(hub['editor'], read_only=True) as db:
        reached = db.hub_status()['position']
    with pytest.raises(spiral_service.ApiError, match='input generation 3'):
        state.update_inputs({'inputs_position': reached, 'expected_input_generation': 3})
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
    response = state.rebuild({'run': {**REQUEST['run'], 'config': {
        **REQUEST['run']['config'], 'model_num_flow_stages': 3}}})
    assert response['stage'] == 'model'
    rebuilt = settle(state)
    assert rebuilt['input_position']['sequence'] == position['sequence']
    assert rebuilt['applied_config']['object_store_input_position'] == position['sequence']
    assert state.session._context.config['object_store_input_position'] == position['sequence']


def test_an_update_with_identical_inputs_republishes_the_manifest(state, hub, device_stubs,
                                                                  no_theta_topology):
    # A commit the adapter does not read moves the position, not the digest:
    # the published manifest must still name the fitted position.
    from test_service_hub_positions import _edit, _names
    state.initialize(REQUEST)
    before = settle(state)

    def published():
        deadline = time.monotonic() + 30
        while True:
            artifact = state.status()['input_manifest_artifact']
            if artifact is not None:
                _, path, _ = state.artifacts.acquire_file(artifact['id'], 'object_store_inputs.json')
                with open(path) as stream:
                    manifest = json.load(stream)['manifest']
                if manifest['position']['sequence'] == state.status()['input_position']['sequence']:
                    return manifest
            assert time.monotonic() < deadline, state.status()['input_manifest_artifact']
            time.sleep(0.02)
    first = published()
    walk = _names(hub['editor'])['walk']
    reached = _edit(hub['editor'], walk, 'review_note', 'not an input')
    state.update_inputs({'inputs_position': reached})
    status = _settled_update(state)
    assert status['input_generation'] == 1 and status['input_digest'] == before['input_digest']
    assert published()['position']['sequence'] == status['input_position']['sequence'] \
        > first['position']['sequence']


def test_a_refused_service_update_leaves_the_fit(state, hub, device_stubs, no_theta_topology,
                                                 monkeypatch):
    import fit_spiral
    state.initialize(REQUEST)
    before = settle(state)

    def refuse(self, inputs):
        raise fit_spiral.InputReplacementRefused('the inputs cannot replace this fit\'s')
    monkeypatch.setattr(fit_spiral.FitContext, 'prepare_input_replacement', refuse)
    response = state.update_inputs({})
    status = _settled_update(state)
    assert status['state'] == 'Idle' and status['input_digest'] == before['input_digest']
    assert status['input_position'] == before['input_position']
    failure = status['input_preparation_error']
    assert failure['build_id'] == response['build_id'] and failure['phase'] == 'preparing'
    assert failure['recovery_required'] is False
    # The lease's lifecycle answers the command with the refusal, and a
    # repeat of the same command replays it instead of updating again.
    import spiral_service
    state.lease().claim('client-token', 'claim-1')
    for _ in range(2):
        with pytest.raises(spiral_service.ApiError, match='could not be prepared'):
            state.lease_lifecycle('client-token', 'session_update_inputs',
                                  {'command_id': 'update-1'}, state.update_inputs)
    assert state.status()['input_generation'] == 0


def test_verbs_that_change_the_fit_wait_for_an_input_update(state, hub, device_stubs,
                                                            no_theta_topology, monkeypatch):
    import threading
    import spiral_service
    state.initialize(REQUEST)
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
                     lambda: state.rebuild({'inputs': 'reload'}),
                     lambda: state.load_checkpoint({'path': 'absent.ckpt'})):
            with pytest.raises(spiral_service.ApiError, match='input update in progress'):
                verb()
    finally:
        release.set()
    status = _settled_update(state)
    assert status['state'] == 'Idle' and status['input_preparation_error'] is None
    assert status['input_generation'] == 1


def test_a_plain_store_has_no_input_updates(store_path, tmp_path):
    import spiral_service
    from test_service_hub_positions import _service_state
    state = _service_state(spiral_service.bind_object_store(store_path, 'svc'), tmp_path)
    try:
        assert state.dataset()['capabilities']['input_updates'] is False
        with pytest.raises(spiral_service.ApiError, match='not a hub replica'):
            state.update_inputs({})
    finally:
        state.close()
