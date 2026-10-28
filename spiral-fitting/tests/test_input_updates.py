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
