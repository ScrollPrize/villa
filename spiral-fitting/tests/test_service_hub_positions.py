"""A service bound to a hub replica: launch binding, and initialize and
full-input rebuilds that link and pin a hub position (stage 2a).

The runtime is real on the CPU with the device stages stubbed, as in
test_runtime_object_store.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip('vc_object_store')
import vc_object_store as vos  # noqa: E402
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import spiral_service  # noqa: E402
from fit_session import resolve_dataset_root  # noqa: E402
from test_object_store_startup import store_path  # noqa: E402,F401
from test_runtime_object_store import (  # noqa: E402,F401
    RUN, _checkpoint, cpu, device_stubs, settle, write_dataset)


@pytest.fixture()
def hub(store_path, tmp_path):
    """The startup fixture's store as a hub, with an editor's replica."""
    vos.hub_init(store_path, 'ingest')
    editor = str(tmp_path / 'editor.vcobj')
    vos.hub_clone(store_path, editor, 'editor')
    return {'hub': store_path, 'editor': editor,
            'service': str(tmp_path / 'replicas' / 'service.vcobj')}


def _service_state(binding, tmp_path):
    root = write_dataset(tmp_path / 'dataset')
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(root), tmp_path / 'output', tmp_path / 'cache')
    return spiral_service.ServiceState(
        dataset_root=str(root), dataset_resolution=resolution,
        object_store=binding,
        startup_run={'z_begin': 0, 'z_end': 64, 'config': dict(RUN)})


@pytest.fixture()
def state(hub, tmp_path):
    state = _service_state(
        spiral_service.bind_object_store(hub['service'], 'svc', hub['hub']), tmp_path)
    yield state
    state.close()


def _names(path):
    with ObjectStore.open(path, read_only=True) as db:
        return {record['name']: record['uuid'] for kind in osi.KINDS
                for record in db.objects_by_kind(kind)}


def _edit(path, uuid, key, value):
    """An editor's saved edit, pushed: returns the hub position it reached."""
    with ObjectStore.open(path, author='editor') as db:
        db.hub_sync(pull=True, push=False)
        db.acquire_leases([uuid])
        db.set_meta(uuid, key, value)
        report = db.hub_sync(pull=True, push=True)
    assert not report['conflicts']
    return report['published_position']


REQUEST = {'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}


def test_the_binding_clones_and_checks_its_hub(hub, tmp_path):
    binding = spiral_service.bind_object_store(hub['service'], 'svc', hub['hub'])
    assert binding.hosted and Path(hub['service']).is_file()
    assert binding.hub == str(Path(hub['hub']).resolve())
    with ObjectStore.open(hub['editor'], read_only=True) as db:
        assert binding.hub_identity == db.hub_status()['hub_replica']
    # An existing replica binds its own hub, with or without the flag.
    assert spiral_service.bind_object_store(hub['service'], 'svc') == binding
    assert spiral_service.bind_object_store(hub['service'], 'svc', hub['hub']) == binding
    advertised = binding.advertisement()['hub']
    assert advertised['identity'] == binding.hub_identity and advertised['position'] > 0
    # A plain store, or another hub's replica, is refused for this hub.
    plain = str(tmp_path / 'plain.vcobj')
    ObjectStore.create(plain, author='x').close()
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='not a replica'):
        spiral_service.bind_object_store(plain, 'svc', hub['hub'])
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='replicates the hub'):
        spiral_service.bind_object_store(hub['service'], 'svc', plain)
    # An unreachable hub fails startup, not the first update. (A fresh
    # process: this one keeps its hub connection open.)
    os.rename(hub['hub'], hub['hub'] + '.moved')
    try:
        bind = subprocess.run(
            [sys.executable, '-c', 'import sys, spiral_service\n'
             'try:\n spiral_service.bind_object_store(sys.argv[1], "svc")\n'
             'except spiral_service.ObjectStoreBindingError as error:\n'
             ' print(error); sys.exit(3)', hub['service']],
            capture_output=True, text=True, env=os.environ)
        assert bind.returncode == 3, bind.stderr
        assert 'cannot reach the hub' in bind.stdout
    finally:
        os.rename(hub['hub'] + '.moved', hub['hub'])
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='cannot clone'):
        spiral_service.bind_object_store(str(tmp_path / 'new.vcobj'), 'svc',
                                         str(tmp_path / 'absent-hub.vcobj'))


def test_initialize_links_and_pins_the_requested_position(state, hub, device_stubs):
    advertised = state.dataset()
    assert advertised['capabilities']['input_positions'] is True
    assert advertised['object_store']['hub']['identity'] == state.object_store.hub_identity
    walk = _names(hub['editor'])['walk']
    edited = _edit(hub['editor'], walk, 'sampling_group', 'reviewed')
    response = state.initialize({**REQUEST, 'inputs_position': edited})
    assert response['build_id']
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
    # Autosaves name the position as well as the digest. (The stubbed device
    # stages leave no model to save, so the archive itself is a stand-in.)
    session = state.session

    def save_checkpoint(path, completed):
        Path(path).write_bytes(b'PK\x03\x04')
        return path
    session._context.save_checkpoint = save_checkpoint
    autosave = Path(session._write_autosave())
    import json
    metadata = json.loads(autosave.with_name('checkpoint_autosave.json').read_text())
    assert metadata['input_position'] == manifest['position']
    assert metadata['input_digest'] == state.session._context.object_store_inputs.digest
    # The editor reaches N by pulling, links included.
    with ObjectStore.open(hub['editor'], author='editor') as db:
        assert db.hub_sync(pull=True, push=False)['pulled_position'] == position['sequence']
        assert len(db.edges_by_kind('point-surface')) == 3
    # A later edit is not part of the fit; a model rebuild keeps the position.
    _edit(hub['editor'], walk, 'sampling_group', 'later')
    response = state.rebuild({'run': {'z_begin': 0, 'z_end': 64, 'config': {
        **RUN, 'model_num_flow_stages': 3}}})
    assert response['stage'] == 'model'
    assert settle(state)['input_position'] == position
    assert state.session._context.config['object_store_input_position'] == position['sequence']
    # A reload rebuild links and pins a new position including it.
    response = state.rebuild({**REQUEST, 'inputs': 'reload'})
    assert response['inputs'] == 'reload'
    reloaded = settle(state)
    assert reloaded['input_position']['sequence'] > position['sequence']
    [entry] = [e for e in state.session._context.object_store_inputs.manifest['objects']
               if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'later'


def test_a_failed_preparation_keeps_the_resident_fit(state, hub, device_stubs,
                                                      monkeypatch):
    state.initialize(REQUEST)
    first = settle(state)
    session_id, context = state.session_id, state.session._context

    def refuse(*args, **kwargs):
        raise osi.PositionError(['inputs kept changing: no hub position could be '
                                 'linked in 4 attempts; retry'])
    monkeypatch.setattr(osi, 'prepare_position', refuse)
    response = state.rebuild({**REQUEST, 'inputs': 'reload'})
    deadline = time.monotonic() + 30
    while state._building:
        assert time.monotonic() < deadline
        time.sleep(0.02)
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
                              {**REQUEST, 'inputs': 'reload', 'command_id': 'c-1'},
                              state.rebuild)
    assert state.session_id == session_id


def test_a_checkpoint_resumes_at_its_recorded_position(state, hub, tmp_path,
                                                       monkeypatch):
    built = []
    monkeypatch.setattr(state, '_build', lambda *args: built.append(args))
    monkeypatch.setattr(osi, 'prepare_position', lambda *args, **kwargs: pytest.fail(
        'a checkpoint resume does not link'))
    position = {'hub': state.object_store.hub_identity, 'sequence': 7}
    path = _checkpoint(tmp_path / 'fit.ckpt', {}, {
        'object_store': {'position': position}, 'dataset_root': state.dataset_root})
    assert state._checkpoint_position(path) == 7
    other = _checkpoint(tmp_path / 'other.ckpt', {}, {
        'object_store': {'position': {**position, 'hub': 'another-hub'}}})
    assert state._checkpoint_position(other) is None
    paths, run, preview, scroll = state._prepare_session_request(REQUEST)
    paths = paths.__class__(**{**paths.manifest(), 'checkpoint': path})
    deadline = time.monotonic() + 30
    state._begin_build(paths, run, preview, scroll)
    while not built:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    _, _, _, pinned_run, _, _ = built[0]
    assert pinned_run.config['object_store_input_position'] == 7
    assert state.status()['input_position']['source'] == 'checkpoint'


def test_positions_need_a_hub_replica(store_path, tmp_path):
    state = _service_state(spiral_service.bind_object_store(store_path, 'svc'), tmp_path)
    try:
        assert state.dataset()['capabilities']['input_positions'] is False
        with pytest.raises(spiral_service.ApiError, match='not a hub replica'):
            state.initialize({**REQUEST, 'inputs_position': 3})
        with pytest.raises(spiral_service.ApiError, match='non-negative integer'):
            state._inputs_position(-1)
        # A client cannot pin a position through the configuration.
        with pytest.raises(spiral_service.ApiError, match='must not carry'):
            state.initialize({'run': {'z_begin': 0, 'z_end': 64, 'config': {
                **RUN, 'object_store_input_position': 3}}})
    finally:
        state.close()


def test_the_preview_index_records_the_position(tmp_path):
    from preview_index import PublishedPreviewIndex
    index = PublishedPreviewIndex(tmp_path)
    position = {'hub': 'hub-identity', 'sequence': 12}
    index.record('state', manifest_path=tmp_path / 'preview.json', session_id='s',
                 generation=1, input_digest='digest', input_position=position)
    assert index._load()['previews']['state']['input_position'] == position
