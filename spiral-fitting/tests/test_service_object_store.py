"""The Spiral service's object-store inputs: the launch binding, frozen inputs
and the session lease, store-backed discovery and input provenance.

The resident runtime is faked where a test needs one; stores are written
with object_store_fixture.
"""
import copy
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import spiral_service  # noqa: E402
from config import Config  # noqa: E402
from fit_session import (SessionState, SpiralInputPaths,  # noqa: E402
                         SpiralRunConfig, resolve_dataset_root)
from service_http import ApiError  # noqa: E402

# The dataset has no Lasagna stores or shell, so runs turn off the losses
# that would need them.
NO_DENSE = {'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}


@pytest.fixture()
def store(tmp_path):
    path = tmp_path / 'inputs.vcobj'
    with ObjectStore.create(str(path), author='ingest', hostname='host'):
        pass
    return str(path)


@pytest.fixture()
def dataset(tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'spiral-scroll.json').write_text(json.dumps({
        'schema_version': 1, 'name': 's1', 'voxel_size_um': 9.6,
        'spiral_outward_sense': 'CW'}))
    (root / 'umbilicus.json').write_text('{}')
    output = tmp_path / 'output'
    output.mkdir()
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(root), output, tmp_path / 'cache')
    return SimpleNamespace(root=root, output=output, resolution=resolution)


def service(dataset, store, **kwargs):
    return spiral_service.ServiceState(
        dataset_root=str(dataset.root), dataset_resolution=dataset.resolution,
        object_store=spiral_service.bind_object_store(store, 'spiral-test'),
        startup_run={'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)},
        **kwargs)


# ---------------------------------------------------------------------------
# Launch binding (step 3)
# ---------------------------------------------------------------------------

def test_binding_resolves_the_store_once_and_records_its_uuid(store, tmp_path,
                                                              monkeypatch):
    monkeypatch.chdir(tmp_path)
    binding = spiral_service.bind_object_store('inputs.vcobj', 'svc')
    with ObjectStore.open(store, read_only=True) as db:
        assert binding.uuid == db.uuid
    assert binding.path == str(Path(store).resolve())
    assert binding.config() == {'object_store_path': binding.path,
                                'object_store_author': 'svc',
                                'object_store_input_position': None}
    assert not binding.hosted
    assert spiral_service.bind_object_store(store).author.startswith(
        'spiral-service@')


def test_binding_fails_clearly_without_a_store_or_the_module(tmp_path,
                                                             monkeypatch):
    with pytest.raises(spiral_service.ObjectStoreBindingError,
                       match='no object store at'):
        spiral_service.bind_object_store(tmp_path / 'missing.vcobj')
    garbage = tmp_path / 'garbage.vcobj'
    garbage.write_text('not a store')
    with pytest.raises(spiral_service.ObjectStoreBindingError,
                       match='cannot open the object store'):
        spiral_service.bind_object_store(garbage)
    monkeypatch.setitem(__import__('sys').modules, 'vc_object_store', None)
    with pytest.raises(spiral_service.ObjectStoreBindingError,
                       match='vc_object_store Python module is unavailable'):
        spiral_service.bind_object_store(garbage)


def _launch(dataset, *extra):
    return ['--dataset', str(dataset.root), '--output',
            str(dataset.output / 'svc'), '--cache', str(dataset.output / 'cache'),
            '--api-key-file', str(dataset.output / 'key'), *extra]


def test_launch_requires_a_usable_object_store(dataset, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        spiral_service.main(_launch(dataset))
    assert raised.value.code == 2
    assert '--object-store' in capsys.readouterr().err
    assert spiral_service.main(_launch(
        dataset, '--object-store', str(tmp_path / 'missing.vcobj'))) == 2
    assert 'Refusing to start: no object store at' in capsys.readouterr().err


def test_launch_config_must_not_set_the_binding(dataset, store, capsys):
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(SystemExit):
            spiral_service.main(_launch(
                dataset, '--object-store', store,
                '--config', json.dumps({key: '/elsewhere.vcobj'})))
        assert f'--config must not set {key}' in capsys.readouterr().err


def test_session_requests_refuse_client_bindings_and_receive_the_service_one(
        dataset, store):
    state = service(dataset, store)
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(ApiError) as raised:
            state._prepare_session_request(
                {'run': {'z_begin': 0, 'z_end': 10,
                         'config': {**NO_DENSE, key: 'chosen'}}})
        assert raised.value.status == 400
        assert raised.value.details == [{
            'field': f'run.config.{key}',
            'message': "Owned by the service's --object-store binding"}]
    # Unset values (a complete default configuration) and the binding's own
    # values (an echoed session request) are not a choice.
    echoed = {**NO_DENSE, 'object_store_path': None,
              'object_store_author': 'spiral-test'}
    for request in (state.startup_session_request(),
                    {'run': {'z_begin': 0, 'z_end': 10, 'config': echoed}}):
        _, run, _, _ = state._prepare_session_request(request)
        assert run.config['object_store_path'] == state.object_store.path
        assert run.config['object_store_author'] == 'spiral-test'


def test_configuration_marks_the_binding_service_owned(dataset, store):
    catalog = service(dataset, store).configuration_catalog()
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        assert catalog['schema']['fields'][key]['service_owned'] is True
    assert catalog['defaults']['object_store_path'] == str(Path(store).resolve())
    assert catalog['defaults']['object_store_author'] == 'spiral-test'
    assert not any(spec.get('service_owned')
                   for key, spec in catalog['schema']['fields'].items()
                   if key not in spiral_service.SERVICE_OWNED_CONFIG_KEYS)


def test_the_binding_wins_over_a_resumed_checkpoints_store(dataset, store,
                                                           monkeypatch):
    import checkpoint_io
    from spiral_progress import NullProgressReporter
    from spiral_runtime import InteractiveFitSession

    stored = Config().as_dict()
    stored.update(object_store_path='/written/against/another.vcobj',
                  object_store_author='someone-else')
    monkeypatch.setattr(checkpoint_io, 'load_checkpoint_cpu',
                        lambda path: {'cfg': copy.deepcopy(stored)})
    state = service(dataset, store)
    checkpoint = dataset.output / 'resume.ckpt'
    checkpoint.write_bytes(b'PK\x03\x04')
    _, run, _, _ = state._prepare_session_request(state.startup_session_request())
    # The checkpoint's model-shaping keys are untouched: the binding is no
    # contradiction to refuse.
    state._reject_overrides_the_checkpoint_contradicts(str(checkpoint), run.config)
    session = InteractiveFitSession.__new__(InteractiveFitSession)
    session.paths = SpiralInputPaths(checkpoint=str(checkpoint))
    session.run_config = run
    session._condition = threading.Condition()
    session._warnings = []
    session._progress_reporter = lambda: NullProgressReporter()
    session._publish_status = lambda: None
    config = session._resolve_session_config(1)
    assert config['object_store_path'] == state.object_store.path
    assert config['object_store_author'] == 'spiral-test'


def test_a_run_keeps_the_resident_binding(dataset, store):
    state = service(dataset, store)
    applied = Config().as_dict()
    applied.update(state.object_store.config())

    class Resident:
        def status(self):
            return {'state': SessionState.Idle, 'applied_config': applied,
                    'current_iteration': 0}

        def run(self, iterations, **kwargs):
            self.kwargs = kwargs
            return iterations

    resident = state.session = Resident()
    state.session_id = 'spiral-1'
    state.session_paths = SpiralInputPaths(output_directory=str(dataset.output))
    state.session_request = {'run': SpiralRunConfig(0, 10).manifest()}
    configuration = Config().as_dict()
    configuration['object_store_path'] = None
    state.run({'iterations': 1, 'expected_session_revision': 0,
               'configuration': configuration,
               'dt_loss_schedule': {'enabled': False, 'last_fraction': 0.0}})
    assert 'object_store_path' not in resident.kwargs['run_config']


# ---------------------------------------------------------------------------
# Session lease and frozen inputs (step 4)
# ---------------------------------------------------------------------------

class Http:
    """The service over real HTTP, as a client sees it."""

    def __init__(self, state):
        self.state = state
        self.server = spiral_service.SpiralServer(('127.0.0.1', 0), ['key'], state)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.commands = 0

    def __call__(self, method, path, body=None, token=None):
        import urllib.error
        import urllib.request
        request = urllib.request.Request(
            f'http://127.0.0.1:{self.server.server_port}{path}', method=method)
        request.add_header('Authorization', 'Bearer key')
        if token is not None:
            request.add_header(spiral_service.LEASE_TOKEN_HEADER, token)
        data = None if body is None else json.dumps(body).encode()
        try:
            with urllib.request.urlopen(request, data=data, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def command(self, path, body=None, token=None):
        self.commands += 1
        return self('POST', path, {**(body or {}), 'command_id': f'c{self.commands}'},
                    token)

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@pytest.fixture()
def http(dataset, store, monkeypatch):
    import spiral_runtime
    from service_lifecycle_transcript import FakeRuntime
    built = []

    def create_session(*args, **kwargs):
        built.append(kwargs)
        return FakeRuntime()
    monkeypatch.setattr(spiral_runtime, 'create_session', create_session)
    client = Http(service(dataset, store))
    client.built = built
    yield client
    client.close()
    client.state.close()


def initialize_body():
    return {'run': {'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)}}


def test_one_client_holds_the_lease_and_gates_mutations(http):
    status, refused = http.command('/session/stop', token='owner')
    assert status == 403 and refused['error'] == 'Claim the session lease first'
    status, claimed = http('POST', '/session/lease/claim', {'command_id': 'a'}, 'owner')
    assert status == 200 and claimed['owned']
    assert http('GET', '/session/status')[1]['lease_id'] == claimed['lease_id']
    # Reconnecting with the same token keeps the lease; another token cannot
    # take it or act through it.
    again = http('POST', '/session/lease/claim', {'command_id': 'b'}, 'owner')[1]
    assert again['lease_id'] == claimed['lease_id']
    assert http('POST', '/session/lease/claim', {'command_id': 'c'}, 'other')[0] == 403
    status, refused = http.command('/session/initialize', initialize_body(), 'other')
    assert (status, refused['error']) == (403, 'This client does not own the session lease')
    status, initialized = http.command('/session/initialize', initialize_body(), 'owner')
    assert status == 200 and initialized['state'] == 'Idle'


def test_a_repeated_initialize_command_builds_once(http):
    http('POST', '/session/lease/claim', {'command_id': 'a'}, 'owner')
    body = {**initialize_body(), 'command_id': 'init-1'}
    first = http('POST', '/session/initialize', body, 'owner')
    second = http('POST', '/session/initialize', body, 'owner')
    assert first[0] == second[0] == 200
    assert len(http.built) == 1
    # Nothing binds inputs into the build: they come from the store.
    assert 'input_bindings' not in http.built[0]


@pytest.mark.parametrize('method,path', [
    ('GET', '/session/input-catalog'), ('GET', '/session/input-content/0123-ab/1'),
    ('POST', '/session/input-changes'), ('POST', '/session/apply-inputs'),
    ('POST', '/session/commit-inputs'), ('POST', '/session/editing/claim')])
def test_the_input_editing_surface_is_gone(http, method, path):
    http('POST', '/session/lease/claim', {'command_id': 'a'}, 'owner')
    body = {'command_id': 'x'} if method == 'POST' else None
    status, payload = http(method, path, body, 'owner')
    assert (status, payload['error']) == (404, 'Unknown endpoint')


def test_releasing_the_lease_tears_the_session_down(http):
    http('POST', '/session/lease/claim', {'command_id': 'a'}, 'owner')
    http.command('/session/initialize', initialize_body(), 'owner')
    session, lease = http.state.session, http.state.session_lease
    assert http('POST', '/session/lease/release', {'command_id': 'x'}, 'other')[0] == 403
    status, released = http('POST', '/session/lease/release', {'command_id': 'r'}, 'owner')
    assert (status, released) == (200, {'released': True})
    assert session.state == SessionState.Closing
    assert http.state.session_lease is None and http.state.session is None
    assert http('GET', '/session/status')[1]['state'] == 'Uninitialized'
    # The lease's file lock is free for the next owner.
    fresh = http('POST', '/session/lease/claim', {'command_id': 'b'}, 'next')[1]
    assert fresh['lease_id'] != lease.id


@pytest.mark.parametrize('user', ['request', 'background', 'command'])
def test_release_waits_for_every_user_of_the_service(dataset, store, user):
    from concurrent.futures import ThreadPoolExecutor
    state = service(dataset, store)
    lease = state.lease()
    lease.claim('owner', 'claim')
    entered, finish = threading.Event(), threading.Event()

    def work(*_):
        entered.set()
        assert finish.wait(5)

    def request():
        with state.service_use():
            work()
    with ThreadPoolExecutor(2) as pool:
        if user == 'background':
            state._start_background(target=work, name='test-construction')
        elif user == 'command':
            pool.submit(lease.coordinator.execute, 'slow', 'test', {}, work)
        else:
            pool.submit(request)
        assert entered.wait(5)
        releasing = pool.submit(state.release_lease, 'owner', 'release')
        with state._service_condition:
            assert state._service_condition.wait_for(lambda: state._closing, timeout=5)
        assert not releasing.done()
        with pytest.raises(ApiError):
            with state.service_use():
                pass
        finish.set()
        assert releasing.result(5) == {'released': True}
    assert state.session_lease is None
    state.close()


def test_a_fitter_that_will_not_close_keeps_the_lease_for_a_retry(http, monkeypatch):
    http('POST', '/session/lease/claim', {'command_id': 'a'}, 'owner')
    http.command('/session/initialize', initialize_body(), 'owner')
    session = http.state.session
    close = session.close

    def timeout(timeout=15.0):
        raise TimeoutError('still reading inputs')
    monkeypatch.setattr(session, 'close', timeout)
    with pytest.raises(TimeoutError):
        http.state.release_lease('owner', 'release')
    assert http.state.session_lease is not None
    monkeypatch.setattr(session, 'close', close)
    assert http.state.release_lease('owner', 'release') == {'released': True}
    assert http.state.session_lease is None


def test_release_during_construction_closes_the_installed_fitter(dataset, store,
                                                                 monkeypatch):
    import sys
    import types
    state = service(dataset, store)
    state.lease().claim('owner', 'claim')
    entered, finish, closed = threading.Event(), threading.Event(), threading.Event()

    def create(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return types.SimpleNamespace(close=closed.set)
    monkeypatch.setitem(sys.modules, 'spiral_runtime',
                        types.SimpleNamespace(create_session=create))
    state._start_background(target=state._build,
                            args=(state.session_id, None, None, None, None, None),
                            name='test-build')
    assert entered.wait(5)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as pool:
        releasing = pool.submit(state.release_lease, 'owner', 'release')
        with state._service_condition:
            assert state._service_condition.wait_for(lambda: state._closing, timeout=5)
        finish.set()
        releasing.result(5)
    assert closed.is_set() and state.session is None
    state.close()


# ---------------------------------------------------------------------------
# Discovery of the bound store's inputs (step 5)
# ---------------------------------------------------------------------------

@pytest.fixture()
def filled_store(tmp_path):
    """Two patches, an absolute PCL attached to one, a fiber at scale 2."""
    import numpy as np
    from object_store_fixture import (frame, pcl_patch_edge, put_fiber,
                                      put_patch, put_pcl)
    path = tmp_path / 'filled.vcobj'
    rows, cols = np.mgrid[:6, :6]
    grid = np.stack([10.0 + rows, np.full(rows.shape, 50.0), 20.0 + cols], -1)
    with ObjectStore.create(str(path), author='ingest', hostname='host') as db:
        near = put_patch(db, grid, name='near')
        put_patch(db, grid + [0, 30, 0], name='far')
        pcl = put_pcl(db, {7: ([11, 50, 21], 3.0)}, absolute=True, name='abs')
        pcl_patch_edge(db, pcl, 7, near, (1.0, 1.0))
        put_fiber(db, [[10, 40, 20], [12, 40, 22], [14, 40, 24]], [0, 2],
                  coordinates=frame(2), name='fiber')
    return str(path)


def test_dataset_advertises_the_binding_and_its_selection(dataset, filled_store):
    state = service(dataset, filled_store)
    advertised = state.dataset()
    assert advertised['object_store'] == {
        'uuid': state.object_store.uuid, 'path': state.object_store.path,
        'author': 'spiral-test', 'hub': None}
    assert advertised['capabilities'] == {
        'input_source': 'object_store', 'inputs_frozen': True,
        'input_reload_rebuild': True, 'input_positions': False, 'input_updates': False}
    # The dataset no longer resolves or advertises patch, fiber or PCL files.
    assert 'pcl_inputs' not in advertised
    assert not {'verified_patches', 'fibers'} & set(advertised['resolved'])
    inputs = advertised['inputs']
    assert inputs['ok'] and inputs['volume_uuid'] == '20260411134726'
    assert inputs['volume_source'] == 'inferred'
    assert {kind: entry['count'] for kind, entry in inputs['selection'].items()} == {
        'surface': 2, 'point_collection': 1, 'fiber': 1}
    assert inputs['selection']['fiber']['scales'] == [[2.0, 1]]
    assert inputs['selection']['surface']['description'] == 'all active'
    assert inputs['relationships'] == {'pcl_patch': 1}
    by_name = {entry['name']: entry for entry in inputs['objects']}
    assert by_name['abs']['pcl_role'] == 'absolute'
    assert 'pcl_role' not in by_name['near']
    with ObjectStore.open(filled_store, read_only=True) as db:
        geometry = db.current_geometry(by_name['fiber']['uuid'])['uuid']
    assert by_name['fiber']['geometry_revision'] == geometry
    assert by_name['fiber']['scale_factor'] == 2.0
    assert all(entry['metadata_sha256'] and entry['object_revision']
               for entry in inputs['objects'])
    # Discovery writes nothing: no refresh ran.
    with ObjectStore.open(filled_store, read_only=True) as db:
        assert len(db.edges_by_kind('point-surface')) == 1


def test_discovery_follows_store_changes_and_reports_refusals(dataset, filled_store):
    state = service(dataset, filled_store)
    first = state.startup_inputs()
    assert state.startup_inputs() is first
    with ObjectStore.open(filled_store, author='editor') as db:
        uuid = db.select_objects(object_kind='surface', fields=['uuid'])[0]['uuid']
        db.set_meta(uuid, 'coordinates', {'volume_uuid': 'another-volume',
                                          'scale_factor': 1})
    refused = state.startup_inputs()
    assert refused['ok'] is False
    assert 'selected objects declare different volumes' in refused['problems'][0]


def test_validation_requires_no_patch_fiber_or_pcl_files(dataset, store):
    state = service(dataset, store)
    assert not (dataset.root / 'verified_patches').exists()
    paths, run, _, _ = state._prepare_session_request(state.startup_session_request())
    assert not {'verified_patches', 'fibers', 'pcls'} & set(paths.manifest())


def test_starting_the_service_and_discovery_do_not_import_torch(dataset, filled_store,
                                                                tmp_path):
    import os
    import socket
    import subprocess
    import sys
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    script = f"""
import json, os, signal, sys, threading, time, urllib.request
import spiral_service

def probe():
    for _ in range(200):
        try:
            request = urllib.request.Request(
                'http://127.0.0.1:{port}/dataset',
                headers={{'Authorization': 'Bearer ' + open({str(tmp_path / 'key')!r}).read().strip()}})
            with urllib.request.urlopen(request, timeout=10) as response:
                dataset = json.loads(response.read())
            break
        except Exception:
            time.sleep(0.05)
    print('RESULT', json.dumps({{'torch': 'torch' in sys.modules,
                               'ok': dataset['inputs']['ok']}}), file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

threading.Thread(target=probe, daemon=True).start()
sys.exit(spiral_service.main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, '-c', script, '--port', str(port),
         '--api-key-file', str(tmp_path / 'key'), '--dataset', str(dataset.root),
         '--output', str(dataset.output / 'svc'), '--object-store', filled_store],
        cwd=str(Path(spiral_service.__file__).parent), capture_output=True,
        text=True, timeout=120, env=dict(os.environ))
    line = next(line for line in result.stderr.splitlines() if line.startswith('RESULT'))
    assert json.loads(line[len('RESULT '):]) == {'torch': False, 'ok': True}
