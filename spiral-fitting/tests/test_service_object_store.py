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
                         SpiralRunConfig)
from service_http import ApiError  # noqa: E402
from service_project import UMBILICUS, project, with_project  # noqa: E402
from object_store_fixture import VOLUME  # noqa: E402
from test_object_store_startup import store_path  # noqa: E402,F401

# The project has no Lasagna stores and the store no shell, so runs turn off
# the losses that would need them.
NO_DENSE = {'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}


@pytest.fixture()
def store(store_path):
    # Populated: a service links and pins a position of it before a build.
    return store_path


@pytest.fixture()
def host(tmp_path):
    # The service's host paths: no dataset directory, no spiral-scroll.json.
    output = tmp_path / 'output'
    output.mkdir()
    return SimpleNamespace(output=output, cache=tmp_path / 'cache',
                           bindings=spiral_service.bind_service_inputs(
                               output, tmp_path / 'cache'))


def service(host, store, bindings=None, **kwargs):
    return spiral_service.ServiceState(
        bindings=bindings or host.bindings,
        object_store=spiral_service.bind_object_store(store, 'spiral-test'),
        startup_run={'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)},
        **kwargs)


# ---------------------------------------------------------------------------
# Launch binding (step 3)
# ---------------------------------------------------------------------------

def test_binding_resolves_the_store_once_and_records_its_uuid(store, tmp_path,
                                                              monkeypatch):
    monkeypatch.chdir(Path(store).parent)
    binding = spiral_service.bind_object_store(Path(store).name, 'svc')
    with ObjectStore.open(store, read_only=True) as db:
        assert binding.uuid == db.uuid
    assert binding.path == str(Path(store).resolve())
    assert binding.config() == {'object_store_path': binding.path,
                                'object_store_author': 'svc',
                                'object_store_input_position': None}
    assert not binding.hosted
    # Shared in place: the store records positions of its own from now on.
    with ObjectStore.open(store, read_only=True) as db:
        assert binding.positions == db.store_positions()['store']
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


def _launch(host, *extra):
    return ['--output',
            str(host.output / 'svc'), '--cache', str(host.output / 'cache'),
            '--api-key-file', str(host.output / 'key'), *extra]


def test_launch_requires_a_usable_object_store(host, tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        spiral_service.main(_launch(host))
    assert raised.value.code == 2
    assert '--object-store' in capsys.readouterr().err
    assert spiral_service.main(_launch(
        host, '--object-store', str(tmp_path / 'missing.vcobj'))) == 2
    assert 'Refusing to start: no object store at' in capsys.readouterr().err


def test_launch_config_must_not_set_the_binding(host, store, capsys):
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(SystemExit):
            spiral_service.main(_launch(
                host, '--object-store', store,
                '--config', json.dumps({key: '/elsewhere.vcobj'})))
        assert f'--config must not set {key}' in capsys.readouterr().err


def test_session_requests_refuse_client_bindings_and_receive_the_service_one(
        host, store):
    state = service(host, store)
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(ApiError) as raised:
            state._prepare_session_request(with_project(
                {'run': {'z_begin': 0, 'z_end': 10,
                         'config': {**NO_DENSE, key: 'chosen'}}}))
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
        _, run, _, _ = state._prepare_session_request(with_project(request))
        assert run.config['object_store_path'] == state.object_store.path
        assert run.config['object_store_author'] == 'spiral-test'


def test_configuration_marks_the_binding_service_owned(host, store):
    catalog = service(host, store).configuration_catalog()
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        assert catalog['schema']['fields'][key]['service_owned'] is True
    assert catalog['defaults']['object_store_path'] == str(Path(store).resolve())
    assert catalog['defaults']['object_store_author'] == 'spiral-test'
    assert not any(spec.get('service_owned')
                   for key, spec in catalog['schema']['fields'].items()
                   if key not in spiral_service.SERVICE_OWNED_CONFIG_KEYS)


def test_the_binding_wins_over_a_resumed_checkpoints_store(host, store,
                                                           monkeypatch):
    import checkpoint_io
    from spiral_progress import NullProgressReporter
    from spiral_runtime import InteractiveFitSession

    stored = Config().as_dict()
    stored.update(object_store_path='/written/against/another.vcobj',
                  object_store_author='someone-else')
    monkeypatch.setattr(checkpoint_io, 'load_checkpoint_cpu',
                        lambda path: {'cfg': copy.deepcopy(stored)})
    state = service(host, store)
    checkpoint = host.output / 'resume.ckpt'
    checkpoint.write_bytes(b'PK\x03\x04')
    _, run, _, _ = state._prepare_session_request(with_project(state.startup_session_request()))
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


def test_a_run_keeps_the_resident_binding(host, store):
    state = service(host, store)
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
    state.session_paths = SpiralInputPaths(output_directory=str(host.output))
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
def http(host, store, monkeypatch):
    import spiral_runtime
    from service_lifecycle_transcript import FakeRuntime
    built = []

    def create_session(*args, **kwargs):
        built.append(kwargs)
        return FakeRuntime()
    monkeypatch.setattr(spiral_runtime, 'create_session', create_session)
    client = Http(service(host, store))
    client.built = built
    yield client
    client.close()
    client.state.close()


def initialize_body():
    return with_project({'run': {'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)}})


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
def test_release_waits_for_every_user_of_the_service(host, store, user):
    from concurrent.futures import ThreadPoolExecutor
    state = service(host, store)
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


def test_release_during_construction_closes_the_installed_fitter(host, store,
                                                                 monkeypatch):
    import sys
    import types
    state = service(host, store)
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


def test_bindings_advertise_the_store_and_its_selection(host, filled_store):
    state = service(host, filled_store)
    advertised = state.binding_advertisement()
    with ObjectStore.open(filled_store, read_only=True) as db:
        positions = db.store_positions()
    assert advertised['object_store'] == {
        'uuid': state.object_store.uuid, 'path': state.object_store.path,
        'author': 'spiral-test', 'hub': None,
        'positions': {'identity': positions['store'], 'position': positions['position'],
                      'position_error': None}}
    assert advertised['capabilities'] == {
        'input_source': 'object_store', 'inputs_frozen': False,
        'input_reload_rebuild': True, 'input_positions': True, 'input_updates': True,
        'winding_inference': False, 'build_missing_lasagna_packs': False,
        'commits': True}
    # Nothing advertises patch, fiber or PCL files, or a host.
    assert 'pcl_inputs' not in advertised
    assert not {'root', 'resolved', 'detected_checkpoints', 'scroll_spec'} & set(advertised)
    inputs = advertised['inputs']
    assert inputs['ok'] and inputs['volume_uuid'] == '20260411134726'
    assert inputs['volume_source'] == 'inferred'
    assert {kind: entry['count'] for kind, entry in inputs['selection'].items()} == {
        'surface': 2, 'point_collection': 1, 'fiber': 1, 'track': 0}
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


def test_discovery_follows_store_changes_and_reports_refusals(host, filled_store):
    state = service(host, filled_store)
    first = state.startup_inputs()
    assert state.startup_inputs() is first
    with ObjectStore.open(filled_store, author='editor') as db:
        uuid = db.select_objects(object_kind='surface', fields=['uuid'])[0]['uuid']
        db.set_meta(uuid, 'coordinates', {'volume_uuid': 'another-volume',
                                          'scale_factor': 1})
    refused = state.startup_inputs()
    assert refused['ok'] is False
    assert 'selected objects declare different volumes' in refused['problems'][0]


def test_validation_requires_no_patch_fiber_or_pcl_files(host, store):
    state = service(host, store)
    paths, run, _, _ = state._prepare_session_request(with_project(state.startup_session_request()))
    assert not {'verified_patches', 'fibers', 'pcls'} & set(paths.manifest())


def test_starting_the_service_and_discovery_do_not_import_torch(host, filled_store,
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
                'http://127.0.0.1:{port}/bindings',
                headers={{'Authorization': 'Bearer ' + open({str(tmp_path / 'key')!r}).read().strip()}})
            with urllib.request.urlopen(request, timeout=10) as response:
                bindings = json.loads(response.read())
            break
        except Exception:
            time.sleep(0.05)
    print('RESULT', json.dumps({{'torch': 'torch' in sys.modules,
                               'ok': bindings['inputs']['ok']}}), file=sys.stderr, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)

threading.Thread(target=probe, daemon=True).start()
sys.exit(spiral_service.main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, '-c', script, '--port', str(port),
         '--api-key-file', str(tmp_path / 'key'),
         '--output', str(host.output / 'svc'), '--object-store', filled_store,
         # The store has no outer shell, so the shell losses are off.
         '--config', json.dumps(NO_DENSE)],
        cwd=str(Path(spiral_service.__file__).parent), capture_output=True,
        text=True, timeout=120, env=dict(os.environ))
    line = next(line for line in result.stderr.splitlines() if line.startswith('RESULT'))
    assert json.loads(line[len('RESULT '):]) == {'torch': False, 'ok': True}


# ---------------------------------------------------------------------------
# The VC3D project's inputs
# ---------------------------------------------------------------------------

def _built(state, request, timeout=60):
    """Initialize with ``request``, capturing what the build is given
    instead of building: ``(paths, scroll)``, or the preparation error."""
    import time
    built = []
    state._build = lambda session_id, previous, paths, run, preview, scroll: (
        built.append((paths, scroll)))
    state.initialize(request)
    deadline = time.monotonic() + timeout
    while state._building and not built:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    return built[0] if built else state.status()['input_preparation_error']


def test_a_request_must_carry_its_project(host, store):
    state = service(host, store)
    with pytest.raises(ApiError) as refused:
        state.initialize(initialize_body() | {'project': None})
    assert refused.value.status == 400
    assert [detail['field'] for detail in refused.value.details] == ['project']


def test_the_build_reads_the_project_of_its_fit_volume(host, store):
    state = service(host, store)
    paths, scroll = _built(state, initialize_body())
    # The fit volume the store's inputs name selects the project's volume:
    # its umbilicus, written where the fit reads it, and the scroll.
    from umbilicus import json_umbilicus_z_to_yx
    assert Path(paths.umbilicus).parent == host.output / 'project-inputs'
    assert json_umbilicus_z_to_yx(paths.umbilicus)(32.0).tolist() == [32.0, 32.0]
    assert (scroll.name, scroll.voxel_size_um, scroll.spiral_outward_sense) == (
        's1', 9.6, 'CW')
    request = state.session_request
    assert request['project'] == {
        'name': 's1', 'volume': VOLUME, 'voxel_size_um': 9.6,
        'z_direction_is_top_to_bottom': False, 'left_handed_coordinates': False,
        'umbilicus': {'uuid': UMBILICUS['uuid'], 'revision': UMBILICUS['revision'],
                      'points': 2}}
    assert request['scroll']['umbilicus_coordinate_scale'] == 1.0


def test_the_scroll_is_the_projects_and_the_sense_a_fit_setting(host, store):
    facts = {'voxel_size_um': 7.91, 'base_shape_zyx': [100, 64, 64],
             'z_direction_is_top_to_bottom': None, 'left_handed_coordinates': None}
    body = initialize_body() | {'project': project(name='PHerc', **facts)}
    failure = _built(service(host, store), body)
    assert 'set the fit setting spiral_outward_sense' in failure['message']
    body['run']['config']['spiral_outward_sense'] = 'ACW'
    _, scroll = _built(service(host, store), body)
    assert (scroll.name, scroll.voxel_size_um, scroll.base_shape_zyx,
            scroll.spiral_outward_sense) == ('PHerc', 7.91, (100, 64, 64), 'ACW')
    # A volume whose catalog orientation decides the sense refuses another.
    failure = _built(service(host, store), initialize_body() | {
        'run': {**body['run'], 'config': {**body['run']['config'],
                                          'spiral_outward_sense': 'ACW'}}})
    assert 'contradicts the catalog' in failure['message']


def test_the_lasagna_dataset_is_the_projects(host, store, tmp_path):
    from lasagna_source import load_lasagna_source, url_identity
    # A remote dataset is identified by its URL, which VC3D sends alone.
    lasagna = {'location': 'https://data.example/p.lasagna.json', 'group': '3'}
    for key in ('normal_x', 'normal_y', 'gradient_magnitude'):
        lasagna[key] = f'https://data.example/{key}.ome.zarr'
    body = initialize_body() | {'project': project(lasagna=lasagna, lasagna_scale=2)}
    paths, scroll = _built(service(host, store), body)
    # The fit reads the project's dataset by identity; its stores need not
    # be readable here.
    source = load_lasagna_source(paths.lasagna)
    assert (source.identity, source.group, source.location) == (
        url_identity(lasagna['location']), '3', lasagna['location'])
    assert source.stores['normal_x'] == lasagna['normal_x']
    assert scroll.lasagna_scale == 2

    # A fit that needs normals reads their pack from the service's cache,
    # which never builds one: without it the build is refused up front.
    needs_normals = initialize_body() | {'project': project(lasagna=lasagna, lasagna_scale=2)}
    needs_normals['run']['config']['loss_weight_dense_normals'] = 1.0
    with pytest.raises(ApiError) as refused:
        service(host, store).initialize(needs_normals)
    [detail] = refused.value.details
    assert detail['field'] == 'lasagna' and 'not in the cache' in detail['message']
    pack = Path(source.pack_directory(paths.cache_directory, 'normals'))
    pack.mkdir(parents=True)
    (pack / 'meta.json').write_text('{}')
    paths, _ = _built(service(host, store), needs_normals)
    assert load_lasagna_source(paths.lasagna) == source

    # A fit that needs normals the project cannot give is refused before
    # anything is linked.
    body = initialize_body()
    body['run']['config']['loss_weight_dense_normals'] = 1.0
    with pytest.raises(ApiError) as refused:
        service(host, store).initialize(body)
    assert {(detail['field'], 'selects no Lasagna dataset' in detail['message'])
            for detail in refused.value.details} == {('lasagna', True)}

    # A local dataset's identity is VC3D's; a remote one's cannot be.
    local = {**lasagna, 'location': '/data/p.lasagna.json', 'identity': 'run-1'}
    paths, _ = _built(service(host, store), initialize_body() | {
        'project': project(lasagna=local, lasagna_scale=2)})
    assert load_lasagna_source(paths.lasagna).identity == 'run-1'
    with pytest.raises(ApiError, match='identified by its URL'):
        service(host, store).initialize(initialize_body() | {
            'project': project(lasagna={**lasagna, 'identity': 'run-1'}, lasagna_scale=2)})


def test_fiber_directions_are_the_project_datasets_cached_samples(host, store):
    from lasagna_source import fiber_directions_path, url_identity
    fibers = {'location': 'https://data.example/f.lasagna.json'}
    body = initialize_body() | {'project': project(fiber_inference=fibers, fiber_scale=4)}
    body['run']['config'].update(input_use_fiber_directions=True,
                                 loss_weight_fiber_directions=1.0)
    # The loss needs them: without the cache's samples the build is refused.
    failure = _built(service(host, store), body)
    assert 'no fiber directions of https://data.example/f.lasagna.json' in failure['message']
    # Enabled without the loss, they are read when the project can give them.
    optional = initialize_body() | {'project': project()}
    optional['run']['config'].update(input_use_fiber_directions=True)
    paths, _ = _built(service(host, store), optional)
    assert paths.fiber_directions == ''

    cached = Path(fiber_directions_path(paths.cache_directory,
                                        url_identity(fibers['location']), 4))
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'')
    paths, _ = _built(service(host, store), body)
    assert paths.fiber_directions == str(cached)
    optional['project'] = project(fiber_inference=fibers, fiber_scale=4)
    assert _built(service(host, store), optional)[0].fiber_directions == str(cached)


def test_a_local_service_extracts_missing_fiber_directions(host, store, tmp_path):
    from fiber_direction_samples import load_fiber_direction_samples
    from lasagna_source import fiber_directions_path
    from test_fiber_direction_samples import _fiber_volumes, write_fiber_dataset

    manifest = write_fiber_dataset(tmp_path / 'fibers', *_fiber_volumes())
    fibers = {'location': str(manifest), 'identity': 'fibers-test'}
    body = initialize_body() | {'project': project(fiber_inference=fibers, fiber_scale=4)}
    body['run']['config'].update(input_use_fiber_directions=True,
                                 loss_weight_fiber_directions=1.0)
    paths, _ = _built(service(host, store, build_missing_lasagna_packs=True), body)
    # Under the identity VC3D sent, at the volume's scale.
    assert paths.fiber_directions == fiber_directions_path(host.cache, 'fibers-test', 4)
    samples = load_fiber_direction_samples(paths.fiber_directions, 0, 2)
    assert len(samples['position_zyx']) == 2
    before = Path(paths.fiber_directions).stat().st_mtime_ns
    _built(service(host, store, build_missing_lasagna_packs=True), body)
    assert Path(paths.fiber_directions).stat().st_mtime_ns == before

    # Without the loss nothing is extracted.
    other = {**fibers, 'identity': 'fibers-unused'}
    optional = initialize_body() | {'project': project(fiber_inference=other, fiber_scale=4)}
    optional['run']['config'].update(input_use_fiber_directions=True)
    paths, _ = _built(service(host, store, build_missing_lasagna_packs=True), optional)
    assert paths.fiber_directions == ''


def test_the_fit_volume_must_be_a_project_volume_with_an_umbilicus(host, store):
    failure = _built(service(host, store),
                     initialize_body() | {'project': project(volume='other-volume')})
    assert f'the fit volume {VOLUME} is not a volume of the project' in failure['message']
    failure = _built(service(host, store),
                     initialize_body() | {'project': project(umbilicus=None)})
    assert 'has no umbilicus for the fit volume' in failure['message']


def test_a_changed_project_volume_rebuilds_everything(host, store):
    state = service(host, store)
    _built(state, initialize_body())
    state.session = SimpleNamespace(status=lambda: {'state': SessionState.Idle,
                                                    'input_generation': 0})
    changed = {**UMBILICUS, 'revision': 'b8e6d4f1-2c3d-4e5f-9a01-1b2c3d4e5f60'}
    for request, stage in ((initialize_body(), 'model'),
                           (initialize_body() | {'project': project(umbilicus=changed)}, 'all'),
                           (initialize_body() | {'project': project(name='renamed')}, 'all')):
        request['run']['config']['model_num_flow_stages'] = 3
        paths, run, preview, scroll = state._prepare_session_request(request)
        with state.lock:
            assert state._rebuild_stage_locked(paths, run, preview, scroll,
                                               SessionState.Idle) == stage


# ---------------------------------------------------------------------------
# Host bindings: no dataset, winding inference bound at launch
# ---------------------------------------------------------------------------

WINDING_MODEL = {**NO_DENSE, 'dense_spacing_mode': 'winding_model'}


def winding_artifact(directory, valid=True):
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'artifact_type': 'winding_inference_crossings' if valid else 'other',
        'format_version': 1}))
    return directory


def test_bindings_advertise_the_winding_inference_binding(host, store, tmp_path):
    unbound = service(host, store).binding_advertisement()
    assert unbound['bindings']['output_directory'] == str(host.output)
    assert unbound['bindings']['cache_directory'] == str(host.cache)
    assert unbound['bindings']['winding_inference'] == {
        'bound': False, 'path': None, 'available': False,
        'error': 'the service was launched without --winding-inference'}
    assert unbound['capabilities']['winding_inference'] is False

    artifact = winding_artifact(tmp_path / 'winding')
    bound = service(host, store, spiral_service.bind_service_inputs(
        host.output, host.cache, artifact)).binding_advertisement()
    assert bound['bindings']['winding_inference'] == {
        'bound': True, 'path': str(artifact), 'available': True, 'error': None}
    assert bound['capabilities']['winding_inference'] is True

    # An artifact that cannot be read is bound but unavailable, and says why.
    broken = winding_artifact(tmp_path / 'broken', valid=False)
    unusable = service(host, store, spiral_service.bind_service_inputs(
        host.output, host.cache, broken)).binding_advertisement()
    status = unusable['bindings']['winding_inference']
    assert (status['bound'], status['available']) == (True, False)
    assert 'unexpected artifact_type' in status['error']
    assert unusable['capabilities']['winding_inference'] is False


def test_configuration_defaults_follow_the_winding_inference_binding(
        host, store, tmp_path):
    def default(bindings=None):
        return service(host, store, bindings).configuration_catalog()[
            'defaults']['input_use_winding_inference']

    assert default() is False
    assert default(spiral_service.bind_service_inputs(
        host.output, host.cache, winding_artifact(tmp_path / 'winding'))) is True
    assert default(spiral_service.bind_service_inputs(
        host.output, host.cache,
        winding_artifact(tmp_path / 'broken', valid=False))) is False


def test_a_missing_winding_inference_binding_is_refused_at_launch(tmp_path):
    with pytest.raises(spiral_service.ServiceBindingError, match='is not a directory'):
        spiral_service.bind_service_inputs(tmp_path / 'out', tmp_path / 'cache',
                                           tmp_path / 'missing')


def test_launch_refuses_retired_and_unusable_bindings(host, store, tmp_path, capsys):
    with pytest.raises(SystemExit):
        spiral_service.main(_launch(host, '--object-store', store,
                                    '--dataset', str(tmp_path)))
    err = capsys.readouterr().err
    assert '--dataset is retired' in err and '--winding-inference' in err
    with pytest.raises(SystemExit):
        spiral_service.main(_launch(host, '--object-store', store,
                                    '--winding-inference', str(tmp_path / 'missing')))
    assert '--winding-inference' in capsys.readouterr().err


def test_requests_cannot_name_host_input_paths(host, store, tmp_path):
    state = service(host, store)
    for key in ('winding_inference', 'dataset_root', 'umbilicus'):
        body = initialize_body()
        body['paths'] = {key: str(tmp_path)}
        with pytest.raises(ApiError) as refused:
            state._prepare_session_request(body)
        assert refused.value.status == 400
        [detail] = refused.value.details
        assert detail['field'] == f'paths.{key}'
    body['paths'] = {'winding_inference': str(tmp_path)}
    with pytest.raises(ApiError) as refused:
        state._prepare_session_request(body)
    assert 'restart the service' in refused.value.details[0]['message']


def test_winding_model_supervision_reads_the_bound_artifact(host, store, tmp_path):
    body = initialize_body()
    body['run']['config'] = {**WINDING_MODEL, 'input_use_winding_inference': True}
    # Without a binding a fit that asks for it fails preparation, saying how
    # to provide one; nothing is built.
    with pytest.raises(ApiError) as refused:
        service(host, store).initialize(body)
    [detail] = [detail for detail in refused.value.details
                if detail['field'] == 'winding_inference']
    assert '--winding-inference PATH' in detail['message']
    # Enabled only by the Python default (the request leaves the toggle
    # out), it is left out instead.
    implicit = initialize_body()
    implicit['run']['config'] = dict(WINDING_MODEL)
    state = service(host, store)
    paths, run, _, _ = state._prepare_session_request(implicit)
    assert paths.winding_inference == ''
    assert run.config['input_use_winding_inference'] is False
    [warning] = [event for event in state.events.read_after(0)['events']
                 if event['severity'] == 'warning']
    assert 'the default configuration enables it' in warning['text']
    # So is one a checkpoint's configuration enables.
    checkpoint = host.output / 'fit' / 'checkpoint.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'')
    state = service(host, store)
    state._checkpoint_durable_cfg = lambda path: (
        {**WINDING_MODEL, 'input_use_winding_inference': True}, None, None, None)
    resumed = initialize_body()
    resumed['paths'] = {'checkpoint': str(checkpoint)}
    resumed['run']['config'] = {}
    request, config = state._bound_session_request(
        resumed, state._request_project(resumed), include_input_config=True)
    assert config['input_use_winding_inference'] is False
    assert request['run']['config'] == {'input_use_winding_inference': False}
    assert request['paths']['winding_inference'] == ''
    [warning] = [event for event in state.events.read_after(0)['events']
                 if event['severity'] == 'warning']
    assert 'the checkpoint it resumes was fitted with it' in warning['text']
    assert '--winding-inference PATH' in warning['text']

    artifact = winding_artifact(tmp_path / 'winding')
    state = service(host, store, spiral_service.bind_service_inputs(
        host.output, host.cache, artifact))
    paths, _, _, _ = state._prepare_session_request(body)
    assert paths.winding_inference == str(artifact)
    assert paths.provenance['winding_inference'] == str(artifact)
    # A fit that does not need it does not read it.
    paths, _, _, _ = state._prepare_session_request(initialize_body())
    assert paths.winding_inference == ''


def test_a_service_initializes_from_project_and_store_alone(host, store, tmp_path):
    # No dataset directory or spiral-scroll.json anywhere: the project and
    # the bindings resolve every input, recorded with the fit volume.
    assert not list(tmp_path.rglob('spiral-scroll.json'))
    state = service(host, store)
    paths, _ = _built(state, initialize_body())
    assert paths.volume_uuid == VOLUME
    recorded = state.session_request['paths']
    assert recorded['volume_uuid'] == VOLUME
    assert 'dataset_root' not in recorded
    assert recorded['provenance'] == {'project': 's1',
                                      'output_directory': str(host.output),
                                      'cache_directory': str(host.cache)}


def test_local_service_packs_before_linking_and_reuses_cache(host, store, tmp_path,
                                                           monkeypatch):
    import numpy as np
    from test_sparse_cuda_cache import write_raw_pack_source
    from lasagna_source import load_lasagna_source
    import object_store_inputs as osi

    lasagna = {'location': str(tmp_path / 'local.lasagna.json'),
               'identity': 'local-test', 'group': '0'}
    for channel in ('normal_x', 'normal_y', 'gradient_magnitude'):
        path = tmp_path / channel
        write_raw_pack_source(path / '0', np.full((16, 16, 16), 128, dtype=np.uint8))
        lasagna[channel] = str(path)
    body = initialize_body() | {'project': project(lasagna=lasagna, lasagna_scale=2)}
    body['run']['config']['loss_weight_dense_spacing'] = 1
    state = service(host, store, build_missing_lasagna_packs=True)
    original = osi.prepare_position
    linked = []

    def link(*args, **kwargs):
        for kind in ('normals', 'grad_mag'):
            assert (host.cache / 'lasagna/local-test' / f'{kind}_g0/meta.json').is_file()
        linked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(osi, 'prepare_position', link)
    paths, _ = _built(state, body)
    assert linked
    assert state.binding_advertisement()['capabilities']['build_missing_lasagna_packs']
    source = load_lasagna_source(paths.lasagna)
    meta = Path(source.pack_directory(paths.cache_directory, 'normals')) / 'meta.json'
    before = meta.stat().st_mtime_ns
    # Cached packs no longer require the local arrays.
    for channel in ('normal_x', 'normal_y', 'gradient_magnitude'):
        Path(lasagna[channel]).rename(tmp_path / (channel + '-unavailable'))
    _built(service(host, store, build_missing_lasagna_packs=True), body)
    assert meta.stat().st_mtime_ns == before


@pytest.mark.parametrize('location', ['https://example.invalid/array', '/missing/array'])
def test_pack_failure_preserves_resident_before_link_refresh(host, store,
                                                            monkeypatch, location):
    import object_store_inputs as osi
    state = service(host, store, build_missing_lasagna_packs=True)
    resident = SimpleNamespace(status=lambda: {"state": SessionState.Idle})
    state.session = resident
    monkeypatch.setattr(osi, 'prepare_position', lambda *a, **k: pytest.fail('must not link'))
    lasagna = {'location': '/local.lasagna.json', 'identity': 'missing-test',
               'group': '0', 'normal_x': location, 'normal_y': location}
    body = initialize_body() | {'project': project(lasagna=lasagna, lasagna_scale=2)}
    failure = _built(state, body)
    assert 'not an accessible local array' in failure['message']
    assert state.session is resident
