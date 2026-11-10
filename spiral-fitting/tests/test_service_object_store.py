"""The Spiral service's object-store inputs: the launch binding, the session
lease, store-backed discovery and the VC3D project's inputs.

The resident runtime is faked where a test needs one; stores are written
with object_store_fixture. Launch flags and the host bindings' advertisement
are covered by test_spiral_service_v2.
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
from cpu_service import wait_until  # noqa: E402
from fit_session import SessionState, SpiralInputPaths  # noqa: E402
from object_store_fixture import VOLUME  # noqa: E402
from service_http import ApiError  # noqa: E402
from service_project import FACTS, UMBILICUS, project, with_project  # noqa: E402

# The project has no Lasagna stores and the store no shell, so runs turn off
# the losses that would need them.
NO_DENSE = {'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}
WINDING_MODEL = {**NO_DENSE, 'dense_spacing_mode': 'winding_model'}


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


def refusal(call, *args):
    """The message and detail messages of the ApiError ``call`` raises."""
    with pytest.raises(ApiError) as refused:
        call(*args)
    return '\n'.join([refused.value.message, *(
        detail['message'] for detail in refused.value.details or [])])


def initialize_body(**config):
    return with_project({'run': {'z_begin': 0, 'z_end': 10,
                                 'config': {**NO_DENSE, **config}}})


# ---------------------------------------------------------------------------
# The launch binding
# ---------------------------------------------------------------------------

def test_binding_resolves_the_store_once_and_records_its_uuid(store_path, tmp_path,
                                                              monkeypatch):
    monkeypatch.chdir(Path(store_path).parent)
    binding = spiral_service.bind_object_store(Path(store_path).name, 'svc')
    with ObjectStore.open(store_path, read_only=True) as db:
        assert binding.uuid == db.uuid
        assert binding.generation == db.status()['generation']
    assert binding.location == str(Path(store_path).resolve()) and not binding.served
    assert binding.config() == {'object_store_location': binding.location,
                                'object_store_author': 'svc',
                                'object_store_link_timeout_s': 300,
                                'object_store_input_position': None}
    assert spiral_service.bind_object_store(store_path).author.startswith('spiral-service@')
    garbage = tmp_path / 'garbage.vcobj'
    garbage.write_text('not a store')
    with pytest.raises(spiral_service.ObjectStoreBindingError,
                       match='cannot open the object store'):
        spiral_service.bind_object_store(garbage)


def test_clients_cannot_choose_the_binding_or_host_input_paths(host, store_path, capsys,
                                                               tmp_path):
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(SystemExit):
            spiral_service.main([
                '--output', str(host.output / 'svc'), '--cache', str(host.output / 'cache'),
                '--api-key-file', str(host.output / 'key'), '--object-store', store_path,
                '--config', json.dumps({key: '/elsewhere.vcobj'})])
        assert f'--config must not set {key}' in capsys.readouterr().err
    state = service(host, store_path)
    for key in spiral_service.SERVICE_OWNED_CONFIG_KEYS:
        with pytest.raises(ApiError) as raised:
            state._prepare_session_request(initialize_body(**{key: 'chosen'}))
        assert raised.value.status == 400
        assert raised.value.details == [{
            'field': f'run.config.{key}',
            'message': "Owned by the service's --object-store binding"}]
    # Unset values (a complete default configuration) and the binding's own
    # values (an echoed session request) are not a choice.
    for request in (state.startup_session_request(),
                    {'run': {'z_begin': 0, 'z_end': 10, 'config': {
                        **NO_DENSE, 'object_store_location': None,
                        'object_store_author': 'spiral-test'}}}):
        paths, run, _, _ = state._prepare_session_request(with_project(request))
        assert run.config['object_store_location'] == state.object_store.location
        assert run.config['object_store_author'] == 'spiral-test'
        assert not {'verified_patches', 'fibers', 'pcls'} & set(paths.manifest())
    # The winding-inference artifact is bound at launch, not named by a request.
    body = initialize_body()
    body['paths'] = {'winding_inference': str(tmp_path)}
    with pytest.raises(ApiError) as refused:
        state._prepare_session_request(body)
    [detail] = refused.value.details
    assert detail['field'] == 'paths.winding_inference'
    assert 'restart the service' in detail['message']


def test_the_binding_wins_over_a_resumed_checkpoints_store(host, store_path,
                                                           monkeypatch):
    import checkpoint_io
    from spiral_progress import NullProgressReporter
    from spiral_runtime import InteractiveFitSession

    stored = Config().as_dict()
    stored.update(object_store_location='/written/against/another.vcobj',
                  object_store_author='someone-else')
    monkeypatch.setattr(checkpoint_io, 'load_checkpoint_cpu',
                        lambda path: {'cfg': copy.deepcopy(stored)})
    state = service(host, store_path)
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
    assert config['object_store_location'] == state.object_store.location
    assert config['object_store_author'] == 'spiral-test'


# ---------------------------------------------------------------------------
# The session lease
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
def http(host, store_path, monkeypatch):
    import spiral_runtime
    from protocol_client_service import FakeRuntime
    built = []

    def create_session(*args, **kwargs):
        built.append(kwargs)
        return FakeRuntime()
    monkeypatch.setattr(spiral_runtime, 'create_session', create_session)
    client = Http(service(host, store_path))
    client.built = built
    yield client
    client.close()
    client.state.close()


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
    # A repeated initialize command builds once; nothing binds inputs into
    # the build: they come from the store.
    body = {**initialize_body(), 'command_id': 'init-1'}
    first = http('POST', '/session/initialize', body, 'owner')
    second = http('POST', '/session/initialize', body, 'owner')
    assert first[0] == second[0] == 200 and first[1]['state'] == 'Idle'
    assert len(http.built) == 1 and 'input_bindings' not in http.built[0]


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
def test_release_waits_for_every_user_of_the_service(host, store_path, user):
    from concurrent.futures import ThreadPoolExecutor
    state = service(host, store_path)
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


def test_release_during_construction_closes_the_installed_fitter(host, store_path,
                                                                 monkeypatch):
    import sys
    import types
    from concurrent.futures import ThreadPoolExecutor
    state = service(host, store_path)
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
    with ThreadPoolExecutor(1) as pool:
        releasing = pool.submit(state.release_lease, 'owner', 'release')
        with state._service_condition:
            assert state._service_condition.wait_for(lambda: state._closing, timeout=5)
        finish.set()
        releasing.result(5)
    assert closed.is_set() and state.session is None
    state.close()


# ---------------------------------------------------------------------------
# Discovery of the bound store's inputs
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
    # GET /bindings never reads the store itself: the selection is read in
    # the background and advertised once it has been read.
    assert state.binding_advertisement()['inputs'] in (
        {'pending': True}, state.startup_inputs())
    state.startup_inputs()
    advertised = state.binding_advertisement()
    with ObjectStore.open(filled_store, read_only=True) as db:
        status = db.status()
    assert advertised['object_store'] == {
        'uuid': state.object_store.uuid, 'generation': status['generation'],
        'location': state.object_store.location, 'served': False,
        'author': 'spiral-test', 'commit_sequence': status['commit_sequence'],
        'position_error': None}
    assert advertised['capabilities']['input_source'] == 'object_store'
    assert advertised['capabilities']['commits'] is True
    # Nothing advertises patch, fiber or PCL files, or a host dataset.
    assert not {'pcl_inputs', 'root', 'resolved', 'detected_checkpoints',
                'scroll_spec'} & set(advertised)
    inputs = advertised['inputs']
    assert inputs['ok'] and inputs['volume_uuid'] == VOLUME
    assert {kind: entry['count'] for kind, entry in inputs['selection'].items()} == {
        'surface': 2, 'point_collection': 1, 'fiber': 1, 'track': 0}
    assert inputs['selection']['fiber']['scales'] == [[2.0, 1]]
    assert inputs['relationships'] == {'pcl_patch': 1}
    by_name = {entry['name']: entry for entry in inputs['objects']}
    assert by_name['abs']['pcl_role'] == 'absolute' and 'pcl_role' not in by_name['near']
    with ObjectStore.open(filled_store, read_only=True) as db:
        assert by_name['fiber']['geometry_revision'] == \
            db.current_geometry(by_name['fiber']['uuid'])['uuid']
        # Discovery writes nothing: no refresh ran.
        assert len(db.edges_by_kind('point-surface')) == 1
    # It follows the store's changes, and reports a selection a fit refuses.
    assert state.startup_inputs() is state.startup_inputs()
    with ObjectStore.open(filled_store, author='editor') as db:
        db.set_meta(by_name['near']['uuid'], 'coordinates',
                    {'volume_uuid': 'another-volume', 'scale_factor': 1})
    refused = state.startup_inputs()
    assert refused['ok'] is False
    assert 'selected objects declare different volumes' in refused['problems'][0]


def test_an_unreachable_served_store_has_no_summary_and_caches_none(host, monkeypatch):
    # A served store out of reach: /bindings says so instead of pending
    # forever, nothing is read or cached for it, and waiting callers fail.
    # (bind_object_store's own refusal of such an endpoint is tested with a
    # server; here its attachment attempts are stood in for, as they take
    # seconds of retries each.)
    def refused(*args, **kwargs):
        raise ConnectionRefusedError('connection refused')
    monkeypatch.setattr(spiral_service.ObjectStoreBinding, 'open', refused)
    binding = spiral_service.ObjectStoreBinding(
        'http://127.0.0.1:1', 'store-uuid', 'generation-1', 'svc', served=True)
    assert binding.head() == (None, 'ConnectionRefusedError: connection refused')
    state = spiral_service.ServiceState(
        bindings=host.bindings, object_store=binding,
        startup_run={'z_begin': 0, 'z_end': 10, 'config': dict(NO_DENSE)})
    advertised = state.binding_advertisement()
    assert advertised['object_store']['commit_sequence'] is None
    assert advertised['object_store']['position_error']
    assert advertised['inputs']['pending'] is True
    assert 'cannot be read' in advertised['inputs']['error']
    assert state._startup_inputs is None and state._startup_inputs_reading is None
    assert 'cannot be read' in refusal(state.startup_inputs)
    assert 'cannot be read' in refusal(state._inputs_position, {
        'store': 'store-uuid', 'generation': 'generation-1', 'commit_sequence': 1})


def test_bindings_never_wait_for_the_store_to_be_read(host, filled_store,
                                                     monkeypatch):
    # Reading a large store's selection takes minutes, longer than a client
    # waits for GET /bindings: the read runs in the background, once.
    import object_store_inputs
    real = object_store_inputs.describe_selection
    release, calls = threading.Event(), []

    def slow(config):
        calls.append(config)
        assert release.wait(30)
        if len(calls) == 1:
            raise RuntimeError('store unreadable')
        return real(config)
    monkeypatch.setattr(object_store_inputs, 'describe_selection', slow)
    state = service(host, filled_store)

    def settled():
        wait_until(lambda: state._startup_inputs_reading is None)

    assert state.binding_advertisement()['inputs'] == {'pending': True}
    assert state.binding_advertisement()['inputs'] == {'pending': True}
    release.set()
    settled()
    # A failed read is reported and not kept: the next request reads again.
    assert len(calls) == 1
    assert any('store unreadable' in event['text']
               for event in state.events.read_after(0)['events'])
    assert state.binding_advertisement()['inputs'] == {'pending': True}
    settled()
    assert len(calls) == 2
    assert state.binding_advertisement()['inputs']['ok'] is True
    assert len(calls) == 2


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
            # The selection is read in the background; ask until it is.
            if not bindings['inputs'].get('pending'):
                break
            time.sleep(0.05)
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
    built = []
    state._build = lambda session_id, previous, paths, run, preview, scroll: (
        built.append((paths, scroll)))
    state.initialize(request)
    wait_until(lambda: built or not state._building, timeout)
    return built[0] if built else state.status()['input_preparation_error']


def test_the_build_reads_the_project_of_its_fit_volume(host, store_path, tmp_path):
    with pytest.raises(ApiError) as refused:
        service(host, store_path).initialize(initialize_body() | {'project': None})
    assert [detail['field'] for detail in refused.value.details] == ['project']
    # No dataset directory or spiral-scroll.json anywhere: the fit volume the
    # store's inputs name selects the project's volume, its umbilicus,
    # written where the fit reads it, and the scroll.
    from umbilicus import json_umbilicus_z_to_yx
    state = service(host, store_path)
    paths, scroll = _built(state, initialize_body())
    assert not list(tmp_path.rglob('spiral-scroll.json'))
    assert paths.volume_uuid == VOLUME
    assert Path(paths.umbilicus).parent == host.output / 'project-inputs'
    assert json_umbilicus_z_to_yx(paths.umbilicus)(32.0).tolist() == [32.0, 32.0]
    assert (scroll.name, scroll.voxel_size_um, scroll.spiral_outward_sense) == (
        's1', 9.6, 'CW')
    request = state.session_request
    assert request['project'] == {
        'name': 's1', 'volume': VOLUME, 'voxel_size_um': 9.6,
        'z_direction_is_top_to_bottom': False, 'left_handed_coordinates': False,
        'umbilicus': {'uuid': UMBILICUS['uuid'], 'generation': UMBILICUS['generation'],
                      'version': UMBILICUS['version'], 'points': 2}}
    assert request['scroll']['umbilicus_coordinate_scale'] == 1.0
    assert request['paths']['volume_uuid'] == VOLUME
    assert 'dataset_root' not in request['paths']
    assert request['paths']['provenance'] == {'project': 's1',
                                              'output_directory': str(host.output),
                                              'cache_directory': str(host.cache)}


def test_the_scroll_is_the_projects_and_the_sense_a_fit_setting(host, store_path):
    facts = {'voxel_size_um': 7.91, 'base_shape_zyx': [100, 64, 64],
             'z_direction_is_top_to_bottom': None, 'left_handed_coordinates': None}
    body = initialize_body() | {'project': project(name='PHerc', **facts)}
    failure = _built(service(host, store_path), body)
    assert 'set the fit setting spiral_outward_sense' in failure['message']
    body['run']['config']['spiral_outward_sense'] = 'ACW'
    _, scroll = _built(service(host, store_path), body)
    assert (scroll.name, scroll.voxel_size_um, scroll.base_shape_zyx,
            scroll.spiral_outward_sense) == ('PHerc', 7.91, (100, 64, 64), 'ACW')
    # A volume whose catalog orientation decides the sense refuses another.
    failure = _built(service(host, store_path), initialize_body(spiral_outward_sense='ACW'))
    assert 'contradicts the catalog' in failure['message']


def test_the_fit_volume_must_be_a_project_volume_with_an_umbilicus(host, store_path,
                                                                   monkeypatch):
    failure = _built(service(host, store_path),
                     initialize_body() | {'project': project(volume='other-volume')})
    assert f'the fit volume {VOLUME} is not a volume of the project' in failure['message']
    # Another volume has one: only the store's selection names the fit
    # volume, so the build refuses it once the inputs are read.
    other = project()
    other['volumes']['other-volume'] = other['volumes'].pop(VOLUME)
    other['volumes'][VOLUME] = {key: value for key, value
                                in other['volumes']['other-volume'].items()
                                if key != 'umbilicus'}
    failure = _built(service(host, store_path), initialize_body() | {'project': other})
    assert f'has no umbilicus for the fit volume {VOLUME}' in failure['message']
    # Reading a large store's selection takes minutes: a project that cannot
    # serve any fit volume, or the configured one, is refused on the request.
    import object_store_inputs

    def unread(*args, **kwargs):
        raise AssertionError('the store was read')
    monkeypatch.setattr(object_store_inputs, 'read_snapshot', unread)
    state = service(host, store_path)
    assert 'has no umbilicus for any of its volumes' in refusal(
        state.initialize, initialize_body() | {'project': project(umbilicus=None)})
    body = initialize_body(fit_volume_uuid=VOLUME) | {'project': project(umbilicus=None)}
    body['project']['volumes']['other-volume'] = {**FACTS, 'umbilicus': UMBILICUS}
    assert f'has no umbilicus for the fit volume {VOLUME}' in refusal(state.initialize, body)
    assert not state._building


def test_the_lasagna_dataset_is_the_projects(host, store_path):
    from lasagna_source import load_lasagna_source, url_identity
    # A remote dataset is identified by its URL, which VC3D sends alone.
    lasagna = {'location': 'https://data.example/p.lasagna.json', 'group': '3'}
    for key in ('normal_x', 'normal_y', 'gradient_magnitude'):
        lasagna[key] = f'https://data.example/{key}.ome.zarr'

    def body(lasagna=lasagna, **config):
        return initialize_body(**config) | {'project': project(lasagna=lasagna, lasagna_scale=2)}
    paths, scroll = _built(service(host, store_path), body())
    # The fit reads the project's dataset by identity; its stores need not
    # be readable here.
    source = load_lasagna_source(paths.lasagna)
    assert (source.identity, source.group, source.location) == (
        url_identity(lasagna['location']), '3', lasagna['location'])
    assert source.stores['normal_x'] == lasagna['normal_x']
    assert scroll.lasagna_scale == 2
    # A fit that needs normals reads their pack from the service's cache,
    # which never builds one: without it the build is refused up front.
    assert 'not in the cache' in refusal(service(host, store_path).initialize,
                                         body(loss_weight_dense_normals=1.0))
    pack = Path(source.pack_directory(paths.cache_directory, 'normals'))
    pack.mkdir(parents=True)
    (pack / 'meta.json').write_text('{}')
    paths, _ = _built(service(host, store_path), body(loss_weight_dense_normals=1.0))
    assert load_lasagna_source(paths.lasagna) == source
    # A project without a dataset cannot give them.
    assert 'selects no Lasagna dataset' in refusal(
        service(host, store_path).initialize, initialize_body(loss_weight_dense_normals=1.0))
    # A local dataset's identity is VC3D's; a remote one's cannot be.
    local = {**lasagna, 'location': '/data/p.lasagna.json', 'identity': 'run-1'}
    paths, _ = _built(service(host, store_path), body(local))
    assert load_lasagna_source(paths.lasagna).identity == 'run-1'
    assert 'identified by its URL' in refusal(
        service(host, store_path).initialize, body({**lasagna, 'identity': 'run-1'}))


def test_local_service_packs_before_linking_and_reuses_cache(host, store_path, tmp_path,
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
    body = initialize_body(loss_weight_dense_spacing=1) | {
        'project': project(lasagna=lasagna, lasagna_scale=2)}
    state = service(host, store_path, build_missing_lasagna_packs=True)
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
    _built(service(host, store_path, build_missing_lasagna_packs=True), body)
    assert meta.stat().st_mtime_ns == before


@pytest.mark.parametrize('flags, builds', [([], True), (['--build-missing-lasagna-packs'], True),
                                            (['--require-prepared-lasagna-packs'], False)])
def test_a_launched_service_builds_missing_packs_unless_told_not_to(
        tmp_path, store_path, monkeypatch, flags, builds):
    class Constructed(Exception):
        pass

    def construct(**kwargs):
        raise Constructed(kwargs['build_missing_lasagna_packs'])
    monkeypatch.setattr(spiral_service, 'ServiceState', construct)
    import sys
    monkeypatch.setattr(sys, 'stdout', sys.stdout)
    monkeypatch.setattr(sys, 'stderr', sys.stderr)
    with pytest.raises(Constructed) as constructed:
        spiral_service.main(['--object-store', str(store_path), '--output', str(tmp_path / 'out'),
                             '--cache', str(tmp_path / 'cache'),
                             '--api-key-file', str(tmp_path / 'key'), *flags])
    assert constructed.value.args == (builds,)


def test_a_service_packs_remote_arrays_as_it_packs_local_ones(host, store_path, tmp_path,
                                                             monkeypatch, request):
    import numpy as np
    from fake_s3 import FakeS3Server
    import remote_zarr
    from test_sparse_cuda_cache import write_raw_pack_source

    channels = ('normal_x', 'normal_y', 'gradient_magnitude')
    for value, channel in enumerate(channels):
        write_raw_pack_source(tmp_path / 'bucket' / channel / '0',
                              np.full((16, 16, 16), 100 + value, dtype=np.uint8))

    def lasagna(identity, root):
        return {'location': '/client/remote_cache/s.lasagna.json', 'identity': identity,
                'group': '0', **{channel: f'{root}/{channel}' for channel in channels}}
    local = lasagna('local-copy', tmp_path / 'bucket')
    s3 = FakeS3Server(tmp_path / 'bucket')
    request.addfinalizer(s3.close)
    remote = lasagna('remote-copy', s3.location())
    monkeypatch.setattr(remote_zarr, 'access_for', lambda location: s3.access())

    def built(dataset):
        _built(service(host, store_path, build_missing_lasagna_packs=True),
               initialize_body(loss_weight_dense_spacing=1) | {
                   'project': project(lasagna=dataset, lasagna_scale=2)})
        return host.cache / 'lasagna' / dataset['identity']
    from_local, from_remote = built(local), built(remote)
    assert s3.gets
    for kind in ('normals_g0', 'grad_mag_g0'):
        for name in ('table.npy', 'channel_0.u8'):
            assert (from_remote / kind / name).read_bytes() == (from_local / kind / name).read_bytes()
        # The fetched arrays are removed once packed.
        assert not (from_remote / f'{kind}.arrays').exists()


@pytest.mark.parametrize('location, refused', [
    ('https://data.example/array', 'its objects cannot be listed'),
    ('/missing/array', 'not an accessible local array')])
def test_pack_failure_preserves_resident_before_link_refresh(host, store_path,
                                                            monkeypatch, location, refused):
    import object_store_inputs as osi
    state = service(host, store_path, build_missing_lasagna_packs=True)
    resident = SimpleNamespace(status=lambda: {"state": SessionState.Idle})
    state.session = resident
    monkeypatch.setattr(osi, 'prepare_position', lambda *a, **k: pytest.fail('must not link'))
    lasagna = {'location': '/local.lasagna.json', 'identity': 'missing-test',
               'group': '0', 'normal_x': location, 'normal_y': location}
    failure = _built(state, initialize_body() | {
        'project': project(lasagna=lasagna, lasagna_scale=2)})
    assert refused in failure['message']
    assert state.session is resident


def test_lasagna_pack_kinds_follow_the_resolved_config():
    source = SimpleNamespace(stores={'normal_x': 'x', 'normal_y': 'y', 'gradient_magnitude': 'g'})
    config = Config().as_dict()
    kinds = spiral_service._lasagna_pack_kinds
    assert kinds({**config, 'dense_spacing_mode': 'winding_model'}, source) == {'normals'}
    assert kinds({**config, 'dense_spacing_mode': 'grad_mag'}, source) == {'normals', 'grad_mag'}
    # Normals without their loss: still read by the link refresh.
    assert kinds({**config, 'dense_spacing_mode': 'winding_model',
                  'loss_weight_dense_normals': 0}, source) == {'normals'}


def test_fiber_directions_are_the_project_datasets_cached_samples(host, store_path, tmp_path):
    from fiber_direction_samples import load_fiber_direction_samples
    from lasagna_source import fiber_directions_path, url_identity
    from test_fiber_direction_samples import _fiber_volumes, write_fiber_dataset
    fibers = {'location': 'https://data.example/f.lasagna.json'}
    loss = {'input_use_fiber_directions': True, 'loss_weight_fiber_directions': 1.0}

    def body(fibers=fibers, **config):
        return initialize_body(**config) | {'project': project(fiber_inference=fibers,
                                                               fiber_scale=4)}
    # The loss needs them: without the cache's samples the build is refused.
    failure = _built(service(host, store_path), body(**loss))
    assert 'no fiber directions of https://data.example/f.lasagna.json' in failure['message']
    # Enabled without the loss, they are read when the project can give them.
    paths, _ = _built(service(host, store_path), body(None, input_use_fiber_directions=True))
    assert paths.fiber_directions == ''
    cached = Path(fiber_directions_path(paths.cache_directory,
                                        url_identity(fibers['location']), 4))
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'')
    assert _built(service(host, store_path), body(**loss))[0].fiber_directions == str(cached)
    assert _built(service(host, store_path), body(input_use_fiber_directions=True)
                  )[0].fiber_directions == str(cached)

    # A local service extracts missing ones, under the identity VC3D sent at
    # the volume's scale, once; without the loss nothing is extracted.
    manifest = write_fiber_dataset(tmp_path / 'fibers', *_fiber_volumes())
    local = {'location': str(manifest), 'identity': 'fibers-test'}
    paths, _ = _built(service(host, store_path, build_missing_lasagna_packs=True),
                      body(local, **loss))
    assert paths.fiber_directions == fiber_directions_path(host.cache, 'fibers-test', 4)
    assert len(load_fiber_direction_samples(paths.fiber_directions, 0, 2)['position_zyx']) == 2
    before = Path(paths.fiber_directions).stat().st_mtime_ns
    _built(service(host, store_path, build_missing_lasagna_packs=True), body(local, **loss))
    assert Path(paths.fiber_directions).stat().st_mtime_ns == before
    paths, _ = _built(service(host, store_path, build_missing_lasagna_packs=True),
                      body({**local, 'identity': 'fibers-unused'},
                           input_use_fiber_directions=True))
    assert paths.fiber_directions == ''


def test_fiber_directions_extract_from_the_copys_remote_manifest(host, store_path, monkeypatch):
    import fiber_direction_samples
    from lasagna_source import fiber_directions_path
    calls = []

    def extract(location, output, path, *, identity, **kwargs):
        calls.append((location, identity))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b'')
    monkeypatch.setattr(fiber_direction_samples, 'extract', extract)
    # VC3D's copy is not on this host; the manifest it was copied from is read,
    # under the identity VC3D sent.
    fibers = {'location': '/client/remote_cache/f/f.lasagna.json', 'identity': 'fibers-copy',
              'remote_manifest': 'https://data.example/f/f.lasagna.json'}
    body = initialize_body(input_use_fiber_directions=True, loss_weight_fiber_directions=1.0) | {
        'project': project(fiber_inference=fibers, fiber_scale=4)}
    paths, _ = _built(service(host, store_path, build_missing_lasagna_packs=True), body)
    assert calls == [('https://data.example/f/f.lasagna.json', 'fibers-copy')]
    assert paths.fiber_directions == fiber_directions_path(host.cache, 'fibers-copy', 4)
    with pytest.raises(ApiError, match=r'remote_manifest must be an HTTP\(S\) URL'):
        service(host, store_path).initialize(initialize_body() | {'project': project(
            fiber_inference={**fibers, 'remote_manifest': 's3://data/f.lasagna.json'})})


def test_a_changed_project_volume_rebuilds_everything(host, store_path):
    state = service(host, store_path)
    _built(state, initialize_body())
    state.session = SimpleNamespace(status=lambda: {'state': SessionState.Idle,
                                                    'input_generation': 0})
    changed = {**UMBILICUS, 'version': UMBILICUS['version'] + 1}
    for request, stage in ((initialize_body(), 'model'),
                           (initialize_body() | {'project': project(umbilicus=changed)}, 'all'),
                           (initialize_body() | {'project': project(name='renamed')}, 'all')):
        request['run']['config']['model_num_flow_stages'] = 3
        paths, run, preview, scroll = state._prepare_session_request(request)
        with state.lock:
            assert state._rebuild_stage_locked(paths, run, preview, scroll,
                                               SessionState.Idle) == stage


# ---------------------------------------------------------------------------
# Winding inference, bound at launch
# ---------------------------------------------------------------------------

def winding_artifact(directory, valid=True):
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({
        'artifact_type': 'winding_inference_crossings' if valid else 'other',
        'format_version': 1}))
    return directory


def test_winding_model_supervision_reads_the_bound_artifact(host, store_path, tmp_path):
    def bound(artifact):
        return spiral_service.bind_service_inputs(host.output, host.cache, artifact)

    def default(state):
        return state.configuration_catalog()['defaults']['input_use_winding_inference']

    def warning(state):
        [event] = [event for event in state.events.read_after(0)['events']
                   if event['severity'] == 'warning']
        return event['text']
    body = initialize_body(**WINDING_MODEL, input_use_winding_inference=True)
    # Without a binding a fit that asks for it fails preparation, saying how
    # to provide one; nothing is built.
    unbound = service(host, store_path)
    assert default(unbound) is False
    assert '--winding-inference PATH' in refusal(unbound.initialize, body)
    # Enabled only by the Python default (the request leaves the toggle
    # out), it is left out instead.
    paths, run, _, _ = unbound._prepare_session_request(initialize_body(**WINDING_MODEL))
    assert paths.winding_inference == ''
    assert run.config['input_use_winding_inference'] is False
    assert 'the default configuration enables it' in warning(unbound)
    # So is one a checkpoint's configuration enables.
    checkpoint = host.output / 'fit' / 'checkpoint.ckpt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'')
    state = service(host, store_path)
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
    assert 'the checkpoint it resumes was fitted with it' in warning(state)
    # An artifact that cannot be read is bound but unavailable.
    broken = service(host, store_path, bound(winding_artifact(tmp_path / 'broken', valid=False)))
    assert default(broken) is False
    status = broken.binding_advertisement()['bindings']['winding_inference']
    assert (status['bound'], status['available']) == (True, False)
    assert 'unexpected artifact_type' in status['error']
    # A usable one is the default and is read by a fit that needs it only.
    artifact = winding_artifact(tmp_path / 'winding')
    state = service(host, store_path, bound(artifact))
    assert default(state) is True
    paths, _, _, _ = state._prepare_session_request(body)
    assert paths.winding_inference == str(artifact)
    assert paths.provenance['winding_inference'] == str(artifact)
    paths, _, _, _ = state._prepare_session_request(initialize_body())
    assert paths.winding_inference == ''
