"""Spiral on a store served by vc_object_server: the fit's inputs, link
refresh, pins and commits through an endpoint (cases C3, C4, S3 and S5 of
the store-location design), read as a store file gives them.

Needs VC_OBJECT_SERVER naming a vc_object_server binary; skipped otherwise.
"""
import dataclasses
import json
import os
import re
import subprocess
from types import SimpleNamespace

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore, object_reference  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import spiral_service  # noqa: E402
from object_store_fixture import edges, fit_config, head, names, write_walk_store  # noqa: E402

SERVER = os.environ.get('VC_OBJECT_SERVER')
pytestmark = [pytest.mark.skipif(not SERVER, reason='set VC_OBJECT_SERVER to a vc_object_server binary'),
              pytest.mark.usefixtures('cpu')]


@pytest.fixture()
def served(tmp_path):
    """The walk store served by a vc_object_server subprocess: ``(endpoint,
    path)``. The file stays readable read-only beside the server."""
    path = write_walk_store(tmp_path / 'served.vcobj')
    server = subprocess.Popen(
        [SERVER, path, '--port', '0', '--strip-cache', str(tmp_path / 'strips')],
        stderr=subprocess.PIPE, text=True)
    try:
        line = server.stderr.readline()
        port = re.search(r':(\d+)\s*$', line)
        assert port, line
        yield f'http://127.0.0.1:{port.group(1)}', path
    finally:
        server.terminate()
        assert server.wait(timeout=30) == 0


def quiet(line):
    pass


def prepare(location, config=None, cache=None, **options):
    config = config or fit_config(location, 'service')
    with osi.open_store(config, cache_directory=cache) as store:
        return osi.prepare_position(store, config, osi.request_from_config(config), log=quiet,
                                    cache_directory=cache, **options)


def set_meta(store, uuid, key, value, request='7b9d0c2e-5a1f-4f6b-9c3d-2e1f0a9b8c7d'):
    """An editor's checked metadata edit through ``store``; returns the receipt."""
    status = store.status()
    return store.submit({'command_version': 3, 'request': request,
                         'store': status['store'], 'generation': status['generation'],
                         'author': 'editor',
                         'expected_objects': {uuid: object_reference(store.current_object_revision(uuid)['record'])},
                         'expected_relationships': {}, 'read_dependencies': {},
                         'expected_commit_sequence': None,
                         'operations': [{'op': 'set_meta', 'uuid': uuid, 'key': key, 'value': value}]})


def read_at(location, pins):
    config = fit_config(location, 'service')
    with osi.open_store(config, read_only=True) as store:
        return osi.read_snapshot(store, dataclasses.replace(osi.request_from_config(config), pins=pins))


# ---------------------------------------------------------------------------
# C3: the fit's inputs through an endpoint
# ---------------------------------------------------------------------------

def test_links_and_pins_through_the_endpoint_as_through_the_file(served, tmp_path):
    endpoint, path = served
    before = head(path)
    prepared = prepare(endpoint, cache=str(tmp_path / 'cache'))
    # The link refresh ran on the server as one link_objects request and
    # committed the links with the marker; the pins are at its commit.
    assert prepared.head == before
    assert prepared.position.commit_sequence == before + 1 == head(path)
    assert len(edges(path)) == 3
    assert set(prepared.pins.references) == set(names(path).values())
    assert any('links committed at commit' in line for line in prepared.lines)
    # Served: one stage, no detector progress; the strip cache is the server's.
    with osi.open_store(fit_config(endpoint, 'service'), read_only=True) as store:
        assert osi.is_served(store)
        assert osi.strip_cache_directory(store, str(tmp_path / 'cache')) is None
        marker = osi.read_link_marker(store)
    assert marker['commit_sequence'] == prepared.position.commit_sequence
    for step in marker['detectors']:
        assert 'cache_directory' not in step['params']
    # The pinned snapshot reads the same through the endpoint and the file.
    hosted = read_at(endpoint, prepared.pins)
    local = read_at(path, prepared.pins)
    assert hosted.digest() == local.digest()
    assert hosted.manifest()['store']['location'] == endpoint
    assert osi._compared(hosted.manifest())['objects'] == osi._compared(local.manifest())['objects']
    # A rerun finds the links current and writes nothing.
    again = prepare(endpoint, cache=str(tmp_path / 'cache'))
    assert again.position == prepared.position and head(path) == before + 1


def test_the_fit_loads_frozen_inputs_through_the_endpoint(served, tmp_path):
    endpoint, path = served
    cache = tmp_path / 'cache'
    hosted = osi.load_fit_inputs(fit_config(endpoint, 'service'), log=quiet, cache_directory=str(cache))
    # The served store's persistent cache lives under the fit's cache.
    assert (cache / osi.HOSTED_CACHE_DIRECTORY).is_dir()
    assert not (cache / osi.STRIP_CACHE_DIRECTORY).exists()
    # The file beside the server opens read-only: the links are current, so
    # a read of its head is the same frozen selection.
    local = osi.load_fit_inputs(fit_config(path, 'service', object_store_refresh_links=False), log=quiet)
    assert hosted.digest == local.digest
    assert hosted.counts == local.counts
    assert osi.input_summary(hosted)['fiber_patch_normals'] == {'kind': 'none', 'ct_volume': None}
    # The pins the fit froze read the same inputs again (a resume, a rank).
    pins = osi.InputPins.from_manifest(hosted.manifest)
    again = osi._compared(read_at(endpoint, pins).manifest())
    frozen = osi._compared(hosted.manifest)
    assert again['objects'] == frozen['objects'] and again['relationships'] == frozen['relationships']


def test_links_off_reads_the_head_without_writing(served, tmp_path):
    endpoint, path = served
    before = head(path)
    prepared = prepare(endpoint, fit_config(endpoint, 'service', object_store_refresh_links=False))
    assert prepared.position.commit_sequence == before == head(path)
    assert edges(path) == []
    assert prepared.pins.pinned


def test_a_position_alone_needs_its_versions_through_an_endpoint(served, tmp_path):
    endpoint, path = served
    prepared = prepare(endpoint)
    config = fit_config(endpoint, 'service')
    walk = names(path)['walk']
    # The file is the server's: edits go through the endpoint.
    with osi.open_store(config) as store:
        set_meta(store, walk, 'sampling_group', 'later')
    with pytest.raises(osi.ObjectStoreInputError, match='served store'):
        read_at(endpoint, osi.InputPins(prepared.position))
    # With its versions, the earlier state reads exactly (the file agrees).
    assert read_at(endpoint, prepared.pins).digest() == read_at(path, prepared.pins).digest()


# ---------------------------------------------------------------------------
# C4 / S5: a file a server holds is bound by its endpoint
# ---------------------------------------------------------------------------

def test_a_served_file_refuses_writers_and_names_the_endpoint(served, tmp_path):
    endpoint, path = served
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='served by vc_object_server'):
        spiral_service.bind_object_store(path, 'svc')
    # The command-line fit's writable open (a link refresh) is refused too;
    # a read-only open of the file still works beside the server.
    with pytest.raises(Exception):
        with osi.open_store(fit_config(path, 'fitter')):
            pass
    with osi.open_store(fit_config(path, 'fitter'), read_only=True) as store:
        assert store.status()['commit_sequence'] == head(path)


# ---------------------------------------------------------------------------
# S3: the service bound to the endpoint
# ---------------------------------------------------------------------------

def service(host, location, **kwargs):
    return spiral_service.ServiceState(
        bindings=host.bindings,
        object_store=spiral_service.bind_object_store(location, 'spiral-test', **kwargs),
        startup_run={'z_begin': 0, 'z_end': 10, 'config': {
            'dense_spacing_mode': 'grad_mag', 'loss_weight_dense_spacing': 0,
            'loss_weight_dense_normals': 0, 'loss_weight_shell_outer': 0}})


@pytest.fixture()
def host(tmp_path):
    output = tmp_path / 'output'
    output.mkdir()
    return SimpleNamespace(output=output, cache=tmp_path / 'cache',
                           bindings=spiral_service.bind_service_inputs(output, tmp_path / 'cache'))


def test_the_service_binds_the_endpoint(served, host):
    endpoint, path = served
    with pytest.raises(spiral_service.ObjectStoreBindingError, match='cannot attach'):
        spiral_service.bind_object_store('http://127.0.0.1:1', 'svc')
    state = service(host, endpoint, link_timeout_s=7)
    binding = state.object_store
    with ObjectStore.open(path, read_only=True) as db:
        status = db.status()
    assert binding.served and binding.location == endpoint
    assert binding.uuid == status['store'] and binding.generation == status['generation']
    assert binding.config() == {'object_store_location': endpoint, 'object_store_author': 'spiral-test',
                                'object_store_link_timeout_s': 7, 'object_store_input_position': None}
    assert binding.head() == (status['commit_sequence'], None)
    advertised = state.binding_advertisement()
    assert advertised['object_store'] == {
        'uuid': binding.uuid, 'generation': binding.generation, 'location': endpoint,
        'served': True, 'author': 'spiral-test', 'commit_sequence': status['commit_sequence'],
        'position_error': None}
    assert advertised['capabilities']['input_positions'] and advertised['capabilities']['input_updates']
    assert advertised['capabilities']['inputs_frozen'] is False
    # The selection is read through the endpoint and follows its commits.
    inputs = state.startup_inputs()
    assert inputs['ok'] and inputs['store']['location'] == endpoint
    assert {kind: entry['count'] for kind, entry in inputs['selection'].items()} == {
        'surface': 2, 'point_collection': 1, 'fiber': 0, 'track': 0}
    assert state.startup_inputs() is inputs
    near = names(path)['near']
    with binding.open() as store:
        set_meta(store, near, 'sampling_group', 'x', request='3f2a6c1d-9e8b-4a7c-8d6e-5f4a3b2c1d0e')
    assert state.startup_inputs() is not inputs
    # A position the client saved through the endpoint is accepted by its receipt.
    position = state._inputs_position({
        'store': binding.uuid, 'generation': binding.generation, 'commit_sequence': head(path),
        'request_uuid': '3f2a6c1d-9e8b-4a7c-8d6e-5f4a3b2c1d0e'})
    assert position.commit_sequence == head(path)
    with pytest.raises(Exception, match='not in the service'):
        state._inputs_position({
            'store': binding.uuid, 'generation': binding.generation, 'commit_sequence': head(path),
            'request_uuid': '00000000-0000-4000-8000-000000000000'})


def test_commits_write_to_the_served_store(served, tmp_path):
    from test_service_commits import _committed, _fit, commit
    endpoint, path = served
    fit = _fit(spiral_service.bind_object_store(endpoint, 'svc'), tmp_path)
    try:
        result = commit(fit, fit.generation())
        assert result['state'] == 'done', result
        [surface] = _committed(path)
        assert surface['uuid'] == result['surface_uuid']
        assert result['version'] >= 1 and result['reused'] is False
        with ObjectStore.open(path, read_only=True) as db:
            assert db.current_geometry(surface['uuid'])['uuid'] == result['geometry_revision']
            assert json.loads(json.dumps(db.object(surface['uuid'])['meta']))['input_position'] \
                == fit.inputs.manifest['position']
        again = commit(fit, fit.generation())
        assert again['reused'] and again['surface_uuid'] == surface['uuid']
    finally:
        fit.state.close()
