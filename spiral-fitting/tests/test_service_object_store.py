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
    (root / 'verified_patches').mkdir()
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
                                'object_store_author': 'svc'}
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
