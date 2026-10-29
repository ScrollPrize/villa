"""A resident session's object-store inputs, through the real runtime.

The service builds a session through spiral_runtime and fit_spiral on the
CPU, with only the device stages (model, optimiser, brick pools) stubbed:
the context reads the store the service is bound to, the adapter's report
reaches status and events, a model-only rebuild keeps the frozen inputs,
input changes are refused, and every rank starts or none does.
"""
import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
import spiral_service  # noqa: E402
from service_project import with_project  # noqa: E402
from fit_session import SessionState, resolve_dataset_root  # noqa: E402
from test_object_store_startup import _edges, store_path  # noqa: E402,F401

RUN = {key: value for key, value in reference.CONFIG_OVERRIDES.items()
       if key not in ('z_begin', 'z_end')}


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def stub_device_stages(fitter):
    """Leave everything but the CUDA-only stages real."""
    def build(self):
        self.start_iteration = 0
    fitter.FitContext.build_device_state = build
    fitter.FitContext.rebuild_model_state = build
    fitter.FitContext.release_setup_only_tracks = lambda self: None


@pytest.fixture()
def device_stubs(monkeypatch):
    import fit_spiral
    for name in ('build_device_state', 'rebuild_model_state',
                 'release_setup_only_tracks'):
        monkeypatch.setattr(fit_spiral.FitContext, name,
                            getattr(fit_spiral.FitContext, name))
    stub_device_stages(fit_spiral)
    loads = []
    load = osi.load_fit_inputs

    def counted(config, **kwargs):
        loads.append(config['object_store_refresh_links'])
        return load(config, **kwargs)
    monkeypatch.setattr(osi, 'load_fit_inputs', counted)
    return loads


def write_dataset(root):
    root.mkdir()
    spec = reference.scroll_spec()
    (root / 'spiral-scroll.json').write_text(json.dumps({
        'schema_version': 1, 'name': spec.name,
        'voxel_size_um': spec.voxel_size_um,
        'spiral_outward_sense': spec.spiral_outward_sense,
        'base_shape_zyx': list(spec.base_shape_zyx)}))
    reference.write_umbilicus(str(root / 'umbilicus.json'))
    return root


@pytest.fixture()
def state(store_path, tmp_path):
    root = write_dataset(tmp_path / 'dataset')
    output = tmp_path / 'output'
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(root), output, tmp_path / 'cache')
    state = spiral_service.ServiceState(
        dataset_root=str(root), dataset_resolution=resolution,
        object_store=spiral_service.bind_object_store(store_path, 'svc'),
        startup_run={'z_begin': 0, 'z_end': 64, 'config': dict(RUN)})
    yield state
    state.close()


def settle(state, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = state.status()
        if not state._building and status['state'] in ('Idle', 'Error'):
            return status
        time.sleep(0.05)
    raise AssertionError(f'the session did not settle: {state.status()["phase"]}')


def test_the_runtime_reads_the_bound_store_and_reports_it(state, store_path,
                                                          device_stubs):
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}))
    status = settle(state)
    assert status['state'] == 'Idle', status['error']
    context = state.session._context
    assert context.config['object_store_path'] == state.object_store.path
    assert context.config['object_store_author'] == 'svc'
    assert device_stubs == [True]
    assert len(_edges(store_path)) == 3
    inputs = context.object_store_inputs
    # The service linked and pinned a position of the shared store first
    # (its lines are preparation events); the runtime read that position.
    preparing = [event['text'] for event in state.events.read_after(0)['events']
                 if event.get('operation') == 'preparing_inputs']
    assert any(line.startswith('link refresh pcl_patch: 3 created') for line in preparing)
    assert status['input_position']['source'] == 'prepared'
    # The adapter's report is in the status and in the event stream.
    report = status['input_report']
    assert any(line.startswith('reading pinned position') for line in report)
    assert f'object-store inputs frozen: digest {inputs.digest[:16]}' in report
    assert any(line.startswith('constraints:') for line in report)
    events = [event for event in state.events.read_after(0)['events']
              if event['kind'] == 'inputs']
    assert [event['text'] for event in events] == report
    # The interactive run directory records the frozen inputs, as a CLI run's does.
    with open(Path(context.out_path) / 'object_store_inputs.json') as stream:
        assert json.load(stream)['digest'] == inputs.digest


def test_status_and_artifacts_carry_the_input_provenance(state, device_stubs):
    from fit_session import read_autosave_metadata
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}))
    status = settle(state)
    inputs = state.session._context.object_store_inputs
    assert status['input_digest'] == inputs.digest
    summary = status['input_summary']
    assert summary['digest'] == inputs.digest
    assert summary['selection'] == {'surface': 2, 'point_collection': 1, 'fiber': 0}
    assert summary['relationships'] == {'pcl_patch': 3}
    # The status lists the inputs read by path, and the store's by digest.
    manifest = status['input_manifest']
    assert manifest['object_store_digest'] == inputs.digest
    assert not {'verified_patches', 'fibers', 'pcls'} & set(manifest)
    # The project's umbilicus, written where the fit reads it.
    assert '/project-inputs/umbilicus-' in manifest['umbilicus']
    assert not {'same_winding_artifact', 'relative_winding_artifact'} & set(status)
    # The whole manifest is an immutable artifact.
    deadline = time.monotonic() + 30
    while state.status()['input_manifest_artifact'] is None:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    artifact = state.status()['input_manifest_artifact']
    assert artifact['input_digest'] == inputs.digest
    _, path, _ = state.artifacts.acquire_file(artifact['id'], 'object_store_inputs.json')
    with open(path) as stream:
        published = json.load(stream)
    assert published == {'digest': inputs.digest,
                         'manifest': json.loads(json.dumps(inputs.manifest))}
    # Autosaves name the inputs they were fitted to. (The stubbed device
    # stages leave no model to save, so the archive itself is a stand-in.)
    session = state.session

    def save_checkpoint(path, completed):
        Path(path).write_bytes(b'PK\x03\x04')
        return path
    session._context.save_checkpoint = save_checkpoint
    autosave = Path(session._write_autosave())
    metadata = json.loads(autosave.with_name('checkpoint_autosave.json').read_text())
    assert metadata['input_digest'] == inputs.digest
    assert read_autosave_metadata(autosave.with_name('checkpoint_autosave.json'))


def test_a_model_rebuild_keeps_the_frozen_inputs(state, store_path, device_stubs):
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}))
    settle(state)
    context = state.session._context
    inputs = context.object_store_inputs
    edges = _edges(store_path)
    response = state.rebuild(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': {
        **RUN, 'model_num_flow_stages': 3}}}))
    assert response['stage'] == 'model'
    assert settle(state)['state'] == 'Idle'
    assert state.session._context is context
    assert context.object_store_inputs is inputs
    assert context.config['model_num_flow_stages'] == 3
    # Nothing reread the store, refreshed links or wrote to it.
    assert device_stubs == [True]
    assert _edges(store_path) == edges


def test_input_refusals_reach_status_and_events(state, store_path, device_stubs):
    # Linking a position fails before any build: nothing is torn down.
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': {
        **RUN, 'pcl_link_window_min_points': 5}}}))
    deadline = time.monotonic() + 60
    while state._building:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    status = state.status()
    assert status['state'] == 'Uninitialized' and device_stubs == []
    failure = status['input_preparation_error']
    assert any('pcl_patch: FAILED' in line for line in failure['problems'])
    assert 'LinkRefreshError' in failure['message']
    [error] = [event for event in state.events.read_after(0)['events']
               if event['kind'] == 'error']
    assert error['text'].startswith('The inputs could not be prepared')


# ---------------------------------------------------------------------------
# Two ranks
# ---------------------------------------------------------------------------

def _rank_session(rank, world_size, rendezvous, root, store, config, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import fit_spiral
    from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig
    from spiral_runtime import InteractiveFitSession, RendezvousEndpoint
    stub_device_stages(fit_spiral)
    statuses = []
    session = InteractiveFitSession(
        SpiralInputPaths(dataset_root=root, umbilicus=f'{root}/umbilicus.json',
                         output_directory=f'{root}-out-{rank}'),
        SpiralRunConfig(0, 64, config=config), SpiralPreviewConfig(),
        reference.scroll_spec(), statuses.append, publishes_outputs=rank == 0,
        rank=rank, world_size=world_size, local_rank=rank,
        rendezvous=RendezvousEndpoint(rendezvous))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = session.status()
        if status['state'] in ('Idle', 'Error'):
            break
        time.sleep(0.05)
    states = [str(entry['state']) for entry in statuses]
    with open(results[rank], 'w') as stream:
        json.dump({'state': str(status['state']), 'error': status['error'],
                   'input_problems': status['input_problems'],
                   'ever_idle': 'Idle' in states,
                   'report': status['input_report']}, stream)
    session.close()


def _two_rank_sessions(tmp_path, store, config):
    import torch.multiprocessing as mp
    root = str(write_dataset(tmp_path / 'dataset'))
    results = [str(tmp_path / f'rank{rank}.json') for rank in range(2)]
    rendezvous = tmp_path / 'rendezvous'
    rendezvous.mkdir()
    mp.spawn(_rank_session, args=(2, str(rendezvous / 'store'), root, store,
                                  config, results), nprocs=2)
    outcome = []
    for path in results:
        with open(path) as stream:
            outcome.append(json.load(stream))
    return outcome


def _bound_config(store, **overrides):
    return {**RUN, 'object_store_path': store, 'object_store_author': 'svc',
            **overrides}


def test_two_rank_sessions_refresh_once_and_both_start(store_path, tmp_path):
    first, second = _two_rank_sessions(tmp_path, store_path, _bound_config(store_path))
    assert first['state'] == second['state'] == 'Idle', (first, second)
    assert len(_edges(store_path)) == 3
    # Rank 0 refreshed and reported; rank 1 only read the frozen inputs.
    assert any('3 created' in line for line in first['report'])
    assert not any('link refresh' in line for line in second['report'])


def test_a_refresh_failure_stops_both_ranks_before_either_starts(store_path, tmp_path):
    outcome = _two_rank_sessions(tmp_path, store_path, _bound_config(
        store_path, pcl_link_window_min_points=5))
    for rank in outcome:
        assert rank['state'] == 'Error' and not rank['ever_idle']
        assert any('pcl_patch: FAILED' in line for line in rank['input_problems'])
    assert outcome[0]['input_problems'] == outcome[1]['input_problems']


# ---------------------------------------------------------------------------
# Rebuilds and checkpoints against frozen inputs (step 8)
# ---------------------------------------------------------------------------

def test_a_reload_rebuild_rereads_the_store_under_an_unchanged_config(
        state, store_path, device_stubs):
    request = {'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}
    state.initialize(with_project(request))
    first = settle(state)['input_digest']
    walk = next(entry['uuid'] for entry in state.dataset()['inputs']['objects']
                if entry['name'] == 'walk')
    with ObjectStore.open(store_path, author='editor') as db:
        db.set_meta(walk, 'sampling_group', 'reviewed')
    # The same configuration is a model-stage rebuild, which keeps the
    # frozen inputs: it neither rereads the store nor sees the edit.
    response = state.rebuild(with_project(request))
    assert (response['stage'], response['inputs']) == ('model', 'keep')
    assert settle(state)['input_digest'] == first
    assert device_stubs == [True]
    response = state.rebuild(with_project({**request, 'inputs': 'reload'}))
    assert (response['stage'], response['inputs']) == ('all', 'reload')
    reloaded = settle(state)
    assert reloaded['state'] == 'Idle'
    assert reloaded['input_digest'] != first
    assert len(device_stubs) == 2
    with pytest.raises(spiral_service.ApiError, match='inputs must be'):
        state.rebuild(with_project({**request, 'inputs': 'refresh'}))


def test_settings_that_shape_the_inputs_need_a_full_rebuild(state, device_stubs):
    from config import OBJECT_STORE_PCL_KEYS, Config, rebuild_stage
    keys = sorted(OBJECT_STORE_PCL_KEYS) + ['input_use_pcl_relative',
                                            'input_use_pcl_same_winding',
                                            'patch_loss_z_margin']
    for key in keys:
        assert rebuild_stage({key}) == 'all', key
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}))
    status = settle(state)
    configuration = {**Config().as_dict(), **status['applied_config'],
                     'pcl_link_window_points': 9}
    with pytest.raises(spiral_service.ApiError, match='requires rebuilding'):
        state.run({'iterations': 1, 'configuration': configuration,
                   'expected_session_revision': state.session_revision,
                   'dt_loss_schedule': {'enabled': False, 'last_fraction': 0.0}})
    response = state.rebuild(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': {
        **RUN, 'pcl_link_window_points': 9}}}))
    assert (response['stage'], response['inputs']) == ('all', 'reload')
    assert settle(state)['applied_config']['pcl_link_window_points'] == 9
    assert len(device_stubs) == 2


def _checkpoint(path, cfg, input_manifest):
    import torch
    torch.save({'cfg': cfg, 'input_manifest': input_manifest}, path)
    return str(path)


def test_checkpoint_refusals_name_the_input_remedies(state, tmp_path, device_stubs):
    state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': dict(RUN)}}))
    status = settle(state)
    live = status['applied_config']
    with open(status['input_manifest_path']) as stream:
        manifest = json.load(stream)['manifest']

    changed = json.loads(json.dumps(manifest))
    changed['objects'][0]['geometry_revision'] = 'an-older-geometry'
    # Written against another store path: that is the binding's to decide.
    cfg = {**live, 'object_store_path': '/elsewhere.vcobj'}
    path = _checkpoint(tmp_path / 'changed.ckpt', cfg,
                       {'object_store': changed, 'dataset_root': state.dataset_root})
    refusal = state._checkpoint_refusal(path, 'object-store inputs differ')
    assert refusal.status == 409
    assert refusal.payload['stage'] == 'all'
    inputs = refusal.payload['inputs']
    assert inputs['legacy'] is False
    assert 'objects' in inputs['problems'][0]
    assert inputs['rebuild'] == {'inputs': 'reload'}
    assert inputs['allow_changed_inputs'] == 'object_store_allow_changed_inputs'

    # The same inputs: only the configuration decides the stage, and the
    # store path the checkpoint names is no difference.
    path = _checkpoint(tmp_path / 'same.ckpt',
                       {**cfg, 'model_num_flow_stages': live['model_num_flow_stages'] + 1},
                       {'object_store': manifest, 'dataset_root': state.dataset_root})
    refusal = state._checkpoint_refusal(path, 'model shape differs')
    assert refusal.payload == {'reasons': ['model shape differs'], 'stage': 'model'}

    path = _checkpoint(tmp_path / 'legacy.ckpt', live,
                       {'dataset_root': state.dataset_root})
    refusal = state._checkpoint_refusal(path, 'no object-store inputs')
    assert refusal.payload['refused'] is True
    assert refusal.payload['inputs']['legacy'] is True
    assert 'object_store_allow_changed_inputs' in refusal.message


# ---------------------------------------------------------------------------
# The CLI and the service freeze the same inputs (step 12)
# ---------------------------------------------------------------------------

def _plane(y, scale=1):
    import numpy as np
    rows, cols = np.mgrid[:8, :8]
    return np.stack([4.0 + rows * 4, np.full(rows.shape, y), 40.0 + cols * 4], -1)


@pytest.fixture()
def varied_store(tmp_path):
    """Every case the selection must agree on: absolute, relative and
    same-winding PCLs by metadata and annotations, a patch and a fiber at
    non-unit scales, and approved, pending, rejected and stale links."""
    import numpy as np
    from object_store_fixture import (array, frame, pcl_patch_edge, put_fiber,
                                      put_patch, put_pcl)
    path = str(tmp_path / 'varied.vcobj')
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        near = put_patch(db, _plane(100.0), (0.25, 0.25), name='near')
        far = put_patch(db, _plane(140.0), (0.25, 0.25), coordinates=frame(2),
                        name='far')
        absolute = put_pcl(db, {1: ([12.0, 100.0, 48.0], 4.0)}, absolute=True,
                           name='absolute')
        relative = put_pcl(db, {3: ([16.0, 100.0, 52.0], 1.0),
                                9: ([20.0, 140.0, 56.0], 2.0)}, name='relative')
        same = put_pcl(db, {5: ([24.0, 100.0, 60.0], None),
                            6: ([28.0, 100.0, 64.0], None)}, name='same')
        db.set_approval(pcl_patch_edge(db, absolute, 1, near, (2.0, 2.0)), 'approved')
        pcl_patch_edge(db, relative, 3, near, (3.0, 3.0))
        db.set_approval(pcl_patch_edge(db, same, 5, near, (5.0, 5.0)), 'rejected')
        pcl_patch_edge(db, same, 6, near, (6.0, 6.0))
        pcl_patch_edge(db, relative, 9, far, (4.0, 4.0))
        put_fiber(db, [[10, 120, 40], [14, 120, 44], [18, 120, 48]], [0, 2],
                  coordinates=frame(2), name='fiber')
        # Moving the far patch while keeping its links bound to the recorded
        # geometry leaves the link to it stale.
        moved = {'kind': 'surface', 'attributes': {'scale': [0.125, 0.125]},
                 'arrays': [array('points', _plane(141.0)[..., ::-1] * 2, '<f4')]}

        def keep_recorded(edge):
            edge = db.edge(edge['uuid'])
            return {'edge_uuid': edge['uuid'],
                    'expected_generation': edge['generation'],
                    'action': 'keep_recorded',
                    'a_revision': db.current_geometry(edge['a'])['uuid'],
                    'b_revision': db.current_geometry(edge['b'])['uuid']}
        db.put_geometry(far, moved, expected_revision=db.current_geometry(far)['uuid'],
                        edge_edits=[keep_recorded(edge) for edge in
                                    db.affected_geometry_edges(far, moved)])
    return path


def _cli_inputs(store, tmp_path, config):
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    umbilicus = str(tmp_path / 'cli-umbilicus.json')
    reference.write_umbilicus(umbilicus)
    context = FitContext(
        FitConfig({**reference.fit_config(), **config}),
        scroll=reference.scroll_spec(), paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cli-cache'), out_base_dir=str(tmp_path / 'cli-out'),
        input_log=lambda line: None)
    context.load_host_inputs()
    return context


@pytest.mark.parametrize('selection', [
    {},
    {'object_store_approved_links_only': True},
    {'object_store_fiber_uuids': [], 'object_store_pcl_tags': []},
])
def test_the_cli_and_the_service_freeze_the_same_inputs(varied_store, tmp_path,
                                                        device_stubs, selection):
    root = write_dataset(tmp_path / 'dataset')
    resolution = spiral_service.bind_service_paths(
        resolve_dataset_root(root), tmp_path / 'output', tmp_path / 'cache')
    state = spiral_service.ServiceState(
        dataset_root=str(root), dataset_resolution=resolution,
        object_store=spiral_service.bind_object_store(varied_store, 'svc'))
    config = {**RUN, 'object_store_refresh_links': False, **selection}
    try:
        state.initialize(with_project({'run': {'z_begin': 0, 'z_end': 64, 'config': config}}))
        assert settle(state)['state'] == 'Idle', state.status()['error']
        served = state.session._context
        cli = _cli_inputs(varied_store, tmp_path, {
            **config, 'object_store_path': varied_store, 'object_store_author': 'cli'})
    finally:
        state.close()
    assert served.object_store_inputs.digest == cli.object_store_inputs.digest
    assert served.object_store_inputs.counts == cli.object_store_inputs.counts
    assert len(served.cross_patch_pcls) == len(cli.cross_patch_pcls)
    assert len(served.unattached_pcl_strips) == len(cli.unattached_pcl_strips)
    manifest = served.object_store_inputs.manifest
    excluded = manifest['excluded_relationships']
    scales = {entry['name']: entry['scale_factor'] for entry in manifest['objects']}
    assert scales['far'] == 2.0
    if 'object_store_fiber_uuids' in selection:
        # Explicitly empty selectors select nothing, not everything.
        assert 'fiber' not in scales and 'absolute' not in scales
        assert manifest['selection']['fiber']['count'] == 0
        assert manifest['relationships'] == []
    else:
        assert excluded['inapplicable'] == 1 and excluded['rejected'] == 1
        if selection.get('object_store_approved_links_only'):
            assert excluded['pending (approved-only)'] == 2
        assert scales['fiber'] == 2.0
        roles = {pcl['name']: pcl['metadata']['winding_is_absolute']
                 for pcl in served.object_store_inputs.point_collections.values()}
        assert roles == {'absolute': True, 'relative': False, 'same': False}
