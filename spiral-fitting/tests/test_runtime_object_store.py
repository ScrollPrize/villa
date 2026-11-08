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

import object_store_reference as reference  # noqa: E402
import spiral_service  # noqa: E402
from cpu_service import (  # noqa: E402
    RUN, request, service_bindings, settle, stub_device_stages, two_ranks,
    wait_until, write_inputs)
from object_store_fixture import edges  # noqa: E402
from service_project import with_project  # noqa: E402

pytestmark = pytest.mark.usefixtures('cpu')


@pytest.fixture()
def state(store_path, tmp_path):
    state = spiral_service.ServiceState(
        bindings=service_bindings(tmp_path),
        object_store=spiral_service.bind_object_store(store_path, 'svc'),
        startup_run={'z_begin': 0, 'z_end': 64, 'config': dict(RUN)})
    yield state
    state.close()


def test_the_runtime_reads_the_bound_store_and_reports_it(state, store_path,
                                                          device_stubs):
    state.initialize(with_project(request()))
    status = settle(state)
    assert status['state'] == 'Idle', status['error']
    context = state.session._context
    assert context.config['object_store_path'] == state.object_store.path
    assert context.config['object_store_author'] == 'svc'
    assert device_stubs == [True]
    assert len(edges(store_path)) == 3
    inputs = context.object_store_inputs
    # The service linked and pinned a position of the shared store first
    # (its lines are preparation events); the runtime read that position.
    preparing = [event['text'] for event in state.events.read_after(0)['events']
                 if event.get('operation') == 'preparing_inputs']
    assert any(line.startswith('link refresh pcl_patch: 3 created') for line in preparing)
    assert status['input_position']['source'] == 'prepared'
    # The adapter's report is in the status and in the event stream.
    report = status['input_report']
    assert any(line.startswith('reading pinned commit') for line in report)
    assert f'object-store inputs frozen: digest {inputs.digest[:16]}' in report
    events = [event for event in state.events.read_after(0)['events']
              if event['kind'] == 'inputs']
    assert [event['text'] for event in events] == report
    # The status carries the input provenance: the store's by digest, the
    # project's umbilicus by path.
    assert status['input_digest'] == inputs.digest
    summary = status['input_summary']
    assert summary['selection'] == {'surface': 2, 'point_collection': 1, 'fiber': 0,
                                    'track': 0}
    assert summary['relationships'] == {'pcl_patch': 3}
    manifest = status['input_manifest']
    assert manifest['object_store_digest'] == inputs.digest
    assert not {'verified_patches', 'fibers', 'pcls'} & set(manifest)
    assert '/project-inputs/umbilicus-' in manifest['umbilicus']
    # The run directory records the frozen inputs, as a CLI run's does, and
    # the whole manifest is an immutable artifact.
    with open(Path(context.out_path) / 'object_store_inputs.json') as stream:
        assert json.load(stream)['digest'] == inputs.digest
    wait_until(lambda: state.status()['input_manifest_artifact'] is not None)
    artifact = state.status()['input_manifest_artifact']
    assert artifact['input_digest'] == inputs.digest
    _, path, _ = state.artifacts.acquire_file(artifact['id'], 'object_store_inputs.json')
    with open(path) as stream:
        assert json.load(stream) == {'digest': inputs.digest,
                                     'manifest': json.loads(json.dumps(inputs.manifest))}
    # Autosaves name the inputs they were fitted to. (The stubbed device
    # stages leave no model to save, so the archive itself is a stand-in.)
    def save_checkpoint(path, completed):
        Path(path).write_bytes(b'PK\x03\x04')
        return path
    context.save_checkpoint = save_checkpoint
    autosave = Path(state.session._write_autosave())
    metadata = json.loads(autosave.with_name('checkpoint_autosave.json').read_text())
    assert metadata['input_digest'] == inputs.digest
    assert metadata['input_position'] == inputs.manifest['position']


def test_input_refusals_reach_status_and_events(state, store_path, device_stubs):
    # Linking a position fails before any build: nothing is torn down.
    state.initialize(with_project(request(pcl_link_window_min_points=5)))
    wait_until(lambda: not state._building, 60)
    status = state.status()
    assert status['state'] == 'Uninitialized' and device_stubs == []
    failure = status['input_preparation_error']
    assert any('point-collection-surface' in line and 'nothing was written' in line
               for line in failure['problems'])
    assert 'LinkRefreshError' in failure['message']
    [error] = [event for event in state.events.read_after(0)['events']
               if event['kind'] == 'error']
    assert error['text'].startswith('The inputs could not be prepared')


# ---------------------------------------------------------------------------
# Two ranks
# ---------------------------------------------------------------------------

def _rank_session(rank, world_size, rendezvous, root, config, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import fit_spiral
    from fit_session import SpiralInputPaths, SpiralPreviewConfig, SpiralRunConfig
    from spiral_runtime import InteractiveFitSession, RendezvousEndpoint
    stub_device_stages(fit_spiral)
    statuses = []
    session = InteractiveFitSession(
        SpiralInputPaths(umbilicus=f'{root}/umbilicus.json',
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


def _two_rank_sessions(tmp_path, store, **overrides):
    config = {**RUN, 'object_store_path': store, 'object_store_author': 'svc', **overrides}
    return two_ranks(_rank_session, tmp_path, str(write_inputs(tmp_path / 'inputs')), config)


def test_two_rank_sessions_refresh_once_and_both_start(store_path, tmp_path):
    first, second = _two_rank_sessions(tmp_path, store_path)
    assert first['state'] == second['state'] == 'Idle', (first, second)
    assert len(edges(store_path)) == 3
    # Rank 0 refreshed and reported; rank 1 only read the frozen inputs.
    assert any('3 created' in line for line in first['report'])
    assert not any('link refresh' in line for line in second['report'])


def test_a_refresh_failure_stops_both_ranks_before_either_starts(store_path, tmp_path):
    outcome = _two_rank_sessions(tmp_path, store_path, pcl_link_window_min_points=5)
    for rank in outcome:
        assert rank['state'] == 'Error' and not rank['ever_idle']
        assert any('point-collection-surface' in line for line in rank['input_problems'])
    assert outcome[0]['input_problems'] == outcome[1]['input_problems']


# ---------------------------------------------------------------------------
# Rebuilds and checkpoints against frozen inputs
# ---------------------------------------------------------------------------

def test_a_model_rebuild_keeps_the_inputs_and_a_reload_rereads_them(
        state, store_path, device_stubs):
    state.initialize(with_project(request()))
    first = settle(state)['input_digest']
    context = state.session._context
    inputs, linked = context.object_store_inputs, edges(store_path)
    walk = next(entry['uuid'] for entry in state.startup_inputs()['objects']
                if entry['name'] == 'walk')
    with ObjectStore.open(store_path, author='editor') as db:
        db.set_meta(walk, 'sampling_group', 'reviewed')
    # A model-stage rebuild keeps the frozen inputs: it neither rereads the
    # store, refreshes links nor sees the edit.
    response = state.rebuild(with_project(request(model_num_flow_stages=3)))
    assert (response['stage'], response['inputs']) == ('model', 'keep')
    assert settle(state)['input_digest'] == first
    assert state.session._context is context and context.object_store_inputs is inputs
    assert context.config['model_num_flow_stages'] == 3
    assert device_stubs == [True] and edges(store_path) == linked
    response = state.rebuild(with_project({**request(), 'inputs': 'reload'}))
    assert (response['stage'], response['inputs']) == ('all', 'reload')
    reloaded = settle(state)
    assert reloaded['state'] == 'Idle' and reloaded['input_digest'] != first
    assert len(device_stubs) == 2
    with pytest.raises(spiral_service.ApiError, match='inputs must be'):
        state.rebuild(with_project({**request(), 'inputs': 'refresh'}))


def test_settings_that_shape_the_inputs_need_a_full_rebuild(state, device_stubs):
    from config import OBJECT_STORE_PCL_KEYS, Config, rebuild_stage
    for key in sorted(OBJECT_STORE_PCL_KEYS) + ['input_use_pcl_relative',
                                                'input_use_pcl_same_winding',
                                                'patch_loss_z_margin']:
        assert rebuild_stage({key}) == 'all', key
    state.initialize(with_project(request()))
    status = settle(state)
    configuration = {**Config().as_dict(), **status['applied_config'],
                     'pcl_link_window_points': 9}
    with pytest.raises(spiral_service.ApiError, match='requires rebuilding'):
        state.run({'iterations': 1, 'configuration': configuration,
                   'expected_session_revision': state.session_revision,
                   'dt_loss_schedule': {'enabled': False, 'last_fraction': 0.0}})
    response = state.rebuild(with_project(request(pcl_link_window_points=9)))
    assert (response['stage'], response['inputs']) == ('all', 'reload')
    assert settle(state)['applied_config']['pcl_link_window_points'] == 9
    assert len(device_stubs) == 2


def _checkpoint(path, cfg, input_manifest):
    import torch
    torch.save({'cfg': cfg, 'input_manifest': input_manifest}, path)
    return str(path)


def test_checkpoint_refusals_name_the_input_remedies(state, tmp_path, device_stubs):
    state.initialize(with_project(request()))
    status = settle(state)
    live, volume = status['applied_config'], state.session_paths.volume_uuid
    assert volume
    with open(status['input_manifest_path']) as stream:
        manifest = json.load(stream)['manifest']

    def refusal(name, cfg, input_manifest, reason):
        return state._checkpoint_refusal(_checkpoint(tmp_path / name, cfg, input_manifest),
                                         reason)

    # Changed inputs (written against another store path: that is the
    # binding's to decide) need a reload.
    changed = json.loads(json.dumps(manifest))
    changed['objects'][0]['geometry_revision'] = 'an-older-geometry'
    cfg = {**live, 'object_store_path': '/elsewhere.vcobj'}
    refused = refusal('changed.ckpt', cfg, {'object_store': changed, 'volume_uuid': volume},
                      'object-store inputs differ')
    assert (refused.status, refused.payload['stage']) == (409, 'all')
    inputs = refused.payload['inputs']
    assert inputs['legacy'] is False and 'objects' in inputs['problems'][0]
    assert inputs['rebuild'] == {'inputs': 'reload'}
    assert inputs['allow_changed_inputs'] == 'object_store_allow_changed_inputs'
    # The same inputs: only the configuration decides the stage.
    refused = refusal('same.ckpt', {**cfg, 'model_num_flow_stages': live['model_num_flow_stages'] + 1},
                      {'object_store': manifest, 'volume_uuid': volume}, 'model shape differs')
    assert refused.payload == {'reasons': ['model shape differs'], 'stage': 'model'}
    refused = refusal('legacy.ckpt', live, {'volume_uuid': volume}, 'no object-store inputs')
    assert refused.payload['refused'] is True and refused.payload['inputs']['legacy'] is True
    assert 'object_store_allow_changed_inputs' in refused.message
    # A checkpoint fitted in another volume is what no rebuild can fix.
    refused = refusal('foreign.ckpt', live, {'object_store': manifest,
                                             'volume_uuid': 'another-volume'}, 'another volume')
    assert refused.payload == {'reasons': ['another volume'], 'refused': True}


# ---------------------------------------------------------------------------
# The CLI and the service freeze the same inputs
# ---------------------------------------------------------------------------

@pytest.fixture()
def varied_store(tmp_path):
    """Every case the selection must agree on: absolute, relative and
    same-winding PCLs by metadata and annotations, a patch and a fiber at
    non-unit scales, and approved, pending, rejected and stale links."""
    from object_store_fixture import (array, frame, pcl_patch_edge, plane, put_fiber,
                                      put_patch, put_pcl)
    path = str(tmp_path / 'varied.vcobj')
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        near = put_patch(db, plane(100.0), (0.25, 0.25), name='near')
        far = put_patch(db, plane(140.0), (0.25, 0.25), coordinates=frame(2), name='far')
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
                 'arrays': [array('points', plane(141.0)[..., ::-1] * 2, '<f4')]}

        def keep_recorded(edge):
            edge = db.edge(edge['uuid'])
            return {'edge_uuid': edge['uuid'],
                    'expected_generation': edge['generation'],
                    'action': 'keep_recorded',
                    'a_revision': db.current_geometry(edge['a'])['uuid'],
                    'b_revision': db.current_geometry(edge['b'])['uuid']}
        db.put_geometry(far, moved, expected_version=db.current_version(far),
                        edge_edits=[keep_recorded(edge) for edge in
                                    db.affected_geometry_edges(far, moved)])
    return path


@pytest.mark.parametrize('selection', [
    {'object_store_approved_links_only': True},
    {'object_store_fiber_uuids': [], 'object_store_pcl_tags': []},
])
def test_the_cli_and_the_service_freeze_the_same_inputs(varied_store, tmp_path,
                                                        device_stubs, selection):
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    state = spiral_service.ServiceState(
        bindings=service_bindings(tmp_path),
        object_store=spiral_service.bind_object_store(varied_store, 'svc'))
    config = {**RUN, 'object_store_refresh_links': False, **selection}
    try:
        state.initialize(with_project(request(**config)))
        assert settle(state)['state'] == 'Idle', state.status()['error']
        served = state.session._context
        umbilicus = str(tmp_path / 'cli-umbilicus.json')
        reference.write_umbilicus(umbilicus)
        cli = FitContext(
            FitConfig({**reference.fit_config(), **config, 'object_store_path': varied_store,
                       'object_store_author': 'cli'}),
            scroll=reference.scroll_spec(), paths=SpiralInputPaths(umbilicus=umbilicus),
            cache_dir=str(tmp_path / 'cli-cache'), out_base_dir=str(tmp_path / 'cli-out'),
            input_log=lambda line: None)
        cli.load_host_inputs()
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
        assert excluded['pending (approved-only)'] == 2
        assert scales['fiber'] == 2.0
        roles = {pcl['name']: pcl['metadata']['winding_is_absolute']
                 for pcl in served.object_store_inputs.point_collections.values()}
        assert roles == {'absolute': True, 'relative': False, 'same': False}
