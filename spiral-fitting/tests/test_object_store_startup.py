"""Startup lifecycle of object-store fit inputs: link refresh, frozen reads,
rank agreement, external frames and resume validation."""
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    VOLUME, pcl_patch_edge, put_fiber, put_patch, put_pcl)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('FIT_SPIRAL_DEVICE', 'cpu')


def _grid(y=100.0, size=8):
    rows, cols = np.mgrid[:size, :size]
    return np.stack([4.0 + rows * 4, np.full(rows.shape, y), 40.0 + cols * 4],
                    axis=-1)


def _on(y, *points):
    """PCL points on the plane y, annotated with no winding."""
    return {index: ([z, y, x], None) for index, (z, x) in enumerate(points)}


@pytest.fixture()
def store_path(tmp_path):
    """A closed store with two patches and a PCL touching each of them."""
    path = str(tmp_path / 'fit.vcobj')
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        put_patch(db, _grid(), (0.25, 0.25), name='near')
        put_patch(db, _grid(140.0), (0.25, 0.25), name='far')
        put_pcl(db, {**_on(100.0, (12.0, 48.0), (16.0, 52.0)),
                     2: ([20.0, 140.0, 56.0], None)}, name='walk')
    return path


def _uuids(path):
    with ObjectStore.open(path, read_only=True) as db:
        return {record['name']: record['uuid'] for kind in osi.KINDS
                for record in db.objects_by_kind(kind)}


def _edges(path, kind='point-patch'):
    with ObjectStore.open(path, read_only=True) as db:
        return sorted((edge['uuid'], edge['a'], edge['b'], edge['approval_state'],
                       edge['generation'], edge['changed_at'])
                      for edge in db.edges_by_kind(kind))


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def fit_config(path, **overrides):
    config = reference.fit_config()
    config.update(object_store_path=path, object_store_author='fitter')
    config.update(overrides)
    return config


def load(path, log=None, **overrides):
    lines = [] if log is None else log
    return osi.load_fit_inputs(fit_config(path, **overrides), log=lines.append)


def test_default_refresh_writes_links_and_freezes_the_inputs(store_path):
    log = []
    inputs = load(store_path, log)
    edges = _edges(store_path)
    assert len(edges) == 3
    assert inputs.counts['pcl attachments'] == 3
    assert any(line.startswith('link refresh pcl_patch: 3 created') for line in log)
    assert any('fiber_patch: skipped (no selected fibers)' in line for line in log)
    assert inputs.manifest['link_refresh']['enabled']
    assert inputs.manifest['materialization']['pcl_patch_overlap_max_pairs'] == 16
    # A rerun reconciles idempotently: same relationships, same frozen inputs.
    again = load(store_path)
    assert _edges(store_path) == edges
    assert again.digest == inputs.digest
    assert any('3 unchanged' in line for line in _refresh_lines(store_path))


def _refresh_lines(path):
    log = []
    load(path, log)
    return log


def test_refresh_needs_an_author(store_path):
    with pytest.raises(osi.LinkRefreshError, match='object_store_author is required'):
        load(store_path, object_store_author=None)


def test_refresh_failures_are_collected_and_stop_the_fit(store_path, tmp_path):
    with ObjectStore.open(store_path, author='ingest') as db:
        put_fiber(db, [[12.0, 100.5, 40.0 + i] for i in range(30)], [0, 29], name='f')
    config = fit_config(store_path, pcl_link_window_points=1,
                        pcl_link_window_min_points=5)
    lasagna = {'normal_x': str(tmp_path / 'absent_x'),
               'normal_y': str(tmp_path / 'absent_y'), 'group': '4', 'scale': 4}
    with pytest.raises(osi.LinkRefreshError) as error:
        osi.load_fit_inputs(config, lasagna=lasagna, log=lambda line: None)
    text = '\n'.join(error.value.problems)
    assert 'link refresh pcl_patch: FAILED: ValueError' in text
    assert 'link refresh fiber_patch: FAILED: RuntimeError' in text
    # The independent overlap detector still ran, and its commit is reported.
    assert 'link refresh patch_overlap: 0 created' in text
    assert 'failed in pcl_patch, fiber_patch; the fit will not train' in text
    assert 'patch_overlap committed before the failure' in text
    assert _edges(store_path) == []


def test_returned_failures_are_failures_and_diagnostics_are_not():
    report = {'edges_created': 1, 'edges_updated': 0, 'edges_skipped_existing': 0,
              'stale_removed': 0, 'committed': False, 'failed': True,
              'issues': [{'severity': 'failure', 'stage': 'validate',
                          'object_uuids': ['u1', 'u2'], 'reason': 'bad frame',
                          'count': 1},
                         {'severity': 'diagnostic', 'stage': 'discover',
                          'object_uuids': [], 'reason': 'too few inliers',
                          'count': 7}]}
    failed = osi.RefreshOutcome('patch_overlap', report=report)
    lines, problems = osi.refresh_report([failed])
    assert failed.failed and problems
    assert '  failure at validate: bad frame (u1, u2)' in lines
    assert '  7 candidate diagnostic(s), e.g. too few inliers' in lines
    report = dict(report, failed=False, committed=True,
                  issues=report['issues'][1:])
    lines, problems = osi.refresh_report([osi.RefreshOutcome('patch_overlap', report=report)])
    assert problems == []


def test_no_refresh_mode_never_writes(store_path):
    before = _sha256(store_path)
    log = []
    inputs = load(store_path, log, object_store_refresh_links=False,
                  object_store_author=None)
    assert _sha256(store_path) == before
    assert _edges(store_path) == []
    assert inputs.counts['pcl attachments'] == 0
    assert "link refresh: off; reading the store's current relationships" in log
    assert not inputs.manifest['link_refresh']['enabled']
    report = '\n'.join(inputs.report())
    assert 'WARNING: no pcl_patch relationships are consumed' in report


def test_approved_only_consumes_approved_links_after_refresh(store_path):
    inputs = load(store_path, object_store_approved_links_only=True)
    assert inputs.counts['pcl attachments'] == 0
    assert inputs.snapshot.exclusions['pending (approved-only)'] == 3
    assert any('WARNING: no pcl_patch relationships are consumed'
               in line for line in inputs.report())
    edge = _edges(store_path)[0][0]
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_approval(edge, 'approved')
    inputs = load(store_path, object_store_approved_links_only=True)
    assert [rel.uuid for rel in inputs.snapshot.relationships] == [edge]
    assert inputs.counts['pcl attachments'] == 1


def test_empty_selectors_skip_detectors_without_writing(store_path):
    log = []
    inputs = load(store_path, log, object_store_pcl_uuids=[],
                  object_store_fiber_tags=['absent'])
    assert 'link refresh pcl_patch: skipped (no selected point collections)' in log
    assert 'link refresh fiber_patch: skipped (no selected fibers)' in log
    assert _edges(store_path) == []
    assert inputs.snapshot.objects_of('point_collection') == []


def test_subset_refresh_leaves_relationships_outside_the_selection(store_path):
    load(store_path)
    names = _uuids(store_path)
    outside = [edge for edge in _edges(store_path) if names['far'] in edge[1:3]]
    assert len(outside) == 1
    # A refresh over {near, walk} must neither touch nor remove the edge to
    # the unselected far patch.
    inputs = load(store_path, object_store_patch_uuids=[names['near']])
    assert [edge for edge in _edges(store_path) if names['far'] in edge[1:3]] == outside
    assert {rel.a for rel in inputs.snapshot.relationships} == {names['near']}


def test_input_toggles_select_nothing_and_pcl_roles_must_agree(store_path):
    config = fit_config(store_path, input_use_verified_patches=False,
                        input_use_fibers=False)
    request = osi.request_from_config(config)
    assert request.selector('patch') == osi.Selector(uuids=())
    assert request.selector('fiber') == osi.Selector(uuids=())
    assert request.selector('point_collection').is_default
    with pytest.raises(osi.ObjectStoreInputError, match='role toggles must all agree'):
        osi.request_from_config(fit_config(store_path, input_use_pcl_relative=False))


def test_normal_source_prefers_lasagna_then_ct_then_none():
    config = reference.fit_config()
    assert osi.fiber_normal_source(config).kind == 'none'
    config.update(object_store_normal_ct_path='/ct.zarr',
                  object_store_normal_ct_level=2,
                  object_store_normal_ct_downsampling=4.0)
    with pytest.raises(osi.ObjectStoreInputError, match='needs object_store_normal_ct_volume_uuid'):
        osi.fiber_normal_source(config, volume_uuid=VOLUME)
    config['object_store_normal_ct_volume_uuid'] = 'other'
    with pytest.raises(osi.ObjectStoreInputError, match='not the fit volume'):
        osi.fiber_normal_source(config, volume_uuid=VOLUME)
    config['object_store_normal_ct_volume_uuid'] = VOLUME
    ct = osi.fiber_normal_source(config, volume_uuid=VOLUME)
    assert (ct.kind, ct.ct_volume, dict(ct.parameters)) == (
        'ct', '/ct.zarr', {'ct_level': 2, 'ct_to_base_scale': 4.0})
    lasagna = osi.fiber_normal_source(
        config, {'normal_x': '/nx', 'normal_y': '/ny', 'group': '4', 'scale': 4},
        volume_uuid=VOLUME)
    assert lasagna.kind == 'lasagna' and lasagna.ct_volume == ''
    assert dict(lasagna.parameters) == {
        'ct_level': 4, 'ct_to_base_scale': 4.0, 'normal_x_volume': '/nx',
        'normal_y_volume': '/ny'}
    assert 'ignoring object_store_normal_ct_path' in lasagna.note
    assert lasagna.provenance()['kind'] == 'lasagna'


def test_fiber_refresh_records_the_chosen_normal_source(tmp_path):
    path = str(tmp_path / 'fibers.vcobj')
    with ObjectStore.create(path, author='ingest', hostname='host') as db:
        patch = put_patch(db, _grid(size=40), (0.25, 0.25))
        # Lying on the patch for 140 voxels, beyond the detector's minimum
        # contact run.
        fiber = put_fiber(db, [[30.0, 100.0, 50.0 + i] for i in range(141)],
                          [0, 140], name='lying')
    log = []
    inputs = load(path, log)
    assert 'fiber-patch normals: none (centerline-only linking)' in log
    assert any(line.startswith('WARNING: fiber-patch links are refreshed '
                               'centerline-only') for line in log)
    assert inputs.manifest['link_refresh']['fiber_patch_normals'] == {
        'kind': 'none', 'ct_volume': None}
    with ObjectStore.open(path, read_only=True) as db:
        [edge] = db.edges_by_kind('same-surface')
    assert {edge['a'], edge['b']} == {patch, fiber}
    assert edge['meta']['normal_method'] == 'centerline'
    assert edge['meta']['normal_source'] is None
    assert inputs.counts['fiber attachments'] == 1


# ---------------------------------------------------------------------------
# Two ranks
# ---------------------------------------------------------------------------

def _rank_worker(rank, world_size, rendezvous, config_paths, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import torch.distributed as dist
    from ddp_helpers import DistributedContext
    with open(config_paths[rank]) as stream:
        config = json.load(stream)
    dist.init_process_group('gloo', store=dist.FileStore(rendezvous, world_size),
                            rank=rank, world_size=world_size)
    try:
        inputs = osi.load_fit_inputs(
            config, dist=DistributedContext(rank, world_size, rank),
            log=lambda line: None)
        result = {'digest': inputs.digest,
                  'attachments': inputs.counts['pcl attachments']}
    except osi.ObjectStoreInputError as error:
        result = {'error': error.problems}
    finally:
        dist.destroy_process_group()
    with open(results[rank], 'w') as stream:
        json.dump(result, stream)


def _two_ranks(tmp_path, configs):
    import torch.multiprocessing as mp
    paths, results = [], []
    for rank, config in enumerate(configs):
        paths.append(str(tmp_path / f'config{rank}.json'))
        results.append(str(tmp_path / f'result{rank}.json'))
        with open(paths[-1], 'w') as stream:
            json.dump(config, stream)
    rendezvous = str(tmp_path / 'rendezvous')
    mp.spawn(_rank_worker, args=(2, rendezvous, paths, results), nprocs=2)
    outcome = []
    for path in results:
        with open(path) as stream:
            outcome.append(json.load(stream))
    return outcome


def test_a_rank_local_read_failure_reaches_every_rank(store_path, tmp_path):
    # Rank 1 alone cannot open its store, after rank 0 refreshed.
    configs = [fit_config(store_path),
               fit_config(str(tmp_path / 'absent.vcobj'))]
    first, second = _two_ranks(tmp_path, configs)
    assert 'error' in first and first == second
    assert all(line.startswith('rank 1: ') for line in first['error'])


def test_two_ranks_refresh_once_and_agree(store_path, tmp_path):
    config = fit_config(store_path)
    first, second = _two_ranks(tmp_path, [config, config])
    assert first == second and first['attachments'] == 3
    assert len(_edges(store_path)) == 3


def test_a_refresh_failure_reaches_every_rank(store_path, tmp_path):
    config = fit_config(store_path, pcl_link_window_min_points=5)
    first, second = _two_ranks(tmp_path, [config, config])
    assert 'error' in first and first == second
    assert any('pcl_patch: FAILED' in line for line in first['error'])


def test_ranks_with_different_inputs_refuse_to_train(store_path, tmp_path):
    configs = [fit_config(store_path),
               fit_config(store_path, pcl_patch_overlap_max_pairs=4)]
    first, second = _two_ranks(tmp_path, configs)
    assert first == second
    assert 'ranks read different object-store inputs' in first['error'][0]


# ---------------------------------------------------------------------------
# External frames and resume
# ---------------------------------------------------------------------------

def test_track_scale_converts_positions_roi_and_crossing_distances():
    from tracks import (PackedTrackCollection, crossing_cache_in_fit_frame,
                        track_store_z_range, tracks_in_fit_frame)
    assert track_store_z_range(10, 20, 1) == (10, 20)
    # Stored z 25..49 are fit z 12.5..24.5, inside [10, 25) only up to 49.
    assert track_store_z_range(10, 25, 2.0) == (20, 50)
    assert track_store_z_range(10, 25, 0.5) == (5, 13)
    coordinates = np.array([[20, 4, 6], [22, 4, 8], [24, 6, 8]], dtype=np.float32)
    packed = PackedTrackCollection(coordinates, [0, 3], [7], [0], [6.0], [1.2])
    fit = tracks_in_fit_frame(packed, 2.0)
    np.testing.assert_array_equal(fit[0], coordinates / 2)
    assert fit.selected_arclengths.tolist() == [3.0]
    assert fit.selected_source_ids.tolist() == [7]
    assert fit.selected_tortuosities.tolist() == [1.2]
    [listed] = tracks_in_fit_frame([coordinates.astype(np.int32)], 2.0)
    np.testing.assert_array_equal(listed, coordinates / 2)
    assert tracks_in_fit_frame(packed, 1) is packed
    cache = {'positions': [4.0, 8.0], 'clearances': [2.0, 1.0],
             'self_local': np.array([1, 2])}
    converted = crossing_cache_in_fit_frame(cache, 2.0)
    assert converted['positions'].tolist() == [2.0, 4.0]
    assert converted['clearances'].tolist() == [1.0, 0.5]
    assert converted['self_local'] is cache['self_local']


def test_converted_tracks_never_rediscover_crossings(tmp_path):
    """A cache that cannot serve the selected tracks is an error once the
    coordinates left their stored integer frame, not a rebuild from them."""
    from build_track_crossings import build_cache
    from test_track_crossing_cache import TrackCrossingCacheTests
    from tracks import (load_track_crossing_cache, load_tracks_from_dbm,
                        prepare_main_phase_tracks, tracks_in_fit_frame)
    path = TrackCrossingCacheTests.make_db(None, tmp_path)
    build_cache(path, show_progress=False)
    cache = load_track_crossing_cache(path)
    tracks, families, source_ids = load_tracks_from_dbm(
        path, 0, 20, return_families=True, return_source_ids=True,
        show_progress=False)
    policy = {'track_crossing_precompute_max': 1,
              'track_max_track_crossing_per_step': 1}

    def prepare(crossing_cache, ids, **options):
        return prepare_main_phase_tracks(
            tracks_in_fit_frame(tracks, 2.0), None, 0.0, 'cpu',
            sampling_config=policy, track_families=families,
            track_source_ids=ids, crossing_cache=crossing_cache, **options)

    # The cache serves its own tracks either way.
    assert 'crossing_index' in prepare(cache, source_ids, rediscover_crossings=False)
    # Tracks the cache does not cover (here: ids it never saw).
    foreign = np.asarray(source_ids, dtype=np.uint64) + np.uint64(1 << 40)
    with pytest.raises(ValueError, match='does not cover the selected tracks'):
        prepare(cache, foreign, rediscover_crossings=False)
    with pytest.raises(ValueError, match='no track crossing cache'):
        prepare(None, source_ids, rediscover_crossings=False)
    with pytest.raises(ValueError, match='track_exclusion_radius'):
        prepare_main_phase_tracks(
            tracks, None, 4.0, 'cpu', sampling_config=policy,
            track_families=families, track_source_ids=source_ids,
            crossing_cache=cache, rediscover_crossings=False)


def test_attached_tracks_need_a_declared_scale(tmp_path):
    from fit_spiral import FitContext
    context = SimpleNamespace(tracks_dbm_path=str(tmp_path / 'tracks.db'),
                              config={'track_scale_factor': None,
                                      'track_crossing_precompute_max': 0,
                                      'track_exclusion_radius': 0.0})
    with pytest.raises(RuntimeError, match='declare track_scale_factor'):
        FitContext._object_store_track_scale(context)
    context.config['track_scale_factor'] = 2.0
    assert FitContext._object_store_track_scale(context) == 2.0
    context.config['track_crossing_precompute_max'] = 8
    context.config['track_exclusion_radius'] = 16.0
    with pytest.raises((RuntimeError, OSError)):
        FitContext._object_store_track_scale(context)
    context.tracks_dbm_path = None
    assert FitContext._object_store_track_scale(context) is None


def _manifest(store_path, **frames):
    return osi.input_manifest(
        osi.read_snapshot(ObjectStore.open(store_path, read_only=True),
                          osi.InputRequest()),
        fit_config(store_path), osi.NormalSource('none'),
        {'tracks': {'scale_factor': 2.0}, **frames})


def test_resume_refuses_changed_inputs_and_frames(store_path):
    inputs = load(store_path)
    recorded = {'object_store': json.loads(json.dumps(inputs.manifest))}
    assert osi.resume_problems(recorded, inputs) == []
    changed = json.loads(json.dumps(recorded))
    changed['object_store']['external_frames'] = {'tracks': {'scale_factor': 2.0}}
    changed['object_store']['materialization']['z_end'] = 32
    [problem] = osi.resume_problems(changed, inputs)
    assert 'external_frames.tracks' in problem and 'materialization.z_end' in problem
    assert 'object_store_allow_changed_inputs' in problem
    warnings = []
    assert osi.resume_problems(changed, inputs, allow_changed=True,
                               log=warnings.append) == []
    assert 'resuming anyway' in warnings[0]
    [problem] = osi.resume_problems({}, inputs)
    assert 'records no object-store inputs' in problem
    # Refreshed relationships are inputs too.
    edge = _edges(store_path)[0][0]
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_approval(edge, 'approved')
    [problem] = osi.resume_problems(recorded, load(store_path))
    assert 'relationships' in problem
    # How links were refreshed is provenance, not an input.
    quiet = load(store_path, object_store_refresh_links=False)
    assert quiet.digest == load(store_path).digest


def test_checkpoint_preflight_checks_the_object_store_inputs(store_path):
    import fit_spiral
    from test_checkpoint_load import _checkpoint, _live_context
    inputs = load(store_path)
    context = _live_context()
    context.object_store_inputs = inputs
    payload = fit_spiral.FitContext._checkpoint_input_manifest(
        SimpleNamespace(interactive_driver=None, object_store_inputs=inputs))
    assert payload['object_store_digest'] == inputs.digest
    verdict = fit_spiral.FitContext.inspect_checkpoint(
        context, _checkpoint(input_manifest=payload))
    assert verdict.accepted, verdict.message()
    stale = json.loads(json.dumps(payload))
    stale['object_store']['external_frames'] = {'tracks': {'scale_factor': 4.0}}
    verdict = fit_spiral.FitContext.inspect_checkpoint(
        context, _checkpoint(input_manifest=stale))
    assert not verdict.accepted
    assert 'external_frames.tracks' in verdict.message()


# ---------------------------------------------------------------------------
# FitContext
# ---------------------------------------------------------------------------

def test_fit_context_loads_frozen_inputs_from_its_configuration(store_path, tmp_path):
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)
    context = FitContext(
        FitConfig(fit_config(store_path)), scroll=reference.scroll_spec(),
        paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), out_base_dir=str(tmp_path / 'out'))
    context.load_host_inputs()
    inputs = context.object_store_inputs
    assert inputs is not None and inputs.digest
    assert len(context.verified_patches) == 2
    assert inputs.manifest['external_frames'] == {
        'umbilicus': {'coordinate_scale': 1.0}}
    assert context._checkpoint_input_manifest()['object_store_digest'] == inputs.digest
    context.resolve_output_path()
    context._write_object_store_manifest()
    with open(os.path.join(context.out_path, 'object_store_inputs.json')) as stream:
        assert json.load(stream)['digest'] == inputs.digest
    # Settings that would relink or re-derive the frozen inputs are refused.
    with pytest.raises(ValueError, match='object-store inputs are frozen'):
        context.apply_config({'pcl_link_distance_tolerance': 3.0},
                             current_iteration=0)
    context._maybe_relink_fibers_for_direction(10 ** 6)


# ---------------------------------------------------------------------------
# Real data (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get('OBJECT_STORE_SMOKE_STORE'),
                    reason='set OBJECT_STORE_SMOKE_STORE to a copy of an ingested dataset store')
def test_real_data_startup_smoke():
    """Refresh and materialize a real ingested store over a z window.

    OBJECT_STORE_SMOKE_STORE names a store the test may write links to (a
    scratch copy); OBJECT_STORE_SMOKE_Z (default 9000:9200) is the window.
    """
    path = os.environ['OBJECT_STORE_SMOKE_STORE']
    z_begin, z_end = map(int, os.environ.get('OBJECT_STORE_SMOKE_Z', '9000:9200').split(':'))
    log = []
    inputs = load(path, log, z_begin=z_begin, z_end=z_end)
    print('\n'.join(log + inputs.report()))
    assert inputs.patches
    assert inputs.counts['pcl attachments'] + inputs.counts['fiber attachments'] \
        + inputs.counts['overlap pairs'] > 0
    assert load(path, z_begin=z_begin, z_end=z_end).digest == inputs.digest
