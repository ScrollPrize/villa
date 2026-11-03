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


def _edges(path, kind='point-surface'):
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
    # A rerun finds the links current (the store's link marker) and runs no
    # detector: same relationships, same frozen inputs.
    rerun = []
    again = load(store_path, rerun)
    assert _edges(store_path) == edges
    assert again.digest == inputs.digest
    assert any(line.startswith('link refresh: skip: unchanged since position') for line in rerun)
    assert not any('running' in line for line in rerun)
    # A forced full refresh reconciles idempotently.
    assert any('3 unchanged' in line
               for line in _refresh_lines(store_path, object_store_full_link_refresh=True))


def _refresh_lines(path, **overrides):
    log = []
    load(path, log, **overrides)
    return log


def test_refresh_needs_an_author(store_path):
    with pytest.raises(osi.LinkRefreshError, match='object_store_author is required'):
        load(store_path, object_store_author=None)


def test_refresh_failures_are_collected_and_stop_the_fit(store_path, tmp_path):
    with ObjectStore.open(store_path, author='ingest') as db:
        put_fiber(db, [[12.0, 100.5, 40.0 + i] for i in range(30)], [0, 29], name='f')
    config = fit_config(store_path, pcl_link_window_points=1,
                        pcl_link_window_min_points=5)
    lasagna = {'pool': str(tmp_path / 'absent_pack'), 'scale': 4}
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
    assert request.selector('surface') == osi.Selector(uuids=())
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
        'ct', '/ct.zarr/2', {'ct_level': 0, 'ct_to_base_scale': 4.0})
    # The store reads the normals pack the fitter loads, never the arrays.
    lasagna = osi.fiber_normal_source(
        config, {'pool': '/cache/lasagna/run-1/normals_g4', 'scale': 4},
        volume_uuid=VOLUME)
    assert lasagna.kind == 'lasagna' and lasagna.ct_volume == ''
    assert dict(lasagna.parameters) == {
        'ct_level': 0, 'ct_to_base_scale': 4.0,
        'normal_pool': '/cache/lasagna/run-1/normals_g4'}
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


def test_packed_normals_link_fibers_as_the_arrays_do(tmp_path, monkeypatch):
    """The refresh reads the fit's normals pack; it links exactly as reading
    the nx/ny arrays it was packed from."""
    import zarr
    from pack_resident_pools import pack_arrays

    arrays = []
    for name, value in (('nx', 128), ('ny', 255)):  # the patch plane's normal, +Y
        array = zarr.open(str(tmp_path / f'{name}.zarr'), mode='w', shape=(192, 128, 224),
                          chunks=(16, 16, 16), dtype='|u1', compressor=None, fill_value=0,
                          zarr_format=2, dimension_separator='/')
        array[:] = value
        arrays.append(str(tmp_path / f'{name}.zarr'))
    pack = pack_arrays(arrays, str(tmp_path / 'normals_g0'), label='normals',
                       lasagna_identity='run-1')

    stores = []
    for name in ('arrays', 'pack'):
        path = str(tmp_path / f'{name}.vcobj')
        with ObjectStore.create(path, author='ingest', hostname='host') as db:
            put_patch(db, _grid(size=40), (0.25, 0.25))
            put_fiber(db, [[30.0, 100.0, 50.0 + i] for i in range(141)], [0, 140], name='lying')
        stores.append(path)
    original = osi.fiber_normal_source
    monkeypatch.setattr(osi, 'fiber_normal_source', lambda config, lasagna, volume: osi.NormalSource(
        'lasagna', '', (('ct_level', 0), ('ct_to_base_scale', 1.0),
                        ('normal_x_volume', arrays[0]), ('normal_y_volume', arrays[1]))))
    load(stores[0])
    monkeypatch.setattr(osi, 'fiber_normal_source', original)
    osi.load_fit_inputs(fit_config(stores[1]), lasagna={'pool': pack, 'scale': 1.0},
                        log=lambda line: None)

    def links(path):
        # Each store's objects have their own UUIDs; the evidence is compared.
        with ObjectStore.open(path, read_only=True) as db:
            return [{key: (sorted(map(json.dumps, value['objects'].values()))
                           if key == 'coordinate_frame' else value)
                     for key, value in edge['meta'].items() if key != 'normal_source'}
                    for edge in db.edges_by_kind('same-surface')]
    from_arrays, from_pack = links(stores[0]), links(stores[1])
    assert from_arrays and from_pack == from_arrays
    assert from_pack[0]['normal_method'] == 'lasagna-nxny-u8-nearest-v1'
    with ObjectStore.open(stores[1], read_only=True) as db:
        [edge] = db.edges_by_kind('same-surface')
    assert edge['meta']['normal_source'] == {'normal_pool': pack}


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


def _replicas(store_path, tmp_path):
    """Two hub clones of the store: the deployed case of one logical store
    seen through different local files."""
    import vc_object_store as vos
    vos.hub_init(store_path, 'ingest')
    replicas = [str(tmp_path / f'{name}.vcobj') for name in ('a', 'b')]
    for path, author in zip(replicas, ('alice', 'bob')):
        vos.hub_clone(store_path, path, author)
    return replicas


def test_replicas_of_one_store_give_one_digest(store_path, tmp_path):
    first, second = _replicas(store_path, tmp_path)
    # Loading on a hosted replica links and publishes at a hub position.
    linked = load(first, object_store_author='alice')
    assert linked.manifest['position']['sequence'] > 0
    with ObjectStore.open(second, author='bob') as db:
        assert db.hub_sync(pull=True, push=False)['pulled']
    # The pull installs the links, and the objects they revise, under the
    # second replica's own revision UUIDs; the current-state digests agree.
    def current(path):
        with ObjectStore.open(path, read_only=True) as db:
            return osi.read_snapshot(db, osi.request_from_config(fit_config(path)))
    a, b = current(first), current(second)
    local = lambda snapshot: {entry['uuid']: entry['object_revision']
                              for entry in snapshot.manifest()['objects']}
    versions = lambda snapshot: {entry['uuid']: entry['revision']
                                 for entry in snapshot.manifest()['relationships']}
    assert local(a).keys() == local(b).keys()
    assert local(a) != local(b) and versions(a) != versions(b)
    assert len(versions(a)) == 3 and a.digest() == b.digest()
    # Reading the pinned position gives one manifest on any replica.
    pinned = load(second, object_store_author='bob', object_store_refresh_links=False)
    assert pinned.manifest['position'] == linked.manifest['position']
    assert pinned.counts == linked.counts and pinned.digest == linked.digest
    assert osi.manifest_differences(linked.manifest, pinned.manifest) == []
    assert osi.resume_problems({'object_store': linked.manifest}, pinned) == []


def test_digest_covers_the_metadata_the_adapter_reads(store_path):
    inputs = load(store_path)
    walk = _uuids(store_path)['walk']
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_meta(walk, 'review_note', 'checked')
    assert load(store_path, object_store_refresh_links=False).digest == inputs.digest
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_meta(walk, 'sampling_group', 'reviewed')
    changed = load(store_path, object_store_refresh_links=False)
    assert changed.digest != inputs.digest
    # The manifest carries the values read, which clients compare directly.
    [entry] = [e for e in changed.manifest['objects'] if e['uuid'] == walk]
    assert entry['metadata']['sampling_group'] == 'reviewed'
    assert osi.content_sha256(entry['metadata']) == entry['metadata_sha256']
    [problem] = osi.resume_problems({'object_store': inputs.manifest}, changed)
    assert 'objects' in problem


def test_a_checkpoint_before_content_identity_compares_geometry(store_path):
    inputs = load(store_path)
    legacy = json.loads(json.dumps(inputs.manifest))
    # The manifest such a checkpoint recorded: no metadata hashes, no
    # relationship content, and revision UUIDs of the replica it ran on.
    for entry in legacy['objects']:
        del entry['metadata_sha256']
        entry['object_revision'] = 'another-replica'
    legacy['relationships'] = [
        {key: entry[key] for key in ('uuid', 'kind', 'family', 'approval_state',
                                     'generation')} | {'revision': 'another'}
        for entry in legacy['relationships']]
    warnings = []
    assert osi.resume_problems({'object_store': legacy}, inputs,
                               log=warnings.append) == []
    assert 'compared on geometry revisions' in warnings[0]
    legacy['objects'][0]['geometry_revision'] = 'older-geometry'
    [problem] = osi.resume_problems({'object_store': legacy}, inputs)
    assert 'objects' in problem


def test_checkpoint_preflight_checks_the_object_store_inputs(store_path):
    import fit_spiral
    from fit_session import SpiralInputPaths
    from test_checkpoint_load import _checkpoint, _live_context
    inputs = load(store_path)
    context = _live_context()
    context.object_store_inputs = inputs
    # A command-line fit records its resolved inputs (paths, fit volume,
    # provenance) beside the object-store inputs.
    payload = fit_spiral.FitContext._checkpoint_input_manifest(
        SimpleNamespace(interactive_driver=None, object_store_inputs=inputs,
                        paths=SpiralInputPaths(volume_uuid='volume-1',
                                               provenance={'input_manifest': '/m.json'})))
    assert payload['object_store_digest'] == inputs.digest
    assert payload['volume_uuid'] == 'volume-1'
    assert payload['provenance'] == {'input_manifest': '/m.json'}
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



def _attach_results(path, tag=None):
    """A correctly framed fit-results collection attached to the near patch."""
    with ObjectStore.open(path, author='fit', hostname='host') as db:
        near = next(r['uuid'] for r in db.objects_by_kind('surface') if r['name'] == 'near')
        results = put_pcl(db, _on(100.0, (12.0, 48.0)), name='results',
                          **({'tags': [tag]} if tag else {}))
        db.set_meta(results, 'role', 'fit_results')
        whole = {'whole_geometry': True}
        db.add_edge({'kind': 'attachment', 'a': results, 'b': near,
                     'a_anchor': whole, 'b_anchor': whole, 'meta': {'role': 'fit_results'}})
    return results


@pytest.mark.parametrize('refresh', [True, False])
def test_attached_collections_are_not_fit_inputs(store_path, refresh):
    if not refresh:
        load(store_path)  # the no-refresh read consumes links made earlier
    results = _attach_results(store_path, tag='walks')
    inputs = load(store_path, object_store_refresh_links=refresh)
    assert inputs.counts['pcl attachments'] == 3
    assert results not in json.dumps(inputs.manifest)
    # A tag matching only the attached collection selects no PCL.
    inputs = load(store_path, object_store_refresh_links=refresh,
                  object_store_pcl_tags=['walks'])
    assert inputs.counts['pcl attachments'] == 0
    assert results not in json.dumps(inputs.manifest)


def test_explicitly_selected_attached_collection_is_refused(store_path):
    results = _attach_results(store_path)
    with pytest.raises(osi.ObjectStoreInputError, match='attached data'):
        load(store_path, object_store_pcl_uuids=[results])
