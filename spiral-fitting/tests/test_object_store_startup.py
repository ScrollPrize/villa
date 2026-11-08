"""Startup lifecycle of object-store fit inputs: link refresh, frozen reads,
rank agreement, external frames and resume validation."""
import hashlib
import dataclasses
import json
import os
from types import SimpleNamespace

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    VOLUME, edges, fit_config, names, on_plane, plane, put_fiber, put_patch, put_pcl)

pytestmark = pytest.mark.usefixtures('cpu')


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def load(path, log=None, **overrides):
    lines = [] if log is None else log
    return osi.load_fit_inputs(fit_config(path, **overrides), log=lines.append)


def test_default_refresh_writes_links_and_freezes_the_inputs(store_path):
    log = []
    inputs = load(store_path, log)
    linked = edges(store_path)
    assert len(linked) == 3
    assert inputs.counts['pcl attachments'] == 3
    assert any(line.startswith('link refresh pcl_patch: 3 created') for line in log)
    assert inputs.manifest['link_refresh']['enabled']
    assert inputs.manifest['materialization']['pcl_patch_overlap_max_pairs'] == 16
    # A rerun finds the links current (the store's link marker) and runs no
    # detector: same relationships, same frozen inputs.
    rerun = []
    again = load(store_path, rerun)
    assert edges(store_path) == linked
    assert again.digest == inputs.digest
    assert any(line.startswith('link refresh: skip: unchanged since commit') for line in rerun)
    assert not any('running' in line for line in rerun)
    # A forced full refresh reconciles idempotently.
    forced = []
    load(store_path, forced, object_store_full_link_refresh=True)
    assert any('3 unchanged' in line for line in forced)
    with pytest.raises(osi.LinkRefreshError, match='object_store_author is required'):
        load(store_path, object_store_author=None)


def test_refresh_failures_are_collected_and_stop_the_fit(store_path, tmp_path):
    with ObjectStore.open(store_path, author='ingest') as db:
        put_fiber(db, [[12.0, 100.5, 40.0 + i] for i in range(30)], [0, 29], name='f')
    config = fit_config(store_path, pcl_link_window_points=1,
                        pcl_link_window_min_points=5)
    lasagna = {'pool': str(tmp_path / 'absent_pack'), 'scale': 4}
    before = _sha256(store_path)
    with pytest.raises(osi.LinkRefreshError) as error:
        osi.load_fit_inputs(config, lasagna=lasagna, log=lambda line: None)
    text = '\n'.join(error.value.problems)
    # The plan is refused whole: no detector ran, nothing was written.
    assert 'nothing was written' in text and 'point-collection-surface' in text
    assert _sha256(store_path) == before
    # A detector that cannot run (the absent normals pack) fails the whole
    # request the same way, the other detectors' links included.
    config = fit_config(store_path)
    with pytest.raises(osi.LinkRefreshError) as error:
        osi.load_fit_inputs(config, lasagna=lasagna, log=lambda line: None)
    text = '\n'.join(error.value.problems)
    assert 'nothing was written' in text and 'fiber-surface' in text
    assert edges(store_path) == []


def test_no_refresh_mode_never_writes(store_path):
    before = _sha256(store_path)
    log = []
    inputs = load(store_path, log, object_store_refresh_links=False,
                  object_store_author=None)
    assert _sha256(store_path) == before
    assert edges(store_path) == []
    assert inputs.counts['pcl attachments'] == 0
    assert any(line.startswith('link refresh: off; pinning commit') for line in log)
    assert not inputs.manifest['link_refresh']['enabled']
    assert 'WARNING: no pcl_patch relationships are consumed' in '\n'.join(inputs.report())


def test_empty_selectors_skip_detectors_without_writing(store_path):
    log = []
    inputs = load(store_path, log, object_store_pcl_uuids=[],
                  object_store_fiber_tags=['absent'])
    assert 'link refresh pcl_patch: skipped (no selected point collections)' in log
    assert 'link refresh fiber_patch: skipped (no selected fibers)' in log
    assert edges(store_path) == []
    assert inputs.snapshot.objects_of('point_collection') == []
    # Input toggles select nothing either; the PCL role toggles must agree.
    request = osi.request_from_config(fit_config(
        store_path, input_use_verified_patches=False, input_use_fibers=False))
    assert request.selector('surface') == osi.Selector(uuids=())
    assert request.selector('fiber') == osi.Selector(uuids=())
    assert request.selector('point_collection').is_default
    with pytest.raises(osi.ObjectStoreInputError, match='role toggles must all agree'):
        osi.request_from_config(fit_config(store_path, input_use_pcl_relative=False))


def test_subset_refresh_leaves_relationships_outside_the_selection(store_path):
    load(store_path)
    uuids = names(store_path)
    outside = [edge for edge in edges(store_path) if uuids['far'] in edge[1:3]]
    assert len(outside) == 1
    # A refresh over {near, walk} must neither touch nor remove the edge to
    # the unselected far patch.
    inputs = load(store_path, object_store_patch_uuids=[uuids['near']])
    assert [edge for edge in edges(store_path) if uuids['far'] in edge[1:3]] == outside
    assert {rel.a for rel in inputs.snapshot.relationships} == {uuids['near']}


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
            put_patch(db, plane(size=40), (0.25, 0.25))
            # Lying on the patch for 140 voxels, beyond the detector's
            # minimum contact run.
            put_fiber(db, [[30.0, 100.0, 50.0 + i] for i in range(141)], [0, 140], name='lying')
        stores.append(path)
    original = osi.fiber_normal_source
    monkeypatch.setattr(osi, 'fiber_normal_source', lambda config, lasagna, volume: osi.NormalSource(
        'lasagna', '', (('ct_level', 0), ('ct_to_base_scale', 1.0),
                        ('normal_x_volume', arrays[0]), ('normal_y_volume', arrays[1]))))
    load(stores[0])
    monkeypatch.setattr(osi, 'fiber_normal_source', original)
    inputs = osi.load_fit_inputs(fit_config(stores[1]), lasagna={'pool': pack, 'scale': 1.0},
                                 log=lambda line: None)
    assert inputs.counts['fiber attachments'] == 1

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
# Two ranks (the end-to-end two-rank sessions are in test_runtime_object_store)
# ---------------------------------------------------------------------------

def _rank_worker(rank, world_size, rendezvous, configs, results):
    os.environ['FIT_SPIRAL_DEVICE'] = 'cpu'
    import torch.distributed as dist
    from ddp_helpers import DistributedContext
    dist.init_process_group('gloo', store=dist.FileStore(rendezvous, world_size),
                            rank=rank, world_size=world_size)
    try:
        inputs = osi.load_fit_inputs(
            configs[rank], dist=DistributedContext(rank, world_size, rank),
            log=lambda line: None)
        result = {'digest': inputs.digest,
                  'attachments': inputs.counts['pcl attachments']}
    except osi.ObjectStoreInputError as error:
        result = {'error': error.problems}
    finally:
        dist.destroy_process_group()
    with open(results[rank], 'w') as stream:
        json.dump(result, stream)


def test_a_rank_local_read_failure_reaches_every_rank(store_path, tmp_path):
    from cpu_service import two_ranks
    # Rank 1 alone cannot open its store, after rank 0 refreshed.
    first, second = two_ranks(_rank_worker, tmp_path, [
        fit_config(store_path), fit_config(str(tmp_path / 'absent.vcobj'))])
    assert 'error' in first and first == second
    assert all(line.startswith('rank 1: ') for line in first['error'])


def test_ranks_with_different_inputs_refuse_to_train(store_path, tmp_path):
    from cpu_service import two_ranks
    first, second = two_ranks(_rank_worker, tmp_path, [
        fit_config(store_path), fit_config(store_path, pcl_patch_overlap_max_pairs=4)])
    assert first == second
    assert 'ranks read different object-store inputs' in first['error'][0]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

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
    edge = edges(store_path)[0][0]
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_approval(edge, 'approved')
    [problem] = osi.resume_problems(recorded, load(store_path))
    assert 'relationships' in problem
    # How links were refreshed is provenance, not an input.
    quiet = load(store_path, object_store_refresh_links=False)
    assert quiet.digest == load(store_path).digest


def test_a_copy_of_the_store_gives_one_digest(store_path, tmp_path):
    """A copied store file: the same inputs under the same identity."""
    import shutil
    linked = load(store_path, object_store_author='alice')
    assert linked.manifest['position']['commit_sequence'] > 0
    assert linked.manifest['link_refresh']['parameters'] == {
        key: fit_config(store_path)[key] for key in osi.LINK_PARAMETER_KEYS}
    copy = str(tmp_path / 'copy.vcobj')
    shutil.copyfile(store_path, copy)

    def current(path):
        with ObjectStore.open(path, read_only=True) as db:
            return osi.read_snapshot(db, osi.request_from_config(fit_config(path)))
    a, b = current(store_path), current(copy)
    versions = lambda snapshot: {entry['uuid']: entry['version']
                                 for entry in snapshot.manifest()['objects']}
    assert versions(a) == versions(b) and a.digest() == b.digest()
    assert len(a.manifest()['relationships']) == 3
    # Reading the pinned position gives one manifest from either file.
    pinned = load(copy, object_store_author='bob', object_store_refresh_links=False)
    assert pinned.manifest['position'] == linked.manifest['position']
    assert pinned.counts == linked.counts and pinned.digest == linked.digest
    assert osi.manifest_differences(linked.manifest, pinned.manifest) == []
    assert osi.resume_problems({'object_store': linked.manifest}, pinned) == []
    # A position ahead of the store's head is refused.
    ahead = dataclasses.replace(osi.InputPosition.from_provenance(linked.manifest['position']),
                                commit_sequence=linked.manifest['position']['commit_sequence'] + 100)
    with pytest.raises(osi.ObjectStoreInputError, match='after this store'):
        osi.load_fit_inputs(fit_config(copy), log=lambda line: None, minimum_position=ahead)


def test_digest_covers_the_metadata_the_adapter_reads(store_path):
    inputs = load(store_path)
    walk = names(store_path)['walk']
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
        del entry['version']
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
    assert (payload['volume_uuid'], payload['provenance']) == (
        'volume-1', {'input_manifest': '/m.json'})
    verdict = fit_spiral.FitContext.inspect_checkpoint(
        context, _checkpoint(input_manifest=payload))
    assert verdict.accepted, verdict.message()
    stale = json.loads(json.dumps(payload))
    stale['object_store']['external_frames'] = {'tracks': {'scale_factor': 4.0}}
    verdict = fit_spiral.FitContext.inspect_checkpoint(
        context, _checkpoint(input_manifest=stale))
    assert not verdict.accepted
    assert 'external_frames.tracks' in verdict.message()


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
    context.resolve_output_path()
    context._write_object_store_manifest()
    with open(os.path.join(context.out_path, 'object_store_inputs.json')) as stream:
        assert json.load(stream)['digest'] == inputs.digest
    # Settings that would relink or re-derive the frozen inputs are refused.
    with pytest.raises(ValueError, match='object-store inputs are frozen'):
        context.apply_config({'pcl_link_distance_tolerance': 3.0},
                             current_iteration=0)


def _context(store_path, tmp_path, **options):
    from config import FitConfig
    from fit_session import SpiralInputPaths
    from fit_spiral import FitContext
    umbilicus = str(tmp_path / 'umbilicus.json')
    reference.write_umbilicus(umbilicus)
    return FitContext(
        FitConfig(fit_config(store_path, **options.pop('config', {}))),
        scroll=reference.scroll_spec(), paths=SpiralInputPaths(umbilicus=umbilicus),
        cache_dir=str(tmp_path / 'cache'), out_base_dir=str(tmp_path / 'out'), **options)


def test_a_resumed_fit_reads_the_checkpoints_pinned_inputs(store_path, tmp_path):
    """A command-line resume reads the inputs at the versions the checkpoint
    recorded (case C6), whatever the store holds now."""
    import torch
    from object_store_fixture import names
    trained = _context(store_path, tmp_path)
    trained.load_host_inputs()
    recorded = trained.object_store_inputs
    checkpoint = str(tmp_path / 'resume.ckpt')
    torch.save({'cfg': dict(recorded.manifest['materialization']),
                'input_manifest': {'object_store': json.loads(json.dumps(recorded.manifest))}},
               checkpoint)
    # A selected object changes after the checkpoint: its current metadata
    # is not what the fit was trained on.
    with ObjectStore.open(store_path, author='editor') as db:
        db.set_meta(names(store_path)['walk'], 'sampling_group', 'reviewed')
    log = []
    context = _context(store_path, tmp_path, resume_path=checkpoint, input_log=log.append)
    context.load_host_inputs()
    inputs = context.object_store_inputs
    assert inputs.digest == recorded.digest
    assert inputs.manifest['position'] == recorded.manifest['position']
    assert {entry['uuid']: entry['version'] for entry in inputs.manifest['objects']} == {
        entry['uuid']: entry['version'] for entry in recorded.manifest['objects']}
    assert osi.resume_problems({'object_store': recorded.manifest}, inputs) == []
    assert any('resuming at the checkpoint' in line for line in log)
    # object_store_allow_changed_inputs resumes against the current inputs.
    current = _context(store_path, tmp_path, resume_path=checkpoint,
                       config={'object_store_allow_changed_inputs': True})
    current.load_host_inputs()
    assert current.object_store_inputs.digest != recorded.digest
    assert current.object_store_inputs.manifest['position']['commit_sequence'] \
        > recorded.manifest['position']['commit_sequence']


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


@pytest.mark.parametrize('refresh', [True, False])
def test_attached_collections_are_not_fit_inputs(store_path, refresh):
    if not refresh:
        load(store_path)  # the no-refresh read consumes links made earlier
    # A correctly framed fit-results collection attached to the near patch.
    with ObjectStore.open(store_path, author='fit', hostname='host') as db:
        results = put_pcl(db, on_plane(100.0, (12.0, 48.0)), name='results', tags=['walks'])
        db.set_meta(results, 'role', 'fit_results')
        whole = {'whole_geometry': True}
        db.add_edge({'kind': 'attachment', 'a': results, 'b': names(store_path)['near'],
                     'a_anchor': whole, 'b_anchor': whole, 'meta': {'role': 'fit_results'}})
    inputs = load(store_path, object_store_refresh_links=refresh)
    assert inputs.counts['pcl attachments'] == 3
    assert results not in json.dumps(inputs.manifest)
    # A tag matching only the attached collection selects no PCL.
    inputs = load(store_path, object_store_refresh_links=refresh,
                  object_store_pcl_tags=['walks'])
    assert inputs.counts['pcl attachments'] == 0
    assert results not in json.dumps(inputs.manifest)
    # Naming it explicitly is refused.
    with pytest.raises(osi.ObjectStoreInputError, match='attached data'):
        load(store_path, object_store_pcl_uuids=[results])
