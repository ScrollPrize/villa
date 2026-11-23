"""Startup lifecycle of object-store fit inputs: pinned positions, frozen
reads, rank agreement, external frames and resume validation."""
import hashlib
import dataclasses
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip('vc_object_store')
from vc_object_store import ObjectStore  # noqa: E402

import object_store_inputs as osi  # noqa: E402
import object_store_reference as reference  # noqa: E402
from object_store_fixture import (  # noqa: E402
    VOLUME, fit_config, names)

pytestmark = pytest.mark.usefixtures('cpu')


def _sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def load(path, log=None, **overrides):
    lines = [] if log is None else log
    return osi.load_fit_inputs(fit_config(path, **overrides), log=lines.append)


def test_umbilicus_owner_survives_request_manifest_and_commit(tmp_path):
    import threading
    import uuid
    from fit_session import SpiralProject
    from spiral_service import ServiceState
    from service_project import project, UMBILICUS

    manifests = []
    for owner in (str(uuid.uuid4()), str(uuid.uuid4())):
        selected = dict(reference=dict(store=owner, generation=UMBILICUS['generation'],
                                       object=UMBILICUS['uuid'], version=UMBILICUS['version']),
                        points_zyx=UMBILICUS['points_zyx'])
        parsed = SpiralProject.from_mapping(project(umbilicus=selected))
        assert parsed.to_mapping()['volumes'][VOLUME]['umbilicus'] == selected
        manifest = parsed.record(VOLUME)
        assert manifest['umbilicus']['reference']['store'] == owner
        manifests.append(manifest)
        input_path = tmp_path / 'inputs.json'
        input_path.write_text(json.dumps({'manifest': {'volume_uuid': VOLUME}}))
        state = SimpleNamespace(lock=threading.RLock(), session_id='s',
                                session=SimpleNamespace(status=lambda: dict(
                                    input_manifest_path=str(input_path), input_digest='digest')),
                                session_request={'project': dict(umbilicus=selected)},
                                object_store=SimpleNamespace(author='test'))
        context = ServiceState._commit_context(state, 's', 1, 'hash')
        assert context.umbilicus == {'reference': selected['reference']}
    assert manifests[0] != manifests[1]


def test_umbilicus_requires_reference_and_preserves_materialized_source(tmp_path):
    from fit_session import SpiralProject, ScrollSpecError, write_project_umbilicus
    from service_project import project, UMBILICUS

    selected = SpiralProject.from_mapping(project()).volumes[VOLUME].umbilicus
    assert selected.reference == dict(store=UMBILICUS['store'], object=UMBILICUS['uuid'],
                                      generation=UMBILICUS['generation'], version=UMBILICUS['version'])
    first = write_project_umbilicus(tmp_path, VOLUME, selected)
    assert json.loads(Path(first).read_text())['source']['reference'] == selected.reference
    other = dataclasses.replace(selected, reference=dict(selected.reference, store=selected.reference['object']))
    second = write_project_umbilicus(tmp_path, VOLUME, other)
    assert first != second
    assert json.loads(Path(first).read_text())['source']['reference'] == selected.reference
    legacy = dict(UMBILICUS)
    legacy.pop('store')
    with pytest.raises(ScrollSpecError, match='regenerate'):
        SpiralProject.from_mapping(project(umbilicus=legacy))
    with pytest.raises(ScrollSpecError, match='regenerate'):
        SpiralProject.from_mapping(project(umbilicus=UMBILICUS))


def test_a_fit_never_writes_the_store(store_path):
    before = _sha256(store_path)
    log = []
    inputs = load(store_path, log)
    assert _sha256(store_path) == before
    assert any(line.startswith('pinning commit') for line in log)
    assert len(inputs.patches) == 3


def test_empty_selectors_select_nothing(store_path):
    log = []
    inputs = load(store_path, log, object_store_patch_uuids=[],
                  object_store_track_tags=['absent'], fit_volume_uuid=VOLUME)
    assert inputs.snapshot.objects_of('surface') == []
    # Input toggles select nothing either.
    request = osi.request_from_config(fit_config(
        store_path, input_use_verified_patches=False, input_use_tracks=False))
    assert request.selector('surface') == osi.Selector(uuids=())
    assert request.selector('track') == osi.Selector(uuids=())


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
        result = {'digest': inputs.digest}
    except osi.ObjectStoreInputError as error:
        result = {'error': error.problems}
    finally:
        dist.destroy_process_group()
    with open(results[rank], 'w') as stream:
        json.dump(result, stream)


def test_a_rank_local_read_failure_reaches_every_rank(store_path, tmp_path):
    from cpu_service import two_ranks
    # Rank 1 alone cannot open its store, after rank 0 pinned it.
    first, second = two_ranks(_rank_worker, tmp_path, [
        fit_config(store_path), fit_config(str(tmp_path / 'absent.vcobj'))])
    assert 'error' in first and first == second
    assert all(line.startswith('rank 1: ') for line in first['error'])


def test_ranks_with_different_inputs_refuse_to_train(store_path, tmp_path):
    from cpu_service import two_ranks
    first, second = two_ranks(_rank_worker, tmp_path, [
        fit_config(store_path), fit_config(store_path, patch_loss_z_margin=4)])
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


def test_a_copy_of_the_store_gives_one_digest(store_path, tmp_path):
    """A copied store file: the same inputs under the same identity."""
    import shutil
    linked = load(store_path)
    assert linked.manifest['position']['commit_sequence'] > 0
    copy = str(tmp_path / 'copy.vcobj')
    shutil.copyfile(store_path, copy)

    def current(path):
        with ObjectStore.open(path, read_only=True) as db:
            return osi.read_snapshot(db, osi.request_from_config(fit_config(path)))
    a, b = current(store_path), current(copy)
    versions = lambda snapshot: {entry['uuid']: entry['version']
                                 for entry in snapshot.manifest()['objects']}
    assert versions(a) == versions(b) and a.digest() == b.digest()
    # Reading the pinned position gives one manifest from either file.
    pinned = load(copy)
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
    assert load(store_path).digest == inputs.digest
    with ObjectStore.open(store_path, author='reviewer') as db:
        db.set_meta(walk, 'source', {'spiral_patch_erode_cells': 2})
    changed = load(store_path)
    assert changed.digest != inputs.digest
    # The manifest carries the values read, which clients compare directly.
    [entry] = [e for e in changed.manifest['objects'] if e['uuid'] == walk]
    assert entry['metadata']['spiral_patch_erode_cells'] == 2
    assert osi.content_sha256(entry['metadata']) == entry['metadata_sha256']
    [problem] = osi.resume_problems({'object_store': inputs.manifest}, changed)
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
    assert len(context.verified_patches) == 3
    assert inputs.manifest['external_frames'] == {
        'umbilicus': {'coordinate_scale': 1.0}}
    context.resolve_output_path()
    context._write_object_store_manifest()
    with open(os.path.join(context.out_path, 'object_store_inputs.json')) as stream:
        assert json.load(stream)['digest'] == inputs.digest
    # Settings that would re-derive the frozen inputs are refused.
    with pytest.raises(ValueError, match='object-store inputs are frozen'):
        context.apply_config({'patch_loss_z_margin': 4},
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
        db.set_meta(names(store_path)['walk'], 'source', {'spiral_patch_erode_cells': 2})
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
    """Pin and materialize a real ingested store over a z window.

    OBJECT_STORE_SMOKE_STORE names the store (read-only);
    OBJECT_STORE_SMOKE_Z (default 9000:9200) is the window.
    """
    path = os.environ['OBJECT_STORE_SMOKE_STORE']
    z_begin, z_end = map(int, os.environ.get('OBJECT_STORE_SMOKE_Z', '9000:9200').split(':'))
    log = []
    inputs = load(path, log, z_begin=z_begin, z_end=z_end)
    print('\n'.join(log + inputs.report()))
    assert inputs.patches
    assert load(path, z_begin=z_begin, z_end=z_end).digest == inputs.digest
