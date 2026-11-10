"""Dataset selection, fingerprints and derived-copy policies share store identities."""
import uuid

import numpy as np
import pytest
from vc_object_store import ObjectStore, extension_uuid, remove_history_and_disable

from vesuvius.neural_tracing.tracers.shared.annotations.fiber_store import (
    FiberStore, _references_digest, fiber_catalog, fibers_digest, fiber_source_version, read_polylines, write_fibers,
)


def source(tmp_path):
    path = tmp_path / 'fibers.vcobj'
    ids = [str(uuid.uuid4()) for _ in range(2)]
    xyz = np.c_[np.arange(32.), np.full(32, 20.), np.full(32, 30.)]
    write_fibers(path, [(dict(id=fid, name=fid, family='H'), xyz) for fid in ids], volume='vol')
    return path, ids


def test_uuid_normalization_and_digest_owner(tmp_path):
    path, ids = source(tmp_path)
    assert [r['id'] for r, _ in read_polylines(path, [ids[0].upper(), ids[0]], volume='vol')] == [ids[0]]
    assert fibers_digest(path, [ids[0].upper(), ids[0]]) == fibers_digest(path, [ids[0]])
    assert read_polylines(path, [], volume='vol') == []
    assert fibers_digest(path, []) == _references_digest([])
    assert fibers_digest(path, []) != fibers_digest(path)
    for operation in (read_polylines, fibers_digest):
        with pytest.raises(ValueError):
            operation(path, ['invalid'])
    ref = fiber_catalog(path, 'vol')[0]['reference']
    assert _references_digest([ref]) != _references_digest([dict(ref, store=str(uuid.uuid4()))])


def test_catalog_measures_selected_geometry_not_ingestion_summary(tmp_path):
    path, ids = source(tmp_path)
    xyz = np.array([[10., 20., 30.], [10., 24., 30.], [13., 24., 30.]])
    with ObjectStore.open(str(path), author='editor') as store:
        version = store.current_version(ids[0])
        with store.write(expected_versions={ids[0]: version}) as tx:
            tx.edit_geometry_sequence(ids[0], dict(array='line_points', start=0, erase_count=32,
                                      parts=[dict(count=3, data=xyz.tobytes())]), version)
    row = next(r for r in fiber_catalog(path, 'vol') if r['id'] == ids[0])
    assert row['point_count'] == 3
    assert row['length'] == 7
    assert row['bbox'] == [10, 20, 30, 13, 24, 30]
    np.testing.assert_array_equal(read_polylines(path, [ids[0]])[0][1], xyz)


def test_strip_index_rejects_changed_ordinal_source(tmp_path):
    from types import SimpleNamespace
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import RunFibers
    path, ids = source(tmp_path)
    index = SimpleNamespace(meta=dict(fibers_digest=fibers_digest(path)), unit_length=np.array([10.]))
    RunFibers(path, index, 1)
    with ObjectStore.open(str(path), author='editor') as store:
        store.delete_object(min(ids))
    with pytest.raises(ValueError, match='source changed'):
        RunFibers(path, index, 1)
    index.meta.clear()
    with pytest.raises(ValueError, match='missing source digest'):
        RunFibers(path, index, 1)


@pytest.mark.parametrize('change', ['translation', 'coordinate_scale', 'missing_reference', 'wrong_owner'])
@pytest.mark.parametrize('history', [False, True])
def test_strip_index_checks_producer_reference_before_certifying_matches(tmp_path, monkeypatch, change, history):
    import json
    from scipy.spatial import cKDTree
    from vesuvius.neural_tracing.tracers.sheets.scripts import match_fibers_to_patches as matcher
    from vesuvius.neural_tracing.tracers.sheets.data.patches import PatchInfo
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip import build_index
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.data import StripIndex, RunFibers
    path, ids = source(tmp_path)
    if not history:
        remove_history_and_disable(path)
    dataset = FiberStore(path, 'vol', split='all')
    xyz = dataset.geometry(0).points
    monkeypatch.setattr(matcher, '_STATE', dict(fibers=dataset, tree=cKDTree(xyz), pos=xyz,
        nrm=np.tile([0., 0., 1.], (len(xyz), 1)), rc=np.c_[np.zeros(len(xyz), int), np.arange(len(xyz))],
        patch=np.zeros(len(xyz), int), cell=np.array([1.]), sample=2., candidates=2, reach=2.,
        max_inward=1., max_outward=1., max_tangential=2., min_run=4., roots=['root'], names=['patch']))
    runs = matcher.match_fiber(0)
    assert runs and runs[0]['fiber_reference'] == dataset.catalog[0]['reference']
    evidence = tmp_path/'runs.jsonl'
    evidence.write_text(''.join(json.dumps(run)+'\n' for run in runs))
    monkeypatch.setattr(build_index, 'build_index', lambda *args: [PatchInfo('patch', 2, 32, 1., 64, (0, 0, 0, 32, 32, 32), 'root')])
    def build(destination):
        build_index.main(['--fibers', str(path), '--runs', str(evidence), '--patches-dir', 'root',
                          '--grid-scale', '1', '--out', str(destination)])
    good = tmp_path/'good'
    build(good)
    assert len(RunFibers(path, StripIndex(good, 'train'), 1)) == 1
    if change in ('translation', 'coordinate_scale'):
        with ObjectStore.open(str(path), author='editor') as store:
            fid = dataset.catalog[0]['id']
            if change == 'translation':
                store.edit_geometry(fid, [dict(array='line_points', element_offset=0,
                                                data=np.asarray(xyz+[0, 100, 0], '<f8').tobytes())],
                                    store.current_version(fid))
            else:
                store.set_meta(fid, 'coordinates', dict(volume_uuid='vol', scale_factor=2))
    else:
        for run in runs:
            if change == 'missing_reference':
                run.pop('fiber_reference')
            else:
                run['fiber_reference'] = dict(run['fiber_reference'], store=str(uuid.uuid4()))
        evidence.write_text(''.join(json.dumps(run)+'\n' for run in runs))
    rejected = tmp_path/'rejected'
    with pytest.raises(ValueError, match='stale or missing fiber reference'):
        build(rejected)
    assert not rejected.exists()


def test_training_coordinates_use_declared_frame_not_registration_routes(tmp_path):
    path, _ = source(tmp_path)
    before = FiberStore(path, 'vol', split='all')
    xyz = before.geometry(0).points.copy()
    references = [row['reference'] for row in before.catalog]
    with ObjectStore.open(str(path), author='editor') as store:
        store.register_volume_transform('vol', 'other', [[2., 0., 0., 100.], [0., 2., 0., 0.], [0., 0., 2., 0.]])
    after = FiberStore(path, 'vol', split='all')
    assert [row['reference'] for row in after.catalog] == references
    np.testing.assert_array_equal(after.geometry(0).points, xyz)
    # This adapter deliberately supports the declared volume/scale only.
    # Foreign-volume registration cannot silently reposition match evidence.
    assert fiber_catalog(path, 'other') == []


def test_strip_resume_requires_complete_input_identity(tmp_path):
    from vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer_all import pending_fibers
    path, _ = source(tmp_path)
    catalog = fiber_catalog(path, 'vol')
    completed = {r['id']: r['reference'] for r in catalog}
    assert pending_fibers(catalog, [0, 1], completed) == []
    completed[catalog[0]['id']] = None  # legacy UUID-only completion
    completed[catalog[1]['id']] = dict(catalog[1]['reference'], generation=str(uuid.uuid4()))
    assert pending_fibers(catalog, [0, 1], completed) == [0, 1]


@pytest.mark.parametrize('change_during_read', [False, True])
@pytest.mark.parametrize('reader', [read_polylines, fiber_catalog])
def test_frozen_evidence_rejects_source_edits(tmp_path, monkeypatch, change_during_read, reader):
    from vesuvius.neural_tracing.tracers.shared.annotations import fiber_store
    path, ids = source(tmp_path)
    expected = fiber_source_version(path)
    assert len(reader(path, volume='vol', source_version=expected)) == 2
    def edit():
        with ObjectStore.open(str(path), author='editor') as store:
            store.set_meta(ids[0], 'coordinates', dict(volume_uuid='vol', scale_factor=2))
    if change_during_read:
        original = fiber_store._read_rows
        def interleaved(*args, **kwargs):
            yield from original(*args, **kwargs)
            edit()
        monkeypatch.setattr(fiber_store, '_read_rows', interleaved)
    else:
        edit()
    with pytest.raises(ValueError, match='source changed'):
        reader(path, volume='vol', source_version=expected)
    with pytest.raises(ValueError, match='missing source version'):
        reader(path, volume='vol', source_version={})


@pytest.mark.parametrize('change', ['metadata', 'geometry'])
@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('history', [False, True])
def test_training_source_change_never_mixes_catalog_and_geometry(tmp_path, change, cached, history):
    path, ids = source(tmp_path)
    if not history:
        remove_history_and_disable(path)
    dataset = FiberStore(path, 'vol', split='all')
    before = dataset.manifest_entries()
    if cached:
        dataset.geometry(0)
    with ObjectStore.open(str(path), author='editor') as store:
        fid = dataset.catalog[0]['id']
        if change == 'metadata':
            store.set_meta(fid, 'coordinates', dict(volume_uuid='vol', scale_factor=2))
        else:
            store.edit_geometry(fid, [dict(array='line_points', element_offset=0,
                                           data=np.array([1., 20., 30.], '<f8').tobytes())],
                                store.current_version(fid))
    for operation in (lambda: dataset.geometry(0), dataset.manifest_entries,
                      lambda: dataset.nearby_fiber_ids([16, 20, 30], 40, ids[1]),
                      lambda: list(dataset._blocks([0, 0, 0], [64, 64, 64]))):
        with pytest.raises(RuntimeError, match='source changed'):
            operation()
    fresh = FiberStore(path, 'vol', split='all')
    assert fresh.manifest_entries() != before
    fresh.geometry(0)


def test_managed_payloads_require_explicit_derived_geometry_policy(tmp_path, monkeypatch):
    path, ids = source(tmp_path)
    fid = ids[0]
    with ObjectStore.open(str(path), author='annotator') as store:
        spec = dict(name='ext_test_manual', extension='test.manual', key_space='objects', versioned=True,
                    columns=[dict(name='owner', type='blob'), dict(name='label', type='integer')],
                    primary_key=['owner'])
        store.create_extension_table(spec)
        with store.write(expected_versions={fid: store.current_version(fid)}) as tx:
            tx.replace_extension_rows(spec['name'], fid, [[extension_uuid(fid), 7]])
        payload = tmp_path / 'notes.txt'
        payload.write_text('manual notes')
        digest = store.put_artifact(payload)
        store.set_meta(fid, 'artifacts', [dict(name='notes.txt', sha256=digest, length=12)])
    rows = read_polylines(path, [fid], volume='vol')
    assert {'vcobj_extensions', 'artifacts'} <= rows[0][0]['meta'].keys()
    target = tmp_path / 'copy.vcobj'
    with pytest.raises(ValueError, match='Managed extension/artifact'):
        write_fibers(target, rows, volume='vol')
    assert not target.exists()
    row, points = rows[0]
    write_fibers(target, [(row, points + [0, 1, 0])], volume='vol', metadata_policy='geometry_only')
    copied, moved = read_polylines(target, [fid], volume='vol')[0]
    assert not {'vcobj_extensions', 'artifacts'} & copied['meta'].keys()
    assert copied['meta']['derivation'] == dict(policy='geometry_only', source=row['reference'])
    np.testing.assert_allclose(moved, points + [0, 1, 0])
    # The selected source and managed payloads remain untouched.
    assert read_polylines(path, [fid], volume='vol')[0][0]['meta'] == row['meta']
    # The snapping consumer explicitly chooses the policy, independent of its numerical sampler.
    from vesuvius.neural_tracing.tracers.fibers.data import fiber_snapping
    monkeypatch.setattr(fiber_snapping, 'snap_native', lambda lines, *args: [p + [0, 1, 0] for p in lines])
    snapped = tmp_path / 'snapped.vcobj'
    fiber_snapping.snap_store(path, snapped, None, None, None, None, None, None, None, {})
    projected = read_polylines(snapped, [fid], volume='vol')[0][0]
    assert projected['meta']['derivation'] == copied['meta']['derivation']
    assert not {'vcobj_extensions', 'artifacts'} & projected['meta'].keys()
