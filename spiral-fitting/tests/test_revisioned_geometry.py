"""Prepare/install uses real CPU geometry and shared derivation helpers."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from config import Config, FitConfig
from dt_targets import DtTargetCacheManager
from fit_spiral import FitContext, PatchAtlas
import test_live_patch_relink as relink


@pytest.fixture
def context(monkeypatch):
    import point_collection
    monkeypatch.setattr(point_collection, 'can_use_surface_index_backend', lambda patches: False)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda states: None)
    ctx = FitContext.__new__(FitContext)
    ctx.config = FitConfig(Config({'influence_enabled': False, 'z_begin': 0, 'z_end': 200,
        'patch_erode_patches': 0, 'pcl_unattached_pcl_min_point_spacing': 0}).as_dict())
    ctx.device = torch.device('cpu')
    ctx.progress = None
    ctx.non_liftable_patch_paths = set()
    ctx._source_verified_patches = {'baseline': relink._flat_patch(50, 10, 10)}
    ctx._source_unverified_patches = {}
    pcl = relink._regular_pcl(5, [[50, 20, 20], [50, 30, 30]])
    ctx._source_point_collections = {5: pcl}
    ctx.next_id = 6
    ctx.verified_patches = {key: copy.copy(patch)
                            for key, patch in ctx._source_verified_patches.items()}
    ctx.verified_patches_list = list(ctx.verified_patches.values())
    ctx._prepare_patch_sampling_cache(ctx.verified_patches_list)
    ctx.patch_atlas = PatchAtlas(ctx.verified_patches, 'cpu').materialize()
    ctx.unverified_patch_atlas = None
    ctx.dt_target_whole_object = False
    ctx.dt_target_cache_manager = DtTargetCacheManager(100)
    ctx.using_tracks = False
    ctx.slice_to_spiral_transform = lambda x: x
    ctx.optimiser = torch.optim.Adam([torch.nn.Parameter(torch.tensor([1.0]))])
    ctx.influence_state = None
    ctx.interactive_dt_resume_iteration = None
    ctx.dist = SimpleNamespace(is_main_process=False)
    ctx.interactive_driver = None
    ctx.verified_patches_path = ''
    ctx.unverified_patches_path = ''
    ctx._derive_point_inputs(ctx.verified_patches, copy.deepcopy(ctx._source_point_collections), {})
    from test_run_boundary_settings import _context
    for name, value in vars(_context()).items():
        if not name.startswith('_') and not hasattr(ctx, name):
            setattr(ctx, name, value)
    return ctx


def test_patch_deletion_rederives_attachments_without_mutating_active_state(context):
    ctx = context
    original = ctx.regular_pcl_catalog[5]
    assert all('on_patch' in p for p in original['points'].values())
    assert all('on_patch' not in p for p in ctx._source_point_collections[5]['points'].values())
    numpy_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    candidate = ctx.prepare_input_changes([
        {'id': 'logical', 'kind': 'patch', 'source_id': 'baseline',
         'deleted': True, 'revision': 2}])
    assert 'baseline' in ctx.verified_patches
    assert all('on_patch' in p for p in original['points'].values())
    assert not candidate.verified_patches
    assert all('on_patch' not in p for p in candidate.regular_pcl_catalog[5]['points'].values())
    assert len(candidate.unattached_pcl_strips) == 1
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    optimiser = ctx.optimiser
    ctx.install_input_changes(candidate)
    assert not ctx.verified_patches
    assert ctx.optimiser is optimiser
    assert ctx.regular_pcl_catalog[5] is not original


def test_failed_mixed_batch_preserves_every_active_input(context):
    ctx = context
    atlas, points = ctx.patch_atlas, ctx.regular_pcl_catalog
    with pytest.raises((OSError, ValueError)):
        ctx.prepare_input_changes([
            {'id': 'logical', 'kind': 'patch', 'source_id': 'baseline', 'deleted': True},
            {'id': 'other', 'kind': 'patch', 'path': '/does-not-exist/revision-patch'}])
    assert ctx.patch_atlas is atlas
    assert ctx.regular_pcl_catalog is points
    assert 'baseline' in ctx.verified_patches


def test_mixed_revisions_keep_fiber_filename_and_restore_role_content(context, tmp_path):
    import json
    ctx = context
    pcl = tmp_path / 'same_windings.json'
    pcl.write_text(json.dumps({'vc_pointcollections_json_version': '1', 'collections': {
        '9': {'name': 'same', 'points': {
            '0': {'p': [80, 80, 50]}, '1': {'p': [110, 110, 50]}}}}}))
    fiber = tmp_path / 'fiber.json'
    fiber.write_text(json.dumps({'type': 'vc3d_fiber', 'line_points': [],
                                'control_points': [[320, 320, 200], [440, 440, 200]]}))
    records = [
        {'id': 'pcl-uuid', 'kind': 'pcl', 'path': str(pcl), 'role': 'same_winding',
         'source_id': '9', 'source_path': str(pcl), 'revision': 1},
        {'id': 'fiber-uuid', 'kind': 'fiber', 'path': str(fiber),
         'source_id': 'fiber-name', 'revision': 1}]
    candidate = ctx.prepare_input_changes(records)
    assert candidate.fiber_catalog['fiber-uuid']['file_basename'] == 'fiber-name.json'
    assert 'fiber-uuid' not in ctx.fiber_catalog
    ctx.install_input_changes(candidate)
    resident_id = ctx._workspace_membership['pcl-uuid']['resident_id']
    ctx.apply_config({'input_use_pcl_same_winding': False}, current_iteration=0)
    assert resident_id not in ctx.regular_pcl_catalog
    assert resident_id in ctx._source_point_collections
    ctx.apply_config({'input_use_pcl_same_winding': True}, current_iteration=0)
    assert resident_id in ctx.regular_pcl_catalog
    ctx.install_input_changes(ctx.prepare_input_changes([
        {**record, 'deleted': True, 'revision': 2} for record in records]))
    assert not ctx.fiber_catalog
    assert resident_id not in ctx.regular_pcl_catalog
    ctx.install_input_changes(ctx.prepare_input_changes([
        {**record, 'revision': 3} for record in records]))
    assert ctx._workspace_membership['pcl-uuid']['resident_id'] == resident_id
    assert ctx.fiber_catalog['fiber-uuid']['file_basename'] == 'fiber-name.json'


def test_invalid_absolute_annotations_fail_during_preparation(context, tmp_path):
    import json
    path = tmp_path / 'abs_winding.json'
    path.write_text(json.dumps({'vc_pointcollections_json_version': '1', 'collections': {
        '0': {'name': 'absolute', 'points': {'0': {'p': [20, 20, 50], 'wind_a': -1}}}}}))
    before = context.regular_pcl_catalog
    with pytest.raises(ValueError, match='non-positive'):
        context.prepare_input_changes([
            {'id': 'absolute-uuid', 'kind': 'pcl', 'path': str(path), 'role': 'absolute'}])
    assert context.regular_pcl_catalog is before


def test_rejected_patch_revision_does_not_remove_previous_geometry(context, monkeypatch):
    import fit_spiral
    original = context.verified_patches['baseline']
    monkeypatch.setattr(fit_spiral, 'load_tifxyz', lambda path: copy.copy(original))

    def reject(candidate):
        del candidate.verified_patches['baseline']
        return ['theta consistency rejection']

    monkeypatch.setattr(FitContext, '_build_theta_crossing_map', reject)
    with pytest.raises(ValueError, match='theta consistency'):
        context.prepare_input_changes([
            {'id': 'patch-uuid', 'kind': 'patch', 'source_id': 'baseline',
             'path': '/immutable/replacement', 'revision': 2}])
    assert context.verified_patches['baseline'] is original


def test_rebuild_adopts_baseline_geometry_without_reading_it_again(context):
    ctx = context
    original = ctx._source_verified_patches['baseline']
    candidate = ctx.prepare_input_changes([{'id': 'baseline-uuid', 'kind': 'patch',
        'source_id': 'baseline', 'path': '/immutable/source/need-not-be-reloaded',
        'role': 'verified', 'revision': 1, 'adopt': True}])
    assert candidate._source_verified_patches['baseline'] is original
    ctx.install_input_changes(candidate)
    assert 'baseline' in ctx.verified_patches


@pytest.mark.parametrize('role', ['verified', 'unverified'])
def test_baseline_adoption_preserves_initial_loader_exclusions(context, monkeypatch, role):
    import fit_spiral
    # A baseline may be absent because it has no valid quads, was eroded
    # away, or was excluded by the ROI, name filter, or source toggle.
    # Registration must not try to reinterpret any of those decisions.
    monkeypatch.setattr(fit_spiral, 'load_tifxyz',
                        lambda path: pytest.fail('excluded baseline was reloaded'))
    original = context._source_verified_patches['baseline']
    candidate = context.prepare_input_changes([{
        'id': 'excluded-uuid', 'kind': 'patch', 'source_id': 'excluded',
        'path': '/immutable/excluded', 'role': role, 'revision': 1,
        'adopt': True,
    }])
    assert candidate._workspace_membership['excluded-uuid'] == {
        'kind': 'patch', 'revision': 1, 'resident_id': 'excluded', 'deleted': False,
    }
    context.install_input_changes(candidate)
    assert context._source_verified_patches == {'baseline': original}
    assert not context._source_unverified_patches
    assert set(context.verified_patches) == {'baseline'}


@pytest.mark.parametrize('role', ['same_winding', 'relative', 'absolute', 'drawn_control_points'])
def test_patch_additions_rederive_one_view_per_collection(context, monkeypatch, role):
    import fit_spiral
    pcl = relink._regular_pcl(5, [[50, 20, 20], [50, 30, 30], [50, 520, 520], [50, 530, 530]])
    fit_spiral.stamp_loaded_pcl_metadata(pcl, '/inputs/pcl.json', role, 5)
    if role == 'absolute':
        for point in pcl['points'].values():
            point['winding_annotation'] = 1.0
    context._source_point_collections = {5: pcl}
    monkeypatch.setattr(fit_spiral, 'load_tifxyz', lambda path: relink._flat_patch(50, 510, 510))
    context.install_input_changes(context.prepare_input_changes([
        {'id': 'new', 'kind': 'patch', 'path': '/immutable/patch'}]))
    assert len(context.cross_patch_pcls) == 1
    assert {key: len(points) for key, points in context.cross_patch_pcls[0]['points_by_patch'].items()} == {
        'baseline': 2, 'new': 2}
    assert all('on_patch' not in point for point in context._source_point_collections[5]['points'].values())


def test_fiber_spacing_reuses_revision_identity(context, tmp_path):
    import json
    path = tmp_path / 'fiber.json'
    path.write_text(json.dumps({'type': 'vc3d_fiber', 'line_points': [], 'control_points': [[0, 0, 200], [400, 0, 200]]}))
    context.install_input_changes(context.prepare_input_changes([
        {'id': 'fiber-id', 'kind': 'fiber', 'path': str(path), 'revision': 2}]))
    before = context._workspace_membership['fiber-id']['resident_id']
    context.apply_config({'pcl_fiber_min_point_spacing': 5.0}, current_iteration=0)
    assert context._workspace_membership['fiber-id']['resident_id'] == before
    assert context._workspace_membership['fiber-id']['revision'] == 2
    assert list(context.fiber_catalog) == ['fiber-id']


@pytest.mark.parametrize('enabled', [True, False])
def test_baseline_fiber_adoption_preserves_startup_exclusions(context, monkeypatch, enabled):
    import fit_spiral
    context.apply_config({'input_use_fibers': enabled}, current_iteration=0)
    monkeypatch.setattr(fit_spiral, 'load_fiber_point_collection',
                        lambda *args, **kwargs: pytest.fail('excluded fiber was reloaded'))
    record = {'id': 'excluded-fiber', 'kind': 'fiber', 'source_id': 'malformed',
              'path': '/immutable/malformed.json', 'revision': 1, 'adopt': True}
    before = set(context._source_point_collections)
    for _ in range(2):
        context.install_input_changes(context.prepare_input_changes([record]))
        assert set(context._source_point_collections) == before
        assert not context.fiber_catalog
        assert context._workspace_membership['excluded-fiber'] == {
            'kind': 'fiber', 'revision': 1, 'resident_id': 6, 'deleted': False}
    assert context.next_id == 7


def test_input_revision_preserves_current_track_policy(context, monkeypatch):
    import fit_spiral
    from unittest.mock import Mock
    ctx = context
    ctx.using_tracks = True
    ctx._refresh_trusted_geometry()
    ctx.tracks = ['retained track']
    ctx.track_sampling_config = fit_spiral.validate_track_sampling_config(ctx.config)
    ctx.track_families = ctx.track_source_ids = None
    ctx.track_crossing_cache = ctx.track_graph = None
    prepare = Mock(return_value=None)
    monkeypatch.setattr(fit_spiral, 'prepare_main_phase_tracks', prepare)
    monkeypatch.setattr(fit_spiral, 'configure_prepared_track_sampling', Mock())
    ctx.apply_config({'track_exclusion_radius': 10, 'track_max_tortuosity': 2,
                      'track_crossing_mode': 'track_walk',
                      'track_crossing_precompute_max': 12,
                      'track_length_bin_weights': [1, 2, 3]}, current_iteration=0)
    prepare.reset_mock()
    ctx.install_input_changes(ctx.prepare_input_changes([
        {'id': 'baseline', 'kind': 'patch', 'deleted': True, 'revision': 2}]))
    policy = prepare.call_args.kwargs['sampling_config']
    expected = fit_spiral.validate_track_sampling_config(ctx.config)
    assert policy.keys() == expected.keys()
    for key in expected:
        np.testing.assert_equal(policy[key], expected[key])


def test_exclusion_radius_preserves_unverified_revisions(context, monkeypatch):
    import fit_spiral
    ctx = context
    ctx._source_unverified_patches = {
        'deleted': relink._flat_patch(100, 510, 510),
        'replaced': relink._flat_patch(100, 610, 610),
    }
    replacement = relink._flat_patch(100, 710, 710)
    added = relink._flat_patch(100, 810, 810)
    monkeypatch.setattr(fit_spiral, 'load_tifxyz',
                        lambda path: {'replacement': replacement, 'added': added}[path])
    records = [
        {'id': 'deleted', 'kind': 'patch', 'role': 'unverified', 'deleted': True, 'revision': 2},
        {'id': 'replaced', 'kind': 'patch', 'role': 'unverified', 'path': 'replacement', 'revision': 2},
        {'id': 'added', 'kind': 'patch', 'role': 'unverified', 'path': 'added', 'revision': 1},
    ]
    ctx.install_input_changes(ctx.prepare_input_changes(records))
    membership = copy.deepcopy(ctx._workspace_membership)
    ctx.unverified_patches_path = '/mutable/dataset'
    monkeypatch.setattr(ctx, '_load_patches_from_dir',
                        lambda path: pytest.fail('revisioned patches reloaded from dataset'))
    for radius in (1, 0):
        ctx.apply_config({'patch_unverified_patch_exclusion_radius': radius}, current_iteration=0)
        assert set(ctx.unverified_patches) == {'replaced', 'added'}
        assert ctx._workspace_membership == membership
        for pid, source in [('replaced', replacement), ('added', added)]:
            assert ctx.unverified_patches[pid] is not source
            torch.testing.assert_close(ctx.unverified_patches[pid].zyxs, source.zyxs)


@pytest.mark.parametrize('dataset_change', ['edit', 'delete'])
def test_adopted_fiber_reingests_snapshot(context, tmp_path, dataset_change):
    import json
    from fit_spiral import load_fiber_point_collection
    dataset = tmp_path / 'dataset.json'
    snapshot = tmp_path / 'snapshot.json'
    document = json.dumps({'type': 'vc3d_fiber', 'line_points': [],
                          'control_points': [[0, 0, 200], [400, 0, 200]]})
    dataset.write_text(document)
    snapshot.write_text(document)
    pcl = load_fiber_point_collection(str(dataset), 6, min_point_spacing=0)
    pcl.update(sampling_group='fibers', file_basename='dataset.json', source_file=str(dataset))
    pcl.setdefault('metadata', {}).update(input_role='fiber', logical_input_kind='fiber',
                                          logical_input_id='dataset', winding_is_absolute=False)
    context._source_point_collections[6] = pcl
    context.next_id = 7
    candidate = context.prepare_input_changes([{
        'id': 'fiber-uuid', 'kind': 'fiber', 'source_id': 'dataset',
        'source_path': str(dataset), 'path': str(snapshot), 'revision': 1, 'adopt': True}])
    assert pcl['source_file'] == str(dataset)
    assert candidate._source_point_collections[6]['points'] == pcl['points']
    context.install_input_changes(candidate)
    if dataset_change == 'delete':
        dataset.unlink()
    else:
        dataset.write_text('unaccepted invalid geometry')
    context.apply_config({'pcl_fiber_min_point_spacing': 5.0}, current_iteration=0)
    assert context.fiber_catalog['fiber-uuid']['source_file'] == str(snapshot)
    assert context._workspace_membership['fiber-uuid']['revision'] == 1
    assert context._workspace_membership['fiber-uuid']['resident_id'] == 6
