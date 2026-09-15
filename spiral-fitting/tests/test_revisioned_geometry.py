"""Prepare/install uses real CPU geometry and shared derivation helpers."""

from geometry_fixtures import context

import copy
import numpy as np
import pytest
import torch
from fit_spiral import FitContext
import geometry_fixtures as relink
from runtime_fixtures import resident


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


@pytest.mark.parametrize('enabled', [False, True])
def test_input_revision_preserves_original_run_dt_schedule(context, enabled):
    context.configure_dt_loss_schedule(
        100, 100, {'enabled': enabled, 'last_fraction': 0.25})
    expected = 175 if enabled else None
    candidate = context.prepare_input_changes([
        {'id': 'logical', 'kind': 'patch', 'source_id': 'baseline',
         'deleted': True, 'revision': 2}],
        current_iteration=150, target_iteration=200)
    assert candidate.run_dt_resume_iteration == expected
    context.install_input_changes(candidate)
    assert context.run_dt_resume_iteration == expected
    context.clear_interactive_run_state()
    assert context.run_dt_resume_iteration is None


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


@pytest.mark.parametrize('metadata', [
    {'format': 'tifxyz'},
    [],
    {'scale': None},
    {'scale': [1]},
    {'scale': ['bad', 1]},
    {'scale': [0, 1]},
    {'scale': [float('inf'), 1]},
])
def test_malformed_patch_metadata_preserves_resident(context, resident, tmp_path, metadata):
    import json
    from tifxyz import save_tifxyz

    save_tifxyz(context._source_verified_patches['baseline'].zyxs.numpy(),
                str(tmp_path), 'draft', 1, 1, 'test')
    path = tmp_path / 'draft'
    meta_path = path / 'meta.json'
    valid_metadata = meta_path.read_text()
    meta_path.write_text(json.dumps(metadata))
    session, _, _, _ = resident
    session._context = context
    atlas, points, optimiser = (context.patch_atlas, context.regular_pcl_catalog,
                                context.optimiser)
    records = [{'id': 'baseline', 'kind': 'patch', 'path': str(path)}]
    result = session.apply_input_changes('malformed', records, timeout=5)
    assert not result['applied']
    assert 'meta.json' in str(result)
    assert context.patch_atlas is atlas
    assert context.regular_pcl_catalog is points
    assert context.optimiser is optimiser
    assert session._context is context
    assert session._completed == 7
    meta_path.write_text(valid_metadata)
    assert session.apply_input_changes('repaired', records, timeout=5)['applied']
    assert context.optimiser is optimiser


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


@pytest.mark.parametrize('role', ['verified', 'unverified'])
@pytest.mark.parametrize('adopt', [True, False])
def test_theta_rejected_baseline_replay(context, monkeypatch, role, adopt):
    import fit_spiral
    # Startup retains pre-validation sources, even after excluding a patch
    # from the active geometry. Replay must repeat that exclusion successfully.
    excluded = relink._flat_patch(50, 510, 510)
    source = getattr(context, f'_source_{role}_patches')
    source['excluded'] = excluded
    build_theta = FitContext._build_theta_crossing_map

    def reject(candidate):
        warnings = []
        if 'excluded' in getattr(candidate, f'{role}_patches'):
            warnings = candidate._exclude_non_liftable_patches(
                ['excluded'] if role == 'verified' else [],
                ['excluded'] if role == 'unverified' else [],
                {'inconsistent_edges': 1})
        return warnings + build_theta(candidate)

    monkeypatch.setattr(FitContext, '_build_theta_crossing_map', reject)
    monkeypatch.setattr(fit_spiral, 'load_tifxyz',
                        lambda path: pytest.fail('baseline was reloaded') if adopt
                        else copy.copy(excluded))
    record = {'id': 'excluded-uuid', 'kind': 'patch', 'source_id': 'excluded',
              'path': '/immutable/excluded', 'role': role, 'revision': 1,
              'adopt': adopt}
    if not adopt:
        with pytest.raises(ValueError, match='theta consistency'):
            context.prepare_input_changes([record])
        assert set(context.verified_patches) == {'baseline'}
        assert source['excluded'] is excluded
        return

    for _ in range(2):
        candidate = context.prepare_input_changes([record])
        assert 'excluded' not in getattr(candidate, f'{role}_patches')
        assert getattr(candidate, f'_source_{role}_patches')['excluded'] is excluded
        assert set(candidate.verified_patches) == {'baseline'}
        assert any('non-liftable patch' in warning for warning in candidate._input_warnings)
        assert candidate._workspace_membership['excluded-uuid'] == {
            'kind': 'patch', 'revision': 1, 'resident_id': 'excluded', 'deleted': False}
        context.install_input_changes(candidate)


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


@pytest.mark.parametrize('radius', [0, 10])
@pytest.mark.parametrize('enabled', [False, True])
def test_combined_role_and_track_settings_use_final_geometry(
        context, monkeypatch, radius, enabled):
    import fit_spiral
    ctx = context
    ctx.config.update({'input_use_pcl_relative': not enabled,
                       'track_exclusion_radius': 5})
    ctx._source_point_collections[5]['metadata']['input_role'] = 'relative'
    ctx.using_tracks = True
    ctx.tracks = ['retained track']
    ctx.track_families = ctx.track_source_ids = None
    ctx.track_crossing_cache = ctx.track_graph = None
    ctx._refresh_trusted_geometry()
    old_tree = ctx.trusted_geometry_tree

    def prepare(*args, anchor_tree, sampling_config, **kwargs):
        return {'flat_zyx_cpu': torch.zeros((1, 3)),
                'anchor_tree': anchor_tree, 'policy': sampling_config}

    monkeypatch.setattr(fit_spiral, 'prepare_main_phase_tracks', prepare)
    ctx.apply_config({'input_use_pcl_relative': enabled,
                      'track_exclusion_radius': radius,
                      'track_max_tortuosity': 2}, current_iteration=0)

    assert bool(ctx.regular_pcl_catalog) == enabled
    assert ctx.trusted_geometry_tree is not old_tree
    assert ctx.prepared_main_tracks['anchor_tree'] is ctx.trusted_geometry_tree
    assert ctx.prepared_main_tracks['policy']['max_tortuosity'] == 2
    assert ctx.preview_extent_tracks[0] is ctx.prepared_main_tracks['flat_zyx_cpu']
