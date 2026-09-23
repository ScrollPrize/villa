"""Run-boundary application of settings that used to demand a rebuild.

Each test builds a FitContext piecemeal (the pattern test_fiber_supervision
uses) and drives FitContext.apply_config the way the interactive runtime does.
"""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from config import Config
from fit_spiral import FitContext, _UnattachedPclStripList


def _context(**overrides):
    context = FitContext.__new__(FitContext)
    context.config = Config().as_dict()
    context.config.update({'z_begin': 0, 'z_end': 200})
    context.config.update(overrides)
    context.shell_map = None
    context.shell_envelope = None
    context.shell_outer_winding_idx = None
    context.shell_patch = None
    context.tracks = []
    context.prepared_main_tracks = None
    context.verified_patches = {}
    context.verified_patches_list = []
    context.cross_patch_pcls = []
    context.unattached_pcl_strips = _UnattachedPclStripList()
    context.unattached_strip_sampling_groups = []
    context.resolved_links = []
    context.link_components = []
    context.fiber_catalog = {}
    context.regular_pcl_catalog = {}
    context.fiber_direction_samples = None
    context.dt_target_cache_manager = SimpleNamespace(
        update_interval=100, reset=Mock())
    context.theta_crossing_map = SimpleNamespace(invalidate=Mock())
    context._rebuild_pcl_sampling_strata = Mock()
    context._refresh_trusted_geometry = Mock()
    context._build_theta_crossing_map = Mock(return_value=[])
    context._make_shell_polar_map = Mock(return_value='rebuilt shell map')
    return context


def _fiber(cid, logical_id, zyxs, hv=None):
    points = {
        i: {"id": i, "collectionId": cid,
            "p": [z[2], z[1], z[0]],
            "zyx": np.asarray(z, dtype=np.float32),
            "winding_annotation": float("nan")}
        for i, z in enumerate(zyxs)
    }
    return {
        "id": cid, "file_basename": f"{logical_id}.json",
        "sampling_group": "fibers",
        "metadata": {"logical_input_id": logical_id,
                     "logical_input_kind": "fiber",
                     "winding_is_absolute": False,
                     "hv_classification": hv or {}},
        "points": points,
        "kept_orig_indices": np.arange(len(zyxs)),
        "control_line_indices": np.arange(len(zyxs)),
        "branches": [],
    }


def _regular(cid, zyxs, attached=()):
    points = {}
    for i, z in enumerate(zyxs):
        point = {"id": i, "collectionId": cid,
                 "zyx": np.asarray(z, dtype=np.float32),
                 "winding_annotation": float("nan")}
        if i in attached:
            point["on_patch"] = {"id": "patch"}
        points[i] = point
    return {"id": cid, "name": f"pcl{cid}", "sampling_group": "regular.json",
            "metadata": {"winding_is_absolute": False}, "points": points}


# --- shell atlas -----------------------------------------------------------------

def test_shell_atlas_settings_rebuild_the_resident_lookup():
    context = _context()
    context.shell_map = 'old shell map'
    context.apply_config({'shell_num_theta_bins': 360}, current_iteration=0)
    assert context.shell_map == 'rebuilt shell map'
    context._make_shell_polar_map.assert_called_once_with()

    # The confidence floor is read live by the lookup; no table rebuild.
    context._make_shell_polar_map.reset_mock()
    context.apply_config({'shell_min_confidence': 0.5}, current_iteration=0)
    context._make_shell_polar_map.assert_not_called()
    assert context.config['shell_min_confidence'] == 0.5


def test_shell_atlas_settings_are_refused_when_tracks_were_shell_filtered():
    context = _context()
    context.shell_map = 'old shell map'
    context.shell_envelope = object()
    with pytest.raises(ValueError, match='filtered this session'):
        context.apply_config(
            {'shell_table_smooth_sigma_z': 8.0}, current_iteration=0)
    assert context.config['shell_table_smooth_sigma_z'] == 4.0
    assert context.shell_map == 'old shell map'


# --- fiber directions -------------------------------------------------------------

def test_fiber_direction_weight_needs_resident_samples():
    context = _context()
    with pytest.raises(ValueError, match='fiber-direction samples'):
        context.apply_config(
            {'loss_weight_fiber_directions': 1.0}, current_iteration=0)
    assert context.config['loss_weight_fiber_directions'] == 0.0
    context.fiber_direction_samples = {'position_zyx': np.zeros((1, 3))}
    context.apply_config(
        {'loss_weight_fiber_directions': 1.0}, current_iteration=0)
    assert context.config['loss_weight_fiber_directions'] == 1.0
    # Lowering to zero never needs the samples.
    context.fiber_direction_samples = None
    context.apply_config(
        {'loss_weight_fiber_directions': 0.0}, current_iteration=0)


# --- sampling weights ------------------------------------------------------------

def test_sampling_weights_rebuild_the_strata_after_validating_every_group():
    context = _context()
    context.cross_patch_pcls = [{'sampling_group': 'relative.json'}]
    context.unattached_strip_sampling_groups = ['fibers']
    with pytest.raises(KeyError, match='fibers'):
        context.apply_config(
            {'pcl_sampling_weights': {'relative': 1.0}}, current_iteration=0)
    assert context.config['pcl_sampling_weights'] is None
    context._rebuild_pcl_sampling_strata.assert_not_called()

    context.apply_config(
        {'pcl_sampling_weights': {'relative': 1.0, 'fibers': 2.0}},
        current_iteration=0)
    context._rebuild_pcl_sampling_strata.assert_called_once_with()
    context._rebuild_pcl_sampling_strata.reset_mock()
    context.apply_config({'pcl_sampling_weights': None}, current_iteration=0)
    context._rebuild_pcl_sampling_strata.assert_called_once_with()


# --- fiber views -----------------------------------------------------------------

def test_fiber_view_settings_are_a_no_op_without_a_fiber_catalog():
    context = _context()
    context.apply_config({'pcl_use_fiber_links': False}, current_iteration=0)
    context._rebuild_pcl_sampling_strata.assert_not_called()
    assert context.config['pcl_use_fiber_links'] is False


def test_fiber_spacing_reloads_the_documents_and_refuses_missing_ones(tmp_path):
    context = _context()
    context.fiber_catalog = {
        'gone': _fiber(1, 'gone', [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])}
    context.fiber_catalog['gone']['source_file'] = str(tmp_path / 'gone.json')
    context.prepare_input_changes = Mock()
    context.install_input_changes = Mock(return_value=[])
    with pytest.raises(ValueError, match='no longer on disk'):
        context.apply_config(
            {'pcl_fiber_min_point_spacing': 5.0}, current_iteration=0)
    assert context.config['pcl_fiber_min_point_spacing'] == 40.0
    context.prepare_input_changes.assert_not_called()

    present = tmp_path / 'present.json'
    present.write_text('{}')
    context.fiber_catalog = {
        'present': _fiber(2, 'present', [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])}
    context.fiber_catalog['present']['source_file'] = str(present)
    context.fiber_catalog['present']['metadata'][
        'logical_input_revision'] = 'abc'
    context.apply_config(
        {'pcl_fiber_min_point_spacing': 5.0}, current_iteration=0)
    context.prepare_input_changes.assert_called_once_with(
        [{'kind': 'fiber', 'path': str(present), 'id': 'present', 'source_id': 'present',
          'revision': 'abc'}])


@pytest.mark.parametrize('exponential', [False, True])
def test_missing_fiber_rejects_lr_changes_without_mutating_optimizer(
        tmp_path, exponential):
    context = _context(optimizer_exp_lr_schedule=exponential)
    context.fiber_catalog = {'gone': {'source_file': str(tmp_path / 'gone.json')}}
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    context.optimiser = torch.optim.SGD(
        [parameter], lr=context.config['optimizer_learning_rate'])
    context.lr_scheduler = None
    context._realign_lr_schedule(10)
    config_before = dict(context.config)
    optimizer_before = copy.deepcopy(context.optimiser.state_dict())
    scheduler = context.lr_scheduler
    scheduler_before = copy.deepcopy(scheduler.state_dict()) if scheduler else None
    horizon_before = context.num_training_steps

    with pytest.raises(ValueError, match='no longer on disk'):
        context.apply_config({
            'pcl_fiber_min_point_spacing': 5.0,
            'optimizer_learning_rate': 0.5,
            'optimizer_lr_final_factor': 0.2,
            'optimizer_num_training_steps': horizon_before + 100,
            'optimizer_exp_lr_schedule': not exponential,
        }, current_iteration=10)

    assert context.config == config_before
    assert context.optimiser.state_dict() == optimizer_before
    assert context.lr_scheduler is scheduler
    if scheduler is not None:
        assert scheduler.state_dict() == scheduler_before
    assert context.num_training_steps == horizon_before
    context.apply_config({}, current_iteration=10)
    assert context.optimiser.state_dict() == optimizer_before
    assert context.num_training_steps == horizon_before


# --- regular strips --------------------------------------------------------------

def test_unattached_spacing_rederives_regular_strips_from_the_catalog():
    context = _context(pcl_unattached_pcl_min_point_spacing=0.0)
    line = [[float(i), 0.0, 0.0] for i in range(6)]
    context.regular_pcl_catalog = {
        # Fully attached: never an unattached strip.
        3: _regular(3, line, attached=range(6)),
        # Two attached, four free: strip over the whole collection.
        4: _regular(4, line, attached=(0, 5)),
        # Absolute windings never form strips.
        5: {**_regular(5, line), 'metadata': {'winding_is_absolute': True}},
    }
    strips = context.unattached_pcl_strips
    strips.append({'id': 4, 'logical_input_kind': None,
                   'zyxs': np.zeros((6, 3), np.float32),
                   'windings': np.zeros(6, np.float32),
                   'radial_offsets': np.zeros(6, np.float32)})
    fiber_strip = {'id': 9, 'logical_input_kind': 'fiber',
                   'zyxs': np.zeros((2, 3), np.float32),
                   'windings': np.zeros(2, np.float32),
                   'radial_offsets': np.zeros(2, np.float32)}
    strips.append(fiber_strip)
    context.unattached_strip_sampling_groups = ['regular.json', 'fibers']

    context.apply_config(
        {'pcl_unattached_pcl_min_point_spacing': 2.5}, current_iteration=0)

    assert [strip['id'] for strip in strips] == [4, 9]
    assert strips[1] is fiber_strip
    # Greedy decimation keeps 0, 3 and the forced last point 5.
    assert strips[0]['zyxs'][:, 0].tolist() == [0.0, 3.0, 5.0]
    assert np.all(strips[0]['windings'] == 0.0)
    assert context.unattached_strip_sampling_groups == ['regular.json', 'fibers']
    assert context.cross_patch_pcls == []
    context._rebuild_pcl_sampling_strata.assert_called_once_with()
    context._refresh_trusted_geometry.assert_called_once_with()


# --- point-collection role toggles ---------------------------------------------


@pytest.mark.parametrize('settings, error', [
    ({'track_max_track_crossing_per_step': -1}, 'max_track_crossing_per_step'),
    ({'track_crossing_precompute_max': 1.5}, 'track_crossing_precompute_max'),
])
def test_invalid_track_policy_preserves_enabled_pcl_inputs(settings, error):
    context = _context(input_use_pcl_same_winding=True)
    context.tracks = ['resident track']
    context.prepared_main_tracks = {}
    pcl = _regular(7, [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    context.regular_pcl_catalog = {7: pcl}
    context.cross_patch_pcls = [pcl]
    config_before = dict(context.config)

    # Model the participation removal that a successfully prepared candidate
    # would install. Invalid settings must be rejected before reaching it.
    context.prepare_input_changes = Mock(return_value=object())

    def install(candidate):
        context.regular_pcl_catalog = {}
        context.cross_patch_pcls = []

    context.install_input_changes = Mock(side_effect=install)
    with pytest.raises(ValueError, match=error):
        context.apply_config({
            'input_use_pcl_same_winding': False, **settings,
        }, current_iteration=0)

    assert context.config == config_before
    assert context.regular_pcl_catalog == {7: pcl}
    assert context.cross_patch_pcls == [pcl]
    assert context.prepared_main_tracks == {}
    context.prepare_input_changes.assert_not_called()
    context.install_input_changes.assert_not_called()
    context.apply_config({}, current_iteration=0)
    assert context.cross_patch_pcls == [pcl]


# --- tracks ---------------------------------------------------------------------

def test_track_crossing_settings_reprepare_the_retained_tracks(monkeypatch):
    import fit_spiral
    context = _context(input_use_tracks=True)
    context.tracks = ['a track']
    context.trusted_geometry_tree = None
    context.track_families = None
    context.track_source_ids = None
    context.track_crossing_cache = None
    context.track_graph = None
    context.progress = None
    context.device = torch.device('cpu')
    prepared = {'flat_zyx_cpu': torch.zeros((1, 3))}
    prepare = Mock(return_value=prepared)
    monkeypatch.setattr(fit_spiral, 'prepare_main_phase_tracks', prepare)
    monkeypatch.setattr(
        fit_spiral, 'configure_prepared_track_sampling', Mock())

    context.apply_config(
        {'track_crossing_precompute_max': 12},
        current_iteration=0)
    prepare.assert_called_once()
    policy = prepare.call_args.kwargs['sampling_config']
    assert policy['crossing_precompute_max'] == 12
    assert context.prepared_main_tracks is prepared

# --- integration steps ------------------------------------------------------------

def test_integration_step_count_is_set_on_the_resident_transform():
    context = _context()
    context.spiral_and_transform = SimpleNamespace(flow_integration_steps=3)
    context.apply_config(
        {'model_num_flow_integration_steps': 5}, current_iteration=0)
    assert context.spiral_and_transform.flow_integration_steps == 5
    assert context.config['model_num_flow_integration_steps'] == 5


# --- visualisation slices -------------------------------------------------------

def test_visualisation_slice_count_is_read_live():
    import inspect
    import fit_spiral
    source = inspect.getsource(fit_spiral.FitContext._build_model_state)
    assert 'output_num_slices_for_visualization' not in source
    assert 'num_slices_for_visualisation' not in (
        fit_spiral.FitContext._MODEL_STAGE_ATTRIBUTES)
    assert 'output_num_slices_for_visualization' in inspect.getsource(
        fit_spiral.FitContext._prepare_png_visualization_inputs)


def test_fiber_link_side_rules_relink_every_fiber_at_a_run_boundary():
    context = _context(pcl_fiber_link_model_direction_step=10)
    context.fiber_catalog = {'f': _fiber(1, 'f', [[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]])}
    context._relink_fibers_to_patches = Mock()

    # Below the direction-switch step the relink uses the umbilicus direction.
    context.apply_config({'pcl_fiber_link_side_filter': True}, current_iteration=4)
    context._relink_fibers_to_patches.assert_called_once_with(
        'umbilicus', iteration=4)
    assert context.config['pcl_fiber_link_side_filter'] is True

    # From that step on it uses the fitted winding; the margin alone relinks too.
    context._relink_fibers_to_patches.reset_mock()
    context.apply_config(
        {'pcl_fiber_link_side_margin_voxels': 1.5}, current_iteration=12)
    context._relink_fibers_to_patches.assert_called_once_with(
        'model', iteration=12)

    # Turning the filter off relinks under no rules (plain attachments).
    context._relink_fibers_to_patches.reset_mock()
    context.apply_config({'pcl_fiber_link_side_filter': False}, current_iteration=12)
    context._relink_fibers_to_patches.assert_called_once_with(
        'model', iteration=12)


def test_fiber_link_side_rules_are_a_no_op_without_a_fiber_catalog():
    context = _context()
    context._relink_fibers_to_patches = Mock()
    context.apply_config({'pcl_fiber_link_side_filter': True}, current_iteration=0)
    context._relink_fibers_to_patches.assert_not_called()
    assert context.config['pcl_fiber_link_side_filter'] is True


def _z_plane_patch(z, y0=-20.0, x0=-20.0, size=5, spacing=10.0):
    from tifxyz import Patch
    grid = torch.zeros((size, size, 3), dtype=torch.float32)
    for i in range(size):
        for j in range(size):
            grid[i, j] = torch.tensor([z, y0 + i * spacing, x0 + j * spacing])
    return Patch(grid, torch.ones(2), None, None)


def _linkable_regular(cid, zyxs, absolute=False):
    return {
        'id': cid, 'name': f'regular{cid}', 'source_file': '/inputs/r.json',
        'sampling_group': '/inputs/r.json',
        'metadata': {'winding_is_absolute': absolute, 'input_role': 'legacy',
                     'resident_collection_id': cid},
        'points': {
            i: {'id': i, 'collectionId': cid, 'p': [z[2], z[1], z[0]],
                'zyx': np.asarray(z, dtype=np.float32),
                'winding_annotation': float(i + 1) if absolute else float('nan')}
            for i, z in enumerate(zyxs)
        },
    }


def test_link_tolerance_relinks_every_collection_at_a_run_boundary():
    if True:
        # Patches at z = 50 and z = 60; points at z = 51.5 are 1.5 from the
        # first, so a tolerance of 1.0 leaves them unattached and 2.0 links
        # them. The fiber at z = 58.5 likewise sits 1.5 from the second.
        context = _context(pcl_link_distance_tolerance=1.0)
        context.verified_patches = {'a': _z_plane_patch(50.0), 'b': _z_plane_patch(60.0)}
        regular = _linkable_regular(3, [[51.5, 0.0, -10.0], [51.5, 0.0, 0.0], [51.5, 0.0, 10.0]])
        context.regular_pcl_catalog = {3: regular}
        fiber = _fiber(9, 'f', [[58.5, 0.0, -10.0], [58.5, 0.0, 0.0], [58.5, 0.0, 10.0]])
        context.fiber_catalog = {'f': fiber}
        context.link_distance_tolerance = 1.0
        # Stale views standing in for what load derived under tolerance 1.0.
        context.unattached_pcl_strips.append({
            'id': 3, 'logical_input_kind': None, 'zyxs': np.zeros((2, 3), np.float32),
            'windings': np.zeros(2, np.float32), 'radial_offsets': np.zeros(2, np.float32)})
        context.unattached_strip_sampling_groups.append('/inputs/r.json')

        context.apply_config({'pcl_link_distance_tolerance': 2.0}, current_iteration=7)

        assert context.link_distance_tolerance == 2.0
        # The catalogs carry the new attachments...
        assert [p['on_patch']['id'] for p in regular['points'].values()] == ['a'] * 3
        assert [p['on_patch']['id'] for p in fiber['points'].values()] == ['b'] * 3
        # ...and the views were re-derived from them: both collections are now
        # cross-patch (all points attached), the stale strip is gone.
        assert sorted(pcl['id'] for pcl in context.cross_patch_pcls) == [3, 9]
        regular_view = next(pcl for pcl in context.cross_patch_pcls if pcl['id'] == 3)
        assert regular_view is not regular  # the catalog stays pristine
        assert list(regular_view['points_by_patch']) == ['a']
        assert len(regular_view['points_by_patch']['a']) == 3
        assert list(context.unattached_pcl_strips) == []
        context._rebuild_pcl_sampling_strata.assert_called_once_with()
        context._build_theta_crossing_map.assert_called_once_with()

        # Tightening it again detaches everything and turns both into strips.
        context._rebuild_pcl_sampling_strata.reset_mock()
        context.apply_config({'pcl_link_distance_tolerance': 1.0}, current_iteration=8)
        assert all('on_patch' not in p for p in regular['points'].values())
        assert all('on_patch' not in p for p in fiber['points'].values())
        assert context.cross_patch_pcls == []
        assert sorted(strip['id'] for strip in context.unattached_pcl_strips) == [3, 9]


def test_window_min_points_must_fit_the_window_at_a_run_boundary():
    context = _context()
    context.regular_pcl_catalog = {3: _linkable_regular(3, [[51.5, 0.0, 0.0]])}
    context._relink_all_points_to_patches = Mock()
    with pytest.raises(ValueError, match='pcl_link_window_min_points'):
        context.apply_config(
            {'pcl_link_window_points': 3, 'pcl_link_window_min_points': 4},
            current_iteration=0)
    context._relink_all_points_to_patches.assert_not_called()
    # Nothing was applied.
    assert context.config['pcl_link_window_points'] == 1
    assert context.config['pcl_link_window_min_points'] == 1
    context.apply_config(
        {'pcl_link_window_points': 3, 'pcl_link_window_min_points': 2},
        current_iteration=5)
    context._relink_all_points_to_patches.assert_called_once_with(iteration=5)


def test_link_settings_are_a_no_op_without_catalogs():
    context = _context()
    context._relink_all_points_to_patches = Mock()
    context.apply_config({'pcl_link_distance_tolerance': 3.0}, current_iteration=0)
    context._relink_all_points_to_patches.assert_not_called()
    assert context.config['pcl_link_distance_tolerance'] == 3.0
