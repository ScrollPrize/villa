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
    context.fiber_direction_samples = None
    context.dt_target_cache_manager = SimpleNamespace(
        update_interval=100, reset=Mock())
    context.theta_crossing_map = SimpleNamespace(invalidate=Mock())
    context._rebuild_pcl_sampling_strata = Mock()
    context._refresh_trusted_geometry = Mock()
    context._build_theta_crossing_map = Mock(return_value=[])
    context._make_shell_polar_map = Mock(return_value='rebuilt shell map')
    return context


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


# --- frozen object-store inputs ---------------------------------------------------

@pytest.mark.parametrize('key, value', [
    ('pcl_use_fiber_links', False),
    ('pcl_fiber_min_point_spacing', 5.0),
    ('pcl_unattached_pcl_min_point_spacing', 2.5),
    ('pcl_link_distance_tolerance', 3.0),
    ('pcl_link_window_points', 3),
    ('input_use_pcl_same_winding', False),
])
def test_settings_shaping_the_frozen_inputs_are_refused(key, value):
    context = _context()
    config_before = dict(context.config)
    with pytest.raises(ValueError, match='object-store inputs are frozen'):
        context.apply_config({key: value}, current_iteration=0)
    assert context.config == config_before
    context._rebuild_pcl_sampling_strata.assert_not_called()


@pytest.mark.parametrize('exponential', [False, True])
def test_frozen_setting_rejects_lr_changes_without_mutating_optimizer(exponential):
    context = _context(optimizer_exp_lr_schedule=exponential)
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

    with pytest.raises(ValueError, match='object-store inputs are frozen'):
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


# --- tracks ---------------------------------------------------------------------

@pytest.mark.parametrize('settings, error', [
    ({'track_max_track_crossing_per_step': -1}, 'max_track_crossing_per_step'),
    ({'track_crossing_precompute_max': 1.5}, 'track_crossing_precompute_max'),
])
def test_invalid_track_policy_is_rejected_before_any_change(settings, error):
    context = _context(input_use_tracks=True)
    context.tracks = ['resident track']
    context.prepared_main_tracks = {}
    config_before = dict(context.config)
    with pytest.raises(ValueError, match=error):
        context.apply_config(
            {'loss_weight_track_radius': 2.0, **settings}, current_iteration=0)
    assert context.config == config_before
    assert context.prepared_main_tracks == {}


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
