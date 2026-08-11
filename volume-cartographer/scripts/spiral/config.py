"""Validated Spiral configuration with built-in defaults."""

from __future__ import annotations

import json
from pathlib import Path


_ENUMS = {
    "model_flow_integration_solver": ["rk4"],
    "model_flow_field_type": ["cartesian", "cylindrical"],
    "patch_strip_sampling": ["straight", "dijkstra"],
    "track_crossing_mode": ["count", "track_walk"],
    "track_radius_target": ["mean", "median"],
    "dense_spacing_mode": ["phase", "grad_mag"],
    "dense_spacing_support_policy": ["product", "minimum"],
    "dt_target_mode": ["strip_median", "whole_object_quantile"],
    "dense_spacing_density_lambda": [
        "inverse_gap", "soft_mass", "soft_mass_wide"],
}

_NULL_TYPES = {
    "pcl_sampling_weights": "dictionary",
    "track_length_bin_weights": "vector",
    "track_max_tortuosity": "number",
    "loss_start_track_dt": "integer",
    "loss_start_unverified_patch_dt": "number",
}

_PREPARED_INPUT_FIELDS = {
    "patch_erode_patches",
    "track_crossing_precompute_max",
    "track_crossing_mode",
    "track_exclusion_radius",
    "dense_spacing_mode",
    "output_first_winding",
    "output_winding_margin",
    "output_step_size",
    "output_num_slices_for_visualization",
    # These values define the discretised outer-shell lookup/atlas. Unlike
    # ordinary shell loss parameters they cannot be changed on an existing
    # prepared shell.
    "shell_num_theta_bins",
    "shell_table_smooth_sigma_z",
    "shell_table_smooth_sigma_theta",
    "shell_min_confidence",
}

_SCALE_WITH_Z_FIELDS = {
    "sample_count_patches_per_step",
    "sample_count_patches_per_step_for_dt",
    "sample_count_unverified_patches_per_step",
    "sample_count_unverified_patches_per_step_for_dt",
    "sample_count_relative_winding_pcls",
    "sample_count_absolute_winding_pcls",
    "sample_count_unattached_pcls_per_step",
    "sample_count_tracks_per_step",
    "sample_count_dense_normal_points",
    "sample_count_regularisation_points",
    "sample_count_dense_spacing_pairs",
    "sample_count_dense_spacing_density_extra_pairs",
    "sample_count_minimum_spacing_independent_samples",
    "sample_count_dense_attachment_points",
    "sample_count_shell_samples",
}


# z_begin/z_end are deliberately heavyweight settings; their metadata records
# every effect so nothing treats them as cheap run-boundary knobs:
#   - host input filtering: patches, PCLs, unattached strips, and tracks are
#     loaded/kept only where they intersect [z_begin, z_end);
#   - dense-store coverage: the Lasagna normal/grad-mag and surf-SDT brick
#     pools are materialised for exactly this z window;
#   - count scaling: every scale_with_z sample count is scaled by the number
#     of slices in the range relative to the 9500-slice reference;
#   - rendering/preview: the preview/export z window and the output-directory
#     name derive from the range;
#   - model/checkpoint-domain compatibility: the flow-field parameter shapes
#     cover the range (plus margin), so resuming a checkpoint requires the
#     optimisation range to lie within the checkpoint's stored model z-range.
_Z_RANGE_DESCRIPTIONS = {
    "z_begin": "First z slice (inclusive) of the fit. Affects host input "
               "filtering, dense-store coverage, per-step count scaling, "
               "rendering, and model/checkpoint z-domain compatibility.",
    "z_end": "One past the last z slice of the fit. Affects host input "
             "filtering, dense-store coverage, per-step count scaling, "
             "rendering, and model/checkpoint z-domain compatibility.",
}

# Configuration keys that shape the model's parameter tensors. A checkpoint
# whose stored value for any of them differs describes a different model, and
# is refused rather than reshaped: a domain/structure change is the explicit
# rebuild path's job. Configuration metadata, so it lives here beside the
# rest of it and is readable without importing the fitter.
CHECKPOINT_MODEL_SHAPE_KEYS = (
    "model_num_flow_integration_steps", "model_flow_integration_solver",
    "model_num_flow_timesteps", "model_flow_bounds_z_margin",
    "model_flow_bounds_radius", "model_flow_voxel_resolution",
    "model_flow_field_type", "model_gap_expander_logit_resolution",
    "model_gap_expander_num_windings", "model_linear_z_resolution",
)


# Configuration keys whose every consumer is built by
# FitContext._build_model_state(). A rebuild that changes only these can keep
# the host inputs and the dense stores and re-run the model stage alone; see
# rebuild_stage() below and FitContext.rebuild_model_state().
#
# This is an audited allowlist, not a prefix rule, because two model-shaped
# keys are read during host preparation:
#   - model_flow_bounds_z_margin sizes the host-side ShellPolarMap that
#     load_host_inputs() filters tracks with;
#   - optimizer_random_seed seeds np.random and torch.random at the top of
#     load_host_inputs() and the pool generators below it, so it reaches
#     every RNG-order-sensitive host decision.
# Both are therefore absent, and a key nobody has audited is absent by
# construction — the safe answer.
MODEL_STAGE_KEYS = frozenset({
    "model_num_flow_integration_steps",
    "model_flow_integration_solver",
    "model_num_flow_timesteps",
    "model_num_flow_stages",
    "model_flow_bounds_radius",
    "model_flow_voxel_resolution",
    "model_flow_field_type",
    "model_flow_field_high_res_lr_scale_initial",
    "model_flow_field_high_res_lr_scale_final",
    "model_flow_field_high_res_lr_ramp_start_step",
    "model_flow_field_high_res_lr_ramp_steps",
    "model_flow_field_direct_lr",
    "model_gap_expander_logit_resolution",
    "model_gap_expander_num_windings",
    "model_gap_expander_lr_scale",
    "model_linear_z_resolution",
    "model_initial_dr_per_winding",
    "model_sym_dirichlet_finite_difference_epsilon",
})


def rebuild_stage(changed_keys):
    """The build stage a set of changed configuration keys requires.

    ``"model"`` when every changed key is on MODEL_STAGE_KEYS, ``"all"``
    otherwise. The stages are one ordinal, not a graph: "all" is the whole
    build as it has always run, and "model" is a strict suffix of it.

    Anything unrecognised falls to "all", so this fails safe: a new key gets
    today's behaviour until somebody audits its consumers.
    """
    return "model" if MODEL_STAGE_KEYS.issuperset(changed_keys) else "all"


def _runtime_impact(key):
    if key in _Z_RANGE_DESCRIPTIONS:
        # Changing the z-range invalidates host inputs, dense stores, and the
        # model's flow-field domain: a new fit, never a run-boundary tweak.
        return "new_fit"
    if key.startswith("model_") or key == "optimizer_random_seed":
        return "new_fit"
    if key.startswith(("input_", "pcl_")) or key in _PREPARED_INPUT_FIELDS:
        return "new_fit"
    return "run_boundary"


def _field_spec(key, default):
    nullable = key in _NULL_TYPES
    if key in _ENUMS:
        kind = "enum"
    elif nullable:
        kind = _NULL_TYPES[key]
    elif type(default) is bool:
        kind = "boolean"
    elif type(default) is int:
        kind = "integer"
    elif type(default) is float:
        kind = "number"
    elif isinstance(default, list):
        kind = "vector"
    elif isinstance(default, dict):
        kind = "dictionary"
    else:
        kind = "string"

    spec = {
        "type": kind,
        "nullable": nullable,
        "label": key.split("_", 1)[-1].replace("_", " ").title(),
        "runtime_impact": _runtime_impact(key),
    }
    if kind in ("integer", "number"):
        spec.update(
            minimum=1 if key == "output_num_slices_for_visualization" else 0,
            maximum=(1_000_000 if key == "output_num_slices_for_visualization"
                     else 1_000_000_000),
            step=1 if kind == "integer" else .01,
        )
        if kind == "number":
            spec["precision"] = 6
    elif kind == "enum":
        spec["values"] = _ENUMS[key]
    elif kind == "vector":
        spec["length"] = 3 if key == "track_length_bin_weights" else len(default)
    if key in _SCALE_WITH_Z_FIELDS:
        spec["scale_with_z"] = True
    if key in _Z_RANGE_DESCRIPTIONS:
        spec["description"] = _Z_RANGE_DESCRIPTIONS[key]
    return spec


class Config:
    def __init__(self, overrides=None):
        # The optimisation z window (see _Z_RANGE_DESCRIPTIONS for the full
        # effect list). Defaults match the historical fit_spiral module
        # globals for the production PHercParis4 dataset.
        self.z_begin = 4000
        self.z_end = 17000
        self.optimizer_random_seed = 1
        self.optimizer_distributed_split_batch = True
        self.optimizer_learning_rate = 3e-05
        self.optimizer_exp_lr_schedule = True
        self.optimizer_lr_final_factor = 0.3
        self.optimizer_num_training_steps = 30000
        self.model_num_flow_integration_steps = 3
        self.model_flow_integration_solver = "rk4"
        self.model_num_flow_timesteps = 1
        self.model_num_flow_stages = 1
        self.model_flow_bounds_z_margin = 160
        self.model_flow_bounds_radius = 3200
        self.model_flow_voxel_resolution = 16
        self.model_flow_field_type = "cartesian"
        self.model_flow_field_high_res_lr_scale_initial = 0.2
        self.model_flow_field_high_res_lr_scale_final = 0.2
        self.model_flow_field_high_res_lr_ramp_start_step = 0
        self.model_flow_field_high_res_lr_ramp_steps = 1
        self.model_flow_field_direct_lr = True
        self.model_gap_expander_logit_resolution = 24
        self.model_gap_expander_num_windings = 130
        self.model_gap_expander_lr_scale = 0.3
        self.model_linear_z_resolution = 48
        self.model_initial_dr_per_winding = 16.0
        self.patch_radius_loss_margin = 0.025
        self.patch_radius_loss_inv = False
        self.patch_loss_z_margin = 0
        self.patch_dt_norm_p = 0.5
        self.patch_dt_within_patch_norm_p = 3.0
        self.patch_dt_loss_margin = 0.025
        self.patch_radius_within_norm_p = 3.0
        self.sample_count_patches_per_step = 360
        self.sample_count_patches_per_step_for_dt = 240
        self.sample_count_points_per_patch = 800
        self.sample_count_unverified_patches_per_step = 120
        self.sample_count_unverified_patches_per_step_for_dt = 80
        self.sample_count_unverified_points_per_patch = 800
        self.sample_count_relative_winding_pcls = 48
        self.sample_count_relative_winding_patch_pairs_per_pcl = 4
        self.sample_count_absolute_winding_pcls = 48
        self.sample_count_absolute_winding_points_per_pcl = 4
        self.sample_count_unattached_pcls_per_step = 84
        self.sample_count_unattached_pcl_points_per_step = 32
        self.sample_count_tracks_per_step = 48000
        self.sample_count_track_points_per_step = 96
        self.sample_count_dense_normal_points = 60000
        self.sample_count_regularisation_points = 4500
        self.sample_count_dense_spacing_pairs = 12000
        self.sample_count_dense_spacing_count_extra_pairs = 0
        self.sample_count_dense_spacing_density_extra_pairs = 24000
        self.sample_count_dense_spacing_density_chunk_pairs = 24000
        self.sample_count_minimum_spacing_independent_samples = 2000
        self.sample_count_dense_attachment_points = 20000
        self.sample_count_patch_dt_target_points = 256
        self.sample_count_dt_target_points_per_strip = 512
        self.sample_count_shell_samples = 24576
        self.sample_count_influence_footprint_points = 2048
        self.sample_count_influence_anchor_lattice_points = 100000
        self.sample_count_influence_anchor_geometry_points = 100000
        self.sample_count_influence_anchor_samples_per_step = 4096
        self.patch_strip_sampling = "straight"
        self.patch_erode_patches = 1
        self.input_disable_patches = False
        self.patch_unverified_patch_radius_loss_margin = 0.025
        self.patch_unverified_patch_radius_loss_inv = False
        self.patch_unverified_patch_radius_within_norm_p = 3.0
        self.patch_unverified_patch_dt_norm_p = 0.5
        self.patch_unverified_patch_dt_within_patch_norm_p = 3.0
        self.patch_unverified_patch_dt_loss_margin = 0.025
        self.patch_unverified_patch_exclusion_radius = 64.0
        self.pcl_rel_winding_adjacent_patches_only = True
        self.pcl_stratified_pcl_sampling = True
        self.pcl_sampling_weights = None
        self.pcl_fiber_min_point_spacing = 40.0
        self.pcl_unattached_pcl_min_point_spacing = 16.0
        self.track_min_sample_spacing = 20.0
        self.track_max_sample_spacing = 60.0
        self.track_length_bin_weights = [0.0, 0.15, 0.85]
        self.track_max_tortuosity = None
        self.track_crossing_precompute_max = 8
        self.track_max_track_crossing_per_step = 1
        self.track_crossing_mode = "track_walk"
        self.track_min_walk_steps_per_track = 24
        self.track_max_walk_steps_per_track = 256
        self.track_min_walks_per_track = 2
        self.track_max_walks_per_track = 4
        self.track_walk_minimum_cycle_travel = 20.0
        self.track_exclusion_radius = 16.0
        self.track_radius_target = "mean"
        self.track_radius_loss_margin = 0.025
        self.track_radius_within_norm_p = 6.0
        self.track_dt_within_track_norm_p = 3.0
        self.track_dt_norm_p = 0.5
        self.track_dt_loss_margin = 0.025
        self.dense_grad_mag_encode_scale = 1000.0
        self.dense_grad_mag_factor = 0.25
        self.dense_spacing_integration_steps = 8
        self.dense_spacing_mode = "phase"
        self.dense_spacing_pair_m_short = [
            3,
            7
        ]
        self.dense_spacing_pair_m_long = [
            5,
            15
        ]
        self.dense_spacing_pair_long_fraction = 0.15
        self.dense_spacing_count_temperature_wv = 0.5
        self.dense_spacing_target_step_wv = 1.0
        self.dense_spacing_max_step_wv = 2.0
        self.dense_spacing_max_steps = 1400
        self.dense_spacing_step_oversample = 1.25
        self.dense_spacing_use_support_gate = True
        self.dense_spacing_support_sigma = 4.0
        self.dense_spacing_support_floor_alpha = 0.05
        self.dense_spacing_support_policy = "product"
        self.dense_spacing_phase_huber_delta = 0.5
        self.dense_spacing_phase_extension_windings = 1.0
        self.dense_spacing_phase_min_center_gap_wv = 4.0
        self.dense_spacing_phase_graze_dot = 0.4
        self.dense_spacing_phase_graze_depth_wv = 1.0
        self.dense_spacing_phase_window_windings = 0.75
        self.dense_spacing_phase_end_free_margin_windings = 0.5
        self.dense_spacing_phase_missing_cost = 0.55
        self.dense_spacing_phase_missing_extend_cost = 0.55
        self.dense_spacing_phase_extra_cost = 0.7
        self.dense_spacing_phase_extra_extend_cost = 0.7
        self.dense_spacing_phase_temperature = 0.1
        self.dense_spacing_phase_band_confidence_cost = 0.25
        self.dense_spacing_phase_top2_margin = 0.1
        self.dense_spacing_phase_min_matched_windings = 2
        self.dense_spacing_phase_min_matched_mass = 1.0
        self.loss_weight_min_spacing = 2.0
        self.loss_weight_dense_spacing_count = 0.0
        self.loss_weight_dense_spacing_density = 12.0
        self.loss_weight_dense_attachment = 0.0
        self.loss_weight_patch_radius = 8.0
        self.loss_weight_patch_dt = 4.0
        self.loss_weight_unverified_patch_radius = 2.0
        self.loss_weight_unverified_patch_dt = 1.0
        self.loss_weight_rel_winding = 5.0
        self.loss_weight_abs_winding = 5.0
        self.loss_weight_unattached_pcl_radius = 2.0
        self.loss_weight_unattached_pcl_dt = 4.0
        self.loss_weight_track_radius = 50.0
        self.loss_weight_track_dt = 10.0
        self.loss_weight_sym_dirichlet = 10.0
        self.loss_weight_dense_normals = 100.0
        self.loss_weight_dense_spacing = 12.0
        self.loss_weight_umbilicus = 1.25
        self.loss_weight_shell_outer = 1.0
        self.loss_weight_shell_patch_radius = 0.0
        self.loss_weight_anchor = 0.0
        self.dense_spacing_density_min_gap_wv = 0.0
        self.dense_spacing_density_max_blind_fraction = 0.75
        self.dense_min_spacing_d_min_wv = 6.0
        self.dense_attachment_scale = 8.0
        self.dense_attachment_warmup_steps = 3000
        self.dense_attachment_ramp_steps = 3000
        self.dense_normals_finite_difference_epsilon = 8.0
        self.model_sym_dirichlet_finite_difference_epsilon = 4.0
        self.optimizer_weight_decay_gap_expander = 0.01
        self.optimizer_weight_decay_flow_field = 0.0
        self.loss_start_patch_dt = 25000
        self.loss_start_track_dt = 10000
        self.loss_start_unverified_patch_dt = None
        self.dt_progressive_windings = False
        self.dt_progressive_inner_winding = 20
        self.dt_progressive_steps = 50000
        self.dt_progressive_exponent = 1.0
        self.dt_target_mode = "strip_median"
        self.dt_target_floating_threshold = 0.25
        self.dt_target_update_interval = 100
        self.dt_target_max_stride = 128
        self.output_first_winding = 10
        self.output_winding_margin = 4
        self.output_step_size = 20
        self.shell_outer_winding_idx = 130
        self.shell_outer_winding_margin = 10
        self.shell_num_theta_bins = 720
        self.shell_huber_delta = 16.0
        self.shell_table_smooth_sigma_z = 4.0
        self.shell_table_smooth_sigma_theta = 1.0
        self.shell_min_confidence = 0.25
        self.output_save_png_visualizations = False
        self.influence_enabled = False
        self.influence_z = 3000.0
        self.influence_windings = 5.0
        self.influence_theta_frac = 0.5
        self.influence_disable_dt_frac = 0.75
        self.influence_sigma = 0.3333
        self.influence_anchor_ramp_power = 2.0
        self.dense_spacing_density_lambda = "inverse_gap"
        self.dense_spacing_density_soft_mass_min_gap_wv = 0.0
        self.output_num_slices_for_visualization = 20

        defaults = vars(self)
        fields = {key: _field_spec(key, value)
                  for key, value in defaults.items()}

        if isinstance(overrides, (str, Path)):
            overrides = json.loads(Path(overrides).read_text())
        overrides = overrides or {}
        unknown = set(overrides) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown Spiral config keys: {sorted(unknown)}")
        values = defaults | overrides
        for key, value in values.items():
            spec = fields[key]
            if value is None and spec["nullable"]:
                continue
            valid = {
                "boolean": lambda: type(value) is bool,
                "integer": lambda: type(value) is int,
                "number": lambda: type(value) in (int, float),
                "string": lambda: isinstance(value, str),
                "enum": lambda: value in spec["values"],
                "vector": lambda: isinstance(value, list),
                "dictionary": lambda: isinstance(value, dict),
            }
            if not valid[spec["type"]]():
                raise ValueError(f"Invalid value for {key}")
            if spec["type"] in ("integer", "number") and not (
                    spec["minimum"] <= value <= spec["maximum"]):
                raise ValueError(f"Out-of-range value for {key}")
            if spec["type"] == "vector" and len(value) != spec["length"]:
                raise ValueError(f"Invalid vector length for {key}")
            if spec["type"] == "vector" and any(
                    type(item) not in (int, float) for item in value):
                raise ValueError(f"Invalid vector value for {key}")
            if spec["type"] == "dictionary" and any(
                    not isinstance(item_key, str)
                    or type(item) not in (int, float)
                    for item_key, item in value.items()):
                raise ValueError(f"Invalid dictionary value for {key}")
        for key, value in overrides.items():
            setattr(self, key, value)

    def as_dict(self):
        return vars(self).copy()

    @classmethod
    def catalog(cls):
        defaults = cls().as_dict()
        fields = {
            key: _field_spec(key, value)
            for key, value in defaults.items()
        }
        presets = {
            path.stem: cls(path).as_dict()
            for path in (Path(__file__).parent / "configs").glob("*.json")
        }
        return {
            "defaults": defaults,
            "schema": {
                # No input path can be taken by a resident session: every path
                # change implies a rebuild, which is the client's default for
                # a path it finds no entry for.
                "paths": {},
                # The keys a rebuild can apply without reloading the session's
                # inputs, advertised so a client can say in advance which kind
                # of rebuild its pending changes would cause. Authoritative
                # answers still come from the service (see rebuild_stage).
                "model_stage_keys": sorted(MODEL_STAGE_KEYS),
                "fields": fields,
            },
            "presets": presets,
        }


def durable_config(values):
    """The checkpoint-durable subset of a configuration.

    Interactive influence state is session-scoped and the anchor weight only
    exists while an influence window is active, so neither is stored in (or
    expected from) a checkpoint's cfg/requested_config/resolved_config.
    Checkpoint compatibility checks must compare stored key sets against
    this durable subset of the schema, not the raw schema.
    """
    return {
        key: item for key, item in dict(values).items()
        if not key.startswith("interactive_influence_")
        and key != "loss_weight_anchor"
    }


class FitConfig:
    """The one explicit fitter configuration: a resolved key -> value mapping.

    A thin dict-style wrapper handed to FitContext, replacing the module
    global `wandb.config` object the fitter used to read. Values must
    already be fully resolved (Config defaults + overrides + any z-range
    scaling); FitConfig performs no resolution of its own because the
    resolution policies legitimately differ per entry point (the CLI
    scales-and-splits for DDP, the interactive runtime round-trips
    checkpoint counts, the golden driver scales without splitting).

    Construction copies the mapping. update() mutates in place, so every
    holder of the same FitConfig (the context, its losses call sites, a
    driver that recorded it) observes run-boundary configuration changes,
    matching the former shared-wandb.config semantics.
    """

    def __init__(self, values):
        self._values = dict(values)

    def __getitem__(self, key):
        return self._values[key]

    def __contains__(self, key):
        return key in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def get(self, key, default=None):
        return self._values.get(key, default)

    def keys(self):
        return self._values.keys()

    def items(self):
        return self._values.items()

    def update(self, values):
        self._values.update(values)

    def __repr__(self):
        return f"FitConfig({self._values!r})"
