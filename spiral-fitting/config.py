"""Validated Spiral configuration with built-in defaults."""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_GAP_EXPANDER_CAPACITY = 144


_ENUMS = {
    "model_flow_integration_solver": ["rk4"],
    "model_flow_field_type": ["cartesian", "cylindrical"],
    "track_crossing_mode": ["count", "track_walk"],
    "track_radius_target": ["mean", "median"],
    "dense_spacing_mode": ["phase", "grad_mag", "winding_model"],
    "dense_spacing_support_policy": ["product", "minimum"],
    "dt_target_mode": ["strip_median", "whole_object_quantile"],
    "dense_spacing_density_lambda": [
        "inverse_gap", "soft_mass", "soft_mass_wide"],
    "optimizer_flow_sobolev_curvature": ["none", "finite_difference", "gauss_newton"],
    "loss_penalty_shape": ["abs", "square"],
    "optimizer_flow_sobolev_preconditioner": ["cg", "gaussian"],
}

_NULL_TYPES = {
    "pcl_sampling_weights": "dictionary",
    "track_length_bin_weights": "vector",
    "track_max_tortuosity": "number",
    "loss_start_track_dt": "integer",
    "loss_start_unverified_patch_dt": "number",
    "patch_uuid_filter_regex": "string",
}

_PREPARED_INPUT_FIELDS = {
    "patch_erode_patches",
    "patch_uuid_filter_regex",
    "track_crossing_precompute_max",
    "track_crossing_mode",
    "track_exclusion_radius",
    "dense_spacing_mode",
    "loss_weight_fiber_directions",
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
    "sample_count_fiber_direction_points",
    "sample_count_regularisation_points",
    "sample_count_dense_spacing_pairs",
    "sample_count_dense_spacing_density_extra_pairs",
    "sample_count_winding_model_relative_pairs",
    "sample_count_winding_model_density_pairs",
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

_INPUT_TOGGLE_DESCRIPTIONS = {
    "input_use_verified_patches":
        "Load verified patches and allow their radius/DT supervision.",
    "input_use_unverified_patches":
        "Load unverified patches and allow their radius/DT supervision.",
    "input_use_tracks":
        "Load tracks and allow track sampling and losses.",
    "input_use_fibers":
        "Load fiber annotations into the point-collection supervision pools.",
    "input_use_fiber_directions":
        "Load packed fiber-direction samples and allow their orientation loss.",
    "input_use_pcl_absolute":
        "Load absolute-winding point-collection inputs.",
    "input_use_pcl_relative":
        "Load relative-winding point-collection inputs.",
    "input_use_pcl_same_winding":
        "Load same-winding point-collection inputs.",
    "input_use_pcl_drawn_control_points":
        "Load drawn-control-point point-collection inputs.",
    "input_use_normals":
        "Allow dense normal stores, sampling, and normal-dependent losses.",
    "input_use_surf_sdt":
        "Allow the surface-SDT store and SDT-dependent phase losses.",
    "input_use_gradient_magnitude":
        "Allow gradient-magnitude dense-spacing supervision.",
    "input_use_winding_inference":
        "Allow compact winding-inference supervision.",
    "input_use_outer_shell":
        "Allow outer-shell losses, lookup maps, and shell-based track filtering.",
}

# These fields were added after durable checkpoints already existed. Missing
# values are unambiguous: historical fits used every available input, so a
# missing toggle means True. Checkpoint readers use this mapping instead of
# weakening strict schema checks for unrelated future fields.
BACKFILLABLE_CONFIG_DEFAULTS = {
    key: True for key in _INPUT_TOGGLE_DESCRIPTIONS
}
BACKFILLABLE_CONFIG_DEFAULTS.update({
    # Historical checkpoints used an unbounded exponential gap map and used
    # model_gap_expander_num_windings for both the physical estimate and the
    # allocated lattice extent.  The checkpoint loader migrates their tensors;
    # these defaults make the added semantic fields schema-compatible too.
    "model_gap_expander_capacity_windings": DEFAULT_GAP_EXPANDER_CAPACITY,
    "model_gap_expander_min_gap": 1.0,
    "model_gap_expander_softplus_bias": 4.0,
})
# The flow-gradient conditioning settings postdate durable checkpoints;
# missing means off, which is exactly the earlier behaviour.
BACKFILLABLE_CONFIG_DEFAULTS.update({
    "optimizer_flow_grad_smoothing": False,
    "optimizer_flow_grad_smoothing_sigma_voxels": 32.0,
    "optimizer_flow_lazy_moments": False,
    "optimizer_flow_grad_smoothing_across_sigma_voxels": 0.0,
    "optimizer_flow_grad_smoothing_low_res_sigma_voxels": 0.0,
    "model_flow_field_low_res_lr_scale": 1.0,
    "optimizer_flow_shared_second_moment": False,
    "optimizer_flow_shared_second_moment_clip_quantile": 0.99,
    "optimizer_flow_grad_clip_median_multiple": 0.0,
})
# The penalty shape postdates durable checkpoints; missing means the
# historical abs/hinge penalties.
BACKFILLABLE_CONFIG_DEFAULTS.update({
    "loss_penalty_shape": "abs",
    "loss_square_scale_voxels": 16.0,
})
# The Sobolev-damped Hessian-free flow step (sobolev_gauss_newton.py) also
# postdates durable checkpoints; missing means the AdamW flow update.
BACKFILLABLE_CONFIG_DEFAULTS.update({
    "optimizer_flow_sobolev_gn": False,
    "optimizer_flow_sobolev_length_voxels": 32.0,
    "optimizer_flow_sobolev_damping": 1.0,
    "optimizer_flow_sobolev_curvature": "none",
    "optimizer_flow_sobolev_pcg_iterations": 5,
    "optimizer_flow_sobolev_pcg_tolerance": 0.1,
    "optimizer_flow_sobolev_inner_cg_iterations": 20,
    "optimizer_flow_sobolev_inner_cg_tolerance": 0.001,
    "optimizer_flow_sobolev_step_scale": 0.1,
    "optimizer_flow_sobolev_trust_radius": 0.0,
    "optimizer_flow_sobolev_max_step_voxels": 2.0,
    "optimizer_flow_sobolev_fd_epsilon_voxels": 1.0,
    "optimizer_flow_sobolev_diagnostic_interval": 0,
    "optimizer_flow_sobolev_irls_floor": 1.0,
    "optimizer_flow_sobolev_residual_growth_limit": 10.0,
    "optimizer_flow_sobolev_rho_reject": True,
    "optimizer_flow_sobolev_rho_poor": 0.25,
    "optimizer_flow_sobolev_rho_good": 0.75,
    "optimizer_flow_sobolev_adapt_damping": True,
    "optimizer_flow_sobolev_damping_increase": 3.0,
    "optimizer_flow_sobolev_damping_decrease": 1.5,
    "optimizer_flow_sobolev_damping_max_factor": 10000.0,
    "optimizer_flow_sobolev_preconditioner": "cg",
    "optimizer_flow_sobolev_evaluate_step": False,
})

_GAP_EXPANDER_DESCRIPTIONS = {
    "model_gap_expander_num_windings": (
        "Legacy/fallback physical winding-count estimate used by exporters; "
        "it does not allocate the gap lattice."),
    "model_gap_expander_capacity_windings": (
        "Allocated gap-lattice capacity, not a claim about the physical "
        "winding count. Must be at least shell_outer_winding_idx + 3."),
    "model_gap_expander_min_gap": (
        "Hard numerical inter-winding gap floor in working voxels. The "
        "minimum-spacing loss remains the separate geological preference."),
    "model_gap_expander_softplus_bias": (
        "Bias of the stable lower-bounded softplus gap parameterisation."),
}

_OPTIMIZER_DESCRIPTIONS = {
    "loss_penalty_shape": (
        "Per-residual penalty of the radius, umbilicus, shell, absolute "
        "winding and track radius terms after their hinge margins. 'abs' is "
        "the historical L1/hinge form (outlier-robust, zero residual-space "
        "curvature). 'square' uses magnitude^2 / (2 * loss_square_scale_"
        "voxels): constant Gauss-Newton weights and a smooth minimum, but "
        "outliers gain quadratic influence and family weights tuned for 'abs' "
        "are only a starting point. The shell term drops its Huber form for "
        "'square'. DT and relative-winding terms are unchanged."),
    "loss_square_scale_voxels": (
        "Voxel scale s of the 'square' penalty magnitude^2 / (2 s): keeps the "
        "loss in voxel units and matches the abs penalty at 2 s."),
    # See README.md, "Flow-gradient conditioning", for literature precedents
    # and the limitations of these custom combinations.
    "optimizer_flow_grad_smoothing": (
        "Gaussian-smooth flow gradients before the optimizer step. The loss "
        "is unchanged, but Adam scaling and masks mean the resulting update "
        "need not retain the kernel profile. Cylindrical smoothing runs "
        "along z and around rings, with optional separate across-ring smoothing."),
    "optimizer_flow_grad_smoothing_sigma_voxels": (
        "Standard deviation in scroll-voxel units of the flow frame: along "
        "z and around cylindrical rings, or isotropically for Cartesian "
        "lattices. These lattice directions approximate sheet directions. "
        "Used for both lattices unless the low-resolution override is set. "
        "Divide by model_flow_voxel_resolution for fine-cell units; coarse "
        "cells are six times wider. Very small widths become identity kernels. "
        "Effective widths are logged at startup when smoothing is enabled."),
    "optimizer_flow_grad_smoothing_across_sigma_voxels": (
        "Standard deviation in scroll-voxel units for smoothing across "
        "cylindrical rings at matching angles. This approximates coupling "
        "across windings; it does not identify sheet boundaries. 0 disables "
        "across-ring smoothing. Ignored for Cartesian lattices."),
    "optimizer_flow_grad_smoothing_low_res_sigma_voxels": (
        "Along-sheet smoothing width for the coarse lattice alone; 0 uses "
        "the fine lattice's width in scroll-voxel units. For example, at the "
        "default 16-voxel fine spacing, 32 voxels is 2 fine cells but about "
        "0.33 coarse cells. The across-ring width is not affected."),
    "optimizer_flow_shared_second_moment": (
        "Use one Adam denominator across cells and components of each "
        "lattice slab, separately for each flow stage and coarse/fine lattice. "
        "Preserves relative first-moment magnitudes before lazy masking and "
        "weight decay, rather than normalizing each entry independently. "
        "The denominator uses a winsorised mean of positive stored second "
        "moments. Full per-entry state is retained; the flag can change "
        "between runs."),
    "optimizer_flow_shared_second_moment_clip_quantile": (
        "Quantile of positive second-moment entries, estimated from a "
        "fixed-stride sample, used to cap values before averaging the full "
        "slab for the shared denominator. Limits outliers' effect on the "
        "common scale. 1 disables the cap; an empty positive sample also "
        "leaves values uncapped."),
    "model_flow_field_low_res_lr_scale": (
        "Coarse flow learning-rate multiplier relative to the scheduled "
        "base rate, independent of the fine flow multiplier. Shared second "
        "moments can change update sizes; use the logged increments to assess "
        "the scale. 0 freezes coarse values, including weight decay, but "
        "moments may still update. Read every step by the fitter; currently "
        "classified as a model-rebuild setting by the configuration catalog."),
    "optimizer_flow_grad_clip_median_multiple": (
        "Clip individual gradient components at this multiple of the median "
        "nonzero absolute component value, per lattice slab, estimated from "
        "a fixed-stride sample. Runs after DDP averaging and nonfinite "
        "sanitization, before smoothing and optimizer moments. Limits spike "
        "propagation but can suppress valid corrections and change vector "
        "direction. 0 disables clipping. Logs the bound and clipped fraction."),
    "optimizer_flow_lazy_moments": (
        "Use SparseAdam-style masked updates on dense flow gradients: "
        "entries with zero gradient after conditioning and influence masks "
        "retain their moments and receive no gradient update. Smoothing can "
        "activate entries without direct samples. Preserves history through "
        "quiet steps, including stale momentum, and does not correct first "
        "touch scaling. Configured weight decay still applies everywhere."),
    # Sobolev-damped Hessian-free flow step: see sobolev_gauss_newton.py and
    # SOBOLEV_GAUSS_NEWTON_PLAN.md. A prototype for controlled comparisons;
    # not shown to improve quality or speed.
    "optimizer_flow_sobolev_gn": (
        "Replace the AdamW update of the flow lattices with a damped step "
        "d solving (H + lambda A) d = -g, where A is a Sobolev metric "
        "(identity plus scaled lattice Laplacian) and H an optional "
        "Hessian-vector product. Every other parameter keeps AdamW. Flow "
        "gradient clipping, smoothing, Adam moments and flow weight decay "
        "are bypassed while enabled; influence masks still constrain the "
        "step. Prototype: fixed damping, no line search or acceptance test."),
    "optimizer_flow_sobolev_length_voxels": (
        "Sobolev length scale l in scroll voxels of the flow frame: A = I - "
        "l^2 Laplacian, so updates are smoothed over roughly this distance. "
        "Converted to cells per lattice (coarse cells are wider). Isotropic "
        "for Cartesian lattices; along z and around rings for cylindrical "
        "ones, which have no radial coupling in A."),
    "optimizer_flow_sobolev_damping": (
        "Damping lambda. With curvature 'none' the step is -(1/lambda) "
        "A^-1 g, so lambda sets the step length; with curvature it trades "
        "the Hessian against the Sobolev metric (large lambda approaches the "
        "Sobolev gradient step) and is the lower bound of the adapted value. "
        "Must be positive."),
    "optimizer_flow_sobolev_curvature": (
        "'none' takes the Sobolev gradient step. 'finite_difference' solves "
        "(H + lambda A) d = -g by PCG with Hessian-vector products from a "
        "forward finite difference of the gradient on the same batch, one "
        "extra forward/backward pass per PCG iteration; on this loss the "
        "operator proved indefinite and asymmetric. 'gauss_newton' uses "
        "the positive semidefinite IRLS Gauss-Newton operator J^T W J of "
        "the residual-shaped losses (radius, DT, umbilicus, shell, abs "
        "winding; the rest stay first order), costing one forward and one "
        "forward/backward per PCG iteration. Non-finite values or "
        "non-positive curvature stop PCG; the last valid iterate, or the "
        "Sobolev step, is used."),
    "optimizer_flow_sobolev_pcg_iterations": (
        "Maximum outer PCG iterations (Hessian-vector products) per step "
        "when curvature is enabled."),
    "optimizer_flow_sobolev_pcg_tolerance": (
        "PCG stops when the residual falls to this fraction of its initial "
        "norm."),
    "optimizer_flow_sobolev_inner_cg_iterations": (
        "Plain CG iterations used to apply the approximate A^-1 "
        "(preconditioner 'cg'). Fixed count for a near-linear preconditioner."),
    "optimizer_flow_sobolev_inner_cg_tolerance": (
        "Relative residual at which the inner CG for A^-1 stops early."),
    "optimizer_flow_sobolev_step_scale": (
        "Multiplier applied to the solved step before it is added to the "
        "flow parameters. Conservative by default; the step is not passed "
        "through Adam moments or the flow learning rate."),
    "optimizer_flow_sobolev_trust_radius": (
        "Scale the joint step down when its Sobolev norm sqrt(d^T A d) over "
        "all flow lattices (normalised flow-box units) exceeds this. 0 "
        "disables the trust region."),
    "optimizer_flow_sobolev_max_step_voxels": (
        "Scale the scaled joint step down so no lattice's per-component RMS "
        "increment exceeds this many scroll voxels of flow velocity. The "
        "raw step is -(1/lambda) A^-1 g, whose size follows the gradient "
        "scale (unlike Adam), so this cap keeps an untuned damping from "
        "producing huge updates. 0 disables the cap; the applied scale is "
        "logged."),
    "optimizer_flow_sobolev_diagnostic_interval": (
        "Every this many steps, check the step's quadratic model on the same "
        "batch: log the actual and predicted loss reduction, rho = "
        "actual / predicted, and (with curvature) the relative asymmetry of "
        "the finite-difference Hessian product on two random directions. "
        "Costs up to four extra forward/backward passes at those steps. 0 "
        "disables."),
    "optimizer_flow_sobolev_irls_floor": (
        "Floor on |residual| in the Gauss-Newton IRLS weights (dL/dr)/r, in "
        "the residual's own units (voxels for radius, distance and shell "
        "terms). Bounds the weight of residuals near zero."),
    "optimizer_flow_sobolev_adapt_damping": (
        "Adapt lambda across steps without retries, between the configured "
        "damping and damping times optimizer_flow_sobolev_damping_max_factor. "
        "With optimizer_flow_sobolev_evaluate_step the rule uses rho = "
        "actual / predicted same-batch reduction: multiply by "
        "optimizer_flow_sobolev_damping_increase when rho < rho_poor, divide "
        "by optimizer_flow_sobolev_damping_decrease when rho > rho_good "
        "(every curvature mode; for the Sobolev gradient step lambda is the "
        "step length). Without evaluation, curvature modes adapt on the "
        "solver outcome instead (poor: non-positive curvature, growing "
        "residual, non-finite values; clean: converged or max iterations). "
        "The adapted value is saved in checkpoints and logged."),
    "optimizer_flow_sobolev_residual_growth_limit": (
        "Stop PCG when its residual norm grows by more than this factor in "
        "one iteration and keep the previous iterate. CG's residual is not "
        "monotone, and a Gauss-Newton operator confined to the sampled cells "
        "can legitimately spike it severalfold, so keep this loose; 0 "
        "disables the check."),
    "optimizer_flow_sobolev_rho_reject": (
        "With optimizer_flow_sobolev_evaluate_step, undo the flow step "
        "exactly (parameters restored bitwise, no optimizer state touched) "
        "when the same-batch loss did not decrease or is non-finite; lambda "
        "then increases. Every other parameter still takes its AdamW step."),
    "optimizer_flow_sobolev_rho_poor": (
        "rho below which lambda increases (and the step counts as poor)."),
    "optimizer_flow_sobolev_rho_good": (
        "rho above which lambda decreases."),
    "optimizer_flow_sobolev_damping_increase": (
        "Factor applied to lambda after a poor solve (must exceed 1)."),
    "optimizer_flow_sobolev_damping_decrease": (
        "Factor lambda is divided by after a clean solve (at least 1)."),
    "optimizer_flow_sobolev_damping_max_factor": (
        "Upper bound of the adapted lambda relative to the configured damping. "
        "At the bound the step -(1/lambda) A^-1 g is tiny, so a run whose "
        "solves keep failing stalls rather than diverges; keep it moderate."),
    "optimizer_flow_sobolev_fd_epsilon_voxels": (
        "RMS size, in scroll voxels of flow velocity, of the parameter "
        "perturbation used for the finite-difference Hessian-vector "
        "product."),
    "optimizer_flow_sobolev_preconditioner": (
        "'cg' applies A^-1 by inner conjugate gradients (rigorous). "
        "'gaussian' HEURISTICALLY uses the Gaussian gradient smoother and "
        "its optimizer_flow_grad_smoothing_* widths as an approximate A^-1, "
        "the only prototype path that couples cylindrical rings; it has not "
        "been shown self-adjoint, so PCG with it is not a true PCG."),
    "optimizer_flow_sobolev_evaluate_step": (
        "Re-evaluate the loss on the same batch after the flow step, at the "
        "cost of one extra forward/backward pass, and log the before/after "
        "values. Diagnostic only: the step is never rejected."),
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
    "model_gap_expander_capacity_windings",
    "model_gap_expander_lr_scale",
    "model_gap_expander_min_gap", "model_gap_expander_softplus_bias",
    "model_initial_dr_per_winding", "model_linear_z_resolution",
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
    "model_flow_field_low_res_lr_scale",
    "model_flow_field_high_res_lr_scale_initial",
    "model_flow_field_high_res_lr_scale_final",
    "model_flow_field_high_res_lr_ramp_start_step",
    "model_flow_field_high_res_lr_ramp_steps",
    "model_flow_field_direct_lr",
    "model_gap_expander_logit_resolution",
    "model_gap_expander_num_windings",
    "model_gap_expander_capacity_windings",
    "model_gap_expander_min_gap",
    "model_gap_expander_softplus_bias",
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
            minimum=(
                1 if key in {
                    "output_num_slices_for_visualization",
                    "theta_crossing_map_update_interval",
                    "dt_target_update_interval",
                } else 0),
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
        # These values remain part of the resolved/checkpoint configuration,
        # but interactive clients edit them through the run-level z controls,
        # not as independent advanced-JSON settings.
        spec["ui_owner"] = "run"
    elif key in _INPUT_TOGGLE_DESCRIPTIONS:
        spec["description"] = _INPUT_TOGGLE_DESCRIPTIONS[key]
    elif key in _GAP_EXPANDER_DESCRIPTIONS:
        spec["description"] = _GAP_EXPANDER_DESCRIPTIONS[key]
    elif key in _OPTIMIZER_DESCRIPTIONS:
        spec["description"] = _OPTIMIZER_DESCRIPTIONS[key]
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
        # Flow-lattice gradient conditioning (see _OPTIMIZER_DESCRIPTIONS).
        # All are read live every step, so they apply at a run boundary
        # without a rebuild. Off by default.
        self.optimizer_flow_grad_smoothing = False
        self.optimizer_flow_grad_smoothing_sigma_voxels = 32.0
        self.optimizer_flow_grad_smoothing_across_sigma_voxels = 0.0
        self.optimizer_flow_grad_smoothing_low_res_sigma_voxels = 0.0
        self.optimizer_flow_lazy_moments = False
        self.optimizer_flow_shared_second_moment = False
        self.optimizer_flow_shared_second_moment_clip_quantile = 0.99
        self.optimizer_flow_grad_clip_median_multiple = 0.0
        # Penalty shape of the residual losses (see _OPTIMIZER_DESCRIPTIONS).
        self.loss_penalty_shape = "abs"
        self.loss_square_scale_voxels = 16.0
        # Sobolev-damped Hessian-free flow step (prototype, off by default;
        # see _OPTIMIZER_DESCRIPTIONS and sobolev_gauss_newton.py).
        self.optimizer_flow_sobolev_gn = False
        self.optimizer_flow_sobolev_length_voxels = 32.0
        self.optimizer_flow_sobolev_damping = 1.0
        self.optimizer_flow_sobolev_curvature = "none"
        self.optimizer_flow_sobolev_pcg_iterations = 5
        self.optimizer_flow_sobolev_pcg_tolerance = 0.1
        self.optimizer_flow_sobolev_inner_cg_iterations = 20
        self.optimizer_flow_sobolev_inner_cg_tolerance = 0.001
        self.optimizer_flow_sobolev_step_scale = 0.1
        self.optimizer_flow_sobolev_trust_radius = 0.0
        self.optimizer_flow_sobolev_max_step_voxels = 2.0
        self.optimizer_flow_sobolev_diagnostic_interval = 0
        self.optimizer_flow_sobolev_irls_floor = 1.0
        self.optimizer_flow_sobolev_residual_growth_limit = 10.0
        self.optimizer_flow_sobolev_rho_reject = True
        self.optimizer_flow_sobolev_rho_poor = 0.25
        self.optimizer_flow_sobolev_rho_good = 0.75
        self.optimizer_flow_sobolev_adapt_damping = True
        self.optimizer_flow_sobolev_damping_increase = 3.0
        self.optimizer_flow_sobolev_damping_decrease = 1.5
        self.optimizer_flow_sobolev_damping_max_factor = 10000.0
        self.optimizer_flow_sobolev_fd_epsilon_voxels = 1.0
        self.optimizer_flow_sobolev_preconditioner = "cg"
        self.optimizer_flow_sobolev_evaluate_step = False
        self.model_num_flow_integration_steps = 3
        self.model_flow_integration_solver = "rk4"
        self.model_num_flow_timesteps = 1
        self.model_num_flow_stages = 2
        self.model_flow_bounds_z_margin = 160
        self.model_flow_bounds_radius = 3200
        self.model_flow_voxel_resolution = 16
        self.model_flow_field_type = "cartesian"
        self.model_flow_field_high_res_lr_scale_initial = 0.2
        self.model_flow_field_high_res_lr_scale_final = 0.2
        self.model_flow_field_high_res_lr_ramp_start_step = 0
        self.model_flow_field_high_res_lr_ramp_steps = 1
        # Both flow lattices' LRs are optimizer_learning_rate times their
        # scale; the low-resolution one had no scale of its own before.
        self.model_flow_field_low_res_lr_scale = 1.0
        self.model_flow_field_direct_lr = True
        self.model_gap_expander_logit_resolution = 24
        # The physical winding estimate and the allocated transform capacity
        # are deliberately separate.  shell_outer_winding_idx is the active
        # hypothesis; num_windings remains the legacy/fallback physical
        # estimate used by exporters, while capacity only shapes the lattice.
        self.model_gap_expander_num_windings = 130
        self.model_gap_expander_capacity_windings = \
            DEFAULT_GAP_EXPANDER_CAPACITY
        self.model_gap_expander_lr_scale = 0.3
        self.model_gap_expander_min_gap = 1.0
        self.model_gap_expander_softplus_bias = 4.0
        self.model_linear_z_resolution = 48
        self.model_initial_dr_per_winding = 16.0
        # Patch/PCL theta=0 topology is transformed only on this cadence. Patch
        # samples use cached node potentials; generic PCL/track walks gather
        # cached signed crossings.
        self.theta_crossing_map_update_interval = 100
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
        self.sample_count_fiber_direction_points = 60000
        self.sample_count_regularisation_points = 4500
        self.sample_count_dense_spacing_pairs = 12000
        self.sample_count_dense_spacing_count_extra_pairs = 0
        self.sample_count_dense_spacing_density_extra_pairs = 24000
        self.sample_count_dense_spacing_density_chunk_pairs = 24000
        self.sample_count_winding_model_relative_pairs = 128000
        self.sample_count_winding_model_density_pairs = 128000
        self.sample_count_minimum_spacing_independent_samples = 2000
        self.sample_count_dense_attachment_points = 20000
        self.sample_count_patch_dt_target_points = 256
        self.sample_count_dt_target_points_per_strip = 512
        self.sample_count_shell_samples = 24576
        self.sample_count_influence_footprint_points = 2048
        self.sample_count_influence_anchor_lattice_points = 100000
        self.sample_count_influence_anchor_geometry_points = 100000
        self.sample_count_influence_anchor_samples_per_step = 4096
        # Exponent applied to patch areas when building patch sampling
        # probabilities: 0 = uniform, 1 = proportional to area.
        self.patch_sampling_area_exponent = 0.5
        self.patch_erode_patches = 1
        # Rebuild-scoped supervision-source switches. A false value is a hard
        # participation gate: the source is not loaded, prepared, sampled, or
        # used by losses. Loss weights and sample counts remain unchanged so
        # re-enabling a source restores its previous tuning.
        self.input_use_verified_patches = True
        self.input_use_unverified_patches = True
        self.input_use_tracks = False
        self.input_use_fibers = True
        self.input_use_fiber_directions = False
        self.input_use_pcl_absolute = True
        self.input_use_pcl_relative = True
        self.input_use_pcl_same_winding = True
        self.input_use_pcl_drawn_control_points = True
        self.input_use_normals = True
        self.input_use_surf_sdt = False
        self.input_use_gradient_magnitude = True
        self.input_use_winding_inference = True
        self.input_use_outer_shell = True
        self.input_disable_patches = False
        # When set, only patch directory entries (uuid-named) whose name
        # matches this regex (re.search) are loaded; None loads everything.
        self.patch_uuid_filter_regex = None
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
        # Cross-fiber links ("branches"): same-winding continuations between
        # fibers. When on, linked collections merge into per-component
        # cross-patch pcls with an explicit fiber graph (winding ties propagate
        # through junctions whether or not the junction points attach to
        # patches), and the unattached-strip loss samples chain walks that hop
        # fibers at junctions. Link endpoints are resolved by their explicit
        # control_point indices (mapped through decimation via
        # kept_orig_indices).
        self.pcl_use_fiber_links = True
        # Include unapproved (pending) links.
        self.pcl_use_pending_fiber_links = False
        self.track_min_sample_spacing = 20.0
        self.track_max_sample_spacing = 60.0
        self.track_length_bin_weights = [0.0, 0.15, 0.85]
        self.track_max_tortuosity = None
        self.track_crossing_precompute_max = 8
        self.track_max_track_crossing_per_step = 2
        self.track_crossing_mode = "count"
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
        self.dense_spacing_mode = "winding_model"
        self.winding_model_relative_pair_delta = [3, 15]
        self.winding_model_huber_delta = 0.5
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
        # Probability of hopping onto the linked fiber at each junction while
        # sampling a chain walk through a link component in the
        # unattached-strip loss.
        self.loss_fiber_link_branch_probability = 0.5
        self.loss_weight_track_radius = 50.0
        self.loss_weight_track_dt = 10.0
        self.loss_weight_sym_dirichlet = 10.0
        self.loss_weight_dense_normals = 100.0
        self.loss_weight_fiber_directions = 0.0
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
        self.fiber_directions_finite_difference_epsilon = 8.0
        self.model_sym_dirichlet_finite_difference_epsilon = 4.0
        self.optimizer_weight_decay_gap_expander = 0.01
        self.optimizer_weight_decay_flow_field = 0.0
        self.loss_start_patch_dt = 25000
        self.loss_start_track_dt = 25000
        self.loss_start_unverified_patch_dt = None
        self.dt_progressive_windings = False
        self.dt_progressive_inner_winding = 20
        self.dt_progressive_steps = 50000
        self.dt_progressive_exponent = 1.0
        self.dt_target_mode = "strip_median"
        self.dt_target_floating_threshold = 0.25
        # Backward-compatible alias. FitContext phase-locks whole-object DT
        # targets to theta_crossing_map_update_interval and keeps both values
        # synchronized when either setting is changed.
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
        if values["model_gap_expander_capacity_windings"] < 3:
            raise ValueError(
                "model_gap_expander_capacity_windings must be at least 3")
        if not (0.0 < values["model_gap_expander_min_gap"]
                < values["model_initial_dr_per_winding"]):
            raise ValueError(
                "model_gap_expander_min_gap must be positive and smaller "
                "than model_initial_dr_per_winding")
        for key, value in overrides.items():
            setattr(self, key, value)

    def as_dict(self):
        return vars(self).copy()

    @classmethod
    def catalog(cls):
        resolved_defaults = cls().as_dict()
        run_owned = {"z_begin", "z_end"}
        defaults = {
            key: value for key, value in resolved_defaults.items()
            if key not in run_owned
        }
        fields = {
            key: _field_spec(key, value)
            for key, value in defaults.items()
        }
        run_fields = {
            key: _field_spec(key, resolved_defaults[key])
            for key in sorted(run_owned)
        }
        presets = {
            path.stem: {
                key: value for key, value in cls(path).as_dict().items()
                if key not in run_owned
            }
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
                # API run-block fields shown in the left-side dock. They are
                # catalogued for clients but deliberately absent from the
                # advanced configuration defaults, fields, and presets.
                "run_fields": run_fields,
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
