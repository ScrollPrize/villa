from typing import Any, MutableMapping

import numpy as np


def setdefault_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Populate default config values for the row/col conditioning dataset."""
    config.setdefault("use_sdt", False)
    config.setdefault("dilation_radius", 1)  # voxels
    config.setdefault("cond_percent", [0.1, 0.5])
    config.setdefault("use_dense_displacement", True)
    config.setdefault("use_velocity_targets", False)
    config.setdefault("use_trace_ode_targets", False)
    config.setdefault("defer_dense_targets_to_trainer", True)
    config.setdefault("lambda_velocity_dir", 0.1)
    config.setdefault("velocity_target_mode", "away_from_conditioning")
    config.setdefault("velocity_target_region", "full")
    config.setdefault("trace_target_mode", "away_from_conditioning")
    config.setdefault("trace_target_region", "full")
    config.setdefault("trace_target_dilation_radius", 1.0)
    config.setdefault("surface_attract_target_mode", "dense_edt")
    config.setdefault("trace_surface_attract_radius", 0.0)
    config.setdefault("use_neighbor_sheet_context", False)
    config.setdefault("neighbor_sheet_required", False)
    config.setdefault("use_trace_validity_targets", False)
    config.setdefault("lambda_trace_validity", 0.0)
    config.setdefault("trace_validity_positive_radius", 2.0)
    config.setdefault("trace_validity_negative_radius", 3.0)
    config.setdefault("trace_validity_margin", 3.0)
    config.setdefault("trace_validity_background_weight", 0.25)
    config.setdefault("trace_validity_pos_weight", 1.0)
    config.setdefault("defer_trace_dilation_to_trainer", False)
    config.setdefault("defer_trace_validity_to_trainer", True)
    config.setdefault("supervise_conditioning", False)
    config.setdefault("cond_supervision_weight", 0.1)
    config.setdefault("use_growth_direction_channels", False)
    config.setdefault("force_recompute_patches", False)
    config.setdefault("use_heatmap_targets", False)
    config.setdefault("heatmap_step_size", 10)
    config.setdefault("heatmap_step_count", 5)
    config.setdefault("heatmap_sigma", 2.0)
    config.setdefault("use_segmentation", False)
    config.setdefault("sample_mode", "wrap")
    config.setdefault("val_num_workers", 0)
    config.setdefault("persistent_workers", False)

    config.setdefault("use_other_wrap_cond", False)
    config.setdefault("use_triplet_wrap_displacement", False)

    config.setdefault("validate_result_tensors", False)
    
    # Patch-finding defaults.
    config.setdefault("overlap_fraction", 0.0)
    config.setdefault("min_span_ratio", 1.0)
    config.setdefault("edge_touch_frac", 0.1)
    config.setdefault("edge_touch_min_count", 10)
    config.setdefault("edge_touch_pad", 0)
    config.setdefault("min_points_per_wrap", 100)
    config.setdefault("scale_normalize_patch_counts", True)
    config.setdefault("patch_count_reference_scale", 0)
    config.setdefault("bbox_pad_2d", 0)
    config.setdefault("require_all_valid_in_bbox", True)
    config.setdefault("skip_chunk_if_any_invalid", False)
    config.setdefault("inner_bbox_fraction", 0.7)

    # conditioning perturbation defaults
    cond_local_perturb = dict(config.get("cond_local_perturb") or {})
    cond_local_perturb.setdefault("enabled", True)
    cond_local_perturb.setdefault("probability", 0.35)
    cond_local_perturb.setdefault("num_blobs", [1, 3])
    cond_local_perturb.setdefault("points_affected", 10)
    cond_local_perturb.setdefault("sigma_fraction_range", [0.04, 0.10])
    cond_local_perturb.setdefault("amplitude_range", [0.25, 1.25])
    cond_local_perturb.setdefault("radius_sigma_mult", 2.5)
    cond_local_perturb.setdefault("max_total_displacement", 6.0)
    config["cond_local_perturb"] = cond_local_perturb

    config.setdefault("displacement_supervision", "vector")


def _require_choice(name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        options = "', '".join(sorted(allowed))
        raise ValueError(f"{name} must be '{options}', got {value!r}")


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _require_finite_range(
    name: str,
    value: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> None:
    _require_finite(name, value)
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must satisfy {min_value} <= value, got {value!r}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must satisfy value <= {max_value}, got {value!r}")


def validate_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Validate row/col conditioning dataset config invariants."""
    displacement_supervision = str(config.get("displacement_supervision", "vector")).lower()
    _require_choice("displacement_supervision", displacement_supervision, {"vector", "normal_scalar"})

    sample_mode = str(config.get("sample_mode", "wrap")).lower()
    _require_choice("sample_mode", sample_mode, {"wrap", "chunk"})

    velocity_target_mode = str(config.get("velocity_target_mode", "away_from_conditioning")).lower()
    _require_choice("velocity_target_mode", velocity_target_mode, {"away_from_conditioning"})

    velocity_target_region = str(config.get("velocity_target_region", "full")).lower()
    _require_choice("velocity_target_region", velocity_target_region, {"full", "conditioning", "hidden"})

    trace_target_mode = str(config.get("trace_target_mode", "away_from_conditioning")).lower()
    _require_choice("trace_target_mode", trace_target_mode, {"away_from_conditioning"})

    trace_target_region = str(config.get("trace_target_region", "full")).lower()
    _require_choice("trace_target_region", trace_target_region, {"full", "conditioning", "hidden"})

    trace_target_dilation_radius = float(config.get("trace_target_dilation_radius", 1.0))
    _require_finite_range("trace_target_dilation_radius", trace_target_dilation_radius, min_value=0.0)

    surface_attract_target_mode = str(config.get("surface_attract_target_mode", "dense_edt")).lower()
    _require_choice("surface_attract_target_mode", surface_attract_target_mode, {"dense_edt", "trace_band"})

    trace_surface_attract_radius = float(config.get("trace_surface_attract_radius", 0.0))
    _require_finite_range("trace_surface_attract_radius", trace_surface_attract_radius, min_value=0.0)

    trace_validity_positive_radius = float(config.get("trace_validity_positive_radius", 2.0))
    _require_finite_range("trace_validity_positive_radius", trace_validity_positive_radius, min_value=0.0)

    trace_validity_negative_radius = float(config.get("trace_validity_negative_radius", 3.0))
    _require_finite_range("trace_validity_negative_radius", trace_validity_negative_radius, min_value=0.0)

    trace_validity_margin = float(config.get("trace_validity_margin", 3.0))
    _require_finite_range("trace_validity_margin", trace_validity_margin, min_value=0.0)

    trace_validity_background_weight = float(config.get("trace_validity_background_weight", 0.25))
    _require_finite_range("trace_validity_background_weight", trace_validity_background_weight, min_value=0.0)

    trace_validity_pos_weight = float(config.get("trace_validity_pos_weight", 1.0))
    _require_finite_range("trace_validity_pos_weight", trace_validity_pos_weight, min_value=0.0)

    if not bool(config.get("defer_dense_targets_to_trainer", True)):
        raise ValueError(
            "defer_dense_targets_to_trainer=False is no longer supported; "
            "dense EDT targets are built in the trainer."
        )
    if (
        bool(config.get("use_trace_validity_targets", False))
        or float(config.get("lambda_trace_validity", 0.0)) > 0.0
    ) and not bool(config.get("defer_trace_validity_to_trainer", True)):
        raise ValueError(
            "defer_trace_validity_to_trainer=False is no longer supported; "
            "trace-validity EDT targets are built in the trainer."
        )

    use_dense_displacement = bool(config.get("use_dense_displacement", False))
    use_triplet_wrap_displacement = bool(config.get("use_triplet_wrap_displacement", False))
    use_velocity_targets = bool(config.get("use_velocity_targets", False))
    use_trace_ode_targets = bool(config.get("use_trace_ode_targets", False))
    use_growth_direction_channels = bool(config.get("use_growth_direction_channels", False))

    if not use_triplet_wrap_displacement and not use_dense_displacement:
        raise ValueError(
            "Regular split no longer supports sparse supervision; "
            "set use_dense_displacement=True."
        )

    if displacement_supervision == "normal_scalar" and use_dense_displacement:
        raise ValueError("displacement_supervision='normal_scalar' is not supported with use_dense_displacement=True")

    unsupported_flags = {
        "use_triplet_wrap_displacement": use_triplet_wrap_displacement,
        "use_other_wrap_cond": bool(config.get("use_other_wrap_cond", False)),
        "use_sdt": bool(config.get("use_sdt", False)),
        "use_heatmap_targets": bool(config.get("use_heatmap_targets", False)),
        "use_segmentation": bool(config.get("use_segmentation", False)),
    }
    enabled_unsupported = [name for name, enabled in unsupported_flags.items() if enabled]
    if enabled_unsupported:
        raise ValueError(
            "rowcol_cond has been consolidated to the active dense trace-ODE split path; "
            f"unsupported options enabled: {enabled_unsupported}"
        )
    if sample_mode != "wrap":
        raise ValueError("rowcol_cond requires sample_mode='wrap'")
