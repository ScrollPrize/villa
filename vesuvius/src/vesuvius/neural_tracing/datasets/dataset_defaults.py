from typing import Any, MutableMapping

import numpy as np


def setdefault_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Populate default config values for the row/col conditioning dataset."""
    config.setdefault("cond_percent", [0.1, 0.5])
    config.setdefault("use_trace_ode_targets", True)
    config.setdefault("lambda_velocity_dir", 0.1)
    config.setdefault("trace_target_dilation_radius", 1.0)
    config.setdefault("surface_attract_target_mode", "trace_band")
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
    config.setdefault("supervise_conditioning", False)
    config.setdefault("cond_supervision_weight", 0.1)
    config.setdefault("use_growth_direction_channels", False)
    config.setdefault("force_recompute_patches", False)
    config.setdefault("sample_mode", "wrap")
    config.setdefault("val_num_workers", 0)
    config.setdefault("persistent_workers", False)

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
    sample_mode = str(config.get("sample_mode", "wrap")).lower()
    _require_choice("sample_mode", sample_mode, {"wrap"})

    trace_target_dilation_radius = float(config.get("trace_target_dilation_radius", 1.0))
    _require_finite_range("trace_target_dilation_radius", trace_target_dilation_radius, min_value=0.0)

    surface_attract_target_mode = str(config.get("surface_attract_target_mode", "trace_band")).lower()
    _require_choice("surface_attract_target_mode", surface_attract_target_mode, {"trace_band"})

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

    use_trace_ode_targets = bool(config.get("use_trace_ode_targets", False))

    if not use_trace_ode_targets:
        raise ValueError("rowcol_cond requires use_trace_ode_targets=True")

    unsupported_flags = {
        "displacement_supervision=normal_scalar": str(config.get("displacement_supervision", "vector")).lower() == "normal_scalar",
        "use_triplet_wrap_displacement": bool(config.get("use_triplet_wrap_displacement", False)),
        "use_other_wrap_cond": bool(config.get("use_other_wrap_cond", False)),
    }
    enabled_unsupported = [name for name, enabled in unsupported_flags.items() if enabled]
    if enabled_unsupported:
        raise ValueError(
            "rowcol_cond has been consolidated to the active dense trace-ODE split path; "
            f"unsupported options enabled: {enabled_unsupported}"
        )
