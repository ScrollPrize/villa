from typing import Any, MutableMapping

import numpy as np


def setdefault_rowcol_cond_dataset_config(config: MutableMapping[str, Any]) -> None:
    """Populate default config values for the row/col conditioning dataset."""
    config.setdefault("use_sdt", False)
    config.setdefault("dilation_radius", 1)  # voxels
    config.setdefault("cond_percent", [0.5, 0.5])
    config.setdefault("use_extrapolation", False)
    config.setdefault("use_dense_displacement", True)
    config.setdefault("extrapolation_method", "linear_edge")
    config.setdefault("supervise_conditioning", False)
    config.setdefault("cond_supervision_weight", 0.1)
    config.setdefault("force_recompute_patches", False)
    config.setdefault("use_heatmap_targets", False)
    config.setdefault("heatmap_step_size", 10)
    config.setdefault("heatmap_step_count", 5)
    config.setdefault("heatmap_sigma", 2.0)
    config.setdefault("use_segmentation", False)
    config.setdefault("sample_mode", "wrap")

    # Other-wrap conditioning defaults.
    config.setdefault("use_other_wrap_cond", False)
    config.setdefault("other_wrap_prob", 0.5)

    # Triplet-wrap displacement defaults.
    config.setdefault("use_triplet_wrap_displacement", False)
    config.setdefault("triplet_dense_weight_mode", "band")
    config.setdefault("triplet_band_padding_voxels", 4.0)
    config.setdefault("triplet_edt_bbox_padding_voxels", 4.0)
    config.setdefault("triplet_band_distance_percentile", 95.0)
    config.setdefault("triplet_gt_vector_dilation_radius", 0.0)
    config.setdefault("use_triplet_direction_priors", True)
    config.setdefault("triplet_direction_prior_mask", "cond")
    config.setdefault("triplet_overlap_mask_filename", "overlap_mask.tif")
    config.setdefault("triplet_warn_missing_overlap_masks", False)
    config.setdefault("triplet_close_check_enabled", True)
    config.setdefault("triplet_close_distance_voxels", 1.0)
    config.setdefault("triplet_close_fraction_threshold", 0.05)
    config.setdefault("triplet_close_print", True)

    config.setdefault("enable_volume_crop_cache", False)
    config.setdefault("volume_crop_cache_max_items", 0)
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
    config.setdefault("min_cond_span", 0.3)
    config.setdefault("inner_bbox_fraction", 0.7)
    config.setdefault("filter_oob_extrap_points", True)

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

    config.setdefault("rbf_downsample_factor", 4)
    config.setdefault("rbf_edge_downsample_factor", 8)
    config.setdefault("rbf_max_points", None)
    config.setdefault("rbf_edge_band_frac", 0.10)
    config.setdefault("rbf_edge_band_cells", None)
    config.setdefault("rbf_edge_min_points", 128)
    config.setdefault("debug_extrapolation_oob", False)
    config.setdefault("debug_extrapolation_oob_every", 100)
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

    triplet_direction_prior_mask = str(config.get("triplet_direction_prior_mask", "cond")).lower()
    _require_choice("triplet_direction_prior_mask", triplet_direction_prior_mask, {"cond", "full"})

    triplet_close_distance_voxels = float(config.get("triplet_close_distance_voxels", 1.0))
    _require_finite_range("triplet_close_distance_voxels", triplet_close_distance_voxels, min_value=0.0)

    triplet_close_fraction_threshold = float(config.get("triplet_close_fraction_threshold", 0.05))
    _require_finite_range(
        "triplet_close_fraction_threshold",
        triplet_close_fraction_threshold,
        min_value=0.0,
        max_value=1.0,
    )

    triplet_edt_bbox_padding_voxels = float(config.get("triplet_edt_bbox_padding_voxels", 4.0))
    _require_finite_range("triplet_edt_bbox_padding_voxels", triplet_edt_bbox_padding_voxels, min_value=0.0)

    use_dense_displacement = bool(config.get("use_dense_displacement", False))
    use_triplet_wrap_displacement = bool(config.get("use_triplet_wrap_displacement", False))

    if not use_triplet_wrap_displacement and not use_dense_displacement:
        raise ValueError(
            "Regular split no longer supports sparse extrapolation supervision; "
            "set use_dense_displacement=True."
        )

    if displacement_supervision == "normal_scalar" and use_dense_displacement:
        raise ValueError("displacement_supervision='normal_scalar' is not supported with use_dense_displacement=True")

    if use_triplet_wrap_displacement:
        if not use_dense_displacement:
            raise ValueError("use_triplet_wrap_displacement=True requires use_dense_displacement=True")
        if config.get("use_extrapolation", True):
            raise ValueError("use_triplet_wrap_displacement=True requires use_extrapolation=False")
        if config.get("use_other_wrap_cond", False):
            raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_other_wrap_cond")
        if config.get("use_sdt", False):
            raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_sdt")
        if config.get("use_heatmap_targets", False):
            raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_heatmap_targets")
        if config.get("use_segmentation", False):
            raise ValueError("use_triplet_wrap_displacement=True is not compatible with use_segmentation")
        if sample_mode != "wrap":
            raise ValueError("use_triplet_wrap_displacement=True requires sample_mode='wrap'")
