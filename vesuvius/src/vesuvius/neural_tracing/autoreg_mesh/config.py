from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch


_ROPE_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _normalize_surface_downsample_factor(value) -> int | tuple[int, int]:
    if isinstance(value, int):
        factor = int(value)
        if factor <= 0:
            raise ValueError("surface_downsample_factor must be positive")
        return factor
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(
            "surface_downsample_factor must be a positive int or a length-2 sequence, "
            f"got {value!r}"
        )
    factors = tuple(int(v) for v in value)
    if any(v <= 0 for v in factors):
        raise ValueError("surface_downsample_factor factors must be positive")
    if factors[0] == factors[1]:
        return int(factors[0])
    return factors


def _normalize_axis_list(name: str, value) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty sequence of axes")
    axes = [int(v) for v in value]
    if any(axis not in {0, 1, 2} for axis in axes):
        raise ValueError(f"{name} axes must be in {{0,1,2}}, got {value!r}")
    if len(set(axes)) != len(axes):
        raise ValueError(f"{name} axes must be unique, got {value!r}")
    return axes


DEFAULT_AUTOREG_MESH_CONFIG: dict = {
    "seed": 0,
    "crop_size": [128, 128, 128],
    "sample_mode": "wrap",
    "frontier_band_width": 4,
    "surface_downsample_factor": 1,
    "use_stored_resolution_only": False,
    "prefilter_show_progress": True,
    "spatial_augmentation": {
        "enabled": True,
        "mirror_prob": 0.5,
        "transpose_prob": 0.25,
        "mirror_axes": [0, 1, 2],
        "transpose_axes": [0, 1, 2],
    },
    "patch_size": [8, 8, 8],
    "offset_num_bins": [16, 16, 16],
    "offset_loss_weight": 1.0,
    "offset_loss_start_step": 0,
    "offset_loss_ramp_steps": 0,
    "direction_order": ["left", "right", "up", "down"],
    "cache_vol_tokens": False,
    "vol_token_cache": None,
    "dinov2_backbone": None,
    "dinov2_config_path": None,
    "input_shape": [128, 128, 128],
    "input_channels": 1,
    "decoder_dim": 192,
    "decoder_depth": 6,
    "decoder_num_heads": 8,
    "cross_attention_every_n_blocks": 1,
    "decoder_mlp_ratio": 4.0,
    "decoder_dropout": 0.0,
    "pointer_temperature": 0.25,
    "coarse_prediction_mode": "joint_pointer",
    "conditioning_feature_debias_mode": "none",
    "conditioning_feature_debias_basis_source": "zero_volume_svd",
    "conditioning_feature_debias_components": 16,
    "rope_base": 100.0,
    "rope_min_period": None,
    "rope_max_period": None,
    "rope_normalize_coords": "separate",
    "rope_shift_coords": 0.05,
    "rope_jitter_coords": 1.05,
    "rope_rescale_coords": 2.0,
    "rope_dtype": "float32",
    "cross_attention_use_rope": True,
    "scheduled_sampling_enabled": False,
    "scheduled_sampling_mode": "linear_full_token_greedy",
    "scheduled_sampling_pattern": "stripwise_full_strip_greedy",
    "scheduled_sampling_max_prob": 0.10,
    "scheduled_sampling_start_step": 0,
    "scheduled_sampling_ramp_steps": 0,
    "scheduled_sampling_offset_feedback_start_step": None,
    "position_refine_enabled": True,
    "position_refine_loss": "huber",
    "position_refine_weight": 0.05,
    "position_refine_start_step": 5000,
    "xyz_soft_loss_enabled": True,
    "xyz_soft_loss_weight": 1.0,
    "xyz_soft_loss_start_step": 0,
    "xyz_soft_loss": "huber",
    "seam_loss_enabled": True,
    "seam_loss_weight": 0.25,
    "seam_loss_start_step": 0,
    "seam_loss": "edge_huber",
    "seam_band_width": 1,
    "triangle_barrier_enabled": True,
    "triangle_barrier_weight": 0.1,
    "triangle_barrier_start_step": 0,
    "triangle_barrier_margin": 0.05,
    "boundary_loss_enabled": False,
    "boundary_loss_weight": 0.0,
    "boundary_loss_start_step": 0,
    "geometry_metric_enabled": True,
    "geometry_metric_weight": 0.01,
    "geometry_metric_start_step": 2000,
    "geometry_metric_loss": "huber",
    "geometry_sd_enabled": True,
    "geometry_sd_weight": 0.005,
    "geometry_sd_start_step": 2000,
    "distance_aware_coarse_targets_enabled": True,
    "distance_aware_coarse_target_radius": 1,
    "distance_aware_coarse_target_sigma": 1.0,
    "distance_aware_coarse_target_loss": "soft_ce",
    "distance_aware_offset_targets_enabled": False,
    "distance_aware_offset_target_radius": 1,
    "distance_aware_offset_target_sigma": 0.75,
    "coarse_continuation_constraint_enabled": True,
    "coarse_continuation_constraint_mode": "hard_mask",
    "coarse_continuation_forward_scale": 4.0,
    "coarse_continuation_backward_scale": 1.5,
    "coarse_continuation_lateral_scale": 3.0,
    "coarse_continuation_min_radius_scale": 2.5,
    "coarse_continuation_empty_fallback": "disable_for_token",
    "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4},
    "scheduler": "constant",
    "scheduler_kwargs": {},
    "batch_size": 1,
    "num_workers": 0,
    "val_num_workers": 0,
    "val_fraction": 0.1,
    "val_split_mode": "spatial_groups",
    "validation_leakage_margin_voxels": 0.0,
    "val_batches_per_log": 4,
    "rollout_val_examples_per_log": 1,
    "rollout_val_max_steps": None,
    "num_steps": 1000,
    "grad_clip": 1.0,
    "mixed_precision": "no",
    "occupancy_loss_weight": 0.0,
    "out_dir": "./autoreg_mesh_runs",
    "log_frequency": 100,
    "ckpt_frequency": 5000,
    "ckpt_at_step_zero": False,
    "save_final_checkpoint": True,
    "load_ckpt": None,
    "load_weights_only": False,
    "wandb_project": None,
    "wandb_entity": None,
    "wandb_run_name": None,
    "wandb_resume": False,
    "wandb_resume_mode": "allow",
    "wandb_run_id": None,
    "wandb_log_images": True,
    "wandb_image_frequency": None,
    "wandb_log_xy_slice_images": True,
    "wandb_xy_slice_image_frequency": None,
    "wandb_xy_slice_mode": "best_xy_slice",
    "wandb_xy_slice_line_thickness": 1,
    "wandb_xy_slice_depth_tolerance": 0.75,
    "volume_only_augmentation": {
        "enabled": True,
        "contrast_prob": 0.3,
        "mult_brightness_prob": 0.3,
        "add_brightness_prob": 0.1,
        "gamma_prob": 0.4,
        "gaussian_noise_prob": 0.2,
        "gaussian_blur_prob": 0.15,
        "slice_illumination_prob": 0.15,
        "lowres_prob": 0.10,
    },
}


def _as_3tuple(name: str, value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


def _resolve_rope_dtype(value):
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ROPE_DTYPE_ALIASES:
            return _ROPE_DTYPE_ALIASES[normalized]
    raise ValueError(f"unsupported rope_dtype value: {value!r}")


def setdefault_autoreg_mesh_config(config: dict) -> dict:
    merged = deepcopy(DEFAULT_AUTOREG_MESH_CONFIG)
    for key, value in dict(config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def validate_autoreg_mesh_config(config: dict) -> dict:
    cfg = setdefault_autoreg_mesh_config(config)

    cfg["crop_size"] = list(_as_3tuple("crop_size", cfg["crop_size"]))
    cfg["input_shape"] = list(_as_3tuple("input_shape", cfg["input_shape"]))
    cfg["patch_size"] = list(_as_3tuple("patch_size", cfg["patch_size"]))
    cfg["offset_num_bins"] = list(_as_3tuple("offset_num_bins", cfg["offset_num_bins"]))

    if cfg["sample_mode"] != "wrap":
        raise ValueError("autoreg_mesh MVP requires sample_mode='wrap'")
    if not isinstance(cfg.get("prefilter_show_progress"), bool):
        raise ValueError("prefilter_show_progress must be a boolean")
    if int(cfg["frontier_band_width"]) <= 0:
        raise ValueError("frontier_band_width must be positive")
    cfg["surface_downsample_factor"] = _normalize_surface_downsample_factor(cfg["surface_downsample_factor"])
    if any(size <= 0 for size in cfg["input_shape"]):
        raise ValueError(f"input_shape must be positive, got {cfg['input_shape']!r}")
    if any(size <= 0 for size in cfg["patch_size"]):
        raise ValueError(f"patch_size must be positive, got {cfg['patch_size']!r}")
    if any(size <= 0 for size in cfg["offset_num_bins"]):
        raise ValueError(f"offset_num_bins must be positive, got {cfg['offset_num_bins']!r}")
    if float(cfg["offset_loss_weight"]) < 0.0:
        raise ValueError("offset_loss_weight must be non-negative")
    if int(cfg["offset_loss_start_step"]) < 0:
        raise ValueError("offset_loss_start_step must be >= 0")
    if int(cfg["offset_loss_ramp_steps"]) < 0:
        raise ValueError("offset_loss_ramp_steps must be >= 0")
    if any(size % patch != 0 for size, patch in zip(cfg["input_shape"], cfg["patch_size"])):
        raise ValueError(
            f"input_shape {cfg['input_shape']!r} must be divisible by patch_size {cfg['patch_size']!r}"
        )
    if int(cfg["decoder_dim"]) <= 0 or int(cfg["decoder_depth"]) <= 0 or int(cfg["decoder_num_heads"]) <= 0:
        raise ValueError("decoder_dim, decoder_depth, and decoder_num_heads must be positive")
    if int(cfg["decoder_dim"]) % int(cfg["decoder_num_heads"]) != 0:
        raise ValueError(
            f"decoder_dim={cfg['decoder_dim']} must be divisible by decoder_num_heads={cfg['decoder_num_heads']}"
        )
    head_dim = int(cfg["decoder_dim"]) // int(cfg["decoder_num_heads"])
    if head_dim % 6 != 0:
        raise ValueError(
            "decoder head_dim must be divisible by 6 for 3D rotary embeddings, "
            f"got decoder_dim={cfg['decoder_dim']} decoder_num_heads={cfg['decoder_num_heads']}"
        )
    if int(cfg["cross_attention_every_n_blocks"]) <= 0:
        raise ValueError("cross_attention_every_n_blocks must be positive")
    if float(cfg["pointer_temperature"]) <= 0.0:
        raise ValueError("pointer_temperature must be positive")
    if str(cfg["coarse_prediction_mode"]) not in {"joint_pointer", "axis_factorized"}:
        raise ValueError("coarse_prediction_mode must be one of {'joint_pointer', 'axis_factorized'}")
    if str(cfg["conditioning_feature_debias_mode"]) not in {"none", "orthogonal_project"}:
        raise ValueError("conditioning_feature_debias_mode must be one of {'none', 'orthogonal_project'}")
    if str(cfg["conditioning_feature_debias_basis_source"]) != "zero_volume_svd":
        raise ValueError("conditioning_feature_debias_basis_source must currently be 'zero_volume_svd'")
    if int(cfg["conditioning_feature_debias_components"]) <= 0:
        raise ValueError("conditioning_feature_debias_components must be positive")
    if cfg.get("rope_normalize_coords") not in {"min", "max", "separate"}:
        raise ValueError("rope_normalize_coords must be one of {'min', 'max', 'separate'}")
    if cfg.get("rope_base") is not None and (cfg.get("rope_min_period") is not None or cfg.get("rope_max_period") is not None):
        raise ValueError("rope_base cannot be set together with rope_min_period or rope_max_period")
    if cfg.get("rope_base") is None and (cfg.get("rope_min_period") is None or cfg.get("rope_max_period") is None):
        raise ValueError("when rope_base is None, both rope_min_period and rope_max_period must be set")
    if cfg.get("rope_shift_coords") is not None and float(cfg["rope_shift_coords"]) < 0.0:
        raise ValueError("rope_shift_coords must be >= 0")
    if cfg.get("rope_jitter_coords") is not None and float(cfg["rope_jitter_coords"]) <= 1.0:
        raise ValueError("rope_jitter_coords must be > 1.0")
    if cfg.get("rope_rescale_coords") is not None and float(cfg["rope_rescale_coords"]) <= 1.0:
        raise ValueError("rope_rescale_coords must be > 1.0")
    _resolve_rope_dtype(cfg.get("rope_dtype"))
    if not isinstance(cfg.get("cross_attention_use_rope"), bool):
        raise ValueError("cross_attention_use_rope must be a boolean")
    if str(cfg["scheduled_sampling_mode"]) != "linear_full_token_greedy":
        raise ValueError("scheduled_sampling_mode must currently be 'linear_full_token_greedy'")
    if str(cfg["scheduled_sampling_pattern"]) != "stripwise_full_strip_greedy":
        raise ValueError("scheduled_sampling_pattern must currently be 'stripwise_full_strip_greedy'")
    if float(cfg["scheduled_sampling_max_prob"]) < 0.0 or float(cfg["scheduled_sampling_max_prob"]) > 1.0:
        raise ValueError("scheduled_sampling_max_prob must be within [0, 1]")
    if int(cfg["scheduled_sampling_start_step"]) < 0:
        raise ValueError("scheduled_sampling_start_step must be >= 0")
    if int(cfg["scheduled_sampling_ramp_steps"]) < 0:
        raise ValueError("scheduled_sampling_ramp_steps must be >= 0")
    if cfg.get("scheduled_sampling_offset_feedback_start_step") is not None and int(cfg["scheduled_sampling_offset_feedback_start_step"]) < 0:
        raise ValueError("scheduled_sampling_offset_feedback_start_step must be None or >= 0")
    if str(cfg["position_refine_loss"]) != "huber":
        raise ValueError("position_refine_loss must currently be 'huber'")
    if float(cfg["position_refine_weight"]) < 0.0:
        raise ValueError("position_refine_weight must be non-negative")
    if int(cfg["position_refine_start_step"]) < 0:
        raise ValueError("position_refine_start_step must be >= 0")
    if str(cfg["xyz_soft_loss"]) != "huber":
        raise ValueError("xyz_soft_loss must currently be 'huber'")
    if float(cfg["xyz_soft_loss_weight"]) < 0.0:
        raise ValueError("xyz_soft_loss_weight must be non-negative")
    if int(cfg["xyz_soft_loss_start_step"]) < 0:
        raise ValueError("xyz_soft_loss_start_step must be >= 0")
    if str(cfg["seam_loss"]) != "edge_huber":
        raise ValueError("seam_loss must currently be 'edge_huber'")
    if float(cfg["seam_loss_weight"]) < 0.0:
        raise ValueError("seam_loss_weight must be non-negative")
    if int(cfg["seam_loss_start_step"]) < 0:
        raise ValueError("seam_loss_start_step must be >= 0")
    if int(cfg["seam_band_width"]) <= 0:
        raise ValueError("seam_band_width must be positive")
    if float(cfg["triangle_barrier_weight"]) < 0.0:
        raise ValueError("triangle_barrier_weight must be non-negative")
    if int(cfg["triangle_barrier_start_step"]) < 0:
        raise ValueError("triangle_barrier_start_step must be >= 0")
    if float(cfg["triangle_barrier_margin"]) < 0.0:
        raise ValueError("triangle_barrier_margin must be non-negative")
    if float(cfg["boundary_loss_weight"]) < 0.0:
        raise ValueError("boundary_loss_weight must be non-negative")
    if int(cfg["boundary_loss_start_step"]) < 0:
        raise ValueError("boundary_loss_start_step must be >= 0")
    if str(cfg["geometry_metric_loss"]) != "huber":
        raise ValueError("geometry_metric_loss must currently be 'huber'")
    if float(cfg["geometry_metric_weight"]) < 0.0:
        raise ValueError("geometry_metric_weight must be non-negative")
    if int(cfg["geometry_metric_start_step"]) < 0:
        raise ValueError("geometry_metric_start_step must be >= 0")
    if float(cfg["geometry_sd_weight"]) < 0.0:
        raise ValueError("geometry_sd_weight must be non-negative")
    if int(cfg["geometry_sd_start_step"]) < 0:
        raise ValueError("geometry_sd_start_step must be >= 0")
    if int(cfg["distance_aware_coarse_target_radius"]) < 0:
        raise ValueError("distance_aware_coarse_target_radius must be >= 0")
    if float(cfg["distance_aware_coarse_target_sigma"]) <= 0.0:
        raise ValueError("distance_aware_coarse_target_sigma must be > 0")
    if str(cfg["distance_aware_coarse_target_loss"]) != "soft_ce":
        raise ValueError("distance_aware_coarse_target_loss must currently be 'soft_ce'")
    if not isinstance(cfg.get("distance_aware_offset_targets_enabled"), bool):
        raise ValueError("distance_aware_offset_targets_enabled must be a boolean")
    if int(cfg["distance_aware_offset_target_radius"]) < 0:
        raise ValueError("distance_aware_offset_target_radius must be >= 0")
    if float(cfg["distance_aware_offset_target_sigma"]) <= 0.0:
        raise ValueError("distance_aware_offset_target_sigma must be > 0")
    if not isinstance(cfg.get("coarse_continuation_constraint_enabled"), bool):
        raise ValueError("coarse_continuation_constraint_enabled must be a boolean")
    if str(cfg["coarse_continuation_constraint_mode"]) != "hard_mask":
        raise ValueError("coarse_continuation_constraint_mode must currently be 'hard_mask'")
    if float(cfg["coarse_continuation_forward_scale"]) <= 0.0:
        raise ValueError("coarse_continuation_forward_scale must be positive")
    if float(cfg["coarse_continuation_backward_scale"]) < 0.0:
        raise ValueError("coarse_continuation_backward_scale must be non-negative")
    if float(cfg["coarse_continuation_lateral_scale"]) <= 0.0:
        raise ValueError("coarse_continuation_lateral_scale must be positive")
    if float(cfg["coarse_continuation_min_radius_scale"]) <= 0.0:
        raise ValueError("coarse_continuation_min_radius_scale must be positive")
    if str(cfg["coarse_continuation_empty_fallback"]) != "disable_for_token":
        raise ValueError("coarse_continuation_empty_fallback must currently be 'disable_for_token'")
    if float(cfg["occupancy_loss_weight"]) < 0.0:
        raise ValueError("occupancy_loss_weight must be non-negative")
    raw_spatial_aug = cfg.get("spatial_augmentation") or {}
    if not isinstance(raw_spatial_aug, dict):
        raise ValueError("spatial_augmentation must be a mapping of augmentation settings")
    spatial_aug = dict(raw_spatial_aug)
    for key, default in DEFAULT_AUTOREG_MESH_CONFIG["spatial_augmentation"].items():
        spatial_aug.setdefault(key, default)
    if not isinstance(spatial_aug.get("enabled"), bool):
        raise ValueError("spatial_augmentation.enabled must be a boolean")
    for key in ("mirror_prob", "transpose_prob"):
        value = float(spatial_aug[key])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"spatial_augmentation.{key} must be within [0, 1]")
    spatial_aug["mirror_axes"] = _normalize_axis_list("spatial_augmentation.mirror_axes", spatial_aug["mirror_axes"])
    spatial_aug["transpose_axes"] = _normalize_axis_list("spatial_augmentation.transpose_axes", spatial_aug["transpose_axes"])
    if bool(cfg.get("cache_vol_tokens", False)) and bool(spatial_aug["enabled"]):
        raise ValueError("spatial_augmentation.enabled=true is incompatible with cache_vol_tokens=true")
    cfg["spatial_augmentation"] = spatial_aug
    if int(cfg["num_steps"]) <= 0:
        raise ValueError("num_steps must be positive")
    if int(cfg["batch_size"]) <= 0:
        raise ValueError("batch_size must be positive")
    if int(cfg["num_workers"]) < 0:
        raise ValueError("num_workers must be non-negative")
    if int(cfg["val_num_workers"]) < 0:
        raise ValueError("val_num_workers must be non-negative")
    if float(cfg["val_fraction"]) < 0.0 or float(cfg["val_fraction"]) > 1.0:
        raise ValueError("val_fraction must be within [0, 1]")
    if str(cfg["val_split_mode"]) not in {"random_samples", "spatial_groups"}:
        raise ValueError("val_split_mode must be one of {'random_samples', 'spatial_groups'}")
    if float(cfg["validation_leakage_margin_voxels"]) < 0.0:
        raise ValueError("validation_leakage_margin_voxels must be non-negative")
    if int(cfg["val_batches_per_log"]) <= 0:
        raise ValueError("val_batches_per_log must be positive")
    if int(cfg["rollout_val_examples_per_log"]) <= 0:
        raise ValueError("rollout_val_examples_per_log must be positive")
    if cfg["rollout_val_max_steps"] is not None and int(cfg["rollout_val_max_steps"]) <= 0:
        raise ValueError("rollout_val_max_steps must be None or positive")
    if int(cfg["log_frequency"]) <= 0:
        raise ValueError("log_frequency must be positive")
    if int(cfg["ckpt_frequency"]) <= 0:
        raise ValueError("ckpt_frequency must be positive")
    if float(cfg["grad_clip"]) <= 0.0:
        raise ValueError("grad_clip must be positive")
    if str(cfg["mixed_precision"]).lower() not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be one of {'no', 'bf16'}")
    if cfg["out_dir"] is None or str(cfg["out_dir"]).strip() == "":
        raise ValueError("out_dir must be a non-empty path")
    if cfg.get("load_ckpt") is not None and str(cfg["load_ckpt"]).strip() == "":
        raise ValueError("load_ckpt must be None or a non-empty path")
    if cfg.get("wandb_resume_mode") not in {"allow", "must", "never", "auto"}:
        raise ValueError("wandb_resume_mode must be one of {'allow', 'must', 'never', 'auto'}")

    if cfg.get("wandb_image_frequency") is None:
        cfg["wandb_image_frequency"] = int(cfg["log_frequency"])
    if int(cfg["wandb_image_frequency"]) <= 0:
        raise ValueError("wandb_image_frequency must be positive")
    if cfg.get("wandb_xy_slice_image_frequency") is None:
        cfg["wandb_xy_slice_image_frequency"] = int(cfg["wandb_image_frequency"])
    if int(cfg["wandb_xy_slice_image_frequency"]) <= 0:
        raise ValueError("wandb_xy_slice_image_frequency must be positive")
    if str(cfg["wandb_xy_slice_mode"]) != "best_xy_slice":
        raise ValueError("wandb_xy_slice_mode must currently be 'best_xy_slice'")
    if int(cfg["wandb_xy_slice_line_thickness"]) <= 0:
        raise ValueError("wandb_xy_slice_line_thickness must be positive")
    if float(cfg["wandb_xy_slice_depth_tolerance"]) <= 0.0:
        raise ValueError("wandb_xy_slice_depth_tolerance must be positive")

    raw_volume_only_aug = cfg.get("volume_only_augmentation") or {}
    if not isinstance(raw_volume_only_aug, dict):
        raise ValueError("volume_only_augmentation must be a mapping of augmentation settings")
    volume_only_aug = dict(raw_volume_only_aug)
    volume_only_aug.setdefault("enabled", True)
    for key, default in DEFAULT_AUTOREG_MESH_CONFIG["volume_only_augmentation"].items():
        volume_only_aug.setdefault(key, default)
    for key in (
        "contrast_prob",
        "mult_brightness_prob",
        "add_brightness_prob",
        "gamma_prob",
        "gaussian_noise_prob",
        "gaussian_blur_prob",
        "slice_illumination_prob",
        "lowres_prob",
    ):
        value = float(volume_only_aug[key])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"volume_only_augmentation.{key} must be within [0, 1]")
    cfg["volume_only_augmentation"] = volume_only_aug

    optimizer = dict(cfg.get("optimizer") or {})
    optimizer_name = str(optimizer.get("name", "adamw")).lower()
    optimizer["name"] = optimizer_name
    if optimizer_name == "muon":
        optimizer.setdefault("learning_rate", 0.02)
        optimizer.setdefault("momentum", 0.95)
        optimizer.setdefault("weight_decouple", True)
        optimizer.setdefault("nesterov", True)
        optimizer.setdefault("ns_steps", 5)
        optimizer.setdefault("use_adjusted_lr", False)
        optimizer.setdefault("adamw_lr", 3e-4)
        optimizer.setdefault("adamw_betas", [0.9, 0.95])
        optimizer.setdefault("adamw_wd", optimizer.get("weight_decay", 1e-4))
        optimizer.setdefault("adamw_eps", 1e-10)
        optimizer.setdefault("maximize", False)
    else:
        optimizer.setdefault("learning_rate", 1e-4)
    optimizer.setdefault("weight_decay", 1e-4)
    if float(optimizer["learning_rate"]) <= 0.0:
        raise ValueError("optimizer.learning_rate must be positive")
    if float(optimizer["weight_decay"]) < 0.0:
        raise ValueError("optimizer.weight_decay must be non-negative")
    if optimizer_name == "muon":
        if float(optimizer["momentum"]) < 0.0 or float(optimizer["momentum"]) >= 1.0:
            raise ValueError("optimizer.momentum must be within [0, 1)")
        if int(optimizer["ns_steps"]) < 1:
            raise ValueError("optimizer.ns_steps must be >= 1")
        if float(optimizer["adamw_lr"]) <= 0.0:
            raise ValueError("optimizer.adamw_lr must be positive")
        adamw_betas = optimizer.get("adamw_betas")
        if not isinstance(adamw_betas, (list, tuple)) or len(adamw_betas) != 2:
            raise ValueError("optimizer.adamw_betas must be a length-2 sequence")
        adamw_betas = [float(v) for v in adamw_betas]
        if any(beta < 0.0 or beta >= 1.0 for beta in adamw_betas):
            raise ValueError("optimizer.adamw_betas entries must be within [0, 1)")
        optimizer["adamw_betas"] = adamw_betas
        if float(optimizer["adamw_wd"]) < 0.0:
            raise ValueError("optimizer.adamw_wd must be non-negative")
        if float(optimizer["adamw_eps"]) <= 0.0:
            raise ValueError("optimizer.adamw_eps must be positive")
        for key in ("weight_decouple", "nesterov", "use_adjusted_lr", "maximize"):
            if not isinstance(optimizer[key], bool):
                raise ValueError(f"optimizer.{key} must be a boolean")
    cfg["optimizer"] = optimizer
    return cfg


def load_autoreg_mesh_config(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return validate_autoreg_mesh_config(raw)
