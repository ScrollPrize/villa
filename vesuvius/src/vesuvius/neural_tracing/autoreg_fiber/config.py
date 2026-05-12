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


LOCAL_DINOVOL_CHECKPOINT = "/home/giorgio/Projects/dinovol/checkpoint_step_352500.pt"


DEFAULT_AUTOREG_FIBER_CONFIG: dict = {
    "seed": 0,
    "crop_size": [128, 128, 128],
    "input_shape": [128, 128, 128],
    "patch_size": [8, 8, 8],
    "offset_num_bins": [16, 16, 16],
    "prompt_length": 8,
    "target_length": 32,
    "point_stride": 1,
    "fiber_cache_paths": [],
    "fiber_cache_manifest_json": None,
    "volume_zarr_url": None,
    "storage_options": {},
    "volume_shape": None,
    "volumes": {},
    "dinov2_backbone": LOCAL_DINOVOL_CHECKPOINT,
    "dinov2_config_path": None,
    "input_channels": 1,
    "decoder_dim": 192,
    "decoder_depth": 6,
    "decoder_num_heads": 8,
    "cross_attention_every_n_blocks": 1,
    "attention_scaling_mode": "standard",
    "decoder_mlp_ratio": 4.0,
    "decoder_dropout": 0.0,
    "pointer_temperature": 0.25,
    "coarse_prediction_mode": "joint_pointer",
    "ddp_find_unused_parameters": None,
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
    "max_fiber_position_embeddings": 4096,
    "tangent_conditioning_enabled": True,
    "scheduled_sampling_enabled": False,
    "scheduled_sampling_mode": "linear_token_greedy",
    "scheduled_sampling_max_prob": 0.10,
    "scheduled_sampling_start_step": 0,
    "scheduled_sampling_ramp_steps": 0,
    "offset_loss_weight": 1.0,
    "offset_loss_start_step": 0,
    "position_refine_enabled": True,
    "position_refine_loss": "huber",
    "position_refine_weight": 0.05,
    "position_refine_start_step": 5000,
    "xyz_soft_loss_enabled": True,
    "xyz_soft_loss": "huber",
    "xyz_soft_loss_weight": 1.0,
    "xyz_soft_loss_start_step": 0,
    "segment_vector_loss_enabled": True,
    "segment_vector_loss": "huber",
    "segment_vector_loss_weight": 0.05,
    "segment_vector_loss_start_step": 0,
    "straightness_loss_enabled": True,
    "straightness_loss": "huber",
    "straightness_loss_weight": 0.0,
    "straightness_loss_start_step": 0,
    "tube_radius_loss_enabled": True,
    "tube_radius_loss": "huber",
    "tube_radius_loss_weight": 0.0,
    "tube_radius_loss_start_step": 0,
    "distance_aware_coarse_targets_enabled": True,
    "distance_aware_coarse_target_radius": 1,
    "distance_aware_coarse_target_sigma": 1.0,
    "distance_aware_coarse_target_loss": "soft_ce",
    "spatial_augmentation_enabled": False,
    "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4},
    "scheduler": "constant",
    "scheduler_kwargs": {},
    "batch_size": 1,
    "num_workers": 0,
    "val_num_workers": 0,
    "val_fraction": 0.1,
    "val_batches_per_log": 4,
    "rollout_val_examples_per_log": 1,
    "rollout_val_max_steps": None,
    "num_steps": 1000,
    "grad_clip": 1.0,
    "mixed_precision": "no",
    "out_dir": "./autoreg_fiber_runs",
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
    "wandb_log_dataset_summary": True,
    "wandb_dataset_table_max_rows": 256,
    "wandb_log_images": True,
    "wandb_image_frequency": None,
    "wandb_log_xy_slice_images": True,
    "wandb_xy_slice_image_frequency": None,
    "wandb_xy_slice_mode": "best_xy_slice",
    "wandb_xy_slice_line_thickness": 2,
    "wandb_xy_slice_depth_tolerance": 0.75,
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


def setdefault_autoreg_fiber_config(config: dict) -> dict:
    merged = deepcopy(DEFAULT_AUTOREG_FIBER_CONFIG)
    for key, value in dict(config or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def validate_autoreg_fiber_config(config: dict) -> dict:
    cfg = setdefault_autoreg_fiber_config(config)
    cfg["crop_size"] = list(_as_3tuple("crop_size", cfg["crop_size"]))
    cfg["input_shape"] = list(_as_3tuple("input_shape", cfg.get("input_shape") or cfg["crop_size"]))
    cfg["patch_size"] = list(_as_3tuple("patch_size", cfg["patch_size"]))
    cfg["offset_num_bins"] = list(_as_3tuple("offset_num_bins", cfg["offset_num_bins"]))
    if cfg.get("volume_shape") is not None:
        cfg["volume_shape"] = list(_as_3tuple("volume_shape", cfg["volume_shape"]))

    positive_int_keys = (
        "prompt_length",
        "target_length",
        "point_stride",
        "input_channels",
        "decoder_dim",
        "decoder_depth",
        "decoder_num_heads",
        "cross_attention_every_n_blocks",
        "max_fiber_position_embeddings",
        "batch_size",
        "num_steps",
        "log_frequency",
        "ckpt_frequency",
    )
    for key in positive_int_keys:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("num_workers", "val_num_workers", "offset_loss_start_step", "position_refine_start_step", "xyz_soft_loss_start_step", "segment_vector_loss_start_step", "straightness_loss_start_step", "tube_radius_loss_start_step"):
        if int(cfg[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if any(size <= 0 for size in cfg["input_shape"]):
        raise ValueError("input_shape values must be positive")
    if any(size <= 0 for size in cfg["patch_size"]):
        raise ValueError("patch_size values must be positive")
    if any(size <= 0 for size in cfg["offset_num_bins"]):
        raise ValueError("offset_num_bins values must be positive")
    if any(size % patch != 0 for size, patch in zip(cfg["input_shape"], cfg["patch_size"], strict=True)):
        raise ValueError(f"input_shape {cfg['input_shape']!r} must be divisible by patch_size {cfg['patch_size']!r}")
    if int(cfg["decoder_dim"]) % int(cfg["decoder_num_heads"]) != 0:
        raise ValueError("decoder_dim must be divisible by decoder_num_heads")
    head_dim = int(cfg["decoder_dim"]) // int(cfg["decoder_num_heads"])
    if head_dim % 6 != 0:
        raise ValueError("decoder head_dim must be divisible by 6 for 3D rotary embeddings")
    if float(cfg["pointer_temperature"]) <= 0.0:
        raise ValueError("pointer_temperature must be positive")
    if int(cfg["cross_attention_every_n_blocks"]) <= 0:
        raise ValueError("cross_attention_every_n_blocks must be positive")
    if str(cfg["attention_scaling_mode"]) not in {"standard", "legacy_double_scaled"}:
        raise ValueError("attention_scaling_mode must be one of {'standard', 'legacy_double_scaled'}")
    if str(cfg["coarse_prediction_mode"]) not in {"joint_pointer", "axis_factorized"}:
        raise ValueError("coarse_prediction_mode must be one of {'joint_pointer', 'axis_factorized'}")
    if cfg.get("ddp_find_unused_parameters") is not None and not isinstance(cfg["ddp_find_unused_parameters"], bool):
        raise ValueError("ddp_find_unused_parameters must be null, true, or false")
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
    if str(cfg["scheduled_sampling_mode"]) != "linear_token_greedy":
        raise ValueError("scheduled_sampling_mode must currently be 'linear_token_greedy'")
    for key in ("val_fraction", "scheduled_sampling_max_prob"):
        value = float(cfg[key])
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be within [0, 1]")
    for key in ("offset_loss_weight", "position_refine_weight", "xyz_soft_loss_weight", "segment_vector_loss_weight", "straightness_loss_weight", "tube_radius_loss_weight"):
        if float(cfg[key]) < 0.0:
            raise ValueError(f"{key} must be non-negative")
    if str(cfg["position_refine_loss"]) != "huber":
        raise ValueError("position_refine_loss must currently be 'huber'")
    if str(cfg["xyz_soft_loss"]) != "huber":
        raise ValueError("xyz_soft_loss must currently be 'huber'")
    if str(cfg["segment_vector_loss"]) != "huber":
        raise ValueError("segment_vector_loss must currently be 'huber'")
    if str(cfg["straightness_loss"]) != "huber":
        raise ValueError("straightness_loss must currently be 'huber'")
    if str(cfg["tube_radius_loss"]) != "huber":
        raise ValueError("tube_radius_loss must currently be 'huber'")
    if int(cfg["distance_aware_coarse_target_radius"]) < 0:
        raise ValueError("distance_aware_coarse_target_radius must be >= 0")
    if float(cfg["distance_aware_coarse_target_sigma"]) <= 0.0:
        raise ValueError("distance_aware_coarse_target_sigma must be > 0")
    if str(cfg["distance_aware_coarse_target_loss"]) != "soft_ce":
        raise ValueError("distance_aware_coarse_target_loss must currently be 'soft_ce'")
    if str(cfg["mixed_precision"]).lower() not in {"no", "bf16"}:
        raise ValueError("mixed_precision must be one of {'no', 'bf16'}")
    if float(cfg["grad_clip"]) <= 0.0:
        raise ValueError("grad_clip must be positive")
    if cfg["out_dir"] is None or str(cfg["out_dir"]).strip() == "":
        raise ValueError("out_dir must be a non-empty path")
    if cfg.get("load_ckpt") is not None and str(cfg["load_ckpt"]).strip() == "":
        raise ValueError("load_ckpt must be None or a non-empty path")
    if cfg.get("wandb_resume_mode") not in {"allow", "must", "never", "auto"}:
        raise ValueError("wandb_resume_mode must be one of {'allow', 'must', 'never', 'auto'}")
    if int(cfg["wandb_dataset_table_max_rows"]) <= 0:
        raise ValueError("wandb_dataset_table_max_rows must be positive")
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
    return cfg


def load_autoreg_fiber_config(path: str | Path) -> dict:
    return validate_autoreg_fiber_config(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "DEFAULT_AUTOREG_FIBER_CONFIG",
    "LOCAL_DINOVOL_CHECKPOINT",
    "load_autoreg_fiber_config",
    "setdefault_autoreg_fiber_config",
    "validate_autoreg_fiber_config",
]
