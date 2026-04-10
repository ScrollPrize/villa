from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_AUTOREG_MESH_CONFIG: dict = {
    "seed": 0,
    "crop_size": [128, 128, 128],
    "sample_mode": "wrap",
    "frontier_band_width": 4,
    "surface_downsample_factor": 1,
    "use_stored_resolution_only": False,
    "patch_size": [8, 8, 8],
    "offset_num_bins": [16, 16, 16],
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
    "pointer_temperature": 1.0,
    "scheduled_sampling_enabled": False,
    "scheduled_sampling_mode": "linear_full_token_greedy",
    "scheduled_sampling_max_prob": 0.10,
    "scheduled_sampling_start_step": 0,
    "scheduled_sampling_ramp_steps": 0,
    "position_refine_enabled": True,
    "position_refine_loss": "huber",
    "position_refine_weight": 0.05,
    "position_refine_start_step": 5000,
    "distance_aware_coarse_targets_enabled": True,
    "distance_aware_coarse_target_radius": 1,
    "distance_aware_coarse_target_sigma": 1.0,
    "distance_aware_coarse_target_loss": "soft_ce",
    "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4},
    "scheduler": "constant",
    "scheduler_kwargs": {},
    "batch_size": 1,
    "num_workers": 0,
    "val_num_workers": 0,
    "val_fraction": 0.1,
    "val_batches_per_log": 4,
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
}


def _as_3tuple(name: str, value) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (int(value), int(value), int(value))
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an int or a length-3 sequence, got {value!r}")
    return tuple(int(v) for v in value)


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
    if int(cfg["frontier_band_width"]) <= 0:
        raise ValueError("frontier_band_width must be positive")
    if int(cfg["surface_downsample_factor"]) <= 0:
        raise ValueError("surface_downsample_factor must be positive")
    if any(size <= 0 for size in cfg["input_shape"]):
        raise ValueError(f"input_shape must be positive, got {cfg['input_shape']!r}")
    if any(size <= 0 for size in cfg["patch_size"]):
        raise ValueError(f"patch_size must be positive, got {cfg['patch_size']!r}")
    if any(size <= 0 for size in cfg["offset_num_bins"]):
        raise ValueError(f"offset_num_bins must be positive, got {cfg['offset_num_bins']!r}")
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
    if str(cfg["scheduled_sampling_mode"]) != "linear_full_token_greedy":
        raise ValueError("scheduled_sampling_mode must currently be 'linear_full_token_greedy'")
    if float(cfg["scheduled_sampling_max_prob"]) < 0.0 or float(cfg["scheduled_sampling_max_prob"]) > 1.0:
        raise ValueError("scheduled_sampling_max_prob must be within [0, 1]")
    if int(cfg["scheduled_sampling_start_step"]) < 0:
        raise ValueError("scheduled_sampling_start_step must be >= 0")
    if int(cfg["scheduled_sampling_ramp_steps"]) < 0:
        raise ValueError("scheduled_sampling_ramp_steps must be >= 0")
    if str(cfg["position_refine_loss"]) != "huber":
        raise ValueError("position_refine_loss must currently be 'huber'")
    if float(cfg["position_refine_weight"]) < 0.0:
        raise ValueError("position_refine_weight must be non-negative")
    if int(cfg["position_refine_start_step"]) < 0:
        raise ValueError("position_refine_start_step must be >= 0")
    if int(cfg["distance_aware_coarse_target_radius"]) < 0:
        raise ValueError("distance_aware_coarse_target_radius must be >= 0")
    if float(cfg["distance_aware_coarse_target_sigma"]) <= 0.0:
        raise ValueError("distance_aware_coarse_target_sigma must be > 0")
    if str(cfg["distance_aware_coarse_target_loss"]) != "soft_ce":
        raise ValueError("distance_aware_coarse_target_loss must currently be 'soft_ce'")
    if float(cfg["occupancy_loss_weight"]) < 0.0:
        raise ValueError("occupancy_loss_weight must be non-negative")
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
    if int(cfg["val_batches_per_log"]) <= 0:
        raise ValueError("val_batches_per_log must be positive")
    if int(cfg["log_frequency"]) <= 0:
        raise ValueError("log_frequency must be positive")
    if int(cfg["ckpt_frequency"]) <= 0:
        raise ValueError("ckpt_frequency must be positive")
    if float(cfg["grad_clip"]) <= 0.0:
        raise ValueError("grad_clip must be positive")
    if str(cfg["mixed_precision"]).lower() != "no":
        raise ValueError(
            "autoreg_mesh currently does not implement mixed precision; set mixed_precision='no'"
        )
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

    optimizer = dict(cfg.get("optimizer") or {})
    optimizer.setdefault("name", "adamw")
    optimizer.setdefault("learning_rate", 1e-4)
    optimizer.setdefault("weight_decay", 1e-4)
    cfg["optimizer"] = optimizer
    return cfg


def load_autoreg_mesh_config(path: str | Path) -> dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    return validate_autoreg_mesh_config(raw)
