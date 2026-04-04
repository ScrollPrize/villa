from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_AUTOREG_MESH_CONFIG: dict = {
    "seed": 0,
    "crop_size": [128, 128, 128],
    "sample_mode": "wrap",
    "frontier_band_width": 1,
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
    "optimizer": {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4},
    "scheduler": "constant",
    "scheduler_kwargs": {},
    "batch_size": 1,
    "num_workers": 0,
    "num_steps": 1000,
    "grad_clip": 1.0,
    "mixed_precision": "no",
    "occupancy_loss_weight": 0.0,
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
    if float(cfg["occupancy_loss_weight"]) < 0.0:
        raise ValueError("occupancy_loss_weight must be non-negative")

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
