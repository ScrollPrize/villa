from typing import Any, Mapping, Optional, Tuple

import torch

from .dinovol_2_eva import Eva, EvaWithChunking


_BACKBONE_DEFAULTS = {
    "input_channels": 1,
    "global_crops_size": (256, 256, 256),
    "local_crops_size": None,
    "embed_dim": 864,
    "patch_size": (16, 16, 16),
    "embedding_type": "default",
    "deeper_embed_patch_chunk_size": None,
    "deeper_embed_batch_chunk_size": 1,
    "depth": 24,
    "num_heads": 16,
    "qkv_bias": True,
    "qkv_fused": False,
    "mlp_ratio": 8.0 / 3.0,
    "swiglu_mlp": True,
    "scale_mlp": True,
    "scale_attn_inner": False,
    "proj_drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
    "drop_path_uniform": False,
    "init_values": None,
    "use_abs_pos_emb": False,
    "use_rot_pos_emb": True,
    "num_reg_tokens": 4,
    "grad_checkpointing": False,
    "block_chunks": 0,
}
_ROPE_KWARG_CONFIG_KEYS = {
    "base": "rope_base",
    "min_period": "rope_min_period",
    "max_period": "rope_max_period",
    "normalize_coords": "rope_normalize_coords",
    "shift_coords": "rope_shift_coords",
    "jitter_coords": "rope_jitter_coords",
    "rescale_coords": "rope_rescale_coords",
    "dtype": "rope_dtype",
}
_ROPE_KWARG_DEFAULTS = {
    "base": 100.0,
    "normalize_coords": "separate",
    "rescale_coords": 2.0,
}
_ROPE_KWARG_FALLBACK_KEYS = {
    "base": "pos_embed_rope_base",
    "min_period": "pos_embed_rope_min_period",
    "max_period": "pos_embed_rope_max_period",
    "normalize_coords": "pos_embed_rope_normalize_coords",
    "shift_coords": "pos_embed_rope_shift_coords",
    "jitter_coords": "pos_embed_rope_jitter_coords",
    "rescale_coords": "pos_embed_rope_rescale_coords",
    "dtype": "pos_embed_rope_dtype",
}
_ROPE_DTYPE_ALIASES = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _as_3tuple(value: int | Tuple[int, int, int]) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(v) for v in value)
    if len(result) != 3:
        raise ValueError(f"expected 3 values and got {len(result)}: {result}")
    return result


def _config_value(
    config: Mapping[str, Any],
    key: str,
    default: Any,
    *,
    fallback_key: Optional[str] = None,
) -> Any:
    if key in config:
        return config[key]
    if fallback_key is not None and fallback_key in config:
        return config[fallback_key]
    return default


def _resolve_rope_dtype(value: Any) -> Any:
    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _ROPE_DTYPE_ALIASES:
            return _ROPE_DTYPE_ALIASES[normalized]
    raise ValueError(f"unsupported rope dtype value: {value!r}")


def _resolve_rope_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    rope_kwargs = dict(config.get("rope_kwargs") or {})
    for rope_key, config_key in _ROPE_KWARG_CONFIG_KEYS.items():
        if rope_key in rope_kwargs:
            continue
        fallback_key = _ROPE_KWARG_FALLBACK_KEYS[rope_key]
        if config_key in config or fallback_key in config:
            rope_kwargs[rope_key] = _config_value(config, config_key, None, fallback_key=fallback_key)

    for rope_key, default_value in _ROPE_KWARG_DEFAULTS.items():
        rope_kwargs.setdefault(rope_key, default_value)

    if "dtype" in rope_kwargs:
        rope_kwargs["dtype"] = _resolve_rope_dtype(rope_kwargs["dtype"])

    return rope_kwargs


def build_dinovol_2_backbone(config: Mapping[str, Any]) -> Eva:
    backbone_config = {key: config.get(key, default) for key, default in _BACKBONE_DEFAULTS.items()}
    if "num_reg_tokens" not in config and "num_register_tokens" in config:
        backbone_config["num_reg_tokens"] = int(config["num_register_tokens"])
    rope_kwargs = _resolve_rope_kwargs(config)
    global_crops_size = _as_3tuple(backbone_config["global_crops_size"])
    local_crop_value = backbone_config["local_crops_size"] or global_crops_size
    local_crops_size = _as_3tuple(local_crop_value)
    block_chunks = int(backbone_config["block_chunks"])
    backbone_cls = EvaWithChunking if block_chunks > 0 else Eva
    kwargs = dict(backbone_config)
    kwargs.update(
        {
            "global_crops_size": global_crops_size,
            "local_crops_size": local_crops_size,
            "patch_size": _as_3tuple(backbone_config["patch_size"]),
            "rope_kwargs": rope_kwargs,
        }
    )
    kwargs.pop("block_chunks")
    if backbone_cls is EvaWithChunking:
        kwargs["block_chunks"] = block_chunks
    return backbone_cls(**kwargs)
