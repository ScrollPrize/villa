"""Resolve and validate a JEM recipe once, before any model or data is built.

Defaults are the settled recipe of plans/JEM_impl.md. Where the JEM paper is
underspecified the defaults are this implementation's adopted choices (named in
`recipe["conventions"]`), not claims about the authors' code.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping

JEM_DEFAULTS: dict[str, Any] = {
    "global_batch_size": 48,
    "source_chunk_size": None,  # sources per forward chunk on each rank; None = the whole local batch
    "categories": 512,
    "head_hidden_dim": 1024,
    "head_bottleneck_dim": 256,
    "structural_dim": 512,
    "head_nlayers": 3,
    "last_layer_type": "linear",
    "alignment_depths": [4, 8, 12],
    "structural_depths": [0, 4, 8],
    "depth_weights": {"4": 0.25, "8": 0.25, "12": 1.0},
    "teacher_temperature": 0.1,
    "student_temperature": 0.1,
    "visible_weight": 0.2,
    "masked_weight": 0.8,
    "global_view_weight": 0.2,
    "local_view_weight": 0.8,
    "entropy_rank_scope": "logical_batch",
    "entropy_view_scope": "crop_type",
    "structural_pair_weighting": "uniform",
    "structural_backend": "contraction",
    "structural_tile": 2048,
    "distance_bands": {"edges": [2.0, 8.0], "weights": [1 / 3, 1 / 3, 1 / 3]},
    "momentum_teacher": 0.994,
    "regularizer_warmup_steps": 12500,
    "global_collapse_schedule": {"start": 1.0, "end": 0.1},
    "structural_schedule": {"warm": 0.05, "final": 1.0},
    "views": {"source_size": 512, "teacher_size": 256, "global_size": 256, "local_size": 144,
              "locals_per_set": 4, "min_overlap_fraction": 0.10, "min_overlap_count": 2,
              "max_student_attempts": 64, "max_teacher_attempts": 16, "max_source_attempts": 64},
    "masking": {},
    "batch_storage": "device",
    "checkpoint_heads": False,
    "checkpoint_losses": True,
    "auxiliaries": {"object_weight": 0.0, "ink_weight": 0.0, "region_weight": 0.0, "point_weight": 0.0},
}

# Settings of the DINO objective that have no meaning for JEM.
DINO_ONLY_KEYS = {
    "dino_loss_weight", "ibot_loss_weight", "koleo_loss_weight", "koleo_distributed", "koleo_loss_group_size",
    "centering", "teacher_temp", "warmup_teacher_temp", "warmup_teacher_temp_steps", "final_momentum_teacher",
    "freeze_last_layer_steps", "freeze_last_layer_epochs", "freeze_last_layer_ratio", "mask_ratio_min_max",
    "mask_sample_probability", "ibot_fused_last_layer", "ibot_masked_loss_chunk_size", "ibot_kernel_backend",
    "ibot_targets", "gram", "point_supervision", "object_losses", "heads",
}

CONVENTIONS = {
    "empty_alignment_subset": "graph-connected zero; 0.2/0.8 not renormalized",
    "depth_weights": "unnormalized weighted sum for alignment and global collapse",
    "structural_depth_reduction": "uniform mean over target depths",
    "view_weights": "ten student views weighted equally (global total 0.2, local total 0.8)",
    "entropy_position_mean": "positions with zero valid vectors excluded on every rank",
    "categorical_tap_norm": "backbone patch LayerNorm shared across depths, applied once per raw tap",
    "structural_target_norm": "non-affine LayerNorm then L2 normalization of raw teacher taps",
    "layer_zero": "patch-embedding output before prefixes and positions",
    "mask_rounding": "round-half-up token count",
    "head_counting": "three affine stages through the bottleneck plus the separate categorical projection",
}


def _merge(defaults, values, path="method"):
    result = deepcopy(defaults)
    for key, value in dict(values).items():
        if key not in defaults:
            raise ValueError(f"Unknown JEM setting {path}.{key}")
        if isinstance(defaults[key], dict) and key not in ("depth_weights", "masking") and isinstance(value, Mapping):
            result[key] = _merge(defaults[key], value, f"{path}.{key}")
        else:
            result[key] = deepcopy(value)
    return result


def resolve_jem_method(method: Mapping[str, Any]) -> dict[str, Any]:
    method = dict(method)
    name = method.pop("name", "jem")
    if name != "jem":
        raise ValueError(f"resolve_jem_method got method {name!r}")
    conflicting = sorted(set(method) & DINO_ONLY_KEYS)
    if conflicting:
        raise ValueError(f"DINO settings are not valid for method 'jem': {conflicting}")
    method.pop("conventions", None)
    recipe = _merge(JEM_DEFAULTS, method)
    recipe["name"] = "jem"

    alignment = [int(d) for d in recipe["alignment_depths"]]
    structural = [int(d) for d in recipe["structural_depths"]]
    if alignment != sorted(set(alignment)) or structural != sorted(set(structural)):
        raise ValueError("JEM depths must be strictly increasing")
    weights = {str(int(k)): float(v) for k, v in dict(recipe["depth_weights"]).items()}
    if set(weights) != {str(d) for d in alignment}:
        raise ValueError(f"depth_weights must name exactly the alignment depths {alignment}")
    recipe.update(alignment_depths=alignment, structural_depths=structural, depth_weights=weights)

    for key in ("categories", "head_hidden_dim", "head_bottleneck_dim", "structural_dim", "head_nlayers",
                "global_batch_size", "structural_tile"):
        if int(recipe[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if recipe["last_layer_type"] not in ("linear", "weight_norm"):
        raise ValueError("JEM categorical heads use last_layer_type 'linear' (or the 'weight_norm' ablation)")
    for key in ("teacher_temperature", "student_temperature"):
        if not float(recipe[key]) > 0:
            raise ValueError(f"{key} must be positive")
    if recipe["entropy_rank_scope"] not in ("logical_batch", "rank"):
        raise ValueError("entropy_rank_scope must be 'logical_batch' or 'rank'")
    if recipe["entropy_view_scope"] not in ("crop_type", "slot"):
        raise ValueError("entropy_view_scope must be 'crop_type' or 'slot'")
    if recipe["structural_pair_weighting"] not in ("uniform", "distance_stratified"):
        raise ValueError("structural_pair_weighting must be 'uniform' or 'distance_stratified'")
    if recipe["structural_backend"] not in ("contraction", "tiled"):
        raise ValueError("structural_backend must be 'contraction' or 'tiled'")
    bands = recipe["distance_bands"]
    edges = [float(edge) for edge in bands["edges"]]
    band_weights = [float(weight) for weight in bands["weights"]]
    if not edges or any(edge <= 0 for edge in edges) or edges != sorted(set(edges)):
        raise ValueError("distance band edges must be positive and strictly increasing")
    if len(band_weights) != len(edges) + 1 or any(w < 0 for w in band_weights) or sum(band_weights) <= 0:
        raise ValueError("distance band weights need one nonnegative weight per band and a positive sum")
    recipe["distance_bands"] = {"edges": edges, "weights": band_weights}
    if not 0.0 <= float(recipe["momentum_teacher"]) <= 1.0:
        raise ValueError("momentum_teacher must be in [0, 1]")
    if int(recipe["regularizer_warmup_steps"]) < 0:
        raise ValueError("regularizer_warmup_steps must be nonnegative")
    if recipe["batch_storage"] not in ("device", "host"):
        raise ValueError("batch_storage must be 'device' or 'host'")
    if recipe["source_chunk_size"] is not None and int(recipe["source_chunk_size"]) < 1:
        raise ValueError("source_chunk_size must be positive or null")
    enabled = {key: value for key, value in recipe["auxiliaries"].items() if float(value) != 0.0}
    if enabled:
        raise NotImplementedError(
            f"JEM auxiliary objectives are not implemented yet (got nonzero {sorted(enabled)}); "
            "the pure-JEM recipe keeps them at zero")
    recipe["conventions"] = dict(CONVENTIONS)
    return recipe


ROPE_AUGMENTATIONS = ("shift_coords", "jitter_coords", "rescale_coords")


def disable_rope_augmentation(backbone: dict[str, Any]) -> dict[str, Any]:
    """JEM trains without RoPE coordinate augmentation: set shift/jitter/rescale off before
    the backbone defaults (which enable rescaling) are applied."""
    rope_kwargs = dict(backbone.get("rope_kwargs") or {})
    for key in ROPE_AUGMENTATIONS:
        flat = f"rope_{key}"
        value = rope_kwargs.get(key, backbone.get(flat, backbone.get(f"pos_embed_{flat}")))
        if value is not None:
            raise ValueError(f"JEM disables RoPE coordinate augmentation; remove rope_kwargs.{key}={value!r}")
        rope_kwargs[key] = None
    return {**backbone, "rope_kwargs": rope_kwargs}


def validate_jem_backbone(backbone: Mapping[str, Any]) -> None:
    if bool(backbone.get("class_token", True)):
        raise ValueError("JEM needs backbone.class_token = false")
    if backbone.get("untie_cls_and_patch_norms") or backbone.get("untie_global_and_local_cls_norm"):
        raise ValueError("JEM disables CLS-only output norms: set untie_cls_and_patch_norms and "
                         "untie_global_and_local_cls_norm to false")
    patch = tuple(int(p) for p in backbone["patch_size"])
    if len(set(patch)) != 1:
        raise ValueError(f"JEM needs a cubic patch size, got {patch}")
    heads = int(backbone["num_heads"])
    dim = int(backbone["embed_dim"])
    if backbone.get("use_rot_pos_emb", True) and backbone.get("rope_type") == "axial" and (dim // heads) % 6:
        raise ValueError(f"axial RoPE needs head_dim divisible by 6, got {dim // heads}; use rope_type 'compass' "
                         "(the recipe does not change width or head count to fit)")
    rope_kwargs = backbone.get("rope_kwargs") or {}
    if any(rope_kwargs.get(key) is not None for key in ROPE_AUGMENTATIONS):
        raise ValueError("JEM disables RoPE coordinate augmentation (shift/jitter/rescale)")
    if int(backbone.get("block_chunks", 0)) > 0:
        raise ValueError("JEM taps need unchunked blocks (block_chunks = 0)")


def lr_from_batch(base_lr: float, global_batch: int, reference_batch: int, rule: str = "sqrt") -> float:
    if rule == "sqrt":
        return float(base_lr) * math.sqrt(global_batch / reference_batch)
    if rule == "linear":
        return float(base_lr) * global_batch / reference_batch
    raise ValueError(f"lr_scaling.rule must be 'sqrt' or 'linear', got {rule!r}")
