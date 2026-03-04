import os.path as osp
from dataclasses import asdict, is_dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from models.resnetall import generate_model
from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.modeling.architecture import Decoder, replace_batchnorm_with_groupnorm
from train_resnet3d_lib.modeling.group_dro import GroupDROComputer
from train_resnet3d_lib.stitch_manager import StitchManager


def save_regression_hyperparameters(model, *, model_cfg, objective_cfg, stitch_cfg):
    model.save_hyperparameters(
        {
            "model_cfg": {
                "size": int(model_cfg.size),
                "enc": str(model_cfg.enc),
                "with_norm": bool(model_cfg.with_norm),
                "total_steps": int(model_cfg.total_steps),
                "n_groups": int(model_cfg.n_groups),
                "group_names": list(model_cfg.group_names),
                "norm": str(model_cfg.norm),
                "group_norm_groups": int(model_cfg.group_norm_groups),
            },
            "objective_cfg": {
                "objective": str(objective_cfg.objective),
                "loss_mode": str(objective_cfg.loss_mode),
                "loss_recipe": str(objective_cfg.loss_recipe),
                "bce_smooth_factor": float(objective_cfg.bce_smooth_factor),
                "soft_label_positive": float(objective_cfg.soft_label_positive),
                "soft_label_negative": float(objective_cfg.soft_label_negative),
                "robust_step_size": objective_cfg.robust_step_size,
                "group_counts": list(objective_cfg.group_counts),
                "group_dro_gamma": float(objective_cfg.group_dro_gamma),
                "group_dro_btl": bool(objective_cfg.group_dro_btl),
                "group_dro_alpha": objective_cfg.group_dro_alpha,
                "group_dro_normalize_loss": bool(objective_cfg.group_dro_normalize_loss),
                "group_dro_min_var_weight": float(objective_cfg.group_dro_min_var_weight),
                "erm_group_topk": int(objective_cfg.erm_group_topk),
            },
            "stitch_cfg": {
                "stitch_all_val": bool(stitch_cfg.stitch_all_val),
                "stitch_downsample": int(stitch_cfg.stitch_downsample),
                "stitch_log_only_downsample": int(stitch_cfg.stitch_log_only_downsample),
                "stitch_log_only_every_n_epochs": int(stitch_cfg.stitch_log_only_every_n_epochs),
                "stitch_train": bool(stitch_cfg.stitch_train),
                "stitch_train_every_n_epochs": int(stitch_cfg.stitch_train_every_n_epochs),
            },
        }
    )


def _normalize_stitch_group_idx_by_segment(stitch_group_idx_by_segment, *, n_groups):
    if stitch_group_idx_by_segment is None:
        return {}
    if not isinstance(stitch_group_idx_by_segment, dict):
        raise TypeError(
            "stitch_group_idx_by_segment must be a dict mapping segment id to group index, "
            f"got {type(stitch_group_idx_by_segment).__name__}"
        )
    normalized_group_map = {}
    for segment_id, group_idx in stitch_group_idx_by_segment.items():
        segment_key = str(segment_id)
        group_idx_i = int(group_idx)
        if group_idx_i < 0 or group_idx_i >= int(n_groups):
            raise ValueError(
                f"stitch_group_idx_by_segment[{segment_key!r}]={group_idx_i} out of range [0, {int(n_groups)})"
            )
        normalized_group_map[segment_key] = group_idx_i
    return normalized_group_map


def _init_group_dro_if_needed(model, *, objective_cfg):
    model.group_dro = None
    if model.objective != "group_dro":
        return
    robust_step_size = objective_cfg.robust_step_size
    if robust_step_size is None:
        raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")
    group_counts = objective_cfg.group_counts
    if group_counts is None:
        raise ValueError("group_counts is required when training.objective is group_dro")

    model.group_dro = GroupDROComputer(
        n_groups=model.n_groups,
        group_counts=group_counts,
        alpha=objective_cfg.group_dro_alpha,
        gamma=objective_cfg.group_dro_gamma,
        adj=objective_cfg.group_dro_adj,
        min_var_weight=objective_cfg.group_dro_min_var_weight,
        step_size=robust_step_size,
        normalize_loss=objective_cfg.group_dro_normalize_loss,
        btl=objective_cfg.group_dro_btl,
    )


def _build_backbone_with_optional_pretrained(*, model_cfg):
    resnet3d_model_depth = getattr(CFG, "resnet3d_model_depth", None)
    if resnet3d_model_depth is None:
        raise KeyError("CFG.resnet3d_model_depth is required")
    resnet3d_model_depth = int(resnet3d_model_depth)
    if resnet3d_model_depth not in {50, 101, 152}:
        raise ValueError(
            "CFG.resnet3d_model_depth must be one of [50, 101, 152], "
            f"got {resnet3d_model_depth}"
        )

    backbone = generate_model(
        model_depth=resnet3d_model_depth,
        n_input_channels=1,
        forward_features=True,
        n_classes=1039,
    )

    norm = str(model_cfg.norm).lower()
    group_norm_groups = int(model_cfg.group_norm_groups)
    init_ckpt_path = getattr(CFG, "init_ckpt_path", None)
    use_pretrained = bool(getattr(CFG, "pretrained", True))
    if use_pretrained and (not init_ckpt_path):
        backbone_pretrained_path = getattr(CFG, "backbone_pretrained_path", None)
        if not isinstance(backbone_pretrained_path, str):
            raise TypeError(
                "CFG.backbone_pretrained_path must be a string when init_ckpt_path is not provided, "
                f"got {type(backbone_pretrained_path).__name__}"
            )
        backbone_pretrained_path = backbone_pretrained_path.strip()
        if not backbone_pretrained_path:
            raise ValueError(
                "CFG.backbone_pretrained_path must be non-empty when init_ckpt_path is not provided"
            )
        if not osp.exists(backbone_pretrained_path):
            raise FileNotFoundError(
                f"Missing backbone pretrained weights: {backbone_pretrained_path}. "
                "Set training_hyperparameters.model.backbone_pretrained_path to a valid file, "
                "or pass --init_ckpt_path to fine-tune from a previous run."
            )
        backbone_ckpt = torch.load(backbone_pretrained_path, map_location="cpu")
        state_dict = backbone_ckpt.get("state_dict", backbone_ckpt)
        conv1_weight = state_dict["conv1.weight"]
        state_dict["conv1.weight"] = conv1_weight.sum(dim=1, keepdim=True)
        backbone.load_state_dict(state_dict, strict=False)
    elif not use_pretrained:
        log("CFG.pretrained=False; skipping backbone pretrained weight loading.")

    if norm == "group":
        replace_batchnorm_with_groupnorm(backbone, desired_groups=group_norm_groups)
    return backbone


def _infer_encoder_dims(backbone):
    was_training = backbone.training
    try:
        backbone.eval()
        with torch.no_grad():
            encoder_dims = [x.size(1) for x in backbone(torch.rand(1, 1, 20, 256, 256))]
    finally:
        if was_training:
            backbone.train()
    return encoder_dims


def _build_stitch_manager(stitch_cfg):
    if isinstance(stitch_cfg, dict):
        payload = dict(stitch_cfg)
    elif is_dataclass(stitch_cfg):
        payload = asdict(stitch_cfg)
    elif hasattr(stitch_cfg, "__dict__"):
        payload = dict(vars(stitch_cfg))
    else:
        raise TypeError(
            "stitch_cfg must be a dict/dataclass/object with attributes, "
            f"got {type(stitch_cfg).__name__}"
        )
    return StitchManager(**payload)


def initialize_regression_state(model, *, model_cfg, objective_cfg, stitch_cfg):
    model.objective = objective_cfg.objective.lower()
    model.loss_mode = objective_cfg.loss_mode.lower()
    model.loss_recipe = objective_cfg.loss_recipe.lower()
    model.bce_smooth_factor = objective_cfg.bce_smooth_factor
    model.soft_label_positive = objective_cfg.soft_label_positive
    model.soft_label_negative = objective_cfg.soft_label_negative
    model.with_norm = model_cfg.with_norm
    model.total_steps = model_cfg.total_steps

    if model.loss_recipe not in {"dice_bce", "bce_only"}:
        raise ValueError(f"training.loss_recipe must be one of ['bce_only', 'dice_bce'], got {model.loss_recipe!r}")
    if not (0.0 <= model.bce_smooth_factor <= 0.5):
        raise ValueError(
            "training_hyperparameters.training.bce_smooth_factor must be in [0.0, 0.5], "
            f"got {model.bce_smooth_factor}"
        )
    if not (0.0 <= model.soft_label_positive <= 1.0):
        raise ValueError(
            "training_hyperparameters.training.soft_label_positive must be in [0.0, 1.0], "
            f"got {model.soft_label_positive}"
        )
    if not (0.0 <= model.soft_label_negative <= 1.0):
        raise ValueError(
            "training_hyperparameters.training.soft_label_negative must be in [0.0, 1.0], "
            f"got {model.soft_label_negative}"
        )
    if model.soft_label_positive <= model.soft_label_negative:
        raise ValueError(
            "training_hyperparameters.training.soft_label_positive must be greater than soft_label_negative, "
            f"got {model.soft_label_positive} <= {model.soft_label_negative}"
        )

    model.n_groups = int(model_cfg.n_groups)
    model.group_names = list(model_cfg.group_names)
    if len(model.group_names) == 0:
        model.group_names = [str(i) for i in range(model.n_groups)]
    if len(model.group_names) != model.n_groups:
        raise ValueError(f"group_names length must be {model.n_groups}, got {len(model.group_names)}")
    model._stitch_group_idx_by_segment = _normalize_stitch_group_idx_by_segment(
        model_cfg.stitch_group_idx_by_segment,
        n_groups=model.n_groups,
    )

    _init_group_dro_if_needed(model, objective_cfg=objective_cfg)

    model.erm_group_topk = int(objective_cfg.erm_group_topk or 0)
    if model.erm_group_topk < 0:
        raise ValueError(f"erm_group_topk must be >= 0, got {model.erm_group_topk}")

    model._ema_decay = float(getattr(CFG, "ema_decay", 0.9))
    model._ema_metrics = {}

    model._eval_threshold = float(getattr(CFG, "eval_threshold", 0.5))
    model._val_eval_metrics = None

    model.loss_func1 = smp.losses.DiceLoss(mode="binary")

    model.backbone = _build_backbone_with_optional_pretrained(model_cfg=model_cfg)

    norm = str(model_cfg.norm).lower()
    group_norm_groups = int(model_cfg.group_norm_groups)
    encoder_dims = _infer_encoder_dims(model.backbone)

    model.decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups)

    if model.with_norm:
        if norm == "group":
            model.normalization = nn.GroupNorm(num_groups=1, num_channels=1)
        else:
            model.normalization = nn.BatchNorm3d(num_features=1)

    model._stitcher = _build_stitch_manager(stitch_cfg)
